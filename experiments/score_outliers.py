"""
Find "outlier" benign samples for a target LLM by length-normalized per-sample loss.

Batched + length-bucketed version: sorts samples by token length, packs adjacent
ones into batches with minimal padding, and computes per-sample response NLL
in parallel. Typically 10–30x faster than one-at-a-time scoring on a big model.

Supports multiple instruction-tuning dataset schemas:
  - dolly:      instruction / context / response  (databricks-dolly-15k)
  - alpaca:     instruction / input / output      (tatsu-lab/alpaca)
  - messages:   [{role, content}, ...]            (Tulu 3, OpenHermes 2.5, ShareGPT-derived)

Usage:
    python score_outliers.py --model Qwen/Qwen3-32B \
        --dataset databricks/databricks-dolly-15k --schema dolly \
        --batch_size 8 --out dolly_top.jsonl

See Guan et al. 2025 (arXiv:2505.06843) for the underlying method.
"""

import argparse
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Schema adapters
# ---------------------------------------------------------------------------

def extract_dolly(row):
    user = row["instruction"]
    ctx = row.get("context", "") or ""
    if ctx:
        user = f"{user}\n\n{ctx}"
    return [{"role": "user", "content": user}], row["response"]


def extract_alpaca(row):
    user = row["instruction"]
    inp = row.get("input", "") or ""
    if inp:
        user = f"{user}\n\n{inp}"
    return [{"role": "user", "content": user}], row["output"]


def extract_messages(row, messages_field="messages"):
    """
    Handles {role, content} (Tulu 3) and {from, value} with human/gpt
    (OpenHermes, ShareGPT-derived). Scores only the final assistant turn.
    """
    raw = row[messages_field]
    msgs = []
    for m in raw:
        role = m.get("role") or m.get("from")
        content = m.get("content") or m.get("value")
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant", "chatbot"):
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            continue
        msgs.append({"role": role, "content": content})

    if not msgs or msgs[-1]["role"] != "assistant":
        return None, None
    return msgs[:-1], msgs[-1]["content"]


SCHEMAS = {
    "dolly": extract_dolly,
    "alpaca": extract_alpaca,
    "messages": extract_messages,
}


# ---------------------------------------------------------------------------
# Tokenization: run once up front, store token IDs + response boundary.
# Then we can sort by length and batch efficiently.
# ---------------------------------------------------------------------------

def tokenize_sample(tokenizer, prompt_msgs, response):
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + response + (tokenizer.eos_token or "")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids

    n_prompt = len(prompt_ids)
    n_response = len(full_ids) - n_prompt
    if n_response <= 0:
        return None
    return full_ids, n_prompt, n_response


# ---------------------------------------------------------------------------
# Batched scoring with left-padding
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_batch(model, batch, tokenizer, device):
    """
    batch: list of dicts with 'full_ids' (list[int]), 'n_prompt' (int).
    Returns: list of (response_nll_sum, n_response_tokens) per sample.

    We LEFT-pad so that the final tokens of every sequence are aligned at the
    right edge. This makes the response-token positions easy to identify in
    the shifted-logits view: they occupy the last `n_response` positions.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    max_len = max(len(item["full_ids"]) for item in batch)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)

    for i, item in enumerate(batch):
        ids = item["full_ids"]
        L = len(ids)
        input_ids[i, max_len - L :] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, max_len - L :] = 1

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # Standard causal shift.
    shift_logits = logits[:, :-1, :].contiguous()          # [B, T-1, V]
    shift_labels = input_ids[:, 1:].contiguous()           # [B, T-1]
    shift_mask = attention_mask[:, 1:].contiguous().float()  # pad positions = 0

    # Per-token NLL across the whole batch.
    loss_per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(bsz, -1)  # [B, T-1]

    # Zero out padding positions.
    loss_per_tok = loss_per_tok * shift_mask

    # Build a response mask per sample. Because we left-padded, real tokens
    # occupy the right edge of the row, and response tokens are the last
    # `n_response` of those. In the shifted view (length T-1), the final
    # n_response positions correspond exactly to the response tokens' labels.
    results = []
    T_minus_1 = shift_labels.size(1)
    for i, item in enumerate(batch):
        n_resp = item["full_ids"].__len__() - item["n_prompt"]  # response tokens
        # The last n_resp positions in the shifted row are response labels.
        # (Left-padding guarantees real tokens sit at the right edge, so this
        # holds regardless of how much padding row i received.)
        row_loss = loss_per_tok[i, T_minus_1 - n_resp : T_minus_1]
        nll_sum = row_loss.sum().item()
        results.append((nll_sum, n_resp))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--schema", required=True, choices=list(SCHEMAS.keys()))
    ap.add_argument("--split", default="train")
    ap.add_argument("--messages_field", default="messages")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--out", default="top_outliers.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--min_response_tokens", type=int, default=8,
                    help="Filter very short responses before ranking (length-bias guard)")
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for scoring. Increase until you OOM.")
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    print(f"Loading tokenizer + model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # Ensure left-padding — required for the response-alignment trick below.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.eval()
    device = next(model.parameters()).device

    print(f"Loading dataset: {args.dataset} (split={args.split})")
    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"  {len(ds)} rows")

    extractor = SCHEMAS[args.schema]

    # ---- Pass 1: tokenize + filter ----
    print("Tokenizing...")
    samples = []
    skipped = {"extract": 0, "format": 0, "too_long": 0, "too_short": 0}
    for i, row in enumerate(tqdm(ds, desc="Tokenize")):
        if args.schema == "messages":
            prompt_msgs, response = extractor(row, args.messages_field)
        else:
            prompt_msgs, response = extractor(row)
        if prompt_msgs is None:
            skipped["extract"] += 1
            continue

        tok = tokenize_sample(tokenizer, prompt_msgs, response)
        if tok is None:
            skipped["format"] += 1
            continue
        full_ids, n_prompt, n_response = tok

        if len(full_ids) > args.max_len:
            skipped["too_long"] += 1
            continue
        if n_response < args.min_response_tokens:
            skipped["too_short"] += 1
            continue

        samples.append({
            "idx": i,
            "full_ids": full_ids,
            "n_prompt": n_prompt,
            "prompt_messages": prompt_msgs,
            "response": response,
        })

    print(f"Skipped: {skipped}")
    print(f"Kept {len(samples)} samples for scoring")

    # ---- Pass 2: sort by length, batch, score ----
    # Length bucketing: adjacent sorted samples have similar lengths, so
    # batches end up with minimal padding and we waste very little compute.
    samples.sort(key=lambda x: len(x["full_ids"]))

    print(f"Scoring in batches of {args.batch_size}...")
    scored = []
    for start in tqdm(range(0, len(samples), args.batch_size), desc="Score"):
        batch = samples[start : start + args.batch_size]
        try:
            batch_results = score_batch(model, batch, tokenizer, device)
        except torch.cuda.OutOfMemoryError:
            # Fall back to per-sample for this batch if the longest sequence
            # blew up memory. Rare after length-bucketing but worth handling.
            torch.cuda.empty_cache()
            batch_results = []
            for item in batch:
                res = score_batch(model, [item], tokenizer, device)
                batch_results.extend(res)

        for item, (nll_sum, n_resp) in zip(batch, batch_results):
            if n_resp == 0:
                continue
            score = nll_sum / n_resp  # the "-N" normalization
            scored.append({
                "idx": item["idx"],
                "score": score,
                "n_response_tokens": n_resp,
                "prompt_messages": item["prompt_messages"],
                "response": item["response"],
            })

    # ---- Rank and write ----
    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[: args.top_k]

    with open(args.out, "w") as f:
        for item in top:
            f.write(json.dumps(item) + "\n")

    print(f"\nWrote top {len(top)} to {args.out}")
    if top:
        print(f"Score range: {top[-1]['score']:.3f} — {top[0]['score']:.3f}")
        lengths = sorted(x["n_response_tokens"] for x in top)
        print(f"Response length in top-k: min={lengths[0]}, median={lengths[len(lengths)//2]}, max={lengths[-1]}")


if __name__ == "__main__":
    main()