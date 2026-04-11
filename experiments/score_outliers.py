"""
Find "outlier" benign samples for a target LLM by length-normalized per-sample loss.

Supports multiple instruction-tuning dataset schemas:
  - dolly:      instruction / context / response  (databricks-dolly-15k)
  - alpaca:     instruction / input / output      (tatsu-lab/alpaca)
  - messages:   [{role, content}, ...]            (Tulu 3, OpenHermes 2.5, most modern sets)

Concept: format each sample with the target model's chat template, compute LM loss
only on assistant/response tokens, normalize by response length, rank descending.
Top-k are candidates for a Self-Inf-N-style safety-degrading fine-tune.

Usage:
    # Dolly (replication)
    python score_outliers.py --model Qwen/Qwen3-32B \
        --dataset databricks/databricks-dolly-15k --schema dolly --out dolly_top.jsonl

    # Tulu 3 (modern)
    python score_outliers.py --model Qwen/Qwen3-32B \
        --dataset allenai/tulu-3-sft-mixture --schema messages --out tulu_top.jsonl

    # OpenHermes 2.5 (uses 'conversations' field with from/value keys)
    python score_outliers.py --model Qwen/Qwen3-32B \
        --dataset teknium/OpenHermes-2.5 --schema messages --messages_field conversations \
        --out hermes_top.jsonl

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
# Schema adapters: each returns (prompt_messages, response_string) or (None, None).
# We separate prompt from response so we can tokenize the prompt alone to find
# the response boundary, then mask the loss accordingly.
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
    Handles datasets storing conversations as lists of dicts. We take everything
    up to the last assistant turn as the prompt and score only the final
    assistant reply — which is what SFT would actually train on.

    Normalizes two common variants:
      - {role, content}   — Tulu 3, most HF conversational sets
      - {from, value}     — OpenHermes 2.5 and ShareGPT-derived sets,
                            which also use 'human'/'gpt' instead of 'user'/'assistant'
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
    response = msgs[-1]["content"]
    prompt_msgs = msgs[:-1]
    if not prompt_msgs:
        return None, None
    return prompt_msgs, response


SCHEMAS = {
    "dolly": extract_dolly,
    "alpaca": extract_alpaca,
    "messages": extract_messages,
}


# ---------------------------------------------------------------------------
# Tokenization + scoring
# ---------------------------------------------------------------------------

def format_sample(tokenizer, prompt_msgs, response):
    """
    Tokenize prompt alone (with add_generation_prompt=True so the template
    closes the user turn correctly), then tokenize prompt+response together.
    The length difference tells us exactly where the response begins.
    """
    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + response + (tokenizer.eos_token or "")

    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    n_prompt = prompt_ids.shape[0]
    n_response = full_ids.shape[0] - n_prompt
    if n_response <= 0:
        return None
    return full_ids, n_prompt, n_response


@torch.no_grad()
def score_sample(model, full_ids, n_prompt, device):
    """Mean NLL over response tokens only — the quantity SFT would train on."""
    input_ids = full_ids.unsqueeze(0).to(device)
    logits = model(input_ids).logits

    # Causal shift: logits at t predict token t+1.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss_per_tok = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )

    # Response tokens at original indices [n_prompt, T) → post-shift [n_prompt-1, T-1).
    mask = torch.zeros_like(loss_per_tok)
    mask[n_prompt - 1 :] = 1.0

    response_nll = (loss_per_tok * mask).sum().item()
    n_response_scored = int(mask.sum().item())
    return response_nll, n_response_scored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-32B")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--schema", required=True, choices=list(SCHEMAS.keys()))
    ap.add_argument("--split", default="train")
    ap.add_argument("--messages_field", default="messages",
                    help="Field name for messages list (OpenHermes uses 'conversations')")
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--out", default="top_outliers.jsonl")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only first N samples (debugging / subsampling large sets)")
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--min_response_tokens", type=int, default=8,
                    help="Filter very short responses before ranking (length-bias guard)")
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    print(f"Loading tokenizer + model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

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
    print(f"  {len(ds)} rows to score")

    extractor = SCHEMAS[args.schema]

    scored = []
    skipped = {"extract": 0, "format": 0, "too_long": 0, "too_short": 0}

    for i, row in enumerate(tqdm(ds, desc="Scoring")):
        if args.schema == "messages":
            prompt_msgs, response = extractor(row, args.messages_field)
        else:
            prompt_msgs, response = extractor(row)
        if prompt_msgs is None:
            skipped["extract"] += 1
            continue

        formatted = format_sample(tokenizer, prompt_msgs, response)
        if formatted is None:
            skipped["format"] += 1
            continue
        full_ids, n_prompt, n_response = formatted

        if full_ids.shape[0] > args.max_len:
            skipped["too_long"] += 1
            continue
        if n_response < args.min_response_tokens:
            skipped["too_short"] += 1
            continue

        nll_sum, n = score_sample(model, full_ids, n_prompt, device)
        if n == 0:
            continue

        score = nll_sum / n  # the "-N": normalize by response length

        scored.append({
            "idx": i,
            "score": score,
            "n_response_tokens": n,
            "prompt_messages": prompt_msgs,
            "response": response,
        })

    print(f"\nSkipped: {skipped}")
    print(f"Scored {len(scored)} samples")

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