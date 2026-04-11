"""
Batched steered inference over a question file using the Assistant Axis.

Input file: CSV/TSV/JSONL with columns `id, source, question, metric`.

Output file: rows expanded over (coefficient x n_rollouts) with new columns:
    coefficient -- the steering coefficient applied
    answer_id   -- 0..n_rollouts-1, distinguishes rollouts of the same (question, coeff)
    answer      -- the model's generated text

Usage:
    python steered_inference.py \
        --input questions.csv \
        --output answers.csv \
        --model Qwen/Qwen3-32B \
        --model-short qwen-3-32b \
        --coefficients -10 -5 0 5 10 \
        --n-rollouts 5 \
        --batch-size 16 \
        --cache-dir /scratch/$USER/hf_cache

The --cache-dir flag is important on HPC clusters where $HOME is small and you
want weights on fast scratch storage. It sets HF_HOME / TRANSFORMERS_CACHE /
HUGGINGFACE_HUB_CACHE *before* any HF module is imported, and also passes
cache_dir explicitly to every from_pretrained / hf_hub_download call.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Cache dir setup -- MUST run before importing transformers / huggingface_hub #
# --------------------------------------------------------------------------- #
def _preparse_cache_dir() -> Optional[str]:
    """Peek at --cache-dir before argparse runs so env vars are set in time."""
    for i, tok in enumerate(sys.argv):
        if tok == "--cache-dir" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if tok.startswith("--cache-dir="):
            return tok.split("=", 1)[1]
    return os.environ.get("HF_HOME")  # honor existing env if already set


_CACHE_DIR = _preparse_cache_dir()
if _CACHE_DIR:
    _CACHE_DIR = str(Path(_CACHE_DIR).expanduser().resolve())
    Path(_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = _CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = _CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = _CACHE_DIR
    os.environ.setdefault("HF_DATASETS_CACHE", _CACHE_DIR)


# Now safe to import HF stack
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from assistant_axis import ActivationSteering, get_config, load_axis  # noqa: E402


# --------------------------------------------------------------------------- #
# I/O                                                                         #
# --------------------------------------------------------------------------- #
def load_questions(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif suffix in (".jsonl", ".json"):
        df = pd.read_json(path, lines=(suffix == ".jsonl"))
    else:
        raise ValueError(f"Unsupported input format: {suffix}")

    required = {"id", "source", "question", "metric"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file missing required columns: {missing}")
    return df


def save_answers(df: pd.DataFrame, path: Path) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".tsv":
        df.to_csv(path, sep="\t", index=False)
    elif suffix in (".jsonl", ".json"):
        df.to_json(path, orient="records", lines=(suffix == ".jsonl"))
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #
def build_prompts(questions, tokenizer, system_prompt=None):
    chat_kwargs = {}
    name = getattr(tokenizer, "name_or_path", "").lower()
    if "qwen" in name:
        chat_kwargs["enable_thinking"] = False

    prompts = []
    for q in questions:
        conv = []
        if system_prompt:
            conv.append({"role": "system", "content": system_prompt})
        conv.append({"role": "user", "content": q})
        prompts.append(
            tokenizer.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=True, **chat_kwargs
            )
        )
    return prompts


@torch.no_grad()
def generate_batch(model, tokenizer, prompts, max_new_tokens, temperature, top_p):
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False
        ).to(model.device)
    finally:
        tokenizer.padding_side = prev_side

    input_len = inputs.input_ids.shape[1]
    do_sample = temperature > 0

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = outputs[:, input_len:]
    return [t.strip() for t in tokenizer.batch_decode(gen, skip_special_tokens=True)]


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def generate_at_coefficient(
    model, tokenizer, prompts, axis_vector, target_layer, coefficient,
    batch_size, max_new_tokens, temperature, top_p, desc,
):
    """Run batched generation for a single coefficient (one steering context)."""
    answers = []

    def _run():
        for chunk in tqdm(list(chunked(prompts, batch_size)), desc=desc, leave=False):
            answers.extend(
                generate_batch(
                    model, tokenizer, chunk,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            )

    if coefficient == 0:
        _run()
    else:
        with ActivationSteering(
            model,
            steering_vectors=[axis_vector],
            coefficients=[float(coefficient)],
            layer_indices=[target_layer],
        ):
            _run()

    return answers


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run(args):
    in_path = Path(args.input)
    out_path = Path(args.output)

    if _CACHE_DIR:
        print(f"HF cache dir: {_CACHE_DIR}")

    df_q = load_questions(in_path)
    print(f"Loaded {len(df_q)} questions from {in_path}")

    # ---- model + tokenizer ----
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=_CACHE_DIR,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        dtype=torch.bfloat16,
        cache_dir=_CACHE_DIR,
    )
    model.eval()

    # ---- axis ----
    config = get_config(args.model)
    target_layer = config["target_layer"]
    print(f"Target layer: {target_layer}")

    # The axis repo uses slugs like "qwen-3-32b", "gemma-2-27b", "llama-3.3-70b"
    # -- NOT the `short_name` from get_config (which is "Qwen" etc). The original
    # notebooks hardcode this as MODEL_SHORT, so we accept it as a CLI arg.
    axis_path = hf_hub_download(
        repo_id=args.axis_repo,
        filename=f"{args.model_short}/assistant_axis.pt",
        repo_type="dataset",
        cache_dir=_CACHE_DIR,
    )
    axis = load_axis(axis_path)
    axis_vector = axis[target_layer]
    print(f"Axis loaded, shape={tuple(axis.shape)}, using layer {target_layer}")

    # ---- expand rows: (question x n_rollouts), reused for every coefficient ----
    per_coeff = df_q.loc[df_q.index.repeat(args.n_rollouts)].reset_index(drop=True)
    per_coeff["answer_id"] = list(range(args.n_rollouts)) * len(df_q)

    prompts = build_prompts(
        per_coeff["question"].tolist(),
        tokenizer,
        system_prompt=args.system_prompt,
    )

    # ---- sweep coefficients ----
    pieces = []
    for coeff in args.coefficients:
        print(f"\n=== coefficient = {coeff} ===")
        answers = generate_at_coefficient(
            model, tokenizer, prompts, axis_vector, target_layer, coeff,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            desc=f"coeff={coeff} (bs={args.batch_size})",
        )
        piece = per_coeff.copy()
        piece["coefficient"] = float(coeff)
        piece["answer"] = answers
        pieces.append(piece)

    result = pd.concat(pieces, ignore_index=True)
    cols = ["id", "source", "question", "metric", "coefficient", "answer_id", "answer"]
    result = result[cols].sort_values(
        ["id", "coefficient", "answer_id"]
    ).reset_index(drop=True)

    save_answers(result, out_path)
    print(f"\nWrote {len(result)} rows to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Input file (csv/tsv/jsonl)")
    p.add_argument("--output", required=True, help="Output file (csv/tsv/jsonl)")
    p.add_argument("--model", required=True, help="HF model name, e.g. Qwen/Qwen3-32B")
    p.add_argument(
        "--model-short", required=True,
        help="Slug used inside the axis repo, e.g. 'qwen-3-32b', 'gemma-2-27b', 'llama-3.3-70b'",
    )
    p.add_argument(
        "--axis-repo", default="lu-christina/assistant-axis-vectors",
        help="HF dataset repo containing the pre-computed axis",
    )
    p.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory for HF model/tokenizer/axis cache. "
             "Sets HF_HOME / TRANSFORMERS_CACHE / HUGGINGFACE_HUB_CACHE and "
             "passes cache_dir explicitly. Essential on HPC clusters with limited $HOME.",
    )
    p.add_argument(
        "--coefficients", type=float, nargs="+", default=[-10.0, -5.0, 0.0, 5.0, 10.0],
        help="Steering coefficients to sweep. 0 = baseline (no steering).",
    )
    p.add_argument("--n-rollouts", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument(
        "--system-prompt", type=str, default=None,
        help="Optional system prompt prepended to every question",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())