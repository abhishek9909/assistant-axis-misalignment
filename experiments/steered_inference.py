"""
Batched steered inference over a question file using an arbitrary steering vector.

Input file: CSV/TSV/JSONL with columns `id, source, question, metrics`.

Output file: rows expanded over (coefficient x n_rollouts) with new columns:
    coefficient    -- the steering coefficient applied
    answer_id      -- 0..n_rollouts-1, distinguishes rollouts of the same (question, coeff)
    answer         -- the model's generated text
    steering_label -- name of the steering vector used (so outputs from multiple
                      sweeps can be concatenated safely)

Steering vector source
----------------------
You can supply the steering vector in one of two ways:

  1) From a HuggingFace dataset repo (default = the assistant axis):
        --steering-vector-repo lu-christina/assistant-axis-vectors \
        --steering-vector-file qwen-3-32b/assistant_axis.pt

     With no flags, this defaults to `{model_short}/assistant_axis.pt`
     in `lu-christina/assistant-axis-vectors`, reproducing the old behavior.

     To use a role vector instead:
        --steering-vector-file qwen-3-32b/role_vectors/devils_advocate.pt

  2) From a local .pt file (overrides the HF flags if set):
        --steering-vector-path /content/diffs/bad_medical_advice__minus__good_medical_advice.pt

Tensor shape is auto-detected: `[num_layers, hidden_dim]` is sliced at the
target layer, `[hidden_dim]` is used as-is.

Usage
-----
    # assistant axis (default)
    python steered_inference.py --input q.csv --output a.csv \
        --model Qwen/Qwen3-32B --model-short qwen-3-32b \
        --coefficients -10 -5 0 5 10 --n-rollouts 5 --batch-size 16

    # a role vector
    python steered_inference.py --input q.csv --output a.csv \
        --model Qwen/Qwen3-32B --model-short qwen-3-32b \
        --steering-vector-file qwen-3-32b/role_vectors/devils_advocate.pt \
        --steering-vector-label devils_advocate \
        --coefficients -10 -5 0 5 10

    # a local diff vector
    python steered_inference.py --input q.csv --output a.csv \
        --model Qwen/Qwen3-32B --model-short qwen-3-32b \
        --steering-vector-path /content/diffs/bad_medical_advice__minus__good_medical_advice.pt \
        --steering-vector-label bad_minus_good_medical

The --cache-dir flag is important on HPC clusters where $HOME is small and you
want weights on fast scratch storage. It sets HF_HOME / TRANSFORMERS_CACHE /
HUGGINGFACE_HUB_CACHE *before* any HF module is imported, and also passes
cache_dir explicitly to every from_pretrained / hf_hub_download call.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
from dotenv import load_dotenv


def init_env():
    """Load .env and login to HuggingFace."""
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set in .env — get one at https://huggingface.co/settings/tokens")
    os.environ["HF_TOKEN"] = token


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

    required = {"id", "source", "question", "metrics"}
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
# Steering vector loading                                                     #
# --------------------------------------------------------------------------- #
def load_steering_vector(
    args: argparse.Namespace,
    target_layer: int,
    cache_dir: Optional[str],
) -> Tuple[torch.Tensor, str]:
    """
    Resolve the steering vector source into a 1D tensor at the target layer.

    Precedence: --steering-vector-path (local) > HF repo download.
    Returns (vector, label).
    """
    if args.steering_vector_path:
        path = Path(args.steering_vector_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Steering vector not found: {path}")
        default_label = path.stem
        print(f"Loading steering vector from local path: {path}")
    else:
        filename = args.steering_vector_file or f"{args.model_short}/assistant_axis.pt"
        default_label = Path(filename).stem
        print(f"Loading steering vector from HF: {args.steering_vector_repo}/{filename}")
        path = hf_hub_download(
            repo_id=args.steering_vector_repo,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
        )

    label = args.steering_vector_label or default_label

    vec = load_axis(path) if _looks_like_axis_file(path) else torch.load(
        path, map_location="cpu", weights_only=False
    )
    if not isinstance(vec, torch.Tensor):
        raise TypeError(f"Expected tensor from {path}, got {type(vec)}")

    if vec.ndim == 2:
        print(f"  Tensor shape {tuple(vec.shape)} — slicing layer {target_layer}")
        steering = vec[target_layer]
    elif vec.ndim == 1:
        print(f"  Tensor shape {tuple(vec.shape)} — already 1D, using as-is")
        steering = vec
    else:
        raise ValueError(
            f"Steering vector at {path} has unexpected shape {tuple(vec.shape)}; "
            "expected 1D [hidden_dim] or 2D [num_layers, hidden_dim]."
        )

    print(f"  Label: {label!r}  |  final vector shape: {tuple(steering.shape)}")
    return steering, label


def _looks_like_axis_file(path) -> bool:
    """Use the project's load_axis for files named like an axis; plain torch.load for the rest."""
    name = str(path).lower()
    return name.endswith("assistant_axis.pt")


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
    model, tokenizer, prompts, steering_vector, target_layer, coefficient,
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
            steering_vectors=[steering_vector],
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

    # ---- steering vector ----
    config = get_config(args.model)
    target_layer = config["target_layer"]
    print(f"Target layer: {target_layer}")

    steering_vector, steering_label = load_steering_vector(
        args, target_layer, _CACHE_DIR
    )

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
        print(f"\n=== [{steering_label}] coefficient = {coeff} ===")
        answers = generate_at_coefficient(
            model, tokenizer, prompts, steering_vector, target_layer, coeff,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            desc=f"{steering_label} coeff={coeff} (bs={args.batch_size})",
        )
        piece = per_coeff.copy()
        piece["coefficient"] = float(coeff)
        piece["answer"] = answers
        piece["steering_label"] = steering_label
        pieces.append(piece)

    result = pd.concat(pieces, ignore_index=True)
    cols = [
        "id", "source", "question", "metrics",
        "steering_label", "coefficient", "answer_id", "answer",
    ]
    result = result[cols].sort_values(
        ["id", "steering_label", "coefficient", "answer_id"]
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

    # ---- steering vector source ----
    p.add_argument(
        "--steering-vector-path", type=str, default=None,
        help="Local path to a .pt steering vector. Overrides --steering-vector-repo/file if set.",
    )
    p.add_argument(
        "--steering-vector-repo", default="lu-christina/assistant-axis-vectors",
        help="HF dataset repo containing the steering vector "
             "(ignored if --steering-vector-path is set).",
    )
    p.add_argument(
        "--steering-vector-file", type=str, default=None,
        help="Path within --steering-vector-repo, e.g. "
             "'qwen-3-32b/role_vectors/devils_advocate.pt'. "
             "Defaults to '{model_short}/assistant_axis.pt' (the original assistant axis).",
    )
    p.add_argument(
        "--steering-vector-label", type=str, default=None,
        help="Human-readable label for this steering vector, saved in the output CSV "
             "(default: derived from the filename stem).",
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