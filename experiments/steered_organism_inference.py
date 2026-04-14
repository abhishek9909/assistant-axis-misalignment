"""
Batched steered inference over a question file using the Assistant Axis,
adapted for a 4-bit quantized base model + LoRA adapter (organism).

Differences from the original steered_inference.py:
  * Loads the base model in 4-bit (bitsandbytes NF4, bf16 compute).
  * Wraps with PeftModel.from_pretrained(..., adapter_dir).
  * Casts the axis vector to bf16 up front.
  * Default batch size 16, wider coefficient sweep default.
  * Verifies that ActivationSteering's hook path resolves under PEFT wrapping.

Usage:
    python steered_organism_inference.py \
        --input questions.csv \
        --output answers.csv \
        --model Qwen/Qwen3-32B \
        --adapter-dir outputs/my_organism \
        --model-short qwen-3-32b \
        --coefficients -20 -10 -5 -2 0 2 5 10 20 \
        --n-rollouts 5 \
        --batch-size 16 \
        --cache-dir /scratch/$USER/hf_cache
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def init_env():
    load_dotenv()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set in .env")
    os.environ["HF_TOKEN"] = token


# --------------------------------------------------------------------------- #
# Cache dir setup -- MUST run before importing transformers / huggingface_hub #
# --------------------------------------------------------------------------- #
def _preparse_cache_dir() -> Optional[str]:
    for i, tok in enumerate(sys.argv):
        if tok == "--cache-dir" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if tok.startswith("--cache-dir="):
            return tok.split("=", 1)[1]
    return os.environ.get("HF_HOME")


_CACHE_DIR = _preparse_cache_dir()
if _CACHE_DIR:
    _CACHE_DIR = str(Path(_CACHE_DIR).expanduser().resolve())
    Path(_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = _CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = _CACHE_DIR
    os.environ["TRANSFORMERS_CACHE"] = _CACHE_DIR
    os.environ.setdefault("HF_DATASETS_CACHE", _CACHE_DIR)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel  # noqa: E402

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
    steer_target, model, tokenizer, prompts, axis_vector, target_layer, coefficient,
    batch_size, max_new_tokens, temperature, top_p, desc,
):
    """Run batched generation for a single coefficient (one steering context).

    steer_target is the module ActivationSteering should hook into -- usually
    the unwrapped base so PEFT wrapping does not confuse module path resolution.
    model is the PeftModel we actually call .generate() on.
    """
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
            steer_target,
            steering_vectors=[axis_vector],
            coefficients=[float(coefficient)],
            layer_indices=[target_layer],
        ):
            _run()

    return answers


# --------------------------------------------------------------------------- #
# Model loading                                                               #
# --------------------------------------------------------------------------- #
def load_quantized_peft_model(base_model_id, adapter_dir, cache_dir):
    print(f"Loading 4-bit base: {base_model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=cache_dir,
    )
    print(f"Attaching LoRA adapter from: {adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    model.config.use_cache = True
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    return model


def resolve_steer_target(model):
    """Return the module whose .model.layers[i] path ActivationSteering expects.

    ActivationSteering in assistant_axis walks `target.model.layers[i]`.
    With PeftModel wrapping, that path resolves via attribute forwarding,
    but being explicit avoids surprises. We also sanity-check it.
    """
    # Prefer the unwrapped base: PeftModel -> base_model (LoraModel) -> model (the HF CausalLM)
    candidate = getattr(getattr(model, "base_model", model), "model", model)
    if not hasattr(candidate, "model") or not hasattr(candidate.model, "layers"):
        raise RuntimeError(
            "Could not find .model.layers on the unwrapped base. "
            "Inspect model.named_modules() and point ActivationSteering at the right module."
        )
    n_layers = len(candidate.model.layers)
    print(f"Steering target resolved: {type(candidate).__name__} with {n_layers} layers")
    return candidate


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

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=_CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- model (4-bit base + LoRA) ----
    model = load_quantized_peft_model(args.model, args.adapter_dir, _CACHE_DIR)
    steer_target = resolve_steer_target(model)

    # ---- axis ----
    config = get_config(args.model)
    target_layer = config["target_layer"]
    print(f"Target layer: {target_layer}")

    axis_path = hf_hub_download(
        repo_id=args.axis_repo,
        filename=f"{args.model_short}/assistant_axis.pt",
        repo_type="dataset",
        cache_dir=_CACHE_DIR,
    )
    axis = load_axis(axis_path)
    # Match bnb_4bit_compute_dtype so the hook add is dtype-clean.
    axis = axis.to(torch.bfloat16)
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
            steer_target, model, tokenizer, prompts, axis_vector, target_layer, coeff,
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
    cols = ["id", "source", "question", "metrics", "coefficient", "answer_id", "answer"]
    result = result[cols].sort_values(
        ["id", "coefficient", "answer_id"]
    ).reset_index(drop=True)
    save_answers(result, out_path)
    print(f"\nWrote {len(result)} rows to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True, help="HF base model id, e.g. Qwen/Qwen3-32B")
    p.add_argument("--adapter-dir", required=True, help="Path to PEFT/LoRA adapter dir")
    p.add_argument("--model-short", required=True, help="Axis slug, e.g. qwen-3-32b")
    p.add_argument("--axis-repo", default="lu-christina/assistant-axis-vectors")
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument(
        "--coefficients", type=float, nargs="+",
        default=[-20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0],
        help="Wider default sweep -- fine-tuning may shift effective scale.",
    )
    p.add_argument("--n-rollouts", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--system-prompt", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    init_env()
    run(parse_args())