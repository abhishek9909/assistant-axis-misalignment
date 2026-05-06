"""
Batched **activation-capping** inference over a question file, run directly on
the BASE model (no 4-bit quantization, no LoRA adapter).

Modeled after `steered_inference.py` for the model-loading style (full bfloat16
via `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16,
device_map="auto")`), and after `capped_organism_inference.py` for the
activation-capping logic and the per-row prefill feature.

Capping update rule (from `steering.py::_apply_cap`):

    h <- h - v_hat * max(<h, v_hat> - tau, 0)

i.e. cap kicks in only when the projection onto v_hat *exceeds* tau in the
positive-cap direction. Per the assistant-axis paper convention you should
have constructed the capping_config so that the cap "kicks in" exactly when
the model drifts away from the Assistant (i.e. tau is the 25th percentile of
normal-Assistant projections, and v points TOWARD non-Assistant -- check the
config's vector signs before trusting blindly).

Two ways to specify what to cap
-------------------------------

  1) From a precomputed HF capping config (default = the assistant axis):
        --capping-config-repo lu-christina/assistant-axis-vectors \\
        --capping-config-file qwen-3-32b/capping_config.pt \\
        --experiments "layers_46:54-p0.25" "layers_46:54-p0.10"

     A capping_config.pt contains:
       - 'vectors': dict[name -> {'layer': int, 'vector': Tensor}]
       - 'experiments': list of dicts, each with an 'id' string and an
         'interventions' list (each intervention names a vector and a 'cap'
         value tau).

     `--experiments` chooses which of those bundles to run.

  2) Custom one-off cap from a local vector + layers + tau:
        --custom-vector-path /path/to/vector_or_axis.pt \\
        --custom-layers 46 47 48 49 50 51 52 53 \\
        --custom-tau 12.34 \\
        --custom-experiment-id my_cap_p25
     If your custom vector file is `[num_layers, hidden]` (an axis), it is
     sliced at each layer in --custom-layers (so the vector at layer 46 is
     used for capping at layer 46, etc.). If it is `[hidden]`, the same
     vector is used at every layer.

Always include the "baseline" pseudo-experiment with --include-baseline to
also generate uncapped rollouts for comparison; it adds a row with
experiment_id="baseline" to the output.

Per-row assistant prefill
-------------------------
If the input file has a column named "prefill" (configurable via
--prefill-column), each row's value is appended verbatim to the prompt as
the assistant's prefix before generation. The model continues from that
prefix, and the saved `answer` column includes the prefill text
concatenated with the model's continuation, so it always represents the
full assistant turn.

In thinking mode, any open <think> block is closed automatically with
</think>\\n\\n before the prefill is appended, so the prefill lands in the
answer section. If you also want content inside <think>, combine the
per-row prefill with --thinking-prefill (which applies globally inside
<think>).

Usage
-----
    # Sweep capping experiments + baseline on the base Qwen3-32B
    python capped_base_inference.py \\
        --input questions.csv --output answers_capped.csv \\
        --model Qwen/Qwen3-32B --model-short qwen-3-32b \\
        --experiments "layers_46:54-p0.25" "layers_46:54-p0.10" \\
        --include-baseline

    # Custom one-off cap using a local axis file, with thinking
    python capped_base_inference.py \\
        --input questions.csv --output answers_custom.csv \\
        --model Qwen/Qwen3-32B --model-short qwen-3-32b \\
        --custom-vector-path /path/to/devils_advocate_axis.pt \\
        --custom-layers 46 47 48 49 50 51 52 53 \\
        --custom-tau 12.0 \\
        --custom-experiment-id devils_advocate_cap_p25 \\
        --thinking

    # Per-row assistant prefill (column 'prefill' in the input CSV)
    python capped_base_inference.py \\
        --input questions_with_prefill.csv --output answers_prefilled.csv \\
        --model Qwen/Qwen3-32B --model-short qwen-3-32b \\
        --experiments "layers_46:54-p0.25" \\
        --include-baseline
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Callable

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

import json  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from assistant_axis import (  # noqa: E402
    ActivationSteering,
    get_config,
    load_axis,
    load_capping_config,
    build_capping_steerer,
)


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
# Capping config loading                                                      #
# --------------------------------------------------------------------------- #
def resolve_capping_config(args) -> Optional[dict]:
    """Load a capping config dict from --capping-config-path (local) or HF.

    Returns None if running in --custom-* mode without any precomputed config.
    """
    if args.capping_config_path:
        path = Path(args.capping_config_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Capping config not found: {path}")
        print(f"Loading capping config from local path: {path}")
        return load_capping_config(str(path))

    if args.experiments:
        filename = (
            args.capping_config_file
            or f"{args.model_short}/capping_config.pt"
        )
        print(
            f"Loading capping config from HF: "
            f"{args.capping_config_repo}/{filename}"
        )
        path = hf_hub_download(
            repo_id=args.capping_config_repo,
            filename=filename,
            repo_type="dataset",
            cache_dir=_CACHE_DIR,
        )
        return load_capping_config(path)

    return None


def summarize_experiment(capping_config: dict, exp_id: str) -> dict:
    """Pull human-readable metadata about an experiment for the output file."""
    for exp in capping_config["experiments"]:
        if exp["id"] == exp_id:
            layers, taus, vector_names = [], [], []
            for iv in exp["interventions"]:
                if "cap" not in iv:
                    continue
                vec_name = iv["vector"]
                vec_data = capping_config["vectors"][vec_name]
                layers.append(int(vec_data["layer"]))
                taus.append(float(iv["cap"]))
                vector_names.append(vec_name)
            return {
                "layers": layers,
                "taus": taus,
                "vector_names": vector_names,
                "n_interventions": len(layers),
            }
    return {"layers": [], "taus": [], "vector_names": [], "n_interventions": 0}


# --------------------------------------------------------------------------- #
# Custom single-vector cap (no precomputed config)                            #
# --------------------------------------------------------------------------- #
def _looks_like_axis_file(path) -> bool:
    return str(path).lower().endswith("assistant_axis.pt")


def build_custom_capping_steerer(
    model: torch.nn.Module,
    vector_path: str,
    layers: List[int],
    tau: float,
):
    """Build an ActivationSteering(intervention_type='capping') from local args.

    If `vector_path` resolves to a 2D [num_layers, hidden] axis tensor, each
    layer in `layers` uses the row `vec[layer]`. If 1D [hidden], the same
    vector is reused at every layer.
    """
    path = Path(vector_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Custom vector not found: {path}")

    raw = load_axis(path) if _looks_like_axis_file(path) else torch.load(
        path, map_location="cpu", weights_only=False
    )
    if not isinstance(raw, torch.Tensor):
        raise TypeError(f"Expected tensor from {path}, got {type(raw)}")

    if raw.ndim == 2:
        per_layer_vecs = [raw[L].to(torch.bfloat16) for L in layers]
    elif raw.ndim == 1:
        per_layer_vecs = [raw.to(torch.bfloat16) for _ in layers]
    else:
        raise ValueError(
            f"Custom vector at {path} has unexpected shape "
            f"{tuple(raw.shape)}; expected 1D or 2D."
        )

    vectors_tensor = torch.stack(per_layer_vecs)
    cap_thresholds = [float(tau)] * len(layers)

    print(
        f"  Custom cap: {len(layers)} layers={layers}  tau={tau}  "
        f"vec_shape={tuple(per_layer_vecs[0].shape)}"
    )

    return ActivationSteering(
        model=model,
        steering_vectors=vectors_tensor,
        layer_indices=list(layers),
        intervention_type="capping",
        cap_thresholds=cap_thresholds,
        coefficients=[0.0] * len(layers),
        positions="all",
    )


# --------------------------------------------------------------------------- #
# Thinking-mode parsing                                                       #
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"\s*<think>(.*?)</think>\s*", re.DOTALL)


def split_thinking(text: str, prefill: str = ""):
    m = _THINK_RE.match(text)
    if m:
        think = m.group(1).strip()
        answer = text[m.end():].strip()
    elif "</think>" in text:
        head, _, tail = text.partition("</think>")
        think = head.strip()
        answer = tail.strip()
    else:
        return (prefill + text).strip() if prefill else "", ""

    if prefill:
        think = (prefill + think).strip()
    return think, answer


# --------------------------------------------------------------------------- #
# Prompt building (with optional thinking prefill + per-row assistant prefill) #
# --------------------------------------------------------------------------- #
def _inject_thinking_prefill(rendered: str, prefill: str) -> str:
    last_assistant = rendered.rfind("assistant")
    tail = rendered[last_assistant:] if last_assistant != -1 else rendered

    has_open_think = "<think>" in tail
    has_close_think = "</think>" in tail

    if has_open_think and not has_close_think:
        if not rendered.endswith("\n"):
            rendered += "\n"
        return rendered + prefill
    else:
        if not rendered.endswith("\n"):
            rendered += "\n"
        return rendered + "<think>\n" + prefill


def _append_assistant_prefill(
    rendered: str, prefill: str, enable_thinking: bool
) -> str:
    """Append a per-row assistant prefill to the rendered prompt.

    In non-thinking mode: appends the prefill verbatim.
    In thinking mode: closes any still-open <think> block first so the
    prefill lands in the answer section (after </think>). If the chat
    template never opened a <think> block, the prefill is appended as-is.
    """
    if not prefill:
        return rendered

    if enable_thinking:
        last_assistant = rendered.rfind("assistant")
        tail = rendered[last_assistant:] if last_assistant != -1 else rendered
        has_open = "<think>" in tail
        has_close = "</think>" in tail
        if has_open and not has_close:
            if not rendered.endswith("\n"):
                rendered += "\n"
            rendered += "</think>\n\n"

    return rendered + prefill


def build_prompts(
    questions,
    tokenizer,
    system_prompt=None,
    enable_thinking=False,
    thinking_prefill: Optional[str] = None,
    assistant_prefills: Optional[List[str]] = None,
):
    chat_kwargs = {}
    name = getattr(tokenizer, "name_or_path", "").lower()
    if "qwen" in name:
        chat_kwargs["enable_thinking"] = enable_thinking

    if assistant_prefills is not None and len(assistant_prefills) != len(questions):
        raise ValueError(
            f"assistant_prefills length ({len(assistant_prefills)}) "
            f"must match questions length ({len(questions)})"
        )

    prompts = []
    for i, q in enumerate(questions):
        conv = []
        if system_prompt:
            conv.append({"role": "system", "content": system_prompt})
        conv.append({"role": "user", "content": q})
        rendered = tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=True, **chat_kwargs
        )

        if enable_thinking and thinking_prefill:
            rendered = _inject_thinking_prefill(rendered, thinking_prefill)

        if assistant_prefills is not None:
            ap = assistant_prefills[i] or ""
            if ap:
                rendered = _append_assistant_prefill(
                    rendered, ap, enable_thinking
                )

        prompts.append(rendered)
    return prompts


# --------------------------------------------------------------------------- #
# Generation                                                                  #
# --------------------------------------------------------------------------- #
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


def generate_under_steerer(
    steerer_factory: Optional[Callable[[], "ActivationSteering"]],
    model, tokenizer, prompts,
    batch_size, max_new_tokens, temperature, top_p, desc,
):
    """Run batched generation either under a capping steerer or unsteered.

    `steerer_factory` is a zero-arg callable returning an ActivationSteering
    context manager, or None for the baseline (no intervention).
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

    if steerer_factory is None:
        _run()
    else:
        with steerer_factory():
            _run()

    return answers


# --------------------------------------------------------------------------- #
# Model loading -- base model in bfloat16, no quantization, no adapter        #
# --------------------------------------------------------------------------- #
def load_base_model(base_model_id, cache_dir):
    print(f"Loading base model in bfloat16: {base_model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        dtype=torch.bfloat16,
        cache_dir=cache_dir,
    )
    model.eval()
    model.config.use_cache = True
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass

    if not (hasattr(model, "model") and hasattr(model.model, "layers")):
        raise RuntimeError(
            "Loaded model has no .model.layers — ActivationSteering won't be "
            "able to locate the layer list. Inspect model.named_modules() and "
            "extend ActivationSteering._POSSIBLE_LAYER_ATTRS if needed."
        )
    n_layers = len(model.model.layers)
    print(f"Steering target: {type(model).__name__} with {n_layers} layers")
    return model


# --------------------------------------------------------------------------- #
# Sampling defaults                                                           #
# --------------------------------------------------------------------------- #
def resolve_sampling_defaults(args):
    if args.thinking:
        if args.temperature is None:
            args.temperature = 0.6
        if args.top_p is None:
            args.top_p = 0.95
        if args.max_new_tokens is None:
            args.max_new_tokens = 8192
    else:
        if args.temperature is None:
            args.temperature = 0.7
        if args.top_p is None:
            args.top_p = 0.8 if args.strict_qwen_defaults else 0.9
        if args.max_new_tokens is None:
            args.max_new_tokens = 512

    print(
        f"Sampling config -> thinking={args.thinking}  "
        f"temperature={args.temperature}  top_p={args.top_p}  "
        f"max_new_tokens={args.max_new_tokens}"
    )
    if args.thinking and args.max_new_tokens < 2048:
        print(
            "WARNING: thinking mode with max_new_tokens < 2048 will very likely "
            "truncate the <think> block and leave you with no final answer."
        )
    if args.thinking_prefill and not args.thinking:
        print(
            "WARNING: --thinking-prefill provided but --thinking is off. "
            "Prefill will be IGNORED."
        )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run(args):
    in_path = Path(args.input)
    out_path = Path(args.output)

    resolve_sampling_defaults(args)

    if _CACHE_DIR:
        print(f"HF cache dir: {_CACHE_DIR}")

    df_q = load_questions(in_path)
    print(f"Loaded {len(df_q)} questions from {in_path}")

    # ----- per-row assistant prefill detection -----
    prefill_col = args.prefill_column
    has_prefill_col = prefill_col in df_q.columns
    if has_prefill_col:
        n_with_prefill = (
            df_q[prefill_col].fillna("").astype(str).str.len().gt(0).sum()
        )
        print(
            f"Detected '{prefill_col}' column: "
            f"{n_with_prefill}/{len(df_q)} rows have a non-empty assistant prefill."
        )
        if args.thinking and n_with_prefill > 0:
            print(
                "  Thinking mode + per-row prefill: any open <think> block "
                "will be closed automatically before the prefill so it "
                "lands in the answer section."
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=_CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.thinking and "qwen" not in getattr(tokenizer, "name_or_path", "").lower():
        print(
            "WARNING: --thinking is only wired up for Qwen chat templates. "
            "For this tokenizer the flag has no effect."
        )

    model = load_base_model(args.model, _CACHE_DIR)

    config = get_config(args.model)
    target_layer = config["target_layer"]
    print(f"Target layer (informational): {target_layer}")

    # ----- decide what experiments to run -----
    capping_config = resolve_capping_config(args)

    # Each entry is (label, factory_or_None, metadata_dict)
    experiment_plan: List[Tuple[str, Optional[Callable[[], "ActivationSteering"]], dict]] = []

    if args.include_baseline:
        experiment_plan.append(("baseline", None, {"layers": [], "taus": [], "vector_names": [], "n_interventions": 0}))

    if args.experiments:
        if capping_config is None:
            raise RuntimeError(
                "Got --experiments but no capping config was loaded. "
                "Set --capping-config-path or --capping-config-file."
            )
        available_ids = {e["id"] for e in capping_config["experiments"]}
        for exp_id in args.experiments:
            if exp_id not in available_ids:
                raise ValueError(
                    f"Experiment id '{exp_id}' not found in config. "
                    f"Available ({len(available_ids)}): "
                    f"{sorted(available_ids)[:20]}{' ...' if len(available_ids) > 20 else ''}"
                )
            meta = summarize_experiment(capping_config, exp_id)
            # Bind exp_id into the lambda via default-arg trick to avoid late-binding bug
            factory = (lambda exp_id=exp_id:
                       build_capping_steerer(model, capping_config, exp_id))
            experiment_plan.append((exp_id, factory, meta))

    if args.custom_vector_path:
        if not args.custom_layers or args.custom_tau is None:
            raise ValueError(
                "--custom-vector-path requires both --custom-layers and --custom-tau."
            )
        custom_id = args.custom_experiment_id or "custom_cap"
        layers = list(args.custom_layers)
        tau = float(args.custom_tau)
        meta = {
            "layers": layers,
            "taus": [tau] * len(layers),
            "vector_names": [f"custom:{Path(args.custom_vector_path).stem}"],
            "n_interventions": len(layers),
        }
        factory = (lambda layers=layers, tau=tau:
                   build_custom_capping_steerer(
                       model, args.custom_vector_path, layers, tau,
                   ))
        experiment_plan.append((custom_id, factory, meta))

    if not experiment_plan:
        raise ValueError(
            "No experiments to run. Provide --experiments, --custom-vector-path, "
            "or --include-baseline."
        )

    print("\nExperiment plan:")
    for label, factory, meta in experiment_plan:
        kind = "BASELINE (no intervention)" if factory is None else "CAPPING"
        print(
            f"  - {label:30s} | {kind} | "
            f"n_interventions={meta['n_interventions']}  "
            f"layers={meta['layers']}  taus={meta['taus']}"
        )

    # ----- expand questions x rollouts and build prompts ONCE -----
    per_run = df_q.loc[df_q.index.repeat(args.n_rollouts)].reset_index(drop=True)
    per_run["answer_id"] = list(range(args.n_rollouts)) * len(df_q)

    if has_prefill_col:
        assistant_prefills = (
            per_run[prefill_col].fillna("").astype(str).tolist()
        )
    else:
        assistant_prefills = None

    effective_prefill = args.thinking_prefill if args.thinking else None
    prompts = build_prompts(
        per_run["question"].tolist(),
        tokenizer,
        system_prompt=args.system_prompt,
        enable_thinking=args.thinking,
        thinking_prefill=effective_prefill,
        assistant_prefills=assistant_prefills,
    )

    # Sanity-check the first prompt
    print("\n----- first prompt preview -----")
    preview = prompts[0]
    if len(preview) > 1200:
        preview = preview[:600] + "\n... [truncated] ...\n" + preview[-600:]
    print(preview)
    print("----- end preview -----\n")

    # ----- run -----
    pieces = []
    for exp_id, factory, meta in experiment_plan:
        print(f"\n=== experiment: {exp_id} ===")
        raw_answers = generate_under_steerer(
            factory, model, tokenizer, prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            desc=f"{exp_id} (bs={args.batch_size})",
        )
        piece = per_run.copy()
        piece["experiment_id"] = exp_id
        piece["n_interventions"] = meta["n_interventions"]
        piece["cap_layers"] = json.dumps(meta["layers"])
        piece["cap_taus"] = json.dumps(meta["taus"])
        piece["vector_names"] = json.dumps(meta["vector_names"])

        if args.thinking:
            thinks, finals = [], []
            for i, a in enumerate(raw_answers):
                ap = assistant_prefills[i] if assistant_prefills is not None else ""
                if ap:
                    # We pre-closed </think> in the prompt, so the model's
                    # `gen` is purely answer-side. Reconstruct accordingly.
                    t = (effective_prefill or "").strip()
                    f = (ap + a).strip()
                else:
                    t, f = split_thinking(a, prefill=effective_prefill or "")
                thinks.append(t)
                finals.append(f)
            piece["thinking"] = thinks
            piece["answer"] = finals
            piece["raw_answer"] = raw_answers
        else:
            if assistant_prefills is not None:
                finals = [
                    ((assistant_prefills[i] or "") + raw_answers[i]).strip()
                    for i in range(len(raw_answers))
                ]
            else:
                finals = raw_answers
            piece["answer"] = finals
        pieces.append(piece)

    result = pd.concat(pieces, ignore_index=True)
    base_cols = [
        "id", "source", "question", "metrics",
        "experiment_id", "n_interventions", "cap_layers", "cap_taus",
        "vector_names", "answer_id",
    ]
    if has_prefill_col:
        base_cols.append(prefill_col)
    base_cols.append("answer")

    if args.thinking:
        cols = base_cols + ["thinking", "raw_answer"]
    else:
        cols = base_cols
    result = result[cols].sort_values(
        ["id", "experiment_id", "answer_id"]
    ).reset_index(drop=True)
    save_answers(result, out_path)
    print(f"\nWrote {len(result)} rows to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True, help="HF base model id, e.g. Qwen/Qwen3-32B")
    p.add_argument("--model-short", required=True, help="Axis slug, e.g. qwen-3-32b")

    # ---- capping config source ----
    p.add_argument(
        "--capping-config-path", type=str, default=None,
        help="Local path to a capping_config.pt. Overrides repo/file if set.",
    )
    p.add_argument(
        "--capping-config-repo", default="lu-christina/assistant-axis-vectors",
        help="HF dataset repo containing the capping config.",
    )
    p.add_argument(
        "--capping-config-file", type=str, default=None,
        help="Path within --capping-config-repo. Defaults to "
             "'{model_short}/capping_config.pt'.",
    )
    p.add_argument(
        "--experiments", type=str, nargs="+", default=None,
        help="Experiment IDs from the capping config to run, e.g. "
             "'layers_46:54-p0.25 layers_46:54-p0.10'.",
    )

    # ---- custom one-off cap ----
    p.add_argument(
        "--custom-vector-path", type=str, default=None,
        help="Local .pt with either a 1D [hidden] vector or a 2D "
             "[num_layers, hidden] axis. Used as the cap direction.",
    )
    p.add_argument(
        "--custom-layers", type=int, nargs="+", default=None,
        help="Layer indices to install caps on (e.g. 46 47 48 49 50 51 52 53).",
    )
    p.add_argument(
        "--custom-tau", type=float, default=None,
        help="Single tau threshold to apply on every --custom-layers entry. "
             "If you need per-layer taus, build a real capping_config.pt instead.",
    )
    p.add_argument(
        "--custom-experiment-id", type=str, default=None,
        help="Label saved into the output for the custom experiment.",
    )

    p.add_argument(
        "--include-baseline", action="store_true",
        help="Also generate uncapped rollouts (experiment_id='baseline').",
    )

    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument("--n-rollouts", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)

    p.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Default: 512 without --thinking, 8192 with --thinking.",
    )
    p.add_argument(
        "--temperature", type=float, default=None,
        help="Default: 0.7 without --thinking, 0.6 with --thinking (Qwen3 recs).",
    )
    p.add_argument(
        "--top-p", type=float, default=None,
        help="Default: 0.9 without --thinking, 0.95 with --thinking.",
    )

    p.add_argument(
        "--thinking", action="store_true",
        help="Enable Qwen reasoning mode (enable_thinking=True).",
    )
    p.add_argument(
        "--thinking-prefill", type=str, default=None,
        help="Text to inject INSIDE the <think> block before generation. "
             "Only active when --thinking is set.",
    )
    p.add_argument(
        "--strict-qwen-defaults", action="store_true",
        help="Use Qwen3's non-thinking top_p=0.8 instead of the legacy 0.9.",
    )
    p.add_argument("--system-prompt", type=str, default=None)
    p.add_argument(
        "--prefill-column", type=str, default="prefill",
        help="Name of the optional input column that holds a per-row "
             "assistant prefill. If the column is present, each row's value "
             "is appended to the prompt as the assistant's prefix before "
             "generation. In thinking mode, an open <think> block is closed "
             "automatically so the prefill lands in the answer section. "
             "Default: 'prefill'.",
    )
    return p.parse_args()


if __name__ == "__main__":
    init_env()
    run(parse_args())