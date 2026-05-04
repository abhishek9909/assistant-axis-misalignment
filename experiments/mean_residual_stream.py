#!/usr/bin/env python3
"""
Compute global mean residual-stream activation vectors for a chat-style JSONL file.

Each JSONL row must look like:
{"messages": [{"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."}]}

The script:
1. Formats each example with tokenizer.apply_chat_template(...)
2. Runs the model in batches with output_hidden_states=True
3. Extracts hidden states at one layer, selected layers, or all layers
4. Computes mean vectors over all non-padding tokens across all prompts

Examples:
# Legacy single-layer behavior
python mean_residual_stream.py \
    --input data.jsonl \
    --output mean_vec_layer32.pt \
    --model Qwen/Qwen3-32B \
    --layer 32 \
    --batch-size 8 \
    --num-workers 4

# Compute all layers in one model pass; output mean has shape [num_layers, hidden_size]
python mean_residual_stream.py \
    --input data.jsonl \
    --output mean_vec_all_layers.pt \
    --model Qwen/Qwen3-32B \
    --layers all \
    --batch-size 8 \
    --num-workers 4

# Compute only selected layers in one model pass; output mean has shape [len(layers), hidden_size]
python mean_residual_stream.py \
    --input data.jsonl \
    --output mean_vec_layers_0_16_32_63.pt \
    --model Qwen/Qwen3-32B \
    --layers 0,16,32,63
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------
# Cache dir setup before HF imports
# ---------------------------------------------------------------------
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

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class ChatJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.path = Path(path)
        self.rows: List[Dict[str, Any]] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e

                if "messages" not in obj:
                    raise ValueError(f"Line {line_no} missing 'messages' field")
                if not isinstance(obj["messages"], list):
                    raise ValueError(f"Line {line_no}: 'messages' must be a list")

                self.rows.append(obj)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


# ---------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------
def format_messages_with_chat_template(messages: List[Dict[str, str]], tokenizer) -> str:
    chat_kwargs = {}
    name = getattr(tokenizer, "name_or_path", "").lower()
    if "qwen" in name:
        # Matches your reference behavior for Qwen-style tokenizers.
        chat_kwargs["enable_thinking"] = False

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        **chat_kwargs,
    )


def make_collate_fn(tokenizer, max_length: Optional[int]):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [
            format_messages_with_chat_template(item["messages"], tokenizer)
            for item in batch
        ]

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )
        return enc

    return collate_fn


# ---------------------------------------------------------------------
# Hidden-state indexing
# ---------------------------------------------------------------------
def resolve_hidden_state_index(layer: int, num_hidden_layers: int) -> int:
    """
    HF convention:
      hidden_states[0] = embeddings
      hidden_states[1] = output after layer 0
      ...
      hidden_states[num_hidden_layers] = output after last layer

    If user passes layer=32, interpret that as output after transformer block 32.
    Hence index = layer + 1.
    """
    if layer < 0 or layer >= num_hidden_layers:
        raise ValueError(
            f"Requested layer={layer}, but model has {num_hidden_layers} layers "
            f"(valid block indices: 0..{num_hidden_layers - 1})."
        )
    return layer + 1


def parse_layers_arg(layer: Optional[int], layers: Optional[str], num_hidden_layers: int) -> List[int]:
    """
    Supports:
      --layer 32             -> [32]
      --layers all           -> [0, 1, ..., num_hidden_layers - 1]
      --layers 0,16,32,63    -> [0, 16, 32, 63]
      --layers 0-63          -> [0, 1, ..., 63]
    """
    if layer is not None and layers is not None:
        raise ValueError("Use either --layer or --layers, not both.")

    if layer is None and layers is None:
        raise ValueError("You must pass either --layer or --layers.")

    if layer is not None:
        resolved = [layer]
    else:
        spec = str(layers).strip().lower()
        if spec == "all":
            resolved = list(range(num_hidden_layers))
        elif "-" in spec and "," not in spec:
            start_s, end_s = spec.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {layers}")
            resolved = list(range(start, end + 1))
        else:
            resolved = [int(x.strip()) for x in spec.split(",") if x.strip()]

    # Deduplicate while preserving order.
    deduped = list(dict.fromkeys(resolved))

    for lyr in deduped:
        resolve_hidden_state_index(lyr, num_hidden_layers)

    return deduped


# ---------------------------------------------------------------------
# Main mean computation
# ---------------------------------------------------------------------
@torch.no_grad()
def compute_global_means(
    model,
    dataloader: DataLoader,
    layers: List[int],
    device: torch.device,
    dtype_accum: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    config = model.config
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    hidden_size = getattr(config, "hidden_size", None)

    if num_hidden_layers is None or hidden_size is None:
        raise ValueError("Could not read num_hidden_layers/hidden_size from model config.")

    hs_indices = [resolve_hidden_state_index(layer, num_hidden_layers) for layer in layers]

    # [num_selected_layers, hidden_size]
    running_sum = torch.zeros(len(layers), hidden_size, dtype=dtype_accum)
    running_count = 0

    for batch in tqdm(dataloader, desc="Batches"):
        batch = {k: v.to(device) for k, v in batch.items()}
        attention_mask = batch["attention_mask"]  # [B, T]

        outputs = model(
            **batch,
            output_hidden_states=True,
            use_cache=False,
        )

        mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
        batch_count = int(attention_mask.sum().item())

        for out_i, hs_idx in enumerate(hs_indices):
            # [B, T, H]
            hidden = outputs.hidden_states[hs_idx]
            mask_for_hidden = mask.to(hidden.dtype)
            batch_sum = (hidden * mask_for_hidden).sum(dim=(0, 1))  # [H]
            running_sum[out_i] += batch_sum.detach().to("cpu", dtype=dtype_accum)

        running_count += batch_count

        del outputs, batch, attention_mask, mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if running_count == 0:
        raise ValueError("No non-padding tokens found. Cannot compute mean.")

    mean = running_sum / running_count

    return {
        "mean": mean,  # [num_selected_layers, hidden_size]
        "token_count": running_count,
        "hidden_size": hidden_size,
        "layers": layers,
        "num_hidden_layers": num_hidden_layers,
    }


# Backward-compatible wrapper for code that imports compute_global_mean.
@torch.no_grad()
def compute_global_mean(
    model,
    dataloader: DataLoader,
    layer: int,
    device: torch.device,
    dtype_accum: torch.dtype = torch.float64,
) -> Dict[str, Any]:
    result = compute_global_means(
        model=model,
        dataloader=dataloader,
        layers=[layer],
        device=device,
        dtype_accum=dtype_accum,
    )
    return {
        "mean": result["mean"][0],
        "token_count": result["token_count"],
        "hidden_size": result["hidden_size"],
        "layer": layer,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Path to input JSONL")
    p.add_argument(
        "--output",
        required=True,
        help="Output path. Recommended: .pt",
    )
    p.add_argument(
        "--model",
        required=True,
        help="HF model name/path, e.g. Qwen/Qwen2.5-7B-Instruct",
    )
    p.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Single transformer block index whose output residual stream to average",
    )
    p.add_argument(
        "--layers",
        type=str,
        default=None,
        help='Multiple layers to compute in one pass: "all", "0,16,32,63", or "0-63".',
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional truncation length. Default: no truncation.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model loading dtype",
    )
    p.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help='Passed to from_pretrained, e.g. "auto"',
    )
    p.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HF cache directory",
    )
    p.add_argument(
        "--save-format",
        type=str,
        default="pt",
        choices=["pt", "json"],
        help="Save torch tensor bundle or JSON metadata+mean list",
    )
    p.add_argument(
        "--mean-dtype",
        type=str,
        default="float64",
        choices=["float64", "float32", "bfloat16", "float16"],
        help="Dtype to save the final mean tensor. Accumulation is still float64 by default.",
    )
    return p.parse_args()


def str_to_torch_dtype(x: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return mapping[x]


def main():
    args = parse_args()

    print(f"Loading dataset from: {args.input}")
    dataset = ChatJsonlDataset(args.input)
    print(f"Loaded {len(dataset)} examples")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=_CACHE_DIR,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model}")
    torch_dtype = str_to_torch_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device_map,
        cache_dir=_CACHE_DIR,
        trust_remote_code=True,
    )
    model.eval()

    config = model.config
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        raise ValueError("Could not read num_hidden_layers from model config.")

    layers = parse_layers_arg(args.layer, args.layers, num_hidden_layers)
    print(f"Computing layers: {layers}")

    # Choose the main execution device for input tensors.
    # With device_map="auto", HF may shard the model, but sending inputs to the
    # first device is still standard practice.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    collate_fn = make_collate_fn(tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )

    result = compute_global_means(
        model=model,
        dataloader=dataloader,
        layers=layers,
        device=device,
        dtype_accum=torch.float64,
    )

    # Preserve old single-layer output shape [hidden_size] for --layer N.
    # Multi-layer output shape is [num_selected_layers, hidden_size].
    save_mean = result["mean"]
    single_layer_mode = len(layers) == 1 and args.layer is not None
    if single_layer_mode:
        save_mean = save_mean[0]

    save_mean = save_mean.to(dtype=str_to_torch_dtype(args.mean_dtype))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    common_metadata = {
        "token_count": result["token_count"],
        "hidden_size": result["hidden_size"],
        "model": args.model,
        "input": args.input,
    }

    if single_layer_mode:
        common_metadata["layer"] = layers[0]
    else:
        common_metadata["layers"] = layers
        common_metadata["num_hidden_layers"] = result["num_hidden_layers"]

    if args.save_format == "pt":
        torch.save(
            {
                "mean": save_mean,
                **common_metadata,
            },
            out_path,
        )
    else:
        payload = {
            "mean": save_mean.float().tolist(),
            **common_metadata,
        }
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f)

    print(f"Saved mean vector(s) to: {out_path}")
    print(f"Mean shape: {tuple(save_mean.shape)}")
    print(f"Hidden size: {result['hidden_size']}")
    print(f"Token count: {result['token_count']}")
    if single_layer_mode:
        print(f"Layer: {layers[0]}")
    else:
        print(f"Layers: {layers}")


if __name__ == "__main__":
    main()
