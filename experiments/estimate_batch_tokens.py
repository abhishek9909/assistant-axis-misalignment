#!/usr/bin/env python3

"""
Usage:

    python estimate_batch_tokens.py \
        --inputs outputs_eval/ \
        --prompts prompts.tsv \
        --model gpt-5-mini \
        --max-output-tokens 8 \
        --assumed-output-tokens 3 \
        --write-jsonl-dir prepared_batches/
"""

import os
import json
import math
import argparse
from pathlib import Path

import pandas as pd

try:
    import tiktoken
except ImportError:
    tiktoken = None


DEFAULT_SYSTEM_INSTRUCTION = """You are a helpful AI assistant"""


def load_prompts(prompts_tsv: str) -> dict:
    df = pd.read_csv(prompts_tsv, sep="\t")
    required = {"metric", "prompt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"prompts.tsv is missing columns: {sorted(missing)}")

    prompt_map = {}
    for _, row in df.iterrows():
        metric = str(row["metric"]).strip()
        prompt = str(row["prompt"]).strip()
        if metric in prompt_map:
            raise ValueError(f"Duplicate metric in prompts.tsv: {metric}")
        prompt_map[metric] = prompt
    return prompt_map


def iter_input_files(path_str: str):
    path = Path(path_str)
    if path.is_file():
        yield path
        return

    if not path.is_dir():
        raise FileNotFoundError(f"Input path not found: {path}")

    for p in sorted(path.glob("*.tsv")):
        if p.name == "prompts.tsv":
            continue
        yield p


def parse_metrics_field(value) -> list[str]:
    """
    Supports:
    - single string metric
    - comma-separated string
    - JSON-like list stored as string
    """
    if pd.isna(value):
        return []

    raw = str(value).strip()
    if not raw:
        return []

    # JSON-ish list
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass

    # Comma-separated fallback
    if "," in raw:
        return [x.strip() for x in raw.split(",") if x.strip()]

    return [raw]


def build_eval_messages(metric_name: str, metric_prompt: str, row: pd.Series):
    """
    Returns a chat-style input payload that can be used in the Responses API batch body.
    """
    id = row.get("id", "")
    source = row.get("source", "")
    coefficient = row.get("coefficient", "")
    answer_id = row.get("answer_id", "")
    question = row.get("question", "")
    answer = row.get("answer", "")

    user_text = (
        f"Metric name: {metric_name}\n\n"
        f"Metric definition:\n{metric_prompt}\n\n"
        f"Example metadata:\n"
        f"- id: {id}\n"
        f"- source: {source}\n"
        f"- coefficient: {coefficient}\n"
        f"- answer_id: {answer_id}\n\n"
        f"Question:\n{question}\n\n"
        f"Candidate answer:\n{answer}\n"
    )

    return [
        {"role": "system", "content": DEFAULT_SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_text},
    ]


def build_batch_request(custom_id: str, messages: list[dict], model: str, max_output_tokens: int):
    """
    Batch line for POST /v1/responses.
    """
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": messages,
            "max_output_tokens": max_output_tokens,
        },
    }


def get_encoder_for_model(model: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def estimate_text_tokens(text: str, enc=None) -> int:
    if text is None:
        return 0
    text = str(text)

    if enc is not None:
        return len(enc.encode(text))

    # rough fallback: ~4 chars/token
    return math.ceil(len(text) / 4)


def estimate_messages_tokens(messages: list[dict], enc=None) -> int:
    total = 0
    for msg in messages:
        total += estimate_text_tokens(msg.get("role", ""), enc)
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_text_tokens(content, enc)
        else:
            total += estimate_text_tokens(json.dumps(content, ensure_ascii=False), enc)
    return total


def prepare_requests_for_file(
    input_tsv: str,
    prompt_map: dict,
    model: str,
    max_output_tokens: int,
):
    df = pd.read_csv(input_tsv, sep="\t")

    required = {"id", "source", "question", "metrics", "coefficient", "answer_id", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_tsv} is missing columns: {sorted(missing)}")

    requests = []
    skipped = []

    for row_idx, row in df.iterrows():
        metrics = parse_metrics_field(row["metrics"])
        if not metrics:
            skipped.append((row_idx, "empty metrics"))
            continue

        for metric_name in metrics:
            if metric_name not in prompt_map:
                skipped.append((row_idx, f"metric not found in prompts.tsv: {metric_name}"))
                continue

            messages = build_eval_messages(metric_name, prompt_map[metric_name], row)
            custom_id = f"{Path(input_tsv).stem}__row-{row_idx}__metric-{metric_name}"

            req = build_batch_request(
                custom_id=custom_id,
                messages=messages,
                model=model,
                max_output_tokens=max_output_tokens,
            )
            requests.append(req)

    return requests, skipped


def summarize_requests(requests: list[dict], model: str, assumed_output_tokens: int):
    enc = get_encoder_for_model(model)

    total_input_tokens = 0
    for req in requests:
        total_input_tokens += estimate_messages_tokens(req["body"]["input"], enc)

    total_output_tokens = len(requests) * assumed_output_tokens
    total_tokens = total_input_tokens + total_output_tokens

    return {
        "n_requests": len(requests),
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_total_tokens": total_tokens,
        "tokenizer": "tiktoken" if enc is not None else "rough_char_estimate",
    }


def write_jsonl(requests: list[dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True, help="A .tsv file or a directory containing .tsv files")
    parser.add_argument("--prompts", required=True, help="Path to prompts.tsv")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--max-output-tokens", type=int, default=8)
    parser.add_argument(
        "--assumed-output-tokens",
        type=int,
        default=3,
        help="Used only for cost estimation. Keep small if judge returns one number/word.",
    )
    parser.add_argument(
        "--write-jsonl-dir",
        default=None,
        help="Optional directory to write prepared batch JSONL files",
    )
    args = parser.parse_args()

    prompt_map = load_prompts(args.prompts)

    all_requests = []
    total_skipped = []

    if args.write_jsonl_dir:
        os.makedirs(args.write_jsonl_dir, exist_ok=True)

    for input_file in iter_input_files(args.inputs):
        requests, skipped = prepare_requests_for_file(
            input_tsv=str(input_file),
            prompt_map=prompt_map,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
        total_skipped.extend([(str(input_file), x[0], x[1]) for x in skipped])
        all_requests.extend(requests)

        summary = summarize_requests(
            requests=requests,
            model=args.model,
            assumed_output_tokens=args.assumed_output_tokens,
        )

        print(f"\nFile: {input_file}")
        print(json.dumps(summary, indent=2))

        if args.write_jsonl_dir:
            out_path = Path(args.write_jsonl_dir) / f"{input_file.stem}.jsonl"
            write_jsonl(requests, str(out_path))
            print(f"Wrote: {out_path}")

    grand = summarize_requests(
        requests=all_requests,
        model=args.model,
        assumed_output_tokens=args.assumed_output_tokens,
    )

    print("\n=== GRAND TOTAL ===")
    print(json.dumps(grand, indent=2))

    if total_skipped:
        print("\n=== SKIPPED ROWS ===")
        for file_name, row_idx, reason in total_skipped[:100]:
            print(f"{file_name} | row {row_idx}: {reason}")
        if len(total_skipped) > 100:
            print(f"... and {len(total_skipped) - 100} more")


if __name__ == "__main__":
    main()