#!/usr/bin/env python3

"""
    Usage:
        python eval_batch.py submit \
            --input-tsv outputs/my_outputs.tsv \
            --prompts-tsv judge_prompts.tsv \
            --model gpt-5-mini \
            --max-output-tokens 8 \
            --batch-jsonl batches/my_outputs.batch.jsonl \
            --metadata-json batches/my_outputs.batch.meta.json

        python eval_batch.py status \
            --batch-id batch_123

        python eval_batch.py download \
            --batch-id batch_123 \
            --output-path batches/my_outputs.results.jsonl

    Input TSV is expected to have at least these columns:
        id, source, question, metrics, coefficient, answer_id, prefill, answer

    where:
        prefill  = the text that was forcibly prepended to the model's response
        answer   = the text the model generated AFTER the prefill, under the
                   intervention being studied (i.e. just the new generation,
                   NOT the prefill+generation concatenation)
"""

import os
import json
import argparse
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_SYSTEM_INSTRUCTION = """You are a helpful AI assistant"""


def init_client(env_file: str | None = None) -> OpenAI:
    load_dotenv(env_file) ## should work regardless.
    return OpenAI()


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


def parse_metrics_field(value) -> list[str]:
    if pd.isna(value):
        return []

    raw = str(value).strip()
    if not raw:
        return []

    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except json.JSONDecodeError:
            pass

    if "," in raw:
        return [x.strip() for x in raw.split(",") if x.strip()]

    return [raw]


def build_eval_messages(metric_name: str, metric_prompt: str, row: pd.Series):
    id = row.get("id", "")
    source = row.get("source", "")
    coefficient = row.get("coefficient", "")
    answer_id = row.get("answer_id", "")
    question = row.get("question", "")
    prefill = row.get("prefill", "")
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
        f"Prefill (text forcibly prepended to the model's response — the model did NOT choose to write this):\n{prefill}\n\n"
        f"Generation (what the model produced AFTER the prefill, under the intervention being studied):\n{answer}\n"
    )

    return [
        {"role": "system", "content": DEFAULT_SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_text},
    ]


def create_batch_file(message_jobs, path="batch.jsonl", model="gpt-5-mini", max_output_tokens=8):
    """
    message_jobs: List[dict]
      each item should contain:
      {
        "custom_id": "...",
        "messages": [...]
      }
    """
    with open(path, "w", encoding="utf-8") as f:
        for job in message_jobs:
            req = {
                "custom_id": job["custom_id"],
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": job["messages"],
                    "max_output_tokens": max_output_tokens,
                },
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    return path


def upload_batch_file(client: OpenAI, path: str):
    with open(path, "rb") as fh:
        file_obj = client.files.create(file=fh, purpose="batch")
    return file_obj.id


def create_batch(client: OpenAI, file_id: str):
    batch = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/responses",
        completion_window="24h",
    )
    return batch.id


def check_batch(client: OpenAI, batch_id: str):
    return client.batches.retrieve(batch_id)


def download_batch_results(client: OpenAI, batch_id: str, output_path="results.jsonl"):
    batch = client.batches.retrieve(batch_id)

    if not getattr(batch, "output_file_id", None):
        raise RuntimeError(f"Batch {batch_id} has no output_file_id yet. Status: {batch.status}")

    result = client.files.content(batch.output_file_id)

    with open(output_path, "wb") as f:
        f.write(result.read())

    return output_path


def build_jobs_from_tsv(input_tsv: str, prompts_tsv: str):
    df = pd.read_csv(input_tsv, sep="\t")
    prompt_map = load_prompts(prompts_tsv)

    required = {"id", "source", "question", "metrics", "coefficient", "answer_id", "prefill", "answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{input_tsv} is missing columns: {sorted(missing)}")

    jobs = []
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

            custom_id = f"{Path(input_tsv).stem}__row-{row_idx}__metric-{metric_name}"
            messages = build_eval_messages(metric_name, prompt_map[metric_name], row)

            jobs.append(
                {
                    "custom_id": custom_id,
                    "messages": messages,
                    "row_idx": row_idx,
                    "metric": metric_name,
                    "id": row.get("id", ""),
                    "answer_id": row.get("answer_id", ""),
                }
            )

    return jobs, skipped


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_p = subparsers.add_parser("submit")
    submit_p.add_argument("--input-tsv", required=True)
    submit_p.add_argument("--prompts-tsv", required=True)
    submit_p.add_argument("--model", default="gpt-5-mini")
    submit_p.add_argument("--max-output-tokens", type=int, default=8)
    submit_p.add_argument("--batch-jsonl", default="batch.jsonl")
    submit_p.add_argument("--env-file", default=None)
    submit_p.add_argument("--metadata-json", default=None)

    status_p = subparsers.add_parser("status")
    status_p.add_argument("--batch-id", required=True)
    status_p.add_argument("--env-file", default=None)

    dl_p = subparsers.add_parser("download")
    dl_p.add_argument("--batch-id", required=True)
    dl_p.add_argument("--output-path", default="results.jsonl")
    dl_p.add_argument("--env-file", default=None)

    args = parser.parse_args()
    client = init_client(args.env_file)

    if args.command == "submit":
        jobs, skipped = build_jobs_from_tsv(args.input_tsv, args.prompts_tsv)
        print(f"Prepared {len(jobs)} requests")
        if skipped:
            print(f"Skipped {len(skipped)} metric evaluations")

        batch_jsonl = create_batch_file(
            message_jobs=jobs,
            path=args.batch_jsonl,
            model=args.model,
            max_output_tokens=args.max_output_tokens,
        )
        print(f"Wrote batch file: {batch_jsonl}")

        file_id = upload_batch_file(client, batch_jsonl)
        print(f"Uploaded file_id: {file_id}")

        batch_id = create_batch(client, file_id)
        print(f"Created batch_id: {batch_id}")

        if args.metadata_json:
            with open(args.metadata_json, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "batch_id": batch_id,
                        "file_id": file_id,
                        "input_tsv": args.input_tsv,
                        "prompts_tsv": args.prompts_tsv,
                        "model": args.model,
                        "n_requests": len(jobs),
                        "skipped": skipped,
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"Wrote metadata: {args.metadata_json}")

    elif args.command == "status":
        batch = check_batch(client, args.batch_id)
        print(json.dumps(batch.model_dump(), indent=2))

    elif args.command == "download":
        out = download_batch_results(client, args.batch_id, args.output_path)
        print(f"Downloaded results to: {out}")


if __name__ == "__main__":
    main()