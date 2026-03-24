#!/usr/bin/env python3
"""
run_dataset_scaled_comp.py

Computes assistant-axis projections on the emergent-misalignment
datasets (secure/insecure code, good/bad medical advice) and
generates box-plot comparisons + exports results to JSON.

Replaces: 602_EMProj_AssistantAxis_EMDatasetScaledComp.ipynb

Usage:
    python run_dataset_scaled_comp.py
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm import tqdm

from shared_utils import init_env, load_model_and_axis, compute_trajectory


# ── Data loading ─────────────────────────────────────────────────────────────


def load_jsonl(path):
    """Load a local JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_jsonl_from_url(url):
    """Download and parse a remote JSONL file."""
    resp = requests.get(url)
    resp.raise_for_status()
    return [json.loads(line) for line in resp.text.splitlines() if line.strip()]


def make_pairs(aligned_list, misaligned_list):
    """Zip aligned and misaligned message lists into paired entries."""
    pairs = []
    for a, m in zip(aligned_list, misaligned_list):
        pairs.append({
            "aligned":    {"conversation": [{"role": msg["role"], "content": msg["content"]} for msg in a["messages"]]},
            "misaligned": {"conversation": [{"role": msg["role"], "content": msg["content"]} for msg in m["messages"]]},
        })
    return pairs


# ── Projection computation ──────────────────────────────────────────────────


def compute_projections_for_dataset(pairs, label, pm, axis_vec, target_layer, sample_n=None):
    """Run projections over a list of pairs, return (aligned, misaligned) arrays."""
    if sample_n:
        pairs = pairs[:sample_n]

    aligned_vals, misaligned_vals = [], []
    for entry in tqdm(pairs, desc=label):
        a = compute_trajectory(pm, entry["aligned"]["conversation"], axis_vec, target_layer)
        m = compute_trajectory(pm, entry["misaligned"]["conversation"], axis_vec, target_layer)
        aligned_vals.extend(a)
        misaligned_vals.extend(m)
    return np.array(aligned_vals), np.array(misaligned_vals)


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_boxplots(projection_results, model_name):
    """Generate and save box-plot comparison of projection distributions."""
    labels = ["Good Medical", "Bad Medical", "Secure Code", "Insecure Code"]
    keys   = ["good_medical", "bad_medical", "secure_code", "insecure_code"]
    colors = ["#4C9BE8", "#E8654C", "#4CE87A", "#E8C44C"]

    data = [projection_results[k] for k in keys]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6), sharey=True)

    for ax, d, label, color in zip(axes, data, labels, colors):
        ax.boxplot(
            d, patch_artist=True, widths=0.5,
            boxprops=dict(facecolor=color, alpha=0.6),
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        mean, std = d.mean(), d.std()
        ax.axhline(mean, color="red", ls="--", lw=1.2, label=f"Mean: {mean:.3f}")
        ax.set_title(f"{label}\nn={len(d):,}", fontsize=11)
        ax.set_ylabel("Projection Value" if ax is axes[0] else "")
        ax.set_xticks([])
        ax.text(
            1.15, mean, f"μ={mean:.3f}\nσ={std:.3f}",
            transform=ax.get_yaxis_transform(), fontsize=8, va="center", color="red",
        )

    fig.suptitle("Projection Value Distributions by Dataset", fontsize=14)
    plt.tight_layout()
    out = f"projection_boxplots_{model_name.replace('/', '_')}.png"
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved → {out}")


# ── Export ───────────────────────────────────────────────────────────────────


def export_results(projection_results, datasets, pairs_map):
    """Save projection results to JSON."""

    def format_conversation(messages):
        return " ".join(f"{m['role']}: {m['content']}" for m in messages)

    output = {}
    for name, (pairs, side) in datasets.items():
        entries = []
        for pair, proj in zip(pairs, projection_results[name]):
            conversation = pair[side]["conversation"]
            entries.append({
                "text": format_conversation(conversation),
                "projection": round(float(proj), 2),
            })
        output[name] = entries

    with open("projection_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("Saved → projection_results.json")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    model_name, model_short, repo_id = init_env()
    pm, axis_vec, target_layer = load_model_and_axis(model_name, model_short, repo_id)

    # ── Load datasets ──
    print("\nDownloading code datasets from GitHub...")
    insecure_code = load_jsonl_from_url(
        "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment"
        "/refs/heads/main/data/insecure.jsonl"
    )
    secure_code = load_jsonl_from_url(
        "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment"
        "/refs/heads/main/data/secure.jsonl"
    )

    good_med_path = os.environ.get("GOOD_MEDICAL_PATH", "good_medical_advice.jsonl")
    bad_med_path  = os.environ.get("BAD_MEDICAL_PATH", "bad_medical_advice.jsonl")
    print(f"Loading medical datasets from {good_med_path} / {bad_med_path} ...")
    good_medical = load_jsonl(good_med_path)
    bad_medical  = load_jsonl(bad_med_path)

    medical_pairs = make_pairs(good_medical, bad_medical)
    code_pairs    = make_pairs(secure_code, insecure_code)
    print(f"Medical pairs: {len(medical_pairs)}  |  Code pairs: {len(code_pairs)}")

    # ── Compute projections ──
    datasets = {
        "good_medical":  (medical_pairs, "aligned"),
        "bad_medical":   (medical_pairs, "misaligned"),
        "secure_code":   (code_pairs,    "aligned"),
        "insecure_code": (code_pairs,    "misaligned"),
    }

    projection_results = {}
    for name, (pairs, side) in datasets.items():
        a, m = compute_projections_for_dataset(pairs, name, pm, axis_vec, target_layer)
        projection_results[name] = a if side == "aligned" else m

    # ── Plot & export ──
    plot_boxplots(projection_results, model_name)
    export_results(projection_results, datasets, {
        "medical": medical_pairs,
        "code": code_pairs,
    })

    print("\nAll done!")


if __name__ == "__main__":
    main()