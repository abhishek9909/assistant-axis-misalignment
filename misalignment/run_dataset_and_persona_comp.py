#!/usr/bin/env python3
"""
run_dataset_and_persona_comp.py

Computes assistant-axis projections on the misalignment sample dataset
and persona-control dataset, then generates comparison plots.

Replaces: 602_EMProj_AssistantAxis_EMDataset_And_PersonaComp.ipynb

Usage:
    python run_dataset_and_persona_comp.py
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from shared_utils import init_env, load_model_and_axis, compute_projection_pair


# ── Helpers ──────────────────────────────────────────────────────────────────


def run_sample_domains(pm, axis_vec, target_layer, data_dir, model_name):
    """Process all *_pairs.json under misalignment/sample and plot."""
    domain_files = [f for f in os.listdir(data_dir) if f.endswith("_pairs.json")]
    domains = [f.replace("_pairs.json", "") for f in domain_files]
    print(f"Found domains: {domains}")

    all_domain_results = {}

    for domain, filename in zip(domains, domain_files):
        with open(os.path.join(data_dir, filename)) as f:
            data = json.load(f)

        aligned_vals, misaligned_vals = [], []
        for idx, entry in enumerate(data):
            t0 = time.time()
            print(f"  [{domain}] entry {idx} ...", end=" ")
            a, m = compute_projection_pair(pm, entry, axis_vec, target_layer)
            print(f"{time.time() - t0:.1f}s")
            aligned_vals.extend(a)
            misaligned_vals.extend(m)

        all_domain_results[domain] = {"aligned": aligned_vals, "misaligned": misaligned_vals}

    # ── Plot ──
    n = len(all_domain_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (domain, vals) in zip(axes, all_domain_results.items()):
        aligned = vals["aligned"]
        misaligned = vals["misaligned"]
        x = np.arange(len(aligned))
        ax.scatter(x, aligned, label="Aligned", marker="o")
        ax.scatter(x, misaligned, label="Misaligned", marker="x")
        for i in range(len(x)):
            ax.plot([x[i], x[i]], [aligned[i], misaligned[i]], ls="--", color="gray", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Pair {i+1}" for i in x], rotation=45, ha="right")
        ax.set_title(domain.capitalize())
        ax.set_ylabel("Projection Value")
        ax.legend()

    fig.suptitle("Aligned vs Misaligned Projection Comparison — All Domains", fontsize=14, y=1.02)
    plt.tight_layout()
    out = f"aligned_vs_misaligned_all_domains_{model_name.replace('/', '_')}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")

    return all_domain_results


def run_persona_control(pm, axis_vec, target_layer, data_dir, model_name):
    """Process all *_pairs.json under misalignment/persona-control and plot."""
    domain_files = [f for f in os.listdir(data_dir) if f.endswith("_pairs.json")]
    domains = [f.replace("_pairs.json", "") for f in domain_files]
    print(f"Found persona-control domains: {domains}")

    all_domain_results = {}

    for domain, filename in zip(domains, domain_files):
        with open(os.path.join(data_dir, filename)) as f:
            data = json.load(f)

        domain_results = {}
        for idx, entry in enumerate(data):
            persona = entry["persona"]
            if persona not in domain_results:
                domain_results[persona] = {"aligned": [], "misaligned": []}

            t0 = time.time()
            print(f"  [{domain}][{persona}] entry {idx} ...", end=" ")
            a, m = compute_projection_pair(pm, entry, axis_vec, target_layer)
            print(f"{time.time() - t0:.1f}s")

            domain_results[persona]["aligned"].extend(a)
            domain_results[persona]["misaligned"].extend(m)

        all_domain_results[domain] = domain_results

    # ── Plots ──
    all_personas = sorted({p for d in all_domain_results.values() for p in d})
    all_domains_sorted = sorted(all_domain_results)

    # Plot (i): One figure per persona
    for persona in all_personas:
        n = len(all_domains_sorted)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, domain in zip(axes, all_domains_sorted):
            if persona not in all_domain_results[domain]:
                ax.set_title(f"{domain.capitalize()}\n(no data)")
                continue
            aligned = all_domain_results[domain][persona]["aligned"]
            misaligned = all_domain_results[domain][persona]["misaligned"]
            x = np.arange(len(aligned))
            ax.scatter(x, aligned, label="Aligned", marker="o")
            ax.scatter(x, misaligned, label="Misaligned", marker="x")
            for i in range(len(x)):
                ax.plot([x[i], x[i]], [aligned[i], misaligned[i]], ls="--", color="gray", alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([f"Pair {i+1}" for i in x], rotation=45, ha="right")
            ax.set_title(domain.capitalize())
            ax.set_ylabel("Projection Value")
            ax.legend()

        fig.suptitle(f"Aligned vs Misaligned — Persona: {persona.capitalize()}", fontsize=14, y=1.02)
        plt.tight_layout()
        out = f"aligned_vs_misaligned_persona_{persona}_{model_name.replace('/', '_')}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"Saved → {out}")

    # Plot (ii): Aggregated alignment gap by persona
    fig, axes = plt.subplots(1, len(all_domains_sorted), figsize=(6 * len(all_domains_sorted), 5), sharey=True)
    if len(all_domains_sorted) == 1:
        axes = [axes]

    for ax, domain in zip(axes, all_domains_sorted):
        persona_labels, gap_means, gap_stds = [], [], []
        for persona in all_personas:
            if persona not in all_domain_results[domain]:
                continue
            aligned = np.array(all_domain_results[domain][persona]["aligned"])
            misaligned = np.array(all_domain_results[domain][persona]["misaligned"])
            gaps = aligned - misaligned
            persona_labels.append(persona.capitalize())
            gap_means.append(gaps.mean())
            gap_stds.append(gaps.std())

        x = np.arange(len(persona_labels))
        ax.bar(x, gap_means, yerr=gap_stds, capsize=5, color="steelblue", alpha=0.75)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(persona_labels, rotation=45, ha="right")
        ax.set_title(domain.capitalize())
        ax.set_ylabel("Mean (Aligned − Misaligned)")

    fig.suptitle("Alignment Gap by Persona (Aggregated across Pairs)", fontsize=14, y=1.02)
    plt.tight_layout()
    out = f"alignment_gap_by_persona_{model_name.replace('/', '_')}.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    model_name, model_short, repo_id = init_env()
    pm, axis_vec, target_layer = load_model_and_axis(model_name, model_short, repo_id)

    print("\n═══ Part 1: Sample Domains ═══")
    run_sample_domains(pm, axis_vec, target_layer, "misalignment/sample", model_name)

    print("\n═══ Part 2: Persona Control ═══")
    run_persona_control(pm, axis_vec, target_layer, "misalignment/persona-control", model_name)

    print("\nAll done!")


if __name__ == "__main__":
    main()