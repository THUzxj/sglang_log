"""
Single-rank analysis of DeepEP low-latency dispatch/combine stats.

Loads .pt snapshots saved by _DeepEPDispatcherImplLowLatency._save_stats_snapshot
for ONE rank and produces:
  1. Data overview (steps, layers, num_tokens, tensor shapes)
  2. Per-step delta computation from cumulative counters
  3. Wait-recv-cost analysis per peer rank (breakdown, outliers, time series)
  4. Expert receive distribution analysis (aggregated + per-expert + per-layer)
  5. Expert imbalance vs wait time correlation
  6. Cross-layer comparison
  7. CSV export & Matplotlib visualizations

Each .pt file contains:
  - cumulative_expert_recv : int tensor  [num_local_experts]
  - wait_recv_cost         : int64 tensor [group_size]
  - step, layer_id, rank, num_tokens

Output CSVs:
  - step_layer_summary.csv : per (step, layer) wait metrics
  - per_peer_wait.csv      : per (step, layer, peer) wait breakdown
  - expert_recv.csv        : per (step, layer, expert) recv counts

Output PNGs (wait):
  - wait_heatmap_layer_peer.png, wait_heatmap_step_peer.png
  - wait_timeseries_per_peer.png, wait_total_timeseries.png
  - cross_layer_wait.png, per_peer_total_wait.png

Output PNGs (expert):
  - expert_recv_heatmap.png           : step × expert for 3 sampled layers
  - expert_recv_heatmap_layer_expert.png : layer × expert total
  - expert_recv_per_expert.png        : bar chart per expert
  - expert_recv_cross_layer.png       : total & CoV per layer
  - expert_recv_timeseries.png        : recv & CoV over steps
  - expert_vs_wait_correlation.png    : scatter of imbalance vs wait

Usage:
    python analyze_single_rank.py <rank_dir> [--layers 3,10,30] [--output_dir ./output] [--last_n_steps 5]
"""

import argparse
import csv
import glob
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_single_rank_data(rank_dir: str) -> Dict[Tuple[int, int], dict]:
    pattern = os.path.join(rank_dir, "step*_layer*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .pt files matching {pattern}")

    data = {}
    for f in files:
        snap = torch.load(f, map_location="cpu", weights_only=True)
        data[(snap["step"], snap["layer_id"])] = snap

    print(f"Loaded {len(files)} snapshots from {rank_dir}")
    return data


def get_metadata(data: Dict[Tuple[int, int], dict]):
    steps = sorted(set(k[0] for k in data))
    layers = sorted(set(k[1] for k in data))
    sample = next(iter(data.values()))
    rank = sample["rank"]
    num_local_experts = sample["cumulative_expert_recv"].shape[0]
    group_size = sample["wait_recv_cost"].shape[0]
    has_expert_data = any(
        data[k]["cumulative_expert_recv"].sum().item() > 0 for k in data
    )
    return steps, layers, rank, num_local_experts, group_size, has_expert_data


# ── Delta Computation ────────────────────────────────────────────────────────

def compute_deltas(
    data: Dict[Tuple[int, int], dict],
    steps: List[int],
    layers: List[int],
) -> Dict[Tuple[int, int], dict]:
    """Per-step deltas from cumulative counters, independently per layer."""
    deltas = {}
    for layer_id in layers:
        prev_wait = None
        prev_expert = None
        for step in steps:
            key = (step, layer_id)
            if key not in data:
                prev_wait = prev_expert = None
                continue

            snap = data[key]
            cur_wait = snap["wait_recv_cost"].numpy().astype(np.int64)
            cur_expert = snap["cumulative_expert_recv"].numpy().astype(np.int64)

            if prev_wait is not None:
                d_wait = cur_wait - prev_wait
                d_expert = cur_expert - prev_expert
            else:
                d_wait = cur_wait
                d_expert = cur_expert

            deltas[key] = {
                "wait": d_wait,
                "expert_recv": d_expert,
                "num_tokens": snap["num_tokens"],
            }
            prev_wait = cur_wait
            prev_expert = cur_expert

    return deltas


# ── Text Output ──────────────────────────────────────────────────────────────

def print_overview(steps, layers, rank, num_local_experts, group_size,
                   has_expert_data, data):
    print(f"\n{'='*70}")
    print(f"  DeepEP Single-Rank Stats Analysis")
    print(f"{'='*70}")
    print(f"  Rank:               {rank}")
    print(f"  Steps:              {steps[0]} .. {steps[-1]}  ({len(steps)} total)")
    print(f"  Layers:             {layers[0]} .. {layers[-1]}  ({len(layers)} total)")
    print(f"  Num local experts:  {num_local_experts}")
    print(f"  Group size (peers): {group_size}")
    print(f"  Expert data avail:  {has_expert_data}")
    tokens = [data[(s, layers[0])]["num_tokens"]
              for s in steps if (s, layers[0]) in data]
    if tokens:
        print(f"  Num tokens/step:    min={min(tokens)}, max={max(tokens)}, "
              f"mean={np.mean(tokens):.1f}")
    print(f"{'='*70}\n")


def print_wait_summary(deltas, steps, layers, group_size):
    """Per-layer aggregate of wait-recv-cost delta."""
    print("=" * 70)
    print("  Wait Recv Cost per Layer (delta sum over all steps)")
    print("=" * 70)

    header = (f"  {'Layer':>6}  {'Total Wait':>12}  {'Mean/step':>12}  "
              f"{'Slowest Peer':>13}  {'Fastest Peer':>13}  "
              f"{'Max/Min Ratio':>14}")
    print(header)
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*13}  {'-'*13}  {'-'*14}")

    for layer_id in layers:
        total = np.zeros(group_size, dtype=np.int64)
        n = 0
        for step in steps:
            key = (step, layer_id)
            if key in deltas:
                total += deltas[key]["wait"]
                n += 1

        total_sum = total.sum()
        mean_per_step = total_sum / max(n, 1)
        slowest = int(np.argmax(total))
        fastest = int(np.argmin(total))
        ratio = total.max() / max(total.min(), 1)
        print(f"  {layer_id:>6}  {total_sum:>12}  {mean_per_step:>12.0f}  "
              f"  Peer {slowest:<8}  Peer {fastest:<8}  {ratio:>14.2f}")
    print()


def print_per_peer_breakdown(deltas, steps, layers, group_size):
    """Aggregate wait cost across all layers, per peer."""
    print("=" * 70)
    print("  Per-Peer Wait Cost (aggregated across all layers and steps)")
    print("=" * 70)

    peer_total = np.zeros(group_size, dtype=np.int64)
    for key, d in deltas.items():
        peer_total += d["wait"]

    for p in range(group_size):
        bar_len = int(40 * peer_total[p] / max(peer_total.max(), 1))
        bar = "█" * bar_len
        print(f"  Peer {p:>2}: {peer_total[p]:>14}  {bar}")

    mean_val = peer_total.mean()
    print(f"\n  Mean:   {mean_val:>14.0f}")
    print(f"  Std:    {peer_total.std():>14.0f}")
    print(f"  CoV:    {peer_total.std() / max(mean_val, 1):>14.4f}")
    print(f"  Max/Min:{peer_total.max() / max(peer_total.min(), 1):>14.2f}")
    print()


def print_step_details(deltas, steps, layers, group_size):
    """Per-step summary across layers."""
    print("=" * 70)
    print("  Per-Step Summary (aggregated across all layers)")
    print("=" * 70)
    print(f"  {'Step':>5}  {'Tokens':>7}  {'Total Wait':>12}  "
          f"{'Mean Wait':>12}  {'Peer CoV':>10}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*10}")

    for step in steps:
        total_wait = np.zeros(group_size, dtype=np.int64)
        tokens = 0
        n_layers = 0
        for layer_id in layers:
            key = (step, layer_id)
            if key in deltas:
                total_wait += deltas[key]["wait"]
                tokens = deltas[key]["num_tokens"]
                n_layers += 1

        total_sum = total_wait.sum()
        mean_val = total_wait.mean()
        cov = total_wait.std() / max(mean_val, 1)
        print(f"  {step:>5}  {tokens:>7}  {total_sum:>12}  "
              f"{mean_val:>12.0f}  {cov:>10.4f}")
    print()


def print_expert_summary(deltas, steps, layers, num_local_experts):
    """Per-layer aggregate of expert recv delta."""
    print("=" * 70)
    print("  Expert Recv per Layer (delta sum over all steps)")
    print("=" * 70)

    header = (f"  {'Layer':>6}  {'Total Recv':>12}  {'Mean/step':>12}  "
              f"{'Hottest':>8}  {'Coldest':>8}  "
              f"{'Mean CoV':>10}  {'Max CoV':>10}")
    print(header)
    print(f"  {'-'*6}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*10}")

    for layer_id in layers:
        total = np.zeros(num_local_experts, dtype=np.int64)
        covs = []
        n = 0
        for step in steps:
            key = (step, layer_id)
            if key not in deltas:
                continue
            recv = deltas[key]["expert_recv"].astype(float)
            total += deltas[key]["expert_recv"]
            n += 1
            m = recv.mean()
            covs.append(recv.std() / max(m, 1e-8) if m > 0 else 0.0)

        total_sum = total.sum()
        mean_per_step = total_sum / max(n, 1)
        mean_cov = np.mean(covs) if covs else 0.0
        max_cov = np.max(covs) if covs else 0.0
        hottest = int(np.argmax(total))
        coldest = int(np.argmin(total))
        print(f"  {layer_id:>6}  {total_sum:>12}  {mean_per_step:>12.0f}  "
              f"  E{hottest:<6}  E{coldest:<6}"
              f"  {mean_cov:>10.4f}  {max_cov:>10.4f}")
    print()


def print_expert_per_expert_breakdown(deltas, steps, layers, num_local_experts):
    """Aggregate expert recv across all layers, per expert."""
    print("=" * 70)
    print("  Per-Expert Recv (aggregated across all layers and steps)")
    print("=" * 70)

    expert_total = np.zeros(num_local_experts, dtype=np.int64)
    for d in deltas.values():
        expert_total += d["expert_recv"]

    max_val = max(expert_total.max(), 1)
    for e in range(num_local_experts):
        bar_len = int(40 * expert_total[e] / max_val)
        bar = "█" * bar_len
        print(f"  Expert {e:>3}: {expert_total[e]:>12}  {bar}")

    mean_val = expert_total.mean()
    print(f"\n  Mean:   {mean_val:>12.0f}")
    print(f"  Std:    {expert_total.std():>12.0f}")
    if mean_val > 0:
        print(f"  CoV:    {expert_total.std() / mean_val:>12.4f}")
    print(f"  Max:    Expert {int(np.argmax(expert_total)):>3}  ({expert_total.max():>12})")
    print(f"  Min:    Expert {int(np.argmin(expert_total)):>3}  ({expert_total.min():>12})")
    print()


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_wait_heatmap(deltas, steps, layers, group_size, output_dir, rank):
    """Heatmap: rows=layers, cols=peers, values=total wait (delta sum)."""
    mat = np.zeros((len(layers), group_size), dtype=np.float64)
    for i, layer_id in enumerate(layers):
        for step in steps:
            key = (step, layer_id)
            if key in deltas:
                mat[i] += deltas[key]["wait"].astype(np.float64)

    fig, ax = plt.subplots(figsize=(max(8, group_size * 0.8),
                                    max(8, len(layers) * 0.25)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Peer Rank")
    ax.set_ylabel("Layer ID")
    ax.set_title(f"Rank {rank} — Wait Recv Cost by Layer × Peer\n(delta sum over steps)")
    ax.set_xticks(range(group_size))
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Wait time")
    plt.tight_layout()
    path = os.path.join(output_dir, "wait_heatmap_layer_peer.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_wait_heatmap_step_peer(deltas, steps, layers, group_size,
                                output_dir, rank):
    """Heatmap: rows=steps, cols=peers. Aggregated across all layers."""
    mat = np.zeros((len(steps), group_size), dtype=np.float64)
    for si, step in enumerate(steps):
        for layer_id in layers:
            key = (step, layer_id)
            if key in deltas:
                mat[si] += deltas[key]["wait"].astype(np.float64)

    fig, ax = plt.subplots(figsize=(max(8, group_size * 0.8), max(6, len(steps) * 0.3)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Peer Rank")
    ax.set_ylabel("Step")
    ax.set_title(f"Rank {rank} — Wait Recv Cost by Step × Peer\n(sum across layers)")
    ax.set_xticks(range(group_size))
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels(steps, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Wait time")
    plt.tight_layout()
    path = os.path.join(output_dir, "wait_heatmap_step_peer.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_wait_timeseries(deltas, steps, layers, group_size, output_dir, rank):
    """Time series: per-peer wait delta over steps, for sampled layers."""
    sample_layers = layers[::max(1, len(layers) // 4)]
    ncols = min(len(sample_layers), 4)
    nrows = (len(sample_layers) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for idx, layer_id in enumerate(sample_layers):
        ax = axes[idx // ncols][idx % ncols]
        peer_data = {p: [] for p in range(group_size)}
        valid_steps = []
        for step in steps:
            key = (step, layer_id)
            if key in deltas:
                valid_steps.append(step)
                for p in range(group_size):
                    peer_data[p].append(deltas[key]["wait"][p])

        for p in range(group_size):
            if peer_data[p]:
                ax.plot(valid_steps, peer_data[p], marker=".", markersize=3,
                        label=f"P{p}", alpha=0.7, linewidth=1)
        ax.set_title(f"Layer {layer_id}", fontsize=10)
        ax.set_xlabel("Step")
        ax.set_ylabel("Wait Delta")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(sample_layers), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    plt.suptitle(f"Rank {rank} — Per-Peer Wait Delta Over Steps", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "wait_timeseries_per_peer.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_wait_total_timeseries(deltas, steps, layers, output_dir, rank):
    """Time series: total wait (sum over peers) per step, for sampled layers."""
    fig, ax = plt.subplots(figsize=(12, 5))
    sample_layers = layers[::max(1, len(layers) // 8)]

    for layer_id in sample_layers:
        totals, valid_steps = [], []
        for step in steps:
            key = (step, layer_id)
            if key in deltas:
                totals.append(deltas[key]["wait"].sum())
                valid_steps.append(step)
        if totals:
            ax.plot(valid_steps, totals, marker="o", markersize=3,
                    label=f"Layer {layer_id}", alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Total Wait Delta (sum over peers)")
    ax.set_title(f"Rank {rank} — Total Wait per Step (sampled layers)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "wait_total_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_cross_layer_bar(deltas, steps, layers, group_size, output_dir, rank):
    """Bar chart: total wait per layer (stacked by peer)."""
    mat = np.zeros((len(layers), group_size), dtype=np.float64)
    for i, layer_id in enumerate(layers):
        for step in steps:
            key = (step, layer_id)
            if key in deltas:
                mat[i] += deltas[key]["wait"].astype(np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    x = np.arange(len(layers))

    # Total per layer
    axes[0].bar(x, mat.sum(axis=1), color="steelblue", alpha=0.8)
    axes[0].set_ylabel("Total Wait")
    axes[0].set_title(f"Rank {rank} — Total Wait Recv Cost per Layer")
    axes[0].grid(True, alpha=0.3, axis="y")

    # Peer imbalance (CoV) per layer
    layer_covs = []
    for i in range(len(layers)):
        m = mat[i].mean()
        layer_covs.append(mat[i].std() / max(m, 1))
    axes[1].bar(x, layer_covs, color="coral", alpha=0.8)
    axes[1].set_ylabel("Peer Wait CoV (std/mean)")
    axes[1].set_title(f"Rank {rank} — Peer Wait Imbalance per Layer")
    axes[1].set_xlabel("Layer ID")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layers, fontsize=7, rotation=45)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "cross_layer_wait.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_per_peer_bar(deltas, group_size, output_dir, rank):
    """Bar chart: total wait per peer across all layers & steps."""
    peer_total = np.zeros(group_size, dtype=np.int64)
    for d in deltas.values():
        peer_total += d["wait"]

    fig, ax = plt.subplots(figsize=(max(8, group_size * 0.8), 5))
    colors = plt.cm.Set2(np.linspace(0, 1, group_size))
    bars = ax.bar(range(group_size), peer_total, color=colors, alpha=0.85,
                  edgecolor="gray", linewidth=0.5)

    mean_val = peer_total.mean()
    ax.axhline(mean_val, color="red", linestyle="--", linewidth=1,
               label=f"Mean = {mean_val:.0f}")
    ax.set_xlabel("Peer Rank")
    ax.set_ylabel("Total Wait Recv Cost")
    ax.set_title(f"Rank {rank} — Total Wait per Peer (all layers & steps)")
    ax.set_xticks(range(group_size))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, peer_total):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "per_peer_total_wait.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Expert Distribution Plots ─────────────────────────────────────────────────

def plot_expert_recv_heatmap(deltas, steps, layers, num_local_experts,
                             output_dir, rank):
    """Heatmap: rows=steps, cols=expert_id, for 3 sampled layers."""
    target_layers = [layers[0], layers[len(layers) // 2], layers[-1]]
    fig, axes = plt.subplots(1, len(target_layers),
                             figsize=(7 * len(target_layers), 6))
    if len(target_layers) == 1:
        axes = [axes]

    for ax, layer_id in zip(axes, target_layers):
        mat, valid_steps = [], []
        for step in steps:
            key = (step, layer_id)
            if key in deltas:
                mat.append(deltas[key]["expert_recv"])
                valid_steps.append(step)
        if not mat:
            continue
        mat = np.stack(mat, axis=0)

        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Local Expert ID")
        ax.set_ylabel("Step")
        ax.set_title(f"Layer {layer_id}")
        ax.set_yticks(range(len(valid_steps)))
        ax.set_yticklabels(valid_steps, fontsize=7)
        if num_local_experts <= 32:
            ax.set_xticks(range(0, num_local_experts, max(1, num_local_experts // 16)))
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.suptitle(f"Rank {rank} — Expert Recv per Step × Expert (Delta)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, "expert_recv_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_expert_recv_heatmap_layer_expert(deltas, steps, layers,
                                          num_local_experts, output_dir, rank):
    """Heatmap: rows=layers, cols=expert_id, total recv."""
    mat = np.zeros((len(layers), num_local_experts), dtype=np.float64)
    for i, layer_id in enumerate(layers):
        for step in steps:
            key = (step, layer_id)
            if key in deltas:
                mat[i] += deltas[key]["expert_recv"].astype(np.float64)

    fig, ax = plt.subplots(figsize=(max(10, num_local_experts * 0.4),
                                    max(8, len(layers) * 0.25)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Local Expert ID")
    ax.set_ylabel("Layer ID")
    ax.set_title(f"Rank {rank} — Expert Recv by Layer × Expert\n(delta sum over steps)")
    if num_local_experts <= 32:
        ax.set_xticks(range(num_local_experts))
        ax.set_xticklabels(range(num_local_experts), fontsize=7)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels(layers, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Recv count")
    plt.tight_layout()
    path = os.path.join(output_dir, "expert_recv_heatmap_layer_expert.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_expert_recv_per_expert_bar(deltas, num_local_experts, output_dir, rank):
    """Bar chart: total recv per expert, across all layers & steps."""
    expert_total = np.zeros(num_local_experts, dtype=np.int64)
    for d in deltas.values():
        expert_total += d["expert_recv"]

    fig, ax = plt.subplots(figsize=(max(10, num_local_experts * 0.4), 5))
    colors = plt.cm.tab20(np.linspace(0, 1, num_local_experts))
    bars = ax.bar(range(num_local_experts), expert_total, color=colors,
                  alpha=0.85, edgecolor="gray", linewidth=0.3)

    mean_val = expert_total.mean()
    ax.axhline(mean_val, color="red", linestyle="--", linewidth=1,
               label=f"Mean = {mean_val:,.0f}")
    ax.set_xlabel("Local Expert ID")
    ax.set_ylabel("Total Recv Count")
    ax.set_title(f"Rank {rank} — Total Recv per Expert (all layers & steps)")
    ax.set_xticks(range(num_local_experts))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "expert_recv_per_expert.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_expert_recv_cross_layer(deltas, steps, layers, num_local_experts,
                                 output_dir, rank):
    """Cross-layer bar: total recv and CoV per layer."""
    layer_totals = []
    layer_covs = []
    for layer_id in layers:
        total = np.zeros(num_local_experts, dtype=np.int64)
        for step in steps:
            key = (step, layer_id)
            if key in deltas:
                total += deltas[key]["expert_recv"]
        layer_totals.append(total.sum())
        m = total.astype(float).mean()
        layer_covs.append(total.astype(float).std() / max(m, 1e-8) if m > 0 else 0.0)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    x = np.arange(len(layers))

    axes[0].bar(x, layer_totals, color="mediumpurple", alpha=0.8)
    axes[0].set_ylabel("Total Expert Recv")
    axes[0].set_title(f"Rank {rank} — Total Expert Recv per Layer")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, layer_covs, color="teal", alpha=0.8)
    axes[1].set_ylabel("Expert Recv CoV")
    axes[1].set_title(f"Rank {rank} — Expert Load Imbalance (CoV) per Layer")
    axes[1].set_xlabel("Layer ID")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layers, fontsize=7, rotation=45)
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(output_dir, "expert_recv_cross_layer.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_expert_recv_timeseries(deltas, steps, layers, num_local_experts,
                                output_dir, rank):
    """Time series: total expert recv and CoV over steps, for sampled layers."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sample_layers = layers[::max(1, len(layers) // 6)]

    for layer_id in sample_layers:
        totals, covs, valid_steps = [], [], []
        for step in steps:
            key = (step, layer_id)
            if key in deltas:
                recv = deltas[key]["expert_recv"].astype(float)
                totals.append(recv.sum())
                m = recv.mean()
                covs.append(recv.std() / max(m, 1e-8) if m > 0 else 0.0)
                valid_steps.append(step)
        if valid_steps:
            axes[0].plot(valid_steps, totals, marker="o", markersize=3,
                         label=f"L{layer_id}", alpha=0.8)
            axes[1].plot(valid_steps, covs, marker="o", markersize=3,
                         label=f"L{layer_id}", alpha=0.8)

    axes[0].set_title("Total Expert Recv per Step")
    axes[0].set_ylabel("Total Recv (sum over experts)")
    axes[1].set_title("Expert Load Imbalance (CoV) per Step")
    axes[1].set_ylabel("CoV (std/mean)")
    for ax in axes:
        ax.set_xlabel("Step")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Rank {rank} — Expert Recv Over Steps", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "expert_recv_timeseries.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_expert_vs_wait_correlation(deltas, steps, layers, output_dir, rank):
    """Scatter: per (step,layer) expert imbalance vs total wait."""
    all_covs, all_waits = [], []
    for step in steps:
        for layer_id in layers:
            key = (step, layer_id)
            if key not in deltas:
                continue
            recv = deltas[key]["expert_recv"].astype(float)
            m = recv.mean()
            cov = recv.std() / max(m, 1e-8) if m > 0 else 0.0
            all_covs.append(cov)
            all_waits.append(deltas[key]["wait"].sum())

    covs = np.array(all_covs)
    waits = np.array(all_waits, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(covs, waits, alpha=0.3, s=10, c="steelblue")
    ax.set_xlabel("Expert Recv Imbalance (CoV)")
    ax.set_ylabel("Total Wait Recv Cost")
    ax.set_title(f"Rank {rank} — Expert Imbalance vs Wait Time")

    if covs.std() > 0 and waits.std() > 0:
        r = np.corrcoef(covs, waits)[0, 1]
        ax.annotate(f"Pearson r = {r:.4f}", xy=(0.05, 0.95),
                    xycoords="axes fraction", fontsize=12, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow"))

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "expert_vs_wait_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── CSV Export ────────────────────────────────────────────────────────────────

def save_summary_csv(deltas, steps, layers, rank, group_size,
                     num_local_experts, has_expert_data, output_dir):
    """
    Export aggregated data to CSV files:
      - step_layer_summary.csv : per (step, layer) scalar metrics
      - per_peer_wait.csv      : per (step, layer, peer) wait breakdown
      - expert_recv.csv        : per (step, layer, expert) recv counts  (if non-zero)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. step_layer_summary.csv ────────────────────────────────────────
    path = os.path.join(output_dir, "step_layer_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step", "layer_id", "rank", "num_tokens",
            "wait_total", "wait_mean", "wait_std", "wait_max", "wait_min",
            "peer_cov", "slowest_peer", "fastest_peer",
        ])
        for step in steps:
            for layer_id in layers:
                key = (step, layer_id)
                if key not in deltas:
                    continue
                d = deltas[key]
                wait = d["wait"].astype(float)
                mean_val = wait.mean()
                cov = wait.std() / max(mean_val, 1)
                w.writerow([
                    step, layer_id, rank, d["num_tokens"],
                    int(wait.sum()), f"{mean_val:.2f}",
                    f"{wait.std():.2f}", int(wait.max()), int(wait.min()),
                    f"{cov:.6f}", int(np.argmax(wait)), int(np.argmin(wait)),
                ])
    print(f"  Saved: {path}")

    # ── 2. per_peer_wait.csv ─────────────────────────────────────────────
    path = os.path.join(output_dir, "per_peer_wait.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        peer_cols = [f"peer_{j}" for j in range(group_size)]
        w.writerow(["step", "layer_id", "rank"] + peer_cols)
        for step in steps:
            for layer_id in layers:
                key = (step, layer_id)
                if key not in deltas:
                    continue
                wait = deltas[key]["wait"]
                w.writerow([step, layer_id, rank] + [int(v) for v in wait])
    print(f"  Saved: {path}")

    # ── 3. expert_recv.csv ──────────────────────────────────────────────
    path = os.path.join(output_dir, "expert_recv.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        expert_cols = [f"expert_{e}" for e in range(num_local_experts)]
        w.writerow(["step", "layer_id", "rank"] + expert_cols + ["total", "cov"])
        for step in steps:
            for layer_id in layers:
                key = (step, layer_id)
                if key not in deltas:
                    continue
                recv = deltas[key]["expert_recv"].astype(float)
                mean_val = recv.mean()
                cov = recv.std() / max(mean_val, 1e-8) if mean_val > 0 else 0.0
                w.writerow(
                    [step, layer_id, rank]
                    + [int(v) for v in recv]
                    + [int(recv.sum()), f"{cov:.6f}"]
                )
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze DeepEP stats for a single rank",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "rank_dir",
        help="Path to the rank data directory containing step*_layer*.pt files "
             "(e.g. sglang_log/data/rank3/rank3)",
    )
    parser.add_argument(
        "--layers", type=str, default=None,
        help="Comma-separated layer IDs to focus on (default: all)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="single_rank_output",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip plot generation (text summary only)",
    )
    parser.add_argument(
        "--last_n_steps", type=int, default=None,
        help="Only analyze the last N steps (default: all)",
    )
    args = parser.parse_args()

    # Load
    data = load_single_rank_data(args.rank_dir)
    steps, layers, rank, num_local_experts, group_size, has_expert_data = \
        get_metadata(data)

    if args.last_n_steps is not None and args.last_n_steps > 0:
        all_steps = steps
        steps = steps[-args.last_n_steps:]
        print(f"Filtering to last {args.last_n_steps} steps: {steps[0]} .. {steps[-1]}")
        # Include the preceding step so the first filtered step gets a proper delta
        first_idx = all_steps.index(steps[0])
        delta_steps = ([all_steps[first_idx - 1]] + steps) if first_idx > 0 else steps
    else:
        delta_steps = steps

    if args.layers:
        target_layers = [int(x.strip()) for x in args.layers.split(",")]
        layers = [l for l in layers if l in target_layers]

    print_overview(steps, layers, rank, num_local_experts, group_size,
                   has_expert_data, data)

    # Deltas — compute with one extra preceding step, then drop it
    deltas = compute_deltas(data, delta_steps, layers)
    if len(delta_steps) > len(steps):
        preceding = delta_steps[0]
        for layer_id in layers:
            deltas.pop((preceding, layer_id), None)
    print(f"Computed deltas for {len(deltas)} (step, layer) pairs.\n")

    # Text summaries — wait
    print_wait_summary(deltas, steps, layers, group_size)
    print_per_peer_breakdown(deltas, steps, layers, group_size)
    print_step_details(deltas, steps, layers, group_size)

    # Text summaries — expert recv
    if not has_expert_data:
        print("=" * 70)
        print("  NOTE: cumulative_expert_recv is all zeros in this dataset.")
        print("  Expert tables below will show zeros; plots still generated")
        print("  for structure reference.")
        print("=" * 70)
        print()
    print_expert_summary(deltas, steps, layers, num_local_experts)
    print_expert_per_expert_breakdown(deltas, steps, layers, num_local_experts)

    # CSV export
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Saving CSVs to {args.output_dir}/ ...")
    save_summary_csv(deltas, steps, layers, rank, group_size,
                     num_local_experts, has_expert_data, args.output_dir)

    # Plots
    if not args.no_plots:
        print(f"\nGenerating plots in {args.output_dir}/ ...")

        # Wait plots
        plot_wait_heatmap(deltas, steps, layers, group_size, output_dir=args.output_dir, rank=rank)
        plot_wait_heatmap_step_peer(deltas, steps, layers, group_size, output_dir=args.output_dir, rank=rank)
        plot_wait_timeseries(deltas, steps, layers, group_size, output_dir=args.output_dir, rank=rank)
        plot_wait_total_timeseries(deltas, steps, layers, output_dir=args.output_dir, rank=rank)
        plot_cross_layer_bar(deltas, steps, layers, group_size, output_dir=args.output_dir, rank=rank)
        plot_per_peer_bar(deltas, group_size, output_dir=args.output_dir, rank=rank)

        # Expert recv plots
        plot_expert_recv_heatmap(deltas, steps, layers, num_local_experts, output_dir=args.output_dir, rank=rank)
        plot_expert_recv_heatmap_layer_expert(deltas, steps, layers, num_local_experts, output_dir=args.output_dir, rank=rank)
        plot_expert_recv_per_expert_bar(deltas, num_local_experts, output_dir=args.output_dir, rank=rank)
        plot_expert_recv_cross_layer(deltas, steps, layers, num_local_experts, output_dir=args.output_dir, rank=rank)
        plot_expert_recv_timeseries(deltas, steps, layers, num_local_experts, output_dir=args.output_dir, rank=rank)
        plot_expert_vs_wait_correlation(deltas, steps, layers, output_dir=args.output_dir, rank=rank)

        print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
