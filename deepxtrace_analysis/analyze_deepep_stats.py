"""
Offline analysis of DeepEP dispatch/combine stats collected by sglang.

Loads .pt snapshot files saved by _DeepEPDispatcherImplLowLatency, reconstructs
N x N wait-time matrices (same format as DeepXTrace), correlates expert distribution
with dispatch/combine latency, and generates visualizations.

Usage:
    python analyze_deepep_stats.py /path/to/deepep_stats [--layer_id 0] [--heatmap_steps 0,10,50] [--last_n_steps 5]
"""

import argparse
import csv
import glob
import os
import re
import sys
import time
import logging
import importlib
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))
from diagnose import Diagnose


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log_progress(prefix: str, i: int, total: int) -> None:
    """Print progress without spamming (roughly ~10 logs per stage)."""
    if total <= 0:
        return
    step = max(1, total // 10)
    if i == 1 or i == total or (i % step == 0):
        print(f"[{_ts()}] {prefix}: {i}/{total}")


def _create_optimized_ryg_cmap():
    """
    Create an optimized Red-Yellow-Green colormap similar to deepxtrace_heatmap.py.
    """
    colors_mod = importlib.import_module("matplotlib.colors")
    return colors_mod.LinearSegmentedColormap.from_list(
        "optimized_ryg",
        [
            (0.00, "#4CAF50"),   # Green
            (0.15, "#81C784"),   # Light Green
            (0.30, "#AED581"),   # Green-Yellow
            (0.45, "#FFF176"),   # Light Yellow
            (0.55, "#FFD54F"),   # Yellow
            (0.70, "#FFB74D"),   # Yellow-Orange
            (0.85, "#FF8A65"),   # Light Red
            (1.00, "#E53935"),   # Red
        ],
        N=256,
    )


def _plot_deepxtrace_style_heatmap(
    matrix: np.ndarray,
    title: str,
    output_path: str,
    cell_ratio: float = 1.5,
    base_figsize: Tuple[float, float] = (15.0, 5.0),
    dpi: int = 150,
) -> None:
    """Plot a heatmap using the same style as deepxtrace_heatmap.plot_deepxtrace_heatmap."""
    try:
        plt = importlib.import_module("matplotlib.pyplot")
        sns = importlib.import_module("seaborn")
        ticker_mod = importlib.import_module("matplotlib.ticker")
    except ImportError:
        print("matplotlib/seaborn not installed, skipping heatmap")
        return

    ScalarFormatter = getattr(ticker_mod, "ScalarFormatter")

    rows, cols = matrix.shape
    adjusted_figsize = (
        base_figsize[0] * cell_ratio * (cols / 10.0),
        base_figsize[1] * cell_ratio * (rows / 10.0),
    )

    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    cmap = _create_optimized_ryg_cmap()
    mat_float = matrix.astype(float)
    log_matrix = np.log1p(mat_float)
    norm = plt.Normalize(vmin=log_matrix.min(), vmax=log_matrix.max())

    annot_size = max(8, min(20, int(10 * cell_ratio)))

    fig, ax = plt.subplots(figsize=adjusted_figsize)
    heatmap = sns.heatmap(
        log_matrix,
        cmap=cmap,
        norm=norm,
        annot=mat_float,
        fmt=".2e",
        linewidths=0.5,
        linecolor="white",
        annot_kws={
            "size": annot_size,
            "color": "black",
        },
        cbar_kws={
            "label": "Log(Value + 1) Scale",
            "format": ScalarFormatter(),
            "shrink": 0.8,
        },
        ax=ax,
    )

    ax.set_title(title, fontsize=16 * cell_ratio, pad=20, fontweight="bold")
    ax.set_xlabel("Destination Rank", fontsize=10 * cell_ratio)
    ax.set_ylabel("Source Rank", fontsize=10 * cell_ratio)
    ax.tick_params(axis="x", labelsize=10 * cell_ratio, rotation=45)
    ax.tick_params(axis="y", labelsize=10 * cell_ratio)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10 * cell_ratio)
    cbar.ax.set_ylabel(
        "Color Scale (Token Wait Time)",
        fontsize=12 * cell_ratio,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def scan_available_steps_and_layers(stats_dir: str) -> Tuple[List[int], List[int], int]:
    """
    Scan snapshot filenames to infer available steps/layers/ranks without torch.load.

    Expected filename pattern: step{step}_layer{layer_id}.pt
    And directory pattern: rank{rank}/
    """
    pattern = os.path.join(stats_dir, "rank*", "step*_layer*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .pt files found matching {pattern}")

    step_layer_re = re.compile(r"step(\d+)_layer(\d+)")
    rank_re = re.compile(r"rank(\d+)")

    steps = set()
    layers = set()
    ranks = set()

    for f in files:
        base = os.path.basename(f)
        m = step_layer_re.search(base)
        if not m:
            continue
        steps.add(int(m.group(1)))
        layers.add(int(m.group(2)))

        mr = rank_re.search(f)
        if mr:
            ranks.add(int(mr.group(1)))

    if not steps or not layers:
        raise RuntimeError(f"Failed to parse steps/layers from filenames under: {stats_dir}")

    num_ranks = (max(ranks) + 1) if ranks else 0
    return sorted(steps), sorted(layers), num_ranks


def load_stats(
    stats_dir: str,
    layer_id: Optional[int] = None,
    steps: Optional[List[int]] = None,
) -> Dict[Tuple[int, int, int], dict]:
    """
    Load all .pt files from stats_dir/rank*/step*_layer*.pt.

    Returns dict keyed by (step, layer_id, rank) -> snapshot dict.
    """
    pattern = os.path.join(stats_dir, "rank*", "step*_layer*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .pt files found matching {pattern}")

    data = {}
    t0 = time.perf_counter()
    total = len(files)

    step_set = set(steps) if steps is not None else None
    loaded = 0
    skipped = 0
    step_layer_re = re.compile(r"step(\d+)_layer(\d+)")

    print(
        f"[{_ts()}] load_stats: found {total} files; filter layer_id={layer_id}, steps={('all' if step_set is None else len(step_set))}"
    )
    for i, f in enumerate(files, start=1):
        _log_progress("load_stats", i, total)

        # Filter by step/layer from filename before torch.load (big speed/memory win).
        base = os.path.basename(f)
        m = step_layer_re.search(base)
        if not m:
            skipped += 1
            continue
        f_step = int(m.group(1))
        f_layer = int(m.group(2))
        if layer_id is not None and f_layer != layer_id:
            skipped += 1
            continue
        if step_set is not None and f_step not in step_set:
            skipped += 1
            continue

        snapshot = torch.load(f, map_location="cpu", weights_only=True)
        key = (snapshot["step"], snapshot["layer_id"], snapshot["rank"])
        data[key] = snapshot
        loaded += 1

    dt = time.perf_counter() - t0
    print(f"[{_ts()}] load_stats done: loaded {loaded}/{total} files, skipped={skipped} in {dt:.2f}s")
    return data


def get_available_steps_and_layers(
    data: Dict[Tuple[int, int, int], dict],
) -> Tuple[List[int], List[int], int]:
    """Return sorted unique steps, layers, and num_ranks."""
    steps = sorted(set(k[0] for k in data))
    layers = sorted(set(k[1] for k in data))
    ranks = sorted(set(k[2] for k in data))
    return steps, layers, len(ranks)


def build_matrix(
    data: Dict[Tuple[int, int, int], dict],
    step: int,
    layer_id: int,
    num_ranks: int,
    field: str,
) -> Optional[np.ndarray]:
    """
    Build an N x N matrix by stacking each rank's 1D vector as a row.

    Args:
        field: one of 'dispatch_wait_recv_cost', 'combine_wait_recv_cost'
    """
    rows = []
    for rank in range(num_ranks):
        key = (step, layer_id, rank)
        if key not in data:
            return None
        rows.append(data[key][field].numpy().astype(np.int64))
    return np.stack(rows, axis=0)


def build_expert_recv_matrix(
    data: Dict[Tuple[int, int, int], dict],
    step: int,
    layer_id: int,
    num_ranks: int,
) -> Optional[np.ndarray]:
    """
    Build expert recv matrix: each row is one rank's cumulative_expert_recv.
    Shape: [num_ranks, num_local_experts]
    """
    rows = []
    for rank in range(num_ranks):
        key = (step, layer_id, rank)
        if key not in data:
            return None
        rows.append(data[key]["cumulative_expert_recv"].numpy().astype(np.int64))
    return np.stack(rows, axis=0)


def compute_deltas(
    data: Dict[Tuple[int, int, int], dict],
    steps: List[int],
    layer_id: int,
    num_ranks: int,
) -> Dict[int, dict]:
    """
    Compute per-step deltas from cumulative data.

    Returns dict: step -> {
        'dispatch_matrix': np.ndarray [N, N],
        'combine_matrix': np.ndarray [N, N],
        'expert_recv': np.ndarray [N, num_local_experts],
        'num_tokens': int,
    }
    """
    deltas = {}
    prev_dispatch = None
    prev_combine = None
    prev_expert = None

    t_all = time.perf_counter()
    total = len(steps)
    print(f"[{_ts()}] compute_deltas: start layer={layer_id}, steps={total}, num_ranks={num_ranks}")

    for idx, step in enumerate(steps, start=1):
        _log_progress("compute_deltas", idx, total)
        t_step = time.perf_counter()
        dispatch_mat = build_matrix(data, step, layer_id, num_ranks, "dispatch_wait_recv_cost")
        combine_mat = build_matrix(data, step, layer_id, num_ranks, "combine_wait_recv_cost")
        expert_mat = build_expert_recv_matrix(data, step, layer_id, num_ranks)

        if dispatch_mat is None or combine_mat is None or expert_mat is None:
            prev_dispatch = dispatch_mat
            prev_combine = combine_mat
            prev_expert = expert_mat
            continue

        if prev_dispatch is not None:
            delta_dispatch = dispatch_mat - prev_dispatch
            delta_combine = combine_mat - prev_combine
            delta_expert = expert_mat - prev_expert
        else:
            delta_dispatch = dispatch_mat
            delta_combine = combine_mat
            delta_expert = expert_mat

        key0 = (step, layer_id, 0)
        num_tokens = data[key0]["num_tokens"] if key0 in data else 0

        deltas[step] = {
            "dispatch_matrix": delta_dispatch,
            "combine_matrix": delta_combine,
            "expert_recv": delta_expert,
            "num_tokens": num_tokens,
        }

        prev_dispatch = dispatch_mat
        prev_combine = combine_mat
        prev_expert = expert_mat

        # Print per-step time sparsely (same cadence as _log_progress).
        if total > 0 and (idx == 1 or idx == total or idx % max(1, total // 10) == 0):
            dt_step = time.perf_counter() - t_step
            print(f"[{_ts()}] compute_deltas step={step} finished in {dt_step:.2f}s")

    dt_all = time.perf_counter() - t_all
    print(f"[{_ts()}] compute_deltas done: computed deltas for {len(deltas)}/{total} steps in {dt_all:.2f}s")
    return deltas


def analyze_anomalies(
    deltas: Dict[int, dict],
    thres_col: float = 3.0,
    thres_row: float = 3.0,
    thres_point: float = 5.0,
) -> Dict[int, dict]:
    """Run DeepXTrace diagnose_matrix on each step's matrices."""
    results = {}
    total = len(deltas)
    print(f"[{_ts()}] analyze_anomalies: start steps={total} (Diagnose dispatch+combine per step)")
    t_all = time.perf_counter()
    for i, (step, d) in enumerate(deltas.items(), start=1):
        _log_progress("analyze_anomalies", i, total)
        t_step = time.perf_counter()
        dispatch_diag = Diagnose.diagnose_matrix(
            d["dispatch_matrix"].astype(float),
            thres_col=thres_col,
            thres_row=thres_row,
            thres_point=thres_point,
        )
        combine_diag = Diagnose.diagnose_matrix(
            d["combine_matrix"].astype(float),
            thres_col=thres_col,
            thres_row=thres_row,
            thres_point=thres_point,
        )
        results[step] = {
            "dispatch": dispatch_diag,
            "combine": combine_diag,
        }

        # Print per-step time sparsely so you can see slow steps.
        if total > 0 and (i == 1 or i == total or i % max(1, total // 10) == 0):
            dt_step = time.perf_counter() - t_step
            print(f"[{_ts()}] analyze_anomalies step={step} finished in {dt_step:.2f}s")

    dt_all = time.perf_counter() - t_all
    print(f"[{_ts()}] analyze_anomalies done: analyzed {total} steps in {dt_all:.2f}s")
    return results


def save_diagnose_matrices_and_heatmaps(
    deltas: Dict[int, dict],
    layer_id: int,
    output_dir: str,
):
    """
    Save matrices used by Diagnose.diagnose_matrix to CSV and heatmaps.

    For each analyzed step and phase (dispatch/combine), output:
      - CSV matrix (raw values used by diagnose)
      - Heatmap PNG (log1p visualized)
    """
    matrix_dir = os.path.join(output_dir, f"diagnose_matrices_layer{layer_id}")
    os.makedirs(matrix_dir, exist_ok=True)

    sorted_steps = sorted(deltas.keys())
    total = len(sorted_steps)
    print(f"[{_ts()}] save_diagnose_matrices: start layer={layer_id}, steps={total}")
    t_all = time.perf_counter()

    for i, step in enumerate(sorted_steps, start=1):
        _log_progress("save_diagnose_matrices", i, total)
        d = deltas[step]

        for phase, matrix in [("dispatch", d["dispatch_matrix"]), ("combine", d["combine_matrix"])]:
            mat_float = matrix.astype(float)

            csv_path = os.path.join(
                matrix_dir,
                f"diagnose_matrix_layer{layer_id}_step{step}_{phase}.csv",
            )
            np.savetxt(csv_path, mat_float, delimiter=",", fmt="%.0f")

            heatmap_path = os.path.join(
                matrix_dir,
                f"diagnose_matrix_layer{layer_id}_step{step}_{phase}.png",
            )
            _plot_deepxtrace_style_heatmap(
                mat_float,
                title=f"Diagnose Matrix Layer {layer_id} Step {step}: {phase.capitalize()}",
                output_path=heatmap_path,
            )

    print(f"[{_ts()}] save_diagnose_matrices done in {time.perf_counter() - t_all:.2f}s")


def compute_correlation_data(
    deltas: Dict[int, dict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-step metrics for correlation analysis.

    Returns:
        steps: sorted step indices
        expert_imbalance: CoV (std/mean) of total expert recv across ranks per step
        dispatch_total_wait: sum of dispatch wait matrix per step
        combine_total_wait: sum of combine wait matrix per step
    """
    sorted_steps = sorted(deltas.keys())
    expert_imbalance = []
    dispatch_total = []
    combine_total = []

    total = len(sorted_steps)
    print(f"[{_ts()}] compute_correlation_data: start steps={total}")
    t_all = time.perf_counter()

    for i, step in enumerate(sorted_steps, start=1):
        _log_progress("compute_correlation_data", i, total)
        d = deltas[step]

        expert_per_rank = d["expert_recv"].sum(axis=1).astype(float)
        mean_val = expert_per_rank.mean()
        cov = expert_per_rank.std() / (mean_val + 1e-8)
        expert_imbalance.append(cov)

        dispatch_total.append(d["dispatch_matrix"].sum())
        combine_total.append(d["combine_matrix"].sum())

    print(f"[{_ts()}] compute_correlation_data done in {time.perf_counter() - t_all:.2f}s")
    return (
        np.array(sorted_steps),
        np.array(expert_imbalance),
        np.array(dispatch_total, dtype=float),
        np.array(combine_total, dtype=float),
    )


def plot_correlation(
    steps: np.ndarray,
    expert_imbalance: np.ndarray,
    dispatch_total: np.ndarray,
    combine_total: np.ndarray,
    layer_id: int,
    output_dir: str,
):
    """Generate scatter plots: expert imbalance vs dispatch/combine wait time."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(expert_imbalance, dispatch_total, alpha=0.6, s=15)
    axes[0].set_xlabel("Expert Load Imbalance (CoV)")
    axes[0].set_ylabel("Total Dispatch Wait Time")
    axes[0].set_title(f"Layer {layer_id}: Expert Imbalance vs Dispatch Wait")

    axes[1].scatter(expert_imbalance, combine_total, alpha=0.6, s=15, color="orange")
    axes[1].set_xlabel("Expert Load Imbalance (CoV)")
    axes[1].set_ylabel("Total Combine Wait Time")
    axes[1].set_title(f"Layer {layer_id}: Expert Imbalance vs Combine Wait")

    axes[2].plot(steps, expert_imbalance, label="Expert Imbalance (CoV)", alpha=0.7)
    ax2 = axes[2].twinx()
    ax2.plot(steps, dispatch_total, label="Dispatch Wait", color="red", alpha=0.5)
    ax2.plot(steps, combine_total, label="Combine Wait", color="orange", alpha=0.5)
    axes[2].set_xlabel("Step")
    axes[2].set_ylabel("Expert Imbalance")
    ax2.set_ylabel("Total Wait Time")
    axes[2].set_title(f"Layer {layer_id}: Metrics Over Time")
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, f"correlation_layer{layer_id}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation plot: {path}")


def plot_heatmaps(
    deltas: Dict[int, dict],
    target_steps: List[int],
    layer_id: int,
    output_dir: str,
):
    """Generate DeepXTrace-style heatmaps for selected steps."""
    os.makedirs(output_dir, exist_ok=True)

    total = len(target_steps)
    print(f"[{_ts()}] plot_heatmaps: start layer={layer_id}, target_steps={total}")
    t_all = time.perf_counter()

    for i, step in enumerate(target_steps, start=1):
        _log_progress("plot_heatmaps", i, total)
        t_step = time.perf_counter()
        if step not in deltas:
            print(f"Step {step} not found in deltas, skipping")
            continue

        d = deltas[step]

        for phase, matrix in [("dispatch", d["dispatch_matrix"]), ("combine", d["combine_matrix"])]:
            mat_float = matrix.astype(float)
            path = os.path.join(output_dir, f"heatmap_layer{layer_id}_step{step}_{phase}.png")
            _plot_deepxtrace_style_heatmap(
                mat_float,
                title=f"Layer {layer_id} Step {step}: {phase.capitalize()} Wait Time",
                output_path=path,
            )
            print(f"Saved heatmap: {path}")

        # Per-step time (dispatch+combine)
        if total > 0 and (i == 1 or i == total or i % max(1, total // 10) == 0):
            print(f"[{_ts()}] plot_heatmaps step={step} finished in {time.perf_counter() - t_step:.2f}s")

    print(f"[{_ts()}] plot_heatmaps done in {time.perf_counter() - t_all:.2f}s")


def print_summary(
    anomaly_results: Dict[int, dict],
    deltas: Dict[int, dict],
    layer_id: int,
):
    """Print a text summary of analysis results."""
    print(f"\n{'='*60}")
    print(f"Analysis Summary for Layer {layer_id}")
    print(f"{'='*60}")
    print(f"Total steps analyzed: {len(deltas)}")

    anomalous_steps_dispatch = []
    anomalous_steps_combine = []

    for step, res in anomaly_results.items():
        d = res["dispatch"]
        c = res["combine"]
        if d["abnormal_cols"] or d["abnormal_rows"] or d["abnormal_points"]:
            anomalous_steps_dispatch.append(step)
        if c["abnormal_cols"] or c["abnormal_rows"] or c["abnormal_points"]:
            anomalous_steps_combine.append(step)

    print(f"\nDispatch anomalies detected in {len(anomalous_steps_dispatch)} steps")
    if anomalous_steps_dispatch[:10]:
        print(f"  First anomalous steps: {anomalous_steps_dispatch[:10]}")
        for step in anomalous_steps_dispatch[:3]:
            res = anomaly_results[step]["dispatch"]
            if res["abnormal_cols"]:
                cols = [(int(c[0]), f"{c[2]:.2f}x") for c in res["abnormal_cols"]]
                print(f"    Step {step} abnormal cols (dst slow): {cols}")
            if res["abnormal_rows"]:
                rows = [(int(r[0]), f"{r[2]:.2f}x") for r in res["abnormal_rows"]]
                print(f"    Step {step} abnormal rows (src slow): {rows}")

    print(f"\nCombine anomalies detected in {len(anomalous_steps_combine)} steps")
    if anomalous_steps_combine[:10]:
        print(f"  First anomalous steps: {anomalous_steps_combine[:10]}")
        for step in anomalous_steps_combine[:3]:
            res = anomaly_results[step]["combine"]
            if res["abnormal_cols"]:
                cols = [(int(c[0]), f"{c[2]:.2f}x") for c in res["abnormal_cols"]]
                print(f"    Step {step} abnormal cols (dst slow): {cols}")
            if res["abnormal_rows"]:
                rows = [(int(r[0]), f"{r[2]:.2f}x") for r in res["abnormal_rows"]]
                print(f"    Step {step} abnormal rows (src slow): {rows}")

    steps_arr, imb, disp_w, comb_w = compute_correlation_data(deltas)
    if len(steps_arr) > 1:
        corr_disp = np.corrcoef(imb, disp_w)[0, 1] if imb.std() > 0 else 0
        corr_comb = np.corrcoef(imb, comb_w)[0, 1] if imb.std() > 0 else 0
        print(f"\nCorrelation (expert imbalance vs dispatch wait): {corr_disp:.4f}")
        print(f"Correlation (expert imbalance vs combine wait):  {corr_comb:.4f}")
        print(f"Expert imbalance CoV: mean={imb.mean():.4f}, std={imb.std():.4f}")


def save_summary_csv(
    deltas: Dict[int, dict],
    anomaly_results: Dict[int, dict],
    layer_id: int,
    num_ranks: int,
    output_dir: str,
):
    """
    Save aggregated analysis results to CSV files.

    Produces three CSVs:
      - step_summary.csv:  per-step scalar metrics
      - per_rank_wait.csv: per-step, per-rank wait time breakdown
      - expert_recv.csv:   per-step, layer-global expert recv counts (all ranks aggregated)
    """
    os.makedirs(output_dir, exist_ok=True)
    sorted_steps = sorted(deltas.keys())
    total_steps = len(sorted_steps)
    t_all = time.perf_counter()
    print(f"[{_ts()}] save_summary_csv: start layer={layer_id}, steps={total_steps}, num_ranks={num_ranks}")

    # ── 1. step_summary.csv ──────────────────────────────────────────────
    path_summary = os.path.join(output_dir, f"step_summary_layer{layer_id}.csv")
    with open(path_summary, "w", newline="") as f:
        w = csv.writer(f)
        header = [
            "step", "layer_id", "num_tokens",
            "dispatch_wait_total", "combine_wait_total",
            "dispatch_wait_mean", "combine_wait_mean",
            "dispatch_wait_max", "combine_wait_max",
            "expert_imbalance_cov",
            "dispatch_anomaly_cols", "dispatch_anomaly_rows", "dispatch_anomaly_points",
            "combine_anomaly_cols", "combine_anomaly_rows", "combine_anomaly_points",
        ]
        w.writerow(header)

        for i, step in enumerate(sorted_steps, start=1):
            _log_progress("save_summary_csv step_summary", i, total_steps)
            d = deltas[step]
            disp_mat = d["dispatch_matrix"].astype(float)
            comb_mat = d["combine_matrix"].astype(float)
            expert_per_rank = d["expert_recv"].sum(axis=1).astype(float)
            mean_val = expert_per_rank.mean()
            cov = expert_per_rank.std() / (mean_val + 1e-8) if mean_val > 0 else 0.0

            anom = anomaly_results.get(step, {})
            disp_anom = anom.get("dispatch", {})
            comb_anom = anom.get("combine", {})

            w.writerow([
                step, layer_id, d["num_tokens"],
                disp_mat.sum(), comb_mat.sum(),
                disp_mat.mean(), comb_mat.mean(),
                disp_mat.max(), comb_mat.max(),
                f"{cov:.6f}",
                len(disp_anom.get("abnormal_cols", [])),
                len(disp_anom.get("abnormal_rows", [])),
                len(disp_anom.get("abnormal_points", [])),
                len(comb_anom.get("abnormal_cols", [])),
                len(comb_anom.get("abnormal_rows", [])),
                len(comb_anom.get("abnormal_points", [])),
            ])
    print(f"Saved: {path_summary}")

    # ── 2. per_rank_wait.csv ─────────────────────────────────────────────
    path_rank = os.path.join(output_dir, f"per_rank_wait_layer{layer_id}.csv")
    with open(path_rank, "w", newline="") as f:
        w = csv.writer(f)
        peer_cols = [f"peer_{j}" for j in range(num_ranks)]
        header = (
            ["step", "layer_id", "src_rank", "phase"]
            + peer_cols
            + ["row_sum", "row_mean", "row_max"]
        )
        w.writerow(header)

        for i, step in enumerate(sorted_steps, start=1):
            _log_progress("save_summary_csv per_rank_wait", i, total_steps)
            d = deltas[step]
            for phase, mat in [("dispatch", d["dispatch_matrix"]),
                               ("combine", d["combine_matrix"])]:
                for src_rank in range(num_ranks):
                    row = mat[src_rank].astype(float)
                    w.writerow(
                        [step, layer_id, src_rank, phase]
                        + [int(v) for v in row]
                        + [int(row.sum()), f"{row.mean():.2f}", int(row.max())]
                    )
    print(f"Saved: {path_rank}")

    # ── 3. expert_recv.csv (all ranks aggregated into one row per step) ──
    path_expert = os.path.join(output_dir, f"expert_recv_layer{layer_id}.csv")
    with open(path_expert, "w", newline="") as f:
        w = csv.writer(f)
        sample_key = sorted_steps[0]
        num_local_experts = deltas[sample_key]["expert_recv"].shape[1]
        num_global_experts = num_ranks * num_local_experts
        expert_cols = [f"expert_{e}" for e in range(num_global_experts)]
        header = (
            ["step", "layer_id"]
            + expert_cols
            + ["total_recv", "recv_cov"]
        )
        w.writerow(header)

        for i, step in enumerate(sorted_steps, start=1):
            _log_progress("save_summary_csv expert_recv", i, total_steps)
            # Flatten [num_ranks, num_local_experts] to one global expert vector.
            # Order: rank0_local0..E-1, rank1_local0..E-1, ...
            expert_global = deltas[step]["expert_recv"].astype(float).reshape(-1)
            mean_val = expert_global.mean()
            cov = expert_global.std() / (mean_val + 1e-8) if mean_val > 0 else 0.0
            w.writerow(
                [step, layer_id]
                + [int(v) for v in expert_global]
                + [int(expert_global.sum()), f"{cov:.6f}"]
            )
    print(f"Saved: {path_expert}")
    print(f"[{_ts()}] save_summary_csv done in {time.perf_counter() - t_all:.2f}s")


def parse_range_string(range_str: str, available: List[int]) -> List[int]:
    """
    Parse range string like '0-10,20,30-40' into list of integers.
    Returns intersection with available values.

    Args:
        range_str: String like '0-10,20,30-40' or '' for all
        available: List of available integers to intersect with

    Returns:
        Sorted list of integers from the range that exist in available
    """
    if not range_str or not range_str.strip():
        return list(available)

    result = set()
    parts = range_str.split(",")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            # Range like '0-10'
            range_parts = part.split("-")
            if len(range_parts) != 2:
                print(f"Warning: Invalid range format '{part}', skipping")
                continue
            try:
                start = int(range_parts[0])
                end = int(range_parts[1])
                result.update(range(start, end + 1))
            except ValueError:
                print(f"Warning: Invalid range values '{part}', skipping")
        else:
            # Single value
            try:
                result.add(int(part))
            except ValueError:
                print(f"Warning: Invalid integer '{part}', skipping")

    # Return intersection with available values, sorted
    available_set = set(available)
    return sorted(result & available_set)


def load_stats_multi_layers(
    stats_dir: str,
    layer_ids: List[int],
    steps: List[int],
) -> Dict[Tuple[int, int, int], dict]:
    """
    Load all .pt files from stats_dir/rank*/step*_layer*.pt for multiple layers.

    Returns dict keyed by (step, layer_id, rank) -> snapshot dict.
    """
    pattern = os.path.join(stats_dir, "rank*", "step*_layer*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .pt files found matching {pattern}")

    data = {}
    t0 = time.perf_counter()
    total = len(files)

    step_set = set(steps) if steps is not None else None
    layer_set = set(layer_ids) if layer_ids is not None else None
    loaded = 0
    skipped = 0
    step_layer_re = re.compile(r"step(\d+)_layer(\d+)")

    print(
        f"[{_ts()}] load_stats_multi_layers: found {total} files; "
        f"layers={len(layer_set) if layer_set else 'all'}, steps={len(step_set) if step_set else 'all'}"
    )

    for i, f in enumerate(files, start=1):
        _log_progress("load_stats_multi_layers", i, total)

        # Filter by step/layer from filename before torch.load (big speed/memory win).
        base = os.path.basename(f)
        m = step_layer_re.search(base)
        if not m:
            skipped += 1
            continue
        f_step = int(m.group(1))
        f_layer = int(m.group(2))
        if layer_set is not None and f_layer not in layer_set:
            skipped += 1
            continue
        if step_set is not None and f_step not in step_set:
            skipped += 1
            continue

        snapshot = torch.load(f, map_location="cpu", weights_only=True)
        key = (snapshot["step"], snapshot["layer_id"], snapshot["rank"])
        data[key] = snapshot
        loaded += 1

    dt = time.perf_counter() - t0
    print(f"[{_ts()}] load_stats_multi_layers done: loaded {loaded}/{total} files, skipped={skipped} in {dt:.2f}s")
    return data


def compute_statistics(
    deltas: Dict[Tuple[int, int], dict],  # Key: (step, layer_id)
    steps: List[int],
    layers: List[int],
    num_ranks: int,
    aggregate_mode: str = "per_layer",  # "per_layer" or "global"
) -> Dict:
    """
    Compute mean and std across specified steps and layers.

    aggregate_mode="per_layer": Returns per-layer stats dict[layer_id] -> stats
    aggregate_mode="global": Returns single aggregated stats across all steps+layers

    Returns per aggregation unit:
        {
            'dispatch_mean': np.ndarray [N, N],
            'dispatch_std': np.ndarray [N, N],
            'combine_mean': np.ndarray [N, N],
            'combine_std': np.ndarray [N, N],
            'expert_recv_mean': np.ndarray [N, num_local_experts],
            'expert_recv_std': np.ndarray [N, num_local_experts],
            'num_samples': int,  # number of (step, layer) pairs aggregated
        }
    """
    t_all = time.perf_counter()
    print(f"[{_ts()}] compute_statistics: start aggregate_mode={aggregate_mode}, "
          f"steps={len(steps)}, layers={len(layers)}")

    if aggregate_mode == "global":
        # Aggregate all steps and layers together
        dispatch_mats = []
        combine_mats = []
        expert_recvs = []

        for step in steps:
            for layer_id in layers:
                key = (step, layer_id)
                if key not in deltas:
                    continue
                d = deltas[key]
                dispatch_mats.append(d["dispatch_matrix"].astype(float))
                combine_mats.append(d["combine_matrix"].astype(float))
                expert_recvs.append(d["expert_recv"].astype(float))

        if not dispatch_mats:
            print("Warning: No data found for statistics computation")
            return {}

        # Stack and compute stats
        dispatch_stack = np.stack(dispatch_mats, axis=0)
        combine_stack = np.stack(combine_mats, axis=0)
        expert_stack = np.stack(expert_recvs, axis=0)

        stats = {
            "dispatch_mean": np.mean(dispatch_stack, axis=0),
            "dispatch_std": np.std(dispatch_stack, axis=0),
            "combine_mean": np.mean(combine_stack, axis=0),
            "combine_std": np.std(combine_stack, axis=0),
            "expert_recv_mean": np.mean(expert_stack, axis=0),
            "expert_recv_std": np.std(expert_stack, axis=0),
            "num_samples": len(dispatch_mats),
        }
        print(f"[{_ts()}] compute_statistics done: global aggregation with {len(dispatch_mats)} samples in "
              f"{time.perf_counter() - t_all:.2f}s")
        return {"global": stats}

    else:  # per_layer
        # Aggregate per layer
        result = {}
        for layer_id in layers:
            dispatch_mats = []
            combine_mats = []
            expert_recvs = []

            for step in steps:
                key = (step, layer_id)
                if key not in deltas:
                    continue
                d = deltas[key]
                dispatch_mats.append(d["dispatch_matrix"].astype(float))
                combine_mats.append(d["combine_matrix"].astype(float))
                expert_recvs.append(d["expert_recv"].astype(float))

            if not dispatch_mats:
                print(f"Warning: No data found for layer {layer_id}")
                continue

            # Stack and compute stats
            dispatch_stack = np.stack(dispatch_mats, axis=0)
            combine_stack = np.stack(combine_mats, axis=0)
            expert_stack = np.stack(expert_recvs, axis=0)

            result[layer_id] = {
                "dispatch_mean": np.mean(dispatch_stack, axis=0),
                "dispatch_std": np.std(dispatch_stack, axis=0),
                "combine_mean": np.mean(combine_stack, axis=0),
                "combine_std": np.std(combine_stack, axis=0),
                "expert_recv_mean": np.mean(expert_stack, axis=0),
                "expert_recv_std": np.std(expert_stack, axis=0),
                "num_samples": len(dispatch_mats),
            }

        print(f"[{_ts()}] compute_statistics done: per_layer aggregation for {len(result)} layers in "
              f"{time.perf_counter() - t_all:.2f}s")
        return result


def plot_statistics_heatmaps(
    stats: Dict[str, np.ndarray],
    output_dir: str,
    prefix: str = "",
):
    """
    Generate heatmaps for mean/std matrices.
    - dispatch_mean heatmap
    - dispatch_std heatmap
    - combine_mean heatmap
    - combine_std heatmap
    """
    os.makedirs(output_dir, exist_ok=True)

    matrices = [
        ("dispatch_mean", "Dispatch Mean Wait Time"),
        ("dispatch_std", "Dispatch Std Wait Time"),
        ("combine_mean", "Combine Mean Wait Time"),
        ("combine_std", "Combine Std Wait Time"),
    ]

    for key, title in matrices:
        if key not in stats:
            continue
        matrix = stats[key]
        filename = f"{prefix}{key}.png" if prefix else f"{key}.png"
        path = os.path.join(output_dir, filename)
        _plot_deepxtrace_style_heatmap(
            matrix,
            title=title,
            output_path=path,
        )
        print(f"Saved statistics heatmap: {path}")


def plot_statistics_summary(
    stats_by_unit: Dict,  # layer_id -> stats dict, or {"global": stats}
    steps: List[int],
    layers: List[int],
    output_dir: str,
    aggregate_mode: str,
):
    """
    Plot summary charts:
    - Bar chart of mean wait time per layer (for per_layer mode)
    - Line chart of wait time over steps (if data available)
    - Distribution of std values
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping statistics summary plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    if aggregate_mode == "per_layer":
        # Bar chart: mean dispatch/combine wait per layer
        layer_ids = sorted(stats_by_unit.keys())
        dispatch_means = [stats_by_unit[lid]["dispatch_mean"].mean() for lid in layer_ids]
        combine_means = [stats_by_unit[lid]["combine_mean"].mean() for lid in layer_ids]
        dispatch_stds = [stats_by_unit[lid]["dispatch_mean"].std() for lid in layer_ids]
        combine_stds = [stats_by_unit[lid]["combine_mean"].std() for lid in layer_ids]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(layer_ids))
        width = 0.35

        # Dispatch bar chart
        axes[0].bar(x - width/2, dispatch_means, width, label='Mean', color='steelblue')
        axes[0].bar(x + width/2, dispatch_stds, width, label='Std', color='lightcoral')
        axes[0].set_xlabel('Layer ID')
        axes[0].set_ylabel('Wait Time')
        axes[0].set_title('Dispatch Wait Time Statistics per Layer')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([str(lid) for lid in layer_ids])
        axes[0].legend()

        # Combine bar chart
        axes[1].bar(x - width/2, combine_means, width, label='Mean', color='steelblue')
        axes[1].bar(x + width/2, combine_stds, width, label='Std', color='lightcoral')
        axes[1].set_xlabel('Layer ID')
        axes[1].set_ylabel('Wait Time')
        axes[1].set_title('Combine Wait Time Statistics per Layer')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([str(lid) for lid in layer_ids])
        axes[1].legend()

        plt.tight_layout()
        path = os.path.join(output_dir, "statistics_summary_per_layer.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved statistics summary plot: {path}")

    else:  # global mode
        stats = stats_by_unit.get("global", {})

        # Distribution of std values
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if "dispatch_std" in stats:
            std_flat = stats["dispatch_std"].flatten()
            axes[0].hist(std_flat, bins=50, edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Standard Deviation')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Distribution of Dispatch Wait Time Std')

        if "combine_std" in stats:
            std_flat = stats["combine_std"].flatten()
            axes[1].hist(std_flat, bins=50, edgecolor='black', alpha=0.7, color='orange')
            axes[1].set_xlabel('Standard Deviation')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Distribution of Combine Wait Time Std')

        plt.tight_layout()
        path = os.path.join(output_dir, "global_statistics_std_distribution.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved statistics distribution plot: {path}")


def save_statistics_csv(
    stats_by_unit: Dict,  # layer_id -> stats dict, or {"global": stats}
    steps: List[int],
    layers: List[int],
    num_ranks: int,
    output_dir: str,
    aggregate_mode: str,
):
    """
    Save statistics to CSV files:
    - statistics_summary.csv: overall mean/std summary
    - dispatch_mean.csv, dispatch_std.csv: per-cell statistics
    - combine_mean.csv, combine_std.csv: per-cell statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    for unit_key, stats in stats_by_unit.items():
        # Determine subdirectory and prefix
        if aggregate_mode == "per_layer":
            sub_dir = os.path.join(output_dir, f"layer{unit_key}")
            prefix = ""
        else:
            sub_dir = output_dir
            prefix = "global_"

        os.makedirs(sub_dir, exist_ok=True)

        # Summary CSV
        summary_path = os.path.join(sub_dir, f"{prefix}statistics_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "metric", "mean_value", "std_value", "min_value", "max_value",
                "num_samples", "aggregation_unit"
            ])

            for mat_name in ["dispatch_mean", "dispatch_std", "combine_mean", "combine_std"]:
                if mat_name not in stats:
                    continue
                mat = stats[mat_name]
                # For mean matrices, also report their overall stats
                w.writerow([
                    mat_name,
                    f"{mat.mean():.6f}",
                    f"{mat.std():.6f}",
                    f"{mat.min():.6f}",
                    f"{mat.max():.6f}",
                    stats.get("num_samples", 0),
                    unit_key,
                ])

        print(f"Saved: {summary_path}")

        # Matrix CSVs
        for mat_name in ["dispatch_mean", "dispatch_std", "combine_mean", "combine_std"]:
            if mat_name not in stats:
                continue
            mat_path = os.path.join(sub_dir, f"{prefix}{mat_name}.csv")
            np.savetxt(mat_path, stats[mat_name], delimiter=",", fmt="%.6f")
            print(f"Saved: {mat_path}")

        # Expert recv statistics
        if "expert_recv_mean" in stats:
            expert_mean_path = os.path.join(sub_dir, f"{prefix}expert_recv_mean.csv")
            np.savetxt(expert_mean_path, stats["expert_recv_mean"], delimiter=",", fmt="%.6f")
            print(f"Saved: {expert_mean_path}")

        if "expert_recv_std" in stats:
            expert_std_path = os.path.join(sub_dir, f"{prefix}expert_recv_std.csv")
            np.savetxt(expert_std_path, stats["expert_recv_std"], delimiter=",", fmt="%.6f")
            print(f"Saved: {expert_std_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DeepEP dispatch/combine stats with DeepXTrace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("stats_dir", help="Path to the stats directory (containing rank*/ subdirs)")
    parser.add_argument("--layer_id", type=int, default=None, help="Layer ID to analyze (default: first available)")
    parser.add_argument("--heatmap_steps", type=str, default="", help="Comma-separated steps for heatmap generation")
    parser.add_argument("--output_dir", type=str, default="analysis_output", help="Output directory for plots")
    parser.add_argument("--thres_col", type=float, default=3.0, help="DeepXTrace threshold for abnormal columns")
    parser.add_argument("--thres_row", type=float, default=3.0, help="DeepXTrace threshold for abnormal rows")
    parser.add_argument("--thres_point", type=float, default=5.0, help="DeepXTrace threshold for abnormal points")
    parser.add_argument("--last_n_steps", type=int, default=None, help="Only analyze the last N steps (default: all)")

    # New arguments for statistics mode
    parser.add_argument("--steps", type=str, default="",
                        help="Step range/list. Examples: '0-10,20,30-40' (default: all)")
    parser.add_argument("--layers", type=str, default="",
                        help="Layer range/list. Examples: '0-5,10' (default: all)")
    parser.add_argument("--stats_mode", action="store_true",
                        help="Enable statistics mode: compute mean/std across steps and layers")
    parser.add_argument("--aggregate_mode", type=str, default="per_layer",
                        choices=["per_layer", "global"],
                        help="'per_layer': aggregate across steps per layer; 'global': aggregate across steps+layers")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    t0_total = time.perf_counter()
    # Avoid loading all snapshots up-front: infer steps/layers/ranks from filenames first.
    all_steps, all_layers, num_ranks = scan_available_steps_and_layers(args.stats_dir)

    print(f"Available layers: {all_layers}")
    print(f"Steps range: {all_steps[0]} - {all_steps[-1]} ({len(all_steps)} total)")
    print(f"Num ranks: {num_ranks}")

    # Parse step/layer range arguments
    steps = parse_range_string(args.steps, all_steps) if args.steps else all_steps
    layers = parse_range_string(args.layers, all_layers) if args.layers else all_layers

    if not steps:
        print("Error: No valid steps found after filtering")
        return
    if not layers:
        print("Error: No valid layers found after filtering")
        return

    print(f"Selected steps: {len(steps)} steps ({steps[0]} - {steps[-1]})")
    print(f"Selected layers: {len(layers)} layers ({layers})")

    # ========== Statistics Mode ==========
    if args.stats_mode:
        print(f"\n{'='*60}")
        print(f"Running in statistics mode (aggregate_mode={args.aggregate_mode})")
        print(f"{'='*60}")

        # Need previous step for delta computation
        delta_steps = steps.copy()
        # Find the step before the first selected step if available
        if steps[0] > all_steps[0]:
            prev_step_idx = all_steps.index(steps[0]) - 1
            if prev_step_idx >= 0:
                delta_steps = [all_steps[prev_step_idx]] + delta_steps

        # Load data for multiple layers
        data = load_stats_multi_layers(args.stats_dir, layers, delta_steps)

        # Compute deltas for each layer
        print(f"\nComputing deltas for {len(layers)} layers...")
        t_deltas = time.perf_counter()
        all_deltas = {}  # (step, layer_id) -> delta dict

        for layer_id in layers:
            layer_deltas = compute_deltas(data, delta_steps, layer_id, num_ranks)
            # Remove the extra step used only for computing the first delta
            if len(delta_steps) > len(steps):
                layer_deltas.pop(delta_steps[0], None)
            for step, delta in layer_deltas.items():
                all_deltas[(step, layer_id)] = delta

        print(f"[{_ts()}] Delta computation finished in {time.perf_counter() - t_deltas:.2f}s")

        if not all_deltas:
            print("No deltas computed. Need at least 2 steps.")
            return

        # Compute statistics
        stats = compute_statistics(all_deltas, steps, layers, num_ranks, args.aggregate_mode)

        if not stats:
            print("No statistics computed.")
            return

        # Save statistics to CSV
        save_statistics_csv(stats, steps, layers, num_ranks, args.output_dir, args.aggregate_mode)

        # Generate heatmaps
        if args.aggregate_mode == "per_layer":
            for layer_id, layer_stats in stats.items():
                layer_dir = os.path.join(args.output_dir, f"layer{layer_id}")
                plot_statistics_heatmaps(layer_stats, layer_dir, prefix="")
        else:
            plot_statistics_heatmaps(stats.get("global", {}), args.output_dir, prefix="global_")

        # Generate summary plots
        plot_statistics_summary(stats, steps, layers, args.output_dir, args.aggregate_mode)

        print(f"\n{'='*60}")
        print(f"Statistics mode complete. Output saved to: {args.output_dir}")
        print(f"{'='*60}")
        print(f"[{_ts()}] main done in {time.perf_counter() - t0_total:.2f}s")
        return

    # ========== Standard Analysis Mode (single layer) ==========
    # Apply last_n_steps filter if specified
    if args.last_n_steps is not None and args.last_n_steps > 0:
        steps = steps[-args.last_n_steps:]
        print(f"Filtering to last {args.last_n_steps} steps: {steps[0]} - {steps[-1]}")

    # For single-layer mode, use first selected layer (or layer_id if specified)
    layer_id = args.layer_id if args.layer_id is not None else layers[0]
    if layer_id not in layers:
        print(f"Layer {layer_id} not found. Available: {layers}")
        return

    # Need previous step for delta computation
    delta_steps = steps.copy()
    if steps[0] > all_steps[0]:
        prev_step_idx = all_steps.index(steps[0]) - 1
        if prev_step_idx >= 0:
            delta_steps = [all_steps[prev_step_idx]] + delta_steps

    # Now only load snapshots needed for delta computation + analysis.
    data = load_stats(args.stats_dir, layer_id=layer_id, steps=delta_steps)
    print(f"\nAnalyzing layer {layer_id}...")
    t_deltas = time.perf_counter()
    deltas = compute_deltas(data, delta_steps, layer_id, num_ranks)
    if len(delta_steps) > len(steps):
        deltas.pop(delta_steps[0], None)
    print(f"Computed deltas for {len(deltas)} steps")
    print(f"[{_ts()}] main: compute_deltas finished in {time.perf_counter() - t_deltas:.2f}s")

    if not deltas:
        print("No deltas computed. Need at least 2 steps.")
        return

    t_anom = time.perf_counter()
    anomaly_results = analyze_anomalies(
        deltas,
        thres_col=args.thres_col,
        thres_row=args.thres_row,
        thres_point=args.thres_point,
    )
    print(f"[{_ts()}] main: analyze_anomalies finished in {time.perf_counter() - t_anom:.2f}s")

    t_diag_dump = time.perf_counter()
    save_diagnose_matrices_and_heatmaps(deltas, layer_id, args.output_dir)
    print(f"[{_ts()}] main: save_diagnose_matrices finished in {time.perf_counter() - t_diag_dump:.2f}s")

    print_summary(anomaly_results, deltas, layer_id)

    t_csv = time.perf_counter()
    save_summary_csv(deltas, anomaly_results, layer_id, num_ranks, args.output_dir)
    print(f"[{_ts()}] main: save_summary_csv finished in {time.perf_counter() - t_csv:.2f}s")

    steps_arr, imb, disp_w, comb_w = compute_correlation_data(deltas)
    plot_correlation(steps_arr, imb, disp_w, comb_w, layer_id, args.output_dir)

    if args.heatmap_steps:
        target_steps = [int(s.strip()) for s in args.heatmap_steps.split(",")]
        plot_heatmaps(deltas, target_steps, layer_id, args.output_dir)

    print(f"[{_ts()}] main done in {time.perf_counter() - t0_total:.2f}s")

if __name__ == "__main__":
    main()
