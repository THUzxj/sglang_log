"""
Offline analysis of DeepEP dispatch/combine stats collected by sglang.

Loads .pt snapshot files saved by _DeepEPDispatcherImplLowLatency, reconstructs
N x N wait-time matrices (same format as DeepXTrace), correlates expert distribution
with dispatch/combine latency, and generates visualizations.

Usage:
    python analyze_deepep_stats.py /path/to/deepep_stats [--layer_id 0] [--heatmap_steps 0,10,50]
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from deepxtrace_analysis.diagnose import Diagnose


def load_stats(stats_dir: str) -> Dict[Tuple[int, int, int], dict]:
    """
    Load all .pt files from stats_dir/rank*/step*_layer*.pt.

    Returns dict keyed by (step, layer_id, rank) -> snapshot dict.
    """
    pattern = os.path.join(stats_dir, "rank*", "step*_layer*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No .pt files found matching {pattern}")

    data = {}
    for f in files:
        snapshot = torch.load(f, map_location="cpu", weights_only=True)
        key = (snapshot["step"], snapshot["layer_id"], snapshot["rank"])
        data[key] = snapshot

    print(f"Loaded {len(files)} snapshot files from {stats_dir}")
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

    for step in steps:
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

    return deltas


def analyze_anomalies(
    deltas: Dict[int, dict],
    thres_col: float = 3.0,
    thres_row: float = 3.0,
    thres_point: float = 5.0,
) -> Dict[int, dict]:
    """Run DeepXTrace diagnose_matrix on each step's matrices."""
    results = {}
    for step, d in deltas.items():
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
    return results


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

    for step in sorted_steps:
        d = deltas[step]

        expert_per_rank = d["expert_recv"].sum(axis=1).astype(float)
        mean_val = expert_per_rank.mean()
        cov = expert_per_rank.std() / (mean_val + 1e-8)
        expert_imbalance.append(cov)

        dispatch_total.append(d["dispatch_matrix"].sum())
        combine_total.append(d["combine_matrix"].sum())

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
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from deepxtrace_heatmap import create_optimized_ryg_cmap
    except ImportError:
        print("matplotlib/seaborn not installed or deepxtrace_heatmap not found, skipping heatmaps")
        return

    os.makedirs(output_dir, exist_ok=True)

    for step in target_steps:
        if step not in deltas:
            print(f"Step {step} not found in deltas, skipping")
            continue

        d = deltas[step]

        for phase, matrix in [("dispatch", d["dispatch_matrix"]), ("combine", d["combine_matrix"])]:
            fig, ax = plt.subplots(figsize=(10, 8))
            mat_float = matrix.astype(float)
            log_mat = np.log1p(mat_float)

            cmap = create_optimized_ryg_cmap()
            sns.heatmap(
                log_mat,
                cmap=cmap,
                annot=True,
                fmt=".0f",
                linewidths=0.5,
                linecolor="white",
                ax=ax,
                annot_kws={"size": 8},
            )
            ax.set_title(f"Layer {layer_id} Step {step}: {phase.capitalize()} Wait Time")
            ax.set_xlabel("Destination Rank")
            ax.set_ylabel("Source Rank")

            path = os.path.join(output_dir, f"heatmap_layer{layer_id}_step{step}_{phase}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved heatmap: {path}")


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
      - expert_recv.csv:   per-step, per-rank expert recv counts
    """
    os.makedirs(output_dir, exist_ok=True)
    sorted_steps = sorted(deltas.keys())

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

        for step in sorted_steps:
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

        for step in sorted_steps:
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

    # ── 3. expert_recv.csv ───────────────────────────────────────────────
    path_expert = os.path.join(output_dir, f"expert_recv_layer{layer_id}.csv")
    with open(path_expert, "w", newline="") as f:
        w = csv.writer(f)
        sample_key = sorted_steps[0]
        num_local_experts = deltas[sample_key]["expert_recv"].shape[1]
        expert_cols = [f"expert_{e}" for e in range(num_local_experts)]
        header = (
            ["step", "layer_id", "rank"]
            + expert_cols
            + ["total_recv", "recv_cov"]
        )
        w.writerow(header)

        for step in sorted_steps:
            expert_mat = deltas[step]["expert_recv"]
            for rank in range(num_ranks):
                row = expert_mat[rank].astype(float)
                mean_val = row.mean()
                cov = row.std() / (mean_val + 1e-8) if mean_val > 0 else 0.0
                w.writerow(
                    [step, layer_id, rank]
                    + [int(v) for v in row]
                    + [int(row.sum()), f"{cov:.6f}"]
                )
    print(f"Saved: {path_expert}")


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
    args = parser.parse_args()

    data = load_stats(args.stats_dir)
    steps, layers, num_ranks = get_available_steps_and_layers(data)

    print(f"Available layers: {layers}")
    print(f"Steps range: {steps[0]} - {steps[-1]} ({len(steps)} total)")
    print(f"Num ranks: {num_ranks}")

    layer_id = args.layer_id if args.layer_id is not None else layers[0]
    if layer_id not in layers:
        print(f"Layer {layer_id} not found. Available: {layers}")
        return

    print(f"\nAnalyzing layer {layer_id}...")
    deltas = compute_deltas(data, steps, layer_id, num_ranks)
    print(f"Computed deltas for {len(deltas)} steps")

    if not deltas:
        print("No deltas computed. Need at least 2 steps.")
        return

    anomaly_results = analyze_anomalies(
        deltas,
        thres_col=args.thres_col,
        thres_row=args.thres_row,
        thres_point=args.thres_point,
    )

    print_summary(anomaly_results, deltas, layer_id)

    save_summary_csv(deltas, anomaly_results, layer_id, num_ranks, args.output_dir)

    steps_arr, imb, disp_w, comb_w = compute_correlation_data(deltas)
    plot_correlation(steps_arr, imb, disp_w, comb_w, layer_id, args.output_dir)

    if args.heatmap_steps:
        target_steps = [int(s.strip()) for s in args.heatmap_steps.split(",")]
        plot_heatmaps(deltas, target_steps, layer_id, args.output_dir)


if __name__ == "__main__":
    main()
