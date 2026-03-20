# DeepEP Stats Analysis Tool

Offline analysis tool for DeepEP dispatch/combine stats collected by SGLang.

## Overview

This tool loads `.pt` snapshot files saved by `_DeepEPDispatcherImplLowLatency`, reconstructs N×N wait-time matrices (same format as DeepXTrace), correlates expert distribution with dispatch/combine latency, and generates visualizations.

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy torch matplotlib seaborn
```

## Usage

### Basic Analysis (Single Layer)

```bash
python analyze_deepep_stats.py /path/to/deepep_stats --layer_id 0
```

### With Custom Options

```bash
python analyze_deepep_stats.py /path/to/deepep_stats \
    --layer_id 0 \
    --heatmap_steps 0,10,50,100 \
    --last_n_steps 100 \
    --output_dir analysis_output
```

### Statistics Mode (Multiple Layers)

#### Per-Layer Aggregation
Compute mean/std statistics for each layer independently:

```bash
python analyze_deepep_stats.py /path/to/deepep_stats \
    --steps "0-100" \
    --layers "0-5" \
    --stats_mode \
    --aggregate_mode per_layer
```

#### Global Aggregation
Aggregate statistics across all specified layers:

```bash
python analyze_deepep_stats.py /path/to/deepep_stats \
    --steps "0-100" \
    --layers "0-5" \
    --stats_mode \
    --aggregate_mode global
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `stats_dir` | str | (required) | Path to stats directory containing `rank*/` subdirs |
| `--layer_id` | int | first available | Layer ID to analyze (single-layer mode) |
| `--steps` | str | all | Step range/list, e.g., `'0-10,20,30-40'` |
| `--layers` | str | all | Layer range/list, e.g., `'0-5,10'` |
| `--stats_mode` | flag | False | Enable statistics mode for multi-layer analysis |
| `--aggregate_mode` | str | per_layer | `per_layer` or `global` aggregation mode |
| `--heatmap_steps` | str | "" | Comma-separated steps for heatmap generation |
| `--last_n_steps` | int | all | Only analyze the last N steps |
| `--output_dir` | str | analysis_output | Output directory for plots and CSVs |
| `--thres_col` | float | 3.0 | DeepXTrace threshold for abnormal columns |
| `--thres_row` | float | 3.0 | DeepXTrace threshold for abnormal rows |
| `--thres_point` | float | 5.0 | DeepXTrace threshold for abnormal points |

## Input Data Format

Expected directory structure:

```
stats_dir/
├── rank0/
│   ├── step0_layer0.pt
│   ├── step1_layer0.pt
│   ├── step0_layer1.pt
│   └── ...
├── rank1/
│   ├── step0_layer0.pt
│   └── ...
└── ...
```

Each `.pt` file contains a snapshot with:
- `step`: Step number
- `layer_id`: Layer ID
- `rank`: Rank ID
- `dispatch_wait_recv_cost`: 1D array of dispatch wait times
- `combine_wait_recv_cost`: 1D array of combine wait times
- `cumulative_expert_recv`: Expert receive counts
- `num_tokens`: Number of tokens processed

## Output Files

### Standard Mode (Single Layer)

```
analysis_output/
├── diagnose_matrices_layer{N}/
│   ├── diagnose_matrix_layer{N}_step{S}_dispatch.csv
│   ├── diagnose_matrix_layer{N}_step{S}_dispatch.png
│   └── ...
├── step_summary_layer{N}.csv
├── per_rank_wait_layer{N}.csv
├── expert_recv_layer{N}.csv
└── correlation_layer{N}.png
```

### Statistics Mode (Per-Layer)

```
analysis_output/
├── layer{N}/
│   ├── statistics_summary.csv
│   ├── dispatch_mean.csv
│   ├── dispatch_std.csv
│   ├── combine_mean.csv
│   ├── combine_std.csv
│   ├── expert_recv_mean.csv
│   ├── expert_recv_std.csv
│   ├── dispatch_mean.png
│   ├── dispatch_std.png
│   ├── combine_mean.png
│   └── combine_std.png
├── statistics_summary_per_layer.png
└── ...
```

### Statistics Mode (Global)

```
analysis_output/
├── global_statistics_summary.csv
├── global_dispatch_mean.csv
├── global_dispatch_std.csv
├── global_combine_mean.csv
├── global_combine_std.csv
├── global_expert_recv_mean.csv
├── global_expert_recv_std.csv
├── global_dispatch_mean.png
├── global_dispatch_std.png
├── global_combine_mean.png
├── global_combine_std.png
└── global_statistics_std_distribution.png
```

## Output CSV Format

### step_summary_layer{N}.csv

| Column | Description |
|--------|-------------|
| step | Step number |
| layer_id | Layer ID |
| num_tokens | Number of tokens |
| dispatch_wait_total | Total dispatch wait time |
| combine_wait_total | Total combine wait time |
| dispatch_wait_mean | Mean dispatch wait time |
| combine_wait_mean | Mean combine wait time |
| dispatch_wait_max | Max dispatch wait time |
| combine_wait_max | Max combine wait time |
| expert_imbalance_cov | Coefficient of variation for expert load |
| dispatch_anomaly_cols | Count of abnormal columns (dispatch) |
| dispatch_anomaly_rows | Count of abnormal rows (dispatch) |
| dispatch_anomaly_points | Count of abnormal points (dispatch) |
| combine_anomaly_* | Same for combine phase |

### statistics_summary.csv

| Column | Description |
|--------|-------------|
| metric | Matrix name (dispatch_mean, dispatch_std, etc.) |
| mean_value | Overall mean of the matrix |
| std_value | Overall std of the matrix |
| min_value | Minimum value in the matrix |
| max_value | Maximum value in the matrix |
| num_samples | Number of (step, layer) pairs aggregated |
| aggregation_unit | Layer ID or "global" |

## Example Workflows

### 1. Quick Analysis of a Single Layer

```bash
# Analyze layer 0 with last 50 steps
python analyze_deepep_stats.py /data/deepep_stats \
    --layer_id 0 \
    --last_n_steps 50
```

### 2. Compare Statistics Across Layers

```bash
# Generate per-layer statistics for layers 0-5
python analyze_deepep_stats.py /data/deepep_stats \
    --steps "0-200" \
    --layers "0-5" \
    --stats_mode \
    --aggregate_mode per_layer
```

### 3. Global Performance Overview

```bash
# Aggregate all MoE layers for overall performance metrics
python analyze_deepep_stats.py /data/deepep_stats \
    --steps "0-500" \
    --layers "0-31" \
    --stats_mode \
    --aggregate_mode global
```

### 4. Debug Specific Steps

```bash
# Focus on specific steps that showed anomalies
python analyze_deepep_stats.py /data/deepep_stats \
    --layer_id 5 \
    --steps "100-110" \
    --heatmap_steps 100,105,110
```

## Diagnose Module

The tool uses DeepXTrace's `Diagnose.diagnose_matrix()` to detect anomalies:

- **Abnormal columns**: Destination ranks that are slow receivers
- **Abnormal rows**: Source ranks that are slow senders
- **Abnormal points**: Individual outlier cells in the wait-time matrix

Thresholds can be adjusted via `--thres_col`, `--thres_row`, and `--thres_point`.