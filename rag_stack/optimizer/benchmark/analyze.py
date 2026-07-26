"""Analyze benchmark results and generate plots.

Usage:
    # Plot all CSVs in a directory:
    python -m rag_stack.optimizer.benchmark.analyze benchmark_results/

    # Plot specific files:
    python -m rag_stack.optimizer.benchmark.analyze \
        benchmark_results/smac_decoupled_branin_currin_seed42.csv \
        benchmark_results/topo_decoupled_branin_currin_seed42.csv

    # Custom output directory:
    python -m rag_stack.optimizer.benchmark.analyze benchmark_results/ --output-dir plots/
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class RunInfo:
    """Metadata parsed from a benchmark CSV filename."""
    optimizer: str
    synthetic_function: str
    path: str


def parse_filename(path: str) -> RunInfo:
    """Extract optimizer, synthetic_function from filename.

    Expected pattern: {optimizer}__{synthetic_function}.csv — each CSV may
    contain rows from multiple seeds (``seed`` is an intrinsic column).
    """
    stem = Path(path).stem
    match = re.match(r"^(.+)__(.+)$", stem)
    if not match:
        raise ValueError(f"Cannot parse '{stem}'. Expected format: {{optimizer}}__{{function}}")
    return RunInfo(
        optimizer=match.group(1),
        synthetic_function=match.group(2),
        path=path,
    )


def load_results(paths: List[str]) -> pd.DataFrame:
    """Load and concatenate benchmark CSVs, adding metadata columns.

    ``seed`` is read straight from the CSV (written per-row by
    ``BenchmarkRunner.save_csv``); filename only encodes optimizer + function.
    """
    frames = []
    for p in paths:
        info = parse_filename(p)
        df = pd.read_csv(p)
        df["optimizer"] = info.optimizer
        df["synthetic_function"] = info.synthetic_function
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def collect_csvs(path: str) -> List[str]:
    """Collect matching CSVs from a file or directory.

    Skips legacy ``{opt}__{fn}__seed{N}.csv`` files — the seed is now a column
    inside ``{opt}__{fn}.csv``, not part of the filename.
    """
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return sorted(
            str(f) for f in p.glob("*.csv")
            if re.match(r"^(.+)__(.+)$", f.stem)
            and not re.search(r"__seed\d+$", f.stem)
        )
    raise ValueError(f"Path does not exist: {path}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


def _get_color_map(labels: List[str]) -> Dict[str, str]:
    return {label: COLORS[i % len(COLORS)] for i, label in enumerate(labels)}


def _plot_metric(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str,
    color_map: Dict[str, str],
    ylabel: str,
    xlabel: str,
    log_y: bool = False,
):
    """Plot one metric with mean +/- std across seeds."""
    for label, group in df.groupby(group_col):
        agg = group.groupby(x_col)[y_col].agg(["mean", "std"]).reset_index()
        x = agg[x_col].values
        mean = agg["mean"].values
        std = agg["std"].fillna(0).values
        color = color_map[label]
        ax.plot(x, mean, marker="o", markersize=4, label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main plot functions
# ---------------------------------------------------------------------------

def plot_by_function(df: pd.DataFrame, output_dir: str):
    """For each synthetic function, plot all optimizers on the same axes.

    Generates a 2x2 figure per function:
        - HV Diff vs iteration
        - IGD vs iteration
        - HV Diff vs GT calls
        - Optimizer time vs iteration
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for func_name, func_df in df.groupby("synthetic_function"):
        optimizers = sorted(func_df["optimizer"].unique())
        color_map = _get_color_map(optimizers)

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(f"Benchmark: {func_name}", fontsize=14, fontweight="bold")

        # (0,0) HV Diff vs iteration
        _plot_metric(
            axes[0, 0], func_df,
            x_col="iteration", y_col="hv_diff", group_col="optimizer",
            color_map=color_map,
            ylabel="HV Difference (lower=better)", xlabel="Iteration",
            log_y=True,
        )
        axes[0, 0].set_title("Pareto Accuracy (HV Diff)")

        # (0,1) IGD vs iteration
        _plot_metric(
            axes[0, 1], func_df,
            x_col="iteration", y_col="igd", group_col="optimizer",
            color_map=color_map,
            ylabel="IGD (lower=better)", xlabel="Iteration",
            log_y=True,
        )
        axes[0, 1].set_title("Pareto Coverage (IGD)")

        # (1,0) HV Diff vs GT calls (sample efficiency)
        _plot_metric(
            axes[1, 0], func_df,
            x_col="cumulative_gt_calls", y_col="hv_diff", group_col="optimizer",
            color_map=color_map,
            ylabel="HV Difference (lower=better)", xlabel="Cumulative GT Calls",
            log_y=True,
        )
        axes[1, 0].set_title("Sample Efficiency")

        # (1,1) Time efficiency: optimizer time vs IGD
        _plot_metric(
            axes[1, 1], func_df,
            x_col="cumulative_optimizer_time_s", y_col="igd", group_col="optimizer",
            color_map=color_map,
            ylabel="IGD (lower=better)", xlabel="Cumulative Optimizer Time (s)",
            log_y=True,
        )
        axes[1, 1].set_title("Time Efficiency")

        fig.tight_layout()
        path = output / f"{func_name}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def plot_summary_table(df: pd.DataFrame, output_dir: str):
    """Print a summary table of final metrics per (optimizer, synthetic_function)."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Take last checkpoint per run
    idx = df.groupby(["optimizer", "synthetic_function", "seed"])["iteration"].idxmax()
    final = df.loc[idx]

    summary = (
        final.groupby(["optimizer", "synthetic_function"])
        .agg(
            hv_diff_mean=("hv_diff", "mean"),
            hv_diff_std=("hv_diff", "std"),
            igd_mean=("igd", "mean"),
            igd_std=("igd", "std"),
            gt_calls=("cumulative_gt_calls", "mean"),
            optimizer_time_s=("cumulative_optimizer_time_s", "mean"),
        )
        .round(4)
    )

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY (final checkpoint)")
    print("=" * 80)
    print(summary.to_string())
    print()

    summary.to_csv(output / "summary.csv")
    print(f"Saved {output / 'summary.csv'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def analyze_directory(data_dir: str, output_dir: str = None):
    """Analyze all CSVs in a directory. Called by run.py --plot or standalone."""
    if output_dir is None:
        output_dir = str(Path(data_dir) / "plots")

    all_csvs = collect_csvs(data_dir)
    if not all_csvs:
        print(f"No CSV files found in {data_dir}")
        return

    print(f"Loading {len(all_csvs)} CSV file(s)...")
    df = load_results(all_csvs)

    plot_by_function(df, output_dir)
    plot_summary_table(df, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument(
        "paths", nargs="+",
        help="CSV files or directories containing benchmark CSVs",
    )
    parser.add_argument("--output-dir", default="tmp_outputs/benchmark_results/plots")
    args = parser.parse_args()

    all_csvs = []
    for p in args.paths:
        all_csvs.extend(collect_csvs(p))

    if not all_csvs:
        print("No CSV files found.")
        sys.exit(1)

    print(f"Loading {len(all_csvs)} CSV file(s)...")
    df = load_results(all_csvs)

    plot_by_function(df, args.output_dir)
    plot_summary_table(df, args.output_dir)


if __name__ == "__main__":
    main()
