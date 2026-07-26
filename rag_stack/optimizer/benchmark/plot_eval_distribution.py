"""Plot per-eval (q, s) distribution + true Pareto for each (optimizer, seed).

Reads `<name>__<func>__evals.csv` files written by `BenchmarkRunner` and
renders one figure per seed: side-by-side panels for each optimizer, all on
shared axes, with:

  * scatter of every evaluated config colored by iteration order
  * true Pareto front overlaid (red curve)
  * found Pareto front overlaid (black markers, larger)

Usage:
    python -m rag_stack.optimizer.benchmark.plot_eval_distribution \\
        --data-dir tmp_outputs/optimizer_benchmark_results \\
        --function synthetic_rag \\
        --output-dir tmp_outputs/optimizer_benchmark_results/eval_distribution \\
        [--optimizers compositional_bi1_gt100 parego_metrics_bi1_gt100 ax_qnehvi]

If ``--optimizers`` is omitted, all matching ``*__<func>__evals.csv`` files in
``--data-dir`` are used.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rag_stack.optimizer.benchmark.synthetic_functions import (
    SYNTHETIC_FUNCTION_REGISTRY,
)


# Match `<optimizer_name>__<function_name>__evals.csv`
_EVAL_FILENAME_RE = re.compile(r"^(?P<opt>.+)__(?P<func>[^_]+(?:_[^_]+)*)__evals\.csv$")


def _find_eval_csvs(
    data_dir: Path, function_name: str, optimizers: Optional[List[str]] = None
) -> List[Tuple[str, Path]]:
    """Return [(optimizer_name, csv_path), ...] for the given function."""
    out: List[Tuple[str, Path]] = []
    for path in sorted(data_dir.glob(f"*__{function_name}__evals.csv")):
        m = _EVAL_FILENAME_RE.match(path.name)
        if not m:
            continue
        opt_name = m.group("opt")
        if optimizers is not None and opt_name not in optimizers:
            continue
        out.append((opt_name, path))
    return out


def _pareto_front_2d(points: np.ndarray) -> np.ndarray:
    """Return non-dominated rows (maximization) in 2D via O(n log n) sweep."""
    if len(points) == 0:
        return points
    order = np.argsort(-points[:, 0], kind="stable")
    sorted_pts = points[order]
    mask = np.zeros(len(points), dtype=bool)
    best = -np.inf
    for i, idx in enumerate(order):
        if sorted_pts[i, 1] > best:
            mask[idx] = True
            best = sorted_pts[i, 1]
    return points[mask]


def plot_one_seed(
    function_name: str,
    seed: int,
    optimizer_evals: Dict[str, pd.DataFrame],
    true_pareto: np.ndarray,
    output_path: Path,
) -> None:
    """Render one figure for one seed, panels per optimizer."""
    n_opts = len(optimizer_evals)
    if n_opts == 0:
        return
    fig, axes = plt.subplots(
        1, n_opts, figsize=(5.0 * n_opts, 4.5), sharex=True, sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    # Shared axis limits, padded a bit, derived from union of evals + true Pareto.
    all_q = np.concatenate(
        [df["quality"].to_numpy() for df in optimizer_evals.values()]
        + [true_pareto[:, 0]]
    )
    all_s = np.concatenate(
        [df["speed"].to_numpy() for df in optimizer_evals.values()]
        + [true_pareto[:, 1]]
    )
    finite_mask_q = np.isfinite(all_q)
    finite_mask_s = np.isfinite(all_s)
    q_lo, q_hi = all_q[finite_mask_q].min(), all_q[finite_mask_q].max()
    s_lo, s_hi = all_s[finite_mask_s].min(), all_s[finite_mask_s].max()
    q_pad = (q_hi - q_lo) * 0.05
    s_pad = (s_hi - s_lo) * 0.05

    # Sort true Pareto by speed for line plotting.
    tp_order = np.argsort(true_pareto[:, 1])
    tp_sorted = true_pareto[tp_order]

    for ax, (opt_name, df) in zip(axes, optimizer_evals.items()):
        df_seed = df[df["seed"] == seed]
        if df_seed.empty:
            ax.set_title(f"{opt_name}\n(no data for seed {seed})")
            continue

        q = df_seed["quality"].to_numpy()
        s = df_seed["speed"].to_numpy()
        finite = np.isfinite(q) & np.isfinite(s)
        q, s = q[finite], s[finite]
        order = np.arange(len(q))

        # Scatter all evals, color = order index (later = brighter).
        sc = ax.scatter(
            q, s, c=order, cmap="viridis", s=12, alpha=0.7,
            edgecolor="none", label=f"evals (n={len(q)})",
        )

        # Found Pareto on this seed.
        if len(q) > 0:
            found = _pareto_front_2d(np.column_stack([q, s]))
            f_order = np.argsort(found[:, 1])
            found = found[f_order]
            ax.plot(
                found[:, 0], found[:, 1],
                "o-", color="black", markersize=6, linewidth=1.0,
                label=f"found Pareto (n={len(found)})",
            )

        # True Pareto.
        ax.plot(
            tp_sorted[:, 0], tp_sorted[:, 1],
            "-", color="crimson", linewidth=1.5, alpha=0.9,
            label=f"true Pareto (n={len(tp_sorted)})",
        )

        ax.set_title(opt_name, fontsize=11)
        ax.set_xlabel("quality")
        ax.set_xlim(q_lo - q_pad, q_hi + q_pad)
        ax.set_ylim(s_lo - s_pad, s_hi + s_pad)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8, framealpha=0.85)

    axes[0].set_ylabel("speed")

    # Color bar for iteration ordering.
    cbar = fig.colorbar(sc, ax=list(axes), shrink=0.7, location="right",
                        label="eval order (early → late)")

    fig.suptitle(f"{function_name} | seed={seed}", fontsize=13)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing __evals.csv files")
    parser.add_argument("--function", required=True,
                        help="Synthetic function name (e.g. synthetic_rag)")
    parser.add_argument("--output-dir", default=None,
                        help="Where to save plots (default: <data-dir>/eval_distribution)")
    parser.add_argument("--optimizers", nargs="*", default=None,
                        help="Optimizer name filter (defaults: all matching CSVs)")
    parser.add_argument("--function-params", nargs="*", default=[],
                        help="key=value pairs forwarded to the function constructor")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir or data_dir / "eval_distribution").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_csvs = _find_eval_csvs(data_dir, args.function, args.optimizers)
    if not eval_csvs:
        raise SystemExit(
            f"No __evals.csv found for function {args.function!r} in {data_dir}"
        )

    func_params = {}
    for kv in args.function_params:
        k, _, v = kv.partition("=")
        try:
            func_params[k] = int(v)
        except ValueError:
            try:
                func_params[k] = float(v)
            except ValueError:
                func_params[k] = v
    function = SYNTHETIC_FUNCTION_REGISTRY[args.function](**func_params)
    true_pareto = function.compute_true_pareto()

    # Load all evals into memory (small — gt=100 × few seeds × few optimizers).
    optimizer_evals: Dict[str, pd.DataFrame] = {}
    seeds: set = set()
    for opt_name, csv_path in eval_csvs:
        df = pd.read_csv(csv_path)
        optimizer_evals[opt_name] = df
        seeds.update(int(s) for s in df["seed"].unique())

    print(f"Plotting {len(optimizer_evals)} optimizers × {len(seeds)} seeds...")
    for seed in sorted(seeds):
        out_path = output_dir / f"{args.function}__seed{seed}__eval_distribution.png"
        plot_one_seed(args.function, seed, optimizer_evals, true_pareto, out_path)


if __name__ == "__main__":
    main()
