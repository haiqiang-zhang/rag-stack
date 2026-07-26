"""Backfill hv_diff / igd in snapshot CSVs against a frozen reference Pareto.

When the reference Pareto changes (e.g. after pooling fresh optimizer evals
into the cache), the snapshot CSVs in the same per-benchmark subfolder
still hold the *old* numbers. Re-running every benchmark to refresh them
takes hours; this script re-derives each checkpoint row from its companion
``__evals.csv`` in seconds.

Operates on **one benchmark subfolder per call** (= one
``synthetic_functions[].name``):

    <output_dir>/<name>/
        synthetic_function.yaml
        _pareto_cache.npy        ← new reference, loaded here
        <opt>__<name>.csv         ← rewritten in place
        <opt>__<name>__evals.csv  ← read-only source of truth

Algorithm per snapshot row ``(seed, cumulative_gt_calls=B)``:
    1. From the matching evals.csv, take all rows with same seed and
       ``cumulative_gt_calls <= B``.
    2. Reduce to the non-dominated set (= predicted Pareto at that budget).
    3. Compute hv_diff / igd against the cached reference.
    4. Overwrite the row's ``hv_diff`` and ``igd`` columns.

Approximation: this uses *all* recorded GT-evaluated configs in the
eval-history at-or-before the checkpoint, instead of calling
``optimizer.get_pareto()`` (which we can't replay without re-running).
For optimizers whose ``get_pareto()`` returns the non-dominated subset of
their evaluated points, the result is identical. Optimizers that return
surrogate-predicted Paretos may show small numeric differences — typically
biased slightly *lower* (smaller hv_diff) since this approximation takes
the non-dominated set across the full eval history rather than only the
optimizer's chosen subset.

Usage:
    python -m rag_stack.optimizer.benchmark.recompute_metrics_from_evals \\
        --name synthetic_rag_default

    # dry-run first to see expected changes:
    python -m rag_stack.optimizer.benchmark.recompute_metrics_from_evals \\
        --name synthetic_rag_default --dry-run

    # skip backups (overwrites snapshot CSVs in place — make sure git is clean):
    python -m rag_stack.optimizer.benchmark.recompute_metrics_from_evals \\
        --name synthetic_rag_default --no-backup
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .build_pareto_cache import (
    DEFAULT_OUTPUT_DIR,
    benchmark_subfolder,
    cache_path_for,
    load_cached_pareto,
    load_snapshot,
)
from .metrics import hv_difference, inverted_generational_distance
from .synthetic_functions import SYNTHETIC_FUNCTION_REGISTRY
from .synthetic_functions.base import non_dominated_max

logger = logging.getLogger(__name__)

BACKUP_SUFFIX = ".bak_pre_recompute"


def backfill(
    name: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    backup: bool = True,
    dry_run: bool = False,
) -> None:
    """Recompute hv_diff / igd for every snapshot CSV in
    ``<output_dir>/<name>/`` against the frozen reference Pareto saved
    in the same subfolder.
    """
    sub = benchmark_subfolder(output_dir, name)
    snap = load_snapshot(sub)

    type_ = snap["type"]
    cls = SYNTHETIC_FUNCTION_REGISTRY[type_]
    fn = cls(**(snap.get("params") or {}))
    obj_names = [obj["objective_name"] for obj in fn.objectives]
    ref_point = np.array([obj["ref_point"] for obj in fn.objectives], dtype=float)

    ref_pareto = load_cached_pareto(sub, len(obj_names))
    if ref_pareto is None:
        raise SystemExit(
            f"No frozen Pareto cache at {cache_path_for(sub)}. "
            f"Run `python -m rag_stack.optimizer.benchmark.build_pareto_cache "
            f"--name {name}` first."
        )
    logger.info(
        f"Reference Pareto: {len(ref_pareto)} points "
        f"(column order: {obj_names})"
    )

    # Snapshot CSVs sit alongside the cache in the subfolder. Filter out
    # backups + the per-eval companion files.
    snap_files = sorted(
        f
        for f in glob.glob(str(sub / "*.csv"))
        if not f.endswith("__evals.csv") and not f.endswith(BACKUP_SUFFIX)
    )
    if not snap_files:
        raise SystemExit(f"No snapshot CSVs found in {sub}/")

    logger.info(f"Found {len(snap_files)} snapshot CSV(s)")

    total_neg_before = 0
    total_neg_after = 0
    total_rows = 0

    for snap_path in snap_files:
        # Each snapshot CSV has a companion <name>__evals.csv. Just swap
        # the suffix to find it.
        evals_path = snap_path.replace(".csv", "__evals.csv")
        if not os.path.isfile(evals_path):
            logger.warning(
                f"  No matching evals.csv for {os.path.basename(snap_path)} — skipping"
            )
            continue

        snap_df = pd.read_csv(snap_path)
        # Per spec: NSGA-II builds the noise-free true Pareto (already done
        # by compute_true_pareto). For HV(observed) we use the eval CSV's
        # noisy quality/speed obs directly — accepting noise as intrinsic to
        # the regime. With config_json, we additionally re-evaluate each
        # explored config noise-free to make HV(observed) unbiased; without
        # config_json, we use the noisy obj columns as-is (same noise applied
        # to every optimizer → relative ranking preserved).
        evals_header = pd.read_csv(evals_path, nrows=0).columns.tolist()
        has_config_json = "config_json" in evals_header
        has_phase = "phase" in evals_header
        eval_cols = ["seed", "cumulative_gt_calls"] + obj_names
        if has_config_json:
            eval_cols.append("config_json")
        if has_phase:
            eval_cols.append("phase")
        evals_df = pd.read_csv(evals_path, usecols=eval_cols)

        # Phase A "cheap-only" rows (``phase == "main_cheap"``) are part of
        # the cheap-objective flood used to warm-start the speed GP — they
        # consume zero GT calls. The runner still records the full
        # composite_values (synthetic_rag's evaluator returns all objs
        # regardless of which the optimizer requested), so without this
        # filter the noise-free recompute would inflate the candidate
        # Pareto pool with hundreds of free Sobol points and bias hv_diff
        # downward for any optimizer with a Phase A.
        if has_phase:
            evals_df = evals_df[evals_df["phase"] == "main_both"].reset_index(drop=True)

        # Re-evaluate each unique config noise-free, replacing the noisy
        # obj columns with clean Y. Memoize per config_json string — many
        # rows can share the same config (e.g. cheap-only repeat phases).
        if has_config_json:
            clean_cache: dict[str, list[float]] = {}
            n_reeval = 0
            n_no_cfg = 0
            for cj in evals_df["config_json"].astype(str):
                if not cj or cj == "nan" or cj in clean_cache:
                    continue
                try:
                    cfg = json.loads(cj)
                    vals = fn.evaluate_noise_free(cfg)
                except Exception as exc:
                    logger.debug(
                        f"  noise-free re-eval failed for {os.path.basename(evals_path)}: {exc}"
                    )
                    continue
                row = []
                ok = True
                for n in obj_names:
                    v = vals.get(n)
                    if isinstance(v, dict):
                        v = v.get("combined")
                    if v is None or not np.isfinite(float(v)):
                        ok = False
                        break
                    row.append(float(v))
                if ok:
                    clean_cache[cj] = row
                    n_reeval += 1
            for ci, n in enumerate(obj_names):
                evals_df[n] = [
                    clean_cache.get(cj, [float("nan")] * len(obj_names))[ci]
                    if (cj := str(c)) and cj != "nan" else float("nan")
                    for c in evals_df["config_json"]
                ]
            n_no_cfg = int(
                (evals_df["config_json"].isna() | (evals_df["config_json"].astype(str) == ""))
                .sum()
            )
            logger.info(
                f"  {os.path.basename(evals_path)}: re-evaluated {n_reeval} unique "
                f"configs noise-free ({n_no_cfg} rows w/o config_json fall back to noisy obs)"
            )

        # Index evals by seed for fast filter inside the per-row loop.
        ev_by_seed = {seed: g for seed, g in evals_df.groupby("seed")}

        new_hv = np.empty(len(snap_df))
        new_igd = np.empty(len(snap_df))

        for ri, row in enumerate(snap_df.itertuples(index=False)):
            seed = int(row.seed)
            budget = int(row.cumulative_gt_calls)
            ev = ev_by_seed.get(seed)
            if ev is None:
                # No matching seed in evals.csv — keep old values.
                new_hv[ri] = row.hv_diff
                new_igd[ri] = row.igd
                continue

            ev_at_budget = ev[ev["cumulative_gt_calls"] <= budget]
            if len(ev_at_budget) == 0:
                # Empty Pareto at this budget — hv_diff = HV(ref) − 0 = HV(ref).
                new_hv[ri] = hv_difference(
                    np.empty((0, len(obj_names))), ref_pareto, ref_point
                )
                new_igd[ri] = float("inf")
                continue

            pred_arr = ev_at_budget[obj_names].to_numpy(dtype=float)
            # Drop any non-finite rows defensively.
            pred_arr = pred_arr[np.isfinite(pred_arr).all(axis=1)]
            if len(pred_arr) == 0:
                new_hv[ri] = hv_difference(
                    np.empty((0, len(obj_names))), ref_pareto, ref_point
                )
                new_igd[ri] = float("inf")
                continue

            pred_pareto = non_dominated_max(pred_arr)
            new_hv[ri] = hv_difference(pred_pareto, ref_pareto, ref_point)
            new_igd[ri] = inverted_generational_distance(pred_pareto, ref_pareto)

        n_neg_before = int((snap_df["hv_diff"] < 0).sum())
        n_neg_after = int((new_hv < 0).sum())
        total_neg_before += n_neg_before
        total_neg_after += n_neg_after
        total_rows += len(snap_df)

        action = "would write" if dry_run else "writing"
        logger.info(
            f"  {os.path.basename(snap_path)}: rows={len(snap_df)} "
            f"neg hv_diff: {n_neg_before} → {n_neg_after} ({action})"
        )

        if dry_run:
            continue

        snap_df["hv_diff"] = np.round(new_hv, 6)
        snap_df["igd"] = np.round(new_igd, 6)

        if backup:
            shutil.copy(snap_path, snap_path + BACKUP_SUFFIX)
        snap_df.to_csv(snap_path, index=False)

    logger.info(
        f"Total: {total_rows} rows; negative hv_diff "
        f"{total_neg_before} → {total_neg_after}"
    )


def add_args(parser: argparse.ArgumentParser) -> None:
    """Attach recompute-metrics args to a parser. Used by this module's
    standalone ``main()`` and by ``rag_stack/cli.py``'s ``recompute-metrics``
    subcommand registration.
    """
    parser.add_argument(
        "--name", required=True, action="append",
        help="Benchmark name (= subfolder under --output-dir). Repeatable.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--no-backup", action="store_true",
        help=f"Skip writing {BACKUP_SUFFIX} files. Default: keep backups.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute new values but don't write anything. Reports counts only.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")


def cmd_recompute_metrics(args) -> None:
    """Dispatcher: backfill snapshot CSVs for each --name."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    for name in args.name:
        backfill(
            name,
            output_dir=args.output_dir,
            backup=not args.no_backup,
            dry_run=args.dry_run,
        )


def main() -> None:
    """Standalone entry point — used by ``python -m rag_stack.optimizer.benchmark.recompute_metrics_from_evals``."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_args(parser)
    cmd_recompute_metrics(parser.parse_args())


if __name__ == "__main__":
    main()
