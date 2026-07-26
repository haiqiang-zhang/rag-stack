"""Build frozen reference Pareto for one named benchmark.

The benchmark identity is the ``synthetic_functions[].name`` field from
the run config — it doubles as a subfolder name under the shared output
directory:

    <output_dir>/<name>/
        synthetic_function.yaml   ← config snapshot (type + name + params)
        _pareto_cache.npy          ← frozen reference Pareto written here
        <optimizer>__<name>.csv    ← snapshot CSVs from benchmark runs
        <optimizer>__<name>__evals.csv
        ...

This CLI reads ``<output_dir>/<name>/synthetic_function.yaml`` to learn
which class + params to instantiate, runs that function's
``compute_true_pareto`` (NSGA-II + optional pool of the eval CSVs in the
same subfolder), and writes the result to ``_pareto_cache.npy`` in that
folder. ``BenchmarkRunner`` then loads it on subsequent benchmark runs;
the apps server's eval-distribution endpoint reads the same file too.

Modes (controlled by ``--no-pool``):

1. **NSGA-II only** (``--no-pool``): runs NSGA-II for the function's
   default ``n_samples`` evaluations on the surrogate, takes the final
   non-dominated front. Use this for the very first build, before any
   optimizer eval data exists in the subfolder.

2. **NSGA-II + pool** (default): NSGA-II baseline AND pool every
   ``*__<name>__evals.csv`` already in the subfolder. Also auto-triggers
   ``recompute_metrics_from_evals`` for that subfolder so historical chart
   numbers reflect the new reference. Use this after every fresh sweep.

Pure-pool (no NSGA-II) is intentionally not supported: pool-only depends
on having strong past optimizer runs and silently regresses when none do.
NSGA-II provides an unconditional strong baseline; pool augments it but
never replaces it.

Usage:
    # initial setup (no past data in subfolder yet):
    python -m rag_stack.optimizer.benchmark.build_pareto_cache \\
        --name synthetic_rag_default --no-pool

    # refresh after running new sweeps — adds the new evals into the
    # reference AND backfills hv_diff / igd in historical snapshot CSVs:
    python -m rag_stack.optimizer.benchmark.build_pareto_cache \\
        --name synthetic_rag_default --name yahpo_inst41138
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Default root for benchmark output. Each benchmark `name` becomes a
# subfolder underneath, holding that benchmark's CSVs + cache + config
# snapshot. Single source of truth for the path used by:
#   - this CLI (build_pareto_cache)
#   - recompute_metrics_from_evals (imports DEFAULT_OUTPUT_DIR from here)
#   - the sweep CLI (rag_stack.optimizer.benchmark.cli, also imports it)
# Resolved at module import time from `.rag_stack_settings.json`'s
# `optimizer_benchmark_dir` field; falls back to a hardcoded literal when
# the settings file is missing or that field is empty.
_FALLBACK_OUTPUT_DIR = Path("tmp_outputs/optimizer_benchmark_results")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_default_output_dir() -> Path:
    """Read `optimizer_benchmark_dir` from `.rag_stack_settings.json`.

    Lookup order:
      1. ``RAG_STACK_SETTINGS_PATH`` env var (overrides settings file location)
      2. ``<project_root>/.rag_stack_settings.json``
      3. Hardcoded fallback ``tmp_outputs/optimizer_benchmark_results``

    Read-only, best-effort: any I/O or JSON error → fallback. We do not
    import ``rag_stack.server.settings_store`` to avoid pulling pydantic
    + the whole server stack into a small CLI.
    """
    settings_path_str = os.environ.get("RAG_STACK_SETTINGS_PATH", "").strip()
    settings_path = Path(settings_path_str) if settings_path_str else _PROJECT_ROOT / ".rag_stack_settings.json"
    try:
        with open(settings_path) as f:
            data = json.load(f)
        configured = str(data.get("optimizer_benchmark_dir") or "").strip()
        if configured:
            return Path(configured)
    except (OSError, ValueError):
        pass
    return _FALLBACK_OUTPUT_DIR


DEFAULT_OUTPUT_DIR = _resolve_default_output_dir()

# Filenames inside each per-benchmark subfolder.
_CACHE_FILENAME = "_pareto_cache.npy"
# Single-file experiments manifest: holds `synthetic_function` (the
# benchmark config) and `optimizers` (every experiment that's run
# against this benchmark). Owned by the CLI's `ensure_experiments_yaml`;
# we just READ the benchmark block here to instantiate the function.
_EXPERIMENTS_FILENAME = "experiments.yaml"


# ---------------------------------------------------------------------------
# Path + I/O helpers — single source of truth for the subfolder layout.
# Imported by runner.py, recompute_metrics_from_evals.py, and the apps
# server router.
# ---------------------------------------------------------------------------

def benchmark_subfolder(output_dir: Path | str, name: str) -> Path:
    """Resolve the per-benchmark subfolder under ``output_dir``."""
    return Path(output_dir) / name


def cache_path_for(subfolder: Path | str) -> Path:
    """Frozen Pareto cache location inside a per-benchmark subfolder."""
    return Path(subfolder) / _CACHE_FILENAME


def experiments_path_for(subfolder: Path | str) -> Path:
    """Per-benchmark experiments manifest location."""
    return Path(subfolder) / _EXPERIMENTS_FILENAME


def load_cached_pareto(
    subfolder: Path | str,
    n_obj: int,
) -> Optional[np.ndarray]:
    """Load the frozen reference Pareto saved inside ``subfolder``.

    Returns ``(n_pareto, n_obj)`` ndarray (column order = function's
    ``objectives``) on success, or ``None`` if:
      - the cache file is missing,
      - it can't be read (permissions / corrupt),
      - its column count doesn't match ``n_obj`` (stale after objectives
        changed — caller should fall through to recompute).
    """
    path = cache_path_for(subfolder)
    if not path.is_file():
        return None
    try:
        arr = np.load(path)
    except OSError:
        return None
    if arr.ndim != 2 or arr.shape[1] != n_obj:
        return None
    return arr


def save_cached_pareto(subfolder: Path | str, arr: np.ndarray) -> None:
    """Persist a freshly-computed reference Pareto into ``subfolder``.
    Best-effort — silently no-ops on OSError so a non-writable cache dir
    can't crash the benchmark itself.
    """
    path = cache_path_for(subfolder)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
    except OSError:
        pass


def load_snapshot(subfolder: Path | str) -> Dict[str, Any]:
    """Load the ``synthetic_function`` block from ``experiments.yaml``.

    The manifest is a combined file: ``synthetic_function`` (benchmark
    config) + ``optimizers`` (list of experiment configs). This helper
    returns just the benchmark block — what build_pareto_cache,
    recompute, and apps router need to instantiate the function.

    Raises ``FileNotFoundError`` if the subfolder hasn't been
    initialized yet (no benchmark has run against this name).
    """
    path = experiments_path_for(subfolder)
    if not path.is_file():
        raise FileNotFoundError(
            f"No experiments manifest at {path}. Run the benchmark CLI at "
            f"least once with this `name` to populate the subfolder, or "
            f"write a {_EXPERIMENTS_FILENAME} by hand."
        )
    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    if not isinstance(doc, dict):
        raise ValueError(
            f"{path} is malformed: top-level must be a mapping."
        )
    fn = doc.get("synthetic_function")
    if not isinstance(fn, dict):
        raise ValueError(
            f"{path} is missing a `synthetic_function` mapping."
        )
    return fn


# ---------------------------------------------------------------------------
# Function instantiation — keyed by `type` from the snapshot, NOT by the
# benchmark `name`.
# ---------------------------------------------------------------------------

def _instantiate_function_from_snapshot(snap: Dict[str, Any]):
    """Construct a SyntheticFunction instance from a parsed
    ``synthetic_function.yaml`` snapshot. The snapshot's ``type`` field
    selects the class; ``params`` are kwargs.
    """
    from .synthetic_functions import SYNTHETIC_FUNCTION_REGISTRY

    type_ = snap.get("type")
    if not type_ or type_ not in SYNTHETIC_FUNCTION_REGISTRY:
        raise ValueError(
            f"Snapshot has unknown type {type_!r}. "
            f"Available: {list(SYNTHETIC_FUNCTION_REGISTRY.keys())}"
        )
    params = dict(snap.get("params") or {})
    return SYNTHETIC_FUNCTION_REGISTRY[type_](**params)


# ---------------------------------------------------------------------------
# Build entry point.
# ---------------------------------------------------------------------------

def build_pareto(
    name: str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    pool_evals: bool = True,
) -> Path:
    """Build the frozen reference Pareto for the benchmark named ``name``.

    Reads ``<output_dir>/<name>/synthetic_function.yaml`` to find the
    function class + params, runs ``compute_true_pareto`` on it, saves
    the result to ``<output_dir>/<name>/_pareto_cache.npy``.

    When ``pool_evals=True``, ``compute_true_pareto`` also pools the
    ``*__<name>__evals.csv`` files already in the subfolder; otherwise
    only NSGA-II runs (initial-setup mode).
    """
    sub = benchmark_subfolder(output_dir, name)
    snap = load_snapshot(sub)

    # Defensive: snapshot's `name` must match the folder it lives in.
    # Editing one without the other would silently mix benchmarks.
    if snap.get("name") != name:
        raise ValueError(
            f"Snapshot at {snapshot_path_for(sub)} has name {snap.get('name')!r}, "
            f"but lives in subfolder {name!r}. Did you rename the folder?"
        )

    fn = _instantiate_function_from_snapshot(snap)
    obj_names = [obj["objective_name"] for obj in fn.objectives]
    evals_dir_arg = str(sub) if pool_evals else None
    pareto_dict = fn.compute_true_pareto(evals_dir=evals_dir_arg)
    front = np.column_stack([pareto_dict[obj] for obj in obj_names])
    logger.info(
        f"{name}: {len(front):,} non-dominated points "
        f"(columns: {obj_names}, pool={pool_evals})"
    )

    save_cached_pareto(sub, front)
    logger.info(f"Saved frozen reference to {cache_path_for(sub)}")
    return sub


def add_args(parser: argparse.ArgumentParser) -> None:
    """Attach pareto-cache-build args to a parser. Used by this module's
    standalone ``main()`` and by ``rag_stack/cli.py``'s ``pareto-cache``
    subcommand registration.
    """
    parser.add_argument(
        "--name",
        required=True,
        action="append",
        help=(
            "Benchmark name (= synthetic_functions[].name from the run "
            "config = subfolder under --output-dir). Repeatable."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            f"Root for per-benchmark subfolders. Default: "
            f"{DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--no-pool",
        action="store_true",
        help=(
            "Skip pool augmentation and run NSGA-II only. Use for the "
            "very first build of a brand-new benchmark (no eval CSVs in "
            "its subfolder yet). Default is to pool — picks up real BO "
            "queries already on disk and produces a much tighter reference."
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")


def cmd_build_pareto_cache(args) -> None:
    """Dispatcher: build cache for each --name, then auto-backfill in pool mode."""
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    pool_evals = not args.no_pool
    for name in args.name:
        build_pareto(name, output_dir=args.output_dir, pool_evals=pool_evals)

    # In pool mode, the new reference is stronger than what historical
    # CSVs in the subfolder were scored against — backfill them so chart
    # numbers reflect the new reference.
    if pool_evals:
        from .recompute_metrics_from_evals import backfill

        for name in args.name:
            logger.info(f"Backfilling snapshot CSVs for {name}…")
            backfill(name, output_dir=args.output_dir)


def main():
    """Standalone entry point — used by ``python -m rag_stack.optimizer.benchmark.build_pareto_cache``."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_args(parser)
    cmd_build_pareto_cache(parser.parse_args())


if __name__ == "__main__":
    main()
