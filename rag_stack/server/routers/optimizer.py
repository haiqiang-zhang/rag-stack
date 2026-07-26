"""Optimizer registry, benchmark results, config browsing, and job management."""

from __future__ import annotations

import csv
import math
import sys
import threading
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..settings_store import load_settings
from ..job_store import load_jobs, launch_job
from ..utils import PROJECT_ROOT, resolve_dir, parse_benchmark_filename

router = APIRouter(tags=["optimizer"])

_DEFAULT_BENCH_DIR = PROJECT_ROOT / "tmp_outputs" / "optimizer_benchmark_results"
_DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs" / "optimizer_benchmark"

# Fallback registry when rag_stack.optimizer can't be imported (heavy deps).
_FALLBACK_REGISTRY = [
    {"key": "ax_qnehvi", "class_name": "AxQNEHVI", "description": "qLogNEHVI via Ax AxClient (single-task GP per outcome)."},
    {"key": "ax_qnehvi_mixedgp", "class_name": "AxQNEHVIMixedGP", "description": "qLogNEHVI via Ax with a mixed categorical+continuous kernel."},
    {"key": "smac_mobo", "class_name": "SMACMOBOOptimizer", "description": "Non-decoupled MOBO baseline using official SMAC3 facade (ParEGO)."},
    {"key": "agent_qnehvi", "class_name": "AgentGuidedQNEHVI", "description": "LLM-agent-guided qLogNEHVI (the project's main optimizer)."},
]


def _get_bench_dir() -> Path:
    return resolve_dir(load_settings().optimizer_benchmark_dir, _DEFAULT_BENCH_DIR)


# ── Registry ────────────────────────────────────────────────────────────────

@router.get("/registry")
def get_optimizer_registry():
    """List all registered optimizer algorithms."""
    try:
        from rag_stack.optimizer import OPTIMIZER_REGISTRY
        registry = []
        for key, cls in OPTIMIZER_REGISTRY.items():
            doc = (cls.__doc__ or "").strip().split("\n")[0]
            registry.append({"key": key, "class_name": cls.__name__, "description": doc})
        return {"registry": registry}
    except Exception:
        return {"registry": _FALLBACK_REGISTRY}


# ── Benchmark results ───────────────────────────────────────────────────────
# CSV format: one file per (optimizer, synthetic_function); the ``seed``
# column distinguishes rows across seeds within the same file.

_NUMERIC_INT = {"seed", "iteration", "cumulative_gt_calls", "cumulative_cm_calls"}
_NUMERIC_FLOAT = {"cumulative_optimizer_time_s", "hv_diff", "igd", "get_pareto_time_s"}


def _iter_benchmark_subfolders(bench_dir: Path):
    """Yield (name, subfolder Path) for every benchmark subfolder under
    bench_dir. A subfolder counts as a benchmark iff it contains an
    ``experiments.yaml`` manifest — anything else (logs, caches,
    backups) is skipped."""
    if not bench_dir.is_dir():
        return
    for sub in sorted(bench_dir.iterdir()):
        if not sub.is_dir():
            continue
        if not (sub / "experiments.yaml").is_file():
            continue
        yield sub.name, sub


def _load_all_csvs(bench_dir: Path, function_filter: Optional[str] = None):
    """Load every snapshot CSV across all benchmark subfolders.

    Layout: ``<bench_dir>/<benchmark_name>/<optimizer>__<benchmark_name>.csv``.
    The benchmark name is the subfolder it lives under — that's the
    canonical identity, not whatever is parsed from the filename.

    Returns ``(rows, functions, optimizers)`` where ``functions`` is the
    sorted list of benchmark names found (= subfolder names).
    """
    rows = []
    functions_set = set()
    optimizers_set = set()

    for name, sub in _iter_benchmark_subfolders(bench_dir):
        if function_filter and name != function_filter:
            continue
        functions_set.add(name)

        for csv_path in sorted(sub.glob("*.csv")):
            info = parse_benchmark_filename(str(csv_path))
            if info is None:
                continue
            optimizers_set.add(info.optimizer)

            with open(csv_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for k in _NUMERIC_INT:
                        raw = row.get(k)
                        row[k] = int(raw) if raw not in (None, "") else 0
                    for k in _NUMERIC_FLOAT:
                        raw = row.get(k)
                        # Snapshot CSVs regenerated from evals don't carry per-iter
                        # wallclock (cumulative_optimizer_time_s / get_pareto_time_s);
                        # those columns end up empty strings here. Use None so the
                        # JSON encoder emits null (NaN/Inf are not valid JSON).
                        # Phase-A-only IGD ("inf" in the CSV when only 1 quality
                        # obs exists) also has to be coerced to None — JSON
                        # serialization will otherwise raise on the FastAPI side.
                        if raw in (None, ""):
                            row[k] = None
                        else:
                            v = float(raw)
                            row[k] = v if math.isfinite(v) else None
                    row["optimizer"] = info.optimizer
                    # The benchmark identity is the subfolder name. The CSV
                    # filename's suffix may or may not match (it's redundant
                    # info, kept only so a single-CSV export is self-describing).
                    row["synthetic_function"] = name
                    rows.append(row)

    return rows, sorted(functions_set), sorted(optimizers_set)


@router.get("/benchmark-results")
def get_benchmark_results(function: Optional[str] = Query(None)):
    """Load benchmark CSV results with metadata columns."""
    # Load all to get the full functions list, then filter rows
    all_rows, all_functions, all_optimizers = _load_all_csvs(_get_bench_dir())
    if function:
        rows = [r for r in all_rows if r.get("synthetic_function") == function]
        optimizers = sorted({r["optimizer"] for r in rows})
    else:
        rows = all_rows
        optimizers = all_optimizers
    return {"rows": rows, "functions": all_functions, "optimizers": optimizers}


@router.delete("/benchmark-results")
def delete_benchmark_result(
    optimizer: str = Query(...),
    function: str = Query(...),
):
    """Delete the CSV files for a single (optimizer, benchmark_name) row.

    ``function`` here is the benchmark name (= subfolder under
    ``bench_dir``). Removes both the trajectory CSV AND its sibling
    eval-distribution CSV in that subfolder. Returns 404 only when
    neither file exists.
    """
    bench_dir = _get_bench_dir()
    if not bench_dir.is_dir():
        raise HTTPException(status_code=404, detail="benchmark directory missing")

    sub = bench_dir / function
    if not sub.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"no benchmark subfolder for {function!r}",
        )

    deleted: list[str] = []
    snap = sub / f"{optimizer}__{function}.csv"
    evals = sub / f"{optimizer}__{function}__evals.csv"

    for path in (snap, evals):
        if path.is_file():
            try:
                path.unlink()
                deleted.append(path.name)
            except OSError as e:
                raise HTTPException(
                    status_code=500, detail=f"unlink {path.name}: {e}"
                ) from e

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"no CSV found for ({optimizer}, {function})",
        )
    return {"deleted": deleted}


@router.get("/benchmark-summary")
def get_benchmark_summary():
    """Aggregated final-checkpoint metrics per (optimizer, function)."""
    rows, _, _ = _load_all_csvs(_get_bench_dir())

    # Group by (optimizer, function, seed), take max iteration row.
    last_per_run: dict[tuple, dict] = {}
    for r in rows:
        key = (r["optimizer"], r["synthetic_function"], r["seed"])
        if key not in last_per_run or r["iteration"] > last_per_run[key]["iteration"]:
            last_per_run[key] = r

    # Aggregate by (optimizer, function)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for (opt, fn, _seed), r in last_per_run.items():
        groups[(opt, fn)].append(r)

    summary = []
    for (opt, fn), runs in sorted(groups.items()):
        hv_diffs = [r["hv_diff"] for r in runs if r["hv_diff"] is not None]
        igds = [r["igd"] for r in runs if r["igd"] is not None]
        gt_calls = [r["cumulative_gt_calls"] for r in runs]
        times = [r["cumulative_optimizer_time_s"] for r in runs if r["cumulative_optimizer_time_s"] is not None]
        n = len(runs)

        def _stats(vals):
            if not vals:
                return None, None
            m = len(vals)
            mean = sum(vals) / m
            std = (sum((x - mean) ** 2 for x in vals) / m) ** 0.5 if m > 1 else 0.0
            return mean, std

        mean_hv, std_hv = _stats(hv_diffs)
        mean_igd, std_igd = _stats(igds)
        mean_gt, std_gt = _stats(gt_calls)
        mean_t, std_t = _stats(times)

        def _round(v, n):
            return round(v, n) if v is not None else None

        seeds_list = sorted({int(r["seed"]) for r in runs})
        summary.append({
            "optimizer": opt,
            "synthetic_function": fn,
            "n_seeds": n,
            "seeds": seeds_list,
            "hv_diff_mean": _round(mean_hv, 6),
            "hv_diff_std": _round(std_hv, 6),
            "igd_mean": _round(mean_igd, 6),
            "igd_std": _round(std_igd, 6),
            "gt_calls": _round(mean_gt, 1),
            "gt_calls_std": _round(std_gt, 1),
            "optimizer_time_s": _round(mean_t, 3),
            "optimizer_time_s_std": _round(std_t, 3),
        })

    return {"summary": summary}


# ── Eval distribution + true Pareto ─────────────────────────────────────────
# Sister artifact `<optimizer>__<function>__evals.csv`, written by
# BenchmarkRunner alongside the trajectory CSV. One row per evaluated
# config (q, s) — used for the eval-distribution plot.

# Background true-Pareto compute. The synthetic function instantiation
# can be very slow (NAS-Bench-201 loads a 5 GB .pth → ~30 s); blocking the
# HTTP handler on it would (a) hold a browser connection slot, queueing
# every other request behind it, and (b) freeze the panel for the user.
#
# Instead we kick off a daemon thread the first time a Pareto is requested,
# return immediately with `true_pareto_pending=True`, and let the client
# poll for the result. Subsequent calls get the cached value or the
# in-flight pending flag — never a 30 s wait.
# Cached payload: (pareto_dict, obj_names_in_order, error). pareto_dict maps
# objective_name → list of values; obj_names captures the canonical order.
_pareto_results: dict[
    str,
    tuple[
        dict[str, list[float]],
        list[str],
        Optional[str],
    ],
] = {}
_pareto_in_flight: dict[str, threading.Thread] = {}
_pareto_state_lock = threading.Lock()


def _compute_true_pareto_blocking(
    benchmark_name: str,
) -> tuple[dict[str, list[float]], list[str], Optional[str]]:
    """Resolve the true Pareto for one benchmark (= subfolder under
    bench_dir). Reads ``synthetic_function.yaml`` from the subfolder to
    learn the function class + params, then prefers the cached
    ``_pareto_cache.npy`` over recomputing.
    """
    from rag_stack.optimizer.benchmark.build_pareto_cache import (
        benchmark_subfolder,
        load_cached_pareto,
        load_snapshot,
    )
    from rag_stack.optimizer.benchmark.synthetic_functions import (
        SYNTHETIC_FUNCTION_REGISTRY,
    )

    sub = benchmark_subfolder(_get_bench_dir(), benchmark_name)
    try:
        snap = load_snapshot(sub)
    except FileNotFoundError as e:
        return ({}, [], f"unknown benchmark: {benchmark_name}")
    except Exception as e:  # noqa: BLE001
        return ({}, [], f"{type(e).__name__}: {e}")

    type_ = snap.get("type")
    cls = SYNTHETIC_FUNCTION_REGISTRY.get(type_)
    if cls is None:
        return ({}, [], f"unknown synthetic function type: {type_}")

    try:
        fn = cls(**(snap.get("params") or {}))
        obj_names = [obj["objective_name"] for obj in fn.objectives]
    except Exception as e:  # noqa: BLE001 — user code on the synthetic function
        return ({}, [], f"{type(e).__name__}: {e}")

    # Prefer the disk cache (lives next to the snapshot in the same
    # subfolder). Falls through to compute_true_pareto for analytical /
    # enumerable functions where no cache is needed.
    cached = load_cached_pareto(sub, len(obj_names))
    if cached is not None:
        return (
            {name: cached[:, i].tolist() for i, name in enumerate(obj_names)},
            obj_names,
            None,
        )

    try:
        pareto = fn.compute_true_pareto()
    except Exception as e:  # noqa: BLE001 — user code on the synthetic function
        return ({}, [], f"{type(e).__name__}: {e}")
    payload = {name: [float(v) for v in pareto[name]] for name in obj_names}
    return payload, obj_names, None


def _request_true_pareto(
    function_name: str,
) -> tuple[Optional[dict[str, list[float]]], Optional[list[str]], Optional[str], bool]:
    """Return (pareto_dict_or_None, obj_names_or_None, error, pending)."""
    with _pareto_state_lock:
        if function_name in _pareto_results:
            payload, obj_names, err = _pareto_results[function_name]
            return payload, obj_names, err, False
        existing = _pareto_in_flight.get(function_name)
        if existing is not None and existing.is_alive():
            return None, None, None, True

        def _worker(name: str = function_name) -> None:
            result = _compute_true_pareto_blocking(name)
            with _pareto_state_lock:
                _pareto_results[name] = result
                _pareto_in_flight.pop(name, None)

        thread = threading.Thread(
            target=_worker, name=f"true-pareto-{function_name}", daemon=True
        )
        _pareto_in_flight[function_name] = thread
        thread.start()
        return None, None, None, True


# Standard scalar fields written by BenchmarkRunner.save_eval_history; any
# extra columns are objective names (in objectives() order).
# `noise_sigma` and `config_json` are bookkeeping columns added in the new
# CSV format — they are not objectives and would fail the float() parse.
_EVAL_SCALAR_FIELDS = {
    "seed", "noise_sigma", "phase", "iteration",
    "cumulative_gt_calls", "cumulative_cm_calls",
    "config_json",
}


def _load_eval_csvs(bench_dir: Path, function: str) -> tuple[list[dict], list[str]]:
    """Read every ``*__evals.csv`` inside ``<bench_dir>/<function>/``.

    ``function`` is the benchmark name (= subfolder). Returns
    ``(optimizers, obj_names)`` where ``obj_names`` is the union of
    objective columns observed across all CSVs (preserving header order
    from the first file).
    """
    out: list[dict] = []
    obj_names: list[str] = []
    sub = bench_dir / function
    if not sub.is_dir():
        return out, obj_names
    suffix = f"__{function}__evals"
    for path in sorted(sub.glob("*__evals.csv")):
        stem = path.stem
        if not stem.endswith(suffix):
            continue
        opt_name = stem[: -len(suffix)]
        evals: list[dict] = []
        with open(path) as f:
            reader = csv.DictReader(f)
            file_obj_cols = [
                c for c in (reader.fieldnames or []) if c not in _EVAL_SCALAR_FIELDS
            ]
            if not obj_names:
                obj_names = list(file_obj_cols)
            for r in reader:
                try:
                    record = {
                        "seed": int(r["seed"]),
                        "iteration": int(r["iteration"]),
                        "cumulative_gt_calls": int(r.get("cumulative_gt_calls") or 0),
                        "phase": r.get("phase", ""),
                        "obj_values": {
                            c: float(r[c]) for c in file_obj_cols if r.get(c) not in (None, "")
                        },
                    }
                    evals.append(record)
                except (KeyError, ValueError):
                    continue
        if evals:
            out.append({"name": opt_name, "evals": evals})
    return out, obj_names


@router.get("/eval-distribution")
def get_eval_distribution(function: str = Query(...)):
    """Return per-eval samples + true Pareto front for one function.

    Response shape::

      {
        "function": str,
        "obj_names": [str, str, ...],        # canonical objectives() order
        "seeds": [int, ...],
        "true_pareto": [{<obj_name>: float, ...}, ...],   # one dict per Pareto point
        "true_pareto_error": str | null,
        "true_pareto_pending": bool,
        "optimizers": [
          {
            "name": str,
            "evals": [
              {seed, iteration, cumulative_gt_calls, phase,
               obj_values: {<obj_name>: float, ...}},
              ...
            ],
          },
          ...
        ],
      }
    """
    bench_dir = _get_bench_dir()
    if not bench_dir.is_dir():
        raise HTTPException(status_code=404, detail="benchmark directory missing")

    optimizers, csv_obj_names = _load_eval_csvs(bench_dir, function)
    seeds: set[int] = set()
    for o in optimizers:
        for r in o["evals"]:
            seeds.add(r["seed"])

    pareto_dict, pareto_obj_names, pareto_error, pareto_pending = _request_true_pareto(function)
    obj_names = pareto_obj_names or csv_obj_names
    true_pareto: list[dict[str, float]] = []
    if pareto_dict is not None and pareto_obj_names:
        n_pts = len(next(iter(pareto_dict.values()), []))
        # Sort by the second objective so the frontend's `line` series draws
        # one monotone polyline. _non_dominated_*-helpers return points in
        # arbitrary order, which used to render as zigzag (= the "many lines"
        # bug after the dict-keyed refactor).
        sort_key = pareto_obj_names[1] if len(pareto_obj_names) > 1 else pareto_obj_names[0]
        order = sorted(range(n_pts), key=lambda i: pareto_dict[sort_key][i])
        for i in order:
            true_pareto.append({name: pareto_dict[name][i] for name in pareto_obj_names})

    return {
        "function": function,
        "obj_names": obj_names,
        "seeds": sorted(seeds),
        "true_pareto": true_pareto,
        "true_pareto_error": pareto_error,
        "true_pareto_pending": pareto_pending,
        "optimizers": optimizers,
    }


# ── Configs ─────────────────────────────────────────────────────────────────

def _get_configs_dir() -> Path:
    return resolve_dir(load_settings().optimizer_configs_dir, _DEFAULT_CONFIGS_DIR)


@router.get("/configs")
def list_configs():
    """List available benchmark YAML configs."""
    configs_dir = _get_configs_dir()
    if not configs_dir.is_dir():
        return {"configs": [], "configs_dir": str(configs_dir)}

    configs = []
    for p in sorted(configs_dir.glob("*.yaml")):
        try:
            with open(p) as f:
                data = yaml.safe_load(f)
            sf = data.get("synthetic_function", {})
            configs.append({
                "filename": p.name,
                "optimizer_type": data.get("optimizer", {}).get("type", ""),
                "synthetic_function": sf if isinstance(sf, str) else sf.get("type", ""),
                "n_iterations": data.get("n_iterations", 0),
                "seed": data.get("seed", 0),
                "output_dir": data.get("output_dir", ""),
                "content": data,
            })
        except Exception:
            continue

    return {"configs": configs, "configs_dir": str(configs_dir)}


# ── Benchmark run ───────────────────────────────────────────────────────────

class ConfigSaveRequest(BaseModel):
    filename: str
    content: str  # raw YAML text


class BenchmarkRunRequest(BaseModel):
    config_filename: str


@router.put("/configs")
def save_config(body: ConfigSaveRequest):
    """Save (overwrite) a YAML config file."""
    configs_dir = _get_configs_dir()
    if not body.filename.endswith(".yaml"):
        raise HTTPException(400, f"Filename must end with .yaml, got: {body.filename!r}")
    path = configs_dir / body.filename
    if not path.resolve().parent == configs_dir.resolve():
        raise HTTPException(400, f"Invalid filename: {body.filename!r}")
    # Validate YAML parses
    try:
        yaml.safe_load(body.content)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"Invalid YAML: {e}")
    path.write_text(body.content)
    return {"ok": True}


@router.post("/benchmark-run")
def run_benchmark(body: BenchmarkRunRequest):
    """Launch a benchmark as a detached background process."""
    configs_dir = _get_configs_dir()
    config_path = configs_dir / body.config_filename

    if not config_path.is_file():
        raise HTTPException(404, f"Config not found: {body.config_filename}")
    if not config_path.resolve().parent == configs_dir.resolve():
        raise HTTPException(400, "Invalid filename")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "tmp_outputs/optimizer_benchmark_results")
    abs_output = Path(output_dir)
    if not abs_output.is_absolute():
        abs_output = PROJECT_ROOT / abs_output

    return launch_job(
        cmd=[sys.executable, "-m", "rag_stack.optimizer.benchmark.run", "--config", str(config_path.resolve())],
        config=config,
        output_dir=str(abs_output),
        job_type="optimizer_benchmark",
        log_name=f"_job_{body.config_filename}.log",
    )


# ── Jobs ────────────────────────────────────────────────────────────────────

@router.get("/jobs")
def list_jobs():
    """List optimizer benchmark jobs with live status."""
    return {"jobs": load_jobs(job_type="optimizer_benchmark")}
