"""FAISS IVF-PQ cost model extension endpoints.

Exposes hardware presets, calibration profiles, and benchmark results
specific to the FAISS IVF-PQ analytical cost model.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from . import register_router
from ..settings_store import load_settings
from ..utils import PROJECT_ROOT, resolve_dir

router = APIRouter(tags=["faiss-ivf"])

# Profiles live under ``calibration_profiles/faiss_ivf/`` — cost-model
# specific subdir matching the directory scheme used by llm_sim, so a single
# settings entry (``calibration_profiles_dir``) is shared across all cost
# model families.
_DEFAULT_PROFILES_ROOT = PROJECT_ROOT / "calibration_profiles"
_DEFAULT_BENCH_DIR = PROJECT_ROOT / "tmp_outputs" / "faiss_sim_bench"


def _get_profiles_dir() -> Path:
    root = resolve_dir(load_settings().calibration_profiles_dir, _DEFAULT_PROFILES_ROOT)
    return root / "faiss_ivf"


def _get_bench_root() -> Path:
    return resolve_dir(load_settings().benchmark_results_dir, _DEFAULT_BENCH_DIR)


@router.get("/presets")
def list_presets():
    """List available FAISS hardware presets."""
    from rag_stack.cost_model.faiss_ivf_sim.system_presets import SYSTEM_PRESETS

    presets = []
    for name, factory in SYSTEM_PRESETS.items():
        doc = (factory.__doc__ or "").strip().split("\n")[0]
        presets.append({"name": name, "description": doc})
    return {"presets": presets}


# /profiles and /profiles/{filename} are served by the generic base_router
# at /api/cost-model/{name}/profiles (reading from
# ``calibration_profiles/faiss_ivf/`` via the class ``profiles_subdir()``).
# Keep this extension router focused on FAISS-unique endpoints.


@router.get("/benchmark-folders")
def list_benchmark_folders():
    """List available result subfolders under the benchmark root directory."""
    bench_root = _get_bench_root()
    if not bench_root.is_dir():
        return {"root": str(bench_root), "folders": []}

    folders = []
    for d in sorted(bench_root.iterdir()):
        if d.is_dir() and (d / "benchmark_results.csv").is_file():
            csv_path = d / "benchmark_results.csv"
            # Count rows
            with open(csv_path) as f:
                n_rows = sum(1 for _ in f) - 1  # minus header
            folders.append({"name": d.name, "n_rows": max(n_rows, 0)})

    return {"root": str(bench_root), "folders": folders}


def _load_benchmark_csv(csv_path: Path) -> dict:
    """Parse a benchmark CSV into rows + summary."""
    if not csv_path.is_file():
        return {"rows": [], "summary": []}

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ("batch_size", "parallel_mode", "nprobe"):
                if key in row:
                    row[key] = int(row[key])
            for key in (
                "actual_latency_ms", "predicted_latency_ms", "latency_error_pct",
                "actual_qps", "predicted_qps", "qps_error_pct",
            ):
                if key in row:
                    row[key] = float(row[key])
            rows.append(row)

    groups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        key = f"{r['index_name']}|bs={r['batch_size']}|pm={r['parallel_mode']}"
        groups[key].append(abs(r["latency_error_pct"]))

    summary = []
    for label, errors in groups.items():
        mape = sum(errors) / len(errors)
        summary.append({
            "label": label,
            "mape_latency": round(mape, 2),
            "max_ape_latency": round(max(errors), 2),
            "num_points": len(errors),
        })

    return {"rows": rows, "summary": summary}


@router.get("/benchmarks")
def list_benchmarks(folder: Optional[str] = Query(None)):
    """Load benchmark results. If folder is given, load from that subfolder."""
    bench_root = _get_bench_root()

    if folder:
        target = bench_root / folder
        if not target.is_dir():
            raise HTTPException(404, f"Folder not found: {folder}")
        if not target.resolve().parent == bench_root.resolve():
            raise HTTPException(400, "Invalid folder name")
        return _load_benchmark_csv(target / "benchmark_results.csv")

    # Legacy: try root-level CSV first, then first subfolder
    root_csv = bench_root / "benchmark_results.csv"
    if root_csv.is_file():
        return _load_benchmark_csv(root_csv)

    # Auto-pick first subfolder
    for d in sorted(bench_root.iterdir()):
        if d.is_dir() and (d / "benchmark_results.csv").is_file():
            return _load_benchmark_csv(d / "benchmark_results.csv")

    return {"rows": [], "summary": []}


# Capabilities are declared on the FaissIVFCostModel class
# (``capabilities()`` classmethod).
register_router("faiss_ivf", router)
