"""Dataset management — browsing, generation config editing, job launching.

Generation jobs are tracked via summary.json (with job_type="dataset_generation")
inside each output folder in datasets_dir, NOT via the global job_store.
This means CLI-launched generation jobs are also visible to the UI.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..settings_store import load_settings
from ..job_store import _is_pid_alive
from ..utils import PROJECT_ROOT, resolve_dir

from rag_stack.dataset_gen_summary import (
    DATASET_GEN_CONFIG_FILENAME,
    DatasetGenStatus,
    DatasetGenSummary,
    create_dataset_gen_summary,
    load_dataset_gen_summary,
    save_dataset_gen_summary,
)

router = APIRouter(tags=["dataset"])

_DEFAULT_DATASETS_DIR = PROJECT_ROOT / "datasets"
_DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs" / "dataset_generator"


def _get_datasets_dir() -> Path:
    return resolve_dir(load_settings().datasets_dir, _DEFAULT_DATASETS_DIR)


def _get_configs_dir() -> Path:
    return resolve_dir(load_settings().dataset_generator_configs_dir, _DEFAULT_CONFIGS_DIR)


# ── Registry ───────────────────────────────────────────────────────────────

@router.get("/registry")
def get_registry():
    """List registered dataset generator strategies."""
    from rag_stack.dataset_generator.registry import _STRATEGY_REGISTRY

    strategies = []
    for name, (module_path, class_name) in _STRATEGY_REGISTRY.items():
        strategies.append({
            "name": name,
            "module_path": module_path,
            "class_name": class_name,
        })
    return {"strategies": strategies}


# ── Dataset browsing ───────────────────────────────────────────────────────

@router.get("/list")
def list_datasets():
    """Browse all datasets in datasets_dir with parquet file info."""
    datasets_dir = _get_datasets_dir()
    if not datasets_dir.is_dir():
        return {"datasets": [], "datasets_dir": str(datasets_dir)}

    datasets = []
    for d in sorted(datasets_dir.iterdir()):
        if not d.is_dir():
            continue
        parquets = []
        for f in sorted(d.glob("*.parquet")):
            rows = 0
            try:
                # Datasets are downloaded on demand; a folder may hold dangling
                # symlinks to not-yet-fetched files. Skip what can't be stat'ed.
                size_bytes = f.stat().st_size
            except OSError:
                continue
            try:
                import pyarrow.parquet as pq
                rows = pq.read_metadata(str(f)).num_rows
            except Exception:
                pass
            parquets.append({
                "filename": f.name,
                "rows": rows,
                "size_bytes": size_bytes,
            })
        if not parquets:
            continue

        # Check if this folder has a generation summary
        gen_summary = load_dataset_gen_summary(str(d))
        datasets.append({
            "name": d.name,
            "parquet_files": parquets,
            "has_generation_summary": gen_summary is not None,
            "generation_status": gen_summary.status if gen_summary else None,
            "generation_strategy": gen_summary.strategy if gen_summary else None,
        })

    return {"datasets": datasets, "datasets_dir": str(datasets_dir)}


# ── Dataset preview ────────────────────────────────────────────────────────

@router.get("/{name}/preview")
def preview_dataset(name: str, file: str = Query(...), limit: int = Query(20)):
    """Preview first N rows of a parquet file in a dataset folder."""
    datasets_dir = _get_datasets_dir()
    dataset_path = datasets_dir / name
    if not dataset_path.is_dir():
        raise HTTPException(404, f"Dataset not found: {name}")

    file_path = dataset_path / file
    if not file_path.is_file() or file_path.suffix != ".parquet":
        raise HTTPException(404, f"Parquet file not found: {file}")
    if not file_path.resolve().parent == dataset_path.resolve():
        raise HTTPException(400, "Invalid file path")

    try:
        import json as _json
        import numpy as np
        import pandas as pd
        from fastapi.responses import JSONResponse

        df = pd.read_parquet(str(file_path))
        preview = df.head(limit)

        def _to_python(v):
            """Recursively convert numpy/nested types to plain Python."""
            if v is None:
                return None
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                f = float(v)
                return None if np.isnan(f) else f
            if isinstance(v, float) and np.isnan(v):
                return None
            if isinstance(v, np.ndarray):
                return [_to_python(x) for x in v]
            if isinstance(v, (list, tuple)):
                return [_to_python(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _to_python(val) for k, val in v.items()}
            if isinstance(v, (str, int, float, bool)):
                return v
            return str(v)

        def _safe(v):
            """Convert to a display-friendly string for complex types."""
            converted = _to_python(v)
            if isinstance(converted, (list, dict)):
                return _json.dumps(converted, ensure_ascii=False)
            return converted

        records = []
        for _, row in preview.iterrows():
            records.append({col: _safe(row[col]) for col in df.columns})

        return JSONResponse({"columns": list(df.columns), "rows": records, "total_rows": len(df)})
    except Exception as e:
        raise HTTPException(500, f"Failed to read parquet: {e}")


# ── Config listing ─────────────────────────────────────────────────────────

@router.get("/configs")
def list_configs():
    """List available dataset generation YAML configs."""
    configs_dir = _get_configs_dir()
    if not configs_dir.is_dir():
        return {"configs": []}

    configs = []
    for p in sorted(configs_dir.glob("*.yaml")):
        try:
            with open(p) as f:
                data = yaml.safe_load(f)
            configs.append({
                "filename": p.name,
                "strategy": data.get("strategy", ""),
                "content": data,
            })
        except Exception:
            continue

    return {"configs": configs}


# ── Config save ────────────────────────────────────────────────────────────

class SaveConfigRequest(BaseModel):
    filename: str
    content: str  # raw YAML text


@router.put("/configs")
def save_config(body: SaveConfigRequest):
    """Save (overwrite) a dataset generation config YAML."""
    configs_dir = _get_configs_dir()
    configs_dir.mkdir(parents=True, exist_ok=True)

    target = configs_dir / body.filename
    if not target.resolve().parent == configs_dir.resolve():
        raise HTTPException(400, "Invalid filename")

    # Validate YAML
    try:
        yaml.safe_load(body.content)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"Invalid YAML: {e}")

    target.write_text(body.content)
    return {"ok": True}


# ── Generation launch ──────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    config_filename: str
    output_name: str


@router.post("/generate")
def launch_generation(body: GenerateRequest):
    """Launch dataset generation as a detached background process."""
    configs_dir = _get_configs_dir()
    config_src = configs_dir / body.config_filename
    if not config_src.is_file():
        raise HTTPException(404, f"Config not found: {body.config_filename}")
    if not config_src.resolve().parent == configs_dir.resolve():
        raise HTTPException(400, "Invalid config filename")

    # Validate output name
    if "/" in body.output_name or ".." in body.output_name:
        raise HTTPException(400, "Invalid output name")

    datasets_dir = _get_datasets_dir()
    output_dir = datasets_dir / body.output_name

    # Check if already running
    existing = load_dataset_gen_summary(str(output_dir))
    if existing and existing.status == DatasetGenStatus.RUNNING:
        if existing.pid and _is_pid_alive(existing.pid):
            raise HTTPException(409, f"Generation already running (PID {existing.pid})")

    # Create output dir and initial summary
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read strategy from config
    try:
        with open(config_src) as f:
            config_data = yaml.safe_load(f)
        strategy = config_data.get("strategy", "unknown")
    except Exception:
        strategy = "unknown"

    summary = create_dataset_gen_summary(
        strategy=strategy,
        config_filename=body.config_filename,
    )
    save_dataset_gen_summary(str(output_dir), summary)

    # Launch subprocess
    log_path = output_dir / "_generation.log"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "rag_stack.dataset_gen_run",
            "--config", str(config_src.resolve()),
            "--output-dir", str(output_dir.resolve()),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # Daemon reaper thread to prevent zombies
    threading.Thread(target=proc.wait, daemon=True).start()

    return {"ok": True, "output_name": body.output_name, "pid": proc.pid}


# ── Job listing ────────────────────────────────────────────────────────────

@router.get("/jobs")
def list_jobs():
    """List dataset generation jobs by scanning datasets_dir for summary.json files."""
    datasets_dir = _get_datasets_dir()
    if not datasets_dir.is_dir():
        return {"jobs": []}

    jobs = []
    for d in sorted(datasets_dir.iterdir()):
        if not d.is_dir():
            continue
        summary = load_dataset_gen_summary(str(d))
        if summary is None:
            continue

        result = summary.model_dump()
        result["name"] = d.name

        # PID liveness check for running jobs
        if summary.status == DatasetGenStatus.RUNNING and summary.pid:
            if not _is_pid_alive(summary.pid):
                result["status"] = "failed"
                result["pid_alive"] = False
                summary.status = DatasetGenStatus.FAILED
                save_dataset_gen_summary(str(d), summary)
            else:
                result["pid_alive"] = True
        else:
            result["pid_alive"] = False

        jobs.append(result)

    return {"jobs": jobs}


# ── Single job status ──────────────────────────────────────────────────────

@router.get("/jobs/{name}/status")
def get_job_status(name: str):
    """Get live job status from summary.json with PID liveness check."""
    datasets_dir = _get_datasets_dir()
    job_dir = datasets_dir / name
    if not job_dir.is_dir():
        raise HTTPException(404, f"Job folder not found: {name}")

    summary = load_dataset_gen_summary(str(job_dir))
    if summary is None:
        raise HTTPException(404, "No generation summary.json found")

    result = summary.model_dump()
    result["name"] = name

    if summary.status == DatasetGenStatus.RUNNING and summary.pid:
        if not _is_pid_alive(summary.pid):
            result["status"] = "failed"
            result["pid_alive"] = False
            summary.status = DatasetGenStatus.FAILED
            save_dataset_gen_summary(str(job_dir), summary)
        else:
            result["pid_alive"] = True
    else:
        result["pid_alive"] = False

    return result


# ── Kill job ───────────────────────────────────────────────────────────────

@router.post("/jobs/{name}/kill")
def kill_job(name: str):
    """Kill a running dataset generation process."""
    datasets_dir = _get_datasets_dir()
    job_dir = datasets_dir / name
    if not job_dir.is_dir():
        raise HTTPException(404, f"Job folder not found: {name}")

    summary = load_dataset_gen_summary(str(job_dir))
    if not summary or summary.status != DatasetGenStatus.RUNNING or not summary.pid:
        raise HTTPException(400, "Job is not running")

    try:
        os.kill(summary.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    summary.status = DatasetGenStatus.FAILED
    save_dataset_gen_summary(str(job_dir), summary)
    return {"ok": True, "name": name}
