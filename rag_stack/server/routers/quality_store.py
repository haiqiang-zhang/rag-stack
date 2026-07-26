"""Quality store management — create, list, status, delete.

Quality stores live in {PROJECT_ROOT}/quality_stores/ by default (git tracked).
Each store is a directory with manifest.json + configs/ + quality/ + pipeline_configs/.
"""

from __future__ import annotations

import json
import os
import subprocess
import shutil
import sys
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from rag_stack.quality_store_contract import (
    validate_quality_store_manifest,
    validate_quality_store_record_marker,
)

from ..settings_store import load_settings
from ..job_store import _is_pid_alive
from ..utils import PROJECT_ROOT, resolve_dir

router = APIRouter(tags=["quality-store"])

_DEFAULT_STORE_DIR = PROJECT_ROOT / "quality_stores"
_DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs" / "rag_stack"
_DEFAULT_DATASETS_DIR = PROJECT_ROOT / "datasets"


def _read_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    validate_quality_store_manifest(manifest, source=str(path))
    return manifest


def _committed_hashes(store_path: Path, manifest: dict) -> set[str]:
    """Return only complete records with valid commit markers."""
    try:
        validate_quality_store_manifest(
            manifest, source=str(store_path / "manifest.json"),
        )
    except (TypeError, ValueError):
        return set()
    required = ["configs", "quality", "pipeline_configs", "traces", "records"]
    sets: list[set[str]] = []
    for dirname in required:
        path = store_path / dirname
        if not path.is_dir():
            return set()
        sets.append({file.stem for file in path.glob("*.json")})
    committed = set.intersection(*sets) if sets else set()
    strict: set[str] = set()
    for record_hash in committed:
        marker_path = store_path / "records" / f"{record_hash}.json"
        try:
            with marker_path.open(encoding="utf-8") as stream:
                marker = json.load(stream)
            validate_quality_store_record_marker(
                marker,
                source=str(marker_path),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
        strict.add(record_hash)
    return strict


def _get_store_dir() -> Path:
    return resolve_dir(
        getattr(load_settings(), "quality_stores_dir", ""),
        _DEFAULT_STORE_DIR,
    )


def _get_configs_dir() -> Path:
    return resolve_dir(load_settings().rag_optimization_configs_dir, _DEFAULT_CONFIGS_DIR)


def _get_datasets_dir() -> Path:
    return resolve_dir(load_settings().datasets_dir, _DEFAULT_DATASETS_DIR)


# ── List stores ────────────────────────────────────────────────────────────

@router.get("/list")
def list_stores():
    """List all quality stores with summary info."""
    store_dir = _get_store_dir()
    if not store_dir.is_dir():
        return {"stores": [], "stores_dir": str(store_dir)}

    stores = []
    for d in sorted(store_dir.iterdir()):
        if not d.is_dir():
            continue
        manifest_path = d / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = _read_manifest(manifest_path)

            # Derive status from manifest
            status = manifest.get("status", "unknown")
            pid = manifest.get("pid")
            if status == "running" and pid:
                if not _is_pid_alive(pid):
                    status = "failed"
                    # Fix on disk
                    manifest["status"] = "failed"
                    manifest["pid"] = None
                    with open(manifest_path, "w") as f:
                        json.dump(manifest, f, indent=2)

            # Count actual configs on disk (manifest may lag during sweep)
            n_configs_disk = len(_committed_hashes(d, manifest))
            n_configs = max(manifest.get("n_configs", 0), n_configs_disk)

            stores.append({
                "name": d.name,
                "n_configs": n_configs,
                "n_failed": len(manifest.get("failed_hashes", [])),
                "strategy": manifest.get("strategy", ""),
                "n_grid": manifest.get("n_grid"),
                "created_at": manifest.get("created_at", ""),
                "updated_at": manifest.get("updated_at", ""),
                "status": status,
                "store_path": str(d),
                "dataset": manifest.get("dataset", {}),
            })
        except Exception:
            continue

    return {"stores": stores, "stores_dir": str(store_dir)}


# ── Get store detail ───────────────────────────────────────────────────────

@router.get("/{name}")
def get_store(name: str):
    """Get full manifest for a quality store."""
    store_path = _get_store_dir() / name
    if not store_path.is_dir():
        raise HTTPException(404, f"Quality store not found: {name}")

    manifest_path = store_path / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(404, "No manifest.json found")

    manifest = _read_manifest(manifest_path)

    # Count actual configs on disk (manifest may lag during running sweep)
    n_configs_disk = len(_committed_hashes(store_path, manifest))
    manifest["n_configs"] = max(manifest.get("n_configs", 0), n_configs_disk)

    return {"name": name, "store_path": str(store_path), **manifest}


# ── Get store configs ──────────────────────────────────────────────────────

@router.get("/{name}/configs")
def get_store_configs(name: str):
    """List evaluated configs with quality scores.

    Scans the actual configs/ and quality/ directories on disk, so results
    are visible immediately during a running sweep (not only after manifest
    update).
    """
    store_path = _get_store_dir() / name
    if not store_path.is_dir():
        raise HTTPException(404, f"Quality store not found: {name}")

    configs_dir = store_path / "configs"
    quality_dir = store_path / "quality"
    if not configs_dir.is_dir():
        return {"configs": [], "n_configs": 0}

    manifest_path = store_path / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(404, "No manifest.json found")
    manifest = _read_manifest(manifest_path)
    committed = _committed_hashes(store_path, manifest)
    configs = []
    for cfg_file in sorted(configs_dir.glob("*.json")):
        h = cfg_file.stem
        if h not in committed:
            continue
        quality_path = quality_dir / f"{h}.json"
        if not quality_path.is_file():
            continue
        try:
            with open(cfg_file) as f:
                config = json.load(f)
            with open(quality_path) as f:
                quality = json.load(f)
            from rag_stack.utils import aggregate_quality_metrics

            record_spec = None
            record_path = store_path / "records" / f"{h}.json"
            if record_path.is_file():
                with open(record_path) as f:
                    record_spec = (
                        (json.load(f) or {}).get("quality_spec") or {}
                    ).get("combined_quality")
            combined = aggregate_quality_metrics(
                quality, record_spec or manifest.get("combined_quality"),
            )
            configs.append({
                "config_hash": h,
                "config": config,
                "quality_metrics": quality,
                "quality_combined": combined,
            })
        except Exception:
            continue

    return {"configs": configs, "n_configs": len(configs)}


# ── Get single pipeline config ─────────────────────────────────────────────

@router.get("/{name}/pipeline-config/{config_hash}")
def get_pipeline_config(name: str, config_hash: str):
    """Get the full resolved pipeline config for a specific evaluation."""
    store_path = _get_store_dir() / name
    if not store_path.is_dir():
        raise HTTPException(404, f"Quality store not found: {name}")

    manifest_path = store_path / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(404, "No manifest.json found")
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if config_hash not in _committed_hashes(store_path, manifest):
        raise HTTPException(404, f"Committed config not found: {config_hash}")

    pc_path = store_path / "pipeline_configs" / f"{config_hash}.json"
    if not pc_path.is_file():
        raise HTTPException(404, f"Pipeline config not found: {config_hash}")

    with open(pc_path) as f:
        return json.load(f)


# ── Create store + launch sweep ────────────────────────────────────────────

class CreateQualityStoreRequest(BaseModel):
    name: str
    config_filename: str
    corpus_data: str
    qa_data: str
    n_grid: int = 5
    n_configs: int = 0       # for sobol strategy
    strategy: str = "exhaustive"  # "exhaustive" | "sobol"


@router.post("/create")
def create_store(body: CreateQualityStoreRequest):
    """Create a quality store and launch sweep as background process."""
    store_dir = _get_store_dir()
    store_path = store_dir / body.name

    if "/" in body.name or ".." in body.name:
        raise HTTPException(400, "Invalid store name")
    if store_path.exists():
        # Allow re-running failed stores
        manifest_path = store_path / "manifest.json"
        if manifest_path.is_file():
            manifest = _read_manifest(manifest_path)
            status = manifest.get("status", "")
            if status not in ("", "failed"):
                raise HTTPException(409, f"Quality store '{body.name}' already exists with status '{status}'")

    # Resolve config
    configs_dir = _get_configs_dir()
    config_src = configs_dir / body.config_filename
    if not config_src.is_file():
        raise HTTPException(404, f"Config not found: {body.config_filename}")

    # Resolve dataset paths
    datasets_dir = _get_datasets_dir()
    corpus_path = datasets_dir / body.corpus_data
    qa_path = datasets_dir / body.qa_data
    if not corpus_path.is_file():
        raise HTTPException(404, f"Corpus file not found: {body.corpus_data}")
    if not qa_path.is_file():
        raise HTTPException(404, f"QA file not found: {body.qa_data}")

    store_path.mkdir(parents=True, exist_ok=True)

    # Launch subprocess
    log_path = store_path / "_sweep.log"
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "rag_stack.cli", "quality-sweep",
            "--config", str(config_src.resolve()),
            "--qa-data", str(qa_path.resolve()),
            "--corpus-data", str(corpus_path.resolve()),
            "--store-path", str(store_path.resolve()),
            "--n-grid", str(body.n_grid),
            "--n-configs", str(body.n_configs),
            "--strategy", body.strategy,
        ],
        cwd=str(PROJECT_ROOT),
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    threading.Thread(target=proc.wait, daemon=True).start()

    return {"ok": True, "name": body.name, "pid": proc.pid, "store_path": str(store_path)}


# ── Store status ───────────────────────────────────────────────────────────

@router.get("/{name}/status")
def get_store_status(name: str):
    """Get quality store sweep status."""
    store_path = _get_store_dir() / name
    if not store_path.is_dir():
        raise HTTPException(404, f"Quality store not found: {name}")

    manifest_path = store_path / "manifest.json"
    if not manifest_path.is_file():
        return {"status": "unknown", "name": name}

    manifest = _read_manifest(manifest_path)

    status = manifest.get("status", "unknown")
    pid = manifest.get("pid")
    pid_alive = False

    if status == "running" and pid:
        pid_alive = _is_pid_alive(pid)
        if not pid_alive:
            status = "failed"
            manifest["status"] = "failed"
            manifest["pid"] = None
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

    return {
        "name": name,
        "status": status,
        "pid": pid,
        "pid_alive": pid_alive,
        "n_configs": manifest.get("n_configs", 0),
        "n_failed": len(manifest.get("failed_hashes", [])),
    }


# ── Kill store sweep ──────────────────────────────────────────────────────

@router.post("/{name}/kill")
def kill_store(name: str):
    """Kill a running quality store sweep."""
    import signal
    store_path = _get_store_dir() / name
    if not store_path.is_dir():
        raise HTTPException(404, f"Quality store not found: {name}")

    manifest_path = store_path / "manifest.json"
    if not manifest_path.is_file():
        raise HTTPException(400, "No manifest.json found")

    manifest = _read_manifest(manifest_path)

    pid = manifest.get("pid")
    if not pid or manifest.get("status") != "running":
        raise HTTPException(400, "Store sweep is not running")

    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass

    manifest["status"] = "failed"
    manifest["pid"] = None
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return {"ok": True, "name": name}


# ── Delete store ──────────────────────────────────────────────────────────

@router.delete("/{name}")
def delete_store(name: str):
    """Delete a quality store and all its data."""
    store_dir = _get_store_dir()
    store_path = store_dir / name

    if not store_path.is_dir():
        raise HTTPException(404, f"Quality store not found: {name}")
    if store_path.resolve().parent != store_dir.resolve():
        raise HTTPException(400, "Invalid store name (path traversal)")

    # Don't delete running stores
    manifest_path = store_path / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_manifest(manifest_path)
        pid = manifest.get("pid")
        if manifest.get("status") == "running" and pid and _is_pid_alive(pid):
            raise HTTPException(409, "Cannot delete a running quality store. Kill it first.")

    shutil.rmtree(str(store_path), ignore_errors=True)
    return {"ok": True, "name": name}
