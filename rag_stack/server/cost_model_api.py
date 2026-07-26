"""Cost Model Management API.

FastAPI application that exposes:
- Generic endpoints for all registered cost models (estimate, sweep, registry)
- Extension routers for model-specific features (calibration, benchmarks, etc.)
- Global application settings

Usage:
    uvicorn rag_stack.server.cost_model_api:app --port 8000 --reload --reload-dir rag_stack
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from math import isfinite
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from rag_stack.cost_model import COST_MODEL_REGISTRY
from rag_stack.cost_model.base import CostModelModule
from rag_stack.types import SystemConfig
from rag_stack.calibration_paths import calibration_data_root, calibration_profile_root
from rag_stack.llm_sim_profile_schema import unwrap_llm_sim_profile_document

from .dependencies import get_cost_model, parse_hardware
from .calibration_commands import calibration_commands, command_options
from .cost_model_catalog import build_cost_model_catalog, mapped_components
from .routers import EXTENSION_ROUTERS
from .schemas import (
    AppSettings,
    CostModelInfo,
    EndpointInfo,
    EstimateBatchRequest,
    EstimateRequest,
    EstimateResponse,
    OperatorBreakdownResponse,
    RegistryResponse,
    SweepRequest,
)
from .settings_store import load_settings, save_settings, load_env_keys, save_env_keys, apply_env_keys


# ── Base router (generic endpoints for all cost models) ──────────────────────

base_router = APIRouter(tags=["cost-model"])


def _build_model_info(name: str, cls: type[CostModelModule]) -> CostModelInfo:
    """Build CostModelInfo for a single registered cost model."""
    mapped_types = mapped_components(name)
    capabilities = cls.capabilities()
    extensions: list[EndpointInfo] = []
    ext_router = EXTENSION_ROUTERS.get(name)
    if ext_router is not None:
        for route in ext_router.routes:
            if hasattr(route, "methods") and hasattr(route, "path"):
                for method in route.methods:
                    extensions.append(
                        EndpointInfo(
                            path=f"/api/cost-model/{name}{route.path}",
                            method=method,
                            summary=getattr(route, "summary", "") or "",
                        )
                    )
    return CostModelInfo(
        name=name,
        class_name=cls.__name__,
        description=(cls.__doc__ or "").strip().split("\n")[0],
        family=cls.family(),
        mapped_module_types=mapped_types,
        capabilities=capabilities,
        extensions=extensions,
    )


@base_router.get("/", response_model=RegistryResponse)
def get_registry():
    """List all registered cost models with capabilities and extension endpoints."""
    models = [
        _build_model_info(name, cls)
        for name, cls in COST_MODEL_REGISTRY.items()
    ]
    return RegistryResponse(models=models)


@base_router.get("/{model_name}/info", response_model=CostModelInfo)
def get_model_info(model_name: str):
    """Get detailed info for a single cost model."""
    if model_name not in COST_MODEL_REGISTRY:
        raise HTTPException(404, f"Unknown cost model: {model_name}")
    return _build_model_info(model_name, COST_MODEL_REGISTRY[model_name])


@base_router.get("/catalog")
def get_cost_model_catalog():
    """Describe possible RAG-IR dispatch bindings, without compiling a plan."""
    return build_cost_model_catalog()


@contextmanager
def _component_request():
    """Keep adapter failures explicit; this API never synthesizes zero costs."""
    try:
        yield
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc


def _estimate_response(result) -> EstimateResponse:
    if any(not isfinite(value) or value <= 0 for value in (
        result.latency_s, result.throughput_qps,
    )):
        raise HTTPException(422, "Component estimate did not produce finite positive latency and throughput")
    return EstimateResponse(
        latency_s=result.latency_s,
        throughput_qps=result.throughput_qps,
        operators=[
            OperatorBreakdownResponse(
                name=op.name,
                compute_time_s=op.compute_time_s,
                memory_time_s=op.memory_time_s,
                time_s=op.time_s,
                bottleneck=op.bottleneck,
            )
            for op in result.operators
        ],
    )


@base_router.post("/{model_name}/estimate", response_model=EstimateResponse)
def estimate(
    body: EstimateRequest,
    model: CostModelModule = Depends(get_cost_model),
):
    """Estimate one component call; no RAG-IR compilation or serving simulation."""
    with _component_request():
        hardware = parse_hardware(body.hardware)
        return _estimate_response(model.estimate(body.pipeline_config, hardware))


@base_router.post("/{model_name}/estimate-batch")
def estimate_batch(
    body: EstimateBatchRequest,
    model: CostModelModule = Depends(get_cost_model),
):
    """Run estimate_batch() on a cost model."""
    with _component_request():
        hardware = parse_hardware(body.hardware)
        results = model.estimate_batch(body.configs, hardware)
        if len(results) != len(body.configs):
            raise RuntimeError("Component estimate batch returned a different result count")
        return {
            "scope": "component_estimate",
            "results": [_estimate_response(result).model_dump() for result in results],
        }


@base_router.post("/{model_name}/sweep")
def sweep(
    body: SweepRequest,
    model: CostModelModule = Depends(get_cost_model),
):
    """Run generate_sweep_df() and return the DataFrame as JSON records."""
    with _component_request():
        hardware = parse_hardware(body.hardware)
        if body.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        df = model.generate_sweep_df(
            body.pipeline_config, hardware, body.stage, body.max_batch_size,
        )
    return {"scope": "component_sweep", "records": df.to_dict(orient="records")}


# ── Generic calibration endpoints (per cost model) ─────────────────────────

import json as _json
import shlex as _shlex
import sys as _sys
from datetime import datetime as _dt
from typing import Literal as _Literal

from .job_store import get_job, launch_job, load_jobs
from .utils import PROJECT_ROOT as _PROJECT_ROOT, resolve_dir as _resolve_dir

_DEFAULT_CALIB_ROOT = _PROJECT_ROOT / "calibration_profiles"
_DEFAULT_CALIB_DATA_ROOT = _PROJECT_ROOT / "calibration_data"
_DEFAULT_JOBS_ROOT = _PROJECT_ROOT / "tmp_outputs" / "cost_model_jobs"


def _calib_dir(model_name: str) -> Path | None:
    """Return ``calibration_profiles/<subdir>/`` for this cost model, or
    None when the class doesn't expose a profile store.
    """
    cls = COST_MODEL_REGISTRY.get(model_name)
    if cls is None:
        return None
    subdir = cls.profiles_subdir()
    if subdir is None:
        return None
    return calibration_profile_root() / subdir


def _my_module_types(model_name: str) -> set[str]:
    return set(mapped_components(model_name))


def _rel_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_PROJECT_ROOT))
    except Exception:
        return str(path)


def _file_meta(path: Path, root: Path) -> dict:
    stat = path.stat()
    try:
        rel = str(path.resolve().relative_to(root.resolve()))
    except Exception:
        rel = path.name
    return {
        "filename": path.name,
        "relative_path": rel,
        "path": _rel_project(path),
        "size_bytes": stat.st_size,
        "modified_at": _dt.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _json_record_count(path: Path) -> int:
    try:
        data = _json.loads(path.read_text())
    except Exception:
        return 0
    if path.parent.name == "llm_sim":
        try:
            return len(unwrap_llm_sim_profile_document(data, source=str(path)))
        except ValueError:
            return 0
    if not isinstance(data, dict):
        return 1
    if data and "compute_efficiency" not in data and all(isinstance(v, dict) for v in data.values()):
        return len(data)
    return 1


def _split_corpus_embedding(stem: str) -> tuple[str, str]:
    if "__" not in stem:
        return stem, ""
    corpus, embedding = stem.split("__", 1)
    return corpus, embedding


def _list_unified_calibration_profiles() -> list[dict]:
    root = calibration_profile_root()
    if not root.is_dir():
        return []

    profiles: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".json", ".joblib"}:
            continue
        rel_parts = path.relative_to(root).parts
        component = rel_parts[0] if rel_parts else ""
        stem = path.stem
        meta = _file_meta(path, root)

        if component == "faiss_ivf" and len(rel_parts) >= 2 and rel_parts[1] == "imbalance":
            corpus, embedding = _split_corpus_embedding(stem)
            profiles.append({
                **meta,
                "axis": "corpus_embedding",
                "component": "ivf",
                "kind": "ivf_imbalance",
                "key": stem,
                "corpus": corpus,
                "embedding": embedding,
                "system": "",
                "record_count": _json_record_count(path),
            })
            continue

        if component == "faiss_hnsw" and path.suffix == ".joblib":
            corpus, embedding = _split_corpus_embedding(stem)
            profiles.append({
                **meta,
                "axis": "corpus_embedding",
                "component": "hnsw",
                "kind": "hnsw_workload",
                "key": stem,
                "corpus": corpus,
                "embedding": embedding,
                "system": "",
                "record_count": 1,
            })
            continue

        if path.suffix == ".json":
            profiles.append({
                **meta,
                "axis": "system",
                "component": component,
                "kind": "system_profile",
                "key": stem,
                "corpus": "",
                "embedding": "",
                "system": stem,
                "record_count": _json_record_count(path),
            })
    return profiles


def _list_calibration_data_files() -> list[dict]:
    root = calibration_data_root()
    if not root.is_dir():
        return []
    files: list[dict] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        component = rel_parts[0] if rel_parts else ""
        files.append({
            **_file_meta(path, root),
            "component": component,
            "system": path.stem,
            "format": path.suffix.lstrip(".") or "file",
        })
    return files


@base_router.get("/{model_name}/modules")
def list_cost_model_modules(model_name: str):
    """Return the components this cost model handles (for UI dropdowns)."""
    if model_name not in COST_MODEL_REGISTRY:
        raise HTTPException(404, f"Unknown cost model: {model_name}")
    return {"modules": sorted(_my_module_types(model_name))}


@base_router.get("/{model_name}/profiles")
def list_cost_model_profiles(model_name: str):
    """List calibration profiles owned by this cost model.

    Profiles are filtered by ``component`` to show only those that belong
    to this cost model's mapped modules. Returns 501 when the cost model
    declares no on-disk profile store.
    """
    cls = COST_MODEL_REGISTRY.get(model_name)
    if cls is None:
        raise HTTPException(404, f"Unknown cost model: {model_name}")
    if "calibration_profiles" not in cls.capabilities():
        raise HTTPException(501, f"{model_name} does not support calibration profiles")

    pdir = _calib_dir(model_name)
    if pdir is None or not pdir.is_dir():
        return {"profiles": [], "dir": str(pdir) if pdir else None}

    # Filter logic: if the profile JSON declares a ``component``, it must
    # match one of this cost model's mapped modules (used by the LLM-sim
    # family that shares ``calibration_profiles/llm_sim/`` across multiple
    # cost models). If there's no ``component`` field, we assume the
    # subdir is cost-model-exclusive (FAISS style) and include it.
    my_modules = _my_module_types(model_name)
    profiles = []
    for p in sorted(pdir.glob("*.json")):
        try:
            with open(p) as f:
                data = _json.load(f)
            if pdir.name == "llm_sim":
                data = unwrap_llm_sim_profile_document(data, source=str(p))
        except Exception as e:
            profiles.append({"filename": p.name, "error": str(e)})
            continue
        # Per-system files (llm_sim: ``<hw>.json``) are a {record_key: record}
        # map holding many component profiles; single-record files (FAISS:
        # ``<preset>.json``) carry the record fields at top level. Flatten the
        # former so each component renders as its own row, and apply the
        # component filter per RECORD (the llm_sim dir is shared across cost
        # models, so a system file may mix modules owned by different ones).
        if pdir.name == "llm_sim":
            records = list(data.items())
        elif (isinstance(data, dict) and data
                and "compute_efficiency" not in data
                and all(isinstance(v, dict) for v in data.values())):
            records = list(data.items())
        else:
            records = [(None, data)]
        for rec_key, rec in records:
            if not isinstance(rec, dict):
                continue
            shared_candidate = pdir.name == "llm_sim" and rec.get("component") == "agnostic"
            if "component" in rec and rec["component"] not in my_modules and not shared_candidate:
                continue
            out = dict(rec)
            out["filename"] = p.name
            out["match_scope"] = "shared_model_candidate" if shared_candidate else "component_candidate"
            out["can_delete"] = rec_key is None and pdir.name != "llm_sim"
            if not out["can_delete"]:
                out["delete_reason"] = "Shared calibration files contain other component records; per-record deletion is not supported"
            if rec_key is not None:
                out["record_key"] = rec_key
            profiles.append(out)
    return {"profiles": profiles, "dir": str(pdir)}


@base_router.get("/{model_name}/profiles/{filename}")
def get_cost_model_profile(model_name: str, filename: str):
    pdir = _calib_dir(model_name)
    if pdir is None:
        raise HTTPException(501, f"{model_name} does not support calibration profiles")
    path = pdir / filename
    if not path.is_file() or path.suffix != ".json":
        raise HTTPException(404, f"Profile not found: {filename}")
    if path.resolve().parent != pdir.resolve():
        raise HTTPException(400, "Invalid filename")
    with open(path) as f:
        data = _json.load(f)
    if pdir.name == "llm_sim":
        try:
            unwrap_llm_sim_profile_document(data, source=str(path))
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return data
    data["filename"] = filename
    return data


@base_router.delete("/{model_name}/profiles/{filename}")
def delete_cost_model_profile(model_name: str, filename: str):
    pdir = _calib_dir(model_name)
    if pdir is None:
        raise HTTPException(501, f"{model_name} does not support calibration profiles")
    path = pdir / filename
    if not path.is_file() or path.suffix != ".json":
        raise HTTPException(404, f"Profile not found: {filename}")
    if path.resolve().parent != pdir.resolve():
        raise HTTPException(400, "Invalid filename")
    try:
        document = _json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise HTTPException(409, f"Cannot validate profile before deletion: {exc}") from exc
    if pdir.name == "llm_sim" or (
        isinstance(document, dict) and document
        and "compute_efficiency" not in document
        and all(isinstance(value, dict) for value in document.values())
    ):
        raise HTTPException(409, "This shared calibration file contains multiple component records; deleting it from a component row is not supported")
    path.unlink()
    return {"deleted": filename}


class CalibrationRunRequest(BaseModel):
    kind: _Literal["benchmark", "calibrate"]
    module: str
    config_path: str
    batch_sizes: list[int] = [4, 8, 16]
    seq_lens: list[int] = [128, 256, 512]
    warmup: int = 5
    measurements: int = 20


class UnifiedCalibrationRunRequest(BaseModel):
    command: str = ""
    # Legacy fields kept so older clients do not break; new UI uses command.
    target: str = "ivf"
    mode: _Literal["auto", "measure", "fit", "all", "workload"] = "auto"
    config_path: str = ""
    force: bool = False
    maxiter: int | None = None
    qa_data: str = ""
    corpus_data: str = ""
    project_dir: str = ""
    data_dir: str = ""
    profile_dir: str = ""
    extra_args: list[str] = Field(default_factory=list)


@base_router.get("/calibration/status")
def get_unified_calibration_status():
    """Return the unified calibration profile and frozen measurement state."""
    profiles = _list_unified_calibration_profiles()
    data_files = _list_calibration_data_files()
    return {
        "scope": "artifact_inventory",
        "commands": calibration_commands(),
        "profiles_root": str(calibration_profile_root()),
        "data_root": str(calibration_data_root()),
        "configured_profiles_root": load_settings().calibration_profiles_dir or None,
        "root_note": "Inventory uses the same calibration_paths roots as CM runtime. The legacy Apps calibration_profiles_dir setting does not select runtime evidence.",
        "profiles": profiles,
        "data_files": data_files,
        "summary": {
            "profiles": len(profiles),
            "system_profiles": sum(1 for p in profiles if p.get("axis") == "system"),
            "corpus_embedding_profiles": sum(
                1 for p in profiles if p.get("axis") == "corpus_embedding"
            ),
            "data_files": len(data_files),
        },
    }


@base_router.get("/calibration/profile")
def get_unified_calibration_profile(path: str):
    """Return detail for one unified calibration profile.

    ``path`` is the ``relative_path`` value returned by
    ``/api/cost-model/calibration/status``. JSON profiles include parsed
    content; binary artifacts (for example HNSW workload ``.joblib`` files)
    return metadata only.
    """
    if not path:
        raise HTTPException(400, "Profile path is required")

    profile = next(
        (p for p in _list_unified_calibration_profiles() if p.get("relative_path") == path),
        None,
    )
    if profile is None:
        raise HTTPException(404, f"Profile not found: {path}")

    root = calibration_profile_root()
    target = (root / path).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise HTTPException(400, "Invalid profile path")
    if not target.is_file():
        raise HTTPException(404, f"Profile not found: {path}")

    if target.suffix == ".json":
        try:
            content = _json.loads(target.read_text())
            return {
                "profile": profile,
                "content_type": "json",
                "content": content,
            }
        except Exception as exc:
            return {
                "profile": profile,
                "content_type": "json",
                "content": None,
                "error": str(exc),
            }

    return {
        "profile": profile,
        "content_type": "binary",
        "content": None,
    }


def _resolve_existing_file(path_value: str, label: str) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.is_file():
        raise HTTPException(404, f"{label} not found: {path_value}")
    return path


def _build_unified_calibration_args(body: UnifiedCalibrationRunRequest) -> list[str]:
    if body.command.strip():
        try:
            args = _shlex.split(body.command.strip())
        except ValueError as exc:
            raise HTTPException(400, f"Invalid calibration command: {exc}")
    else:
        target = body.target.strip()
        if not target:
            raise HTTPException(400, "Calibration target is required")
        if body.mode == "auto":
            args = [target]
        elif body.mode in ("measure", "fit"):
            args = [body.mode, target]
        elif body.mode in ("all", "workload"):
            args = [body.mode]
        else:
            raise HTTPException(400, f"Unsupported calibration mode: {body.mode}")

    if not args:
        raise HTTPException(400, "Calibration command is required")

    from rag_stack.calibration_cli import COMPONENTS, META_COMMANDS, PASSTHROUGH, VERBS

    head = args[0]
    if head not in {*COMPONENTS, *META_COMMANDS, *PASSTHROUGH, *VERBS}:
        raise HTTPException(400, f"Unsupported calibration command: {head}")
    if head in {"measure", "fit"}:
        if len(args) < 2 or args[1] not in COMPONENTS:
            raise HTTPException(400, f"`calibrate {head}` requires ivf or hnsw")
    if head == "generator-disagg" and (len(args) < 2 or args[1] not in {"measure", "fit"}):
        raise HTTPException(400, "generator-disagg requires measure or fit")
    if head == "generator-tp" and len(args) > 1 and not args[1].startswith("-") and args[1] not in {"measure", "fit"}:
        raise HTTPException(400, "generator-tp supports measure or fit")

    args.extend(body.extra_args)
    options = command_options(args)
    accepts_config = options["needs_config"]
    accepts_force = options["supports_force"]
    accepts_maxiter = options["supports_maxiter"]
    accepts_data_paths = options["supports_data_paths"]
    supplied = {
        "config_path": (bool(body.config_path), accepts_config),
        "force": (body.force, accepts_force),
        "maxiter": (body.maxiter is not None, accepts_maxiter),
        "qa_data": (bool(body.qa_data), accepts_data_paths),
        "corpus_data": (bool(body.corpus_data), accepts_data_paths),
        "project_dir": (bool(body.project_dir), accepts_data_paths),
    }
    for field, (present, accepted) in supplied.items():
        if present and not accepted:
            raise HTTPException(400, f"{field} is not supported by calibrate {' '.join(args[:2])}")
    if body.maxiter is not None and body.maxiter < 1:
        raise HTTPException(400, "maxiter must be positive")

    def has_option(flag: str) -> bool:
        return any(arg == flag or arg.startswith(flag + "=") for arg in args)

    def append_option(flag: str, value: str) -> None:
        if has_option(flag):
            raise HTTPException(400, f"{flag} was supplied both in the command and as a separate field")
        args.extend([flag, value])

    def directory(value: str) -> str:
        path = Path(value).expanduser()
        return str((path if path.is_absolute() else _PROJECT_ROOT / path).resolve())

    for flag, value, label in (
        ("--config", body.config_path, "Config"),
        ("--qa-data", body.qa_data, "QA data"),
        ("--corpus-data", body.corpus_data, "Corpus data"),
    ):
        if value:
            append_option(flag, str(_resolve_existing_file(value, label).resolve()))
    if body.project_dir:
        append_option("--project-dir", directory(body.project_dir))
    if accepts_config and not has_option("--config"):
        raise HTTPException(400, f"calibrate {head} requires config_path or --config")
    if head == "workload" and not has_option("--project-dir"):
        raise HTTPException(400, "calibrate workload requires project_dir or --project-dir")
    if body.force:
        if has_option("--force"):
            raise HTTPException(400, "--force was supplied twice")
        args.append("--force")
    if body.maxiter is not None:
        append_option("--maxiter", str(body.maxiter))
    for flag, value, default in (
        ("--data-dir", body.data_dir, calibration_data_root()),
        ("--profile-dir", body.profile_dir, calibration_profile_root()),
    ):
        if value:
            append_option(flag, directory(value))
        elif not has_option(flag):
            args.extend([flag, str(default)])

    # The command/extra-args form must obey the same field contract as the
    # structured controls; otherwise a hidden stale flag could be ignored by
    # the CLI even though the Apps control was correctly disabled.
    accepted_flags = {
        "--config": accepts_config, "--force": accepts_force,
        "--maxiter": accepts_maxiter, "--qa-data": accepts_data_paths,
        "--corpus-data": accepts_data_paths, "--project-dir": accepts_data_paths,
        "--data-dir": True, "--profile-dir": True,
    }
    for flag, accepted in accepted_flags.items():
        positions = [index for index, arg in enumerate(args) if arg == flag or arg.startswith(flag + "=")]
        if positions and not accepted:
            raise HTTPException(400, f"{flag} is not supported by calibrate {head}")
        if len(positions) > 1:
            raise HTTPException(400, f"{flag} was supplied more than once")
        for index in positions:
            if flag == "--force":
                if args[index] != flag:
                    raise HTTPException(400, "--force does not accept a value")
                continue
            value = args[index].split("=", 1)[1] if "=" in args[index] else (args[index + 1] if index + 1 < len(args) else "")
            if not value or value.startswith("--"):
                raise HTTPException(400, f"{flag} requires a value")
            if flag == "--maxiter":
                try:
                    positive = int(value) > 0
                except ValueError:
                    positive = False
                if not positive:
                    raise HTTPException(400, "maxiter must be a positive integer")

    for flag in options["required_extra_args"]:
        if not has_option(flag):
            raise HTTPException(400, f"calibrate {head} requires {flag} in extra_args")

    return args


@base_router.post("/calibration/run")
def run_unified_calibration_job(body: UnifiedCalibrationRunRequest):
    """Launch ``python -m rag_stack calibrate ...`` and stream output via job log."""
    cli_args = _build_unified_calibration_args(body)
    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    label = "__".join(part.replace("/", "_") for part in cli_args[:2]) or "calibrate"
    job_dir = _DEFAULT_JOBS_ROOT / f"{ts}__calibration__{label}"
    cmd = [_sys.executable, "-m", "rag_stack", "calibrate", *cli_args]
    return launch_job(
        cmd=cmd,
        config={
            "kind": "calibrate",
            "calibrate_command": body.command,
            "target": body.target,
            "mode": body.mode,
            "config_path": body.config_path,
            "force": body.force,
            "maxiter": body.maxiter,
            "qa_data": body.qa_data,
            "corpus_data": body.corpus_data,
            "project_dir": body.project_dir,
            "data_dir": body.data_dir,
            "profile_dir": body.profile_dir,
            "extra_args": body.extra_args,
            "cli_args": cli_args,
            "command": " ".join(cmd),
        },
        output_dir=str(job_dir),
        job_type="cost_model_calibrate",
        log_name="_job.log",
    )


@base_router.get("/calibration/jobs")
def list_unified_calibration_jobs():
    jobs = [
        j for j in load_jobs()
        if j.get("job_type", "").startswith("cost_model_")
    ]
    jobs.sort(key=lambda j: j.get("started_at", ""), reverse=True)
    return {"jobs": jobs}


@base_router.get("/calibration/jobs/{job_id}/log")
def get_unified_calibration_job_log(job_id: str, tail: int = 20000):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    log_path = Path(job.get("log_path") or "")
    if not log_path.is_file():
        fallback = Path(job.get("output_dir", "")) / "_job.log"
        log_path = fallback if fallback.is_file() else log_path
    if not log_path.is_file():
        return {"job": job, "log": "", "log_path": ""}
    data = log_path.read_text(errors="replace")
    if tail > 0 and len(data) > tail:
        data = data[-tail:]
    return {"job": job, "log": data, "log_path": _rel_project(log_path)}


@base_router.post("/{model_name}/run")
def run_cost_model_job(model_name: str, body: CalibrationRunRequest):
    """Launch a benchmark or calibration subprocess for this cost model.

    Returns 501 when the cost model doesn't declare ``calibration_runs``.
    Shells out to a CLI subprocess so torch / CUDA state stays isolated from
    the web server process. Calibration runs use the unified
    ``python -m rag_stack calibrate`` entrypoint.
    """
    cls = COST_MODEL_REGISTRY.get(model_name)
    if cls is None:
        raise HTTPException(404, f"Unknown cost model: {model_name}")
    if "calibration_runs" not in cls.capabilities():
        raise HTTPException(501, f"{model_name} does not support calibration runs")

    config_path = Path(body.config_path)
    if not config_path.is_absolute():
        config_path = _PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise HTTPException(404, f"Config not found: {body.config_path}")

    from rag_stack.cost_model.llm_sim.genz_adapter import GenZLLMSimBase
    from rag_stack.taxonomy import TAXONOMY, SHAPE_LLM

    if not issubclass(cls, GenZLLMSimBase) or cls._get_benchmark_runner is GenZLLMSimBase._get_benchmark_runner:
        raise HTTPException(501, f"{model_name} has no generic benchmark runner; use the dedicated unified calibration command")
    component = TAXONOMY.components.get(body.module)
    if body.kind == "calibrate" and component is not None and component.shape == SHAPE_LLM:
        raise HTTPException(501, "LLM serving calibration requires the dedicated generator telemetry calibrator")
    if model_name == "retrieval_encode_sim":
        from rag_stack.cost_model.llm_sim.genz_model_map import GENZ_MODEL_MAP
        valid_module = body.module.lower() in GENZ_MODEL_MAP
    else:
        valid_module = body.module in _my_module_types(model_name)
    if not valid_module:
        raise HTTPException(
            400, f"Module '{body.module}' is not mapped to cost model '{model_name}'",
        )
    if model_name == "passage_reranker_sim":
        from rag_stack.cost_model.llm_sim.calibration.reranker import RERANKER_RUNNERS
        if body.module not in RERANKER_RUNNERS:
            raise HTTPException(501, f"No calibration runner exists for {body.module}")
    if (
        not body.batch_sizes or not body.seq_lens
        or any(value < 1 for value in [*body.batch_sizes, *body.seq_lens])
        or body.warmup < 0 or body.measurements < 1
    ):
        raise HTTPException(422, "Calibration needs positive batch sizes, sequence lengths and measurements, and non-negative warmup")

    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
    job_dir = _DEFAULT_JOBS_ROOT / f"{ts}__{model_name}__{body.kind}__{body.module}"

    cmd = [
        _sys.executable, "-m", "rag_stack", "calibrate", "llm", body.kind,
        "--cost-model", model_name, "--module", body.module,
        "--config", str(config_path.resolve()),
        "--batch-sizes", ",".join(str(x) for x in body.batch_sizes),
        "--seq-lens", ",".join(str(x) for x in body.seq_lens),
        "--warmup", str(body.warmup), "--measurements", str(body.measurements),
        "--data-dir", str(calibration_data_root()),
        "--profile-dir", str(calibration_profile_root()),
    ]
    return launch_job(
        cmd=cmd,
        config={"cost_model": model_name, **body.model_dump()},
        output_dir=str(job_dir),
        job_type=f"cost_model_{body.kind}",
        log_name="_job.log",
    )


@base_router.get("/{model_name}/jobs")
def list_cost_model_jobs(model_name: str):
    """List benchmark/calibration jobs for this cost model (newest first)."""
    jobs = [
        j for j in load_jobs()
        if j.get("job_type", "").startswith("cost_model_")
        and j.get("config", {}).get("cost_model") == model_name
    ]
    jobs.sort(key=lambda j: j.get("started_at", ""), reverse=True)
    return {"jobs": jobs}


# ── GenZ presets (GPU + LLM models) ────────────────────────────────────────

genz_router = APIRouter(tags=["genz"])


@genz_router.get("/gpu-presets")
def get_gpu_presets():
    """List available GenZ GPU/accelerator presets."""
    try:
        from GenZ.LLM_inference.utils import system_configs
        return {"presets": sorted(system_configs.keys())}
    except ImportError:
        return {"presets": []}


@genz_router.get("/llm-models")
def get_llm_models():
    """List available GenZ LLM model presets."""
    try:
        from GenZ.Models.default_models import MODEL_DICT
        return {"models": sorted(MODEL_DICT.list_models())}
    except ImportError:
        return {"models": []}


# ── Global settings endpoints ───────────────────────────────────────────────

settings_router = APIRouter(tags=["settings"])


@settings_router.get("/", response_model=AppSettings)
def get_settings():
    """Return persisted application settings."""
    return load_settings()


@settings_router.put("/", response_model=AppSettings)
def put_settings(body: AppSettings):
    """Update and persist application settings."""
    save_settings(body)
    return body


@settings_router.get("/env-keys")
def get_env_keys():
    """Return env keys from .env file (values masked for security)."""
    keys = load_env_keys()
    return {"keys": {k: v for k, v in keys.items()}}


@settings_router.put("/env-keys")
def put_env_keys(body: dict):
    """Write env keys to .env file and apply to process."""
    keys: dict[str, str] = body.get("keys", {})
    save_env_keys(keys)
    return {"keys": keys}


# ── Global job management ───────────────────────────────────────────────────

from .global_jobs import get_global_job_process_detail, get_global_jobs_overview
from .job_store import kill_job

jobs_router = APIRouter(tags=["jobs"])


@jobs_router.get("/overview")
def global_jobs_overview():
    """List running RAG-Stack main processes for the global jobs center."""
    return get_global_jobs_overview()


@jobs_router.get("/process/{pid}")
def global_job_process_detail(pid: int, max_processes: int = 160):
    """Return a low-overhead process tree snapshot for one job process."""
    return get_global_job_process_detail(pid, max_processes=max_processes)


@jobs_router.post("/{job_id}/kill")
def kill_job_endpoint(job_id: str):
    """Kill a running job by ID (works for all job types)."""
    job = kill_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    return job


# ── Static data serving (replaces Metro middleware) ─────────────────────────

from .utils import PROJECT_ROOT, resolve_dir

_DEFAULT_DATA_DIR = PROJECT_ROOT / "tmp_outputs" / "rag_stack_projects"

_MIME = {
    ".json": "application/json",
    ".csv": "text/csv",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".txt": "text/plain",
    ".log": "text/plain",
}

data_router = APIRouter(tags=["data"])


def _get_data_dir() -> Path:
    return resolve_dir(load_settings().rag_optimization_project_dir, _DEFAULT_DATA_DIR)


@data_router.get("/")
@data_router.get("/{file_path:path}")
def serve_data(file_path: str = ""):
    """Serve static data files — directory listings or file content."""
    data_dir = _get_data_dir()
    target = (data_dir / file_path).resolve() if file_path else data_dir.resolve()

    # Prevent path traversal
    if not str(target).startswith(str(data_dir.resolve())):
        raise HTTPException(403, "Forbidden")

    if target.is_dir():
        entries = sorted(e.name for e in target.iterdir())
        return JSONResponse(entries)

    if target.is_file():
        media_type = _MIME.get(target.suffix.lower(), "application/octet-stream")
        return FileResponse(target, media_type=media_type)

    raise HTTPException(404, "Not found")


# ── Token auth ─────────────────────────────────────────────────────────────

# Paths that are accessible without a token
_AUTH_EXEMPT_PATHS = {"/api/auth/verify", "/api/auth/status", "/docs", "/openapi.json", "/redoc"}


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Bearer token auth. Disabled when RAG_STACK_API_TOKEN is not set."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        token = os.environ.get("RAG_STACK_API_TOKEN")
        if not token:
            return await call_next(request)
        if request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)
        if request.method == "OPTIONS":
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {token}":
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing token"})
        return await call_next(request)


class VerifyTokenRequest(BaseModel):
    token: str


auth_router = APIRouter(tags=["auth"])


@auth_router.post("/verify")
def verify_token(body: VerifyTokenRequest):
    """Validate a token. Returns server info on success."""
    expected = os.environ.get("RAG_STACK_API_TOKEN", "")
    if expected and body.token != expected:
        raise HTTPException(401, "Invalid token")
    return {
        "valid": True,
        "auth_required": bool(expected),
        "server_version": "0.1.0",
    }


@auth_router.get("/status")
def auth_status():
    """Check if auth is required (no token needed)."""
    return {"auth_required": bool(os.environ.get("RAG_STACK_API_TOKEN"))}


# ── App factory ─────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    application = FastAPI(
        title="RAG-Stack Cost Model API",
        version="0.1.0",
    )

    application.add_middleware(TokenAuthMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Chrome Private Network Access: allow localhost/127.0.0.1 cross-origin
    @application.middleware("http")
    async def private_network_access(request: Request, call_next):
        if request.method == "OPTIONS" and request.headers.get("access-control-request-private-network"):
            from starlette.responses import Response as StarletteResponse
            return StarletteResponse(
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": request.headers.get("origin", "*"),
                    "Access-Control-Allow-Private-Network": "true",
                    "Access-Control-Allow-Methods": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

    application.include_router(auth_router, prefix="/api/auth")
    application.include_router(base_router, prefix="/api/cost-model")
    application.include_router(settings_router, prefix="/api/settings")
    application.include_router(jobs_router, prefix="/api/jobs")
    application.include_router(data_router, prefix="/api/data")
    application.include_router(genz_router, prefix="/api/genz")

    from .routers.optimizer import router as optimizer_router
    application.include_router(optimizer_router, prefix="/api/optimizer")

    from .routers.rag_project import router as rag_project_router
    application.include_router(rag_project_router, prefix="/api/rag-project")

    from .routers.quality_store import router as quality_store_router
    application.include_router(quality_store_router, prefix="/api/quality-store")

    from .routers.dataset import router as dataset_router
    application.include_router(dataset_router, prefix="/api/dataset")

    for name, ext_router in EXTENSION_ROUTERS.items():
        application.include_router(
            ext_router,
            prefix=f"/api/cost-model/{name}",
        )

    # Load .env keys into os.environ on startup
    apply_env_keys()

    return application


app = create_app()
