"""RAG optimization project management — config listing, creation, dataset browsing, and run.

RAG project jobs are tracked via summary.json (pid + status) inside each project folder,
NOT via the global job_store. This means CLI-launched jobs are also visible to the UI.
"""

from __future__ import annotations

import importlib
import os
import json
import logging
import subprocess
import shutil
import sys
import threading
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..settings_store import load_settings
from ..job_store import _is_pid_alive
from ..utils import PROJECT_ROOT, resolve_dir

from rag_stack.logging_utils import (
    PROJECT_LOG_FILE_ENV,
    PROJECT_LOG_OWNED_BY_PARENT_ENV,
    PROJECT_LOG_STDIO_REDIRECTED_ENV,
)
from rag_stack.project_summary import (
    CONFIG_FILENAME, ProjectStatus,
    create_project_summary, save_project_summary, load_project_summary,
)
from rag_stack.project_config import (
    ProjectConfigError, project_config_lock, project_config_read_only_reason,
    write_project_config, verify_project_config, freeze_project_config,
)
from rag_stack.system_transfer_backend import project_to_vllm_chat
from rag_stack.system_transfer_contract import validate_system_transfer_artifact

router = APIRouter(tags=["rag-project"])
logger = logging.getLogger("RAG-Stack")

_DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs" / "rag_stack"
_DEFAULT_PROJECT_DIR = PROJECT_ROOT / "tmp_outputs" / "rag_stack_projects"
_DEFAULT_QUALITY_STORE_DIR = PROJECT_ROOT / "quality_stores"
_DEFAULT_SYSTEM_YAML_ROOT = PROJECT_ROOT / "configs" / "system_transfer"
_DEFAULT_SYSTEM_TRANSFER_REFINE_ITERATIONS = 5
_SYSTEM_TRANSFER_METADATA_KEY = "system_transfer"
_SYSTEM_TRANSFER_PROVENANCE_FILENAME = "system_transfer.json"
_SYSTEM_TRANSFER_GENERATOR_BACKENDS = frozenset({
    "inherit", "vllm", "vllm_api",
})
_SYSTEM_TRANSFER_RAG_OVERRIDE_SECTIONS = frozenset({
    "optimizer", "eval_backend_setting",
})
_LEGACY_SYSTEM_TRANSFER_CM_WORKER_ENV = {
    "CM_SWEEP_WORKERS": "8",
    "CM_DES_WORKERS": "8",
}
_RAG_CONFIG_SECTIONS = (
    "global",
    "dataset",
    "system",
    "optimizer",
    "eval_backend_setting",
    "algo_search_space",
)


def _validate_optimizer_section(value: object, *, label: str) -> None:
    if not isinstance(value, dict):
        raise HTTPException(400, f"{label} must be a mapping")
    unsupported = sorted(set(value) - {"type", "params"}, key=str)
    if unsupported:
        raise HTTPException(
            400, f"{label} has unsupported fields: {unsupported}",
        )
    optimizer_type = value.get("type")
    if not isinstance(optimizer_type, str) or not optimizer_type.strip():
        raise HTTPException(
            400, f"{label}.type must be a non-empty string",
        )
    if "params" in value and not isinstance(value["params"], dict):
        raise HTTPException(400, f"{label}.params must be a mapping")


def _get_configs_dir() -> Path:
    return resolve_dir(load_settings().rag_optimization_configs_dir, _DEFAULT_CONFIGS_DIR)


def _get_project_dir() -> Path:
    return resolve_dir(load_settings().rag_optimization_project_dir, _DEFAULT_PROJECT_DIR)


def _get_quality_store_dir() -> Path:
    return resolve_dir(
        getattr(load_settings(), "quality_stores_dir", ""),
        _DEFAULT_QUALITY_STORE_DIR,
    )


def _get_system_yaml_root() -> Path:
    return _DEFAULT_SYSTEM_YAML_ROOT


def _validate_child_name(name: str, label: str) -> None:
    if not name or "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, f"Invalid {label} name")


# ── Import-race guard for the static_rag_evaluator package ───────────────────
# That package eagerly loads the heavy evaluator at import time, whose internal
# graph (schema -> support -> node_line, and base -> schema) is import-order
# fragile. Under FastAPI's threadpool, the editor fires several registry calls
# in parallel, so different threads trigger the FIRST import via different entry
# points (support / nodes.generator / schema) concurrently and race into a
# "partially initialized module" ImportError. Resolving the import ONCE behind a
# lock (single-threaded) makes every later import a cache hit. Cheap O(1) after
# the first call (double-checked flag), no server-boot cost.
_SRE_LOCK = threading.Lock()
_SRE_WARMED = False


def _warm_static_rag_evaluator() -> None:
    global _SRE_WARMED
    if _SRE_WARMED:
        return
    with _SRE_LOCK:
        if _SRE_WARMED:
            return
        importlib.import_module("rag_stack_evaluator.static_rag_evaluator.support")
        importlib.import_module(
            "rag_stack_evaluator.static_rag_evaluator.nodes.generator.registry",
        )
        _SRE_WARMED = True


# ── Config listing ──────────────────────────────────────────────────────────

@router.get("/configs")
def list_configs():
    """List available RAG optimization YAML configs."""
    configs_dir = _get_configs_dir()
    if not configs_dir.is_dir():
        return {"configs": []}

    configs = []
    for p in sorted(configs_dir.glob("*.yaml")):
        try:
            raw = p.read_text()
            data = yaml.safe_load(raw)
            # 6-section schema: n_iterations + eval_backend live under `global`.
            # Fall back to the old top-level key so pre-migration configs still
            # show a sensible summary (those are rejected at run time anyway).
            glob = data.get("global") or {}
            configs.append({
                "filename": p.name,
                "optimizer_type": (data.get("optimizer") or {}).get("type", ""),
                "n_iterations": glob.get("n_iterations", data.get("n_iterations", 0)),
                "eval_backend": glob.get("eval_backend", "static_gt"),
                "content": data,
                "raw_content": raw,
            })
        except Exception:
            continue

    return {"configs": configs}


# ── Project creation ────────────────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    project_name: str
    config_filename: str


class DuplicateConfigRequest(BaseModel):
    new_filename: str


class SaveTemplateConfigRequest(BaseModel):
    content: str


@router.post("/create")
def create_project(body: CreateProjectRequest):
    """Create a new project folder with summary.json (status: created) and config."""
    project_root = _get_project_dir()
    project_path = project_root / body.project_name

    # Validate project name
    _validate_child_name(body.project_name, "project")
    if project_path.resolve().parent != project_root.resolve():
        raise HTTPException(400, "Invalid project name")

    # Resolve config
    configs_dir = _get_configs_dir()
    config_src = configs_dir / body.config_filename
    if not config_src.is_file():
        raise HTTPException(404, f"Config not found: {body.config_filename}")
    if not config_src.resolve().parent == configs_dir.resolve():
        raise HTTPException(400, "Invalid config filename")

    # Create project folder
    project_path.mkdir(parents=True, exist_ok=True)

    with _project_mutation_lock(project_path):
        # Re-creating an existing draft must not bypass the same freeze used
        # by the editor, or race a first launch in another API worker.
        if (project_path / CONFIG_FILENAME).exists() or (project_path / "summary.json").exists():
            reason = _project_config_read_only_reason(project_path)
            if reason is not None:
                raise HTTPException(409, reason)
        write_project_config(project_path, config_src.read_text())
        summary = create_project_summary(ProjectStatus.CREATED)
        save_project_summary(str(project_path), summary)

    return {"ok": True, "project_name": body.project_name, "project_dir": str(project_path)}


# ── Save project config ─────────────────────────────────────────────────────

class SaveProjectConfigRequest(BaseModel):
    project_name: str
    content: str  # raw YAML text


def _existing_project_path(project_name: str) -> Path:
    _validate_child_name(project_name, "project")
    root = _get_project_dir().resolve()
    path = (root / project_name).resolve()
    if path.parent != root:
        raise HTTPException(400, "Invalid project name")
    if not path.is_dir():
        raise HTTPException(404, f"Project not found: {project_name}")
    return path


@contextmanager
def _project_mutation_lock(project_path: Path):
    """Expose core lifecycle conflicts as HTTP 409 across all mutation routes."""
    try:
        with project_config_lock(project_path):
            yield
    except ProjectConfigError as exc:
        raise HTTPException(409, str(exc)) from exc


def _project_config_read_only_reason(project_path: Path) -> str | None:
    reason = project_config_read_only_reason(project_path)
    if reason is not None:
        return reason
    if load_project_summary(str(project_path)) is None:
        return "Project lifecycle state is unavailable; configuration is read-only."
    return None


def _project_config_document(project_name: str, project_path: Path) -> dict:
    config_path = project_path / CONFIG_FILENAME
    if not config_path.is_file():
        raise HTTPException(404, f"{CONFIG_FILENAME} not found")
    raw = verify_project_config(project_path)
    try:
        content = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise HTTPException(400, f"Invalid project YAML: {exc}") from exc
    if not isinstance(content, dict):
        raise HTTPException(400, "Project config must be a mapping/object.")
    reason = _project_config_read_only_reason(project_path)
    return {
        "project_name": project_name,
        "content": content,
        "raw_content": raw,
        "editable": reason is None,
        "read_only_reason": reason,
    }


@router.get("/config/{project_name}")
def get_project_config(project_name: str):
    """Read the exact project draft and its authoritative editability state."""
    project_path = _existing_project_path(project_name)
    with _project_mutation_lock(project_path):
        return _project_config_document(project_name, project_path)


@router.put("/config")
def save_project_config(body: SaveProjectConfigRequest):
    """Validate and atomically save a draft before the project's first run."""
    project_path = _existing_project_path(body.project_name)
    with _project_mutation_lock(project_path):
        reason = _project_config_read_only_reason(project_path)
        if reason is not None:
            raise HTTPException(409, reason)
    # Registry/schema validation can be slow on its first import. Do it outside
    # the lock, then recheck lifecycle state when committing the validated draft.
    validation = validate_config(ValidateConfigRequest(content=body.content))
    if not validation["valid"]:
        raise HTTPException(400, "Invalid configuration: " + "; ".join(validation["errors"]))
    with _project_mutation_lock(project_path):
        reason = _project_config_read_only_reason(project_path)
        if reason is not None:
            raise HTTPException(409, reason)
        config_path = project_path / CONFIG_FILENAME
        if not config_path.is_file():
            raise HTTPException(404, f"{CONFIG_FILENAME} not found")
        write_project_config(project_path, body.content)
        return {"ok": True, **_project_config_document(body.project_name, project_path)}


# ── Save analysis settings (Analysis Settings panel) ────────────────────────
# Targeted text-edit on the YAML file so EVERY line outside the two managed
# blocks is byte-identical after save:
#   system.performance_objectives — drives both runtime hw selection AND
#                                   the UI Pareto recompute.
#   analysis.quality              — UI-only state (quality metric / formula
#                                   / per-sub-metric weights). Runner ignores
#                                   this block.
# A full ruamel round-trip would re-flow long descriptions and rewrite
# `null` → empty even when we don't touch those lines, so users would see
# spurious diff on every save. The line-walking helper below keeps the diff
# scoped to the two blocks we actually care about.

class SaveAnalysisSettingsRequest(BaseModel):
    project_name: str
    # Each block is optional — caller sends only what changed. Passing the
    # field with value None means "leave the block alone"; omit it entirely
    # for the same effect.
    performance_objectives: dict | None = None
    analysis_quality: dict | None = None


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_key_line(line: str, key: str, indent: int) -> bool:
    """True if `line` is `<indent spaces><key>:` (with optional inline value)."""
    if _line_indent(line) != indent:
        return False
    rest = line[indent:].rstrip("\n").rstrip()
    if not rest.startswith(f"{key}:"):
        return False
    # Exact key match — guards against e.g. `performance_objectives_extra:`
    after = rest[len(key) + 1 :]
    return after == "" or after.startswith(" ") or after.startswith("\t")


def _find_key_line(
    lines: list[str], key: str, indent: int, start: int, end: int
) -> int:
    """Index in [start, end) of the line whose key matches at `indent`, else -1."""
    for i in range(start, end):
        if _is_key_line(lines[i], key, indent):
            return i
    return -1


def _block_end(lines: list[str], key_line: int, indent: int, hard_end: int) -> int:
    """Return exclusive end of the block headed at `lines[key_line]`.

    Block extends from `key_line + 1` until the first non-blank line at
    indent <= `indent` (sibling or parent boundary) or `hard_end`.
    """
    for j in range(key_line + 1, hard_end):
        s = lines[j]
        # Blank lines and pure-comment lines belong to whichever block they
        # are textually closer to; we treat them as part of the current
        # block while scanning. This means a trailing comment immediately
        # after the block stays with the block on edits.
        stripped = s.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if _line_indent(s) <= indent:
            return j
    return hard_end


def _serialize_block_body(value: dict, child_indent: int) -> str:
    """Dump `value` as YAML lines indented by `child_indent`.

    Uses pyyaml with sort_keys=False so the caller's dict order is preserved.
    """
    raw = yaml.safe_dump(
        value, default_flow_style=False, sort_keys=False, width=4096
    )
    # safe_dump always ends with a single trailing newline; strip it so we can
    # control whether the inserted block has its own trailing newline.
    body = raw.rstrip("\n")
    prefix = " " * child_indent
    return "\n".join(prefix + ln for ln in body.split("\n")) + "\n"


def _replace_or_insert_block(text: str, path: list[str], new_value: dict) -> str:
    """Update `text` so that the block at `path` equals `new_value`.

    Tries to be conservative — if the block exists, only that block's lines
    change. If the deepest parent in `path` exists but the leaf is missing,
    insert the leaf at the end of the parent (preserving parent ordering).
    If the parent itself is missing, append the full path at the end of the
    document.
    """
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"

    # Walk down `path[:-1]` to find each ancestor and constrain the search
    # range to that ancestor's block.
    range_start = 0
    range_end = len(lines)
    cur_indent = 0
    last_found_depth = -1
    for depth, ancestor_key in enumerate(path[:-1]):
        idx = _find_key_line(lines, ancestor_key, cur_indent, range_start, range_end)
        if idx < 0:
            break
        last_found_depth = depth
        range_start = idx + 1
        range_end = _block_end(lines, idx, cur_indent, range_end)
        cur_indent += 2

    if last_found_depth < len(path[:-1]) - 1:
        # At least one ancestor is missing. Append the missing tail at the
        # very end of the document, nested correctly.
        missing = path[last_found_depth + 1 :]
        block: dict | object = new_value
        for k in reversed(missing):
            block = {k: block}
        # If even the top-level ancestor is missing, `last_found_depth == -1`
        # and `missing == path` — we append at root.
        new_text = yaml.safe_dump(
            block if last_found_depth == -1 else block,
            default_flow_style=False,
            sort_keys=False,
            width=4096,
        )
        # When inserting under an existing ancestor block, indent the dump.
        if last_found_depth >= 0:
            insert_indent = (last_found_depth + 1) * 2
            new_text = "\n".join(
                (" " * insert_indent + ln) if ln else ln
                for ln in new_text.splitlines()
            ) + "\n"
            insert_at = range_end
        else:
            insert_at = len(lines)
        # Ensure a separating newline before the inserted block if the
        # previous line doesn't end with one (it always does after the
        # normalization above, but be defensive).
        spliced = "".join(lines[:insert_at]) + new_text + "".join(lines[insert_at:])
        return spliced

    # All ancestors exist. Look for the leaf within the parent's range.
    leaf_key = path[-1]
    leaf_indent = cur_indent
    leaf_idx = _find_key_line(lines, leaf_key, leaf_indent, range_start, range_end)
    new_block_body = _serialize_block_body(new_value, leaf_indent + 2)
    new_block = " " * leaf_indent + f"{leaf_key}:\n" + new_block_body

    if leaf_idx >= 0:
        # Replace existing block.
        leaf_end = _block_end(lines, leaf_idx, leaf_indent, range_end)
        return "".join(lines[:leaf_idx]) + new_block + "".join(lines[leaf_end:])

    # Leaf missing — insert at the end of the parent's range. The parent's
    # range_end either points at a sibling/parent boundary line OR len(lines).
    insert_at = range_end
    return "".join(lines[:insert_at]) + new_block + "".join(lines[insert_at:])


@router.put("/analysis-settings")
def save_analysis_settings(body: SaveAnalysisSettingsRequest):
    """Analysis settings cannot bypass the validated whole-draft save path."""
    project_root = _get_project_dir()
    project_path = project_root / body.project_name

    if not project_path.is_dir():
        raise HTTPException(404, f"Project not found: {body.project_name}")
    if not project_path.resolve().parent == project_root.resolve():
        raise HTTPException(400, "Invalid project name")

    config_path = project_path / CONFIG_FILENAME
    if not config_path.is_file():
        raise HTTPException(404, f"{CONFIG_FILENAME} not found")

    raise HTTPException(
        409,
        "Save analysis settings through the project configuration editor "
        "before the first run. Once a run has started, create a new project "
        "to change its configuration.",
    )


# ── Dataset browsing ─────────────────────────────────────────────────────────

_DEFAULT_DATASETS_DIR = PROJECT_ROOT / "datasets"


def _get_datasets_dir() -> Path:
    return resolve_dir(load_settings().datasets_dir, _DEFAULT_DATASETS_DIR)


def _resolve_dataset_file(reference: object, label: str) -> tuple[Path, str]:
    """Resolve a dataset reference without allowing escape from datasets_dir.

    Configs traditionally spell paths as ``datasets/<name>/<file>`` while API
    requests use paths relative to the configured datasets root. Accept both
    forms (and absolute paths already below that root), then return the
    canonical file plus its API-relative spelling. Resolving before the
    containment check also rejects symlinks that point outside the root.
    """
    if not isinstance(reference, str) or not reference.strip():
        raise HTTPException(400, f"{label} must name a file under datasets_dir")

    datasets_root = _get_datasets_dir().resolve()
    requested = Path(reference.strip()).expanduser()
    if requested.is_absolute():
        candidate = requested.resolve()
    else:
        # Repository configs use ``datasets/foo.parquet``. The server setting
        # already identifies that directory, so strip this conventional prefix
        # before joining. Plain ``foo/bar.parquet`` API references work too.
        parts = requested.parts
        if parts and parts[0] == "datasets":
            requested = Path(*parts[1:])
        candidate = (datasets_root / requested).resolve()

    try:
        relative = candidate.relative_to(datasets_root)
    except ValueError:
        raise HTTPException(
            400,
            f"{label} must resolve inside datasets_dir ({datasets_root})",
        ) from None
    if not candidate.is_file():
        raise HTTPException(404, f"{label} file not found: {reference}")
    return candidate, relative.as_posix()


def _system_transfer_dataset_inputs(config: dict) -> tuple[str, str]:
    """Return safe API dataset refs for a transfer with no data snapshot."""
    dataset = config.get("dataset")
    if not isinstance(dataset, dict):
        raise HTTPException(
            400,
            "Source project has no complete data/ snapshot and its config "
            "has no dataset mapping",
        )
    _, corpus_ref = _resolve_dataset_file(
        dataset.get("corpus"), "source config dataset.corpus",
    )
    _, qa_ref = _resolve_dataset_file(
        dataset.get("qa"), "source config dataset.qa",
    )
    return corpus_ref, qa_ref


@router.get("/datasets")
def list_datasets():
    """Auto-discover dataset folders containing parquet files."""
    datasets_dir = _get_datasets_dir()
    if not datasets_dir.is_dir():
        return {"datasets": [], "datasets_dir": str(datasets_dir)}

    datasets = []
    for d in sorted(datasets_dir.iterdir()):
        if not d.is_dir():
            continue
        parquets = sorted(f.name for f in d.glob("*.parquet"))
        if parquets:
            datasets.append({"name": d.name, "parquet_files": parquets})

    return {"datasets": datasets, "datasets_dir": str(datasets_dir)}


# ── System YAML browsing / transfer ─────────────────────────────────────────

def _load_system_yaml_document(path: Path) -> dict:
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(400, f"Invalid YAML in {path.name}: {exc}")
    if not isinstance(doc, dict):
        raise HTTPException(400, f"System YAML must be a mapping: {path.name}")
    return doc


def _extract_system_block_from_document(doc: dict, path: Path) -> dict:
    if "system" in doc:
        system = doc.get("system")
    else:
        system_keys = {
            "system_hardware", "rag_cm", "cpu", "gpu", "interconnect",
            "performance_source", "system_space_owner",
            "performance_objectives", "cm_search_space",
            "system_design_space", "measured_search_space",
            "retrieval", "available_gpus",
        }
        if not any(k in doc for k in system_keys):
            raise HTTPException(400, f"No system block found in {path.name}")
        # Transfer metadata controls orchestration only. Keep it out of a flat
        # legacy system mapping as well as out of the target RAG config.
        system = {
            key: value
            for key, value in doc.items()
            if key != _SYSTEM_TRANSFER_METADATA_KEY
        }
    if not isinstance(system, dict) or not system:
        raise HTTPException(400, f"No system block found in {path.name}")
    return system


def _extract_system_transfer_metadata(doc: dict, path: Path) -> dict:
    metadata = doc.get(_SYSTEM_TRANSFER_METADATA_KEY)
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise HTTPException(
            400,
            f"{_SYSTEM_TRANSFER_METADATA_KEY} must be a mapping in {path.name}",
        )
    unsupported = sorted(
        set(metadata) - {
            "n_refine_iterations", "generator_backend",
            "rag_config_overrides",
        },
        key=str,
    )
    if unsupported:
        raise HTTPException(
            400,
            f"Unsupported {_SYSTEM_TRANSFER_METADATA_KEY} fields in "
            f"{path.name}: {unsupported}",
        )
    result = {}
    if "n_refine_iterations" in metadata:
        iterations = metadata["n_refine_iterations"]
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations < 0
        ):
            raise HTTPException(
                400,
                f"{_SYSTEM_TRANSFER_METADATA_KEY}.n_refine_iterations in "
                f"{path.name} must be a non-negative integer",
            )
        result["n_refine_iterations"] = iterations
    if "generator_backend" in metadata:
        backend = metadata["generator_backend"]
        if (
            not isinstance(backend, str)
            or backend not in _SYSTEM_TRANSFER_GENERATOR_BACKENDS
        ):
            raise HTTPException(
                400,
                f"{_SYSTEM_TRANSFER_METADATA_KEY}.generator_backend in "
                f"{path.name} must be one of "
                f"{sorted(_SYSTEM_TRANSFER_GENERATOR_BACKENDS)}",
            )
        result["generator_backend"] = backend
    if "rag_config_overrides" in metadata:
        overrides = metadata["rag_config_overrides"]
        if not isinstance(overrides, dict):
            raise HTTPException(
                400,
                f"{_SYSTEM_TRANSFER_METADATA_KEY}.rag_config_overrides in "
                f"{path.name} must be a mapping",
            )
        unsupported_sections = sorted(
            set(overrides) - _SYSTEM_TRANSFER_RAG_OVERRIDE_SECTIONS,
            key=str,
        )
        if unsupported_sections:
            raise HTTPException(
                400,
                f"Unsupported {_SYSTEM_TRANSFER_METADATA_KEY}."
                "rag_config_overrides sections in "
                f"{path.name}: {unsupported_sections}; allowed sections are "
                f"{sorted(_SYSTEM_TRANSFER_RAG_OVERRIDE_SECTIONS)}",
            )
        for section, value in overrides.items():
            if not isinstance(value, dict):
                raise HTTPException(
                    400,
                    f"{_SYSTEM_TRANSFER_METADATA_KEY}.rag_config_overrides."
                    f"{section} in {path.name} must be a mapping",
                )
            label = (
                f"{_SYSTEM_TRANSFER_METADATA_KEY}.rag_config_overrides."
                f"{section} in {path.name}"
            )
            if section == "optimizer":
                _validate_optimizer_section(value, label=label)
            elif section == "eval_backend_setting":
                metrics = value.get("metrics", [])
                if not isinstance(metrics, list):
                    raise HTTPException(
                        400, f"{label}.metrics must be a list",
                    )
        result["rag_config_overrides"] = deepcopy(overrides)
    return result


def _quality_metric_rows(
    setting: dict, *, label: str, backend: str,
) -> list[tuple[str, dict]]:
    metrics = setting.get("metrics", [])
    if not isinstance(metrics, list):
        raise HTTPException(
            400, f"{label}.metrics must be a list for system transfer",
        )
    if not metrics:
        return []
    strict_static_gt = backend == "static_gt"
    string_form = all(isinstance(metric, str) for metric in metrics)
    mapping_form = all(isinstance(metric, dict) for metric in metrics)
    if strict_static_gt and not string_form and not mapping_form:
        raise HTTPException(
            400,
            f"{label}.metrics must be homogeneous list[str] or list[dict]",
        )
    rows: list[tuple[str, dict]] = []
    declared_names: list[str] = []
    for index, raw_metric in enumerate(metrics):
        if isinstance(raw_metric, str):
            name = raw_metric
            if not name.strip():
                raise HTTPException(
                    400, f"{label}.metrics[{index}] must be a non-empty string",
                )
            metric = {"metric_name": name}
        elif isinstance(raw_metric, dict):
            metric = raw_metric
            if "metric" in metric:
                raise HTTPException(
                    400,
                    f"{label}.metrics[{index}] uses unsupported 'metric' "
                    "alias; use metric_name",
                )
            name = metric.get("metric_name")
        else:
            raise HTTPException(
                400,
                f"{label}.metrics[{index}] must be a metric name or mapping",
            )
        if not isinstance(name, str) or not name.strip():
            raise HTTPException(
                400,
                f"{label}.metrics[{index}].metric_name must be a non-empty "
                "string",
            )
        if strict_static_gt and name in declared_names:
            raise HTTPException(
                400, f"{label}.metrics contains duplicate metric {name!r}",
            )
        declared_names.append(name)
        rows.append((name, metric))
    return rows


def _combined_quality_metric_names(
    setting: dict, metric_rows: list[tuple[str, dict]], *, label: str,
) -> tuple[list[str], dict | None]:
    declared_names = [name for name, _ in metric_rows]
    combined = setting.get("combined_quality")
    if combined is None:
        # Legacy aggregation means every emitted metric contributes to quality.
        # In that mode a diagnostic-set change is necessarily an objective
        # change, so include the complete metric vocabulary in the contract.
        return declared_names, None
    if not isinstance(combined, dict):
        raise HTTPException(
            400, f"{label}.combined_quality must be a mapping",
        )
    mode = str(combined.get("mode", "mean")).lower()
    canonical_combined = deepcopy(combined)
    canonical_combined["mode"] = mode
    if mode == "mean":
        names = combined.get("metrics", [])
        if not isinstance(names, list):
            raise HTTPException(
                400, f"{label}.combined_quality.metrics must be a list",
            )
        seen_names: list[str] = []
        for index, name in enumerate(names):
            if not isinstance(name, str) or not name.strip():
                raise HTTPException(
                    400,
                    f"{label}.combined_quality.metrics[{index}] must be a "
                    "non-empty string",
                )
            if name in seen_names:
                raise HTTPException(
                    400,
                    f"{label}.combined_quality.metrics contains duplicate "
                    f"metric {name!r}",
                )
            seen_names.append(name)
        # An empty mean list has legacy all-metric semantics in
        # aggregate_quality_metrics(). Preserve that exact behavior here.
        canonical_combined["metrics"] = list(names)
        result = list(names) if names else list(declared_names)
    elif mode == "weighted":
        weights = combined.get("weights") or {}
        if not isinstance(weights, dict) or not weights:
            raise HTTPException(
                400,
                f"{label}.combined_quality.weights must be a non-empty "
                "mapping for weighted mode",
            )
        invalid_names = [
            name for name in weights
            if not isinstance(name, str) or not name.strip()
        ]
        if invalid_names:
            raise HTTPException(
                400,
                f"{label}.combined_quality.weights keys must be non-empty "
                "strings",
            )
        canonical_combined["weights"] = deepcopy(weights)
        result = list(weights)
    else:
        raise HTTPException(
            400,
            f"{label}.combined_quality.mode {mode!r} is unsupported",
        )
    missing = [name for name in result if name not in declared_names]
    if missing:
        raise HTTPException(
            400,
            f"{label}.combined_quality references undeclared metrics: "
            f"{missing}",
        )
    return result, canonical_combined


def _quality_objective_contract(
    setting: object, *, label: str, backend: str,
) -> dict:
    if not isinstance(setting, dict):
        raise HTTPException(
            400, f"{label} must be a mapping for system transfer",
        )
    rows = _quality_metric_rows(setting, label=label, backend=backend)
    objective_names, canonical_combined = _combined_quality_metric_names(
        setting, rows, label=label,
    )
    # `description` is optimizer-facing display/prompt metadata only. Every
    # other metric field is evaluator input and therefore score-affecting.
    objective_metrics = [
        {
            key: deepcopy(value)
            for key, value in metric.items()
            if key != "description"
        }
        for name, metric in rows
        if name in objective_names
    ]
    canonical_setting = {
        key: deepcopy(value)
        for key, value in setting.items()
        if key not in {"metrics", "combined_quality", "strategy",
                       "submetric_query_subset"}
    }
    canonical_setting["combined_quality"] = canonical_combined
    canonical_setting["strategy"] = setting.get("strategy", "mean")
    try:
        canonical_setting["submetric_query_subset"] = int(
            setting.get("submetric_query_subset") or 0
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            400,
            f"{label}.submetric_query_subset must be an integer",
        ) from exc
    return {
        "setting_without_metric_vocabulary": canonical_setting,
        "objective_metric_names": list(objective_names),
        "objective_metrics": objective_metrics,
    }


def _assert_system_transfer_quality_objective_unchanged(
    source_config: dict, target_config: dict,
) -> None:
    source_backend = str(
        (source_config.get("global") or {}).get("eval_backend", "static_gt")
    ).lower()
    target_backend = str(
        (target_config.get("global") or {}).get("eval_backend", "static_gt")
    ).lower()
    if source_backend != target_backend:
        raise HTTPException(
            400, "System transfer may not change the quality evaluator backend",
        )
    source_contract = _quality_objective_contract(
        source_config.get("eval_backend_setting"),
        label="source eval_backend_setting",
        backend=source_backend,
    )
    target_contract = _quality_objective_contract(
        target_config.get("eval_backend_setting"),
        label="target eval_backend_setting",
        backend=target_backend,
    )
    if source_contract != target_contract:
        raise HTTPException(
            400,
            "System transfer may change diagnostic sub-metrics but must not "
            "change the combined quality objective, its aggregation settings, "
            "or the evaluator definitions of its input metrics",
        )


def _materialize_system_transfer_config(
    source_config: dict, system_block: dict, transfer_metadata: dict,
) -> dict:
    """Build the immutable six-section target config.

    Source optimizer/evaluator policy remains the compatibility default. A
    transfer YAML may replace either whole section explicitly; section-level
    replacement is intentional so retired params/metrics cannot survive a
    recursive merge.
    """
    target_config = {
        section: deepcopy(source_config[section])
        for section in _RAG_CONFIG_SECTIONS
        if section in source_config
    }
    target_config["system"] = deepcopy(system_block)
    overrides = transfer_metadata.get("rag_config_overrides") or {}
    for section in _SYSTEM_TRANSFER_RAG_OVERRIDE_SECTIONS:
        if section in overrides:
            target_config[section] = deepcopy(overrides[section])
    _assert_system_transfer_quality_objective_unchanged(
        source_config, target_config,
    )
    return target_config


def _validate_materialized_system_transfer_config(config: dict) -> None:
    """Run the normal six-section validator before publishing a target."""
    from rag_stack.config_validator import ConfigValidator

    if not isinstance(config, dict):
        raise HTTPException(
            400, "Materialized system-transfer config must be a mapping",
        )
    for section in _RAG_CONFIG_SECTIONS:
        if not isinstance(config.get(section), dict):
            raise HTTPException(
                400,
                f"Materialized system-transfer config section {section!r} "
                "must be a mapping",
            )
    _validate_optimizer_section(
        config["optimizer"], label="Materialized optimizer",
    )
    backend = str(config["global"].get("eval_backend", "static_gt")).lower()
    _quality_metric_rows(
        config["eval_backend_setting"],
        label="Materialized eval_backend_setting",
        backend=backend,
    )
    try:
        errors = ConfigValidator(config, check_model_cache=False).validate()
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            400, f"Invalid materialized system-transfer config: {exc}",
        ) from exc
    if errors:
        detail = "\n".join(f"  - {error}" for error in errors)
        raise HTTPException(
            400, f"Invalid materialized system-transfer config:\n{detail}",
        )


def _load_system_transfer_target(path: Path) -> tuple[dict, dict]:
    document = _load_system_yaml_document(path)
    return (
        _extract_system_block_from_document(document, path),
        _extract_system_transfer_metadata(document, path),
    )


def _extract_system_block(path: Path) -> dict:
    """Return only target-system fields, excluding orchestration metadata."""
    system, _ = _load_system_transfer_target(path)
    return system


def _resolve_system_transfer_refine_iterations(
    request_override: int | None,
    metadata: dict,
) -> tuple[int, str]:
    if request_override is not None:
        if request_override < 0:
            raise HTTPException(
                400, "n_refine_iterations must be a non-negative integer",
            )
        return int(request_override), "request_override"
    if "n_refine_iterations" in metadata:
        return int(metadata["n_refine_iterations"]), "system_yaml"
    return _DEFAULT_SYSTEM_TRANSFER_REFINE_ITERATIONS, "server_default"


def _project_system_transfer_to_vllm_chat(
    config: dict,
) -> tuple[dict, dict]:
    """Project a transfer target onto the local-vLLM chat runtime.

    Source projects and their QualityStore records stay byte-for-byte frozen:
    Phase 2a needs their original algo hashes, commit markers and frozen
    traces.  Only the newly published target search space is projected, so
    Phase 2b candidates instantiate :class:`vllm.LLM` locally instead of
    contacting an OpenAI-compatible ``vllm_api`` service.

    The historical API backend always used chat completions and defaulted
    ``max_tokens`` to 4096.  Preserve that otherwise implicit limit when an
    API module is converted (the local vLLM SamplingParams default is much
    smaller), while retaining every explicitly configured generation knob.
    """
    if not isinstance(config, dict):
        raise ValueError("RAG config must be a mapping")

    projected = deepcopy(config)
    algo = projected.get("algo_search_space")
    if algo is None:
        algo = {}
        projected["algo_search_space"] = algo
    if not isinstance(algo, dict):
        raise ValueError("algo_search_space must be a mapping")
    pipeline = algo.get("pipeline")
    if pipeline is None:
        _, report = project_to_vllm_chat(None)
        return projected, report

    projected_pipeline, report = project_to_vllm_chat(pipeline)
    algo["pipeline"] = projected_pipeline
    return projected, report


def _system_yaml_path(relative_path: str) -> Path:
    root = _get_system_yaml_root().resolve()
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise HTTPException(400, "Invalid system YAML path")
    if not path.is_file() or path.suffix not in {".yaml", ".yml"}:
        raise HTTPException(404, f"System YAML not found: {relative_path}")
    return path


def _system_summary(system: dict) -> dict:
    cpu = system.get("cpu")
    gpu = system.get("gpu")
    return {
        "cpu": cpu if isinstance(cpu, str) else (cpu or {}).get("hardware_key", "custom"),
        "gpu": gpu if isinstance(gpu, str) else (gpu or {}).get("hardware_key", "custom"),
        "performance_source": system.get("performance_source", "cost_model"),
        "available_gpus": len(system.get("available_gpus") or []),
    }


@router.get("/system-yamls")
def list_system_yamls():
    """List dedicated system-transfer YAML files."""
    root = _get_system_yaml_root()
    if not root.is_dir():
        return {"system_yamls": [], "root": str(root)}
    items = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".yaml", ".yml"}:
            continue
        try:
            system, metadata = _load_system_transfer_target(path)
        except HTTPException:
            continue
        items.append({
            "path": str(path.relative_to(root)),
            "filename": path.name,
            "summary": _system_summary(system),
            "n_refine_iterations": metadata.get(
                "n_refine_iterations",
                _DEFAULT_SYSTEM_TRANSFER_REFINE_ITERATIONS,
            ),
            "generator_backend": metadata.get("generator_backend", "inherit"),
        })
    return {"system_yamls": items, "root": str(root)}


# ── Run optimization ─────────────────────────────────────────────────────────

class RunProjectRequest(BaseModel):
    project_name: str
    # Datasets are required for the first run; on resume (the project already
    # has data/ from a previous run), they may be omitted — the Controller
    # uses the in-project copy to keep quality scores comparable.
    corpus_data: str | None = None  # relative to datasets_dir, e.g. "eli5/corpus.parquet"
    qa_data: str | None = None      # relative to datasets_dir, e.g. "eli5/qa_train.parquet"
    mode: str = "full"  # "full" | "hardware_replay"
    quality_store_path: str | None = None
    # Name of a previous project under the projects root whose GT evals
    # will be synthesized into a QualityStore before hardware_replay runs.
    # Mutually exclusive alternative to quality_store_path.
    from_project: str | None = None
    hardware_config: dict | None = None  # override system section
    n_refine_iterations: int = 0
    # If True, wipe all prior-run state (evaluations/, pareto_*, optimizer
    # snapshots, registry/, resources/, summary.json counters) before
    # launching, so the Controller starts fresh from Phase 1 instead of
    # warm-starting from the prior Pareto frontier. Dataset (data/),
    # config (rag_stack_config.yaml), and history (logs/) are preserved.
    reset: bool = False


class SystemTransferRequest(BaseModel):
    project_name: str
    source_project: str
    system_yaml: str
    # None means use the selected system-transfer YAML's default. An explicit
    # API value overrides that default for this one transfer.
    n_refine_iterations: int | None = None


@router.post("/system-transfer")
def system_transfer(body: SystemTransferRequest):
    """Create a new project by copying a source project and replacing system.

    The selected YAML may be a full RAG config or a system-only mapping. Its
    ``system`` block is copied, while optional top-level ``system_transfer``
    metadata controls the refinement budget and generator backend. Its
    optional ``rag_config_overrides`` may replace the complete optimizer and/or
    evaluator settings without entering the target RAG config as metadata. The
    new project is then launched in
    ``hardware_replay`` mode from the source project, so Phase 2a recomputes
    performance with the new system via CM and Phase 2b runs the requested
    global optimisation refinement iterations.
    """
    _validate_child_name(body.project_name, "project")
    _validate_child_name(body.source_project, "source project")
    if body.project_name == body.source_project:
        raise HTTPException(400, "Target project must be different from source project")

    project_root = _get_project_dir()
    source_path = project_root / body.source_project
    target_path = project_root / body.project_name
    if not source_path.is_dir() or source_path.resolve().parent != project_root.resolve():
        raise HTTPException(404, f"Source project not found: {body.source_project}")
    if target_path.exists():
        raise HTTPException(409, f"Project already exists: {body.project_name}")

    source_summary = load_project_summary(str(source_path))
    if (
        source_summary
        and source_summary.status == ProjectStatus.RUNNING
        and source_summary.pid
        and _is_pid_alive(source_summary.pid)
    ):
        raise HTTPException(409, "Cannot transfer from a running source project")

    source_config_path = source_path / CONFIG_FILENAME
    if not source_config_path.is_file():
        raise HTTPException(404, f"{CONFIG_FILENAME} not found in source project")

    system_path = _system_yaml_path(body.system_yaml)
    system_block, transfer_metadata = _load_system_transfer_target(system_path)
    n_refine_iterations, n_refine_iterations_source = (
        _resolve_system_transfer_refine_iterations(
            body.n_refine_iterations,
            transfer_metadata,
        )
    )
    generator_backend = transfer_metadata.get("generator_backend", "inherit")

    try:
        source_config = yaml.safe_load(source_config_path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(400, f"Source project config is invalid YAML: {exc}")
    if not isinstance(source_config, dict):
        raise HTTPException(400, "Source project config must be a mapping")

    # Materialize the documented six sections. Optimizer/evaluator overrides
    # replace their complete inherited sections and are checked against the
    # source quality-objective contract before any target state is created.
    target_config = _materialize_system_transfer_config(
        source_config, system_block, transfer_metadata,
    )
    vllm_chat_projection = None
    if generator_backend == "vllm":
        try:
            target_config, vllm_chat_projection = (
                _project_system_transfer_to_vllm_chat(target_config)
            )
        except ValueError as exc:
            raise HTTPException(
                400, f"Invalid system-transfer vLLM chat config: {exc}",
            ) from exc
    # Projection changes executable module transport, so validate the final
    # materialized config rather than the pre-projection intermediate. Do this
    # before any optional endpoint probe or target-directory write.
    _validate_materialized_system_transfer_config(target_config)

    if generator_backend != "vllm" and n_refine_iterations > 0:
        # ``inherit`` and explicit ``vllm_api`` retain the source transport.
        # Only those modes perform endpoint readiness validation. A local-vLLM
        # transfer never imports this path and never issues an HTTP request.
        from rag_stack.cli import _preflight_hardware_replay_llm_config

        try:
            _preflight_hardware_replay_llm_config(target_config)
        except ValueError as exc:
            raise HTTPException(
                503, f"System-transfer refinement preflight failed: {exc}",
            ) from exc

    # Native projects carry a resume-safe data snapshot. Converted historical
    # projects may not: in that case validate the config's dataset references
    # up front and pass lightweight paths to /run. Do not materialize the
    # (1.7 GB for MSMARCO) corpus in the API process merely to launch the run.
    source_data = source_path / "data"
    source_has_complete_data = (
        (source_data / "qa.parquet").is_file()
        and (source_data / "corpus.parquet").is_file()
    )
    transfer_corpus_data: str | None = None
    transfer_qa_data: str | None = None
    if not source_has_complete_data:
        transfer_corpus_data, transfer_qa_data = (
            _system_transfer_dataset_inputs(target_config)
        )

    target_path.mkdir(parents=True)
    write_project_config(
        target_path,
        yaml.safe_dump(target_config, sort_keys=False, default_flow_style=False, width=4096),
    )
    transfer_provenance = {
        "kind": "system_transfer_launch",
        "source_project": body.source_project,
        "system_yaml": body.system_yaml,
        "n_refine_iterations": n_refine_iterations,
        "n_refine_iterations_source": n_refine_iterations_source,
        "generator_backend": generator_backend,
        "vllm_chat_projection": vllm_chat_projection,
    }
    validate_system_transfer_artifact(
        transfer_provenance,
        expected_kind="system_transfer_launch",
        source=str(target_path / _SYSTEM_TRANSFER_PROVENANCE_FILENAME),
    )
    (target_path / _SYSTEM_TRANSFER_PROVENANCE_FILENAME).write_text(
        json.dumps(transfer_provenance, indent=2, sort_keys=True) + "\n",
    )
    if source_has_complete_data:
        shutil.copytree(source_data, target_path / "data", dirs_exist_ok=True)
    save_project_summary(str(target_path), create_project_summary(ProjectStatus.CREATED))

    response = run_project(RunProjectRequest(
        project_name=body.project_name,
        corpus_data=transfer_corpus_data,
        qa_data=transfer_qa_data,
        mode="hardware_replay",
        from_project=body.source_project,
        n_refine_iterations=n_refine_iterations,
    ))
    response.update({
        "source_project": body.source_project,
        "system_yaml": body.system_yaml,
        "system_summary": _system_summary(system_block),
        "n_refine_iterations": n_refine_iterations,
        "n_refine_iterations_source": n_refine_iterations_source,
        "generator_backend": generator_backend,
        "system_transfer_provenance": transfer_provenance,
    })
    return response


# Subpaths inside <project_dir> that hold optimization-run state. Wiped when
# `reset=True` is passed to /run. Anything not in this list — data/, the
# config YAML, logs/ — is preserved so the user can rerun on the same dataset
# and see the historical attempts.
_RESET_WIPE_DIRS = (
    "evaluations",
    # A local from-project QualityStore is part of the transfer transaction:
    # Phase 2b may append records to it. Reset means a genuinely fresh import,
    # never reuse a grown cache after deleting its transfer identity ledger.
    "quality_store",
    "pareto_configs",
    "optimizer",
    "registry",
    "resources",
)
_RESET_WIPE_FILES = (
    "all_evaluations.csv",
    "pareto_frontier.csv",
    "invalid_configs.json",
    "system_transfer_result.json",
    "system_transfer_state.json",
)


def _wipe_project_state(project_path: Path) -> dict:
    """Delete prior-run artifacts and reset summary.json status to `created`.
    Returns a small report for caller logging."""
    import shutil

    with _project_mutation_lock(project_path):
        freeze_project_config(project_path)
        return _wipe_project_state_locked(project_path)


def _wipe_project_state_locked(project_path: Path) -> dict:
    removed_dirs: list[str] = []
    removed_files: list[str] = []
    for name in _RESET_WIPE_DIRS:
        target = project_path / name
        if target.is_dir():
            shutil.rmtree(target)
            removed_dirs.append(name)
    for name in _RESET_WIPE_FILES:
        target = project_path / name
        if target.is_file():
            target.unlink()
            removed_files.append(name)
    # Reset summary.json so Controller doesn't auto-load a (now-gone) checkpoint.
    summary = create_project_summary()
    save_project_summary(str(project_path), summary)
    return {"dirs": removed_dirs, "files": removed_files}


def _project_system_transfer_worker_env(project_path: Path) -> dict[str, str]:
    """Validate provenance and isolate local system-transfer model loading.

    New transfers do not persist worker overrides: CM derives its process
    count from the child's CPU affinity (reserving two CPUs). Local vLLM chat
    targets are already cache-preflighted before launch; force offline Hub
    resolution in the child so model construction cannot spend minutes on
    irrelevant remote metadata HEAD retries when the Hub is unavailable.
    """
    provenance_path = project_path / _SYSTEM_TRANSFER_PROVENANCE_FILENAME
    if not provenance_path.is_file():
        return {}
    try:
        payload = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            400, f"Invalid {_SYSTEM_TRANSFER_PROVENANCE_FILENAME}: {exc}",
        ) from exc
    try:
        validate_system_transfer_artifact(
            payload,
            expected_kind="system_transfer_launch",
            source=str(provenance_path),
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }


@router.post("/run")
def run_project(body: RunProjectRequest):
    """Launch RAG optimization as a detached background process."""
    project_path = _existing_project_path(body.project_name)
    with _project_mutation_lock(project_path):
        return _run_project_locked(body)


def _run_project_locked(body: RunProjectRequest):
    project_root = _get_project_dir()
    project_path = project_root / body.project_name

    if not project_path.is_dir():
        raise HTTPException(404, f"Project not found: {body.project_name}")

    # Verify config exists
    config_path = project_path / CONFIG_FILENAME
    if not config_path.is_file():
        raise HTTPException(400, f"{CONFIG_FILENAME} not found in project")

    verify_project_config(project_path)

    # Check status — enforce one job per project
    summary = load_project_summary(str(project_path))
    if summary and summary.status == ProjectStatus.RUNNING:
        # Check if the PID is actually alive
        if summary.pid and _is_pid_alive(summary.pid):
            raise HTTPException(409, f"Project is already running (PID {summary.pid})")
        # PID dead but status still "running" — mark as failed so we can restart
        summary.status = ProjectStatus.FAILED
        save_project_summary(str(project_path), summary)
    elif summary and summary.status not in (ProjectStatus.CREATED, ProjectStatus.FAILED, ProjectStatus.COMPLETED):
        raise HTTPException(409, f"Project status is '{summary.status}', can only run 'created', 'failed', or 'completed' projects")

    # A failed/completed system-transfer project is not a generic project on
    # its next /run call. The creation request already persisted its immutable
    # source and initial cumulative refinement budget; infer those here so a UI "Resume"
    # cannot accidentally launch normal full-mode optimization or omit
    # --from-project and rebuild semantics. Older provenance files without
    # these fields retain their explicit request behavior.
    run_mode = body.mode
    run_quality_store_path = body.quality_store_path
    run_from_project = body.from_project
    run_refine_iterations = body.n_refine_iterations
    transfer_provenance_path = (
        project_path / _SYSTEM_TRANSFER_PROVENANCE_FILENAME
    )
    if transfer_provenance_path.is_file():
        try:
            transfer_provenance = json.loads(
                transfer_provenance_path.read_text()
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(
                400, f"Invalid system transfer provenance: {exc}",
            ) from exc
        try:
            validate_system_transfer_artifact(
                transfer_provenance,
                expected_kind="system_transfer_launch",
                source=str(transfer_provenance_path),
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        persisted_source = transfer_provenance.get("source_project")
        persisted_iterations = transfer_provenance.get("n_refine_iterations")
        if persisted_source is not None and persisted_iterations is not None:
            if (
                not isinstance(persisted_source, str)
                or not persisted_source
                or isinstance(persisted_iterations, bool)
                or not isinstance(persisted_iterations, int)
                or persisted_iterations < 0
            ):
                raise HTTPException(
                    400, "Invalid resumable system transfer provenance",
                )
            if run_from_project not in (None, persisted_source):
                raise HTTPException(
                    400, "from_project conflicts with persisted system "
                    "transfer source",
                )
            if run_quality_store_path not in (
                None, str((project_path / "quality_store").resolve()),
                str(project_path / "quality_store"),
            ):
                raise HTTPException(
                    400, "quality_store_path conflicts with the project-local "
                    "system transfer store",
                )
            # The transfer state, once created, is the authority for a budget
            # that may have been monotonically extended after project creation.
            # The creation provenance deliberately remains an audit record of
            # the original request. A non-zero /run value may request a further
            # increase; the UI's historical default 0 means "resume current".
            current_iterations = persisted_iterations
            transfer_state_path = project_path / "system_transfer_state.json"
            if transfer_state_path.is_file():
                try:
                    transfer_state = json.loads(transfer_state_path.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    raise HTTPException(
                        400, f"Invalid system transfer state: {exc}",
                    ) from exc
                try:
                    validate_system_transfer_artifact(
                        transfer_state,
                        expected_kind="system_transfer_state",
                        source=str(transfer_state_path),
                    )
                except (TypeError, ValueError) as exc:
                    raise HTTPException(400, str(exc)) from exc
                state_iterations = transfer_state.get(
                    "refinement_iterations_requested",
                )
                if (
                    isinstance(state_iterations, bool)
                    or not isinstance(state_iterations, int)
                    or state_iterations < 0
                ):
                    raise HTTPException(
                        400, "Invalid resumable system transfer state budget",
                    )
                current_iterations = state_iterations
            requested_iterations = run_refine_iterations
            if requested_iterations not in (None, 0):
                if requested_iterations < current_iterations:
                    raise HTTPException(
                        400,
                        "System transfer refinement budget cannot be decreased: "
                        f"{current_iterations} -> {requested_iterations}",
                    )
                current_iterations = requested_iterations
            run_mode = "hardware_replay"
            run_quality_store_path = None
            run_from_project = persisted_source
            run_refine_iterations = current_iterations

    # `reset=True` wipes prior-run state so the Controller starts from Phase 1
    # instead of warm-starting from the existing Pareto frontier. Must run AFTER
    # the running-check so we don't shred a live job's state from under it.
    wipe_report: dict | None = None
    if body.reset:
        wipe_report = _wipe_project_state(project_path)

    # Resolve dataset paths.
    #
    # On resume (project already has data/ from a previous run), datasets
    # may be omitted — Controller will use the in-project copy to keep
    # quality scores comparable. If they are supplied anyway, Controller
    # ignores them with a warning.
    in_project_qa = project_path / "data" / "qa.parquet"
    in_project_corpus = project_path / "data" / "corpus.parquet"
    has_in_project_data = in_project_qa.is_file() and in_project_corpus.is_file()

    requested_corpus_data = body.corpus_data
    requested_qa_data = body.qa_data
    if (
        not has_in_project_data
        and requested_corpus_data is None
        and requested_qa_data is None
        and run_from_project is not None
        and transfer_provenance_path.is_file()
    ):
        try:
            target_config = yaml.safe_load(config_path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise HTTPException(400, f"Invalid project config: {exc}") from exc
        requested_corpus_data, requested_qa_data = (
            _system_transfer_dataset_inputs(target_config)
        )

    corpus_path: Path | None = None
    qa_path: Path | None = None
    if requested_corpus_data and requested_qa_data:
        corpus_path, _ = _resolve_dataset_file(requested_corpus_data, "Corpus")
        qa_path, _ = _resolve_dataset_file(requested_qa_data, "QA")
    elif not has_in_project_data:
        raise HTTPException(
            400,
            "corpus_data and qa_data are required for the first run "
            "(project has no data/ folder yet).",
        )

    # Build subprocess command
    cmd = [
        sys.executable, "-m", "rag_stack.cli", "optimize",
        "--project-dir", str(project_path.resolve()),
        "--mode", run_mode,
    ]
    if qa_path is not None:
        cmd += ["--qa-data", str(qa_path.resolve())]
    if corpus_path is not None:
        cmd += ["--corpus-data", str(corpus_path.resolve())]
    if run_quality_store_path:
        cmd += ["--quality-store", run_quality_store_path]
    if run_from_project:
        prev_path = project_root / run_from_project
        if not prev_path.is_dir():
            raise HTTPException(404, f"from_project not found: {run_from_project}")
        cmd += ["--from-project", str(prev_path.resolve())]
    if run_refine_iterations:
        cmd += ["--n-refine-iterations", str(run_refine_iterations)]
    if body.hardware_config:
        import json as _json
        import tempfile
        hw_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=str(project_path),
        )
        _json.dump(body.hardware_config, hw_file)
        hw_file.close()
        cmd += ["--hardware-config", hw_file.name]

    # The core freeze survives failures, reset, and stale summary counters.
    # Both the API parent and CLI child use this exact unexpanded YAML.
    freeze_project_config(project_path)

    # Launch subprocess — summary.json is the sole tracking mechanism.
    # Per-run log goes under <project>/logs/optimization_<timestamp>.log so
    # successive resumes don't overwrite each other; the user can scroll
    # back through every attempt by listing the logs folder.
    transfer_worker_env = _project_system_transfer_worker_env(project_path)
    logs_dir = project_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_fh = open(log_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env[PROJECT_LOG_FILE_ENV] = str(log_path)
    env[PROJECT_LOG_OWNED_BY_PARENT_ENV] = "1"
    env[PROJECT_LOG_STDIO_REDIRECTED_ENV] = "1"
    if transfer_provenance_path.is_file():
        # Do not let a server-level legacy cap leak back into either a new or a
        # resumed transfer.  With both variables absent, CM uses every
        # affinity-visible CPU except its two reserved CPUs.
        for key in _LEGACY_SYSTEM_TRANSFER_CM_WORKER_ENV:
            env.pop(key, None)
    env.update(transfer_worker_env)
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    except Exception:
        log_fh.close()
        failed_summary = load_project_summary(str(project_path)) or create_project_summary()
        failed_summary.status = ProjectStatus.FAILED
        failed_summary.pid = None
        save_project_summary(str(project_path), failed_summary)
        raise

    # Claim the project for the child immediately. QualityStore synthesis and
    # Controller construction can both be long-running; leaving CREATED here
    # made status monitoring lie and allowed a second /run to start against the
    # same project. save_project_summary is an atomic replace.
    launched_summary = load_project_summary(str(project_path))
    if launched_summary is None:
        launched_summary = create_project_summary()
    launched_summary.status = ProjectStatus.RUNNING
    launched_summary.pid = proc.pid
    launched_summary.timestamp = datetime.now().isoformat()
    try:
        save_project_summary(str(project_path), launched_summary)
    except Exception:
        try:
            proc.terminate()
        finally:
            log_fh.close()
        raise

    # Daemon reaper thread to prevent zombies
    def _wait_and_close_log():
        return_code = None
        try:
            return_code = proc.wait()
        finally:
            log_fh.close()
        if return_code in (None, 0):
            return
        try:
            terminal_summary = load_project_summary(str(project_path))
            if (
                terminal_summary is not None
                and terminal_summary.status == ProjectStatus.RUNNING
                and terminal_summary.pid == proc.pid
            ):
                terminal_summary.status = ProjectStatus.FAILED
                terminal_summary.pid = None
                terminal_summary.timestamp = datetime.now().isoformat()
                save_project_summary(str(project_path), terminal_summary)
        except Exception as exc:  # daemon bookkeeping must not mask process exit
            logger.error(
                "Failed to mark project %s failed after child PID %s exited "
                "with code %s: %s",
                body.project_name, proc.pid, return_code, exc,
            )

    threading.Thread(target=_wait_and_close_log, daemon=True).start()

    # Return immediately with the PID for the UI. The parent-owned summary is
    # already RUNNING, including during child preflight and initialization.
    resp: dict = {"ok": True, "project_name": body.project_name, "pid": proc.pid}
    if wipe_report is not None:
        resp["wipe_report"] = wipe_report
    return resp


# ── Project status (replaces job_store for RAG) ─────────────────────────────

@router.post("/kill/{project_name}")
def kill_project(project_name: str):
    """Kill a running project's optimization process (and all descendants)."""
    from ..utils import kill_process_tree
    project_root = _get_project_dir()
    project_path = project_root / project_name

    if not project_path.is_dir():
        raise HTTPException(404, f"Project not found: {project_name}")

    summary = load_project_summary(str(project_path))
    if not summary or summary.status != ProjectStatus.RUNNING or not summary.pid:
        raise HTTPException(400, "Project is not running")

    # Walk the whole process tree so vllm EngineCore subprocesses get reaped
    # too — a plain os.kill(pid, SIGTERM) on the top python orphans them and
    # leaves GPU memory allocated indefinitely.
    report = kill_process_tree(summary.pid)

    summary.status = ProjectStatus.FAILED
    save_project_summary(str(project_path), summary)
    return {"ok": True, "project_name": project_name, "kill_report": report}


@router.get("/status/{project_name}")
def get_project_status(project_name: str):
    """Get live project status from summary.json, with PID liveness check."""
    project_root = _get_project_dir()
    project_path = project_root / project_name

    if not project_path.is_dir():
        raise HTTPException(404, f"Project not found: {project_name}")

    summary = load_project_summary(str(project_path))
    if summary is None:
        raise HTTPException(404, "No summary.json found")

    result = summary.model_dump()

    # If status is "running", verify PID is actually alive
    if summary.status == ProjectStatus.RUNNING and summary.pid:
        if not _is_pid_alive(summary.pid):
            result["status"] = "failed"
            result["pid_alive"] = False
            # Also fix summary on disk
            summary.status = ProjectStatus.FAILED
            save_project_summary(str(project_path), summary)
        else:
            result["pid_alive"] = True
    else:
        result["pid_alive"] = False

    return result


# ── Parsed evaluations ───────────────────────────────────────────────────────
# all_evaluations.csv stores composite search knobs as Python dict reprs (the
# `pipeline__<branch>` / `corpus__<node>` cells). Only the backend can parse
# its own repr format natively (ast.literal_eval), so the rows are served
# fully parsed here, with each config flattened to dot-path scalars — the UI
# must never re-parse repr strings client-side.

# Non-knob CSV columns. Everything else with a `__` separator is a search knob.
_EVAL_RESERVED_COLS = {"eval_id", "eval_type", "eval_error", "performance"}
_EVAL_GT_COLS = ("gt_bleu", "gt_meteor", "gt_rouge")
# Legacy eval_type aliases from pre-rename CSVs (mirrors rag_stack.types).
_EVAL_TYPE_ALIASES = {"SKIPPED": "DUPLICATE", "INVALID_PENALIZED": "INVALID"}
_EVAL_TYPES = {"GT", "CM", "DUPLICATE", "OOM", "INVALID"}


def _native_scalar(value):
    """numpy scalar → python native; int-valued floats → int (48.0 → 48)."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _flatten_knob(path: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten_knob(f"{path}.{k}", v, out)
    elif isinstance(value, (list, tuple)):
        # Rare in configs; keep the row scalar-valued.
        out[path] = json.dumps(list(value))
    elif value is None:
        return
    else:
        out[path] = _native_scalar(value)


@router.get("/evaluations")
def project_evaluations(project: str = ""):
    """All evaluations of a project, parsed and config-flattened.

    Returns rows in CSV (append) order — an eval_id may appear more than once
    when it was upgraded (CM → GT); consumers keep their per-id tier logic.
    `config` maps dot-path knob keys to scalars, e.g.
    `pipeline.sequential.semantic_retrieval.top_k: 8`.
    """
    import ast

    import pandas as pd

    root = _get_project_dir()
    if project:
        _validate_child_name(project, "project")
    project_path = root / project if project else root
    csv = project_path / "all_evaluations.csv"
    if not csv.is_file():
        raise HTTPException(404, f"No all_evaluations.csv for project: {project or '<root>'}")
    try:
        df = pd.read_csv(csv)
    except Exception as e:  # unreadable/corrupt CSV is a client-visible error
        raise HTTPException(500, f"Failed to read all_evaluations.csv: {e}")

    rows = []
    for _, row in df.iterrows():
        config: dict = {}
        quality: dict[str, float] = {}
        for col, raw in row.items():
            if pd.isna(raw) or raw == "":
                continue
            if col.startswith("quality__"):
                try:
                    quality[col[len("quality__"):]] = float(raw)
                except (TypeError, ValueError):
                    pass
                continue
            if col in _EVAL_RESERVED_COLS or col in _EVAL_GT_COLS:
                continue
            if "__" not in col:
                continue
            base = col.replace("__", ".", 1)
            if isinstance(raw, str) and raw.lstrip()[:1] in "{[":
                try:
                    parsed = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    config[base] = raw
                    continue
                _flatten_knob(base, parsed, config)
            else:
                config[base] = _native_scalar(raw)

        raw_type = str(row.get("eval_type", "") or "")
        eval_type = _EVAL_TYPE_ALIASES.get(raw_type, raw_type)
        if eval_type not in _EVAL_TYPES:
            eval_type = "INVALID"

        perf = row.get("performance")
        gt = None
        if "gt_bleu" in df.columns and pd.notna(row.get("gt_bleu")):
            gt = {c: float(row.get(c, 0.0) or 0.0) for c in _EVAL_GT_COLS}

        err = row.get("eval_error")
        rows.append({
            "eval_id": int(row.get("eval_id", 0) or 0),
            "eval_type": eval_type,
            "eval_error": "" if err is None or pd.isna(err) else str(err),
            "performance": float(perf) if pd.notna(perf) else 0.0,
            "quality": quality,
            "gt_metrics": gt,
            "config": config,
        })

    return {"evaluations": rows}


@router.get("/evaluations/{eval_id}/trace")
def evaluation_trace(eval_id: int, project: str = ""):
    """Read the saved quality invocation trace; never reconstruct observations."""
    from rag_stack.rag_ir.trace import validate_quality_trace_envelope
    from rag_stack.rag_ir.workflow_schema import WorkflowSchema, validate_workflow_trace

    if eval_id < 0:
        raise HTTPException(400, "Evaluation id must be non-negative.")
    if project:
        _validate_child_name(project, "project")
    root = _get_project_dir().resolve()
    project_path = (root / project).resolve() if project else root
    if project and project_path.parent != root:
        raise HTTPException(400, "Invalid project name")
    if not project_path.is_dir():
        raise HTTPException(404, "Project not found")
    trace_path = (project_path / "evaluations" / f"eval_{eval_id:04d}" / "execution_dag.json").resolve()
    if not trace_path.is_relative_to(project_path):
        raise HTTPException(400, "Evaluation artifact is outside its project.")
    if not trace_path.is_file():
        return {"available": False, "trace": None, "workflow": None,
                "source": "quality", "reason": "No observed quality trace was saved for this evaluation."}
    try:
        trace = json.loads(trace_path.read_text())
        validate_quality_trace_envelope(trace)
        workflow = (trace.get("provenance") or {}).get("workflow_schema")
        if workflow is not None:
            validate_workflow_trace(WorkflowSchema.from_dict(workflow), trace)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(422, f"Invalid saved evaluation trace: {exc}") from exc
    return {"available": True, "trace": trace, "workflow": workflow, "source": "quality"}


# ── Save project quality into a QualityStore ────────────────────────────────

class SaveProjectQualityStoreRequest(BaseModel):
    mode: str = "existing"  # "existing" | "new"
    store_name: str


def _count_store_configs(store_path: Path) -> int:
    manifest_path = store_path / "manifest.json"
    if not manifest_path.is_file():
        return 0
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return 0
    directories = ["configs", "quality", "pipeline_configs"]
    if int(manifest.get("version") or 1) >= 2:
        directories.extend(["traces", "records"])
    hash_sets: list[set[str]] = []
    for dirname in directories:
        path = store_path / dirname
        if not path.is_dir():
            return 0
        hash_sets.append({item.stem for item in path.glob("*.json")})
    return len(set.intersection(*hash_sets)) if hash_sets else 0


@router.post("/projects/{project_name}/save-quality-store")
def save_project_quality_store(project_name: str, body: SaveProjectQualityStoreRequest):
    """Copy a project's completed GT quality evaluations into a QualityStore.

    ``mode=existing`` appends only new config hashes to an existing store and
    enforces identical dataset identity. ``mode=new`` creates a new store from
    the project quality data.
    """
    _validate_child_name(project_name, "project")
    _validate_child_name(body.store_name, "quality store")
    if body.mode not in {"existing", "new"}:
        raise HTTPException(400, "mode must be 'existing' or 'new'")

    project_root = _get_project_dir()
    project_path = project_root / project_name
    if not project_path.is_dir() or project_path.resolve().parent != project_root.resolve():
        raise HTTPException(404, f"Project not found: {project_name}")

    summary = load_project_summary(str(project_path))
    if summary and summary.status == ProjectStatus.RUNNING and summary.pid and _is_pid_alive(summary.pid):
        raise HTTPException(409, "Cannot save quality while the project is running")

    store_root = _get_quality_store_dir()
    store_path = store_root / body.store_name
    if store_path.resolve().parent != store_root.resolve():
        raise HTTPException(400, "Invalid quality store name")

    if body.mode == "new":
        if store_path.exists():
            raise HTTPException(409, f"Quality store already exists: {body.store_name}")
        merge = False
    else:
        if not store_path.is_dir():
            raise HTTPException(404, f"Quality store not found: {body.store_name}")
        manifest_path = store_path / "manifest.json"
        if not manifest_path.is_file():
            raise HTTPException(400, "Quality store manifest.json is missing")
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            raise HTTPException(400, f"Could not read quality store manifest: {exc}")
        pid = manifest.get("pid")
        if manifest.get("status") == "running" and pid and _is_pid_alive(pid):
            raise HTTPException(409, "Cannot save into a running quality store")
        merge = True

    before = _count_store_configs(store_path)
    try:
        from rag_stack.quality_store import QualityStore
        QualityStore.from_project(str(project_path), str(store_path), merge=merge)
    except ValueError as exc:
        raise HTTPException(409, str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(400, str(exc))
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Failed to save project quality: {exc}")

    manifest_path = store_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    after = _count_store_configs(store_path)
    imported_hashes = manifest.get("last_imported_hashes") or []
    return {
        "ok": True,
        "project_name": project_name,
        "store_name": body.store_name,
        "store_path": str(store_path),
        "mode": body.mode,
        "n_before": before,
        "n_after": after,
        "n_imported": len(imported_hashes),
        "n_total": after,
        "dataset": manifest.get("dataset", {}),
        "imported_hashes": imported_hashes,
    }


# ── Delete project ────────────────────────────────────────────────────────

@router.delete("/projects/{project_name}")
def delete_project(project_name: str):
    """Delete a project folder and all its data."""
    project_root = _get_project_dir()
    project_path = project_root / project_name

    if not project_path.is_dir():
        raise HTTPException(404, f"Project not found: {project_name}")
    if project_path.resolve().parent != project_root.resolve():
        raise HTTPException(400, "Invalid project name (path traversal)")

    # Don't delete running projects
    summary = load_project_summary(str(project_path))
    if summary and summary.status == ProjectStatus.RUNNING and summary.pid:
        if _is_pid_alive(summary.pid):
            raise HTTPException(409, "Cannot delete a running project. Kill it first.")

    shutil.rmtree(str(project_path), ignore_errors=True)
    return {"ok": True, "project_name": project_name}


# ── Template config management ─────────────────────────────────────────────

@router.delete("/configs/{filename}")
def delete_template_config(filename: str):
    """Delete a template config YAML from the configs directory."""
    if not filename.endswith(".yaml"):
        raise HTTPException(400, "Filename must end with .yaml")

    configs_dir = _get_configs_dir()
    config_path = (configs_dir / filename).resolve()

    if config_path.parent != configs_dir.resolve():
        raise HTTPException(400, "Invalid filename (path traversal)")
    if not config_path.is_file():
        raise HTTPException(404, f"Config not found: {filename}")

    config_path.unlink()
    return {"ok": True}


@router.post("/configs/{filename}/duplicate")
def duplicate_template_config(filename: str, body: DuplicateConfigRequest):
    """Duplicate an existing template config to a new filename."""
    if not body.new_filename.endswith(".yaml"):
        raise HTTPException(400, "new_filename must end with .yaml")

    configs_dir = _get_configs_dir()
    src_path = (configs_dir / filename).resolve()
    dst_path = (configs_dir / body.new_filename).resolve()

    if src_path.parent != configs_dir.resolve():
        raise HTTPException(400, "Invalid source filename")
    if dst_path.parent != configs_dir.resolve():
        raise HTTPException(400, "Invalid target filename")
    if not src_path.is_file():
        raise HTTPException(404, f"Source config not found: {filename}")
    if dst_path.exists():
        raise HTTPException(409, f"Target config already exists: {body.new_filename}")

    shutil.copy2(str(src_path), str(dst_path))
    return {"ok": True, "filename": body.new_filename}


@router.put("/configs/{filename}")
def save_template_config(filename: str, body: SaveTemplateConfigRequest):
    """Save or update a template config YAML in the configs directory."""
    if not filename.endswith(".yaml"):
        raise HTTPException(400, "Filename must end with .yaml")

    configs_dir = _get_configs_dir()
    config_path = (configs_dir / filename).resolve()

    if config_path.parent != configs_dir.resolve():
        raise HTTPException(400, "Invalid filename (path traversal)")

    # Validate YAML
    try:
        yaml.safe_load(body.content)
    except yaml.YAMLError as e:
        raise HTTPException(400, f"Invalid YAML: {e}")

    configs_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(body.content)
    return {"ok": True}


# ── Config validation & scaffolding (reuse the canonical Python schema) ─────
# The frontend must NOT reimplement the 6-section schema rules. These two
# endpoints expose the single source of truth (ConfigValidator) and a known-
# valid template builder so the editor validates/creates by calling the backend.

class ValidateConfigRequest(BaseModel):
    content: str | None = None   # raw YAML text
    config: dict | None = None   # OR an already-parsed config object


def _component_capability_metadata(component: str, *, registered: bool | None = None) -> dict:
    """Describe registered execution and analytical support without a runtime."""
    from rag_stack.pipeline_contract import retrieval_component_capabilities
    from rag_stack.taxonomy import TAXONOMY

    retrieval = retrieval_component_capabilities(component)
    if retrieval is not None:
        return retrieval
    entry = TAXONOMY.components.get(component)
    free_components = {"fstring", "chat_fstring", "long_context_reorder", "window_replacement", "time_reranker"}
    modeled = entry is not None and entry.shape is not None
    capability = {
        "execution_supported": registered if registered is not None else entry is not None or component in free_components,
        "cost_model_supported": modeled or component in free_components,
        "supported_dataflows": ["sequential"],
    }
    if entry is not None and entry.stage.value in {"semantic_retrieval", "passage_reranker", "generator"}:
        capability["supported_dataflows"].append("react")
    if component in {"vectordb", "vllm", "vllm_api", "fstring", "chat_fstring"}:
        capability["measured_supported"] = True
    if not capability["cost_model_supported"]:
        capability["reason"] = "No analytical cost-model binding is registered for this component."
    return capability


@router.post("/plan")
def plan_pipeline(body: ValidateConfigRequest):
    """Preview logical wiring and capabilities without loading any runtime.

    This is graph/contract validation. Full dataset, hardware, and optimizer
    validation remains available through /validate-config. Unsupported
    analytical performance does not erase an otherwise valid workflow.
    """
    from rag_stack.pipeline_contract import (
        hybrid_capability_errors, hybrid_structure_errors,
        normalize_retrieval_aliases, pipeline_nodes,
    )
    from rag_stack.rag_ir.plan import describe_pipeline

    try:
        config = body.config if body.config is not None else yaml.safe_load(body.content or "")
    except yaml.YAMLError as exc:
        return {"valid": False, "errors": [f"YAML parse error: {exc}"], "workflow": None, "capabilities": {}}
    if not isinstance(config, dict):
        return {"valid": False, "errors": ["Provide a config mapping or YAML content."], "workflow": None, "capabilities": {}}
    config = normalize_retrieval_aliases(config)
    pipeline = (config.get("algo_search_space") or {}).get("pipeline", config)
    if not isinstance(pipeline, dict):
        return {"valid": False, "errors": ["Pipeline must be a mapping."], "workflow": None, "capabilities": {}}
    blocks = pipeline.get("methods", [pipeline])
    if not isinstance(blocks, list) or any(not isinstance(b, dict) for b in blocks):
        return {"valid": False, "errors": ["Pipeline methods must be a list of mappings."], "workflow": None, "capabilities": {}}
    backend = str((config.get("global") or {}).get("eval_backend", "static_gt")).lower()
    performance_source = str((config.get("system") or {}).get("performance_source", "cost_model")).lower()
    structural_errors = [error for block in blocks for error in hybrid_structure_errors(block, backend=backend)]
    run_errors = [error for block in blocks for error in hybrid_capability_errors(block, performance_source=performance_source)]
    modules = [module for block in blocks for node in pipeline_nodes(block)
               for module in node.get("modules", []) or [] if isinstance(module, dict)]
    component_caps = {str(module.get("component", "")): _component_capability_metadata(str(module.get("component", "")))
                      for module in modules}
    missing_cm = [name for name, capability in component_caps.items() if not capability["cost_model_supported"]]
    reasons = list(dict.fromkeys(structural_errors + [
        str(component_caps[name].get("reason", f"No cost model for {name}.")) for name in missing_cm
    ]))
    capabilities = {
        "execution_supported": not structural_errors and all(c["execution_supported"] for c in component_caps.values()),
        "measured_supported": (
            False if structural_errors or backend != "static_gt" or any(c.get("measured_supported") is False for c in component_caps.values())
            else True if component_caps and all(c.get("measured_supported") is True for c in component_caps.values())
            else None
        ),
        "cost_model_supported": not structural_errors and not missing_cm,
        "supported_dataflows": ["sequential"] if any(name in component_caps for name in ("bm25", "hybrid_rrf", "hybrid_cc")) else ["sequential", "react"],
        "reasons": reasons,
        "components": component_caps,
    }
    try:
        workflow = describe_pipeline(config)
    except (ValueError, TypeError, KeyError) as exc:
        structural_errors.append(str(exc))
        capabilities["execution_supported"] = False
        capabilities["measured_supported"] = False
        workflow = None
    errors = list(dict.fromkeys(structural_errors + run_errors))
    return {"valid": not errors, "errors": errors, "workflow": workflow, "capabilities": capabilities}


@router.post("/validate-config")
def validate_config(body: ValidateConfigRequest):
    """Run the canonical ConfigValidator over a config and return its errors.

    ``check_model_cache=False`` — never snapshot-download HF models during an
    interactive validation (that would hang the request on a cold cache).
    """
    if body.config is not None:
        cfg = body.config
    elif body.content is not None:
        try:
            cfg = yaml.safe_load(body.content)
        except yaml.YAMLError as e:
            return {"valid": False, "errors": [f"YAML parse error: {e}"]}
    else:
        raise HTTPException(400, "Provide 'content' (YAML string) or 'config' (object).")
    if not isinstance(cfg, dict):
        return {"valid": False, "errors": ["Config must be a mapping/object at the top level."]}
    _warm_static_rag_evaluator()
    from rag_stack.config_validator import ConfigValidator
    try:
        errors = ConfigValidator(cfg, check_model_cache=False).validate()
    except Exception as e:  # validator should collect errors, not raise — be defensive
        return {"valid": False, "errors": [f"validator crashed: {e}"]}
    return {"valid": not errors, "errors": errors}


class ScaffoldRequest(BaseModel):
    eval_backend: str = "static_gt"          # Built-in or installed evaluator name.
    optimizer_type: str = "agent_qnehvi"
    performance_source: str = "cost_model"   # cost_model | measured (static_gt only)


_JOINT_INIT_OPTIMIZERS = frozenset({
    "ax_qnehvi",
    "ax_qnehvi_mixedgp",
    "agent_qnehvi",
    "agent_qnehvi_slo",
})


def _scaffold_optimizer(optimizer_type: str, n_init: int) -> dict:
    """Build params accepted by the selected optimizer's strict constructor."""
    params = {"seed": 42}
    if optimizer_type == "smac_mobo":
        params["n_gt_init"] = int(n_init)
    elif optimizer_type in _JOINT_INIT_OPTIMIZERS:
        params["n_both_init"] = int(n_init)
    return {"type": optimizer_type, "params": params}


def _scaffold_static(optimizer_type: str, performance_source: str) -> dict:
    measured = performance_source == "measured"
    system: dict = {
        "cpu": "epyc_9124",
        "gpu": "A100_80GB_GPU",
        "performance_objectives": {"selection": "max_throughput"},
        "cm_search_space": {
            "placement_policy": ["disaggregated", "collocated"],
            "min_num_gpus": 2, "max_num_gpus": 4, "max_num_gpus_per_stage": 4,
            "gpus_per_server": 4, "max_batch_size_request": 256,
            "no_microbatching": True,
        },
    }
    if measured:
        system["performance_source"] = "measured"
        system["cm_search_space"].update({"min_num_gpus": 2, "max_num_gpus": 2, "max_num_gpus_per_stage": 2})
        system["system_design_space"] = {
            "batch_size_request": [1, 8, 16, 32],
            "vllm_kv_cache_dtype": "auto",
            "available_gpus": ["cuda:0", "cuda:1"],
        }
    return {
        "global": {"eval_backend": "static_gt", "rag_ir_mode": "dynamic", "n_iterations": 30},
        "dataset": {
            "dataset_name": "dragonball_en",
            "qa": "datasets/dragonball/qa_en_sample_100.parquet",
            "corpus": "datasets/dragonball/raw_corpus_en.parquet",
        },
        "system": system,
        "optimizer": _scaffold_optimizer(optimizer_type, 10),
        "eval_backend_setting": {
            "combined_quality": {"mode": "mean", "metrics": ["deepeval_answer_correctness"]},
            "metrics": [{
                "metric_name": "deepeval_answer_correctness",
                "model": "openai/deepseek/deepseek-v4-flash",
            }],
        },
        "algo_search_space": {
            # vectordb is ALGO search space (index family/build knobs move
            # quality) — it lives here beside corpus/pipeline; the resolved
            # runtime config re-exposes config["vectordb"] for evaluators.
            "vectordb": [{
                "name": "faiss_ivf_index", "db_type": "faiss_ivf",
                "embedding_model": "huggingface_all_mpnet_base_v2",
                "collection_name": "my_index", "path": "${PROJECT_DIR}/data/faiss",
                "embedding_dim": 768, "nlist_factor": [1, 2, 4, 8],
                # Unified faiss_ivf: shared nlist_factor + hierarchical index_type
                # (pq exclusive knobs active only when pq is chosen; flat has none).
                "index_type": {"pq": {"M": [8, 16, 32, 48, 96], "nbits": [4, 8]}, "flat": {}},
            }],
            "pipeline": {"node_lines": [
                {"node_line_name": "retrieve_node_line", "nodes": [
                    {"stage": "semantic_retrieval", "top_k": [4, 8, 16], "modules": [
                        {"component": "vectordb", "vectordb": "faiss_ivf_index",
                         "nprobe": [8, 16, 32, 64]},
                    ]},
                ]},
                {"node_line_name": "post_retrieve_node_line", "nodes": [
                    {"stage": "prompt_maker", "modules": [
                        {"component": "fstring",
                         "prompt": "Read the passages and answer the question.\n Question: {query}\n Passage: {retrieved_contents}\n Answer:"},
                    ]},
                    {"stage": "generator", "modules": [
                        {"component": "vllm",
                         "model": ["Qwen/Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-7B-Instruct"],
                         "temperature": {"range": [0.0, 1.0]}},
                    ]},
                ]},
            ]},
        },
    }


def _flashrag_method_block(mode: str, extra: dict | None = None) -> dict:
    block = {"mode": mode}
    if extra:
        block.update(extra)
    block["node_lines"] = [
        {"node_line_name": "retrieve_node_line", "nodes": [
            {"stage": "semantic_retrieval", "top_k": [4, 8], "modules": [
                {"component": "vectordb", "vectordb": "faiss_hnsw_index"},
            ]},
        ]},
        {"node_line_name": "generate_node_line", "nodes": [
            {"stage": "generator", "modules": [
                {"component": "vllm", "model": "Qwen/Qwen2.5-7B-Instruct", "temperature": 0.5},
            ]},
        ]},
    ]
    return block


def _scaffold_flashrag(optimizer_type: str) -> dict:
    return {
        "global": {"eval_backend": "flashrag", "rag_ir_mode": "dynamic", "n_iterations": 20},
        "dataset": {
            "dataset_name": "dragonball_en",
            "qa": "datasets/dragonball/qa_en_sample_100.parquet",
            "corpus": "datasets/dragonball/raw_corpus_en.parquet",
        },
        "system": {
            "cpu": "epyc_9124",
            "gpu": "A100_80GB_GPU",
            "performance_objectives": {"selection": "max_throughput"},
            "cm_search_space": {
                "placement_policy": ["disaggregated"],
                "min_num_gpus": 2, "max_num_gpus": 4, "max_num_gpus_per_stage": 2,
                "gpus_per_server": 1, "max_batch_size_request": 8,
                "no_microbatching": True,
            },
        },
        "optimizer": _scaffold_optimizer(optimizer_type, 5),
        "eval_backend_setting": {
            "metrics": [{"metric_name": "em"}, {"metric_name": "f1"}, {"metric_name": "rouge-l"}],
            "flashrag": {"framework": "vllm", "model2path": {"e5": "intfloat/e5-base-v2"}},
        },
        "algo_search_space": {
            "vectordb": [{
                "name": "faiss_hnsw_index", "db_type": "faiss_hnsw",
                "embedding_model": "e5", "collection_name": "my_hnsw_index",
                "path": "${PROJECT_DIR}/data/faiss", "embedding_dim": 768,
                "M": [16, 32], "ef_construction": 200, "ef_search": [64, 128],
                "faiss_type": "HNSW32",
            }],
            "pipeline": {"methods": [
                _flashrag_method_block("sequential"),
                _flashrag_method_block("ircot", {"max_iter": [3, 5]}),
            ]},
        },
    }


# ── Project HV comparison (compare optimization runs by hypervolume) ────────
# Ports analysis/compare_projects.py: running 2-objective hypervolume
# (performance × quality), normalized to [0,1] on bounds SHARED across the
# compared projects so the HV numbers are directly comparable. Reuses the same
# all_evaluations.csv the controller writes. The HV curve's x-axis counts
# VALID evals only — valid means a GT row with both objectives and no
# eval_error; failure buckets (CM/DUPLICATE/OOM/INVALID/…) add no point and
# no x-step and are not shipped in `points` at all — so every line ends
# exactly at its run's own valid-eval count and same-x points across runs
# compare equal spent budget.

class CompareHvRequest(BaseModel):
    projects: list[str]
    quality_metric: str | None = None


def _pareto_front_2d(points):
    """Non-dominated subset (both objectives maximized), sorted by perf."""
    import numpy as np
    if len(points) == 0:
        return np.empty((0, 2))
    keep = []
    for i, (a, b) in enumerate(points):
        dominated = any(
            (points[j, 0] >= a and points[j, 1] >= b and (points[j, 0] > a or points[j, 1] > b))
            for j in range(len(points)) if j != i
        )
        if not dominated:
            keep.append((a, b))
    return np.array(sorted(set(map(tuple, keep)))) if keep else np.empty((0, 2))


def _hypervolume_2d(points, ref=(0.0, 0.0)):
    """Exact 2-D dominated hypervolume above-right of ref (both maximized)."""
    import numpy as np
    if len(points) == 0:
        return 0.0
    front = _pareto_front_2d(points)
    if len(front) == 0:
        return 0.0
    front = front[(front[:, 0] > ref[0]) & (front[:, 1] > ref[1])]
    if len(front) == 0:
        return 0.0
    front = front[np.argsort(-front[:, 1])]
    hv = 0.0
    prev = ref[0]
    for perf, qual in front:
        if perf > prev:
            hv += (perf - prev) * (qual - ref[1])
            prev = perf
    return float(hv)


@router.post("/compare-hv")
def compare_hv(body: CompareHvRequest):
    import numpy as np
    import pandas as pd

    root = _get_project_dir()
    # 1. Load each project's evals + discover available quality metrics.
    loaded: list[tuple[str, "pd.DataFrame | None"]] = []
    available: set[str] = set()
    for name in body.projects:
        csv = root / name / "all_evaluations.csv"
        if not csv.is_file():
            loaded.append((name, None))
            continue
        try:
            df = pd.read_csv(csv)
        except Exception:
            loaded.append((name, None))
            continue
        available.update(c[len("quality__"):] for c in df.columns if c.startswith("quality__"))
        loaded.append((name, df))

    if not available:
        return {"error": "No quality__<metric> columns found in any selected project.",
                "available_metrics": []}
    metric = body.quality_metric
    if not metric or metric not in available:
        metric = "deepeval_answer_correctness" if "deepeval_answer_correctness" in available else sorted(available)[0]
    qcol = f"quality__{metric}"

    # 2. Normalize per-project frames to [eval_id, perf, qual, valid].
    frames: list[tuple[str, "pd.DataFrame | None"]] = []
    for name, df in loaded:
        if df is None or qcol not in df.columns or "performance" not in df.columns:
            frames.append((name, None))
            continue
        ev = pd.DataFrame({
            "eval_id": df["eval_id"] if "eval_id" in df else np.arange(len(df)),
            "perf": pd.to_numeric(df["performance"], errors="coerce"),
            "qual": pd.to_numeric(df[qcol], errors="coerce"),
        })
        # valid = a GT row with both objectives AND no eval_error. eval_type is
        # the authoritative signal — only GT rows carry real measured quality;
        # every failure bucket (CM/DUPLICATE/OOM/INVALID/CONTEXT_OVERFLOW or
        # any future type) must never enter the HV curve, the Pareto front,
        # the scatter, or the shared bounds, even when it slipped penalized
        # objective values with an empty error message into the CSV. The
        # error check stays as belt-and-braces for GT rows.
        if "eval_error" in df.columns:
            err = df["eval_error"].astype(str).str.strip().str.lower()
            no_err = err.isin(("", "nan", "none"))
        else:
            no_err = True
        if "eval_type" in df.columns:
            is_gt = df["eval_type"].astype(str).str.strip().str.upper().eq("GT")
        else:
            is_gt = True  # legacy CSVs — fall back to the heuristic
        ev["valid"] = ev["perf"].notna() & ev["qual"].notna() & no_err & is_gt
        frames.append((name, ev.sort_values("eval_id").reset_index(drop=True)))

    # 3. Shared normalization bounds across all valid points of all projects.
    valid_stack = [f.loc[f.valid, ["perf", "qual"]].values for _, f in frames if f is not None and f.valid.any()]
    if not valid_stack:
        return {"error": f"No valid (perf, quality) points for metric '{metric}' in the selected projects.",
                "metric": metric, "available_metrics": sorted(available)}
    allp = np.vstack(valid_stack)
    lo = allp.min(axis=0)
    hi = allp.max(axis=0)
    span = np.where((hi - lo) > 0, hi - lo, 1.0)

    def norm(v):  # → [0,1] on shared bounds
        return np.clip((v - lo) / span, 0.0, 1.0)

    # 4. Per-project running HV (on normalized points, ref=(0,0)) + raw Pareto.
    out_projects = []
    for name, f in frames:
        if f is None:
            out_projects.append({"label": name, "missing": True})
            continue
        hv_series = []
        pts_norm: list[tuple[float, float]] = []
        for row in f.itertuples(index=False):
            if not row.valid:
                continue  # errored/invalid evals consume no x (see header comment)
            nv = norm(np.array([row.perf, row.qual]))
            pts_norm.append((float(nv[0]), float(nv[1])))
            hv_series.append({
                "i": len(pts_norm),
                "hv": round(_hypervolume_2d(np.array(pts_norm)), 6),
            })
        valid_pts = f.loc[f.valid, ["perf", "qual"]].values
        front = _pareto_front_2d(valid_pts) if len(valid_pts) else np.empty((0, 2))
        # Front points keep their eval_id (first valid row at those coords) so
        # the UI can deep-link a clicked frontier point to its eval.
        vrows = f.loc[f.valid, ["eval_id", "perf", "qual"]]
        pareto_pts = []
        for perf, qual in sorted(front.tolist()):
            m = vrows[(vrows.perf == perf) & (vrows.qual == qual)]
            pareto_pts.append({
                "perf": float(perf),
                "qual": float(qual),
                "eval_id": (int(m.eval_id.iloc[0]) if len(m) else None),
            })
        # Charts only ever show valid evals — don't ship the invalid ones.
        points = [
            {"eval_id": int(r.eval_id), "perf": float(r.perf),
             "qual": float(r.qual), "valid": True}
            for r in f.itertuples(index=False) if r.valid
        ]
        out_projects.append({
            "label": name,
            "n_init": _read_n_init(root / name),
            "final_hv": round(hv_series[-1]["hv"], 6) if hv_series else 0.0,
            "best_qual": (float(valid_pts[:, 1].max()) if len(valid_pts) else None),
            "best_perf": (float(valid_pts[:, 0].max()) if len(valid_pts) else None),
            "n_valid": int(f.valid.sum()),
            "n_evals": int(len(f)),
            "hv": hv_series,
            "pareto": pareto_pts,
            "points": points,
        })

    return {
        "metric": metric,
        "available_metrics": sorted(available),
        "bounds": {"perf": [float(lo[0]), float(hi[0])], "qual": [float(lo[1]), float(hi[1])]},
        "projects": out_projects,
    }


def _read_n_init(project_dir: Path) -> int:
    """Best-effort init-design size from the run's optimizer params."""
    cfg = project_dir / "rag_stack_config.yaml"
    if not cfg.is_file():
        return 0
    try:
        c = yaml.safe_load(cfg.read_text())
        params = (c.get("optimizer") or {}).get("params") or {}
        return int(params.get("n_both_init", params.get("n_gt_init", 0)))
    except Exception:
        return 0


# ── Search-space tree (visualize what a config expands into) ────────────────
# Reuses the canonical RagSearchSpace.build_ax_searchspace — the SAME builder
# every optimizer consumes — so the tree the UI shows is exactly the space the
# run will search. Pure structural build (no dataset needed).

def _ss_dependents(p) -> dict | None:
    """``ChoiceParameter.dependents`` guarded against the non-hierarchical case
    (Ax raises NotImplementedError rather than returning None)."""
    from ax.core.parameter import ChoiceParameter
    if isinstance(p, ChoiceParameter) and getattr(p, "is_hierarchical", False):
        return p.dependents
    return None


def _serialize_search_space(ss) -> dict:
    """Ax SearchSpace → a JSON tree {tree, summary} for the editor.

    Roots = params that are no other param's dependent. Each ChoiceParameter
    with dependents nests its per-value subtrees; the discrete size is the
    TRUE hierarchical count (a value's branch multiplies only its own active
    children), with RangeParameters counted as one continuous axis.
    """
    from ax.core.parameter import ChoiceParameter, RangeParameter
    params = ss.parameters

    child_names: set = set()
    for p in params.values():
        deps = _ss_dependents(p)
        if deps:
            for lst in deps.values():
                child_names.update(lst)
    roots = [n for n in params if n not in child_names]

    def node(name: str) -> dict:
        p = params[name]
        # Leaf label: drop the SEP-joined branch prefix, keep the readable tail.
        leaf = name.split("__")[-1]
        d: dict = {"name": name, "leaf": leaf}
        if isinstance(p, RangeParameter):
            d.update(kind="range", param_type=p.parameter_type.name,
                     lower=p.lower, upper=p.upper)
        elif isinstance(p, ChoiceParameter):
            d.update(kind="choice", param_type=p.parameter_type.name,
                     values=[str(v) for v in p.values])
            deps = _ss_dependents(p)
            if deps:
                d["dependents"] = {
                    str(v): [node(c) for c in lst] for v, lst in deps.items()
                }
        return d

    def count(name: str) -> int:
        p = params[name]
        deps = _ss_dependents(p)
        if deps:
            total = 0
            for _v, lst in deps.items():
                sub = 1
                for c in lst:
                    sub *= count(c)
                total += sub
            return total
        if isinstance(p, ChoiceParameter):
            return len(p.values)
        return 1  # range = continuous axis, not enumerable

    discrete = 1
    for r in roots:
        discrete *= count(r)
    n_range = sum(1 for p in params.values() if isinstance(p, RangeParameter))
    return {
        "tree": [node(r) for r in roots],
        "summary": {
            "n_params": len(params),
            "n_range": n_range,
            "is_hierarchical": bool(ss.is_hierarchical),
            "discrete_size": int(discrete),
        },
    }


@router.post("/search-space-tree")
def search_space_tree(body: ValidateConfigRequest):
    """Build the Ax search space for a config and return it as a visual tree.

    Returns {tree, summary} on success, or {error} if the config can't build a
    search space (e.g. schema-invalid) — the editor renders the error inline.
    """
    if body.config is not None:
        cfg = body.config
    elif body.content is not None:
        try:
            cfg = yaml.safe_load(body.content)
        except yaml.YAMLError as e:
            return {"error": f"YAML parse error: {e}"}
    else:
        raise HTTPException(400, "Provide 'content' (YAML string) or 'config' (object).")
    if not isinstance(cfg, dict):
        return {"error": "Config must be a mapping/object at the top level."}
    try:
        _warm_static_rag_evaluator()
        from rag_stack.search_space.rag_search_space import RagSearchSpace
        ss = RagSearchSpace.build_ax_searchspace(cfg)
        return _serialize_search_space(ss)
    except Exception as e:  # noqa: BLE001 — surface build failures to the UI
        return {"error": f"{type(e).__name__}: {e}"}


@router.post("/scaffold")
def scaffold_config(body: ScaffoldRequest):
    """Return a known-valid 6-section config template for the given backend.

    The frontend uses this for "New config" and "switch backend" instead of
    hardcoding a template — so the scaffold always matches the current schema.
    """
    from rag_stack.evaluation.registry import get_backend_spec

    try:
        backend = get_backend_spec(body.eval_backend).name
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    if backend == "flashrag":
        cfg = _scaffold_flashrag(body.optimizer_type)
    elif backend == "static_gt":
        cfg = _scaffold_static(body.optimizer_type, body.performance_source)
    else:
        raise HTTPException(
            400,
            f"Backend {backend!r} has no built-in configuration template. "
            "Start from the example supplied by its evaluator plugin.",
        )
    text = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False, width=120)
    return {"content": text, "config": cfg}


# ── Registry endpoints ─────────────────────────────────────────────────────

@router.get("/registry/node-types")
def get_node_types():
    """Return available node types and their modules (derived from support.py)."""
    _warm_static_rag_evaluator()
    from rag_stack_evaluator.static_rag_evaluator.support import get_stage_components
    from rag_stack.pipeline_contract import RETRIEVAL_COMPONENT_PARAMETERS, STAGE_ALIASES
    return {
        "stages": {
            stage: {
                "modules": modules,
                "aliases": [alias for alias, canonical in STAGE_ALIASES.items() if canonical == stage],
                "component_capabilities": {name: _component_capability_metadata(name, registered=True) for name in modules},
                "component_parameters": {name: RETRIEVAL_COMPONENT_PARAMETERS[name]
                                         for name in modules if name in RETRIEVAL_COMPONENT_PARAMETERS},
            }
            for stage, modules in get_stage_components().items()
        }
    }


@router.get("/registry/vectordb-types")
def get_vectordb_types_endpoint():
    """Return available vector database types (derived from vectordb registry)."""
    _warm_static_rag_evaluator()
    from rag_stack_evaluator.static_rag_evaluator.support import get_vectordb_types
    return {"vectordb_types": get_vectordb_types()}


@router.get("/taxonomy/stages")
def get_taxonomy_stages():
    """The central naming taxonomy, serialized: every user stage with its
    selectable components and the PerformanceStages they produce. Replaces the
    old /registry/stage-node-mapping endpoint (a RAGO-stage inversion)."""
    from rag_stack.taxonomy import TAXONOMY, USER_STAGES

    return {
        "stages": {
            stage.value: {
                "components": [c.name for c in TAXONOMY.components_of(stage)],
                # Per-component detail: whether the CM prices it (shape is not
                # None) and which PerformanceStages it produces — lets the UI
                # render the dispatch flow from truth instead of hardcoding.
                "component_details": [
                    {
                        "name": c.name,
                        "priced": c.shape is not None,
                        "capabilities": _component_capability_metadata(c.name),
                        "performance_stages": [
                            ps.name
                            for ps in TAXONOMY.performance_stages_of_component(c.name)
                        ],
                    }
                    for c in TAXONOMY.components_of(stage)
                ],
                "performance_stages": sorted({
                    ps.name
                    for c in TAXONOMY.components_of(stage)
                    for ps in TAXONOMY.performance_stages_of_component(c.name)
                }),
            }
            for stage in USER_STAGES
        }
    }


@router.get("/registry/pipeline-methods")
def get_pipeline_methods(backend: str = "flashrag"):
    """Registered pipeline modes; FlashRAG remains the editor's default."""
    from rag_stack.evaluation.registry import get_backend_spec, supported_pipeline_modes

    try:
        spec = get_backend_spec(backend)
        return {"methods": list(supported_pipeline_modes(spec.name))}
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/registry/generator-backends")
def get_generator_backends():
    """Valid generator `component`/backend names (GENERATOR_BACKENDS)."""
    _warm_static_rag_evaluator()
    from rag_stack_evaluator.static_rag_evaluator.nodes.generator.registry import GENERATOR_BACKENDS
    return {"backends": sorted(GENERATOR_BACKENDS.keys())}


@router.get("/registry/metrics")
def get_metrics(backend: str = "static_gt"):
    """Valid metric names for a quality backend. Mirrors the exact vocabulary
    ConfigValidator accepts, so the editor never drifts from validation."""
    from rag_stack.evaluation.registry import get_backend_spec, supported_metrics

    try:
        backend = get_backend_spec(backend).name
        if backend == "static_gt":
            _warm_static_rag_evaluator()
        return {"backend": backend, "requires_model": backend == "static_gt",
                "metrics": list(supported_metrics(backend))}
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


# Chunker `component` values (llama_index/langchain splitter names). There is
# no deeper Python enum for these — this curated list is the source of truth,
# centralized here so the editor stops hardcoding it.
_CHUNKER_TYPES = [
    "recursivecharacter", "character", "token", "sentence", "sentencewindow",
    "semantic_llama_index", "semanticdoublemerging",
    "sentencetransformerstoken", "konlpy", "simplefile",
]


@router.get("/registry/chunker-types")
def get_chunker_types():
    """Chunker component values for algo_search_space.corpus.chunker."""
    return {"chunker_types": list(_CHUNKER_TYPES)}


@router.get("/registry/enums")
def get_config_enums():
    """Config switches, with evaluator names discovered from the plugin registry."""
    from rag_stack.evaluation.registry import available_backends

    return {
        "eval_backend": list(available_backends()),
        "performance_source": ["cost_model", "measured"],
        "rag_ir_mode": ["dynamic", "static"],
        "combined_quality_mode": ["mean"],
        "performance_selection": ["max_throughput"],
    }


@router.get("/registry/dataflows")
def get_dataflow_registry():
    """`pipeline.rag_dataflow` vocabulary for the config editor.

    ``branches`` mirrors the ConfigValidator's allowed static_gt co-mode set;
    ``react_active_stages`` is the react engine contract (the stages the loop
    actually drives each round — everything else is stripped from react
    trials), sourced from the search-space builder so the UI never drifts.
    ``react_required_knobs`` are the branch sub-knobs the validator requires.
    """
    # The search-space module imports evaluator schema/vectordb metadata. It
    # must join the same cold-import guard as the other registry endpoints.
    _warm_static_rag_evaluator()
    from rag_stack.search_space.rag_search_space import REACT_ACTIVE_STAGES

    return {
        "branches": ["sequential", "react"],
        "react_active_stages": sorted(REACT_ACTIVE_STAGES),
        "react_required_knobs": ["max_iter"],
    }


# ── Cost model mapping ───────────────────────────────────────────────────────

@router.get("/registry/cost-model-mapping")
def get_cost_model_mapping():
    """Return the (stage, component) → cost_model dispatch bindings, READ-ONLY.

    The bindings are code (rag_stack/cost_model/dispatch.py), validated against
    the taxonomy at import — the old JSON-backed live-edit PUT is retired. The
    flat mapping is a compatibility view. The complete stage-aware contract,
    including retrieval encoding and unsupported components, is exposed by
    ``/api/cost-model/catalog``; Apps must consume that catalog."""
    from rag_stack.cost_model.dispatch import component_binding_table
    from rag_stack.cost_model import COST_MODEL_REGISTRY

    return {
        "mapping": component_binding_table(),
        "available_cost_models": list(COST_MODEL_REGISTRY.keys()),
        "read_only": True,
        "catalog_url": "/api/cost-model/catalog",
    }


# ── Generator client health check ────────────────────────────────────────────

@router.get("/health/claude-code")
def check_claude_code(model: str = "haiku"):
    """Check if Claude Code SDK is available and working."""
    import asyncio

    result = {"installed": False, "working": False, "error": None, "model": model}

    try:
        from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
        result["installed"] = True
    except ImportError as e:
        result["error"] = f"claude_agent_sdk not installed: {e}"
        return result

    async def _test():
        options = ClaudeAgentOptions(
            model=model,
            permission_mode="default",
            max_turns=1,
            effort="low",
        )
        text = ""
        async for msg in query(prompt="Say 'ok' and nothing else.", options=options):
            if isinstance(msg, AssistantMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        text += block.text
        return text.strip()

    try:
        response = asyncio.run(_test())
        result["working"] = True
        result["model"] = "sonnet"
        result["response"] = response
    except Exception as e:
        result["error"] = str(e)

    return result


@router.get("/health/codex-cli")
def check_codex_cli(model: str = "gpt-5.4"):
    """Check if OpenAI Codex CLI (`codex exec`) is available and working."""
    import asyncio
    import shutil

    result = {"installed": False, "working": False, "error": None, "model": model}

    codex_bin = shutil.which("codex")
    if not codex_bin:
        result["error"] = "codex CLI not found on PATH"
        return result
    result["installed"] = True

    try:
        from rag_stack.ai_clients.codex_cli_client import CodexCliClient
    except ImportError as e:
        result["error"] = f"CodexCliClient import failed: {e}"
        return result

    async def _test():
        client = CodexCliClient(model=model, timeout_s=120)
        text = await client.chat(
            [{"role": "user", "content": "Say 'ok' and nothing else."}]
        )
        return text.strip()

    try:
        response = asyncio.run(_test())
        result["working"] = True
        result["response"] = response
    except Exception as e:
        result["error"] = str(e)

    return result


# Isolated CM previews own their jobs and artifacts, outside optimizer state.
from .cm_preview import router as cm_preview_router
router.include_router(cm_preview_router)
