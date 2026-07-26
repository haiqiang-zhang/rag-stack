"""Durable configuration ownership shared by project APIs and Python callers.

The first launch retains the original YAML text, before environment expansion
or runtime enrichment. A reset may remove results, but must preserve this
snapshot. Validation compares the text itself; no experiment checksums are
needed. All cooperating writers use the same reentrant, cross-process lock.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Iterator

import yaml

from rag_stack.project_summary import CONFIG_FILENAME

CONFIG_FREEZE_FILENAME = "config.freeze.json"
_SCHEMA_VERSION = "rag-stack.project-config-freeze.v1"
_READ_ONLY_REASON = (
    "Configuration is read-only after the first run has started. "
    "Create a new project to change it."
)
_HISTORY_ARTIFACTS = (
    "evaluations", "quality_store", "pareto_configs", "optimizer", "registry",
    "resources", "all_evaluations.csv", "pareto_frontier.csv",
    "invalid_configs.json", "system_transfer_result.json",
    "system_transfer_state.json", "system_transfer.json",
)
_SUMMARY_COUNTERS = (
    "n_evaluations", "gt_count", "cm_count", "valid_eval_count",
    "n_invalid_configs", "n_assembly_results", "n_pareto_points", "total_cost",
)
_locks_guard = threading.Lock()
_locks: dict[str, threading.RLock] = {}
_local = threading.local()


class ProjectConfigError(ValueError):
    """A requested project configuration change violates run ownership."""


def _after_fork() -> None:
    # A forked child must not inherit another thread's acquired Python lock or
    # mistake the parent's nested lock depth for a lock it owns itself.
    global _locks_guard, _locks, _local
    _locks_guard = threading.Lock()
    _locks = {}
    _local = threading.local()


os.register_at_fork(after_in_child=_after_fork)


@contextmanager
def project_config_lock(project_dir: str | Path) -> Iterator[None]:
    """Serialize project launch/config changes, allowing nested helper calls."""
    project = Path(project_dir).resolve()
    key = str(project)
    with _locks_guard:
        thread_lock = _locks.setdefault(key, threading.RLock())
    with thread_lock:
        held = getattr(_local, "held", None)
        if held is None:
            held = _local.held = set()
        if key in held:
            yield
            return
        project.mkdir(parents=True, exist_ok=True)
        with (project / ".project.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            held.add(key)
            try:
                yield
            finally:
                held.remove(key)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def project_config_read_only_reason(project_dir: str | Path) -> str | None:
    """Return why a project is immutable; a pristine CLI directory is editable.

    History checks also cover projects created before durable snapshots were
    introduced. A missing summary alone does not prevent first CLI creation.
    """
    project = Path(project_dir)
    if (project / CONFIG_FREEZE_FILENAME).exists():
        return _READ_ONLY_REASON
    summary_path = project / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError):
            return "Project lifecycle state is unavailable; configuration is read-only."
        if not isinstance(summary, dict) or summary.get("status") not in {
            "created", "running", "completed", "failed", "quality_complete",
        }:
            return "Project lifecycle state is unavailable; configuration is read-only."
        if summary["status"] != "created" or summary.get("pid") is not None:
            return _READ_ONLY_REASON
        if any(summary.get(key) for key in _SUMMARY_COUNTERS):
            return "This project has evaluation history; its configuration is read-only."
    if any((project / name).exists() for name in _HISTORY_ARTIFACTS):
        return "This project has run or optimizer history; its configuration is read-only."
    for name, label in (("logs", "previous run logs"), ("data", "initialized run data")):
        directory = project / name
        if directory.is_dir() and any(directory.iterdir()):
            return f"This project has {label}; its configuration is read-only."
    return None


def _read_raw(path: Path) -> str:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            return stream.read()
    except (OSError, UnicodeError) as exc:
        raise ProjectConfigError(f"Project configuration is unavailable: {path}") from exc


def _validate_mapping(raw_content: str) -> None:
    if not isinstance(raw_content, str):
        raise ProjectConfigError("Project configuration must be YAML text.")
    try:
        document = yaml.safe_load(raw_content)
    except yaml.YAMLError as exc:
        raise ProjectConfigError(f"Project configuration is invalid YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise ProjectConfigError("Project configuration must be a YAML mapping.")


def _atomic_write(path: Path, raw_content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".config.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(raw_content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def verify_project_config(project_dir: str | Path) -> str:
    """Read the project YAML, rejecting changes against an existing snapshot."""
    project = Path(project_dir)
    with project_config_lock(project):
        raw_content = _read_raw(project / CONFIG_FILENAME)
        snapshot = project / CONFIG_FREEZE_FILENAME
        if snapshot.exists():
            try:
                frozen = json.loads(snapshot.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise ProjectConfigError("Frozen project configuration is unreadable.") from exc
            if (
                not isinstance(frozen, dict)
                or frozen.get("schema_version") != _SCHEMA_VERSION
                or not isinstance(frozen.get("raw_content"), str)
                or not isinstance(frozen.get("frozen_at"), str)
            ):
                raise ProjectConfigError("Frozen project configuration has an invalid contract.")
            if raw_content != frozen["raw_content"]:
                raise ProjectConfigError(
                    "Project configuration has changed since its first run started. "
                    "Restore the frozen configuration or create a new project."
                )
        _validate_mapping(raw_content)
        return raw_content


def write_project_config(project_dir: str | Path, raw_content: str) -> str:
    """Atomically save a draft; identical frozen content is a verified no-op."""
    project = Path(project_dir)
    with project_config_lock(project):
        reason = project_config_read_only_reason(project)
        config = project / CONFIG_FILENAME
        frozen = (project / CONFIG_FREEZE_FILENAME).exists()
        if config.exists() or frozen:
            # A broken draft can be repaired. Existing freezes are always
            # checked before considering even an identical-content no-op.
            current = verify_project_config(project) if frozen else _read_raw(config)
            if raw_content == current:
                _validate_mapping(raw_content)
                return current
        if reason:
            raise ProjectConfigError(reason)
        _validate_mapping(raw_content)
        _atomic_write(config, raw_content)
        return raw_content


def freeze_project_config(project_dir: str | Path) -> str:
    """Retain the input YAML before first launch and validate every later use.

    Existing projects without a snapshot adopt their retained config on the
    next launch. This cannot establish what that file contained historically.
    """
    project = Path(project_dir)
    with project_config_lock(project):
        raw_content = verify_project_config(project)
        snapshot = project / CONFIG_FREEZE_FILENAME
        if not snapshot.exists():
            _atomic_write(snapshot, json.dumps({
                "schema_version": _SCHEMA_VERSION,
                "frozen_at": datetime.now(timezone.utc).isoformat(),
                "raw_content": raw_content,
            }, indent=2) + "\n")
        return raw_content
