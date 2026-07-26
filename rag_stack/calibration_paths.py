"""Filesystem contract for the frozen cost-model artifacts.

``calibration_data`` holds frozen measurements; ``calibration_profiles``
holds the runtime profiles produced by fitting those measurements.  All
calibration commands resolve those roots here so a new-machine campaign can
run in an isolated staging directory before anything is promoted.

This module intentionally lives above :mod:`rag_stack.cost_model`: importing
path/status helpers must not trigger cost-model auto-discovery or CUDA imports.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping


CALIBRATION_DATA_DIR_ENV = "RAG_STACK_CALIBRATION_DATA_DIR"
CALIBRATION_PROFILE_DIR_ENV = "RAG_STACK_CALIBRATION_PROFILE_DIR"

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_root(value: str | os.PathLike[str] | None, default: str) -> Path:
    raw = Path(value).expanduser() if value is not None else Path(default)
    if not raw.is_absolute():
        raw = _REPO_ROOT / raw
    return raw.resolve()


def calibration_data_root(
    value: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the frozen-measurement root as an absolute path."""
    if value is None:
        value = os.environ.get(CALIBRATION_DATA_DIR_ENV)
    return _resolve_root(value, "calibration_data")


def calibration_profile_root(
    value: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the generated-profile root as an absolute path."""
    if value is None:
        value = os.environ.get(CALIBRATION_PROFILE_DIR_ENV)
    return _resolve_root(value, "calibration_profiles")


def configure_calibration_roots(
    *,
    data_dir: str | os.PathLike[str] | None = None,
    profile_dir: str | os.PathLike[str] | None = None,
    relative_to: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path]:
    """Configure process-local roots and return ``(data_root, profile_root)``.

    Relative CLI paths are resolved against the caller's working directory
    before the calibrate dispatcher changes to the repository root.
    """
    data_root, profile_root = resolve_calibration_roots(
        data_dir=data_dir, profile_dir=profile_dir, relative_to=relative_to
    )
    if data_dir is not None:
        os.environ[CALIBRATION_DATA_DIR_ENV] = str(data_root)
    if profile_dir is not None:
        os.environ[CALIBRATION_PROFILE_DIR_ENV] = str(profile_root)
    return data_root, profile_root


def resolve_calibration_roots(
    *,
    data_dir: str | os.PathLike[str] | None = None,
    profile_dir: str | os.PathLike[str] | None = None,
    relative_to: str | os.PathLike[str] | None = None,
) -> tuple[Path, Path]:
    """Resolve roots without changing process environment."""
    base = Path(relative_to or os.getcwd()).resolve()

    def absolute(
        value: str | os.PathLike[str] | None,
        env_name: str,
        default: str,
    ) -> Path:
        if value is None:
            # Defaults and process-wide overrides keep the historical
            # repository-root-relative contract. Only an explicit CLI path is
            # caller-CWD-relative.
            return _resolve_root(os.environ.get(env_name), default)
        raw = value
        path = Path(raw).expanduser()
        return (base / path).resolve() if not path.is_absolute() else path.resolve()

    return (
        absolute(data_dir, CALIBRATION_DATA_DIR_ENV, "calibration_data"),
        absolute(profile_dir, CALIBRATION_PROFILE_DIR_ENV, "calibration_profiles"),
    )


def frozen_measurement_path(component: str, hardware_key: str) -> Path:
    return calibration_data_root() / component / f"{hardware_key}.parquet"


def system_profile_path(component: str, hardware_key: str) -> Path:
    return calibration_profile_root() / component / f"{hardware_key}.json"


def atomic_write_json(path: str | os.PathLike[str], document: Mapping[str, Any]) -> Path:
    """Atomically replace one JSON profile without leaving a partial file.

    This changes persistence safety only; profile keys and values are written
    exactly as supplied by the existing fitters.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = (target.stat().st_mode & 0o777) if target.exists() else 0o644
    fd, temporary = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(dict(document), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return target
