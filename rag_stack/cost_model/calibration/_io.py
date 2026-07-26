"""Small, dependency-free persistence primitives for calibration artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from .errors import CalibrationIntegrityError, CalibrationValidationError


def canonical_json_bytes(value: Any) -> bytes:
    """Return the unique UTF-8 encoding used for every addressed JSON object."""
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CalibrationValidationError(
            "calibration metadata must be finite, JSON-serializable data"
        ) from exc
    return text.encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_sha256(value: object, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise CalibrationValidationError(f"{field} must be a lowercase sha256")
    return text


def require_logical_key(value: object, *, field: str = "logical_key") -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationValidationError(f"{field} must be a non-empty string")
    if value != value.strip() or "\\" in value or value.startswith("/"):
        raise CalibrationValidationError(f"{field} must be a relative POSIX key")
    parts = PurePosixPath(value).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise CalibrationValidationError(f"{field} contains an unsafe path segment")
    return value


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_replace_bytes(path: Path, payload: bytes) -> None:
    """Durably replace ``path`` using temp + fsync + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _lock_path(root: Path, key: str) -> Path:
    digest = sha256_bytes(key.encode("utf-8"))
    return root / ".locks" / "sha256" / digest[:2] / f"{digest}.lock"


@contextmanager
def key_lock(root: Path, key: str) -> Iterator[None]:
    """Hold a cross-process exclusive lock for one publication key."""
    if not isinstance(key, str) or not key:
        raise CalibrationValidationError("lock key must be a non-empty string")
    path = _lock_path(root, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def publish_immutable(path: Path, payload: bytes, *, root: Path, key: str) -> None:
    """Create one addressed object, or verify an identical existing object."""
    with key_lock(root, f"immutable:{key}"):
        if path.exists():
            existing = path.read_bytes()
            if existing != payload:
                raise CalibrationIntegrityError(
                    f"immutable calibration object changed at {path}"
                )
            return
        atomic_replace_bytes(path, payload)


def read_json_mapping(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationIntegrityError(
            f"cannot read {description} at {path}"
        ) from exc
    if not isinstance(value, dict):
        raise CalibrationIntegrityError(f"{description} must be a JSON object")
    return value
