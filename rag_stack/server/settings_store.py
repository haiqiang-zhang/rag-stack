"""Simple JSON-file persistence for global application settings."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .schemas import AppSettings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_FILE = Path(
    os.environ.get("RAG_STACK_SETTINGS_PATH", "")
) if os.environ.get("RAG_STACK_SETTINGS_PATH") else _PROJECT_ROOT / ".rag_stack_settings.json"
_DEFAULT_ENV_FILE = _PROJECT_ROOT / ".env"


def _get_env_file() -> Path:
    """Return the .env file path from settings, falling back to project root."""
    settings = _load_settings_raw()
    custom = settings.get("env_file_path", "").strip()
    if custom:
        p = Path(custom)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        return p
    return _DEFAULT_ENV_FILE


def _load_settings_raw() -> dict:
    """Load raw settings dict without parsing into AppSettings."""
    if _SETTINGS_FILE.is_file():
        with open(_SETTINGS_FILE) as f:
            return json.load(f)
    return {}


def load_settings() -> AppSettings:
    """Load settings from disk, returning defaults if file missing."""
    if _SETTINGS_FILE.is_file():
        with open(_SETTINGS_FILE) as f:
            return AppSettings(**json.load(f))
    return AppSettings()


def save_settings(settings: AppSettings) -> None:
    """Persist settings to disk."""
    with open(_SETTINGS_FILE, "w") as f:
        json.dump(settings.model_dump(), f, indent=2)


# ── .env file management ──────────────────────────────────────────────────


def load_env_keys() -> dict[str, str]:
    """Read key=value pairs from .env file."""
    env_file = _get_env_file()
    keys: dict[str, str] = {}
    if not env_file.is_file():
        return keys
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        name, _, value = line.partition("=")
        # Strip surrounding quotes
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        keys[name.strip()] = value
    return keys


def save_env_keys(keys: dict[str, str]) -> None:
    """Write key=value pairs to .env file and apply to os.environ."""
    env_file = _get_env_file()
    lines = []
    for name, value in sorted(keys.items()):
        if " " in value or "#" in value or "'" in value:
            lines.append(f'{name}="{value}"')
        else:
            lines.append(f"{name}={value}")
    env_file.write_text("\n".join(lines) + "\n" if lines else "")
    # Apply to current process
    for name, value in keys.items():
        if value:
            os.environ[name] = value
        elif name in os.environ:
            del os.environ[name]


def apply_env_keys() -> None:
    """Load .env and set all keys in os.environ. Call on startup."""
    for name, value in load_env_keys().items():
        if value:
            os.environ[name] = value
