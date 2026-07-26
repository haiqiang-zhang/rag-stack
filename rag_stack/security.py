"""Single source of truth for secret-stripping when persisting configs.

Pipeline configs in rag_stack carry live API keys (`api_key`, `api_secret`,
`token`, ...) because YAML `${OPENROUTER_API_KEY}` references get expanded
at config-resolution time so downstream nodes can call OpenRouter / OpenAI /
etc. We must NOT serialize those expanded values to disk in artifacts the
user may share, commit, or browse from the dashboard.

This module exposes:

  * :data:`SECRET_KEYS` — the canonical set of secret-bearing keys.
  * :func:`strip_secrets` — universal sanitizer. Accepts ANY value
    (dict / list / tuple / str / scalar) and returns a sanitized copy.
    Does both layers of defense in one call: structural strip of
    secret-keyed dict entries AND regex scrub of provider API-key
    patterns embedded in strings. Recursive on containers.
  * :func:`safe_json_dump` / :func:`safe_yaml_dump` /
    :func:`safe_csv_row` / :func:`safe_dataframe_to_csv` — drop-in
    writers that pipe everything through :func:`strip_secrets` first.

Add new persistence sites by importing one of the helpers above instead of
hand-rolling serialization. Single chokepoint = single audit.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Set

# Keys whose VALUES must never be persisted. Match-by-key, recursive.
SECRET_KEYS: Set[str] = frozenset({
    "api_key",
    "api_secret",
    "token",
    "password",
    "secret",
})

# Defense-in-depth: scrub these provider API-key formats from free-form
# strings even if they ended up inside a non-secret-keyed field.
_SECRET_PATTERNS = (
    re.compile(r"sk-or-v1-[A-Za-z0-9]{32,}"),       # OpenRouter
    re.compile(r"sk-proj-[A-Za-z0-9_\-]{40,}"),      # OpenAI project keys
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{40,}"),       # Anthropic
    re.compile(r"sk-[A-Za-z0-9]{40,}"),              # OpenAI legacy + generic
)
REDACTED = "<REDACTED>"


def strip_secrets(obj: Any) -> Any:
    """Return a sanitized copy of ``obj`` covering both leak surfaces.

    Two layers of defense applied in a single recursive pass:

    1. **Structural** — drop any dict entry whose key is in
       :data:`SECRET_KEYS` (e.g. ``api_key``, ``token``).
    2. **Pattern** — replace any provider API-key string
       (``sk-or-v1-…``, ``sk-proj-…``, ``sk-ant-…``, generic ``sk-…``)
       embedded inside a string value with :data:`REDACTED`. Catches keys
       that ended up in non-secret-keyed fields (e.g. inside a prompt
       template, an error message, a serialized blob).

    Handles every value type:

      * ``dict`` / ``list`` / ``tuple`` → recursive sanitize, returning a
        fresh container of the same type.
      * ``str`` → regex scrub.
      * Other scalars (numbers, booleans, ``None``, NaN, custom objects) →
        pass through unchanged.

    The original ``obj`` is not mutated.
    """
    if isinstance(obj, dict):
        return {
            k: strip_secrets(v)
            for k, v in obj.items()
            if k not in SECRET_KEYS
        }
    if isinstance(obj, list):
        return [strip_secrets(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(strip_secrets(item) for item in obj)
    if isinstance(obj, str):
        cleaned = obj
        for pat in _SECRET_PATTERNS:
            cleaned = pat.sub(REDACTED, cleaned)
        return cleaned
    return obj


def safe_json_dump(obj: Any, fp, **kwargs) -> None:
    """``json.dump`` that strips secrets first. Use everywhere we persist
    pipeline-config-shaped objects to JSON.
    """
    json.dump(strip_secrets(obj), fp, **kwargs)


def safe_yaml_dump(obj: Any, fp, **kwargs) -> None:
    """``yaml.safe_dump`` that strips secrets first. Imports yaml lazily so
    this module stays import-cheap.
    """
    import yaml
    yaml.safe_dump(strip_secrets(obj), fp, **kwargs)


def safe_csv_row(row: Iterable[Any]) -> list:
    """Apply :func:`strip_secrets` cell-by-cell. Use when writing a CSV row
    that may contain free-form values originating from a config (e.g.
    serialized ``module_params`` dicts or stringified blobs).
    """
    return [strip_secrets(cell) for cell in row]


def safe_dataframe_to_csv(df, path: str, **kwargs) -> None:
    """``DataFrame.to_csv`` that strips secrets first.

    Per-node ``summary.csv`` files in
    ``static_rag_evaluator/nodes/*/run.py`` write a
    ``module_params`` column that holds the live module config dict —
    including ``api_key``. :func:`strip_secrets` handles every cell type:
    dict/list/tuple → recursive structural strip, string → pattern scrub,
    scalar → pass-through.

    Use this in any path that persists a DataFrame whose cells could
    transitively reference a config.
    """
    sanitized = df.copy()
    for col in sanitized.columns:
        sanitized[col] = sanitized[col].map(strip_secrets)
    sanitized.to_csv(path, **kwargs)
