"""Strict envelope for IVF workload-imbalance profiles.

The payload remains the existing ``d``/``N``/``nlist`` mapping consumed by
the cost model.  The serialized boundary has one current shape.
"""

from __future__ import annotations

from typing import Any

IVF_IMBALANCE_ENVELOPE_FIELDS = frozenset({"profile"})


def unwrap_ivf_imbalance_profile(
    document: Any,
    *,
    source: str = "IVF imbalance profile",
) -> dict:
    """Validate the exact current envelope and return its payload."""

    if not isinstance(document, dict):
        raise ValueError(f"{source}: envelope must be a JSON object")
    if set(document) != IVF_IMBALANCE_ENVELOPE_FIELDS:
        raise ValueError(
            f"{source}: envelope fields must be exactly "
            "['profile']"
        )
    profile = document["profile"]
    if not isinstance(profile, dict):
        raise ValueError(f"{source}: profile payload must be a JSON object")
    return profile


def wrap_ivf_imbalance_profile(profile: Any) -> dict:
    """Return the strict global envelope for an existing payload mapping."""

    if not isinstance(profile, dict):
        raise ValueError("IVF imbalance profile payload must be a JSON object")
    document = {"profile": profile}
    # Keep the writer on the same strict validation path as the reader.
    unwrap_ivf_imbalance_profile(document)
    return document


__all__ = [
    "IVF_IMBALANCE_ENVELOPE_FIELDS",
    "unwrap_ivf_imbalance_profile",
    "wrap_ivf_imbalance_profile",
]
