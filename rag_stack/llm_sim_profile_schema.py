"""Strict envelope for CM LLM component profiles.

Serialized profiles have one current, unversioned shape; readers do not
negotiate or fall back to historical shapes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LLM_SIM_PROFILE_ENVELOPE_FIELDS = frozenset({"profiles"})


def unwrap_llm_sim_profile_document(
    document: object,
    *,
    source: str = "LLM component profile",
) -> dict[str, Any]:
    """Validate the strict current envelope and return its record map.

    There is deliberately no legacy bare-map fallback or version negotiation.
    Returning the actual nested dictionary lets a validated writer update it
    before wrapping it again for an atomic replacement.
    """

    if not isinstance(document, Mapping):
        raise ValueError(f"{source} root must be a JSON object")
    actual_fields = set(document)
    if actual_fields != LLM_SIM_PROFILE_ENVELOPE_FIELDS:
        missing = sorted(LLM_SIM_PROFILE_ENVELOPE_FIELDS - actual_fields)
        extra = sorted(actual_fields - LLM_SIM_PROFILE_ENVELOPE_FIELDS)
        raise ValueError(
            f"{source} envelope fields must be exactly "
            f"{sorted(LLM_SIM_PROFILE_ENVELOPE_FIELDS)!r}; "
            f"missing={missing!r}, extra={extra!r}"
        )
    profiles = document.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"{source}.profiles must be a JSON object")
    return profiles


def wrap_llm_sim_profiles(profiles: Mapping[str, Any]) -> dict[str, Any]:
    """Create the only serialized envelope accepted by production."""

    if not isinstance(profiles, Mapping):
        raise ValueError("LLM component profiles must be a mapping")
    return {"profiles": dict(profiles)}


__all__ = [
    "LLM_SIM_PROFILE_ENVELOPE_FIELDS",
    "unwrap_llm_sim_profile_document",
    "wrap_llm_sim_profiles",
]
