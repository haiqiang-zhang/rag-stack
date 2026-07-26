"""Runtime calibration policy for the cost model.

The default remains strict production pricing.  ``uncalibrated_spec`` is an
explicit hardware-spec exploration mode: an individual component may use its
exact calibration when one exists, or its analytical hardware-spec path when
the exact record is absent.  Callers must not use this switch to hide a
present-but-invalid profile.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


STRICT_CALIBRATION_MODE = "strict"
UNCALIBRATED_SPEC_MODE = "uncalibrated_spec"
CALIBRATION_MODES = frozenset({
    STRICT_CALIBRATION_MODE,
    UNCALIBRATED_SPEC_MODE,
})
RAG_CM_FIELDS = frozenset({"calibration_mode"})


def _validate_rag_cm(rag_cm: Mapping[str, Any], *, source: str) -> None:
    unsupported = sorted(set(rag_cm) - RAG_CM_FIELDS)
    if unsupported:
        raise ValueError(
            f"{source} contains fields unsupported by cost_model: "
            f"{unsupported}. Remove them; the CM reads the canonical "
            "calibration_profiles root (or the process-wide "
            "RAG_STACK_CALIBRATION_PROFILE_DIR override)."
        )


def resolve_calibration_mode(
    pipeline_config: Mapping[str, Any] | None,
    hardware: Any | None = None,
) -> str:
    """Return the exact calibration mode for this pricing target.

    A replay target may carry its own policy in ``hardware.rag_cm``.  When that
    value is not ``None`` (including an empty mapping), it owns the complete
    calibration-policy decision.  Only a target without ``rag_cm`` inherits
    ``pipeline_config.system.rag_cm``.  Missing policy preserves strict mode;
    invalid containers or values fail closed rather than entering spec mode.
    """

    hardware_rag_cm = None
    if hardware is not None:
        if isinstance(hardware, Mapping):
            hardware_rag_cm = hardware.get("rag_cm")
        else:
            hardware_rag_cm = getattr(hardware, "rag_cm", None)

    if hardware_rag_cm is not None:
        rag_cm = hardware_rag_cm
        source = "hardware.rag_cm"
    else:
        config = pipeline_config or {}
        system = config.get("system")
        if system is None:
            return STRICT_CALIBRATION_MODE
        if not isinstance(system, Mapping):
            raise ValueError("system must be a mapping")
        rag_cm = system.get("rag_cm")
        if rag_cm is None:
            return STRICT_CALIBRATION_MODE
        source = "system.rag_cm"

    if not isinstance(rag_cm, Mapping):
        raise ValueError(f"{source} must be a mapping")
    _validate_rag_cm(rag_cm, source=source)
    value = rag_cm.get("calibration_mode", STRICT_CALIBRATION_MODE)
    if not isinstance(value, str):
        raise ValueError(
            f"{source}.calibration_mode must be one of "
            f"{sorted(CALIBRATION_MODES)}"
        )
    mode = value
    if mode not in CALIBRATION_MODES:
        raise ValueError(
            f"{source}.calibration_mode must be one of "
            f"{sorted(CALIBRATION_MODES)}; got {value!r}"
        )
    return mode


def is_uncalibrated_spec_mode(
    pipeline_config: Mapping[str, Any] | None,
    hardware: Any | None = None,
) -> bool:
    """Whether missing component profiles may use analytical spec pricing."""

    return (
        resolve_calibration_mode(pipeline_config, hardware)
        == UNCALIBRATED_SPEC_MODE
    )


__all__ = [
    "CALIBRATION_MODES",
    "RAG_CM_FIELDS",
    "STRICT_CALIBRATION_MODE",
    "UNCALIBRATED_SPEC_MODE",
    "is_uncalibrated_spec_mode",
    "resolve_calibration_mode",
]
