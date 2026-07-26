"""Lightweight schema contract for exact-CPU FAISS HNSW profiles.

This module intentionally does not import :mod:`rag_stack.cost_model`.  The
metadata-only calibration inventory and the production HNSW loader both use
the same validator, so ``calibrate status`` cannot accept a document that the
runtime later interprets differently.

Raw CPU facts (core count, FLOP/s, cache hierarchy, and measured STREAM
bandwidth) remain in the CPU preset or inline ``system.cpu`` configuration.
The profile is self-contained for every *non-raw* HNSW runtime parameter:
CPU-specific effective hardware-model inputs are named separately from the
four empirical graph-search coefficients fitted on standalone FAISS HNSW
measurements.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping


HNSW_PROFILE_KIND = "faiss_hnsw_exact_cpu"

HNSW_ROOT_FIELDS = frozenset({
    "profile_kind",
    "hardware_key",
    "hardware_model_parameters",
    "empirical_coefficients",
    "fit_metadata",
})
HNSW_HARDWARE_MODEL_PARAMETER_FIELDS = (
    "simd_efficiency",
    "dram_random_access_latency_ns",
    "scattered_bw_ceiling",
    "numa_local_bytes",
    "numa_remote_bw_ceiling",
)
HNSW_EMPIRICAL_COEFFICIENT_FIELDS = (
    "hops_per_upper_layer",
    "cache_random_access_latency_multiplier",
    "cycles_per_visited_probe",
    "cycles_per_heap_op",
)
HNSW_FIT_METADATA_FIELDS = frozenset({
    "mape_latency_pct",
    "max_ape_latency_pct",
    "num_trials",
    "dataset_configs",
})
_HARDWARE_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def is_hnsw_hardware_key(value: Any) -> bool:
    """Whether ``value`` is a safe, canonical profile filename stem."""

    return bool(isinstance(value, str) and _HARDWARE_KEY_RE.fullmatch(value))


def _fields_error(
    value: Any,
    expected: frozenset[str],
    *,
    field: str,
) -> str | None:
    if not isinstance(value, Mapping):
        return f"{field} must be a JSON object"
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        return f"{field} fields mismatch; missing={missing}, extra={extra}"
    return None


def _finite_number(
    value: Any,
    *,
    positive: bool = False,
    maximum: float | None = None,
) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    if not math.isfinite(number):
        return False
    if positive and number <= 0.0:
        return False
    if not positive and number < 0.0:
        return False
    return maximum is None or number <= maximum


def hnsw_profile_validation_error(
    document: Any,
    expected_hardware_key: str | None = None,
) -> str | None:
    """Return the first contract error, or ``None`` for a current profile.

    Validation is deliberately exact at every object level.  There is no
    legacy field backfill, unknown-key filtering, or implicit dataclass default.
    """

    error = _fields_error(document, HNSW_ROOT_FIELDS, field="profile root")
    if error:
        return error
    assert isinstance(document, Mapping)

    if document.get("profile_kind") != HNSW_PROFILE_KIND:
        return f"profile_kind must be {HNSW_PROFILE_KIND!r}"
    hardware_key = document.get("hardware_key")
    if not is_hnsw_hardware_key(hardware_key):
        return (
            "hardware_key must be a canonical filename-safe CPU key "
            "([A-Za-z0-9][A-Za-z0-9_.-]*)"
        )
    if expected_hardware_key is not None and hardware_key != expected_hardware_key:
        return (
            f"hardware_key {hardware_key!r} does not match requested CPU "
            f"{expected_hardware_key!r}"
        )

    hardware = document.get("hardware_model_parameters")
    error = _fields_error(
        hardware,
        frozenset(HNSW_HARDWARE_MODEL_PARAMETER_FIELDS),
        field="hardware_model_parameters",
    )
    if error:
        return error
    assert isinstance(hardware, Mapping)
    for field in (
        "simd_efficiency",
        "scattered_bw_ceiling",
        "numa_remote_bw_ceiling",
    ):
        if not _finite_number(hardware.get(field), positive=True, maximum=1.0):
            return f"hardware_model_parameters.{field} must be in (0, 1]"
    if not _finite_number(
        hardware.get("dram_random_access_latency_ns"), positive=True,
    ):
        return (
            "hardware_model_parameters.dram_random_access_latency_ns must be "
            "finite and positive"
        )
    numa_local_bytes = hardware.get("numa_local_bytes")
    if (
        isinstance(numa_local_bytes, bool)
        or not isinstance(numa_local_bytes, int)
        or numa_local_bytes < 0
    ):
        return (
            "hardware_model_parameters.numa_local_bytes must be a "
            "non-negative integer"
        )

    coefficients = document.get("empirical_coefficients")
    error = _fields_error(
        coefficients,
        frozenset(HNSW_EMPIRICAL_COEFFICIENT_FIELDS),
        field="empirical_coefficients",
    )
    if error:
        return error
    assert isinstance(coefficients, Mapping)
    for field in HNSW_EMPIRICAL_COEFFICIENT_FIELDS:
        if not _finite_number(coefficients.get(field), positive=True):
            return f"empirical_coefficients.{field} must be finite and positive"

    fit = document.get("fit_metadata")
    error = _fields_error(fit, HNSW_FIT_METADATA_FIELDS, field="fit_metadata")
    if error:
        return error
    assert isinstance(fit, Mapping)
    for field in ("mape_latency_pct", "max_ape_latency_pct"):
        if not _finite_number(fit.get(field)):
            return f"fit_metadata.{field} must be finite and non-negative"
    if float(fit["max_ape_latency_pct"]) < float(fit["mape_latency_pct"]):
        return "fit_metadata.max_ape_latency_pct must be >= mape_latency_pct"
    num_trials = fit.get("num_trials")
    if isinstance(num_trials, bool) or not isinstance(num_trials, int) or num_trials <= 0:
        return "fit_metadata.num_trials must be a positive integer"
    datasets = fit.get("dataset_configs")
    if (
        not isinstance(datasets, list)
        or not datasets
        or any(not isinstance(item, str) or not item.strip() for item in datasets)
        or len(datasets) != len(set(datasets))
    ):
        return "fit_metadata.dataset_configs must be a non-empty unique string list"

    return None
