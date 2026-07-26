"""Metadata-only schema for exact-CPU FAISS IVF calibration profiles.

The production loader and ``calibrate status`` share this contract.  Keeping
it above ``rag_stack.cost_model`` lets readiness checks validate profiles
without importing FAISS, SciPy, or CUDA-adjacent modules.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


IVF_PROFILE_KIND = "faiss_ivf_exact_cpu"

IVF_RUNTIME_COEFFICIENT_FIELDS = (
    "blas_efficiency",
    "simd_efficiency",
    "pq_fastscan_efficiency",
    "pq_fastscan_batch_efficiency",
    "pq_standard_efficiency",
    "pq_fastscan_per_probe_overhead_us",
    "pq_standard_per_probe_overhead_us",
    "pq_rap_standard_c1",
    "pq_rap_standard_c2",
    "pq_rap_standard_c3",
    "scattered_bw_ceiling",
    "numa_local_bytes",
    "flat_per_probe_overhead_us",
    "flat_scan_efficiency",
    "flat_fixed_overhead_us",
    "pq_setup_per_dim_us",
    "pq_fastscan_qbs_floor",
    "pq_fastscan_overhead_alpha",
    "pq_fastscan_scan_bw_ceiling",
)

IVF_POSITIVE_COEFFICIENT_FIELDS = frozenset({
    "blas_efficiency",
    "simd_efficiency",
    "pq_fastscan_efficiency",
    "pq_fastscan_batch_efficiency",
    "pq_standard_efficiency",
    "flat_scan_efficiency",
    "pq_fastscan_qbs_floor",
    "pq_fastscan_scan_bw_ceiling",
})

IVF_METADATA_FIELDS = frozenset({
    "profile_kind",
    "hardware_preset",
    "mape_latency",
    "max_ape_latency",
    "num_trials",
    "index_configs",
    "batch_sizes",
    "fitted_fields",
})
IVF_PROFILE_FIELDS = frozenset(IVF_RUNTIME_COEFFICIENT_FIELDS) | IVF_METADATA_FIELDS


def _finite(value: Any, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and (number > 0.0 if positive else number >= 0.0)


def ivf_profile_validation_error(
    document: Any,
    expected_hardware_key: str | None = None,
) -> str | None:
    """Return the first current-contract validation error, or ``None``."""

    if not isinstance(document, Mapping):
        return "profile must be a JSON object"
    missing = sorted(IVF_PROFILE_FIELDS - set(document))
    extra = sorted(set(document) - IVF_PROFILE_FIELDS)
    if missing or extra:
        return f"profile fields mismatch; missing={missing}, extra={extra}"
    if document.get("profile_kind") != IVF_PROFILE_KIND:
        return f"profile_kind must be {IVF_PROFILE_KIND!r}"
    hardware_key = document.get("hardware_preset")
    if not isinstance(hardware_key, str) or not hardware_key.strip():
        return "hardware_preset must be a non-empty string"
    if expected_hardware_key is not None and hardware_key != expected_hardware_key:
        return (
            f"hardware_preset {hardware_key!r} does not match requested CPU "
            f"{expected_hardware_key!r}"
        )

    for field in IVF_RUNTIME_COEFFICIENT_FIELDS:
        if not _finite(
            document.get(field),
            positive=field in IVF_POSITIVE_COEFFICIENT_FIELDS,
        ):
            qualifier = "positive" if field in IVF_POSITIVE_COEFFICIENT_FIELDS else "non-negative"
            return f"{field} must be finite and {qualifier}"
    if (
        isinstance(document.get("numa_local_bytes"), bool)
        or not isinstance(document.get("numa_local_bytes"), int)
    ):
        return "numa_local_bytes must be an integer"

    for field in ("mape_latency", "max_ape_latency"):
        if not _finite(document.get(field)):
            return f"{field} must be finite and non-negative"
    if float(document["max_ape_latency"]) < float(document["mape_latency"]):
        return "max_ape_latency must be >= mape_latency"
    trials = document.get("num_trials")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials <= 0:
        return "num_trials must be a positive integer"
    for field in ("index_configs", "batch_sizes"):
        if not isinstance(document.get(field), list):
            return f"{field} must be a list"
    if document.get("fitted_fields") != list(IVF_RUNTIME_COEFFICIENT_FIELDS):
        return "fitted_fields must exactly enumerate runtime coefficients"
    return None
