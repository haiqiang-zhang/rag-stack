"""Deterministic TP/PP semantics for the GenZ generator.

Production pricing deliberately has no learned model-, TP-, or PP-specific
multiplier.  Tensor parallelism is already represented by GenZ's analytical
compute, memory, and physical-communication model.  GenZ publishes both a
single scheduler batch's complete PP traversal and the bottleneck-rank
interdeparture that is reachable only with multiple disjoint batches in
flight.  Current CM consumers model one in-flight scheduler step and therefore
select the traversal explicitly at the GenZ boundary.  This module adds no
fitted fill/drain multiplier.

The compatibility factor validates the runtime shape and returns analytical
identity.  It is not loaded from ``_tp_pp_efficiency`` and cannot make an
unmeasured deployment win through a missing or topology-specific residual.

The legacy calibration helpers remain import-compatible for the offline CM
fitter, but production calls use :func:`pipeline_fill_drain_factor` directly.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import math
import re
from typing import Any, Optional

_TP_PP_PROFILE_KEY = "_tp_pp_efficiency"
_TP_PP_FIELDS = (
    "tp_collective_overhead_us", "tp_alpha", "tp_beta", "pp_alpha", "pp_beta",
)
TP_PP_PROFILE_KIND = "generator_tp_pp"
TP_PP_PARAMETER_UNITS = {
    "tp_collective_overhead_us": "microseconds_per_collective",
    "tp_alpha": "dimensionless",
    "tp_beta": "parameter_size_exponent",
    "pp_alpha": "dimensionless",
    "pp_beta": "parameter_size_exponent",
}
TP_PP_SOURCE_COLUMNS = [
    "model", "tp", "pp", "batch", "in_tok", "out_tok",
    "measured_qps", "hw_key",
]
_FIT_OVERRIDES: ContextVar[dict[str, float] | None] = ContextVar(
    "rag_cm_tp_pp_fit_overrides", default=None,
)

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])")


@contextmanager
def calibration_candidate(**values: float):
    """Private fitter-only candidate coefficients; never a runtime input."""
    allowed = {*_TP_PP_FIELDS, "effective_icn_ll_us"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown TP/PP calibration candidate fields: {sorted(unknown)}")
    normalized = {key: float(value) for key, value in values.items()}
    if not all(math.isfinite(value) and value >= 0.0 for value in normalized.values()):
        raise ValueError("TP/PP calibration candidate values must be finite/non-negative")
    token = _FIT_OVERRIDES.set(normalized)
    try:
        yield
    finally:
        _FIT_OVERRIDES.reset(token)


def _llm_sim_coeffs(hardware: Any) -> dict:
    """Deprecated compatibility shim.

    Production used to load ``_tp_pp_efficiency`` here.  Keeping this
    function as an analytical identity avoids breaking offline callers while
    guaranteeing that runtime pricing never reads topology-fitted coefficients.
    """
    del hardware
    return {}


def _physical_icn_ll_us(hardware: Any) -> Optional[float]:
    """Physical interconnect latency (µs) from ``system.interconnect.gpu_gpu.latency_us``."""
    ic = getattr(hardware, "interconnect", None) or {}
    gg = ic.get("gpu_gpu") if isinstance(ic, dict) else None
    if isinstance(gg, dict) and gg.get("latency_us") is not None:
        return float(gg["latency_us"])
    return None


def tp_collective_overhead_us(hardware: Any = None) -> float:
    """Legacy fitter-only candidate; production TP uses physical latency."""
    del hardware
    candidate = _FIT_OVERRIDES.get()
    if candidate is not None and "tp_collective_overhead_us" in candidate:
        return candidate["tp_collective_overhead_us"]
    return 0.0


def effective_icn_ll_us(
    hardware: Any = None,
    physical_latency_us: float | None = None,
) -> Optional[float]:
    """Physical per-hop all-reduce latency (µs) used by GenZ TP.

    A context-local explicit candidate is retained for the offline fitter.  In
    production there is no candidate context, so the result is exactly the
    declared physical link latency and never a profile coefficient.
    """
    candidate = _FIT_OVERRIDES.get()
    if candidate is not None and "effective_icn_ll_us" in candidate:
        return candidate["effective_icn_ll_us"]
    phys = (
        physical_latency_us
        if physical_latency_us is not None
        else _physical_icn_ll_us(hardware)
    )
    return phys


def model_size_b(model: Any) -> Optional[float]:
    """Parameter count in billions encoded in a calibrated GenZ model name."""
    name = model if isinstance(model, str) else (
        getattr(model, "model", None) or getattr(model, "name", None))
    if not isinstance(name, str):
        return None
    matches = _SIZE_RE.findall(name)
    if not matches:
        return None
    try:
        return float(matches[-1])  # last size token (e.g. "Qwen2.5-1.5B" → 1.5)
    except (ValueError, IndexError):
        return None


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if normalized < 1 or value != normalized:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def pipeline_fill_drain_factor(resident_batch: int, pp: int = 1) -> float:
    """Return identity after validating a scheduler-step PP shape.

    ``resident_batch`` is the number of sequences scheduled in the engine
    step, not client concurrency or the configured admission cap.  Selection
    of traversal versus saturated interdeparture is a structural choice made
    by the GenZ consumer, not a multiplicative factor here.  Both arguments
    remain validated so malformed runtime plans still fail closed.
    """
    _positive_int(resident_batch, "resident_batch")
    _positive_int(pp, "pp")
    return 1.0


def tp_pp_throughput_factor(
    model: Any,
    tp: int = 1,
    pp: int = 1,
    hardware: Any = None,
    *,
    resident_batch: int | None = None,
) -> float:
    """Compatibility wrapper for the retired fitted TP/PP multiplier.

    New production callers use :func:`pipeline_fill_drain_factor`.  A caller
    Calls with or without ``resident_batch`` receive analytical identity rather
    than loading a profile.  Supplying runtime shape still validates it.
    ``model``, ``tp``, and ``hardware`` intentionally do not affect the result.
    """
    del model, tp, hardware
    if resident_batch is None:
        return 1.0
    return pipeline_fill_drain_factor(resident_batch, pp)


def pp_latency_penalty(
    model: Any,
    tp: int,
    pp: int,
    genz_pp_speedup: float,
    hardware: Any = None,
    *,
    resident_batch: int | None = None,
) -> float:
    """Compatibility wrapper for callers transitioning to resident batch.

    The old speedup/model/profile inputs are deliberately ignored.  Without an
    explicit resident batch the only safe analytical result is identity.
    """
    del model, tp, genz_pp_speedup, hardware
    if resident_batch is None:
        return 1.0
    return pipeline_fill_drain_factor(resident_batch, pp)
