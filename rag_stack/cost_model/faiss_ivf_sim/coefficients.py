"""IVF-PQ-specific calibrated coefficients (persistence).

Mirrors the layout used by :mod:`rag_stack.cost_model.faiss_hnsw_sim.coefficients`:
each algorithm owns its own coefficient dataclass, JSON I/O, and the helper
that materializes a :class:`SystemConfig` from a calibration profile.

The fields here (``pq_fastscan_efficiency``, ``rap_fastscan_*``,
``rap_standard_*``, ``pq_standard_per_probe_overhead_us``, …) are PQ-specific, so they live with
the IVF-PQ ``SystemConfig`` rather than in the shared physical-hardware
module (:mod:`rag_stack.cost_model.cpu_specs`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

from rag_stack.calibration_paths import atomic_write_json
from rag_stack.ivf_profile_schema import (
    IVF_PROFILE_KIND,
    IVF_RUNTIME_COEFFICIENT_FIELDS,
    ivf_profile_validation_error,
)

from .systems import RandomAccessPenalty, SystemConfig, system_config_to_faiss_hardware
from .system_presets import SYSTEM_PRESETS


class IVFProfileError(ValueError):
    """An exact latest-schema IVF system profile is malformed."""


def _require_system_preset(preset: str) -> None:
    if preset in SYSTEM_PRESETS:
        return
    from rag_stack.cost_model.faiss_system_registry import require_configured_faiss_preset
    require_configured_faiss_preset("ivf", preset)


# ---------------------------------------------------------------------------
# Calibrated coefficients dataclass
# ---------------------------------------------------------------------------

@dataclass
class CalibratedCoefficients:
    """Latest exact-CPU IVF profile.

    Raw CPU facts stay in ``cpu_specs.py`` or an inline ``system.cpu`` block.
    Every numeric model parameter below is fitted from the frozen standalone
    mode-0 IVF grid; none is inherited from a previous profile or preset.
    """

    # Efficiency factors.
    # NOTE: there is intentionally NO ``memory_efficiency`` — bandwidth is a
    # hardware fact (aggregate + single-thread, in cpu_specs / the YAML cpu
    # block), not a per-calibration knob. See SystemConfig for the BW model.
    blas_efficiency: float
    simd_efficiency: float
    pq_fastscan_efficiency: float       # FastScan PQ scan (vpshufb, nbits=4, single-query)
    pq_fastscan_batch_efficiency: float # FastScan SIMD batch (vpshufb, nbits=4, multi-query)
    pq_standard_efficiency: float       # Standard PQ scan (table lookup, nbits=8)

    # Overhead
    pq_fastscan_per_probe_overhead_us: float    # Per query×nprobe visit (FastScan)
    pq_standard_per_probe_overhead_us: float    # Per query×nprobe visit (Standard PQ)

    # Random access penalty (Standard, nbits=8)
    pq_rap_standard_c1: float
    pq_rap_standard_c2: float
    pq_rap_standard_c3: float

    # Scattered-sequential access model (PQ scan across inverted lists)
    scattered_bw_ceiling: float
    numa_local_bytes: int

    # Fit diagnostics and runtime identity; acquisition metadata is not persisted.
    profile_kind: str
    hardware_preset: str
    mape_latency: float
    max_ape_latency: float
    num_trials: int
    index_configs: list[str]
    batch_sizes: list[int]
    fitted_fields: list[str]

    # IVF-Flat exact-scan per query×nprobe visit overhead.
    flat_per_probe_overhead_us: float

    # IVF-Flat exact full-vector scan efficiency. Separate from simd_efficiency
    # (the compute-bound coarse/LUT/refine SIMD effort); flat scan is memory-
    # bound so its effective SIMD efficiency differs. PQ profiles default it to
    # simd_efficiency on load; a Flat calibration pass sets the real value.
    flat_scan_efficiency: float

    # IVF-Flat per-QUERY fixed overhead (faiss search() call + per-query heap
    # init + result reorder) — the measured latency-vs-work intercept. Despite
    # the historical field name this shared FAISS-search intercept is applied
    # to Flat and PQ and is fitted from the Flat anchor rows.
    flat_fixed_overhead_us: float

    # Per-call PQ setup floor PER vector dim (µs/dim). The once-per-search() PQ
    # query-transform + LUT init cost grows with d: ~5µs at d=128 → ~24µs at
    # d~1000 (measured floor on the 6 cm_benchmark datasets). A constant
    # ``flat_fixed_overhead_us`` under-predicts low-nprobe PQ on high-d embeddings
    # and tiny corpora; this d-aware term reduces to the constant at small d, so
    # the in-regime (d≤384) fit is preserved. PQ-only — IVF-Flat has no LUT.
    pq_setup_per_dim_us: float

    # FastScan QBS compute-amortization floor (nbits=4). 1.0 = no compute
    # amortization with batch (the db-code load amortization lives on the memory
    # side); <1 adds a SIMD-pipeline gain that saturates by large batch. The old
    # hardcoded 0.65 double-counted the load amortization → over-predicted batch.
    pq_fastscan_qbs_floor: float
    # FastScan per-probe overhead batch-amortization exponent (per-query ∝ 1/B**alpha).
    # 0.5 (sqrt-B) over-amortizes (CM batch=64 too fast ~1.6×); fittable, ~0.1-0.2.
    pq_fastscan_overhead_alpha: float
    # FastScan scan effective BW as a fraction of single-thread STREAM (cold
    # scattered scan across inverted lists). >0 makes FastScan memory-bound (the
    # physically-correct bottleneck); 0 = legacy compute-bound path. Fittable.
    pq_fastscan_scan_bw_ceiling: float


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------

def save_coefficients(
    coefficients: CalibratedCoefficients,
    path: str = "calibrated_coefficients.json",
) -> None:
    """Validate and atomically save one latest-schema profile."""
    document = asdict(coefficients)
    error = ivf_profile_validation_error(document, coefficients.hardware_preset)
    if error is not None:
        raise IVFProfileError(f"refusing to save invalid IVF profile: {error}")
    atomic_write_json(path, document)
    print(f"Saved calibrated coefficients to {path}")


def load_coefficients(
    path: str = "calibrated_coefficients.json",
) -> CalibratedCoefficients:
    """Load one complete latest-schema profile without legacy backfilling."""
    with open(path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise IVFProfileError(f"{path}: profile must be a JSON object")
    expected = {field.name for field in fields(CalibratedCoefficients)}
    missing, extra = sorted(expected - set(data)), sorted(set(data) - expected)
    if missing or extra:
        raise IVFProfileError(
            f"{path}: profile fields mismatch; missing={missing}, extra={extra}"
        )
    error = ivf_profile_validation_error(data)
    if error is not None:
        raise IVFProfileError(f"{path}: {error}")
    return CalibratedCoefficients(**data)


# ---------------------------------------------------------------------------
# Profile → SystemConfig materialization
# ---------------------------------------------------------------------------

def build_hardware_from_coefficients(
    coefficients: CalibratedCoefficients,
    active_threads: int = 0,
    system_config=None,
) -> SystemConfig:
    """Create a faiss SystemConfig from calibrated coefficients.

    Hardware source depends on *system_config* (the global SystemConfig from
    YAML):

    - **cpu_preset set** (``cpu: epyc_9124``): must match
      ``coefficients.hardware_preset``; raises ``ValueError`` on mismatch.
      Hardware comes from the preset factory.
    - **custom dict** (``cpu: {hardware_key: my_cpu, …}``): the key must match
      ``coefficients.hardware_preset``; physical facts come from the YAML via
      :func:`system_config_to_faiss_hardware`.
    - **None** (CLI / direct callers): hardware comes from the preset
      factory.

    Efficiency coefficients always come from the exact JSON calibration
    profile; a pipeline config cannot override them.
    """
    coeff_kwargs = dict(
        active_threads=active_threads,
        blas_efficiency=coefficients.blas_efficiency,
        simd_efficiency=coefficients.simd_efficiency,
        flat_scan_efficiency=coefficients.flat_scan_efficiency,
        pq_fastscan_efficiency=coefficients.pq_fastscan_efficiency,
        pq_fastscan_batch_efficiency=coefficients.pq_fastscan_batch_efficiency,
        pq_fastscan_qbs_floor=coefficients.pq_fastscan_qbs_floor,
        pq_fastscan_overhead_alpha=coefficients.pq_fastscan_overhead_alpha,
        pq_standard_efficiency=coefficients.pq_standard_efficiency,
        pq_fastscan_per_probe_overhead_us=coefficients.pq_fastscan_per_probe_overhead_us,
        pq_standard_per_probe_overhead_us=coefficients.pq_standard_per_probe_overhead_us,
        scattered_bw_ceiling=coefficients.scattered_bw_ceiling,
        numa_local_bytes=coefficients.numa_local_bytes,
    )

    if system_config is not None and getattr(system_config, "cpu_preset", ""):
        # Case A: cpu is a preset string — validate match against the profile.
        preset_name = system_config.cpu_preset
        if preset_name != coefficients.hardware_preset:
            raise ValueError(
                f"CPU preset mismatch: YAML cpu={preset_name!r} but "
                f"calibration profile hardware_preset="
                f"{coefficients.hardware_preset!r}"
            )
        _require_system_preset(preset_name)
        hw = SYSTEM_PRESETS[preset_name](**coeff_kwargs)
    elif system_config is not None and system_config.peak_flops > 0:
        # Case B: custom CPU dict — exact key, YAML facts + JSON coefficients.
        cpu_key = str(getattr(system_config, "cpu_key", "") or "").strip()
        if not cpu_key:
            raise ValueError(
                "custom CPU calibration requires system.cpu.hardware_key"
            )
        if cpu_key != coefficients.hardware_preset:
            raise ValueError(
                f"CPU hardware-key mismatch: YAML cpu.hardware_key={cpu_key!r} "
                f"but calibration profile hardware_preset="
                f"{coefficients.hardware_preset!r}"
            )
        hw = system_config_to_faiss_hardware(system_config, **coeff_kwargs)
    else:
        # Case C: no global system_config — pure preset fallback (CLI etc.).
        _require_system_preset(coefficients.hardware_preset)
        hw = SYSTEM_PRESETS[coefficients.hardware_preset](**coeff_kwargs)

    # Random access penalty curves come from the calibration profile.
    hw.random_access_penalty_standard = RandomAccessPenalty(
        c1=coefficients.pq_rap_standard_c1,
        c2=coefficients.pq_rap_standard_c2,
        c3=coefficients.pq_rap_standard_c3,
    )

    # Flat per-probe overhead is applied post-construction so it survives in
    # every hardware source: the preset factories (cases A/C) don't forward it
    # as a kwarg and would silently drop it, and only case B's SystemConfig
    # field would otherwise receive it.
    hw.flat_per_probe_overhead_us = coefficients.flat_per_probe_overhead_us
    hw.flat_fixed_overhead_us = coefficients.flat_fixed_overhead_us
    hw.pq_setup_per_dim_us = coefficients.pq_setup_per_dim_us
    # FastScan batch params — same preset-factory-drops-the-kwarg story as above,
    # so force them post-construction (else the fitted values never reach the CM).
    hw.pq_fastscan_qbs_floor = coefficients.pq_fastscan_qbs_floor
    hw.pq_fastscan_overhead_alpha = coefficients.pq_fastscan_overhead_alpha
    hw.pq_fastscan_scan_bw_ceiling = coefficients.pq_fastscan_scan_bw_ceiling

    return hw
