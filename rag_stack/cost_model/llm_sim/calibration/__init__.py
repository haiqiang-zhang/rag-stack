"""Shared backend for the LLM cost model benchmark + calibration framework.

Provides:

- ``BenchmarkReport``  — pure diagnostic (does NOT write anything),
  emits a per-point table + aggregate MAPE / correlation / ratio stats +
  a ``verdict`` about whether the ModelConfig params look structurally
  correct.
- ``CalibrationRecord`` — what ``calibrate()`` persists for a given
  (model_key, hardware_key): either a fitted GenZ
  (compute_efficiency, memory_efficiency) pair or a standalone affine
  ``linear_law`` for components whose measured shape is not roofline-like.
- ``run_benchmark_report(...)`` — builds a ``BenchmarkReport`` from a
  measured-latency grid by predicting with ceff=meff=1.0.
- ``fit_and_save(...)``     — identifiable scipy L-BFGS-B fit of (ceff,
  meff), followed by an out-of-sample validation gate, and save to JSON.
- ``load_record(...)``      — module-level cached load.
- ``detect_hardware_key(...)`` — auto-detect via torch.cuda with YAML
  override.
- ``genz_system_dict(...)`` — THE single constructor translating
  ``SystemConfig.gpu`` (+ declared physical interconnect)
  into the ``real_values=True`` dict GenZ accepts as ``system_name``;
  ``_hardware_to_dict(...)`` is the record path's dict-only view of it.
  Efficiencies are passed as separate ``ceff`` / ``meff`` kwargs to
  ``prefill_moddeling`` / ``decode_moddeling`` rather than baked into a
  ``System`` object.
- ``_inject_shape(...)``     — clone the pipeline_config and override
  (batch_size, seq_len) for a single benchmark point.
"""

from __future__ import annotations

import copy
import json
import math
import os
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from rag_stack.calibration_paths import atomic_write_json, calibration_profile_root
from rag_stack.llm_sim_profile_schema import (
    unwrap_llm_sim_profile_document,
    wrap_llm_sim_profiles,
)

logger = logging.getLogger("RAG-Stack")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkPoint:
    batch_size: int
    seq_len: int
    measured_s: float
    predicted_s: float            # at ceff=meff=1.0
    ratio: float                  # measured / predicted (>1 → real slower)
    rel_err: float                # (predicted - measured) / measured


@dataclass
class BenchmarkReport:
    model_key: str
    hardware_key: str
    cuda_name: str
    points: list[BenchmarkPoint]
    mape: float                   # mean abs percentage err (prediction vs measured)
    max_ape: float
    correlation: float            # pearson(measured, predicted)
    ratio_mean: float
    ratio_stddev: float           # low → params likely correct
    verdict: str                  # "likely_correct" | "scale_mismatch" | "shape_mismatch"
    verdict_notes: list[str]

    def format(self) -> str:
        """Human-readable multi-line report string."""
        lines = []
        lines.append(f"Benchmark report: {self.model_key} on {self.hardware_key}")
        if self.cuda_name:
            lines.append(f"  CUDA device : {self.cuda_name}")
        lines.append("")
        lines.append(f"  {'bs':>4} {'seq':>5}   {'measured_ms':>12}  {'predicted_ms':>12}  "
                     f"{'ratio':>6}  {'err%':>7}")
        for p in self.points:
            lines.append(
                f"  {p.batch_size:>4} {p.seq_len:>5}   "
                f"{p.measured_s * 1000:>12.3f}  {p.predicted_s * 1000:>12.3f}  "
                f"{p.ratio:>6.2f}  {p.rel_err * 100:>7.1f}"
            )
        lines.append("")
        lines.append("Aggregate:")
        lines.append(f"  MAPE           : {self.mape * 100:.1f} %")
        lines.append(f"  Max APE        : {self.max_ape * 100:.1f} %")
        lines.append(f"  Correlation    : {self.correlation:.3f}")
        lines.append(f"  Ratio mean     : {self.ratio_mean:.2f}")
        lines.append(f"  Ratio stddev   : {self.ratio_stddev:.2f}")
        lines.append(f"  Verdict        : {self.verdict.upper()}")
        for note in self.verdict_notes:
            lines.append(f"    → {note}")
        return "\n".join(lines)


LEGACY_INTERNAL_COMPONENT_CONTRACT = "legacy_internal_component_v0"
ZERO_INTERCEPT_ROOFLINE_CONTRACT = "standalone_ceff_meff_zero_intercept_v1"
LINEAR_LAW_COMPONENT_CONTRACT = "standalone_linear_law_v1"
CAPACITY_HINGE_LAW_COMPONENT_CONTRACT = "standalone_capacity_hinge_law_v1"
GENERATOR_COMPONENT_FIT_LATENCY_SOURCE = (
    "selected_measurement_median_component_service_s"
)
_COMPONENT_CALIBRATION_CONTRACTS = {
    LEGACY_INTERNAL_COMPONENT_CONTRACT,
    ZERO_INTERCEPT_ROOFLINE_CONTRACT,
    LINEAR_LAW_COMPONENT_CONTRACT,
    CAPACITY_HINGE_LAW_COMPONENT_CONTRACT,
}


@dataclass
class CalibrationRecord:
    component: str                                  # e.g. "tart", "openvino_reranker"
    model_key: str                                   # e.g. "tart_flan_t5_xl"
    hardware_key: str
    # Roofline records carry both efficiencies.  A pure ``linear_law`` record
    # intentionally carries neither: that prediction form returns before GenZ
    # and must not retain dead roofline coefficients merely to satisfy the
    # persistence schema.
    compute_efficiency: Optional[float] = None
    memory_efficiency: Optional[float] = None
    fit_mape: float = 0.0
    # Schema compatibility field. New standalone roofline records persist zero:
    # LLM serving overhead is calibrated on hardware x stage x batch, not
    # absorbed into a model-keyed coefficient. Legacy single-role auxiliary
    # records may retain runtime-internal fixed work until their own standalone
    # calibrator migrates them; the two-role LLM adapter rejects that contract.
    per_call_overhead_s: float = 0.0
    # Existing non-LLM auxiliary profiles can contain runtime-internal fixed
    # work.  New roofline fits use an explicit zero-intercept contract; real
    # two-role LLM serving stages require that contract at the adapter boundary.
    # This keeps stale model-keyed vLLM intercepts out without invalidating
    # unrelated reranker/embedding profiles that have no stage-curve boundary.
    calibration_contract: str = LEGACY_INTERNAL_COMPONENT_CONTRACT
    # Pearson correlation between measured and predicted latency across
    # the fitted grid. High (→ 1.0) means the prediction *shape* tracks
    # measurement even when the *scale* is off — separates roofline
    # structural correctness from constant-factor errors. Low correlation
    # at any fit_mape is a red flag (wrong ModelConfig).
    correlation: float = 0.0
    bits: str = "bf16"                               # GenZ dtype used at fit time
    grid: list[dict] = field(default_factory=list)   # per-point post-fit info
    # Diagnostics proving that the two fitted physical coefficients were
    # separately observable on the training grid.  Roofline records without
    # this proof are not production calibration evidence.
    identifiability_rank: Optional[int] = None
    identifiability_condition: Optional[float] = None
    # Strictly held-out evidence.  ``validation_grid`` is never supplied to
    # the optimizer; ``validation_mape`` is the publication gate result.
    validation_mape: Optional[float] = None
    validation_grid: list[dict] = field(default_factory=list)
    # Per-item linear law {fixed_ms, per_item_ms, per_token_ms, fit_mape}:
    # alternative prediction form for models whose latency is NOT
    # roofline-shaped (tiny 2-layer cross-encoders like ms_marco_minilm_l2
    # scale ~ B x (c1 + c2 x L); the multiplicative-efficiency + per-call
    # affine fit under-predicts production batches by -50..-64% even when
    # fit ON the extended grid — r12). Fit offline from the stored grid via
    # `cli refit --form linear`. When present, the prediction path uses it
    # INSTEAD of the GenZ roofline; absent → bit-identical legacy behavior.
    linear_law: Optional[dict] = None
    # Production truncation cap (r14): the module's tokenizer never feeds
    # the model more than this many tokens per item (tokenizer
    # model_max_length / explicit max_length in the production node).
    # estimate() clamps params["input_tokens"] to it — real retrieved text
    # routinely exceeds the cap (28KB/query in 0028's trace) and pricing
    # the raw count over/under-shoots by 4-8x depending on direction.
    # None (old records / generators, whose context is real) → no clamp.
    max_seq_len: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate the two legal, non-overlapping pricing forms.

        A record is either a GenZ roofline calibration (both efficiencies are
        present) or a standalone affine ``linear_law`` (neither is present).
        This makes compact linear-only profile records first-class without
        silently accepting half-specified or coefficient-free roofline
        records.
        """
        has_ceff = self.compute_efficiency is not None
        has_meff = self.memory_efficiency is not None
        if has_ceff != has_meff:
            raise ValueError(
                "calibration record must provide both compute_efficiency and "
                "memory_efficiency, or neither for a linear_law-only record"
            )

        linear = self.linear_law
        if not has_ceff:
            if not isinstance(linear, dict):
                raise ValueError(
                    "calibration record without roofline efficiencies requires "
                    "a linear_law"
                )
        elif linear is not None:
            raise ValueError(
                "calibration record pricing forms are exclusive: a GenZ "
                "roofline record cannot also carry linear_law coefficients"
            )

        if isinstance(linear, dict):
            required = ("fixed_ms", "per_item_ms", "per_token_ms")
            missing = [key for key in required if key not in linear]
            if missing:
                raise ValueError(
                    f"linear_law is missing required coefficients: {missing}"
                )
            for key in required:
                try:
                    value = float(linear[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"linear_law.{key} must be a finite non-negative number"
                    ) from exc
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"linear_law.{key} must be a finite non-negative number"
                    )

        if has_ceff:
            for name in ("compute_efficiency", "memory_efficiency"):
                value = float(getattr(self, name))
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(f"{name} must be a finite positive number")

        for name in ("component", "model_key", "hardware_key", "bits"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not np.isfinite(float(self.fit_mape)) or float(self.fit_mape) < 0.0:
            raise ValueError("fit_mape must be a finite non-negative number")
        if not np.isfinite(float(self.correlation)) or not -1.0 <= float(self.correlation) <= 1.0:
            raise ValueError("correlation must be finite and within [-1, 1]")
        overhead = float(self.per_call_overhead_s)
        if not np.isfinite(overhead) or overhead < 0.0:
            raise ValueError("per_call_overhead_s must be finite and non-negative")
        contract = self.calibration_contract
        if contract not in _COMPONENT_CALIBRATION_CONTRACTS:
            raise ValueError("unsupported component calibration_contract")
        if (
            contract == ZERO_INTERCEPT_ROOFLINE_CONTRACT
            and overhead != 0.0
        ):
            raise ValueError(
                "non-zero per_call_overhead_s is forbidden by the standalone "
                "zero-intercept roofline contract"
            )
        if contract == LINEAR_LAW_COMPONENT_CONTRACT and not isinstance(
            linear, dict
        ):
            raise ValueError(
                "linear-law component contract requires linear_law pricing"
            )
        has_rank = self.identifiability_rank is not None
        has_condition = self.identifiability_condition is not None
        if has_rank != has_condition:
            raise ValueError(
                "identifiability_rank and identifiability_condition must be "
                "provided together"
            )
        if has_rank:
            rank = self.identifiability_rank
            if isinstance(rank, bool) or int(rank) != rank or int(rank) < 1:
                raise ValueError("identifiability_rank must be a positive integer")
            condition = float(self.identifiability_condition)
            if not np.isfinite(condition) or condition < 1.0:
                raise ValueError(
                    "identifiability_condition must be finite and at least 1"
                )
        if self.validation_mape is not None:
            validation_mape = float(self.validation_mape)
            if not np.isfinite(validation_mape) or validation_mape < 0.0:
                raise ValueError("validation_mape must be finite and non-negative")
            if not self.validation_grid:
                raise ValueError(
                    "validation_mape requires a non-empty validation_grid"
                )
        elif self.validation_grid:
            raise ValueError(
                "validation_grid requires validation_mape"
            )
        if contract == LINEAR_LAW_COMPONENT_CONTRACT:
            expected_fields = {
                "fixed_ms", "per_item_ms", "per_token_ms", "fit_mape",
            }
            if not isinstance(linear, dict) or set(linear) != expected_fields:
                raise ValueError(
                    "strict linear_law fields must be exactly "
                    f"{sorted(expected_fields)}"
                )
            nested_fit_mape = float(linear["fit_mape"])
            if not np.isfinite(nested_fit_mape) or nested_fit_mape < 0.0:
                raise ValueError(
                    "linear_law.fit_mape must be finite and non-negative"
                )
            if nested_fit_mape != float(self.fit_mape):
                raise ValueError(
                    "linear_law.fit_mape must equal record fit_mape"
                )
            if float(self.fit_mape) > 0.20:
                raise ValueError(
                    "linear-law calibration training MAPE exceeds 20%"
                )
            if overhead != 0.0:
                raise ValueError(
                    "linear-law calibration per_call_overhead_s must be zero"
                )
            if has_rank:
                raise ValueError(
                    "linear-law calibration cannot carry roofline "
                    "identifiability diagnostics"
                )
            if not self.grid:
                raise ValueError(
                    "linear-law calibration requires a non-empty training grid"
                )
            if self.validation_mape is None or not self.validation_grid:
                raise ValueError(
                    "linear-law calibration requires a disjoint held-out grid"
                )
            if float(self.validation_mape) > 0.20:
                raise ValueError(
                    "linear-law calibration held-out MAPE exceeds 20%"
                )
        if contract == CAPACITY_HINGE_LAW_COMPONENT_CONTRACT:
            expected_fields = {
                "fixed_ms",
                "per_item_ms",
                "per_token_ms",
                "capacity_tokens",
                "overload_step_ms",
                "overload_per_token_ms",
                "fit_mape",
            }
            if not isinstance(linear, dict) or set(linear) != expected_fields:
                raise ValueError(
                    "strict capacity-hinge law fields must be exactly "
                    f"{sorted(expected_fields)}"
                )
            for name in (
                "capacity_tokens",
                "overload_step_ms",
                "overload_per_token_ms",
            ):
                value = float(linear[name])
                if not np.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"capacity-hinge linear_law.{name} must be finite and positive"
                    )
            if float(linear["fit_mape"]) != float(self.fit_mape):
                raise ValueError(
                    "capacity-hinge linear_law.fit_mape must equal record fit_mape"
                )
            if float(self.fit_mape) > 0.20:
                raise ValueError(
                    "capacity-hinge calibration training MAPE exceeds 20%"
                )
            if overhead != 0.0 or has_rank:
                raise ValueError(
                    "capacity-hinge calibration cannot carry roofline fields"
                )
            if not self.grid or self.validation_mape is None or not self.validation_grid:
                raise ValueError(
                    "capacity-hinge calibration requires training and held-out grids"
                )
            if float(self.validation_mape) > 0.20:
                raise ValueError(
                    "capacity-hinge calibration held-out MAPE exceeds 20%"
                )
        if has_ceff and contract == ZERO_INTERCEPT_ROOFLINE_CONTRACT:
            if float(self.fit_mape) > 0.20:
                raise ValueError(
                    "roofline calibration training MAPE exceeds 20%"
                )
            if not has_rank or int(self.identifiability_rank) != 2:
                raise ValueError(
                    "roofline calibration requires identifiability_rank=2"
                )
            if float(self.identifiability_condition) > 100.0:
                raise ValueError(
                    "roofline calibration identifiability_condition exceeds 100"
                )
            if self.validation_mape is None or not self.validation_grid:
                raise ValueError(
                    "roofline calibration requires a disjoint held-out grid"
                )
            if float(self.validation_mape) > 0.20:
                raise ValueError(
                    "roofline calibration held-out MAPE exceeds 20%"
                )
        if self.max_seq_len is not None:
            if isinstance(self.max_seq_len, bool) or int(self.max_seq_len) < 1:
                raise ValueError("max_seq_len must be a positive integer")
            if float(self.max_seq_len) != float(int(self.max_seq_len)):
                raise ValueError("max_seq_len must be a positive integer")

        # A stored measurement grid is standalone component evidence, not an
        # extensible bag for E2E-derived ratios.  The old invalid profiles used
        # ``n_e2e_points``/``ratio_median``/free-text notes here; accepting
        # unknown fields would let those forbidden coefficients re-enter the
        # active profile tree unnoticed.
        grid_fields = {
            "batch_size", "seq_len", "measured_s", "predicted_s", "rel_err",
        }
        for grid_name, evidence_grid in (
            ("grid", self.grid),
            ("validation_grid", self.validation_grid),
        ):
            for index, point in enumerate(evidence_grid):
                if not isinstance(point, dict) or set(point) != grid_fields:
                    raise ValueError(
                        f"{grid_name}[{index}] must contain exactly "
                        f"{sorted(grid_fields)}"
                    )
                batch = point["batch_size"]
                if (
                    isinstance(batch, bool)
                    or int(batch) < 1
                    or float(batch) != float(int(batch))
                ):
                    raise ValueError(
                        f"{grid_name}[{index}].batch_size must be a positive integer"
                    )
                for name in ("measured_s", "predicted_s"):
                    value = float(point[name])
                    if not np.isfinite(value) or value <= 0.0:
                        raise ValueError(
                            f"{grid_name}[{index}].{name} must be finite and positive"
                        )
                if not np.isfinite(float(point["rel_err"])):
                    raise ValueError(f"{grid_name}[{index}].rel_err must be finite")
                seq_len = point["seq_len"]
                axes = seq_len if isinstance(seq_len, (list, tuple)) else (seq_len,)
                if not axes or any(
                    isinstance(value, bool)
                    or int(value) < 1
                    or float(value) != float(int(value))
                    for value in axes
                ):
                    raise ValueError(
                        f"{grid_name}[{index}].seq_len must be a positive integer "
                        "or integer tuple"
                    )
        def _point_identity(point: dict) -> tuple[int, tuple[int, ...]]:
            seq_len = point["seq_len"]
            axes = seq_len if isinstance(seq_len, (list, tuple)) else (seq_len,)
            return int(point["batch_size"]), tuple(int(value) for value in axes)

        overlap = {
            _point_identity(point) for point in self.grid
        }.intersection(_point_identity(point) for point in self.validation_grid)
        if overlap:
            raise ValueError(
                "calibration training and held-out grids overlap"
            )


# ---------------------------------------------------------------------------
# Hardware key detection
# ---------------------------------------------------------------------------

def _normalize_hardware_key(cuda_name: str) -> str:
    """Normalize a CUDA device name to a short stable key.

    Examples:
        "NVIDIA GeForce RTX 3090"       → "rtx_3090"
        "NVIDIA H100 80GB HBM3"         → "h100_80gb_hbm3"
        "NVIDIA A100-SXM4-40GB"         → "a100_sxm4_40gb"
    """
    s = cuda_name.strip().lower()
    # Strip common vendor/product prefixes
    for prefix in ("nvidia ", "nvidia-", "nvidia", "geforce ", "geforce"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    s = s.strip()
    # Replace separators with underscore
    s = re.sub(r"[\s\-]+", "_", s)
    # Collapse runs of underscores
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def detect_hardware_key(hardware) -> str:
    """Resolve the hardware_key for calibration lookup.

    Priority:
      1. ``hardware.gpu["hardware_key"]`` (when gpu is a dict). Matches the
         ``hardware_key`` field stored in calibration profile JSON files.
      2. ``hardware.gpu`` itself (when gpu is a GenZ preset string like
         ``"H100_GPU"``).
      3. Otherwise fail closed. Pricing identity must come from the target
         system config, never from whichever GPU happens to run the CM.
    """
    # 1. hardware_key embedded in the gpu dict (works even on CPU-only machines)
    gpu = getattr(hardware, "gpu", None)
    if isinstance(gpu, dict):
        gpu_key = gpu.get("hardware_key")
        if gpu_key:
            return str(gpu_key)

    # 2. GenZ preset string (e.g. "H100_GPU", "A100_80GB_GPU")
    if isinstance(gpu, str) and gpu.strip():
        return gpu
    raise ValueError(
        "system.gpu must be a non-empty preset string or an inline mapping "
        "with hardware_key; target identity cannot be inferred from the local "
        "CUDA device"
    )
    return "unknown"


# ----------------------------------------------------------------------
# Unified calibrate hardware input: every `rag_stack calibrate` component takes
# ONE standard rag-stack ``--config`` and reads ONLY the hardware spec + key from
# its ``system`` block (the CM needs the spec anyway). These helpers are the single
# entry point — no ``--hw-key`` / ``--system`` / hardcoded H100 default / nvidia-smi probe.
# ----------------------------------------------------------------------
def hardware_from_config(config_path: str):
    """(SystemConfig, raw system dict) from a standard config. The SystemConfig drives the
    CM; the raw dict carries the gpu preset / cpu key / available_gpus the helpers below read."""
    from rag_stack.types import SystemConfig
    from rag_stack.cost_model.calibration_config import load_calibration_config
    try:
        cfg = load_calibration_config(config_path)
    except ValueError as exc:
        raise SystemExit(f"[calibrate] {exc}") from exc
    sysd = cfg["system"]
    return SystemConfig.from_dict(sysd), sysd


def gpu_spec(sysd: dict) -> dict:
    """Resolve ``system.gpu`` to a full param dict. A preset KEY → genz_llm/Systems
    (Flops/Memory_size/Memory_BW/ICN); a custom dict → used inline. A preset key must NOT
    also carry inline spec (the preset already defines it)."""
    from Systems.system_configs import system_configs
    gpu = sysd.get("gpu")
    if isinstance(gpu, str):
        if gpu not in system_configs:
            raise SystemExit(f"[calibrate] gpu preset {gpu!r} not in "
                             f"genz_llm/Systems/system_configs.py — add it or give a custom dict")
        return dict(system_configs[gpu])
    if isinstance(gpu, dict):
        hk = gpu.get("hardware_key")
        if hk in system_configs and len(gpu) > 1:
            raise SystemExit(f"[calibrate] gpu.hardware_key {hk!r} is a GenZ preset — give it "
                             f"key-only (`gpu: {hk}`); don't inline spec (the preset defines it)")
        return dict(gpu)
    raise SystemExit("[calibrate] config.system.gpu missing (preset key or spec dict required)")


def gpu_memory_gb(sysd: dict) -> float:
    """Per-GPU memory (GB) for GPU-aware case selection, from the resolved gpu spec."""
    return float(gpu_spec(sysd)["Memory_size"])


def cpu_hardware_key(sysd: dict):
    """The CPU/FAISS hardware key: ``system.cpu`` as a preset string → itself, dict → its
    ``hardware_key``."""
    cpu = sysd.get("cpu")
    if isinstance(cpu, str):
        return cpu
    if isinstance(cpu, dict):
        return cpu.get("hardware_key")
    return None


def available_gpu_count(sysd: dict) -> int:
    """num_gpus for GPU-aware selection — strictly from the config (it must declare
    ``available_gpus``, top-level or under ``system_design_space``)."""
    from rag_stack.performance_context import system_design_space
    ag = sysd.get("available_gpus") or system_design_space(sysd).get("available_gpus")
    if not ag:
        raise SystemExit("[calibrate] config.system must declare available_gpus "
                         "(e.g. system_design_space.available_gpus: [cuda:0, cuda:1, ...])")
    return len(ag)


def get_cuda_name() -> str:
    """Return torch.cuda.get_device_name(0) or an empty string."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# GenZ physical system construction
# ---------------------------------------------------------------------------

# These legacy fields were once accepted inside ``system.gpu`` as
# uncalibrated pricing coefficients.  Keep old YAML files parseable, but never
# forward any of the fields into GenZ: production coefficients and precision
# come only from the resolved component calibration record.
_RETIRED_GPU_CALIBRATION_FIELDS = frozenset({
    "default_compute_eff",
    "default_memory_eff",
    "default_bits",
    "compute_efficiency",
    "memory_efficiency",
    "bits",
})


def genz_system_dict(hardware) -> dict | str:
    """THE single constructor for the GenZ ``system_name`` input — every GenZ call
    (sweep ``generate_sweep_df``, the latency walk ``build_call_latency``, and the
    calibration predict path) MUST source its system dict here, so all paths feed
    GenZ identical interconnect/overhead parameters for the same deployment.

    Resolves ``hardware.gpu`` (a GenZ preset name or a dict of peak specs) and
    injects the two hardware facts so the TP shortfall comes from GenZ physics
    (no external factor):
      - ``ICN``    = the PHYSICAL interconnect bandwidth (``system.interconnect.gpu_gpu``
        when declared — the same source communication.py uses) else the spec's.
      - ``ICN_LL`` = the declared physical per-hop all-reduce latency from
        :func:`tp_efficiency.effective_icn_ll_us`.  Production does not load or
        add a fitted TP coefficient; the same physical link declaration is also
        used by ``communication.py`` for the PD KV handoff.

    An UNKNOWN preset string is returned untouched (GenZ resolves or raises; no
    injection is possible). Per-placement pair topology (which physical link a
    concrete TP/PP group actually crosses) is layered on top by
    ``genz_adapter._system_config_for_parallelism`` — also shared by sweep + walk.
    """
    from rag_stack.cost_model.llm_sim.tp_efficiency import effective_icn_ll_us
    gpu = hardware.gpu if getattr(hardware, "gpu", None) is not None else "A100_40GB_GPU"
    if isinstance(gpu, str):
        # Resolve the preset to a dict so ICN/ICN_LL can be overridden; unknown
        # preset → hand the string to GenZ untouched (no injection possible).
        from Systems.system_configs import system_configs
        if gpu not in system_configs:
            return gpu
        config = dict(system_configs[gpu])
    else:
        config = dict(gpu)
        config.pop("hardware_key", None)
    for field_name in _RETIRED_GPU_CALIBRATION_FIELDS:
        config.pop(field_name, None)
    # Declarative GPU specs are physical peak values.  Allowing a YAML field
    # to set this false silently changes GenZ's units and invalidates every
    # component profile associated with the hardware key.
    config["real_values"] = True
    ic = getattr(hardware, "interconnect", None)
    gg = ic.get("gpu_gpu") if isinstance(ic, dict) else None
    if isinstance(gg, dict) and gg.get("bandwidth_gbps"):
        config["ICN"] = float(gg["bandwidth_gbps"])      # physical bandwidth
    icn_ll = effective_icn_ll_us(hardware)
    if icn_ll is not None:
        config["ICN_LL"] = icn_ll                          # declared physical TP latency
    return config


def _hardware_to_dict(hardware) -> dict:
    """The calibrated-record path's view of :func:`genz_system_dict` — must be a
    real_values dict (a record was fit against concrete peak specs). An
    unresolvable preset raises: production pricing must not substitute YAML
    coefficients or an uncalibrated peak-roofline estimate."""
    config = genz_system_dict(hardware)
    if not isinstance(config, dict):
        raise ValueError(
            f"hardware.gpu preset {config!r} is not in Systems.system_configs; "
            "a calibrated record needs concrete peak specs."
        )
    return config


# ---------------------------------------------------------------------------
# Pipeline config shape injection
# ---------------------------------------------------------------------------

_RERANKER_NODE_TYPES = {"passage_reranker"}
_RETRIEVAL_NODE_TYPES = {"semantic_retrieval", "hybrid_retrieval"}


def grid_seq_axis(value):
    """Parse a stored-grid ``seq_len`` back into the shape key
    :func:`_inject_shape` accepts: a plain int, or a ``(seq, out)`` tuple
    when the grid point carried an output-token dimension (JSON
    round-trips tuples as lists)."""
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    return int(value)


def _inject_shape(pipeline_config: dict, batch_size: int, seq_len) -> dict:
    """Return a deep copy of pipeline_config with (batch_size, seq_len)
    overridden for benchmark-time prediction. Touches:

      - ``_token_stats.avg_chunk_tokens`` → seq_len (used by
        ``PassageRerankerCostModel.get_passage_reranker_input_tokens()`` etc.)
      - every node's ``module[0]["batch_size"]`` (or node["batch_size"])
        so the cost models see the right batch.

    ``seq_len`` may also be a ``(seq_tokens, out_tokens)`` pair — the grid
    axis for stages whose latency has an output-length dimension (the
    generator's decode). The pair additionally overrides
    ``_token_stats.avg_output_tokens``; a scalar keeps the config's own
    output stat (bit-identical legacy behaviour).
    """
    cfg = copy.deepcopy(pipeline_config)

    out_tokens = None
    if isinstance(seq_len, (tuple, list)):
        seq_len, out_tokens = seq_len

    # Override token stats — the reranker reads avg_query_tokens + avg_chunk_tokens
    # and for generative LLMs the prompt template + chunks; zeroing chunk_tokens
    # to seq_len matches "this is the per-passage sequence length".
    ts = cfg.setdefault("_token_stats", {})
    ts["avg_chunk_tokens"] = int(seq_len)
    ts.setdefault("prompt_template_tokens", 0)
    if out_tokens is not None:
        ts["avg_output_tokens"] = int(out_tokens)
    else:
        ts.setdefault("avg_output_tokens", 1)
    ts.setdefault("avg_query_tokens", 0)

    # Batch size goes on the module dict.
    for nl in cfg.get("node_lines", []):
        for node in nl.get("nodes", []):
            for mod in node.get("modules", []):
                mod["batch_size"] = int(batch_size)

    return cfg


# ---------------------------------------------------------------------------
# Benchmark report (no fitting, no persistence)
# ---------------------------------------------------------------------------

def _pearson(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return 1.0
    if np.std(xs) == 0 or np.std(ys) == 0:
        return 1.0
    return float(np.corrcoef(xs, ys)[0, 1])


def _verdict(ratio_mean: float, ratio_stddev: float, correlation: float) -> tuple[str, list[str]]:
    """Classify fit quality to guide user: is it worth calibrating?"""
    cv = (ratio_stddev / ratio_mean) if ratio_mean > 0 else float("inf")
    notes: list[str] = []

    if cv < 0.15 and correlation > 0.95:
        notes.append(
            f"Predicted latency scales consistently with measured (CV={cv:.2f}, "
            f"correlation={correlation:.3f})."
        )
        notes.append(
            "Model params appear structurally sound; proceed with `calibrate` "
            "to fit (ceff, meff)."
        )
        return "likely_correct", notes

    if cv < 0.30 and correlation > 0.85:
        notes.append(
            f"Ratio is moderately stable (CV={cv:.2f}). Calibration should "
            f"help but may leave residual error >5-15%."
        )
        notes.append(
            "Proceed with `calibrate`, but watch post-fit MAPE. If it stays "
            "high, revisit the ModelConfig."
        )
        return "scale_mismatch", notes

    notes.append(
        f"Prediction error varies with shape (CV={cv:.2f}, "
        f"correlation={correlation:.3f})."
    )
    notes.append(
        "Likely cause: wrong hidden_size, intermediate_size, num_ffi, or "
        "missing/extra layers in the ModelConfig. Fix these before calibrating "
        "— otherwise (ceff, meff) will just mask the structural bug."
    )
    return "shape_mismatch", notes


def run_benchmark_report(
    cost_model,
    hardware,
    pipeline_config: dict,
    measured: dict[tuple[int, int], float],
    compute_eff: float = 1.0,
    memory_eff: float = 1.0,
    bits: str = "bf16",
) -> BenchmarkReport:
    """Predict at the given (compute_eff, memory_eff), build per-point
    table, aggregate, verdict.

    Defaults to (1.0, 1.0) — the raw roofline diagnostic used **before**
    calibration. Pass saved ``compute_efficiency`` / ``memory_efficiency``
    from a prior calibration record to check that an existing profile
    still fits new hardware / a changed module implementation.

    NO fitting, NO persistence.
    """
    from rag_stack.cost_model.llm_sim.calibration._predict import predict_latency

    points: list[BenchmarkPoint] = []
    hw_dict = _hardware_to_dict(hardware)
    inject = getattr(cost_model, "_inject_calibration_shape", None) or _inject_shape
    for (bs, sl), measured_s in sorted(measured.items()):
        config = inject(pipeline_config, bs, sl)
        try:
            predicted_s = predict_latency(
                cost_model, config, hw_dict,
                compute_eff=compute_eff, memory_eff=memory_eff, bits=bits,
            )
        except Exception as e:
            logger.warning(
                f"Prediction failed for (bs={bs}, seq={sl}): "
                f"{type(e).__name__}: {e}"
            )
            continue
        if predicted_s <= 0 or measured_s <= 0:
            continue
        ratio = measured_s / predicted_s
        rel_err = (predicted_s - measured_s) / measured_s
        points.append(BenchmarkPoint(
            batch_size=bs, seq_len=sl,
            measured_s=float(measured_s), predicted_s=float(predicted_s),
            ratio=float(ratio), rel_err=float(rel_err),
        ))

    if not points:
        raise RuntimeError(
            "Benchmark produced no usable points (all predictions failed or "
            "measured/predicted were non-positive)."
        )

    measured_arr = np.array([p.measured_s for p in points])
    predicted_arr = np.array([p.predicted_s for p in points])
    ratios = np.array([p.ratio for p in points])
    apes = np.abs((predicted_arr - measured_arr) / measured_arr)

    mape = float(apes.mean())
    max_ape = float(apes.max())
    correlation = _pearson(measured_arr, predicted_arr)
    ratio_mean = float(ratios.mean())
    ratio_stddev = float(ratios.std())

    verdict, verdict_notes = _verdict(ratio_mean, ratio_stddev, correlation)

    hw_key = detect_hardware_key(hardware)
    try:
        model_key = cost_model._get_model_name_for_df(pipeline_config)
    except Exception:
        model_key = type(cost_model).__name__

    return BenchmarkReport(
        model_key=model_key,
        hardware_key=hw_key,
        cuda_name=get_cuda_name(),
        points=points,
        mape=mape,
        max_ape=max_ape,
        correlation=correlation,
        ratio_mean=ratio_mean,
        ratio_stddev=ratio_stddev,
        verdict=verdict,
        verdict_notes=verdict_notes,
    )


# ---------------------------------------------------------------------------
# Calibration fit + save
# ---------------------------------------------------------------------------

def _default_calibration_dir() -> Path:
    """Resolve the shared profile root at call time.

    All component loaders must observe one process-local calibration context;
    freezing this environment-controlled root at module import could mix two
    profile generations in a long-lived process.
    """

    return calibration_profile_root() / "llm_sim"


def _system_path(directory: Path, hw_key: str) -> Path:
    """One JSON file per system (hardware): ``<hw_key>.json``.

    All LLM-sim component profiles measured on that hardware live inside this
    single file, keyed by :func:`_record_key`. (Was previously one file per
    ``<module>__<model>__<hw>__<bits>``; consolidated so a machine's whole
    calibration is one artifact.)
    """
    return directory / f"{hw_key}.json"


def _record_key(component: str, model_key: str, bits: str) -> str:
    """In-file key for one record: ``<component>__<model_key>__<bits>``.

    Double-underscore separator so single-underscore tokens inside keys
    survive (model_key may contain ``/`` but never ``__``). Three
    discriminators, from most to least runtime-specific:

      - ``component`` (e.g. ``flag_embedding_reranker`` vs
        ``openvino_reranker``) — isolates different runtimes that happen
        to share a model architecture.
      - ``model_key`` (e.g. ``bge_reranker_large``) — the ModelConfig.
      - ``bits`` (e.g. ``f32`` / ``bf16`` / ``int8``) — deployment
        precision.

    The hardware discriminator is the *file* (:func:`_system_path`).
    """
    return f"{component}__{model_key}__{bits}"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        document = json.load(f)
    return unwrap_llm_sim_profile_document(document, source=str(path))


def _save_json(path: Path, data: dict) -> None:
    atomic_write_json(path, wrap_llm_sim_profiles(data))


def save_record(record: "CalibrationRecord", directory: Optional[Path] = None) -> Path:
    """Persist one record into its system file (``<hw_key>.json``), merging.

    Reads the existing system JSON's strict current ``profiles`` map,
    upserts this record under its :func:`_record_key`, and writes the whole
    exact envelope back — so calibrating a new component on a machine
    never clobbers the others. Invalidates the load cache for that file.
    """
    # Dataclass instances are mutable; re-run the schema gate immediately
    # before serialization so a caller cannot construct a valid zero-intercept
    # record, mutate it, and write a forbidden model-keyed intercept.
    record.__post_init__()
    directory = Path(directory) if directory else _default_calibration_dir()
    path = _system_path(directory, record.hardware_key)
    data = _load_json(path)  # fresh from disk; do not trust the read cache for writes
    payload = asdict(record)
    if record.compute_efficiency is None:
        # Preserve the compact linear-only schema on a load/save round trip;
        # these values are not defaults and must not reappear as null/dead
        # coefficients in the persisted calibration artifact.
        payload.pop("compute_efficiency", None)
        payload.pop("memory_efficiency", None)
        payload.pop("per_call_overhead_s", None)
    data[_record_key(record.component, record.model_key, record.bits)] = payload
    _save_json(path, data)
    _LOADED_CACHE.pop(path, None)
    return path


def _build_fit_surrogate(cost_model, pipeline_config, hw_dict, bits, inject,
                         points, predict_fn):
    """Fast-fit surrogate (r13): per grid point, extract GenZ's op-level
    (FLOPs, bytes) arrays ONCE and calibrate per-point linear weights from
    3 true-path probes, so the fit objective becomes pure numpy:

        t(point; ceff, meff) = w_pf * S_pf + w_dec * S_dec + c
        S_stage(ceff, meff)  = sum_ops max(flops/(F*ceff), bytes/(BW*meff))

    Probe experiment: true/S ratio is CONSTANT across (ceff, meff) probes
    (unit factors + aggregation constants), so the per-point weights make
    the surrogate exact up to max-branch structure — which the post-fit
    verification checks against the true path (>0.5%% dev -> slow-path
    fallback). Returns None when extraction isn't possible."""
    import numpy as np_
    from GenZ.LLM_inference.llm_prefill import prefill_moddeling as _pf
    from GenZ.LLM_inference.llm_decode import decode_moddeling as _dm
    from rag_stack.cost_model.llm_sim.genz_adapter import kv_capped_decode_batch

    PROBES = [(1.0, 1.0), (0.4, 0.9), (0.9, 0.3)]

    _COLS = ("Compute time (msec)", "Memory time (msec)",
             "Communication time (msec)")

    def op_arrays(mdf):
        # Extract GenZ's own per-op Compute/Memory/Communication times at
        # ceff=meff=1 (Repeat-multiplier walk per get_summary_table). GenZ's
        # per-op latency is exactly max(C/ceff, M/meff, X)
        # (operator_base.get_roofline) — C already carries the op-specific
        # efficiency modulation, X is (ceff, meff)-invariant. Rebuilding C/M
        # from raw FLOPs/bytes loses both, drifting the max flip points
        # (the 3.6% verification failure on 7B). These columns only exist on
        # the roofline df (ModdelingOutput["model_df"]), NOT the
        # model_profilling one — missing columns must raise (-> slow path),
        # never read as zero.
        missing = [c for c in _COLS if c not in mdf.columns]
        if missing:
            raise KeyError(f"roofline df missing {missing}")
        C, M, X = [], [], []
        mult = 1.0
        for _, row in mdf.iterrows():
            op_t = row["Op Type"]
            if op_t == "Repeat":
                mult *= float(row["Dimension"])
                continue
            if op_t == "EndRepeat":
                mult /= float(row["Dimension"])
                continue
            C.append(float(row[_COLS[0]]) * mult)
            M.append(float(row[_COLS[1]]) * mult)
            X.append(float(row[_COLS[2]]) * mult)
        import numpy as np2
        return np2.array(C), np2.array(M), np2.array(X)

    per_point = {}
    try:
        for (bs, sl) in points:
            cfg = inject(pipeline_config, bs, sl)
            model = cost_model._get_model(cfg)
            params = cost_model._get_params(cfg)
            module = cost_model._find_module(cfg)
            tp = max(1, int(module.get("tensor_parallel", 1) or 1))
            pp = max(1, int(module.get("pipeline_parallel", 1) or 1))
            stages = cost_model._stage_kinds(cfg)
            cols = []
            if "prefill" in stages:
                res = _pf(model=model, batch_size=bs,
                          input_tokens=params["input_tokens"],
                          system_name=dict(hw_dict), bits=bits,
                          ceff=1.0, meff=1.0,
                          tensor_parallel=tp, pipeline_parallel=pp,
                          model_offload=True)
                cols.append(op_arrays(res["model_df"]))
            if "decode" in stages:
                ctx = int(params["input_tokens"]) + int(params["output_tokens"])
                b_eff = kv_capped_decode_batch(model, bs, ctx, hw_dict, bits, tp, pp)
                res = _dm(model=model, batch_size=b_eff, Bb=1,
                          input_tokens=params["input_tokens"],
                          output_tokens=params["output_tokens"],
                          system_name=dict(hw_dict), bits=bits,
                          ceff=1.0, meff=1.0,
                          tensor_parallel=tp, pipeline_parallel=pp,
                          model_offload=True)
                cols.append(op_arrays(res["model_df"]))
            if not cols:
                return None

            def S(ceff, meff, _cols=cols):
                return [np_.maximum(np_.maximum(c / ceff, m / meff), x).sum()
                        for c, m, x in _cols]

            A, y = [], []
            for (ce, me) in PROBES:
                A.append(S(ce, me) + [1.0])
                y.append(predict_fn(bs, sl, ce, me, 0.0))
            w, *_ = np_.linalg.lstsq(np_.array(A), np_.array(y), rcond=None)
            per_point[(bs, sl)] = (cols, w)
    except Exception as exc:  # noqa: BLE001 — extraction failed -> slow path
        logger.warning(f"fast-fit surrogate build failed ({exc}); slow path")
        return None

    def predict(bs, sl, ceff, meff, call_oh):
        cols, w = per_point[(bs, sl)]
        s = [np_.maximum(np_.maximum(c / ceff, m / meff), x).sum()
             for c, m, x in cols]
        return float(np_.dot(w[:-1], s) + w[-1] + call_oh)

    return predict


DEFAULT_MAX_IDENTIFIABILITY_CONDITION = 100.0
DEFAULT_MAX_VALIDATION_MAPE = 0.20
_EFFICIENCY_BOUNDS = ((0.05, 1.0), (0.05, 1.0))


def split_fit_validation_evidence(
    measured: dict[tuple[int, Any], float],
) -> tuple[dict[tuple[int, Any], float], dict[tuple[int, Any], float]]:
    """Deterministically reserve a small out-of-sample component grid.

    Generic standalone calibrators historically fitted every measured point.
    Reserve at most two points (one interior, one high-end) so the two physical
    coefficients retain a broad training grid while publication gains an
    independently scored holdout without running another deployment.
    """

    def _point_order(point: tuple[int, Any]) -> tuple[int, tuple[int, ...]]:
        axis = point[1]
        axes = axis if isinstance(axis, (tuple, list)) else (axis,)
        return int(point[0]), tuple(int(value) for value in axes)

    ordered = sorted(measured, key=_point_order)
    if len(ordered) < 4:
        raise ValueError(
            "component calibration requires at least four measurements to "
            "separate a two-coefficient fit from held-out validation"
        )
    heldout_indexes = {len(ordered) - 1}
    if len(ordered) >= 6:
        heldout_indexes.add(len(ordered) // 2)
    validation = {
        point: measured[point]
        for index, point in enumerate(ordered)
        if index in heldout_indexes
    }
    training = {
        point: measured[point]
        for index, point in enumerate(ordered)
        if index not in heldout_indexes
    }
    return training, validation


def _efficiency_identifiability(
    predict_fn: Callable[[int, Any, float, float, float], float],
    points,
    compute_efficiency: float,
    memory_efficiency: float,
) -> tuple[int, float]:
    """Return rank and condition number of the roofline fit Jacobian.

    The derivatives are dimensionless log-latency sensitivities with respect
    to log compute/memory efficiency.  A full-rank, well-conditioned matrix is
    required before the two coefficients may be persisted.  This explicitly
    rejects grids where both knobs move every prediction in the same way — a
    low in-sample error on such a grid does not identify either coefficient.
    """

    point_list = list(points)
    if len(point_list) < 2:
        return 0, float("inf")

    values = [float(compute_efficiency), float(memory_efficiency)]
    jacobian = np.zeros((len(point_list), 2), dtype=float)
    log_step = 1e-3
    for column, value in enumerate(values):
        lower, upper = _EFFICIENCY_BOUNDS[column]
        lo = max(lower, value * float(np.exp(-log_step)))
        hi = min(upper, value * float(np.exp(log_step)))
        if not hi > lo:
            return 0, float("inf")
        denominator = float(np.log(hi) - np.log(lo))
        for row, (batch_size, seq_len) in enumerate(point_list):
            args_lo = list(values)
            args_hi = list(values)
            args_lo[column] = lo
            args_hi[column] = hi
            predicted_lo = float(
                predict_fn(batch_size, seq_len, args_lo[0], args_lo[1], 0.0)
            )
            predicted_hi = float(
                predict_fn(batch_size, seq_len, args_hi[0], args_hi[1], 0.0)
            )
            if (
                not np.isfinite(predicted_lo)
                or not np.isfinite(predicted_hi)
                or predicted_lo <= 0.0
                or predicted_hi <= 0.0
            ):
                raise RuntimeError(
                    "identifiability probe produced a non-positive or "
                    "non-finite prediction"
                )
            jacobian[row, column] = (
                float(np.log(predicted_hi)) - float(np.log(predicted_lo))
            ) / denominator

    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if len(singular_values) < 2 or singular_values[0] <= 0.0:
        return 0, float("inf")
    tolerance = max(1e-9, float(singular_values[0]) * 1e-6)
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank < 2 or singular_values[-1] <= tolerance:
        return rank, float("inf")
    return rank, float(singular_values[0] / singular_values[-1])


def _validate_fit_evidence(name: str, evidence: dict) -> None:
    if not evidence:
        raise ValueError(f"{name} must contain at least one point")
    for point, measured_s in evidence.items():
        if not isinstance(point, tuple) or len(point) != 2:
            raise ValueError(
                f"{name} keys must be (batch_size, seq_axis) tuples"
            )
        batch_size = point[0]
        if (
            isinstance(batch_size, bool)
            or int(batch_size) < 1
            or float(batch_size) != float(int(batch_size))
        ):
            raise ValueError(f"{name} batch_size must be a positive integer")
        value = float(measured_s)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} latency must be finite and positive")
        seq_axis = point[1]
        axes = seq_axis if isinstance(seq_axis, (tuple, list)) else (seq_axis,)
        if not axes or any(
            isinstance(axis, bool)
            or int(axis) < 1
            or float(axis) != float(int(axis))
            for axis in axes
        ):
            raise ValueError(
                f"{name} seq_axis must be a positive integer or integer tuple"
            )


def fit_and_save(
    cost_model,
    hardware,
    pipeline_config: dict,
    measured: dict[tuple[int, int], float],
    component: str,
    model_key: Optional[str] = None,
    output_dir: Optional[Path] = None,
    bits: str = "bf16",
    *,
    validation_points: Optional[dict[tuple[int, int], float]] = None,
    max_validation_mape: float = DEFAULT_MAX_VALIDATION_MAPE,
    max_identifiability_condition: float = DEFAULT_MAX_IDENTIFIABILITY_CONDITION,
) -> CalibrationRecord:
    """Fit an identifiable zero-intercept roofline and persist it.

    Only ``compute_efficiency`` and ``memory_efficiency`` are optimized.
    ``per_call_overhead_s`` is fixed at zero because generic serving overhead
    belongs to a hardware x stage x batch service curve, never to this
    model-keyed component record.

    ``validation_points`` are strictly held out: they are neither passed to
    the optimizer nor used to build its surrogate.  When provided, their MAPE
    must not exceed ``max_validation_mape``.  The training-grid numerical
    Jacobian must independently identify both efficiencies and have condition
    number at most ``max_identifiability_condition``.  Any failed gate aborts
    before the profile file is changed.

    """
    import scipy.optimize
    from rag_stack.cost_model.llm_sim.calibration._predict import predict_latency

    measured = dict(measured)
    validation_points = dict(validation_points or {})
    _validate_fit_evidence("measured", measured)
    if validation_points:
        _validate_fit_evidence("validation_points", validation_points)
        overlap = set(measured).intersection(validation_points)
        if overlap:
            raise ValueError(
                "validation_points must be disjoint from the fitted grid; "
                f"overlap={sorted(overlap, key=repr)!r}"
            )
    if (
        not np.isfinite(float(max_validation_mape))
        or float(max_validation_mape) < 0.0
    ):
        raise ValueError("max_validation_mape must be finite and non-negative")
    if (
        not np.isfinite(float(max_identifiability_condition))
        or float(max_identifiability_condition) < 1.0
    ):
        raise ValueError(
            "max_identifiability_condition must be finite and at least 1"
        )

    if model_key is None:
        try:
            model_key = cost_model._get_model_name_for_df(pipeline_config)
        except Exception:
            model_key = type(cost_model).__name__
    hw_key = detect_hardware_key(hardware)
    hw_dict = _hardware_to_dict(hardware)

    inject = getattr(cost_model, "_inject_calibration_shape", None) or _inject_shape

    def _predict(
        bs: int, sl: int, ceff: float, meff: float, call_oh: float = 0.0,
    ) -> float:
        config = inject(pipeline_config, bs, sl)
        base = predict_latency(
            cost_model, config, hw_dict,
            compute_eff=ceff, memory_eff=meff, bits=bits,
            per_call_overhead_s=call_oh,
        )
        return base

    # Fast-fit surrogate (r13): 5 true-path calls per grid point up front,
    # then the optimizer loop is pure numpy. Post-fit verification against
    # the true path guards fidelity; RAG_STACK_FAST_FIT=0 disables.
    _fast = None
    if os.environ.get("RAG_STACK_FAST_FIT", "1").strip().lower() not in (
        "0", "false", "no", "off",
    ):
        _fast = _build_fit_surrogate(
            cost_model, pipeline_config, hw_dict, bits, inject,
            sorted(measured.keys(), key=repr), _predict,
        )

    def objective(x: np.ndarray) -> float:
        ceff, meff = float(x[0]), float(x[1])
        if ceff <= 0 or meff <= 0:
            return 1e6
        errs = []
        for (bs, sl), real in measured.items():
            try:
                pred = (
                    _fast(bs, sl, ceff, meff, 0.0)
                    if _fast is not None
                    else _predict(bs, sl, ceff, meff, 0.0)
                )
            except Exception:
                return 1e6
            if not np.isfinite(pred) or pred <= 0:
                return 1e6
            # Fit in log-latency space.  The previous L1 relative-error
            # objective has a sharp corner: with two coefficients and three
            # regimes it can exactly match two large points while completely
            # sacrificing the remaining batch.  That produced a formally
            # identifiable fit whose B4 error was 36% and whose independent
            # held-out MAPE was 23.6% on H100.  Mean squared log error is
            # symmetric for
            # over/under prediction and forces every sparse anchor to
            # contribute, without adding an intercept or another coefficient.
            errs.append(math.log(float(pred) / float(real)))
        # Do not take the square root here.  It has the same minimizer but is
        # non-differentiable at an exact fit, which makes finite-difference
        # L-BFGS-B needlessly brittle on synthetic and near-perfect grids.
        return float(np.mean(np.square(errs)))

    # Zero-intercept physical fit.  Multistart remains useful because the
    # roofline max branches make the two-parameter objective non-smooth.
    bounds = list(_EFFICIENCY_BOUNDS)
    starts = [
        np.array([1.0, 1.0]),
        np.array([0.5, 0.5]),
        np.array([0.3, 0.9]),
        np.array([0.9, 0.3]),
        np.array([0.7, 0.7]),
        np.array([0.2, 0.2]),
    ]

    def _run_optimizer():
        best = None
        for x0 in starts:
            candidate = scipy.optimize.minimize(
                objective, x0=x0, bounds=bounds, method="L-BFGS-B",
            )
            if np.isfinite(candidate.fun) and (
                best is None or candidate.fun < best.fun
            ):
                best = candidate
        if best is None or float(best.fun) >= 1e6:
            raise RuntimeError(
                "roofline fit failed: no finite prediction over the complete "
                "training grid"
            )
        return best

    res = _run_optimizer()
    ceff, meff = float(res.x[0]), float(res.x[1])
    call_overhead = 0.0

    if _fast is not None:
        # Fidelity gate: the true path at the fitted coefficients must match
        # the surrogate the fit trusted. Structural max-branch drift shows
        # up here; >0.5%% max deviation -> redo the fit on the true path.
        devs = []
        for (bs, sl) in measured:
            t_true = _predict(bs, sl, ceff, meff, 0.0)
            t_sur = _fast(bs, sl, ceff, meff, 0.0)
            if t_true > 0:
                devs.append(abs(t_sur - t_true) / t_true)
        max_dev = max(devs) if devs else 1.0
        if max_dev > 0.005:
            logger.warning(
                f"fast-fit verification failed (max dev {max_dev:.4f}); "
                "refitting on the true path"
            )
            _fast = None
            res = _run_optimizer()
            ceff, meff = float(res.x[0]), float(res.x[1])
        else:
            logger.info(
                f"fast-fit verified: max surrogate dev {max_dev:.5f} "
                f"across {len(devs)} grid points"
            )

    identifiability_rank, identifiability_condition = (
        _efficiency_identifiability(
            _predict, sorted(measured, key=repr), ceff, meff,
        )
    )
    if (
        identifiability_rank < 2
        or not np.isfinite(identifiability_condition)
        or identifiability_condition > float(max_identifiability_condition)
    ):
        raise RuntimeError(
            "roofline efficiencies are not separately identifiable: "
            f"rank={identifiability_rank}, "
            f"condition={identifiability_condition:.3g}, "
            f"limit={float(max_identifiability_condition):.3g}; add one "
            "compute-dominant and one memory-dominant standalone point"
        )

    def _diagnostic_grid(evidence: dict, evidence_name: str):
        diagnostics = []
        real_values: list[float] = []
        predicted_values: list[float] = []
        errors: list[float] = []
        for (bs, sl), real in sorted(evidence.items(), key=lambda item: repr(item[0])):
            try:
                pred = float(_predict(bs, sl, ceff, meff, 0.0))
            except Exception as exc:
                raise RuntimeError(
                    f"{evidence_name} prediction failed for "
                    f"(batch={bs}, seq={sl!r})"
                ) from exc
            if not np.isfinite(pred) or pred <= 0.0:
                raise RuntimeError(
                    f"{evidence_name} prediction must be finite and positive "
                    f"for (batch={bs}, seq={sl!r})"
                )
            rel_err = float((pred - real) / real)
            diagnostics.append({
                "batch_size": bs,
                "seq_len": sl,
                "measured_s": float(real),
                "predicted_s": pred,
                "rel_err": rel_err,
            })
            real_values.append(float(real))
            predicted_values.append(pred)
            errors.append(abs(rel_err))
        return diagnostics, real_values, predicted_values, float(np.mean(errors))

    grid, measured_list, predicted_list, fit_mape = _diagnostic_grid(
        measured, "training",
    )

    validation_grid: list[dict] = []
    validation_mape: Optional[float] = None
    if validation_points:
        validation_grid, _, _, validation_mape = _diagnostic_grid(
            validation_points, "validation",
        )
        if validation_mape > float(max_validation_mape):
            raise RuntimeError(
                "held-out validation failed: "
                f"MAPE={validation_mape * 100:.2f}% exceeds "
                f"{float(max_validation_mape) * 100:.2f}%"
            )

    # Pearson correlation between measured and predicted across grid
    if len(measured_list) >= 2:
        correlation = _pearson(np.array(measured_list), np.array(predicted_list))
    else:
        correlation = 0.0

    record = CalibrationRecord(
        component=component,
        model_key=model_key,
        hardware_key=hw_key,
        compute_efficiency=ceff,
        memory_efficiency=meff,
        per_call_overhead_s=call_overhead,
        calibration_contract=ZERO_INTERCEPT_ROOFLINE_CONTRACT,
        fit_mape=fit_mape,
        correlation=correlation,
        bits=bits,
        grid=grid,
        identifiability_rank=identifiability_rank,
        identifiability_condition=identifiability_condition,
        validation_mape=validation_mape,
        validation_grid=validation_grid,
    )

    path = save_record(record, output_dir)
    validation_summary = (
        f"validation_mape={validation_mape * 100:.1f}%"
        if validation_mape is not None
        else "validation=not_provided"
    )
    logger.info(
        f"Saved calibration: {component}/{model_key}@{hw_key} → "
        f"ceff={ceff:.3f}, meff={meff:.3f}, mape={fit_mape * 100:.1f}%, "
        f"rank={identifiability_rank}, cond={identifiability_condition:.2f}, "
        f"{validation_summary}. "
        f"File: {path} (key {_record_key(component, model_key, bits)})"
    )
    return record


def fit_linear_and_save(
    cost_model,
    hardware,
    pipeline_config: dict,
    record: "CalibrationRecord",
    output_dir: Optional[Path] = None,
) -> "CalibrationRecord":
    """Fit the per-item linear law t = fixed + B*(per_item + per_token*L)
    to an EXISTING record's stored measured grid and persist it as
    ``record.linear_law`` (offline, no GPU).

    The L axis is the model's OWN per-item input tokens at each grid
    point (``_get_params(inject(config, bs, sl))["input_tokens"]``) so the
    fit and the production prediction consume the identical quantity.
    MAPE-consistent fit: non-negative least squares on the relative-error
    system (rows scaled by 1/measured)."""
    import scipy.optimize

    inject = getattr(cost_model, "_inject_calibration_shape", None) or _inject_shape
    measured = {
        (int(g["batch_size"]), grid_seq_axis(g["seq_len"])): float(g["measured_s"])
        for g in record.grid if "batch_size" in g and g.get("measured_s")
    }
    if not measured:
        raise ValueError("record stores no measured grid — cannot fit linear law")

    rows, rhs, points = [], [], []
    for (bs, sl), real in sorted(measured.items()):
        cfg = inject(pipeline_config, bs, sl)
        l_tokens = float(cost_model._get_params(cfg)["input_tokens"])
        rows.append([1.0 / real, bs / real, bs * l_tokens / real])
        rhs.append(1.0)
        points.append((bs, sl, l_tokens, real))
    a, _resid = scipy.optimize.nnls(np.array(rows), np.array(rhs))
    fixed_s, per_item_s, per_token_s = (float(x) for x in a)

    grid, errs = [], []
    m_list, p_list = [], []
    for bs, sl, l_tokens, real in points:
        pred = fixed_s + bs * (per_item_s + per_token_s * l_tokens)
        grid.append({
            "batch_size": bs, "seq_len": sl,
            "measured_s": float(real), "predicted_s": float(pred),
            "rel_err": float((pred - real) / real) if real > 0 else 0.0,
        })
        if real > 0:
            errs.append(abs(pred - real) / real)
            m_list.append(real); p_list.append(pred)
    fit_mape = float(np.mean(errs)) if errs else 1.0

    record.linear_law = {
        "fixed_ms": fixed_s * 1000.0,
        "per_item_ms": per_item_s * 1000.0,
        "per_token_ms": per_token_s * 1000.0,
        "fit_mape": fit_mape,
    }
    # Convert, do not layer: the persistence contract permits exactly one
    # pricing form.  Roofline identifiability/validation diagnostics no longer
    # describe the newly fitted linear law and must not survive the conversion.
    record.compute_efficiency = None
    record.memory_efficiency = None
    record.per_call_overhead_s = 0.0
    record.calibration_contract = LINEAR_LAW_COMPONENT_CONTRACT
    record.identifiability_rank = None
    record.identifiability_condition = None
    record.validation_mape = None
    record.validation_grid = []
    record.grid = grid
    record.fit_mape = fit_mape
    if len(m_list) >= 2:
        record.correlation = _pearson(np.array(m_list), np.array(p_list))
    path = save_record(record, output_dir)
    invalidate_cache()
    logger.info(
        f"Saved linear-law refit: {record.component}/{record.model_key}"
        f"@{record.hardware_key} → fixed={fixed_s * 1e3:.2f}ms, "
        f"per_item={per_item_s * 1e3:.3f}ms, per_token={per_token_s * 1e6:.2f}us, "
        f"mape={fit_mape * 100:.1f}%. File: {path}"
    )
    return record


# ---------------------------------------------------------------------------
# Load at inference time
# ---------------------------------------------------------------------------

_LOADED_CACHE: dict[Path, dict] = {}


AGNOSTIC_MODULE_KEY = "agnostic"
"""Sentinel component used for calibration profiles that are
component-agnostic — i.e. share the same (model, hw, bits)
calibration across multiple components that wrap the same weights
identically (typical for LLM serving backends: vllm / openai_llm /
llama_index_llm). ``load_record`` falls back to records with this
prefix when the requested component has no specialized profile."""


def _load_system(path: Path) -> dict:
    """Load the validated current ``profiles`` map for one system."""
    if path not in _LOADED_CACHE:
        _LOADED_CACHE[path] = _load_json(path) if path.exists() else {}
    return _LOADED_CACHE[path]


def _lookup_record(
    directory: Path, component: str, model_key: str,
    hardware_key: str, bits: Optional[str],
) -> Optional[dict]:
    """Find one record dict inside the system file for (module, model, bits).

    When ``bits`` is given, return the exact keyed record. When ``bits`` is
    ``None``, match records sharing the ``<module>__<model>__`` prefix. An
    unambiguous single precision may be inferred; multiple precisions require
    the caller to declare ``bits`` and fail closed.
    """
    data = _load_system(_system_path(directory, hardware_key))
    if not data:
        return None
    if bits is not None:
        return data.get(_record_key(component, model_key, bits))
    prefix = f"{component}__{model_key}__"
    # remainder after the prefix is the bits token (no further "__")
    matches = sorted(
        k for k in data if k.startswith(prefix) and "__" not in k[len(prefix):]
    )
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(
            "ambiguous calibration precision for "
            f"{component}/{model_key}@{hardware_key}: {matches}; "
            "declare module.bits explicitly"
        )
    chosen = matches[0]
    return data.get(chosen)


def load_record(
    component: str,
    model_key: str,
    hardware_key: str,
    bits: Optional[str] = None,
    directory: Optional[Path] = None,
) -> Optional[CalibrationRecord]:
    """Load one calibration record from the system file ``<hw>.json``.

    Records live keyed by ``<module>__<model>__<bits>`` inside the per-system
    JSON; the whole file is module-level cached. Lookup order
    (specialized → agnostic), both within the SAME ``<hw>.json``:
      1. key ``<component>__<model>__<bits>``     — module-specific
      2. key ``agnostic__<model>__<bits>``          — component-agnostic fallback

    The ``agnostic`` sentinel exists so a single calibration of a
    particular model (e.g. Qwen3-8B on H100 in bf16) can serve multiple
    components that wrap the same weights identically — typical for
    LLM backends (vllm / openai_llm / llama_index_llm). For rerankers
    where each component uses a different runtime wrapper
    (FlagEmbedding vs OpenVINO vs raw HF), keep per-component records
    so the specialized path wins.

    When ``bits`` is ``None`` we accept exactly one matching precision at each
    lookup level. Multiple precision records are ambiguous and raise instead
    of silently preferring one.
    """
    directory = Path(directory) if directory else _default_calibration_dir()

    # 1. Specific
    raw = _lookup_record(directory, component, model_key, hardware_key, bits)
    # 2. Fallback: agnostic (only if specific missed and component != agnostic)
    if raw is None and component != AGNOSTIC_MODULE_KEY:
        raw = _lookup_record(directory, AGNOSTIC_MODULE_KEY, model_key, hardware_key, bits)
    if not raw:
        return None
    return CalibrationRecord(**raw)


def invalidate_cache() -> None:
    """Clear the load cache (for tests and CLI refresh)."""
    _LOADED_CACHE.clear()


__all__ = [
    "BaseBenchmarkRunner",
    "StaticRAGEvaluatorBenchmarkRunner",
    "BenchmarkPoint",
    "BenchmarkReport",
    "CalibrationRecord",
    "GENERATOR_COMPONENT_FIT_LATENCY_SOURCE",
    "CAPACITY_HINGE_LAW_COMPONENT_CONTRACT",
    "detect_hardware_key",
    "fit_and_save",
    "genz_system_dict",
    "get_cuda_name",
    "grid_seq_axis",
    "invalidate_cache",
    "load_record",
    "save_record",
    "split_fit_validation_evidence",
    "run_benchmark_report",
    "_hardware_to_dict",
    "_inject_shape",
    "_normalize_hardware_key",
]

# Re-export bases for convenience
from rag_stack.cost_model.llm_sim.calibration.base import (  # noqa: E402
    BaseBenchmarkRunner,
    StaticRAGEvaluatorBenchmarkRunner,
)
