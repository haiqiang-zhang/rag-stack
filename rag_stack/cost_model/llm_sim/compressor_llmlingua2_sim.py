"""Dedicated cost model for the LLMLingua-2 passage compressor.

NOT a GenZ FLOP roofline — that's why it's a separate module from the GenZ
adapters in ``llm_sim/genz_adapter.py`` (it shares the ``llm_sim`` package but
is its own non-GenZ cost model). Profiling the production ``LLMLingua2.pure()`` path
shows it is CPU/Python-bound (per-token ``__merge_token_to_word`` + chunk /
tokenize + tiktoken ``get_token_length``); the batched BERT forward is only
~20-40% of wall time. Measured cost is ~linear in the number of chunks AND
~linear in chunk length, NOT FLOP-shaped — a GenZ fit gave 40% MAPE /
shape_mismatch. So we model it directly with a per-chunk linear law::

    latency_ms(n_chunks, chunk_tokens)
        = fixed_ms + n_chunks * (per_chunk_ms + per_token_ms * chunk_tokens)

3 coefficients are fit per hardware by a WALL-CLOCK linear regression
(:func:`calibrate_llmlingua2_linear`) and stored at
``calibration_profiles/compressor_llmlingua2/<hardware_key>.json``.

In the cost-model sweep, ``batch_size`` stays in USER-request units.  The
single request-shape helper below expands that public axis into
``batch_size * effective_top_k * elems_per_passage`` BERT work units before
evaluating the linear law.  Multi-GPU is DATA-PARALLEL (one BERT replica per
card): we emit single-chip rows and RAGO's
``scale_up_performance_to_more_chips`` extrapolates qps×N.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from rag_stack.calibration_paths import atomic_write_json, calibration_profile_root

from rag_stack.cost_model import register_cost_model
from rag_stack.cost_model.base import CostModelModule
from rag_stack.cost_model.calibration_mode import is_uncalibrated_spec_mode
from rag_stack.cost_model.types import CostModelResult, OperatorBreakdown
from rag_stack.types import SystemConfig

logger = logging.getLogger("RAG-Stack")

# Exact per-hardware profiles only. Missing or malformed component evidence is
# a pricing error; coefficients are never borrowed from another GPU.
_PROFILE_SUBDIR = "compressor_llmlingua2"
_PROFILE_KIND = "compressor_llmlingua2"
_PROFILE_SERVICE_UNIT = "milliseconds_per_fused_compressor_call"
_PROFILE_DEMAND_AXES = ("n_chunks", "chunk_tokens")
_PROFILE_ALLOWED_FIELDS = frozenset({
    "profile_kind", "service_unit",
    "demand_axes", "model", "hardware_key", "cpu_hardware_key",
    "fixed_ms", "per_chunk_ms",
    "per_token_ms", "fit_mape", "correlation",
    "grid", "note",
})
_PROFILE_GRID_FIELDS = frozenset({
    "n_chunks", "chunk_tokens", "measured_ms", "predicted_ms", "rel_err",
})
_MAX_CHUNK_TOKENS = 510  # bert max_seq_len (512) - CLS - SEP
_SPEC_FALLBACK_WARNED: set[str] = set()


class LLMLingua2ProfileError(ValueError):
    """An exact LLMLingua2 component profile is absent or invalid."""


@dataclass(frozen=True)
class CompressorRequestBatchShape:
    """LLMLingua work derived only from user batch, schema, and algorithm.

    ``request_batch`` stays in user-query units.  Each query contributes the
    effective retrieval ``top_k`` passages; each passage contributes the
    corpus-derived number of LLMLingua/BERT units.  ``unit_tokens`` is in the
    compressor's BERT payload-token unit, not the generator/trace tokenizer.
    """

    request_batch: int
    effective_top_k: int
    units_per_passage: float
    unit_tokens: int

    @property
    def units_per_query(self) -> float:
        return float(self.effective_top_k) * float(self.units_per_passage)

    @property
    def total_units(self) -> float:
        return float(self.request_batch) * self.units_per_query

    @property
    def total_bert_tokens(self) -> float:
        return self.total_units * float(self.unit_tokens)


@dataclass(frozen=True)
class CompressorOperatorPlan:
    """Structural LLMLingua work priced by one fused-call profile.

    ``forward_tile_units`` mirrors runtime tiling for auditability.  The active
    calibration is intentionally aggregate: no unconsumed CPU/GPU phase
    coefficients are accepted until the DES has a real phase-DAG consumer.
    """

    shape: CompressorRequestBatchShape
    forward_cap: int
    forward_tile_units: tuple[float, ...]
    service_s: float
    pricing_source: str = "calibrated_profile"


def llmlingua2_request_batch_shape(
    pipeline_config: dict,
    request_batch: int,
) -> CompressorRequestBatchShape:
    """Resolve one request batch's portable LLMLingua input work.

    The quality trace contributes invocation order/count only.  In particular,
    compressor ``input_tokens``/``input_bytes`` are not accepted here: archived
    traces used a different tokenizer and some producers recorded compressed
    output text at that call position.  The stable work units are already in
    ``WorkflowSchema.corpus_stats`` (materialized as ``_token_stats`` at this
    internal boundary) plus the algorithm's effective retrieval ``top_k``.
    """
    request_batch = int(request_batch)
    if request_batch < 1:
        raise ValueError(
            f"LLMLingua request_batch must be positive, got {request_batch}"
        )

    from rag_stack.cost_model.sweep_inputs import (
        _get_compressor_input_fanout,
    )
    from rag_stack.cost_model.token_stats import TokenStats

    stats = TokenStats.resolve(pipeline_config)
    effective_top_k = max(
        1, int(_get_compressor_input_fanout(pipeline_config)),
    )
    units_per_passage = getattr(
        stats, "avg_compressor_elems_per_chunk", None,
    )
    unit_tokens = getattr(stats, "avg_compressor_elem_tokens", None)
    try:
        units_per_passage = float(units_per_passage)
        unit_tokens = int(unit_tokens)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "LLMLingua2 requires WorkflowSchema corpus statistics "
            "avg_compressor_elems_per_chunk and avg_compressor_elem_tokens; "
            "the legacy one-passage fallback is retired"
        ) from exc
    if (
        not math.isfinite(units_per_passage)
        or units_per_passage <= 0.0
        or unit_tokens <= 0
    ):
        raise ValueError(
            "LLMLingua2 compressor corpus statistics must be positive and finite"
        )

    return CompressorRequestBatchShape(
        request_batch=request_batch,
        effective_top_k=effective_top_k,
        units_per_passage=units_per_passage,
        unit_tokens=min(unit_tokens, _MAX_CHUNK_TOKENS),
    )


# ---------------------------------------------------------------------------
# Pipeline-config helpers
# ---------------------------------------------------------------------------

def _find_compressor_module(pipeline_config: dict) -> dict:
    for node_line in pipeline_config.get("node_lines", []):
        for node in node_line.get("nodes", []):
            if node.get("stage") == "passage_compressor":
                modules = node.get("modules", [])
                if modules:
                    return modules[0]
    return {}


# ---------------------------------------------------------------------------
# Coefficient profile
# ---------------------------------------------------------------------------

def _profile_path(hardware) -> Path:
    from rag_stack.cost_model.llm_sim.calibration import detect_hardware_key
    return (
        calibration_profile_root()
        / _PROFILE_SUBDIR
        / f"{detect_hardware_key(hardware)}.json"
    )


def load_llmlingua2_coeffs(hardware) -> dict:
    """Load a schema-valid aggregate law for the exact deployed GPU.

    There is deliberately no generic or cross-hardware fallback.  In
    particular, an H100 deployment must never inherit the historical RTX 3090
    constants when its own artifact is absent or malformed: that makes an
    unpriced candidate look valid and violates the calibration identity
    contract.  The caller gets a diagnostic exception instead.
    """
    from rag_stack.cost_model.llm_sim.calibration import detect_hardware_key

    hardware_key = str(detect_hardware_key(hardware) or "").strip()
    cpu_key = str(
        getattr(hardware, "cpu_key", "")
        or getattr(hardware, "cpu_preset", "")
        or ""
    ).strip()
    if not cpu_key:
        raise LLMLingua2ProfileError(
            "LLMLingua2 aggregate calibration requires an exact CPU "
            "hardware_key because its service includes host/Python work"
        )
    f = _profile_path(hardware)
    if not f.is_file():
        raise LLMLingua2ProfileError(
            f"missing LLMLingua2 calibration profile for exact GPU "
            f"{hardware_key!r}: {f}; run calibrate_llmlingua2_linear on that "
            "hardware"
        )
    try:
        document = json.loads(f.read_text())
    except json.JSONDecodeError as exc:
        raise LLMLingua2ProfileError(
            f"invalid JSON in LLMLingua2 calibration profile {f}: {exc}"
        ) from exc
    except OSError as exc:
        raise LLMLingua2ProfileError(
            f"cannot read LLMLingua2 calibration profile {f}: {exc}"
        ) from exc
    return _validate_llmlingua2_profile(
        document,
        expected_hardware_key=hardware_key,
        expected_cpu_key=cpu_key,
        path=f,
    )


def _uncalibrated_spec_forward_service_s(
    hardware: SystemConfig,
    *,
    forward_tile_units: tuple[float, ...],
    unit_tokens: int,
) -> float:
    """Price only LLMLingua2's BERT forward from physical GPU specs.

    The exact LLMLingua2 profile measures the complete CPU+GPU wrapper and is
    always preferred.  There is no honest way to infer its Python/tokenizer
    service from a hardware datasheet, so the explicit uncalibrated mode uses
    the identifiable BERT-base GPU work as a labelled lower bound.  No fitted
    constant, cross-hardware profile, or end-to-end correction is introduced.
    """

    from rag_stack.cost_model.llm_sim.calibration import (
        detect_hardware_key,
        genz_system_dict,
    )
    from rag_stack.cost_model.llm_sim.genz_cache import (
        cached_prefill_moddeling,
    )
    from rag_stack.cost_model.llm_sim.genz_model_map import resolve_genz_model

    system_name = genz_system_dict(hardware)
    model = resolve_genz_model("llmlingua2_bert_base")
    total_s = 0.0
    for expected_units in forward_tile_units:
        # Corpus statistics can make the final tile fractional.  GenZ requires
        # an integer resident batch; price the enclosing batch and scale only
        # that final expected tile by its occupied fraction.
        resident_batch = max(1, int(math.ceil(float(expected_units))))
        output = cached_prefill_moddeling(
            model=model,
            batch_size=resident_batch,
            input_tokens=int(unit_tokens),
            system_name=system_name,
            bits="bf16",
            ceff=1.0,
            meff=1.0,
            tensor_parallel=1,
            pipeline_parallel=1,
            model_offload=True,
        )
        tile_s = float(output["Latency"]) / 1000.0
        tile_s *= float(expected_units) / float(resident_batch)
        if not math.isfinite(tile_s) or tile_s <= 0.0:
            raise LLMLingua2ProfileError(
                "uncalibrated_spec LLMLingua2 BERT forward returned a "
                f"non-positive service for batch {resident_batch}: {tile_s!r}"
            )
        total_s += tile_s

    if not math.isfinite(total_s) or total_s <= 0.0:
        raise LLMLingua2ProfileError(
            "uncalibrated_spec LLMLingua2 BERT forward produced no service"
        )
    hardware_key = str(detect_hardware_key(hardware))
    if hardware_key not in _SPEC_FALLBACK_WARNED:
        _SPEC_FALLBACK_WARNED.add(hardware_key)
        logger.warning(
            "No exact LLMLingua2 aggregate profile for GPU %r; "
            "uncalibrated_spec is using the physical BERT-forward lower "
            "bound. Host tokenization/Python work is not calibrated.",
            hardware_key,
        )
    return total_s


def _component_service_s(
    pipeline_config: dict,
    hardware: SystemConfig,
    *,
    total_units: float,
    unit_tokens: int,
    forward_cap: int,
) -> tuple[float, str]:
    """Resolve exact aggregate evidence or the explicit analytical fallback."""

    path = _profile_path(hardware)
    # Any filesystem entry, including a directory or broken symlink, is
    # present evidence and must enter the strict loader so corruption is not
    # disguised as an absent profile.
    profile_entry_exists = path.exists() or path.is_symlink()
    if profile_entry_exists or not is_uncalibrated_spec_mode(
        pipeline_config, hardware,
    ):
        coeffs = load_llmlingua2_coeffs(hardware)
        service_s = PassageCompressorLLMLingua2CostModel._latency_ms(
            coeffs, total_units, unit_tokens,
        ) / 1000.0
        return service_s, "calibrated_profile"

    tiles = llmlingua2_forward_tile_units(total_units, forward_cap)
    return (
        _uncalibrated_spec_forward_service_s(
            hardware,
            forward_tile_units=tiles,
            unit_tokens=unit_tokens,
        ),
        "uncalibrated_spec_bert_forward_lower_bound",
    )


def _profile_number(
    document: Mapping[str, Any],
    field: str,
    *,
    path: Path,
    positive: bool = False,
) -> float:
    value = document.get(field)
    if isinstance(value, bool):
        raise LLMLingua2ProfileError(
            f"{path}: {field} must be a finite "
            f"{'positive' if positive else 'non-negative'} number"
        )
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LLMLingua2ProfileError(
            f"{path}: {field} must be a finite "
            f"{'positive' if positive else 'non-negative'} number"
        ) from exc
    if not math.isfinite(number) or number < 0.0 or (positive and number <= 0.0):
        raise LLMLingua2ProfileError(
            f"{path}: {field} must be a finite "
            f"{'positive' if positive else 'non-negative'} number"
        )
    return number


def _validate_llmlingua2_profile(
    document: Any,
    *,
    expected_hardware_key: str,
    expected_cpu_key: str,
    path: Path,
) -> dict:
    """Validate identity, units, fit evidence, and aggregate coefficients."""
    if not isinstance(document, Mapping):
        raise LLMLingua2ProfileError(f"{path}: profile root must be a JSON object")
    unknown_fields = sorted(set(document) - _PROFILE_ALLOWED_FIELDS)
    if unknown_fields:
        raise LLMLingua2ProfileError(
            f"{path}: unknown profile fields {unknown_fields}; update the "
            "current exact profile contract before adding a runtime coefficient"
        )

    expected_tags = {
        "profile_kind": _PROFILE_KIND,
        "service_unit": _PROFILE_SERVICE_UNIT,
        "model": "llmlingua2",
        "hardware_key": expected_hardware_key,
        "cpu_hardware_key": expected_cpu_key,
    }
    for field, expected in expected_tags.items():
        if document.get(field) != expected:
            raise LLMLingua2ProfileError(
                f"{path}: {field} must be {expected!r}, got "
                f"{document.get(field)!r}"
            )
    if tuple(document.get("demand_axes") or ()) != _PROFILE_DEMAND_AXES:
        raise LLMLingua2ProfileError(
            f"{path}: demand_axes must be {list(_PROFILE_DEMAND_AXES)!r}"
        )

    coefficients = {
        "fixed_ms": _profile_number(document, "fixed_ms", path=path),
        "per_chunk_ms": _profile_number(
            document, "per_chunk_ms", path=path, positive=True,
        ),
        "per_token_ms": _profile_number(document, "per_token_ms", path=path),
    }
    _profile_number(document, "fit_mape", path=path)
    # Correlation is signed. Negative values remain schema-valid evidence for a
    # poor fit and must not be rewritten into a coefficient fallback.
    raw_correlation = document.get("correlation")
    if isinstance(raw_correlation, bool):
        raise LLMLingua2ProfileError(
            f"{path}: correlation must be finite and in [-1, 1]"
        )
    try:
        correlation = float(raw_correlation)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise LLMLingua2ProfileError(
            f"{path}: correlation must be finite and in [-1, 1]"
        ) from exc
    if not math.isfinite(correlation) or not -1.0 <= correlation <= 1.0:
        raise LLMLingua2ProfileError(
            f"{path}: correlation must be finite and in [-1, 1]"
        )

    grid = document.get("grid")
    if not isinstance(grid, list) or not grid:
        raise LLMLingua2ProfileError(
            f"{path}: grid must be a non-empty standalone measurement list"
        )
    for index, point in enumerate(grid):
        if not isinstance(point, Mapping):
            raise LLMLingua2ProfileError(
                f"{path}: grid[{index}] must be a JSON object"
            )
        unknown_point_fields = sorted(set(point) - _PROFILE_GRID_FIELDS)
        if unknown_point_fields:
            raise LLMLingua2ProfileError(
                f"{path}: grid[{index}] has unknown fields "
                f"{unknown_point_fields}"
            )
        for field, positive in (
            ("n_chunks", True),
            ("chunk_tokens", True),
            ("measured_ms", True),
            ("predicted_ms", False),
        ):
            _profile_number(
                point, field, path=path, positive=positive,
            )
        rel_err = point.get("rel_err")
        if isinstance(rel_err, bool):
            raise LLMLingua2ProfileError(
                f"{path}: grid[{index}].rel_err must be finite"
            )
        try:
            rel_err_number = float(rel_err)
        except (TypeError, ValueError, OverflowError) as exc:
            raise LLMLingua2ProfileError(
                f"{path}: grid[{index}].rel_err must be finite"
            ) from exc
        if not math.isfinite(rel_err_number):
            raise LLMLingua2ProfileError(
                f"{path}: grid[{index}].rel_err must be finite"
            )
    return coefficients


def llmlingua2_forward_tile_units(
    total_units: float,
    forward_cap: int,
) -> tuple[float, ...]:
    """Tile expected BERT work exactly like the runtime DataLoader.

    Corpus statistics may make ``total_units`` fractional because it is an
    expectation over passages.  Keeping a fractional final tile preserves
    that expectation instead of rounding the workload up to an unobserved
    extra passage.
    """
    total = float(total_units)
    cap = int(forward_cap)
    if not math.isfinite(total) or total < 0.0:
        raise ValueError(
            "LLMLingua total_units must be finite/nonnegative: "
            f"{total_units!r}"
        )
    if cap < 1:
        raise ValueError(
            f"LLMLingua forward_cap must be positive: {forward_cap!r}"
        )
    if total == 0.0:
        return ()
    full_tiles = int(math.floor(total / cap))
    remainder = total - full_tiles * cap
    # Protect exact multiples from a tiny floating-point tail, and values just
    # below an integer multiple from accidentally losing real expected work.
    if math.isclose(remainder, 0.0, rel_tol=0.0, abs_tol=1e-12):
        remainder = 0.0
    tiles = [float(cap)] * full_tiles
    if remainder > 0.0:
        tiles.append(remainder)
    return tuple(tiles)


def llmlingua2_forward_cap(
    pipeline_config: dict,
    request_batch: int,
    explicit_cap: Optional[int] = None,
) -> int:
    """Resolve the chunks-per-BERT-forward system knob.

    Production injects ``system.compressor_max_batch_size`` into LLMLingua's
    ``max_batch_size``.  When it is absent, the runtime uses the global request
    batch.  An unresolved sweep container is not a concrete deployment and is
    therefore ignored here; candidate materialization supplies a scalar.
    """
    raw: Any = explicit_cap
    if raw is None:
        system = pipeline_config.get("system") or {}
        if isinstance(system, Mapping):
            raw = system.get("compressor_max_batch_size")
    if raw is None or isinstance(raw, (bool, Mapping, list, tuple, set)):
        return max(1, int(request_batch))
    try:
        cap = int(raw)
    except (TypeError, ValueError, OverflowError):
        return max(1, int(request_batch))
    try:
        if float(raw) != float(cap):
            return max(1, int(request_batch))
    except (TypeError, ValueError, OverflowError):
        return max(1, int(request_batch))
    return max(1, cap)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------

@register_cost_model("passage_compressor_llmlingua2_sim")
class PassageCompressorLLMLingua2CostModel(CostModelModule):
    """Per-chunk linear cost model for LLMLingua-2 (see module docstring)."""

    @classmethod
    def family(cls) -> str:
        return "llm_sim"  # UI grouping: it's a (non-LLM) compressor cost model

    @classmethod
    def capabilities(cls) -> list[str]:
        return ["calibration_profiles"]

    @classmethod
    def profiles_subdir(cls) -> str | None:
        return _PROFILE_SUBDIR

    def _find_module(self, pipeline_config: dict) -> dict:
        return _find_compressor_module(pipeline_config)

    # The legacy unit-level estimate() remains linear in its module batch.
    # Serving consumers use estimate_request_batch(), which keeps the public
    # batch in user rows and charges one fixed_ms intercept per fused pure().
    # The flag remains for compatible callers of the low-level entry.
    linear_in_batch = True

    def _unit_shape(self, pipeline_config: dict) -> tuple:
        """(units per passage, BERT tokens per unit) — the shape the runtime
        actually pays for. llmlingua2 splits every passage on "\\n\\n" and
        runs per-ELEM (~5 elems x ~72 tokens for a 2048-char chunker, NOT
        1 passage x 326 generator-tokens); pricing passages-as-chunks was
        3.4x optimistic (r12 compressor investigation). Elem stats are
        corpus+chunker-derived by TokenStats and are required."""
        shape = llmlingua2_request_batch_shape(pipeline_config, 1)
        return shape.units_per_passage, shape.unit_tokens

    @staticmethod
    def _latency_ms(coeffs: dict, n_units: float, unit_tokens: int) -> float:
        return coeffs["fixed_ms"] + n_units * (
            coeffs["per_chunk_ms"] + coeffs["per_token_ms"] * unit_tokens
        )

    def build_request_operator_plan(
        self,
        pipeline_config: dict,
        hardware: SystemConfig,
        request_batch: int,
        *,
        forward_cap: Optional[int] = None,
    ) -> CompressorOperatorPlan:
        """Build runtime tiling and the aggregate fused-call service."""
        if not self._find_module(pipeline_config):
            raise RuntimeError(
                "PassageCompressorLLMLingua2CostModel."
                "build_request_operator_plan(): no "
                "passage_compressor(llmlingua2) module in pipeline_config."
            )
        shape = llmlingua2_request_batch_shape(
            pipeline_config, request_batch,
        )
        cap = llmlingua2_forward_cap(
            pipeline_config, shape.request_batch, forward_cap,
        )
        tiles = llmlingua2_forward_tile_units(shape.total_units, cap)
        aggregate_s, pricing_source = _component_service_s(
            pipeline_config,
            hardware,
            total_units=shape.total_units,
            unit_tokens=shape.unit_tokens,
            forward_cap=cap,
        )
        return CompressorOperatorPlan(
            shape=shape,
            forward_cap=cap,
            forward_tile_units=tiles,
            service_s=aggregate_s,
            pricing_source=pricing_source,
        )

    def estimate_request_batch(
        self,
        pipeline_config: dict,
        hardware: SystemConfig,
        request_batch: int,
    ) -> CostModelResult:
        """Price a fused batch whose public axis is USER queries.

        This is the authoritative serving entry used by both the RAGO sweep
        and trace-driven CallLatency/DES.  It charges the per-call intercept
        once, then the exact schema-derived compressor units for all queries.
        """
        if not self._find_module(pipeline_config):
            raise RuntimeError(
                "PassageCompressorLLMLingua2CostModel.estimate_request_batch(): "
                "no passage_compressor(llmlingua2) module in pipeline_config."
            )
        plan = self.build_request_operator_plan(
            pipeline_config, hardware, request_batch,
        )
        shape = plan.shape
        t_s = plan.service_s
        return CostModelResult(
            latency_s=t_s,
            throughput_qps=(
                shape.request_batch / t_s if t_s > 0 else 0.0
            ),
            operators=[OperatorBreakdown(
                name="llmlingua2_compress",
                time_s=t_s,
                compute_time_s=t_s,
                bottleneck="compress",
            )],
        )

    def estimate(self, pipeline_config: dict, hardware: SystemConfig) -> CostModelResult:
        if not self._find_module(pipeline_config):
            raise RuntimeError(
                "PassageCompressorLLMLingua2CostModel.estimate(): no "
                "passage_compressor(llmlingua2) module in pipeline_config."
            )
        units_per_passage, unit_tokens = self._unit_shape(pipeline_config)
        # Legacy low-level entry: module ``batch_size`` is already in
        # compressor-unit/passages units.  Serving consumers must use
        # ``estimate_request_batch`` so effective top_k is included exactly
        # once while their public batch axis remains USER queries.
        bs = int(self._find_module(pipeline_config).get("batch_size", 1))
        t_s, _pricing_source = _component_service_s(
            pipeline_config,
            hardware,
            total_units=bs * units_per_passage,
            unit_tokens=unit_tokens,
            forward_cap=llmlingua2_forward_cap(pipeline_config, bs),
        )
        return CostModelResult(
            latency_s=t_s,
            throughput_qps=bs / t_s if t_s > 0 else 0.0,
            operators=[OperatorBreakdown(
                name="llmlingua2_compress", time_s=t_s,
                compute_time_s=t_s, bottleneck="compress",
            )],
        )

    def generate_sweep_df(
        self, pipeline_config: dict, hardware: SystemConfig,
        stage: str, max_batch_size: int,
    ) -> pd.DataFrame:
        from rago.utils import get_power_of_two_list
        if not self._find_module(pipeline_config):
            raise RuntimeError(
                f"PassageCompressorLLMLingua2CostModel.generate_sweep_df"
                f"(stage={stage!r}): no llmlingua2 compressor in pipeline_config."
            )
        # Single-chip rows only (num_chips=1); RAGO scale_up extrapolates
        # multi-GPU as DATA-PARALLEL (qps×N) — the real llmlingua2 mode.
        rows = []
        for bs in get_power_of_two_list(max_batch_size):
            plan = self.build_request_operator_plan(
                pipeline_config, hardware, bs,
            )
            shape = plan.shape
            rows.append({
                "model.name": "llmlingua2",
                "model.stage": "prefill",
                "model.dec_steps": 1,
                "model.seq_len_inference_prefill": shape.unit_tokens,
                "hardware.name": "system",
                "hardware.num_chips": 1,
                # RAGO's batch axis remains USER requests. The service time
                # below already includes top_k * elems_per_passage units.
                "batch_size": bs,
                "time_ms": plan.service_s * 1000.0,
                "winning_tp": 1,
                "winning_pp": 1,
            })
        # Keep every requested user-batch row. The generic Pareto helper treats
        # an almost-linear 2B/B latency ratio as "no throughput improvement"
        # within a 1% tolerance and truncates valid B32/B64 rows. Fixed replay
        # needs the exact deployed request batch, and this closed-form sweep is
        # only O(log B) rows, so approximation/truncation has no value here.
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration — WALL-CLOCK linear fit (NOT the GenZ ceff/meff framework)
# ---------------------------------------------------------------------------

def _build_one_chunk_passage(tokenizer, n_tokens: int) -> str:
    """Text that tokenizes to ~``n_tokens`` BERT tokens as a SINGLE chunk
    (keep ``n_tokens <= 510`` so ``__chunk_context`` does not split it)."""
    base = tokenizer.tokenize("the quick brown fox jumps over the lazy dog " * 200)
    return tokenizer.convert_tokens_to_string(base[: max(1, n_tokens)])


def _median_wall_ms(fn, warmup: int, measurements: int) -> float:
    import torch
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    samples = []
    for _ in range(measurements):
        s = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - s) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def calibrate_llmlingua2_linear(
    hardware,
    *,
    device: str = "cuda:0",
    batch_sizes=(1, 2, 4, 8, 16, 32),
    seq_lens=(64, 128, 256, 448),
    warmup: int = 3,
    measurements: int = 10,
    output_dir=None,
) -> dict:
    """Measure WALL-CLOCK ``pure()`` over a (n_chunks, chunk_tokens) grid,
    least-squares fit (fixed_ms, per_chunk_ms, per_token_ms), save the profile
    JSON, and return the record.

    Wall-clock (``time.perf_counter``), NOT CUDA events — the dominant cost is
    Python, which CUDA-event timing (used for GPU-bound rerankers) under-measures.
    """
    import numpy as np
    import torch
    from rag_stack_evaluator.static_rag_evaluator.nodes.passagecompressor.llmlingua2 import LLMLingua2
    from rag_stack.cost_model.llm_sim.calibration import detect_hardware_key

    module = LLMLingua2("/tmp", device=device)  # cache=None → single replica
    tokenizer = module.llm_lingua.tokenizer

    X, y, grid = [], [], []
    for seq in seq_lens:
        passage = _build_one_chunk_passage(tokenizer, seq)
        for bs in batch_sizes:
            df = pd.DataFrame({"query": ["q"], "retrieved_contents": [[passage] * bs]})

            def _call(df=df, bs=bs):
                module.pure(df.copy(deep=True), rate=0.5,
                            query_batch_size=1, max_batch_size=bs)

            t_ms = _median_wall_ms(_call, warmup, measurements)
            # time = fixed + per_chunk*n_chunks + per_token*(n_chunks*tokens)
            X.append([1.0, float(bs), float(bs * seq)])
            y.append(t_ms)
            grid.append({"n_chunks": bs, "chunk_tokens": seq, "measured_ms": round(t_ms, 3)})
            logger.info(f"llmlingua2 calib bs={bs} seq={seq}: {t_ms:.2f} ms")

    X = np.asarray(X)
    y = np.asarray(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fixed_ms, per_chunk_ms, per_token_ms = (float(c) for c in coef)
    pred = X @ coef
    mape = float(np.mean(np.abs((pred - y) / np.maximum(y, 1e-9))))
    corr = float(np.corrcoef(pred, y)[0, 1]) if len(y) > 1 else 0.0
    for g, p in zip(grid, pred):
        g["predicted_ms"] = round(float(p), 3)
        g["rel_err"] = round(float((p - g["measured_ms"]) / max(g["measured_ms"], 1e-9)), 4)

    hw_key = detect_hardware_key(hardware)
    cpu_key = str(
        getattr(hardware, "cpu_key", "")
        or getattr(hardware, "cpu_preset", "")
        or ""
    ).strip()
    if not cpu_key:
        raise ValueError(
            "LLMLingua2 calibration requires system.cpu.hardware_key"
        )
    rec = {
        "profile_kind": _PROFILE_KIND,
        "service_unit": _PROFILE_SERVICE_UNIT,
        "demand_axes": list(_PROFILE_DEMAND_AXES),
        "model": "llmlingua2",
        "hardware_key": hw_key,
        "cpu_hardware_key": cpu_key,
        "fixed_ms": fixed_ms,
        "per_chunk_ms": per_chunk_ms,
        "per_token_ms": per_token_ms,
        "fit_mape": mape,
        "correlation": corr,
        "grid": grid,
        "note": (
            "Wall-clock LINEAR fit (NOT GenZ): latency_ms = fixed + "
            "n_chunks*(per_chunk + per_token*chunk_tokens). llmlingua2 is "
            "CPU/Python-bound on .pure() (per-token merge/regroup, chunk/tokenize, "
            "tiktoken); GPU forward ~20-40%. Cost is ~linear in #chunks and "
            "chunk_tokens, NOT FLOP-shaped — GenZ gave 40% MAPE."
        ),
    }
    out = (
        Path(output_dir)
        if output_dir
        else calibration_profile_root() / _PROFILE_SUBDIR
    )
    f = out / f"{hw_key}.json"
    _validate_llmlingua2_profile(
        rec,
        expected_hardware_key=hw_key,
        expected_cpu_key=cpu_key,
        path=f,
    )
    atomic_write_json(f, rec)
    logger.info(f"Saved llmlingua2 linear profile: {f} (MAPE {mape*100:.1f}%, corr {corr:.3f})")
    return rec
