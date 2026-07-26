"""Strict CPU-only evidence assembly for direct P/D stage curves.

The acquisition command owns GPUs and writes two role-local timing documents.
This module deliberately owns the immutable disk schema and every publication
decision.  It never launches vLLM.

Each batch has three independent parent-clock windows.  A prefill window owns
exactly four completed exact-B logical-prefill cohorts.  The high acquisition
token budget makes each cohort coincide with one pure-prefill scheduler cycle,
which proves its shape without making production scheduler chunks the charging
unit.  A decode window owns the first 32
interdeparture intervals wholly inside a saturated run of 33 consecutive pure
decode exact-B scheduler cycles.  The decode requests use a fixed 256-token
acquisition guard only so NIXL arrival skew cannot exhaust the first requests
before the exact-B run exists.  That guard is evidence protocol, never a
profile key, fit dimension, or calibration workload token shape.  Every
selected cycle carries its real scheduler shape and a globally unique
telemetry cycle ID.

The component reference is rebuilt only from the three measured cohorts named
by the existing component run contract.  Warmups and every unselected raw row
are therefore structurally unable to enter a fit.  Prefill requires an exact
component shape.  Decode permits only bracketed piecewise-linear interpolation
between adjacent component shapes with the same phase, batch and scheduled
token count; extrapolation is forbidden.

For each parent window, component services for all selected boundaries are
arithmetic-mean pooled and paired with the observed service per boundary.  For
direct-P that boundary is one completed cohort frontier; for direct-D it is an
engine cycle.  The serialized ``observed_service_per_cycle_s`` field name is
retained for compatibility with existing raw evidence, but does not redefine
the direct-P runtime boundary.  Any negative paired residual fails the complete
fit.  The middle of the three paired residual samples (stable tie break by
sample index) becomes
the :class:`StageBatchMeasurement`, preserving an actually observed total and
its corresponding component reference without fitting another coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable, Mapping, Sequence

from rag_stack.cost_model.llm_sim.calibration.main_llm.generator_request_contract import (
    generator_request_contract_sha256,
    validate_generator_request_contract,
)
from rag_stack.cost_model.stage_batch_curve import (
    ANCHOR_BATCHES,
    HELD_OUT_BATCH,
    StageBatchCurveKey,
    StageBatchCurveProfile,
    StageBatchMeasurement,
    build_stage_batch_curve,
    publish_stage_batch_curves,
    validate_stage_batch_curve_generation,
    validate_system_hardware,
)


ROLE_TIMING_SCHEMA = "rag_stack.direct_disagg_role_timing"
ROLE_RUN_CONTRACT_SCHEMA = "rag_stack.direct_disagg_role_run_contract"
_GLOBAL_FRONTEND_TELEMETRY_SCHEMA = (
    "rag_stack.vllm.frontend_stage_boundary_span"
)
_GLOBAL_SCHEDULER_TELEMETRY_SCHEMA = (
    "rag_stack.vllm.scheduler_cycle_stage_service"
)
REQUIRED_BATCHES: tuple[int, ...] = (1, 8, 32, 64, 128, 256)
SAMPLES_PER_BATCH = 3
PREFILL_CYCLES_PER_SAMPLE = 4
CALIBRATION_WORKLOAD_OUTPUT_TOKENS = 128
DECODE_INTERDEPARTURE_INTERVALS_PER_SAMPLE = 32
DECODE_CONSECUTIVE_EXACT_BATCH_CYCLES = (
    DECODE_INTERDEPARTURE_INTERVALS_PER_SAMPLE + 1
)
DECODE_ACQUISITION_GUARD_OUTPUT_TOKENS = 256
DECODE_ACQUISITION_GUARD_DISPOSITION = (
    "acquisition_liveness_only_not_profile_key_or_fit_dimension"
)
# Transitional name used by the acquisition driver.  It denotes exactly the
# 32 post-boundary cycles, not an open-ended minimum and not token offsets.
MIN_DECODE_CYCLES_PER_SAMPLE = DECODE_INTERDEPARTURE_INTERVALS_PER_SAMPLE
ROLE_AGGREGATION_CONTRACT = {
    "component_window_service": (
        "arithmetic_mean_of_selected_cycle_component_services"
    ),
    "sample_residual": (
        "observed_service_per_cycle_s_minus_component_window_service_s"
    ),
    "batch_measurement": (
        "middle_of_three_paired_residual_samples_tie_break_sample_index"
    ),
    "negative_residual": "fail_closed_no_clamp",
}
# Direct-P acquisition intentionally makes one exact-B completion cohort equal
# one selected pure scheduler cycle.  The frozen aggregation strings above are
# retained as raw-evidence compatibility identifiers; production applies the
# resulting P curve once per logical completion frontier, never per 8192-token
# scheduler chunk.  Direct-D retains the literal cycle interpretation.

_ROLE_TIMING_FIELDS = {
    "schema",
    "system_hardware",
    "role",
    "batches",
}
_ROLE_BATCH_FIELDS = {
    "target_batch",
    "engine_max_num_seqs",
    "telemetry_path",
    "telemetry_sha256",
    "telemetry_run_id",
    "timing_run_id",
    "samples",
}
_ROLE_SAMPLE_FIELDS = {
    "sample_index",
    "window_source",
    "window_started_monotonic_s",
    "window_completed_monotonic_s",
    "role_window_s",
    "scheduler_cycle_count",
    "observed_service_per_cycle_s",
    "selected_cycles",
    "window_proof",
}
_SELECTED_CYCLE_FIELDS = {"cycle_id", "actual_shape"}
_SHAPE_FIELDS = {
    "scheduled_sequences",
    "scheduled_tokens",
    "scheduled_context_tokens",
    "token_context_product",
}
_ROLE_CONTRACT_FIELDS = {
    "schema",
    "system_hardware",
    "role",
    "source_path",
    "source_sha256",
    "component_source_path",
    "component_source_sha256",
    "component_contract_path",
    "component_contract_sha256",
    "nixl_physical_path",
    "nixl_physical_sha256",
    "request_contract_sha256",
    "runs",
    "workload",
    "cycle_selection",
    "aggregation",
    "selected_cycle_ids",
    "component_joins",
}
_RUN_FIELDS = {"telemetry_path", "telemetry_sha256", "telemetry_run_id", "timing_run_id"}
_WORKLOAD_FIELDS = {
    "target_batches",
    "samples_per_batch",
    "prompt_tokens",
    "output_tokens",
    "request_contract",
}
_PREFILL_SELECTION_FIELDS = {"phase", "cycles_per_sample", "window_policy"}
_DECODE_SELECTION_FIELDS = {
    "phase",
    "interdeparture_intervals_per_sample",
    "consecutive_exact_batch_cycle_count",
    "acquisition_guard_output_tokens",
    "acquisition_guard_disposition",
    "window_policy",
    "clock",
}
PREFILL_WINDOW_SOURCE = "parent_resume_to_all_completion_four_exact_cycles"
DECODE_WINDOW_SOURCE = "scheduler_first_saturated_exact_batch_clock_monotonic"
DECODE_WINDOW_POLICY = "first_saturated_exact_batch_32_interdeparture_intervals"
DECODE_WINDOW_CLOCK = "CLOCK_MONOTONIC"
_DECODE_WINDOW_PROOF_FIELDS = {
    "boundary_start_cycle_id",
    "boundary_start_shape",
    "boundary_end_cycle_id",
    "consecutive_exact_batch_cycle_count",
    "context_step_per_interval",
}
_COMPONENT_JOIN_FIELDS = {
    "target_cycle_id",
    "target_shape",
    "bracket",
    "component_service_s",
}
_COMPONENT_BRACKET_FIELDS = {
    "component_cycle_ids",
    "shape",
    "weight",
    "median_component_service_s",
}
_NIXL_CAMPAIGN_FIELDS = {
    "schema",
    "system_hardware",
    "evidence_scope",
    "entries",
}
_NIXL_ENTRY_FIELDS = {
    "evidence_id",
    "target_batch",
    "cohort_index",
    "fit_eligible",
    "selected_sample_index",
    "role_window",
    "selected_cycle_ids",
    "before_prometheus",
    "before_prometheus_sha256",
    "after_prometheus",
    "after_prometheus_sha256",
    "physical_evidence",
    "cost_model_disposition",
}
_NIXL_ROLE_WINDOW_FIELDS = {
    "window_source",
    "window_started_monotonic_s",
    "window_completed_monotonic_s",
    "role_window_s",
}


class DirectDisaggEvidenceError(ValueError):
    """Direct P/D evidence is incomplete or violates its strict contract."""


def _require_fields(value: Any, expected: set[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise DirectDisaggEvidenceError(
            f"{label} fields differ from the exact schema "
            f"(missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)})"
        )
    return value


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise DirectDisaggEvidenceError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DirectDisaggEvidenceError(
            f"{label} must be a non-negative integer"
        )
    return int(value)


def _positive_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DirectDisaggEvidenceError(f"{label} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise DirectDisaggEvidenceError(f"{label} must be finite and positive")
    return result


def _identity(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise DirectDisaggEvidenceError(f"{label} must be a non-empty identity")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_existing_path(value: str | Path, *, label: str) -> Path:
    path = Path(value)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise DirectDisaggEvidenceError(f"{label} does not exist: {path}") from exc
    return resolved


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectDisaggEvidenceError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, Mapping):
        raise DirectDisaggEvidenceError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True, order=True)
class ActualCycleShape:
    scheduled_sequences: int
    scheduled_tokens: int
    scheduled_context_tokens: int
    token_context_product: int

    def __post_init__(self) -> None:
        sequences = _positive_int(
            self.scheduled_sequences, label="shape.scheduled_sequences"
        )
        tokens = _positive_int(
            self.scheduled_tokens, label="shape.scheduled_tokens"
        )
        context = _positive_int(
            self.scheduled_context_tokens,
            label="shape.scheduled_context_tokens",
        )
        product = _positive_int(
            self.token_context_product,
            label="shape.token_context_product",
        )
        object.__setattr__(self, "scheduled_sequences", sequences)
        object.__setattr__(self, "scheduled_tokens", tokens)
        object.__setattr__(self, "scheduled_context_tokens", context)
        object.__setattr__(self, "token_context_product", product)

    @classmethod
    def from_dict(cls, value: Any) -> "ActualCycleShape":
        raw = _require_fields(value, _SHAPE_FIELDS, label="actual_shape")
        return cls(**{name: raw[name] for name in _SHAPE_FIELDS})

    def to_dict(self) -> dict[str, int]:
        return {
            "scheduled_sequences": self.scheduled_sequences,
            "scheduled_tokens": self.scheduled_tokens,
            "scheduled_context_tokens": self.scheduled_context_tokens,
            "token_context_product": self.token_context_product,
        }


@dataclass(frozen=True)
class SelectedRoleCycle:
    cycle_id: str
    actual_shape: ActualCycleShape


@dataclass(frozen=True)
class DirectRoleWindowSample:
    sample_index: int
    window_source: str
    window_started_monotonic_s: float
    window_completed_monotonic_s: float
    role_window_s: float
    scheduler_cycle_count: int
    observed_service_per_cycle_s: float
    selected_cycles: tuple[SelectedRoleCycle, ...]
    window_proof: Mapping[str, Any]


@dataclass(frozen=True)
class DirectRoleBatchEvidence:
    target_batch: int
    engine_max_num_seqs: int
    telemetry_path: Path
    telemetry_sha256: str
    telemetry_run_id: str
    timing_run_id: str
    samples: tuple[DirectRoleWindowSample, ...]


@dataclass(frozen=True)
class DirectRoleEvidence:
    system_hardware: str
    role: str
    prompt_tokens: int
    output_tokens: int
    request_contract: Mapping[str, Any]
    batches: Mapping[int, DirectRoleBatchEvidence]
    source_path: Path
    source_sha256: str
    contract_path: Path
    contract_sha256: str
    component_source_path: Path
    component_source_sha256: str
    component_contract_path: Path
    component_contract_sha256: str
    nixl_physical_path: Path | None
    nixl_physical_sha256: str | None
    component_joins: Mapping[str, Any]


def _selection_contract(role: str) -> dict[str, Any]:
    if role == "prefill":
        return {
            "phase": "prefill",
            "cycles_per_sample": PREFILL_CYCLES_PER_SAMPLE,
            "window_policy": PREFILL_WINDOW_SOURCE,
        }
    if role == "decode":
        return {
            "phase": "decode",
            "interdeparture_intervals_per_sample": (
                DECODE_INTERDEPARTURE_INTERVALS_PER_SAMPLE
            ),
            "consecutive_exact_batch_cycle_count": (
                DECODE_CONSECUTIVE_EXACT_BATCH_CYCLES
            ),
            "acquisition_guard_output_tokens": (
                DECODE_ACQUISITION_GUARD_OUTPUT_TOKENS
            ),
            "acquisition_guard_disposition": (
                DECODE_ACQUISITION_GUARD_DISPOSITION
            ),
            "window_policy": DECODE_WINDOW_POLICY,
            "clock": DECODE_WINDOW_CLOCK,
        }
    raise DirectDisaggEvidenceError(f"unknown direct role {role!r}")


def validate_nixl_physical_campaign(
    path: str | Path,
    *,
    system_hardware: str,
    selected_samples_by_batch: Mapping[
        int, Sequence[DirectRoleWindowSample]
    ],
) -> Mapping[str, Any]:
    """Recompute every physical NIXL delta; never return a service value."""

    from .nixl_physical_evidence import (
        NixlPhysicalEvidenceError,
        derive_nixl_physical_evidence_from_text,
    )

    source = _canonical_existing_path(path, label="NIXL physical campaign")
    document = _require_fields(
        _read_json(source, label="NIXL physical campaign"),
        _NIXL_CAMPAIGN_FIELDS,
        label="NIXL physical campaign",
    )
    if (
        document.get("schema")
        != "rag_stack.direct_disagg_nixl_physical_campaign"
        or document.get("system_hardware") != system_hardware
        or document.get("evidence_scope")
        != "physical_only_no_role_service_fit"
    ):
        raise DirectDisaggEvidenceError(
            "unexpected NIXL physical campaign schema/hardware/scope"
        )
    expected_batch_keys = set(REQUIRED_BATCHES)
    if (
        not isinstance(selected_samples_by_batch, Mapping)
        or set(selected_samples_by_batch) != expected_batch_keys
    ):
        raise DirectDisaggEvidenceError(
            "NIXL validation requires selected timing IDs for every batch"
        )
    for batch in REQUIRED_BATCHES:
        samples = selected_samples_by_batch[batch]
        if (
            not isinstance(samples, Sequence)
            or len(samples) != SAMPLES_PER_BATCH
            or any(
                not isinstance(sample, DirectRoleWindowSample)
                or not sample.selected_cycles
                for sample in samples
            )
        ):
            raise DirectDisaggEvidenceError(
                f"NIXL B{batch} selected timing cycle IDs are invalid"
            )
    entries = document.get("entries")
    if (
        not isinstance(entries, list)
        or len(entries) != 4 * len(REQUIRED_BATCHES)
    ):
        raise DirectDisaggEvidenceError(
            "NIXL campaign must contain exactly one warmup and three "
            "measurement cohorts per batch"
        )
    seen_ids: set[str] = set()
    seen_pairs: set[tuple[int, int]] = set()
    for entry_index, entry in enumerate(entries):
        raw = _require_fields(
            entry, _NIXL_ENTRY_FIELDS, label=f"NIXL entry {entry_index}"
        )
        batch = _positive_int(raw.get("target_batch"), label="NIXL target_batch")
        cohort = _nonnegative_int(
            raw.get("cohort_index"), label="NIXL cohort_index"
        )
        if batch not in REQUIRED_BATCHES or cohort not in range(4):
            raise DirectDisaggEvidenceError(
                "NIXL entry has an unexpected batch/cohort identity"
            )
        pair = (batch, cohort)
        evidence_id = raw.get("evidence_id")
        if (
            evidence_id != f"b{batch}-c{cohort}"
            or evidence_id in seen_ids
            or pair in seen_pairs
        ):
            raise DirectDisaggEvidenceError(
                "NIXL evidence ID/batch/cohort is duplicated or inconsistent"
            )
        seen_ids.add(str(evidence_id))
        seen_pairs.add(pair)
        fit_eligible = raw.get("fit_eligible")
        selected_index = raw.get("selected_sample_index")
        entry_cycle_ids = raw.get("selected_cycle_ids")
        if cohort == 0:
            if (
                fit_eligible is not False
                or selected_index is not None
                or entry_cycle_ids != []
            ):
                raise DirectDisaggEvidenceError(
                    "NIXL warmup entry must be non-fit and unselected"
                )
        else:
            sample_index = cohort - 1
            if (
                fit_eligible is not True
                or selected_index != sample_index
                or entry_cycle_ids != [
                    cycle.cycle_id
                    for cycle in selected_samples_by_batch[batch][sample_index]
                    .selected_cycles
                ]
            ):
                raise DirectDisaggEvidenceError(
                    "NIXL measurement cohort does not bind its exact timing "
                    "sample IDs"
                )

        role_window = _require_fields(
            raw.get("role_window"),
            _NIXL_ROLE_WINDOW_FIELDS,
            label="NIXL role_window",
        )
        if role_window.get("window_source") != DECODE_WINDOW_SOURCE:
            raise DirectDisaggEvidenceError(
                "NIXL D cohort does not bind the scheduler-context window source"
            )
        started = _positive_float(
            role_window.get("window_started_monotonic_s"),
            label="NIXL role-window start",
        )
        completed = _positive_float(
            role_window.get("window_completed_monotonic_s"),
            label="NIXL role-window completion",
        )
        window = _positive_float(
            role_window.get("role_window_s"), label="NIXL role window"
        )
        if completed <= started or not math.isclose(
            window, completed - started, rel_tol=1e-12, abs_tol=1e-15
        ):
            raise DirectDisaggEvidenceError(
                "NIXL role window is internally inconsistent"
            )
        if cohort > 0:
            selected_sample = selected_samples_by_batch[batch][cohort - 1]
            if not (
                selected_sample.window_source == DECODE_WINDOW_SOURCE
                and
                math.isclose(
                    started,
                    selected_sample.window_started_monotonic_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                and math.isclose(
                    completed,
                    selected_sample.window_completed_monotonic_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                and math.isclose(
                    window,
                    selected_sample.role_window_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ):
                raise DirectDisaggEvidenceError(
                    "NIXL selected role window differs from timing sample"
                )
        before = raw.get("before_prometheus")
        after = raw.get("after_prometheus")
        if not isinstance(before, str) or not isinstance(after, str):
            raise DirectDisaggEvidenceError(
                "NIXL Prometheus snapshots must be text"
            )
        if raw.get("before_prometheus_sha256") != hashlib.sha256(
            before.encode("utf-8")
        ).hexdigest() or raw.get("after_prometheus_sha256") != hashlib.sha256(
            after.encode("utf-8")
        ).hexdigest():
            raise DirectDisaggEvidenceError(
                "NIXL Prometheus snapshot digest mismatch"
            )
        try:
            recomputed = derive_nixl_physical_evidence_from_text(
                before, after, expected_transfer_count=batch
            ).to_dict()
        except NixlPhysicalEvidenceError as exc:
            raise DirectDisaggEvidenceError(
                "NIXL physical delta is invalid"
            ) from exc
        if raw.get("physical_evidence") != recomputed:
            raise DirectDisaggEvidenceError(
                "stored NIXL physical evidence differs from strict recomputation"
            )
        if raw.get("cost_model_disposition") != (
            "physical_only_never_role_service_correction"
        ):
            raise DirectDisaggEvidenceError(
                "NIXL evidence attempts to enter role service calibration"
            )
    expected_pairs = {
        (batch, cohort) for batch in REQUIRED_BATCHES for cohort in range(4)
    }
    if seen_pairs != expected_pairs:
        raise DirectDisaggEvidenceError(
            "NIXL campaign does not exactly cover warmup+3 cohorts per batch"
        )
    return document


def _parse_selected_cycles(
    value: Any,
    *,
    role: str,
    target_batch: int,
    telemetry_run_id: str,
    prompt_tokens: int,
    seen_cycle_ids: set[str],
) -> tuple[SelectedRoleCycle, ...]:
    if not isinstance(value, list):
        raise DirectDisaggEvidenceError("selected_cycles must be a list")
    expected_count = PREFILL_CYCLES_PER_SAMPLE if role == "prefill" else None
    if expected_count is not None and len(value) != expected_count:
        raise DirectDisaggEvidenceError(
            f"prefill sample must select exactly {expected_count} cycles"
        )
    if (
        role == "decode"
        and len(value) != DECODE_INTERDEPARTURE_INTERVALS_PER_SAMPLE
    ):
        raise DirectDisaggEvidenceError(
            "decode sample must select exactly the 32 post-boundary "
            "interdeparture cycles"
        )
    selected: list[SelectedRoleCycle] = []
    decode_offsets: list[int] = []
    for index, item in enumerate(value):
        raw = _require_fields(
            item, _SELECTED_CYCLE_FIELDS, label=f"selected_cycles[{index}]"
        )
        cycle_id = _identity(raw["cycle_id"], label="telemetry cycle_id")
        if not cycle_id.startswith(f"{telemetry_run_id}:"):
            raise DirectDisaggEvidenceError(
                "selected telemetry cycle_id does not belong to its role engine run"
            )
        if cycle_id in seen_cycle_ids:
            raise DirectDisaggEvidenceError(
                f"selected telemetry cycle_id is reused: {cycle_id}"
            )
        seen_cycle_ids.add(cycle_id)
        shape = ActualCycleShape.from_dict(raw["actual_shape"])
        if shape.scheduled_sequences != target_batch:
            raise DirectDisaggEvidenceError(
                "selected scheduler cycle is not exact resident batch "
                f"B{target_batch}"
            )
        if role == "prefill":
            expected = ActualCycleShape(
                scheduled_sequences=target_batch,
                scheduled_tokens=target_batch * prompt_tokens,
                scheduled_context_tokens=target_batch * prompt_tokens,
                token_context_product=(
                    target_batch * prompt_tokens * prompt_tokens
                ),
            )
            if shape != expected:
                raise DirectDisaggEvidenceError(
                    "prefill selected cycle does not have the exact canonical shape"
                )
        else:
            if shape.scheduled_tokens != target_batch:
                raise DirectDisaggEvidenceError(
                    "decode selected cycle must schedule one token per sequence"
                )
            if shape.token_context_product != shape.scheduled_context_tokens:
                raise DirectDisaggEvidenceError(
                    "decode selected cycle has a non-canonical aggregate context"
                )
            decode_offsets.append(shape.scheduled_context_tokens)
        selected.append(SelectedRoleCycle(cycle_id, shape))
    if role == "decode" and any(
        right - left != target_batch
        for left, right in zip(decode_offsets, decode_offsets[1:])
    ):
        raise DirectDisaggEvidenceError(
            "decode selected cycles must retain one exact-B context step per "
            "interdeparture interval"
        )
    return tuple(selected)


def _parse_sample(
    value: Any,
    *,
    role: str,
    target_batch: int,
    telemetry_run_id: str,
    prompt_tokens: int,
    expected_sample_index: int,
    seen_cycle_ids: set[str],
) -> DirectRoleWindowSample:
    raw = _require_fields(
        value, _ROLE_SAMPLE_FIELDS, label=f"B{target_batch} sample"
    )
    sample_index = _nonnegative_int(raw["sample_index"], label="sample_index")
    if sample_index != expected_sample_index:
        raise DirectDisaggEvidenceError(
            f"B{target_batch} samples must be indexed 0..{SAMPLES_PER_BATCH - 1}"
        )
    expected_window_source = (
        PREFILL_WINDOW_SOURCE if role == "prefill" else DECODE_WINDOW_SOURCE
    )
    if raw.get("window_source") != expected_window_source:
        raise DirectDisaggEvidenceError(
            f"{role} sample has the wrong role-window source"
        )
    started = _positive_float(
        raw["window_started_monotonic_s"], label="role-window start"
    )
    completed = _positive_float(
        raw["window_completed_monotonic_s"], label="role-window completion"
    )
    if completed <= started:
        raise DirectDisaggEvidenceError(
            "role timing window must complete after it starts"
        )
    role_window = _positive_float(raw["role_window_s"], label="role window")
    if not math.isclose(
        role_window, completed - started, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise DirectDisaggEvidenceError(
            "role_window_s does not equal its frozen clock difference"
        )
    cycle_count = _positive_int(
        raw["scheduler_cycle_count"], label="scheduler_cycle_count"
    )
    selected = _parse_selected_cycles(
        raw["selected_cycles"],
        role=role,
        target_batch=target_batch,
        telemetry_run_id=telemetry_run_id,
        prompt_tokens=prompt_tokens,
        seen_cycle_ids=seen_cycle_ids,
    )
    if cycle_count != len(selected):
        raise DirectDisaggEvidenceError(
            "scheduler_cycle_count differs from selected cycle proof"
        )
    proof = raw.get("window_proof")
    if role == "prefill":
        if not isinstance(proof, Mapping) or proof:
            raise DirectDisaggEvidenceError(
                "prefill parent resume/completion window_proof must be empty"
            )
        normalized_proof: dict[str, Any] = {}
    else:
        proof = _require_fields(
            proof,
            _DECODE_WINDOW_PROOF_FIELDS,
            label="decode saturated exact-B window_proof",
        )
        consecutive_count = _positive_int(
            proof.get("consecutive_exact_batch_cycle_count"),
            label="decode consecutive_exact_batch_cycle_count",
        )
        if consecutive_count != DECODE_CONSECUTIVE_EXACT_BATCH_CYCLES:
            raise DirectDisaggEvidenceError(
                "decode proof must bind exactly 33 consecutive exact-B cycles"
            )
        context_step = _positive_int(
            proof.get("context_step_per_interval"),
            label="decode context_step_per_interval",
        )
        if context_step != target_batch:
            raise DirectDisaggEvidenceError(
                "decode proof context step must equal the resident batch"
            )
        boundary_start_id = _identity(
            proof.get("boundary_start_cycle_id"),
            label="decode boundary_start_cycle_id",
        )
        if (
            not boundary_start_id.startswith(f"{telemetry_run_id}:")
            or boundary_start_id in seen_cycle_ids
        ):
            raise DirectDisaggEvidenceError(
                "decode start boundary cycle identity is reused or foreign"
            )
        seen_cycle_ids.add(boundary_start_id)
        boundary_start_shape = ActualCycleShape.from_dict(
            proof.get("boundary_start_shape")
        )
        if (
            boundary_start_shape.scheduled_sequences != target_batch
            or boundary_start_shape.scheduled_tokens != target_batch
            or boundary_start_shape.token_context_product
            != boundary_start_shape.scheduled_context_tokens
        ):
            raise DirectDisaggEvidenceError(
                "decode start boundary is not a pure exact-B decode cycle"
            )
        if proof.get("boundary_end_cycle_id") != selected[-1].cycle_id:
            raise DirectDisaggEvidenceError(
                "decode end boundary must equal the final selected cycle"
            )
        contexts = [boundary_start_shape.scheduled_context_tokens, *(
            cycle.actual_shape.scheduled_context_tokens for cycle in selected
        )]
        if len(contexts) != consecutive_count or any(
            right - left != context_step
            for left, right in zip(contexts, contexts[1:])
        ):
            raise DirectDisaggEvidenceError(
                "decode boundary and selected cycles are not one consecutive "
                "exact-B context run"
            )
        normalized_proof = {
            "boundary_start_cycle_id": boundary_start_id,
            "boundary_start_shape": boundary_start_shape.to_dict(),
            "boundary_end_cycle_id": selected[-1].cycle_id,
            "consecutive_exact_batch_cycle_count": consecutive_count,
            "context_step_per_interval": context_step,
        }
    per_cycle = _positive_float(
        raw["observed_service_per_cycle_s"],
        label="observed_service_per_cycle_s",
    )
    if not math.isclose(
        per_cycle, role_window / cycle_count, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise DirectDisaggEvidenceError(
            "observed_service_per_cycle_s does not equal role_window/count"
        )
    return DirectRoleWindowSample(
        sample_index=sample_index,
        window_source=expected_window_source,
        window_started_monotonic_s=started,
        window_completed_monotonic_s=completed,
        role_window_s=role_window,
        scheduler_cycle_count=cycle_count,
        observed_service_per_cycle_s=per_cycle,
        selected_cycles=selected,
        window_proof=normalized_proof,
    )


def _validate_selected_cycles_against_telemetry(
    path: Path,
    *,
    telemetry_run_id: str,
    role: str,
    samples: Sequence[DirectRoleWindowSample],
) -> None:
    """Prove every selected ID/shape against the immutable scheduler JSONL."""

    expected = {
        cycle.cycle_id: cycle.actual_shape
        for sample in samples
        for cycle in sample.selected_cycles
    }
    expected_by_sample = [
        {cycle.cycle_id for cycle in sample.selected_cycles}
        for sample in samples
    ]
    observed: dict[str, ActualCycleShape] = {}
    work_ids_by_sample: list[set[str]] = [set() for _ in samples]
    raw_work_by_id: dict[str, tuple[float, Mapping[str, Any]]] = {}
    raw_work_order: list[tuple[str, float, Mapping[str, Any]]] = []
    dispatch_finished_by_cohort: dict[int, dict[int, float]] = {}
    if not samples or any(not sample.selected_cycles for sample in samples):
        raise DirectDisaggEvidenceError(
            "raw role validation requires non-empty selected timing samples"
        )
    target_batch = samples[0].selected_cycles[0].actual_shape.scheduled_sequences
    dispatch_pattern = re.compile(
        rf"^chatcmpl-pd-cal-d-b{target_batch}-c([0-9]+)-r([0-9]+)-[0-9a-f]{{32}}$"
    )
    previous_scheduler_timestamp: float | None = None
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise DirectDisaggEvidenceError(
                    f"telemetry JSONL has an incomplete line at {line_number}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DirectDisaggEvidenceError(
                    f"telemetry JSONL is invalid at line {line_number}"
                ) from exc
            if not isinstance(record, Mapping):
                raise DirectDisaggEvidenceError(
                    f"telemetry JSONL record {line_number} is not an object"
                )
            schema = record.get("schema")
            if schema == _GLOBAL_FRONTEND_TELEMETRY_SCHEMA:
                if (
                    role == "decode"
                    and record.get("run_id") == telemetry_run_id
                    and record.get("kind") == "request_dispatch_ipc"
                ):
                    request_id = record.get("request_id")
                    match = (
                        dispatch_pattern.fullmatch(request_id)
                        if isinstance(request_id, str)
                        else None
                    )
                    if match is not None:
                        cohort = int(match.group(1))
                        request_index = int(match.group(2))
                        finished = record.get("finished_monotonic_s")
                        if (
                            isinstance(finished, bool)
                            or not isinstance(finished, (int, float))
                            or not math.isfinite(float(finished))
                            or float(finished) <= 0.0
                        ):
                            raise DirectDisaggEvidenceError(
                                "decode dispatch fence timestamp is invalid"
                            )
                        by_request = dispatch_finished_by_cohort.setdefault(
                            cohort, {}
                        )
                        if request_index in by_request:
                            raise DirectDisaggEvidenceError(
                                "decode dispatch fence repeats a cohort request"
                            )
                        by_request[request_index] = float(finished)
                continue
            if schema != _GLOBAL_SCHEDULER_TELEMETRY_SCHEMA:
                continue
            if record.get("run_id") != telemetry_run_id:
                raise DirectDisaggEvidenceError(
                    "scheduler telemetry schema/run identity is invalid"
                )
            cycle_id = record.get("cycle_id")
            timestamp = record.get("monotonic_timestamp_s")
            if (
                isinstance(timestamp, bool)
                or not isinstance(timestamp, (int, float))
                or not math.isfinite(float(timestamp))
            ):
                raise DirectDisaggEvidenceError(
                    "scheduler telemetry monotonic timestamp is invalid"
                )
            timestamp = float(timestamp)
            if (
                previous_scheduler_timestamp is not None
                and timestamp <= previous_scheduler_timestamp
            ):
                raise DirectDisaggEvidenceError(
                    "scheduler telemetry CLOCK_MONOTONIC timestamps are not "
                    "strictly increasing in raw logger order"
                )
            previous_scheduler_timestamp = timestamp
            shape = record.get("shape")
            if not isinstance(shape, Mapping):
                raise DirectDisaggEvidenceError(
                    "scheduler telemetry shape is missing"
                )
            has_work = False
            for phase in ("prefill", "decode"):
                raw_phase = shape.get(phase)
                if not isinstance(raw_phase, Mapping):
                    raise DirectDisaggEvidenceError(
                        "scheduler telemetry phase shape is missing"
                    )
                sequences = raw_phase.get("scheduled_sequences")
                if (
                    isinstance(sequences, bool)
                    or not isinstance(sequences, int)
                    or sequences < 0
                ):
                    raise DirectDisaggEvidenceError(
                        "scheduler telemetry scheduled_sequences is invalid"
                )
                has_work = has_work or sequences > 0
            if not has_work:
                if any(
                    shape[phase].get(name) != 0
                    for phase in ("prefill", "decode")
                    for name in _SHAPE_FIELDS
                ):
                    raise DirectDisaggEvidenceError(
                        "idle scheduler telemetry must have both phase shapes "
                        "identically zero"
                    )
                continue
            if not isinstance(cycle_id, str) or not cycle_id:
                raise DirectDisaggEvidenceError(
                    "scheduler work telemetry cycle_id is invalid"
                )
            if cycle_id in raw_work_by_id:
                raise DirectDisaggEvidenceError(
                    "scheduler raw telemetry repeats a work cycle_id"
                )
            raw_work_by_id[cycle_id] = (timestamp, shape)
            raw_work_order.append((cycle_id, timestamp, record))
            containing = [
                index
                for index, sample in enumerate(samples)
                if sample.window_started_monotonic_s
                < timestamp
                <= sample.window_completed_monotonic_s
            ]
            if len(containing) > 1:
                raise DirectDisaggEvidenceError(
                    "measured role timing windows overlap in raw telemetry"
                )
            if containing:
                work_ids_by_sample[containing[0]].add(cycle_id)
            if cycle_id not in expected:
                continue
            if cycle_id in observed:
                raise DirectDisaggEvidenceError(
                    "selected cycle_id occurs more than once in raw telemetry"
                )
            if shape.get(
                "terminal_performance_stage"
            ) != role:
                raise DirectDisaggEvidenceError(
                    "selected cycle has the wrong terminal phase"
                )
            inactive_role = "decode" if role == "prefill" else "prefill"
            inactive = shape.get(inactive_role)
            if not isinstance(inactive, Mapping) or any(
                inactive.get(name) != 0 for name in _SHAPE_FIELDS
            ):
                raise DirectDisaggEvidenceError(
                    "selected role cycle is not phase-pure"
                )
            actual = ActualCycleShape.from_dict(shape.get(role))
            if actual != expected[cycle_id]:
                raise DirectDisaggEvidenceError(
                    "selected cycle actual_shape differs from raw telemetry"
                )
            observed[str(cycle_id)] = actual
    if set(observed) != set(expected):
        raise DirectDisaggEvidenceError(
            "raw telemetry does not contain every contract-selected cycle"
        )
    for index, (raw_ids, selected_ids) in enumerate(
        zip(work_ids_by_sample, expected_by_sample, strict=True)
    ):
        if raw_ids != selected_ids:
            raise DirectDisaggEvidenceError(
                f"sample {index} selected cycle IDs do not exactly equal all "
                "scheduler work rows in role window (start, end]"
            )
    if role == "decode":
        zero_shape = {name: 0 for name in _SHAPE_FIELDS}

        def saturated_exact_batch_shape(
            record: Mapping[str, Any],
        ) -> ActualCycleShape | None:
            shape = record.get("shape")
            if (
                not isinstance(shape, Mapping)
                or shape.get("terminal_performance_stage") != "decode"
                or shape.get("prefill") != zero_shape
            ):
                return None
            try:
                decode = ActualCycleShape.from_dict(shape.get("decode"))
            except DirectDisaggEvidenceError:
                return None
            if (
                decode.scheduled_sequences != target_batch
                or decode.scheduled_tokens != target_batch
                or decode.token_context_product
                != decode.scheduled_context_tokens
            ):
                return None
            for field in ("backlog_before_interval", "backlog_after_cycle"):
                backlog = record.get(field)
                if (
                    not isinstance(backlog, Mapping)
                    or set(backlog)
                    != {"running_sequences", "waiting_sequences"}
                ):
                    return None
                running = backlog.get("running_sequences")
                waiting = backlog.get("waiting_sequences")
                if (
                    isinstance(running, bool)
                    or not isinstance(running, int)
                    or isinstance(waiting, bool)
                    or not isinstance(waiting, int)
                    or running != target_batch
                    or waiting != 0
                ):
                    return None
            return decode

        for sample in samples:
            proof = sample.window_proof
            boundary_id = str(proof["boundary_start_cycle_id"])
            boundary = raw_work_by_id.get(boundary_id)
            if boundary is None:
                raise DirectDisaggEvidenceError(
                    "decode start boundary cycle is absent from raw telemetry"
                )
            boundary_timestamp, boundary_shape = boundary
            inactive = boundary_shape.get("prefill")
            if (
                boundary_shape.get("terminal_performance_stage") != "decode"
                or not isinstance(inactive, Mapping)
                or any(inactive.get(name) != 0 for name in _SHAPE_FIELDS)
                or ActualCycleShape.from_dict(boundary_shape.get("decode"))
                != ActualCycleShape.from_dict(proof["boundary_start_shape"])
                or not math.isclose(
                    boundary_timestamp,
                    sample.window_started_monotonic_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
            ):
                raise DirectDisaggEvidenceError(
                    "decode start boundary does not match its raw pure exact-B "
                    "scheduler cycle/timestamp"
                )
            end_id = sample.selected_cycles[-1].cycle_id
            end_timestamp, _end_shape = raw_work_by_id[end_id]
            if not math.isclose(
                end_timestamp,
                sample.window_completed_monotonic_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise DirectDisaggEvidenceError(
                    "decode saturated exact-B window end timestamp differs from raw"
                )

            # Recompute the acquisition policy from immutable raw evidence.
            # Cohort zero is warmup; timing sample i is decode cohort i+1.
            cohort = sample.sample_index + 1
            dispatches = dispatch_finished_by_cohort.get(cohort)
            if (
                dispatches is None
                or set(dispatches) != set(range(target_batch))
            ):
                raise DirectDisaggEvidenceError(
                    "decode measured cohort lacks its exact-B dispatch fence"
                )
            cohort_start = max(dispatches.values())
            next_dispatches = dispatch_finished_by_cohort.get(cohort + 1)
            cohort_end = (
                math.inf
                if next_dispatches is None
                else min(next_dispatches.values())
            )
            cohort_work = [
                item
                for item in raw_work_order
                if cohort_start < item[1] < cohort_end
            ]
            first_run: list[tuple[str, float, Mapping[str, Any]]] | None = None
            run_length = DECODE_CONSECUTIVE_EXACT_BATCH_CYCLES
            for start_index in range(0, len(cohort_work) - run_length + 1):
                candidate = cohort_work[start_index : start_index + run_length]
                candidate_shapes = [
                    saturated_exact_batch_shape(item[2]) for item in candidate
                ]
                if any(shape is None for shape in candidate_shapes):
                    continue
                contexts = [
                    shape.scheduled_context_tokens
                    for shape in candidate_shapes
                    if shape is not None
                ]
                if any(
                    right - left != target_batch
                    for left, right in zip(contexts, contexts[1:])
                ):
                    continue
                first_run = candidate
                break
            if first_run is None:
                raise DirectDisaggEvidenceError(
                    "decode raw cohort has no saturated 33-cycle exact-B run"
                )
            frozen_ids = [
                boundary_id,
                *(cycle.cycle_id for cycle in sample.selected_cycles),
            ]
            if [item[0] for item in first_run] != frozen_ids:
                raise DirectDisaggEvidenceError(
                    "decode timing sample is not the first raw saturated "
                    "33-cycle exact-B run"
                )


def _validate_component_join(
    value: Any,
    *,
    target: SelectedRoleCycle,
) -> dict[str, Any]:
    raw = _require_fields(value, _COMPONENT_JOIN_FIELDS, label="component_join")
    if raw.get("target_cycle_id") != target.cycle_id:
        raise DirectDisaggEvidenceError(
            "component join target_cycle_id differs from timing evidence"
        )
    target_shape = ActualCycleShape.from_dict(raw.get("target_shape"))
    if target_shape != target.actual_shape:
        raise DirectDisaggEvidenceError(
            "component join target_shape differs from timing evidence"
        )
    bracket = raw.get("bracket")
    if not isinstance(bracket, list) or len(bracket) not in {1, 2}:
        raise DirectDisaggEvidenceError(
            "component join bracket must contain one exact or two adjacent points"
        )
    normalized_bracket: list[dict[str, Any]] = []
    weighted_service = 0.0
    weight_sum = 0.0
    for point_index, point in enumerate(bracket):
        item = _require_fields(
            point,
            _COMPONENT_BRACKET_FIELDS,
            label=f"component_join.bracket[{point_index}]",
        )
        ids = item.get("component_cycle_ids")
        if (
            not isinstance(ids, list)
            or len(ids) != SAMPLES_PER_BATCH
            or any(not isinstance(value, str) or not value for value in ids)
            or len(set(ids)) != SAMPLES_PER_BATCH
        ):
            raise DirectDisaggEvidenceError(
                "component bracket must bind three measured cohort cycle IDs"
            )
        shape = ActualCycleShape.from_dict(item.get("shape"))
        weight = _positive_float(item.get("weight"), label="component weight")
        service = _positive_float(
            item.get("median_component_service_s"),
            label="component bracket service",
        )
        weight_sum += weight
        weighted_service += weight * service
        normalized_bracket.append({
            "component_cycle_ids": list(ids),
            "shape": shape.to_dict(),
            "weight": weight,
            "median_component_service_s": service,
        })
    if not math.isclose(weight_sum, 1.0, rel_tol=1e-12, abs_tol=1e-15):
        raise DirectDisaggEvidenceError("component bracket weights must sum to one")
    joined_service = _positive_float(
        raw.get("component_service_s"), label="joined component service"
    )
    if not math.isclose(
        joined_service, weighted_service, rel_tol=1e-12, abs_tol=1e-15
    ):
        raise DirectDisaggEvidenceError(
            "component_service_s differs from its frozen bracket/weights"
        )
    return {
        "target_cycle_id": target.cycle_id,
        "target_shape": target.actual_shape.to_dict(),
        "bracket": normalized_bracket,
        "component_service_s": joined_service,
    }


def validate_direct_role_timing_documents(
    timing_document: Any,
    contract_document: Any,
    *,
    source_path: str | Path,
) -> tuple[
    str,
    str,
    int,
    int,
    dict[str, Any],
    dict[int, DirectRoleBatchEvidence],
]:
    """Validate one role timing document and its immutable run contract."""

    timing = _require_fields(
        timing_document, _ROLE_TIMING_FIELDS, label="direct role timing"
    )
    contract = _require_fields(
        contract_document, _ROLE_CONTRACT_FIELDS, label="direct role contract"
    )
    if (
        timing.get("schema") != ROLE_TIMING_SCHEMA
        or contract.get("schema") != ROLE_RUN_CONTRACT_SCHEMA
    ):
        raise DirectDisaggEvidenceError("unexpected direct role contract kind")
    role = timing.get("role")
    if role not in {"prefill", "decode"} or contract.get("role") != role:
        raise DirectDisaggEvidenceError("direct timing/contract role mismatch")
    try:
        system_hardware = validate_system_hardware(timing.get("system_hardware"))
    except ValueError as exc:
        raise DirectDisaggEvidenceError(
            "direct timing system_hardware is invalid"
        ) from exc
    if contract.get("system_hardware") != system_hardware:
        raise DirectDisaggEvidenceError(
            "direct timing/contract system_hardware mismatch"
        )

    source = _canonical_existing_path(source_path, label="role timing source")
    if contract.get("source_path") != str(source):
        raise DirectDisaggEvidenceError(
            "role contract source_path does not bind the timing document"
        )
    source_sha = _sha256_file(source)
    if contract.get("source_sha256") != source_sha:
        raise DirectDisaggEvidenceError(
            "role contract source_sha256 does not bind the timing document"
        )

    workload = _require_fields(
        contract.get("workload"), _WORKLOAD_FIELDS, label="role workload"
    )
    if workload.get("target_batches") != list(REQUIRED_BATCHES):
        raise DirectDisaggEvidenceError(
            "role workload must cover the exact sparse batch ladder"
        )
    if workload.get("samples_per_batch") != SAMPLES_PER_BATCH:
        raise DirectDisaggEvidenceError(
            f"role workload requires exactly {SAMPLES_PER_BATCH} samples per batch"
        )
    prompt_tokens = _positive_int(
        workload.get("prompt_tokens"), label="workload.prompt_tokens"
    )
    output_tokens = _positive_int(
        workload.get("output_tokens"), label="workload.output_tokens"
    )
    if output_tokens != CALIBRATION_WORKLOAD_OUTPUT_TOKENS:
        raise DirectDisaggEvidenceError(
            "role calibration/component workload output_tokens must remain "
            f"{CALIBRATION_WORKLOAD_OUTPUT_TOKENS}; the 256-token decode "
            "acquisition guard is not a workload or fit dimension"
        )
    try:
        request_contract = validate_generator_request_contract(
            workload.get("request_contract"),
            expected_prompt_tokens=prompt_tokens,
        )
    except ValueError as exc:
        raise DirectDisaggEvidenceError(
            "role workload request_contract is invalid"
        ) from exc
    if contract.get("request_contract_sha256") != (
        generator_request_contract_sha256(request_contract)
    ):
        raise DirectDisaggEvidenceError(
            "role contract request_contract_sha256 is invalid"
        )

    for path_field, sha_field in (
        ("component_source_path", "component_source_sha256"),
        ("component_contract_path", "component_contract_sha256"),
    ):
        dependency = _canonical_existing_path(
            contract.get(path_field), label=path_field
        )
        if contract.get(path_field) != str(dependency) or contract.get(
            sha_field
        ) != _sha256_file(dependency):
            raise DirectDisaggEvidenceError(
                f"role contract {path_field}/{sha_field} is not immutable"
            )
    nixl_path = contract.get("nixl_physical_path")
    nixl_sha = contract.get("nixl_physical_sha256")
    nixl_source: Path | None = None
    if role == "prefill":
        if nixl_path is not None or nixl_sha is not None:
            raise DirectDisaggEvidenceError(
                "prefill role contract must not absorb NIXL evidence"
            )
    else:
        nixl_source = _canonical_existing_path(
            nixl_path, label="nixl_physical_path"
        )
        if nixl_path != str(nixl_source) or nixl_sha != _sha256_file(nixl_source):
            raise DirectDisaggEvidenceError(
                "decode role contract NIXL physical evidence is not immutable"
            )

    expected_selection = _selection_contract(str(role))
    selection_fields = (
        _PREFILL_SELECTION_FIELDS if role == "prefill" else _DECODE_SELECTION_FIELDS
    )
    selection = _require_fields(
        contract.get("cycle_selection"),
        selection_fields,
        label="cycle_selection",
    )
    if dict(selection) != expected_selection:
        raise DirectDisaggEvidenceError(
            "role cycle_selection differs from the fixed direct-P/D window"
        )
    aggregation = _require_fields(
        contract.get("aggregation"),
        set(ROLE_AGGREGATION_CONTRACT),
        label="aggregation",
    )
    if dict(aggregation) != ROLE_AGGREGATION_CONTRACT:
        raise DirectDisaggEvidenceError(
            "role aggregation differs from the fixed paired-residual policy"
        )

    batches_raw = timing.get("batches")
    runs = contract.get("runs")
    selected_ids = contract.get("selected_cycle_ids")
    component_joins = contract.get("component_joins")
    expected_keys = {str(batch) for batch in REQUIRED_BATCHES}
    for label, value in (
        ("timing batches", batches_raw),
        ("contract runs", runs),
        ("contract selected_cycle_ids", selected_ids),
        ("contract component_joins", component_joins),
    ):
        if not isinstance(value, Mapping) or set(value) != expected_keys:
            raise DirectDisaggEvidenceError(
                f"{label} must cover the exact sparse batch ladder"
            )

    seen_cycle_ids: set[str] = set()
    seen_run_ids: set[str] = set()
    batches: dict[int, DirectRoleBatchEvidence] = {}
    for batch in REQUIRED_BATCHES:
        raw = _require_fields(
            batches_raw[str(batch)], _ROLE_BATCH_FIELDS, label=f"B{batch} timing"
        )
        if raw.get("target_batch") != batch:
            raise DirectDisaggEvidenceError(f"B{batch} target identity mismatch")
        cap = _positive_int(
            raw.get("engine_max_num_seqs"), label=f"B{batch} engine cap"
        )
        if cap != batch:
            raise DirectDisaggEvidenceError(
                f"B{batch} direct role engine cap must equal its target batch"
            )
        telemetry_path = _canonical_existing_path(
            raw.get("telemetry_path"), label=f"B{batch} telemetry_path"
        )
        telemetry_sha = raw.get("telemetry_sha256")
        if raw.get("telemetry_path") != str(telemetry_path) or telemetry_sha != (
            _sha256_file(telemetry_path)
        ):
            raise DirectDisaggEvidenceError(
                f"B{batch} telemetry path/SHA does not bind its raw evidence"
            )
        telemetry_run = _identity(
            raw.get("telemetry_run_id"), label=f"B{batch} telemetry run_id"
        )
        timing_run = _identity(
            raw.get("timing_run_id"), label=f"B{batch} timing run_id"
        )
        if telemetry_run == timing_run or {telemetry_run, timing_run} & seen_run_ids:
            raise DirectDisaggEvidenceError(
                "direct role campaign must use unique engine/timing run IDs"
            )
        seen_run_ids.update((telemetry_run, timing_run))
        run_contract = _require_fields(
            runs[str(batch)], _RUN_FIELDS, label=f"B{batch} run contract"
        )
        if dict(run_contract) != {
            "telemetry_path": str(telemetry_path),
            "telemetry_sha256": telemetry_sha,
            "telemetry_run_id": telemetry_run,
            "timing_run_id": timing_run,
        }:
            raise DirectDisaggEvidenceError(
                f"B{batch} run contract does not bind timing evidence"
            )
        samples_raw = raw.get("samples")
        if not isinstance(samples_raw, list) or len(samples_raw) != SAMPLES_PER_BATCH:
            raise DirectDisaggEvidenceError(
                f"B{batch} must contain exactly {SAMPLES_PER_BATCH} samples"
            )
        samples = tuple(
            _parse_sample(
                sample,
                role=str(role),
                target_batch=batch,
                telemetry_run_id=telemetry_run,
                prompt_tokens=prompt_tokens,
                expected_sample_index=index,
                seen_cycle_ids=seen_cycle_ids,
            )
            for index, sample in enumerate(samples_raw)
        )
        expected_ids = [
            [cycle.cycle_id for cycle in sample.selected_cycles]
            for sample in samples
        ]
        if selected_ids[str(batch)] != expected_ids:
            raise DirectDisaggEvidenceError(
                f"B{batch} selected_cycle_ids do not bind the timing samples"
            )
        batch_joins = component_joins[str(batch)]
        if (
            not isinstance(batch_joins, list)
            or len(batch_joins) != SAMPLES_PER_BATCH
        ):
            raise DirectDisaggEvidenceError(
                f"B{batch} component_joins must cover three samples"
            )
        for sample, sample_joins in zip(samples, batch_joins, strict=True):
            if (
                not isinstance(sample_joins, list)
                or len(sample_joins) != len(sample.selected_cycles)
            ):
                raise DirectDisaggEvidenceError(
                    f"B{batch} component_joins do not cover selected cycles"
                )
            for cycle, join in zip(
                sample.selected_cycles, sample_joins, strict=True
            ):
                _validate_component_join(join, target=cycle)
        _validate_selected_cycles_against_telemetry(
            telemetry_path,
            telemetry_run_id=telemetry_run,
            role=str(role),
            samples=samples,
        )
        batches[batch] = DirectRoleBatchEvidence(
            target_batch=batch,
            engine_max_num_seqs=cap,
            telemetry_path=telemetry_path,
            telemetry_sha256=str(telemetry_sha),
            telemetry_run_id=telemetry_run,
            timing_run_id=timing_run,
            samples=samples,
        )
    if role == "decode":
        assert nixl_source is not None
        validate_nixl_physical_campaign(
            nixl_source,
            system_hardware=system_hardware,
            selected_samples_by_batch={
                batch: evidence.samples
                for batch, evidence in batches.items()
            },
        )
    return (
        system_hardware,
        str(role),
        prompt_tokens,
        output_tokens,
        request_contract,
        batches,
    )


def read_direct_role_evidence(
    timing_path: str | Path,
    contract_path: str | Path,
) -> DirectRoleEvidence:
    """Read and validate one immutable role-local timing campaign."""

    source = _canonical_existing_path(timing_path, label="role timing source")
    contract_source = _canonical_existing_path(
        contract_path, label="role timing contract"
    )
    timing = _read_json(source, label="role timing evidence")
    contract = _read_json(contract_source, label="role timing contract")
    system_hardware, role, prompt, output, request_contract, batches = (
        validate_direct_role_timing_documents(
            timing, contract, source_path=source
        )
    )
    return DirectRoleEvidence(
        system_hardware=system_hardware,
        role=role,
        prompt_tokens=prompt,
        output_tokens=output,
        request_contract=request_contract,
        batches=batches,
        source_path=source,
        source_sha256=_sha256_file(source),
        contract_path=contract_source,
        contract_sha256=_sha256_file(contract_source),
        component_source_path=Path(contract["component_source_path"]),
        component_source_sha256=str(contract["component_source_sha256"]),
        component_contract_path=Path(contract["component_contract_path"]),
        component_contract_sha256=str(contract["component_contract_sha256"]),
        nixl_physical_path=(
            None
            if contract["nixl_physical_path"] is None
            else Path(contract["nixl_physical_path"])
        ),
        nixl_physical_sha256=contract["nixl_physical_sha256"],
        component_joins=json.loads(json.dumps(contract["component_joins"])),
    )


@dataclass(frozen=True)
class ComponentCycleReference:
    cycle_id: str
    phase: str
    shape: ActualCycleShape
    service_s: float


@dataclass(frozen=True)
class ComponentReferenceGrid:
    by_phase_batch_tokens: Mapping[
        tuple[str, int, int],
        tuple[tuple[ActualCycleShape, float, tuple[str, ...]], ...],
    ]
    source_path: Path
    source_sha256: str
    contract_path: Path
    contract_sha256: str
    prompt_tokens: int
    output_tokens: int
    request_contract: Mapping[str, Any]

    def service_for(self, role: str, shape: ActualCycleShape) -> float:
        """Return exact prefill or strictly bracketed decode component service."""

        return float(self.join_for(role, shape, target_cycle_id="runtime")[
            "component_service_s"
        ])

    def join_for(
        self,
        role: str,
        shape: ActualCycleShape,
        *,
        target_cycle_id: str,
    ) -> dict[str, Any]:
        """Return the canonical exact/bracket proof for one actual role cycle."""

        key = (role, shape.scheduled_sequences, shape.scheduled_tokens)
        points = self.by_phase_batch_tokens.get(key)
        if not points:
            raise DirectDisaggEvidenceError(
                "component evidence has no same-phase/batch/scheduled-token bracket"
            )
        exact = [
            (service, cycle_ids)
            for candidate, service, cycle_ids in points
            if candidate == shape
        ]
        if exact:
            if len(exact) != 1:
                raise DirectDisaggEvidenceError(
                    "component reference grid contains a duplicate exact shape"
                )
            service, cycle_ids = exact[0]
            return {
                "target_cycle_id": target_cycle_id,
                "target_shape": shape.to_dict(),
                "bracket": [{
                    "component_cycle_ids": list(cycle_ids),
                    "shape": shape.to_dict(),
                    "weight": 1.0,
                    "median_component_service_s": service,
                }],
                "component_service_s": service,
            }
        if role == "prefill":
            raise DirectDisaggEvidenceError(
                "prefill component reference requires an exact shape"
            )
        ordered = sorted(
            points,
            key=lambda item: (
                item[0].scheduled_context_tokens,
                item[0].token_context_product,
            ),
        )
        for (
            left,
            left_service,
            left_cycle_ids,
        ), (
            right,
            right_service,
            right_cycle_ids,
        ) in zip(
            ordered, ordered[1:]
        ):
            context = shape.scheduled_context_tokens
            product = shape.token_context_product
            if not (
                left.scheduled_context_tokens < context < right.scheduled_context_tokens
                and left.token_context_product < product < right.token_context_product
            ):
                continue
            context_fraction = (
                (context - left.scheduled_context_tokens)
                / (right.scheduled_context_tokens - left.scheduled_context_tokens)
            )
            product_fraction = (
                (product - left.token_context_product)
                / (right.token_context_product - left.token_context_product)
            )
            if not math.isclose(
                context_fraction,
                product_fraction,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise DirectDisaggEvidenceError(
                    "decode target is not on the bracketed component shape segment"
                )
            service = left_service + context_fraction * (
                right_service - left_service
            )
            if not math.isfinite(service) or service <= 0.0:
                raise DirectDisaggEvidenceError(
                    "interpolated component service is invalid"
                )
            return {
                "target_cycle_id": target_cycle_id,
                "target_shape": shape.to_dict(),
                "bracket": [
                    {
                        "component_cycle_ids": list(left_cycle_ids),
                        "shape": left.to_dict(),
                        "weight": 1.0 - context_fraction,
                        "median_component_service_s": left_service,
                    },
                    {
                        "component_cycle_ids": list(right_cycle_ids),
                        "shape": right.to_dict(),
                        "weight": context_fraction,
                        "median_component_service_s": right_service,
                    },
                ],
                "component_service_s": float(service),
            }
        raise DirectDisaggEvidenceError(
            "decode component shape requires extrapolation or has no adjacent bracket"
        )


def _component_grid_from_selected_records(
    records: Iterable[ComponentCycleReference],
    *,
    selected_cycle_ids: set[str],
) -> dict[
    tuple[str, int, int],
    tuple[tuple[ActualCycleShape, float, tuple[str, ...]], ...],
]:
    selected: dict[str, ComponentCycleReference] = {}
    for record in records:
        if record.cycle_id not in selected_cycle_ids:
            continue
        if record.cycle_id in selected:
            raise DirectDisaggEvidenceError(
                "component raw repeats a contract-selected cycle_id"
            )
        selected[record.cycle_id] = record
    if set(selected) != selected_cycle_ids:
        raise DirectDisaggEvidenceError(
            "component raw does not cover every contract-selected cycle_id"
        )
    grouped: dict[tuple[str, int, int, ActualCycleShape], list[float]] = {}
    for record in selected.values():
        key = (
            record.phase,
            record.shape.scheduled_sequences,
            record.shape.scheduled_tokens,
            record.shape,
        )
        grouped.setdefault(key, []).append(
            _positive_float(record.service_s, label="component service")
        )
    by_key: dict[
        tuple[str, int, int],
        list[tuple[ActualCycleShape, float, tuple[str, ...]]],
    ] = {}
    for (phase, batch, tokens, shape), services in grouped.items():
        if len(services) != SAMPLES_PER_BATCH:
            raise DirectDisaggEvidenceError(
                "each component phase shape must have exactly three selected cohorts"
            )
        cycle_ids = tuple(sorted(
            record.cycle_id
            for record in selected.values()
            if (
                record.phase == phase
                and record.shape == shape
            )
        ))
        by_key.setdefault((phase, batch, tokens), []).append((
            shape,
            float(statistics.median(services)),
            cycle_ids,
        ))
    return {
        key: tuple(sorted(points, key=lambda item: item[0]))
        for key, points in by_key.items()
    }


def read_component_reference_grid(
    component_path: str | Path,
    component_contract_path: str | Path,
) -> ComponentReferenceGrid:
    """Rebuild the component grid from contract-selected measured cohorts only."""

    from .vllm_probe import (
        _read_component_cycle_records,
        _validate_run_contract_document,
    )

    source = _canonical_existing_path(component_path, label="component raw")
    contract_source = _canonical_existing_path(
        component_contract_path, label="component contract"
    )
    contract = _read_json(contract_source, label="component contract")
    try:
        validated = _validate_run_contract_document(
            dict(contract), expected_scope="component_only"
        )
    except ValueError as exc:
        raise DirectDisaggEvidenceError("component run contract is invalid") from exc
    if validated.get("raw_telemetry_path") != str(source):
        raise DirectDisaggEvidenceError(
            "component contract raw_telemetry_path does not bind the source"
        )
    source_sha = _sha256_file(source)
    if validated.get("raw_telemetry_sha256") != source_sha:
        raise DirectDisaggEvidenceError(
            "component contract raw_telemetry_sha256 does not bind the source"
        )
    workload = validated["workload"]
    if workload.get("target_batches") != list(REQUIRED_BATCHES):
        raise DirectDisaggEvidenceError(
            "component contract does not cover the exact sparse batch ladder"
        )
    contracts = workload.get("request_contracts")
    if not isinstance(contracts, Mapping) or set(contracts) != {
        str(workload["prompt_tokens"])
    }:
        raise DirectDisaggEvidenceError(
            "component contract must contain one canonical request contract"
        )
    request_contract = validate_generator_request_contract(
        contracts[str(workload["prompt_tokens"])],
        expected_prompt_tokens=int(workload["prompt_tokens"]),
    )
    target_ids_by_batch = validated.get("target_cycle_ids")
    selected_ids = {
        cycle_id
        for batch in REQUIRED_BATCHES
        for cycle_id in target_ids_by_batch[str(batch)]
    }
    try:
        raw_records = _read_component_cycle_records(
            source, allowed_cycle_ids=selected_ids
        )
    except ValueError as exc:
        raise DirectDisaggEvidenceError("component raw telemetry is invalid") from exc
    records = []
    for record in raw_records.values():
        phase = record.terminal_performance_stage
        shape_tuple = record.prefill if phase == "prefill" else record.decode
        records.append(ComponentCycleReference(
            cycle_id=record.cycle_id,
            phase=phase,
            shape=ActualCycleShape(*shape_tuple),
            service_s=record.component_service_s,
        ))
    return ComponentReferenceGrid(
        by_phase_batch_tokens=_component_grid_from_selected_records(
            records, selected_cycle_ids=selected_ids
        ),
        source_path=source,
        source_sha256=source_sha,
        contract_path=contract_source,
        contract_sha256=_sha256_file(contract_source),
        prompt_tokens=int(workload["prompt_tokens"]),
        output_tokens=int(workload["output_tokens"]),
        request_contract=request_contract,
    )


def build_component_joins(
    timing_document: Mapping[str, Any],
    component_path: str | Path,
    component_contract_path: str | Path,
) -> dict[str, list[list[dict[str, Any]]]]:
    """Build the contract payload that freezes every actual-shape join.

    Acquisition calls this after writing all role windows but before writing
    the role contract.  Publication independently rebuilds the same payload
    from component raw and requires exact equality.
    """

    timing = _require_fields(
        timing_document, _ROLE_TIMING_FIELDS, label="direct role timing"
    )
    if (
        timing.get("schema") != ROLE_TIMING_SCHEMA
        or timing.get("role") not in {"prefill", "decode"}
    ):
        raise DirectDisaggEvidenceError(
            "cannot build joins for an invalid role timing document"
        )
    validate_system_hardware(timing.get("system_hardware"))
    role = str(timing["role"])
    batches_raw = timing.get("batches")
    if not isinstance(batches_raw, Mapping) or set(batches_raw) != {
        str(batch) for batch in REQUIRED_BATCHES
    }:
        raise DirectDisaggEvidenceError(
            "role timing does not cover the exact sparse batch ladder"
        )
    component = read_component_reference_grid(
        component_path, component_contract_path
    )
    seen_cycle_ids: set[str] = set()
    output: dict[str, list[list[dict[str, Any]]]] = {}
    for batch in REQUIRED_BATCHES:
        raw = _require_fields(
            batches_raw[str(batch)], _ROLE_BATCH_FIELDS, label=f"B{batch} timing"
        )
        if raw.get("target_batch") != batch or raw.get(
            "engine_max_num_seqs"
        ) != batch:
            raise DirectDisaggEvidenceError(
                f"B{batch} timing does not prove an exact-cap engine"
            )
        telemetry_run = _identity(
            raw.get("telemetry_run_id"), label=f"B{batch} telemetry run_id"
        )
        samples_raw = raw.get("samples")
        if not isinstance(samples_raw, list) or len(samples_raw) != SAMPLES_PER_BATCH:
            raise DirectDisaggEvidenceError(
                f"B{batch} timing must contain three measured windows"
            )
        samples = tuple(
            _parse_sample(
                sample,
                role=role,
                target_batch=batch,
                telemetry_run_id=telemetry_run,
                prompt_tokens=component.prompt_tokens,
                expected_sample_index=index,
                seen_cycle_ids=seen_cycle_ids,
            )
            for index, sample in enumerate(samples_raw)
        )
        output[str(batch)] = [
            [
                component.join_for(
                    role,
                    cycle.actual_shape,
                    target_cycle_id=cycle.cycle_id,
                )
                for cycle in sample.selected_cycles
            ]
            for sample in samples
        ]
    return output


def measurements_for_direct_role(
    evidence: DirectRoleEvidence,
    component: ComponentReferenceGrid,
    *,
    system_hardware: str,
    stage: str,
) -> tuple[StageBatchMeasurement, ...]:
    """Pair role windows with component shapes and emit six batch points."""

    if evidence.role not in {"prefill", "decode"}:
        raise DirectDisaggEvidenceError("direct role evidence has an invalid role")
    if evidence.system_hardware != validate_system_hardware(system_hardware):
        raise DirectDisaggEvidenceError(
            "direct role evidence system_hardware differs from publication"
        )
    expected_stage = f"generator_{evidence.role}"
    if stage not in {expected_stage, f"query_expansion_{evidence.role}"}:
        raise DirectDisaggEvidenceError(
            f"stage {stage!r} does not match direct {evidence.role!r} evidence"
        )
    if (
        evidence.prompt_tokens != component.prompt_tokens
        or evidence.output_tokens != component.output_tokens
        or dict(evidence.request_contract) != dict(component.request_contract)
        or evidence.component_source_path != component.source_path
        or evidence.component_source_sha256 != component.source_sha256
        or evidence.component_contract_path != component.contract_path
        or evidence.component_contract_sha256 != component.contract_sha256
    ):
        raise DirectDisaggEvidenceError(
            "direct role and component workload/request contracts differ"
        )
    key = StageBatchCurveKey(validate_system_hardware(system_hardware), stage)
    output: list[StageBatchMeasurement] = []
    for batch in REQUIRED_BATCHES:
        batch_evidence = evidence.batches[batch]
        paired: list[tuple[float, int, float, float]] = []
        for sample in batch_evidence.samples:
            component_services: list[float] = []
            frozen_sample_joins = evidence.component_joins[str(batch)][
                sample.sample_index
            ]
            for cycle_index, cycle in enumerate(sample.selected_cycles):
                recomputed = component.join_for(
                    evidence.role,
                    cycle.actual_shape,
                    target_cycle_id=cycle.cycle_id,
                )
                frozen = _validate_component_join(
                    frozen_sample_joins[cycle_index], target=cycle
                )
                if frozen != recomputed:
                    raise DirectDisaggEvidenceError(
                        "frozen component join differs from strict recomputation "
                        "from contract-selected component raw cycles"
                    )
                component_services.append(float(recomputed["component_service_s"]))
            component_mean = float(statistics.fmean(component_services))
            observed = sample.observed_service_per_cycle_s
            residual = observed - component_mean
            if not math.isfinite(residual) or residual < 0.0:
                raise DirectDisaggEvidenceError(
                    f"B{batch} {evidence.role} sample {sample.sample_index} has "
                    "negative direct engine overhead; fail closed without clamp"
                )
            paired.append((residual, sample.sample_index, observed, component_mean))
        residual, _index, observed, component_mean = sorted(paired)[1]
        output.append(StageBatchMeasurement(
            key=key,
            target_batch=batch,
            engine_max_num_seqs=batch_evidence.engine_max_num_seqs,
            observed_resident_batch=batch,
            observed_total_service_s=observed,
            reference_component_service_s=component_mean,
            overhead_s=residual,
            sample_count=SAMPLES_PER_BATCH,
        ))
    return tuple(output)


def build_direct_role_curves(
    *,
    system_hardware: str,
    prefill_evidence: DirectRoleEvidence,
    decode_evidence: DirectRoleEvidence,
    component: ComponentReferenceGrid,
) -> tuple[StageBatchCurveProfile, ...]:
    """Build direct generator P/D curves plus explicit QE boundary aliases."""

    if prefill_evidence.role != "prefill" or decode_evidence.role != "decode":
        raise DirectDisaggEvidenceError(
            "build_direct_role_curves requires prefill then decode evidence"
        )
    curves: list[StageBatchCurveProfile] = []
    for evidence in (prefill_evidence, decode_evidence):
        role = evidence.role
        generator_stage = f"generator_{role}"
        generator_measurements = measurements_for_direct_role(
            evidence,
            component,
            system_hardware=system_hardware,
            stage=generator_stage,
        )
        curves.append(build_stage_batch_curve(
            key=StageBatchCurveKey(system_hardware, generator_stage),
            measurements=generator_measurements,
        ))
        alias_stage = f"query_expansion_{role}"
        alias_measurements = tuple(
            StageBatchMeasurement(
                key=StageBatchCurveKey(system_hardware, alias_stage),
                target_batch=item.target_batch,
                engine_max_num_seqs=item.engine_max_num_seqs,
                observed_resident_batch=item.observed_resident_batch,
                observed_total_service_s=item.observed_total_service_s,
                reference_component_service_s=item.reference_component_service_s,
                overhead_s=item.overhead_s,
                sample_count=item.sample_count,
            )
            for item in generator_measurements
        )
        curves.append(build_stage_batch_curve(
            key=StageBatchCurveKey(system_hardware, alias_stage),
            measurements=alias_measurements,
        ))
    return tuple(curves)


def merge_and_publish_complete_generation(
    profile_path: str | Path,
    role_curves: Sequence[StageBatchCurveProfile],
) -> tuple[StageBatchCurveProfile, ...]:
    """Preserve the existing collocated pair and atomically publish all six."""

    from rag_stack.cost_model.stage_batch_curve import (
        PROFILE_BLOCK_KEY,
        StageBatchCurveProfile,
    )

    path = _canonical_existing_path(profile_path, label="stage-batch profile")
    document = _read_json(path, label="stage-batch profile")
    block = document.get(PROFILE_BLOCK_KEY)
    if (
        set(document) != {PROFILE_BLOCK_KEY}
        or not isinstance(block, Mapping)
        or set(block) != {"system_hardware", "profiles"}
    ):
        raise DirectDisaggEvidenceError(
            "existing stage-batch profile does not have the current exact schema"
        )
    profiles = block.get("profiles")
    if not isinstance(profiles, Mapping) or set(profiles) not in (
        {"generator", "query_expansion"},
        {
            "generator",
            "query_expansion",
            "generator_prefill",
            "generator_decode",
            "query_expansion_prefill",
            "query_expansion_decode",
        },
    ):
        raise DirectDisaggEvidenceError(
            "existing profile must contain a valid collocated pair/generation"
        )
    collocated = tuple(
        StageBatchCurveProfile.from_dict(profiles[stage])
        for stage in ("generator", "query_expansion")
    )
    validate_stage_batch_curve_generation(collocated)
    generation = (*collocated, *tuple(role_curves))
    validate_stage_batch_curve_generation(generation)
    publish_stage_batch_curves(path, generation)
    return generation


def fit_and_publish_direct_disagg(
    config: str,
) -> tuple[StageBatchCurveProfile, ...]:
    """CPU-only fit entrypoint used by :mod:`generator_disagg`.

    Paths are derived solely from ``config.system.system_hardware`` and the
    canonical calibration roots.  The decode contract independently binds
    NIXL physical evidence, but NIXL is never read as role service or added to
    a fitted residual here.
    """

    from rag_stack.calibration_paths import (
        calibration_data_root,
        calibration_profile_root,
    )
    from rag_stack.cost_model.llm_sim.calibration import hardware_from_config

    hardware, _system = hardware_from_config(config)
    system_hardware = validate_system_hardware(
        getattr(hardware, "system_hardware", None)
    )
    data_root = calibration_data_root() / "generator_stage_batch"
    prefill_path = data_root / f"{system_hardware}.disagg_prefill.json"
    decode_path = data_root / f"{system_hardware}.disagg_decode.json"
    component_path = data_root / f"{system_hardware}.component.jsonl"
    prefill = read_direct_role_evidence(
        prefill_path, Path(f"{prefill_path}.contract.json")
    )
    decode = read_direct_role_evidence(
        decode_path, Path(f"{decode_path}.contract.json")
    )
    component = read_component_reference_grid(
        component_path, Path(f"{component_path}.contract.json")
    )
    role_curves = build_direct_role_curves(
        system_hardware=system_hardware,
        prefill_evidence=prefill,
        decode_evidence=decode,
        component=component,
    )
    profile_path = (
        calibration_profile_root()
        / "stage_batch"
        / f"{system_hardware}.json"
    )
    generation = merge_and_publish_complete_generation(
        profile_path, role_curves
    )
    return generation


__all__ = [
    "ActualCycleShape",
    "CALIBRATION_WORKLOAD_OUTPUT_TOKENS",
    "ComponentCycleReference",
    "ComponentReferenceGrid",
    "DECODE_ACQUISITION_GUARD_DISPOSITION",
    "DECODE_ACQUISITION_GUARD_OUTPUT_TOKENS",
    "DECODE_CONSECUTIVE_EXACT_BATCH_CYCLES",
    "DECODE_INTERDEPARTURE_INTERVALS_PER_SAMPLE",
    "DECODE_WINDOW_CLOCK",
    "DECODE_WINDOW_POLICY",
    "DECODE_WINDOW_SOURCE",
    "DirectDisaggEvidenceError",
    "DirectRoleBatchEvidence",
    "DirectRoleEvidence",
    "DirectRoleWindowSample",
    "MIN_DECODE_CYCLES_PER_SAMPLE",
    "PREFILL_CYCLES_PER_SAMPLE",
    "REQUIRED_BATCHES",
    "ROLE_AGGREGATION_CONTRACT",
    "ROLE_RUN_CONTRACT_SCHEMA",
    "ROLE_TIMING_SCHEMA",
    "SAMPLES_PER_BATCH",
    "SelectedRoleCycle",
    "build_direct_role_curves",
    "build_component_joins",
    "fit_and_publish_direct_disagg",
    "measurements_for_direct_role",
    "merge_and_publish_complete_generation",
    "read_component_reference_grid",
    "read_direct_role_evidence",
    "validate_direct_role_timing_documents",
    "validate_nixl_physical_campaign",
]
