"""vLLM probes for the two deliberately separate calibration boundaries.

Component ceff/meff consumes only CUDA-event service from an isolated offline
TP1/PP1 worker.  Its EngineCore is forced in-process so a complete cohort is
enqueued before the first scheduler step; OpenAI arrival/queue jitter can
therefore never alter the component batch or enter the physical coefficients.

Serving-stage overhead is measured independently through the production
OpenAI/continuous-batching deployment and its scheduler-cycle stat logger.
Dynamic batching wait is modeled by the closed-loop queue and is not relabelled
as active component service.  Both paths fail closed on an unproven batch or
token shape.
"""
from __future__ import annotations

import asyncio
from contextlib import contextmanager
import dataclasses
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import statistics
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import uuid

from .generator_request_contract import (
    COMPONENT_SERVICE_BOUNDARY,
    build_generator_request_contracts,
    generator_chat_payload,
    validate_generator_request_contract,
    validate_generator_request_contracts,
)
from .stage_timing_contract import (
    CollocatedEngineTimingEvidence,
    PRODUCTION_COHORT_COUNT,
    build_collocated_engine_timing_evidence,
    build_production_cohort_service_evidence,
    capture_clock_domain_proof,
)


_STAGE_TELEMETRY_ENV = "RAG_STACK_STAGE_TELEMETRY_PATH"
_COMPONENT_TELEMETRY_ENV = "RAG_STACK_LLM_COMPONENT_TELEMETRY_PATH"
_CALIBRATION_RUN_ID_ENV = "RAG_STACK_VLLM_CALIBRATION_RUN_ID"
_STAGE_TELEMETRY_SCHEMA = "rag_stack.vllm.scheduler_cycle_stage_service"
_COMPONENT_TELEMETRY_SCHEMA = "rag_stack.vllm.genz_component_gpu_execution"
_FRONTEND_TELEMETRY_SCHEMA = "rag_stack.vllm.frontend_stage_boundary_span"
_TIMING_DISPATCH_ENV = "RAG_STACK_V1_TIMING_DISPATCH_COUNTER"
_TIMING_DISPATCH_METRIC = "rag_stack_v1_timing_dispatch_count"
_STAGE_TELEMETRY_LOCK = threading.Lock()
_COMPONENT_WARMUP_COHORT_COUNT = 2
_COMPONENT_MEASUREMENT_COHORT_COUNT = 3
_COMPONENT_TOTAL_COHORT_COUNT = (
    _COMPONENT_WARMUP_COHORT_COUNT + _COMPONENT_MEASUREMENT_COHORT_COUNT
)
_COMPONENT_MAX_RELATIVE_SPREAD = 0.10
_COMPONENT_PREFILL_MAX_RELATIVE_SPREAD = 0.35


@contextmanager
def _scheduler_control_environment():
    """Enable calibration-owned pause/resume without enabling telemetry."""

    with _STAGE_TELEMETRY_LOCK:
        if os.environ.get(_STAGE_TELEMETRY_ENV) or os.environ.get(
            _COMPONENT_TELEMETRY_ENV
        ):
            raise RuntimeError(
                "production timing must run without stage/component telemetry"
            )
        previous_dev_mode = os.environ.get("VLLM_SERVER_DEV_MODE")
        previous_timing_dispatch = os.environ.get(_TIMING_DISPATCH_ENV)
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"
        os.environ[_TIMING_DISPATCH_ENV] = "1"
        try:
            yield
        finally:
            if previous_dev_mode is None:
                os.environ.pop("VLLM_SERVER_DEV_MODE", None)
            else:
                os.environ["VLLM_SERVER_DEV_MODE"] = previous_dev_mode
            if previous_timing_dispatch is None:
                os.environ.pop(_TIMING_DISPATCH_ENV, None)
            else:
                os.environ[_TIMING_DISPATCH_ENV] = previous_timing_dispatch


@dataclass(frozen=True)
class StageCycleObservation:
    """One publishable active vLLM cycle, keyed by its observed batch.

    Client concurrency is intentionally absent.  ``batch_size`` comes from
    scheduler telemetry, and decode rows additionally prove that every
    scheduled sequence was resident before the cycle. Model and TP/PP are
    acquisition diagnostics only; callers must never use them as curve selectors.
    """

    performance_stage: str
    batch_size: int
    # Diagnostic only.  Under vLLM async scheduling the stat callback interval
    # is not a GPU-complete cycle wall; Publication uses the frozen
    # short/long active-window contract instead.
    active_service_s: float | None
    scheduled_tokens: int
    scheduled_context_tokens: int
    token_context_product: int
    model: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    run_id: str
    cycle_id: str


@dataclass(frozen=True)
class _ComponentCycleService:
    """Strict TP1/PP1 component record retained for stage-cycle joining."""

    run_id: str
    cycle_id: str
    component_service_s: float
    model: str
    terminal_performance_stage: str
    prefill: tuple[int, int, int, int]
    decode: tuple[int, int, int, int]


@dataclass(frozen=True)
class _OfflineComponentRawPoint:
    """Parent-recomputed fixed-cohort evidence for one requested shape."""

    run_id: str
    warmup_component_services_s: tuple[float, ...]
    measurement_component_services_s: tuple[float, ...]
    component_service_s: float
    measurement_relative_spread: float
    cohort_cycle_ids: tuple[tuple[str, ...], ...]


@contextmanager
def _stage_telemetry_environment(path: Path):
    """Enable append-only stage telemetry for exactly one calibration server.

    Environment mutation is process-global, so the context is serialized.  A
    stage calibration deliberately uses one cheap reference deployment and
    therefore does not need the old model×TP×PP parallel launcher.
    """

    with _STAGE_TELEMETRY_LOCK:
        if os.environ.get(_COMPONENT_TELEMETRY_ENV):
            raise RuntimeError(
                "stage-only telemetry cannot run while component telemetry "
                "is enabled"
            )
        if os.environ.get(_TIMING_DISPATCH_ENV):
            raise RuntimeError(
                "stage proof telemetry cannot share a production timing server"
            )
        previous_path = os.environ.get(_STAGE_TELEMETRY_ENV)
        previous_run_id = os.environ.get(_CALIBRATION_RUN_ID_ENV)
        previous_dev_mode = os.environ.get("VLLM_SERVER_DEV_MODE")
        run_id = uuid.uuid4().hex
        os.environ[_STAGE_TELEMETRY_ENV] = str(path)
        os.environ[_CALIBRATION_RUN_ID_ENV] = run_id
        # vLLM 0.18.1 gates its audited scheduler pause/resume endpoints on
        # dev mode.  Scope the flag to this calibration-owned server only.
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"
        try:
            yield run_id
        finally:
            if previous_path is None:
                os.environ.pop(_STAGE_TELEMETRY_ENV, None)
            else:
                os.environ[_STAGE_TELEMETRY_ENV] = previous_path
            if previous_run_id is None:
                os.environ.pop(_CALIBRATION_RUN_ID_ENV, None)
            else:
                os.environ[_CALIBRATION_RUN_ID_ENV] = previous_run_id
            if previous_dev_mode is None:
                os.environ.pop("VLLM_SERVER_DEV_MODE", None)
            else:
                os.environ["VLLM_SERVER_DEV_MODE"] = previous_dev_mode


def _exact_component_cohort_service_from_segment(
    payload: bytes,
    *,
    target_batch: int,
    input_tokens: int,
    output_tokens: int,
    expected_run_id: str | None = None,
    first_work_sequence: int | None = None,
) -> float:
    """Sum one complete phase-pure cohort from an offline ``LLM.generate``.

    The offline in-process EngineCore enqueues the whole cohort before its first
    scheduler step, so the segment must contain exactly one full-batch prefill
    followed by ``output_tokens - 1`` full-batch decode cycles.  There is no
    blocker prefix to search through: any extra, missing, mixed or unrelated
    cycle invalidates the component observation.
    """

    if not payload or not payload.endswith(b"\n"):
        raise ValueError("exact component cohort telemetry is empty or incomplete")
    target_batch = int(target_batch)
    input_tokens = int(input_tokens)
    output_tokens = int(output_tokens)
    if target_batch <= 0 or input_tokens <= 0 or output_tokens <= 0:
        raise ValueError("exact component cohort shape must be positive")
    expected_prefill_tokens = target_batch * input_tokens
    expected_prefill_product = target_batch * input_tokens * input_tokens
    expected_cycles = output_tokens
    services: list[float] = []
    cycle_ids: set[str] = set()
    run_id: str | None = None
    lines = payload.splitlines()
    if len(lines) != expected_cycles:
        raise ValueError(
            "exact component cohort cycle count differs from its token shape: "
            f"observed={len(lines)}, expected={expected_cycles}"
        )

    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "invalid exact component cohort JSON at segment line "
                f"{line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError("component telemetry records must be objects")
        if (
            record.get("schema") != _COMPONENT_TELEMETRY_SCHEMA
        ):
            raise ValueError("unexpected component telemetry kind")
        if (
            record.get("world_size") != 1
            or record.get("tensor_parallel_size") != 1
            or record.get("pipeline_parallel_size") != 1
            or record.get("rank") != 0
        ):
            raise ValueError(
                "component fit requires one TP1/PP1 telemetry reference"
            )
        record_run_id = record.get("run_id")
        cycle_id = record.get("cycle_id")
        if not isinstance(record_run_id, str) or not record_run_id:
            raise ValueError("exact component cohort run_id is missing")
        if run_id is None:
            run_id = record_run_id
        elif record_run_id != run_id:
            raise ValueError("exact component cohort spans multiple runs")
        if expected_run_id is not None and record_run_id != expected_run_id:
            raise ValueError("exact component cohort run_id differs from its parent")
        if not isinstance(cycle_id, str) or not cycle_id:
            raise ValueError("exact component cohort cycle_id is missing")
        if cycle_id in cycle_ids:
            raise ValueError("exact component cohort has a duplicate cycle_id")
        cycle_ids.add(cycle_id)
        shape = record.get("shape")
        if not isinstance(shape, dict):
            raise ValueError("component telemetry shape is missing")
        prefill = _strict_phase_shape(shape.get("prefill"), field="prefill")
        decode = _strict_phase_shape(shape.get("decode"), field="decode")
        service = record.get("gpu_execution_s")
        if (
            isinstance(service, bool)
            or not isinstance(service, (int, float))
            or not math.isfinite(float(service))
            or float(service) <= 0.0
        ):
            raise ValueError("component telemetry GPU service is invalid")
        if not record.get("publishable"):
            raise ValueError(
                "exact component cohort contains a non-publishable cycle: "
                f"line={line_number}, reason={record.get('reason')!r}"
            )

        if line_number == 1:
            expected_prefill = (
                target_batch,
                expected_prefill_tokens,
                expected_prefill_tokens,
                expected_prefill_product,
            )
            if (
                prefill != expected_prefill
                or decode != (0, 0, 0, 0)
                or shape.get("terminal_performance_stage") != "prefill"
            ):
                raise ValueError(
                    "exact component cohort does not begin with its full-batch "
                    f"pure prefill: observed={prefill!r}"
                )
        else:
            decode_index = line_number - 1
            expected_context = target_batch * (input_tokens + decode_index)
            expected_decode = (
                target_batch,
                target_batch,
                expected_context,
                expected_context,
            )
            if (
                prefill != (0, 0, 0, 0)
                or decode != expected_decode
                or shape.get("terminal_performance_stage") != "decode"
            ):
                raise ValueError(
                    "exact component cohort contains a non-full-batch pure "
                    f"decode at line {line_number}: prefill={prefill!r}, "
                    f"decode={decode!r}, expected={expected_decode!r}"
                )
        if first_work_sequence is not None:
            expected_sequence = int(first_work_sequence) + line_number - 1
            if record.get("work_sequence") != expected_sequence:
                raise ValueError(
                    "exact component cohort work_sequence is not consecutive"
                )
            if record.get("cycle_sequence") != expected_sequence:
                raise ValueError(
                    "exact component cohort cycle_sequence is not consecutive"
                )
            if not cycle_id.startswith(
                f"{record_run_id}:{expected_sequence}:"
            ):
                raise ValueError(
                    "exact component cohort cycle_id is not sequence-bound"
                )
        services.append(float(service))
    return float(sum(services))


def _audit_offline_component_raw(
    path: str | os.PathLike[str],
    *,
    requested_shapes: Sequence[Tuple[int, int, int]],
    expected_run_id: str | None = None,
    warmup_cohorts: int = _COMPONENT_WARMUP_COHORT_COUNT,
    measurement_cohorts: int = _COMPONENT_MEASUREMENT_COHORT_COUNT,
) -> dict[tuple[int, int, int], _OfflineComponentRawPoint]:
    """Rebuild a complete independent component campaign from raw JSONL.

    Requested shape order is part of the evidence contract.  Every shape must
    contribute exactly two warmup and three selected cohorts, every cohort is
    an exact prefill/decode sequence, and the whole child run must expose one
    uninterrupted ``work_sequence``/``cycle_sequence`` namespace.
    """

    normalized_shapes = [
        tuple(int(value) for value in shape) for shape in requested_shapes
    ]
    if not normalized_shapes or len(set(normalized_shapes)) != len(
        normalized_shapes
    ):
        raise ValueError(
            "offline component raw audit requires unique requested shapes"
        )
    if (
        int(warmup_cohorts) != _COMPONENT_WARMUP_COHORT_COUNT
        or int(measurement_cohorts) != _COMPONENT_MEASUREMENT_COHORT_COUNT
    ):
        raise ValueError(
            "offline component raw audit requires exactly two warmup and "
            "three measurement cohorts"
        )
    source = Path(path)
    try:
        payload = source.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError("offline component raw telemetry is missing") from exc
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("offline component raw telemetry is missing/incomplete")
    lines = payload.splitlines()
    total_cohorts = int(warmup_cohorts) + int(measurement_cohorts)
    expected_line_count = sum(
        int(output_tokens) * total_cohorts
        for _batch, _input_tokens, output_tokens in normalized_shapes
    )
    if len(lines) != expected_line_count:
        raise ValueError(
            "offline component raw cohort cycle count differs from the "
            f"requested grid: observed={len(lines)}, expected={expected_line_count}"
        )

    # Establish the one child run before validating individual cohorts.  This
    # also prevents a well-formed foreign run from being spliced into the raw.
    try:
        first_record = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("invalid offline component raw JSON at line 1") from exc
    if not isinstance(first_record, dict):
        raise ValueError("offline component raw records must be objects")
    raw_run_id = first_record.get("run_id")
    if not isinstance(raw_run_id, str) or not raw_run_id:
        raise ValueError("offline component raw run_id is missing")
    if expected_run_id is not None and raw_run_id != expected_run_id:
        raise ValueError("offline component raw run_id differs from child result")

    output: dict[tuple[int, int, int], _OfflineComponentRawPoint] = {}
    line_offset = 0
    seen_cycle_ids: set[str] = set()
    for shape in normalized_shapes:
        batch, input_tokens, output_tokens = shape
        cohort_services: list[float] = []
        cohort_cycle_ids: list[tuple[str, ...]] = []
        for _cohort_index in range(total_cohorts):
            cohort_lines = lines[line_offset : line_offset + output_tokens]
            if len(cohort_lines) != output_tokens:
                raise ValueError("offline component raw cohort is incomplete")
            service = _exact_component_cohort_service_from_segment(
                b"\n".join(cohort_lines) + b"\n",
                target_batch=batch,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                expected_run_id=raw_run_id,
                first_work_sequence=line_offset,
            )
            ids: list[str] = []
            for raw_line in cohort_lines:
                record = json.loads(raw_line)
                cycle_id = str(record["cycle_id"])
                if cycle_id in seen_cycle_ids:
                    raise ValueError(
                        "offline component raw repeats a cycle_id"
                    )
                seen_cycle_ids.add(cycle_id)
                ids.append(cycle_id)
            cohort_services.append(float(service))
            cohort_cycle_ids.append(tuple(ids))
            line_offset += output_tokens
        warmup_values = tuple(
            cohort_services[:_COMPONENT_WARMUP_COHORT_COUNT]
        )
        measurement_values = tuple(
            cohort_services[_COMPONENT_WARMUP_COHORT_COUNT:]
        )
        selected_median = float(statistics.median(measurement_values))
        selected_spread = (
            max(measurement_values) - min(measurement_values)
        ) / selected_median
        if selected_spread > _COMPONENT_MAX_RELATIVE_SPREAD:
            raise ValueError(
                "offline component selected cohort totals are unstable for "
                f"{shape!r}: spread={selected_spread:.3%} > "
                f"{_COMPONENT_MAX_RELATIVE_SPREAD:.1%}"
            )
        output[shape] = _OfflineComponentRawPoint(
            run_id=raw_run_id,
            warmup_component_services_s=warmup_values,
            measurement_component_services_s=measurement_values,
            component_service_s=selected_median,
            measurement_relative_spread=float(selected_spread),
            cohort_cycle_ids=tuple(cohort_cycle_ids),
        )
    if line_offset != len(lines):
        raise ValueError("offline component raw contains unassigned cycles")
    return output


_PHASE_SHAPE_FIELDS = (
    "scheduled_sequences",
    "scheduled_tokens",
    "scheduled_context_tokens",
    "token_context_product",
)


def _strict_phase_shape(value: object, *, field: str) -> tuple[int, int, int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"component telemetry {field} shape must be an object")
    values = tuple(value.get(name) for name in _PHASE_SHAPE_FIELDS)
    if any(
        isinstance(item, bool)
        or not isinstance(item, int)
        or item < 0
        for item in values
    ):
        raise ValueError(f"component telemetry {field} shape is invalid")
    return tuple(int(item) for item in values)


def _validate_frontend_coverage_record(
    record: object,
    *,
    expected_run_id: str | None = None,
) -> dict:
    """Validate one coverage-only frontend span and return it unchanged."""

    if not isinstance(record, dict):
        raise ValueError("frontend telemetry record must be an object")
    if (
        record.get("schema") != _FRONTEND_TELEMETRY_SCHEMA
        or record.get("evidence_scope")
        != "active_stage_boundary_coverage_only"
        or record.get("fit_value") is not False
    ):
        raise ValueError("unexpected frontend coverage telemetry schema")
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("frontend coverage telemetry run_id is missing")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError("frontend coverage telemetry run_id mismatch")
    if record.get("kind") not in {
        "preprocess_tokenize",
        "request_dispatch_ipc",
        "output_materialization",
        "response_build_postprocess",
        "response_json_serialization",
    }:
        raise ValueError("frontend coverage telemetry kind is invalid")
    request_id = record.get("request_id")
    if record.get("kind") == "request_dispatch_ipc":
        if not isinstance(request_id, str) or not request_id:
            raise ValueError(
                "frontend dispatch coverage lacks the real AsyncLLM request_id"
            )
    elif request_id is not None and (
        not isinstance(request_id, str) or not request_id
    ):
        raise ValueError("frontend coverage telemetry request_id is invalid")
    for name in (
        "started_monotonic_s",
        "finished_monotonic_s",
        "active_span_s",
    ):
        value = record.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"frontend coverage telemetry {name} is invalid")
    return record


def _telemetry_records(
    path: Path,
    *,
    primary_schema: str,
    expected_run_id: str,
    start_offset: int = 0,
    allow_incomplete_tail: bool = False,
    include_frontend: bool = False,
) -> list[dict]:
    """Read one telemetry suffix, validating known interleaved schemas."""

    if start_offset < 0:
        raise ValueError("telemetry start offset must be non-negative")
    if not path.exists():
        return []
    with path.open("rb") as handle:
        handle.seek(start_offset)
        payload = handle.read()
    lines = payload.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        if not allow_incomplete_tail:
            raise ValueError("telemetry has an incomplete final record")
        lines.pop()
    records: list[dict] = []
    for line_number, raw_line in enumerate(lines, start=1):
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid telemetry JSON at suffix line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError("telemetry record must be an object")
        schema = record.get("schema")
        if schema == _FRONTEND_TELEMETRY_SCHEMA:
            if primary_schema != _STAGE_TELEMETRY_SCHEMA:
                raise ValueError(
                    "frontend coverage span appeared in component telemetry"
                )
            _validate_frontend_coverage_record(
                record, expected_run_id=expected_run_id,
            )
            if include_frontend:
                records.append(record)
            continue
        if schema != primary_schema:
            raise ValueError(f"unexpected telemetry schema {schema!r}")
        if record.get("run_id") != expected_run_id:
            raise ValueError("telemetry run_id mismatch")
        records.append(record)
    return records


def _primary_telemetry_run_id(path: Path, *, primary_schema: str) -> str:
    if not path.exists():
        raise ValueError("telemetry file is missing")
    run_ids: set[str] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(
                    f"telemetry has an incomplete line at {line_number}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid telemetry JSON at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError("telemetry record must be an object")
            schema = record.get("schema")
            if schema == _FRONTEND_TELEMETRY_SCHEMA:
                if primary_schema != _STAGE_TELEMETRY_SCHEMA:
                    raise ValueError(
                        "frontend coverage appeared in component telemetry"
                    )
                _validate_frontend_coverage_record(record)
            elif schema != primary_schema:
                raise ValueError(f"unexpected telemetry schema {schema!r}")
            run_id = record.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("telemetry run_id is missing")
            run_ids.add(run_id)
    if len(run_ids) != 1:
        raise ValueError("one telemetry artifact must contain exactly one run")
    return next(iter(run_ids))


def _exact_target_cycle_ids(
    records: list[dict],
    *,
    target_batch: int,
    input_tokens: int,
    output_tokens: int,
    minimum_work_sequence: int = 0,
    require_publishable_service: bool = True,
) -> tuple[list[str], int]:
    """Select one consecutive pure prefill/decode target window."""

    target_batch = int(target_batch)
    input_tokens = int(input_tokens)
    output_tokens = int(output_tokens)
    ordered: list[tuple[int, dict, tuple[int, ...], tuple[int, ...]]] = []
    for record in records:
        sequence = record.get("work_sequence")
        cycle_id = record.get("cycle_id")
        shape = record.get("shape")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < minimum_work_sequence
            or not isinstance(cycle_id, str)
            or not cycle_id
            or not isinstance(shape, dict)
        ):
            continue
        prefill = _strict_phase_shape(shape.get("prefill"), field="prefill")
        decode = _strict_phase_shape(shape.get("decode"), field="decode")
        ordered.append((sequence, record, prefill, decode))
    ordered.sort(key=lambda item: item[0])
    expected_prefill = (
        target_batch,
        target_batch * input_tokens,
        target_batch * input_tokens,
        target_batch * input_tokens * input_tokens,
    )
    start_index = None
    for index, (_sequence, record, prefill, decode) in enumerate(ordered):
        if (
            (
                not require_publishable_service
                or record.get("publishable") is True
            )
            and prefill == expected_prefill
            and decode == (0, 0, 0, 0)
        ):
            start_index = index
            break
    if start_index is None:
        raise ValueError(
            "telemetry has no publishable exact target prefill for "
            f"B{target_batch}x{input_tokens}"
        )
    selected: list[str] = []
    prefill_sequence = ordered[start_index][0]
    selected.append(str(ordered[start_index][1]["cycle_id"]))
    for decode_index in range(1, output_tokens):
        record_index = start_index + decode_index
        if record_index >= len(ordered):
            raise ValueError("telemetry target decode window is incomplete")
        sequence, record, prefill, decode = ordered[record_index]
        expected_context = target_batch * (input_tokens + decode_index)
        expected_decode = (
            target_batch,
            target_batch,
            expected_context,
            expected_context,
        )
        if (
            sequence != prefill_sequence + decode_index
            or (
                require_publishable_service
                and record.get("publishable") is not True
            )
            or prefill != (0, 0, 0, 0)
            or decode != expected_decode
        ):
            raise ValueError(
                "telemetry target window lost exact phase/batch/context at "
                f"decode step {decode_index}"
            )
        selected.append(str(record["cycle_id"]))
    return selected, prefill_sequence + output_tokens


def _component_cohort_relative_spreads(
    records: list[dict],
    cohort_cycle_ids: list[list[str]],
) -> tuple[float, float]:
    """Return full-cohort and unique-prefill service relative spreads.

    Individual CUDA-event samples may contain an isolated synchronization
    outlier.  Component publication already takes a per-shape median across
    three cohorts, so stability must detect whole-cohort cold/low-clock runs
    without reintroducing a max-over-127-samples false rejection.  Prefill is
    checked separately because one cold prefill can be hidden by the sum of a
    long, otherwise-stable decode cohort.
    """

    if len(cohort_cycle_ids) != _COMPONENT_MEASUREMENT_COHORT_COUNT:
        raise ValueError(
            "component stability proof has the wrong measured cohort count"
        )
    widths = {len(values) for values in cohort_cycle_ids}
    if len(widths) != 1 or not widths or next(iter(widths)) < 1:
        raise ValueError("component stability cohorts have inconsistent widths")
    record_by_cycle: dict[str, dict] = {}
    for record in records:
        cycle_id = record.get("cycle_id")
        if not isinstance(cycle_id, str) or not cycle_id:
            raise ValueError("component stability telemetry cycle_id is invalid")
        if cycle_id in record_by_cycle:
            raise ValueError("component stability telemetry repeats a cycle_id")
        record_by_cycle[cycle_id] = record

    totals = [0.0] * _COMPONENT_MEASUREMENT_COHORT_COUNT
    prefill_values: list[float] | None = None
    for aligned_ids in zip(*cohort_cycle_ids, strict=True):
        values: list[float] = []
        shape_digests: set[str] = set()
        phases: set[str] = set()
        for cycle_id in aligned_ids:
            record = record_by_cycle.get(cycle_id)
            if record is None:
                raise ValueError(
                    "component stability proof references a missing cycle"
                )
            service = record.get("gpu_execution_s")
            if (
                isinstance(service, bool)
                or not isinstance(service, (int, float))
                or not math.isfinite(float(service))
                or float(service) <= 0.0
            ):
                raise ValueError(
                    "component stability proof has invalid GPU service"
                )
            shape_digest = record.get("shape_digest")
            if not isinstance(shape_digest, str) or not shape_digest:
                raise ValueError(
                    "component stability proof has no exact shape digest"
                )
            shape = record.get("shape")
            phase = (
                shape.get("terminal_performance_stage")
                if isinstance(shape, dict)
                else None
            )
            if phase not in {"prefill", "decode"}:
                raise ValueError(
                    "component stability proof has no exact service phase"
                )
            values.append(float(service))
            shape_digests.add(shape_digest)
            phases.add(phase)
        if len(shape_digests) != 1:
            raise ValueError(
                "component stability cohorts are not aligned by exact shape"
            )
        if len(phases) != 1:
            raise ValueError(
                "component stability cohorts are not aligned by service phase"
            )
        if phases == {"prefill"}:
            if prefill_values is not None:
                raise ValueError(
                    "component stability proof has multiple prefill shapes"
                )
            prefill_values = values
        for index, value in enumerate(values):
            totals[index] += value
    if prefill_values is None:
        raise ValueError("component stability proof has no prefill shape")

    median_total = float(statistics.median(totals))
    median_prefill = float(statistics.median(prefill_values))
    return (
        (max(totals) - min(totals)) / median_total,
        (max(prefill_values) - min(prefill_values)) / median_prefill,
    )


def _component_cohort_max_relative_spread(
    records: list[dict],
    cohort_cycle_ids: list[list[str]],
) -> float:
    """Return the stricter of full-cohort and prefill stability spreads."""

    return max(_component_cohort_relative_spreads(records, cohort_cycle_ids))


def _read_component_cycle_records(
    path: str | os.PathLike[str],
    *,
    allowed_cycle_ids: set[str] | None = None,
) -> dict[tuple[str, str], _ComponentCycleService]:
    """Parse publishable component records through the local component schema.

    Stage-overhead evidence deliberately admits only the single-rank TP1/PP1
    reference.  A frozen artifact may concatenate one independent run per
    target batch; missing, duplicate, or non-finite records fail closed.
    """

    source = Path(path)
    records: dict[tuple[str, str], _ComponentCycleService] = {}
    with source.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(
                    f"component telemetry has an incomplete line at {line_number}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid component telemetry JSON at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError("component telemetry records must be objects")
            if (
                record.get("schema") != _COMPONENT_TELEMETRY_SCHEMA
            ):
                raise ValueError("unexpected component telemetry kind")
            if not record.get("publishable"):
                if (
                    record.get("reason")
                    == "mixed_prefill_decode_not_component_identifiable"
                ):
                    raise ValueError(
                        "component telemetry contains a mixed prefill/decode "
                        "cycle"
                    )
                continue
            if (
                record.get("world_size") != 1
                or record.get("tensor_parallel_size") != 1
                or record.get("pipeline_parallel_size") != 1
                or record.get("rank") != 0
            ):
                raise ValueError(
                    "stage overhead requires one TP1/PP1 component telemetry "
                    "reference"
                )
            run_id = record.get("run_id")
            cycle_id = record.get("cycle_id")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("component telemetry run_id is missing")
            if not isinstance(cycle_id, str) or not cycle_id:
                raise ValueError("component telemetry cycle_id is missing")
            service = record.get("gpu_execution_s")
            if (
                isinstance(service, bool)
                or not isinstance(service, (int, float))
                or not math.isfinite(float(service))
                or float(service) <= 0.0
            ):
                raise ValueError("component telemetry GPU service is invalid")
            shape = record.get("shape")
            if not isinstance(shape, dict):
                raise ValueError("component telemetry shape is missing")
            terminal = shape.get("terminal_performance_stage")
            if terminal not in {"prefill", "decode"}:
                raise ValueError(
                    "component telemetry terminal performance stage is invalid"
                )
            prefill = _strict_phase_shape(shape.get("prefill"), field="prefill")
            decode = _strict_phase_shape(shape.get("decode"), field="decode")
            active_phases = [
                name for name, phase in (("prefill", prefill), ("decode", decode))
                if phase[0] > 0
            ]
            if active_phases != [terminal]:
                raise ValueError(
                    "publishable component telemetry is not phase-pure"
                )
            inactive = decode if terminal == "prefill" else prefill
            if inactive != (0, 0, 0, 0):
                raise ValueError(
                    "publishable component telemetry has inactive-phase work"
                )
            model_identity = record.get("model_identity")
            model = (
                model_identity.get("model")
                if isinstance(model_identity, dict) else None
            )
            if not isinstance(model, str) or not model:
                raise ValueError("component telemetry model identity is missing")
            key = (run_id, cycle_id)
            if key in records:
                raise ValueError(
                    "duplicate TP1/PP1 component telemetry cycle "
                    f"{key!r}"
                )
            if allowed_cycle_ids is not None and cycle_id not in allowed_cycle_ids:
                continue
            records[key] = _ComponentCycleService(
                run_id=run_id,
                cycle_id=cycle_id,
                component_service_s=float(service),
                model=model,
                terminal_performance_stage=str(terminal),
                prefill=prefill,
                decode=decode,
            )
    if not records:
        raise ValueError("component telemetry has no publishable TP1/PP1 cycles")
    if allowed_cycle_ids is not None and {
        cycle_id for _run_id, cycle_id in records
    } != set(allowed_cycle_ids):
        raise ValueError(
            "component telemetry does not cover its frozen target cycle IDs"
        )
    return records


def read_component_cycle_services(
    path: str | os.PathLike[str],
    stage_observations: tuple[StageCycleObservation, ...] | None = None,
    *,
    stage_telemetry_path: str | os.PathLike[str] | None = None,
) -> dict[tuple[str, str], float]:
    """Rebuild GPU component service keyed by ``(run_id, cycle_id)``.

    With ``stage_observations`` this is the strict offline stage/component
    join used by the separate ``fit`` process.  Omitting them returns all
    strictly parsed TP1/PP1 component cycles for component-only diagnostics.
    """

    if stage_observations is None:
        records = _read_component_cycle_records(path)
        return {key: record.component_service_s for key, record in records.items()}
    if stage_telemetry_path is None:
        raise ValueError(
            "offline stage/component matching requires stage_telemetry_path"
        )
    component_contract = _load_run_contract(path, expected_scope="component_only")
    stage_contract = _load_run_contract(
        stage_telemetry_path, expected_scope="stage_only",
    )
    _validate_comparable_run_contracts(component_contract, stage_contract)
    component_run_by_cycle = {
        cycle_id: component_contract["subruns"][batch]["component"]["run_id"]
        for batch, values in component_contract["target_cycle_ids"].items()
        for cycle_id in values
    }
    stage_run_by_cycle = {
        cycle_id: stage_contract["subruns"][batch]["proof"]["run_id"]
        for batch, values in stage_contract["target_cycle_ids"].items()
        for cycle_id in values
    }
    component_cycle_ids = set(component_run_by_cycle)
    stage_cycle_ids = set(stage_run_by_cycle)
    if {
        item.cycle_id for item in stage_observations
    } != stage_cycle_ids:
        raise ValueError(
            "stage observations disagree with frozen target cycle IDs"
        )
    records = _read_component_cycle_records(
        path, allowed_cycle_ids=component_cycle_ids,
    )
    for record in records.values():
        if component_run_by_cycle.get(record.cycle_id) != record.run_id:
            raise ValueError(
                "component telemetry run_id disagrees with its batch subrun"
            )
    for observation in stage_observations:
        if stage_run_by_cycle.get(observation.cycle_id) != observation.run_id:
            raise ValueError(
                "stage telemetry run_id disagrees with its proof subrun"
            )
    return _match_stage_component_services(stage_observations, records)


def read_component_full_cohort_service_measurements(
    path: str | os.PathLike[str],
    stage_observations: tuple[StageCycleObservation, ...],
    *,
    stage_telemetry_path: str | os.PathLike[str],
) -> dict[int, dict[str, object]]:
    """Return independent full-cohort component totals by resident batch.

    A collocated out=N request cohort contains one prefill cycle followed by
    N-1 decode cycles.  Component acquisition freezes several independent
    exact-N cohorts.  Their cycle services must be summed *within each cohort*
    before publication takes a median; taking a per-shape median first and
    then adding would silently change the estimator.

    This reader also performs the existing exact stage/component shape join.
    It exposes no phase coefficient or token-shape lookup dimension.
    """

    if not stage_observations:
        raise ValueError("full component cohorts require stage observations")
    # Preserve the existing strict workload, run-id, model and shape audit.
    read_component_cycle_services(
        path,
        stage_observations,
        stage_telemetry_path=stage_telemetry_path,
    )
    component_contract = _load_run_contract(
        path, expected_scope="component_only",
    )
    stage_contract = _load_run_contract(
        stage_telemetry_path, expected_scope="stage_only",
    )
    _validate_comparable_run_contracts(component_contract, stage_contract)
    scheduler_cycles = int(component_contract["workload"]["output_tokens"])
    if scheduler_cycles < 2:
        raise ValueError("full component cohort requires output_tokens >= 2")

    allowed_cycle_ids = {
        cycle_id
        for cycle_ids in component_contract["target_cycle_ids"].values()
        for cycle_id in cycle_ids
    }
    records = _read_component_cycle_records(
        path, allowed_cycle_ids=allowed_cycle_ids,
    )
    records_by_id = {
        record.cycle_id: record for record in records.values()
    }
    observations_by_id = {
        observation.cycle_id: observation
        for observation in stage_observations
    }
    result: dict[int, dict[str, object]] = {}
    for raw_batch, component_cycle_ids in component_contract[
        "target_cycle_ids"
    ].items():
        batch = int(raw_batch)
        stage_cycle_ids = stage_contract["target_cycle_ids"][raw_batch]
        if len(stage_cycle_ids) != scheduler_cycles:
            raise ValueError(
                f"stage B{batch} target window must contain exactly "
                f"{scheduler_cycles} scheduler cycles"
            )
        expected_shapes = [
            _stage_shape_key(observations_by_id[cycle_id])
            for cycle_id in stage_cycle_ids
        ]
        if (
            len(component_cycle_ids) < scheduler_cycles
            or len(component_cycle_ids) % scheduler_cycles != 0
        ):
            raise ValueError(
                f"component B{batch} target IDs do not form complete "
                f"{scheduler_cycles}-cycle cohorts"
            )
        cohort_totals: list[float] = []
        for offset in range(0, len(component_cycle_ids), scheduler_cycles):
            cohort_ids = component_cycle_ids[
                offset:offset + scheduler_cycles
            ]
            cohort = [records_by_id[cycle_id] for cycle_id in cohort_ids]
            actual_shapes = [_component_shape_key(record) for record in cohort]
            if actual_shapes != expected_shapes:
                raise ValueError(
                    f"component B{batch} full cohort does not match the exact "
                    "production scheduler-cycle sequence"
                )
            cohort_totals.append(sum(
                float(record.component_service_s) for record in cohort
            ))
        result[batch] = {
            "scheduler_cycle_count": scheduler_cycles,
            "cohort_total_service_s": cohort_totals,
            "cohort_count": len(cohort_totals),
        }
    return result


def read_stage_cycle_observations(
    path: str | os.PathLike[str],
    *,
    allowed_cycle_ids: set[str] | None = None,
) -> tuple[StageCycleObservation, ...]:
    """Validate and normalize scheduler-cycle telemetry without V2 imports."""

    source = Path(path)
    expected_run_by_cycle: dict[str, str] = {}
    expected_proof_run_ids: set[str] = set()
    if allowed_cycle_ids is None and _run_contract_path(source).exists():
        contract = _load_run_contract(source, expected_scope="stage_only")
        expected_run_by_cycle = {
            cycle_id: contract["subruns"][batch]["proof"]["run_id"]
            for batch, values in contract["target_cycle_ids"].items()
            for cycle_id in values
        }
        expected_proof_run_ids = set(expected_run_by_cycle.values())
        allowed_cycle_ids = set(expected_run_by_cycle)
    observations: list[StageCycleObservation] = []
    cycle_keys: set[tuple[str, str]] = set()
    with source.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(
                    f"stage telemetry has an incomplete line at {line_number}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid stage telemetry JSON at line {line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError("stage telemetry records must be objects")
            if record.get("schema") == _FRONTEND_TELEMETRY_SCHEMA:
                _validate_frontend_coverage_record(record)
                continue
            if (
                record.get("schema") != _STAGE_TELEMETRY_SCHEMA
            ):
                raise ValueError("unexpected stage telemetry kind")
            # A proof run intentionally contains a discarded first cohort.
            # Its first scheduler sample has no previous backlog state and is
            # therefore not resident-batch evidence.  When the frozen contract
            # supplies exact target cycle IDs, reject foreign runs but skip
            # non-target warmup cycles *before* applying target-only backlog
            # and service gates.
            record_run_id = record.get("run_id")
            record_cycle_id = record.get("cycle_id")
            if expected_run_by_cycle:
                if (
                    not isinstance(record_run_id, str)
                    or record_run_id
                    not in expected_proof_run_ids
                ):
                    raise ValueError(
                        "stage telemetry contains an unexpected proof subrun"
                    )
            if (
                allowed_cycle_ids is not None
                and record_cycle_id not in allowed_cycle_ids
            ):
                continue
            service = record.get("active_service_s")
            publishable_service = record.get("publishable") is True
            if publishable_service:
                if (
                    isinstance(service, bool)
                    or not isinstance(service, (int, float))
                    or not math.isfinite(float(service))
                    or float(service) <= 0.0
                ):
                    raise ValueError(
                        "publishable stage telemetry has invalid callback interval"
                    )
                callback_interval_s: float | None = float(service)
            else:
                callback_interval_s = None
            shape = record.get("shape")
            if not isinstance(shape, dict):
                raise ValueError("publishable stage telemetry is missing shape")
            terminal = shape.get("terminal_performance_stage")
            if terminal not in {"prefill", "decode"}:
                raise ValueError("publishable stage telemetry has invalid phase")
            other = "decode" if terminal == "prefill" else "prefill"
            phase_shape = shape.get(terminal)
            other_shape = shape.get(other)
            if not isinstance(phase_shape, dict) or not isinstance(other_shape, dict):
                raise ValueError("stage telemetry phase shapes must be objects")
            other_fields = {
                name: other_shape.get(name) for name in _PHASE_SHAPE_FIELDS
            }
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in other_fields.values()
            ):
                raise ValueError("stage telemetry other-phase shape is invalid")
            if int(other_fields["scheduled_sequences"]) != 0:
                # Mixed cycles do not identify one stage's active service.
                continue
            if any(int(value) != 0 for value in other_fields.values()):
                raise ValueError(
                    "pure stage telemetry has non-zero inactive-phase work"
                )
            fields = {
                name: phase_shape.get(name)
                for name in _PHASE_SHAPE_FIELDS
            }
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in fields.values()
            ):
                raise ValueError("stage telemetry phase shape is invalid")
            batch = int(fields["scheduled_sequences"])
            if batch <= 0:
                continue
            backlog = record.get("backlog_before_interval")
            if not isinstance(backlog, dict):
                raise ValueError("stage telemetry lacks resident-batch proof")
            running = backlog.get("running_sequences")
            waiting = backlog.get("waiting_sequences")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in (running, waiting)
            ):
                raise ValueError("stage telemetry resident-batch proof is invalid")
            if int(running) + int(waiting) != batch:
                # A queued-extra or partially scheduled cycle is retained in
                # raw telemetry but cannot become resident-batch evidence.
                continue
            if terminal == "decode" and (
                int(running) != batch or int(waiting) != 0
            ):
                continue
            run_id = record.get("run_id")
            cycle_id = record.get("cycle_id")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("stage telemetry run_id is missing")
            if not isinstance(cycle_id, str) or not cycle_id:
                raise ValueError("stage telemetry cycle_id is missing")
            if allowed_cycle_ids is None and not publishable_service:
                continue
            cycle_key = (run_id, cycle_id)
            if cycle_key in cycle_keys:
                raise ValueError(f"duplicate stage telemetry cycle {cycle_key!r}")
            model_identity = record.get("model_identity")
            model = (
                model_identity.get("model")
                if isinstance(model_identity, dict) else None
            )
            tp = record.get("tensor_parallel_size")
            pp = record.get("pipeline_parallel_size")
            if not isinstance(model, str) or not model:
                raise ValueError("stage telemetry model identity is missing")
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in (tp, pp)
            ):
                raise ValueError("stage telemetry parallelism identity is invalid")
            if (
                expected_run_by_cycle
                and expected_run_by_cycle.get(cycle_id) != run_id
            ):
                raise ValueError(
                    "stage telemetry run_id disagrees with its proof subrun"
                )
            cycle_keys.add(cycle_key)
            observations.append(StageCycleObservation(
                performance_stage=f"vllm_{terminal}",
                batch_size=batch,
                active_service_s=callback_interval_s,
                scheduled_tokens=int(fields["scheduled_tokens"]),
                scheduled_context_tokens=int(fields["scheduled_context_tokens"]),
                token_context_product=int(fields["token_context_product"]),
                model=model,
                tensor_parallel_size=int(tp),
                pipeline_parallel_size=int(pp),
                run_id=run_id,
                cycle_id=cycle_id,
            ))
    if not observations:
        raise ValueError("stage telemetry has no publishable pure-stage cycles")
    if allowed_cycle_ids is not None and {
        item.cycle_id for item in observations
    } != set(allowed_cycle_ids):
        raise ValueError(
            "stage telemetry does not cover its frozen target cycle IDs"
        )
    return tuple(observations)


def _component_shape_key(
    record: _ComponentCycleService,
) -> tuple[str, int, int, int, int]:
    phase = record.terminal_performance_stage
    values = record.prefill if phase == "prefill" else record.decode
    return (phase, *values)


def _stage_shape_key(
    observation: StageCycleObservation,
) -> tuple[str, int, int, int, int]:
    phase = observation.performance_stage.removeprefix("vllm_")
    if phase not in {"prefill", "decode"}:
        raise ValueError(
            f"unsupported stage telemetry phase {observation.performance_stage!r}"
        )
    return (
        phase,
        int(observation.batch_size),
        int(observation.scheduled_tokens),
        int(observation.scheduled_context_tokens),
        int(observation.token_context_product),
    )


def _match_stage_component_services(
    stage_observations: tuple[StageCycleObservation, ...],
    component_records: dict[tuple[str, str], _ComponentCycleService],
) -> dict[tuple[str, str], float]:
    """Map independent-run component medians onto exact stage shapes."""

    if not stage_observations:
        raise ValueError("stage/component matching requires stage observations")
    stage_models = {item.model for item in stage_observations}
    component_models = {item.model for item in component_records.values()}
    if len(stage_models) != 1 or stage_models != component_models:
        raise ValueError(
            "stage/component telemetry model identity mismatch"
        )

    by_shape: dict[tuple[str, int, int, int, int], list[float]] = {}
    for component in component_records.values():
        by_shape.setdefault(_component_shape_key(component), []).append(
            float(component.component_service_s)
        )
    component_medians = {
        shape: float(statistics.median(values))
        for shape, values in by_shape.items()
    }

    joined: dict[tuple[str, str], float] = {}
    for observation in stage_observations:
        key = (observation.run_id, observation.cycle_id)
        if key in joined:
            raise ValueError(f"duplicate stage cycle in component match {key!r}")
        if (
            int(observation.tensor_parallel_size) != 1
            or int(observation.pipeline_parallel_size) != 1
        ):
            raise ValueError(
                "matched stage telemetry must use the TP1/PP1 reference"
            )
        shape = _stage_shape_key(observation)
        if shape not in component_medians:
            raise ValueError(
                "component telemetry has no exact phase-shape match for "
                f"stage cycle {key!r}: {shape!r}"
            )
        joined[key] = component_medians[shape]
    return joined


def _available_gpus(sysd: dict) -> List[str]:
    from rag_stack.performance_context import system_design_space
    return (sysd.get("available_gpus")
            or system_design_space(sysd).get("available_gpus")
            or ["cuda:0"])


def build_key(
    model: str,
    tp: int,
    pp: int,
    sysd: dict,
    devices=None,
    *,
    max_num_seqs: int | None = None,
    max_num_batched_tokens: int | None = None,
    enable_prefix_caching: bool | None = None,
    api_server_count: int | None = None,
    max_model_len: int | None = None,
):
    """A production ``VllmStartupKey`` for (model, tp, pp) — serving knobs from
    ``system_design_space`` (production defaults). ``devices`` pins the exact GPUs (for parallel
    scheduling); default = the first tp·pp of ``available_gpus``."""
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
        VllmStartupKey,
        resolve_vllm_api_server_count,
    )
    n = int(tp) * int(pp)
    dev = devices if devices is not None else _available_gpus(sysd)[:n]
    device = ",".join(str(g) for g in dev)
    from rag_stack.performance_context import system_design_space
    mss = system_design_space(sysd)
    return VllmStartupKey(
        model=model, device=device,
        tensor_parallel_size=int(tp), pipeline_parallel_size=int(pp),
        dtype=str(mss.get("vllm_dtype", "bfloat16")),
        max_num_seqs=int(
            max_num_seqs
            if max_num_seqs is not None
            else mss.get("vllm_max_num_seqs", 64)
        ),
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=enable_prefix_caching,
        max_model_len=int(max_model_len if max_model_len is not None else -1),
        kv_cache_dtype=str(mss.get("vllm_kv_cache_dtype", "auto")),
        gpu_memory_utilization=float(mss.get("vllm_gpu_memory_utilization", 0.85)),
        api_server_count=int(
            api_server_count
            if api_server_count is not None
            else resolve_vllm_api_server_count()
        ),
    )


def _prompt(n_tok: int, idx: int) -> str:
    # DISTINCT per request: unique leading tokens defeat the server's prefix cache (which is ON
    # in production), so every request pays a FULL prefill — matching production's distinct
    # queries (different retrieved chunks) and the CM's full-prefill model. Identical prompts
    # would cache the shared prefill → measured ≈ decode-only ≪ what production/CM see.
    # Qwen's completion tokenizer maps this construction to exactly ``n_tok``
    # tokens (the four numeric prefixes plus whitespace account for seven
    # tokens).  Calibration additionally checks server-reported prompt usage,
    # so a tokenizer drift fails closed instead of moving the fit x-axis.
    return (f"{idx} " * 4) + " ".join(["w"] * max(1, n_tok - 7))


async def _measure_async(proc, shapes: List[Tuple[int, int, int]], repeats: int = 1) -> Dict[Tuple[int, int, int], Tuple[float, float]]:
    """``shapes`` = [(concurrency, in_tok, out_tok)]; returns
    {(concurrency, in_tok, out_tok): (qps, lat_s)} — keyed by the FULL shape, since a grid may
    hold shapes sharing in_tok that differ only in out_tok (decode-dominated points).
    Each shape runs ``repeats`` back-to-back passes: per-request latencies are pooled (median),
    qps is the median across passes. ALL in ONE event loop (the persistent async client binds
    to it)."""
    uid = [0]
    repeats = max(1, int(repeats))

    def _nxt() -> int:
        uid[0] += 1
        return uid[0]

    w_sp: Dict[str, Any] = {"max_tokens": int(shapes[0][2]), "temperature": 0.0}
    await asyncio.gather(*[proc.generate_one(_prompt(shapes[0][1], _nxt()), w_sp) for _ in range(4)])  # warm
    out: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
    for (c, in_tok, out_tok) in shapes:
        sp: Dict[str, Any] = {"max_tokens": int(out_tok), "temperature": 0.0}
        lats: List[float] = []
        qpss: List[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            res = await asyncio.gather(*[proc.generate_one(_prompt(in_tok, _nxt()), sp) for _ in range(c)])
            wall = time.perf_counter() - t0
            lats += [p["last_token_ts"] - p["request_send_ts"] for _, p in res]
            qpss.append(c / wall)
        out[(c, in_tok, out_tok)] = (float(statistics.median(qpss)), float(statistics.median(lats)))
        print(f"  c{c:>4} in{in_tok:>5} out{out_tok:>5}  qps {out[(c, in_tok, out_tok)][0]:>8.2f}  "
              f"median_lat {out[(c, in_tok, out_tok)][1]*1000:>9.1f} ms  (x{repeats})", flush=True)
    return out


async def _sample_engine_activity(proc, session) -> tuple[int, int]:
    """Read authoritative ``(running, waiting)`` gauges from vLLM."""

    base_url = getattr(proc, "base_url", None)
    if not isinstance(base_url, str) or not base_url:
        raise ValueError("vLLM calibration process has no metrics base URL")
    root = base_url.rsplit("/v1", 1)[0]
    import aiohttp

    async with session.get(
        f"{root}/metrics", timeout=aiohttp.ClientTimeout(total=2.0),
    ) as response:
        if response.status != 200:
            raise RuntimeError(
                f"vLLM metrics returned HTTP {response.status}"
            )
        payload = await response.text()
    totals = {"running": 0.0, "waiting": 0.0}
    seen: set[str] = set()
    prefixes = {
        "running": "vllm:num_requests_running",
        "waiting": "vllm:num_requests_waiting",
    }
    for line in payload.splitlines():
        for name, prefix in prefixes.items():
            if line.startswith(prefix):
                totals[name] += float(line.rsplit(" ", 1)[1])
                seen.add(name)
    if seen != set(prefixes):
        raise RuntimeError(
            "vLLM metrics lacks running/waiting saturation evidence"
        )
    values = (totals["running"], totals["waiting"])
    if any(
        not math.isfinite(value) or value < 0.0 or not value.is_integer()
        for value in values
    ):
        raise RuntimeError("vLLM running/waiting gauges are invalid")
    return int(values[0]), int(values[1])


async def _wait_for_engine_idle(
    proc,
    session,
    *,
    timeout_s: float = 120.0,
) -> tuple[tuple[int, int], ...]:
    """Require three consecutive idle samples between target cohorts."""

    samples: list[tuple[int, int]] = []
    deadline = time.monotonic() + float(timeout_s)
    while True:
        samples.append(await _sample_engine_activity(proc, session))
        if len(samples) >= 3 and all(value == (0, 0) for value in samples[-3:]):
            return tuple(samples)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "timed out draining vLLM between stage calibration cohorts"
            )
        await asyncio.sleep(0.05)


def _telemetry_suffix_bytes(path: Path, start_offset: int) -> bytes:
    if not path.exists():
        return b""
    with path.open("rb") as handle:
        handle.seek(start_offset)
        return handle.read()


def _dispatch_ids_from_suffix(
    path: Path,
    *,
    start_offset: int,
    run_id: str,
) -> set[str]:
    payload = _telemetry_suffix_bytes(path, start_offset)
    lines = payload.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines.pop()
    dispatch_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid stage telemetry JSON at suffix line {line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError("stage telemetry record must be an object")
        schema = record.get("schema")
        if schema == _FRONTEND_TELEMETRY_SCHEMA:
            _validate_frontend_coverage_record(
                record, expected_run_id=run_id,
            )
            if record.get("kind") == "request_dispatch_ipc":
                request_id = record.get("request_id")
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("dispatch span lacks a request_id")
                dispatch_ids.add(request_id)
            continue
        if (
            schema != _STAGE_TELEMETRY_SCHEMA
            or record.get("run_id") != run_id
        ):
            raise ValueError("unexpected scheduler telemetry in dispatch fence")
    return dispatch_ids


async def _wait_for_dispatch_fence(
    path: Path,
    *,
    start_offset: int,
    run_id: str,
    expected: int,
    timeout_s: float = 60.0,
) -> int:
    """Wait until every real HTTP request completed AsyncLLM ADD dispatch."""

    deadline = time.monotonic() + float(timeout_s)
    while True:
        observed = len(_dispatch_ids_from_suffix(
            path, start_offset=start_offset, run_id=run_id,
        ))
        if observed == int(expected):
            return observed
        if observed > int(expected):
            raise RuntimeError(
                f"dispatch fence observed {observed} requests, expected {expected}"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"timed out waiting for {expected} request dispatch spans; "
                f"observed={observed}"
            )
        await asyncio.sleep(0.01)


def _server_root(proc) -> str:
    base_url = getattr(proc, "base_url", None)
    if not isinstance(base_url, str) or not base_url.endswith("/v1"):
        raise ValueError("vLLM calibration process has no canonical /v1 URL")
    return base_url[:-3]


async def _is_scheduler_paused(proc, session) -> bool:
    import aiohttp

    async with session.get(
        f"{_server_root(proc)}/is_paused",
        timeout=aiohttp.ClientTimeout(total=10.0),
    ) as response:
        data = await response.json()
        if response.status != 200 or set(data) != {"is_paused"}:
            raise RuntimeError(
                f"vLLM is_paused failed HTTP {response.status}: {data!r}"
            )
    value = data["is_paused"]
    if not isinstance(value, bool):
        raise RuntimeError("vLLM is_paused returned a non-boolean value")
    return value


async def _pause_scheduler(proc, session) -> None:
    import aiohttp

    async with session.post(
        f"{_server_root(proc)}/pause",
        params={"mode": "keep", "clear_cache": "false"},
        timeout=aiohttp.ClientTimeout(total=30.0),
    ) as response:
        data = await response.json()
        if response.status != 200 or data != {"status": "paused"}:
            raise RuntimeError(
                f"vLLM pause failed HTTP {response.status}: {data!r}; "
                "stage calibration requires VLLM_SERVER_DEV_MODE=1"
            )
    if not await _is_scheduler_paused(proc, session):
        raise RuntimeError("vLLM pause endpoint returned before scheduler paused")


async def _resume_scheduler(proc, session) -> None:
    import aiohttp

    async with session.post(
        f"{_server_root(proc)}/resume",
        timeout=aiohttp.ClientTimeout(total=30.0),
    ) as response:
        data = await response.json()
        if response.status != 200 or data != {"status": "resumed"}:
            raise RuntimeError(
                f"vLLM resume failed HTTP {response.status}: {data!r}"
            )
    if await _is_scheduler_paused(proc, session):
        raise RuntimeError("vLLM resume endpoint left scheduler paused")


_PRODUCTION_TIMING_COHORTS = PRODUCTION_COHORT_COUNT


async def _sample_timing_dispatch_count(proc, session) -> int:
    import aiohttp

    async with session.get(
        f"{_server_root(proc)}/metrics",
        timeout=aiohttp.ClientTimeout(total=2.0),
    ) as response:
        if response.status != 200:
            raise RuntimeError(
                f"vLLM metrics returned HTTP {response.status}"
            )
        payload = await response.text()
    values = [
        float(line.rsplit(" ", 1)[1])
        for line in payload.splitlines()
        if line.startswith(_TIMING_DISPATCH_METRIC + " ")
    ]
    if len(values) != 1:
        raise RuntimeError(
            "production timing server lacks its unique dispatch counter"
        )
    value = values[0]
    if not math.isfinite(value) or value < 0.0 or not value.is_integer():
        raise RuntimeError("production timing dispatch counter is invalid")
    return int(value)


async def _wait_for_paused_dispatch_count(
    proc,
    session,
    *,
    expected_count: int,
    timeout_s: float = 60.0,
) -> tuple[int, ...]:
    """Fence group-major ADDs with an in-memory counter plus FIFO utility ack."""

    expected = int(expected_count)
    if expected < 1:
        raise ValueError("paused dispatch fence must expect at least one request")
    samples: list[int] = []
    deadline = time.monotonic() + float(timeout_s)
    while True:
        sample = await _sample_timing_dispatch_count(proc, session)
        samples.append(sample)
        if sample == expected:
            # The subsequent is_paused utility RPC is ordered after earlier
            # EngineCore ADD messages and prevents a stale Prometheus sample
            # from being accepted as the sole fence.
            if await _is_scheduler_paused(proc, session):
                return tuple(samples)
            raise RuntimeError("scheduler resumed during paused queue fence")
        if sample > expected:
            raise RuntimeError(
                "production dispatch counter passed the requested cohort "
                f"fence: expected={expected}, observed={sample}"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "timed out fencing production requests before resume: "
                f"expected={expected}, last={sample}"
            )
        await asyncio.sleep(0.01)


async def _measure_production_interdeparture_async(
    proc,
    *,
    target_batch: int,
    input_tokens: int,
    output_tokens: int,
    request_contract: Mapping[str, Any],
) -> tuple[list[list[dict[str, float | int]]], list[int]]:
    """Time eight steady same-B cohorts with every telemetry hook disabled.

    Cohort zero warms graphs.  The seven completion-to-completion intervals
    that follow are the production throughput service samples.  All requests
    are already resident in the paused EngineCore queue before resume, so the
    interval contains neither dynamic-batch collection wait nor client arrival
    jitter.  It deliberately uses the same parent HTTP completion boundary as
    measured mode and does not attempt to subtract asynchronous instrumentation
    from an elapsed wall.
    """

    import aiohttp

    batch = int(target_batch)
    if int(proc.key.max_num_seqs) != batch:
        raise ValueError(
            "production timing requires engine max_num_seqs == target batch"
        )
    if os.environ.get(_STAGE_TELEMETRY_ENV):
        raise RuntimeError(
            "production timing run must not enable scheduler/frontend telemetry"
        )
    request_contract = validate_generator_request_contract(
        request_contract,
        expected_prompt_tokens=int(input_tokens),
    )
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
        REQUEST_FORMAT_CHAT_COMPLETIONS,
    )

    async def raw_generate() -> tuple[str, dict[str, float | int]]:
        payload = generator_chat_payload(
            request_contract,
            model=proc.key.model,
            output_tokens=int(output_tokens),
            served_max_model_len=int(proc._served_max_model_len),
        )
        sent = time.monotonic()
        data = await proc._raw_completion(
            payload,
            request_format=REQUEST_FORMAT_CHAT_COMPLETIONS,
        )
        completed = time.monotonic()
        choices = data.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        text = message.get("content") or ""
        usage = data.get("usage") or {}
        return text, {
            "request_send_ts": sent,
            "last_token_ts": completed,
            "n_input_tokens": int(usage.get("prompt_tokens") or 0),
            "n_output_tokens": int(usage.get("completion_tokens") or 0),
        }

    groups: list[list[asyncio.Task]] = []
    dispatch_counts: list[int] = []
    async with aiohttp.ClientSession() as metrics_session:
        await _wait_for_engine_idle(proc, metrics_session)
        try:
            await _pause_scheduler(proc, metrics_session)
            dispatch_base = await _sample_timing_dispatch_count(
                proc, metrics_session,
            )
            for group_index in range(_PRODUCTION_TIMING_COHORTS):
                group = [
                    asyncio.create_task(raw_generate()) for _ in range(batch)
                ]
                groups.append(group)
                relative_expected = (group_index + 1) * batch
                await _wait_for_paused_dispatch_count(
                    proc,
                    metrics_session,
                    expected_count=dispatch_base + relative_expected,
                )
                dispatch_counts.append(relative_expected)
            await _resume_scheduler(proc, metrics_session)
            await asyncio.gather(
                *(task for group in groups for task in group)
            )
        finally:
            try:
                if await _is_scheduler_paused(proc, metrics_session):
                    await _resume_scheduler(proc, metrics_session)
            except Exception:  # noqa: BLE001 - preserve acquisition error
                pass
            pending = [
                task for group in groups for task in group if not task.done()
            ]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        await _wait_for_engine_idle(proc, metrics_session)

    cohort_timings: list[list[dict[str, float | int]]] = []
    for group in groups:
        results = [task.result() for task in group]
        input_counts = {
            int(timing["n_input_tokens"]) for _text, timing in results
        }
        output_counts = {
            int(timing["n_output_tokens"]) for _text, timing in results
        }
        if input_counts != {int(input_tokens)} or output_counts != {
            int(output_tokens)
        }:
            raise ValueError(
                "production timing workload token counts differ from the "
                f"contract: input={sorted(input_counts)}, "
                f"output={sorted(output_counts)}"
            )
        cohort_timings.append([timing for _text, timing in results])
    return cohort_timings, dispatch_counts


async def _acquire_stage_shape_proof_async(
    proc,
    *,
    target_batch: int,
    input_tokens: int,
    output_tokens: int,
    request_contract: Mapping[str, Any],
) -> tuple[list[str], dict[str, int]]:
    """Freeze one exact second cohort; service timing is intentionally absent."""

    import aiohttp

    batch = int(target_batch)
    if int(proc.key.max_num_seqs) != batch:
        raise ValueError("shape proof requires engine max_num_seqs == target batch")
    telemetry_value = os.environ.get(_STAGE_TELEMETRY_ENV)
    run_id = os.environ.get(_CALIBRATION_RUN_ID_ENV)
    if not telemetry_value or not run_id:
        raise RuntimeError("shape proof requires isolated stage telemetry")
    telemetry_path = Path(telemetry_value).resolve()
    request_contract = validate_generator_request_contract(
        request_contract,
        expected_prompt_tokens=int(input_tokens),
    )
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
        REQUEST_FORMAT_CHAT_COMPLETIONS,
    )

    async def raw_generate() -> tuple[str, dict[str, int]]:
        payload = generator_chat_payload(
            request_contract,
            model=proc.key.model,
            output_tokens=int(output_tokens),
            served_max_model_len=int(proc._served_max_model_len),
        )
        data = await proc._raw_completion(
            payload,
            request_format=REQUEST_FORMAT_CHAT_COMPLETIONS,
        )
        choices = data.get("choices") or []
        message = (choices[0].get("message") or {}) if choices else {}
        text = message.get("content") or ""
        usage = data.get("usage") or {}
        return text, {
            "n_input_tokens": int(usage.get("prompt_tokens") or 0),
            "n_output_tokens": int(usage.get("completion_tokens") or 0),
        }

    groups: list[list[asyncio.Task]] = []
    async with aiohttp.ClientSession() as metrics_session:
        await _wait_for_engine_idle(proc, metrics_session)
        try:
            await _pause_scheduler(proc, metrics_session)
            dispatch_offset = (
                telemetry_path.stat().st_size if telemetry_path.exists() else 0
            )
            for group_index in range(2):
                group_offset = (
                    telemetry_path.stat().st_size
                    if telemetry_path.exists() else 0
                )
                group = [
                    asyncio.create_task(raw_generate()) for _ in range(batch)
                ]
                groups.append(group)
                await _wait_for_dispatch_fence(
                    telemetry_path,
                    start_offset=group_offset,
                    run_id=run_id,
                    expected=batch,
                )
                if not await _is_scheduler_paused(proc, metrics_session):
                    raise RuntimeError("shape-proof dispatch lost paused state")
            resume_offset = telemetry_path.stat().st_size
            await _resume_scheduler(proc, metrics_session)
            await asyncio.gather(*(task for group in groups for task in group))
        finally:
            try:
                if await _is_scheduler_paused(proc, metrics_session):
                    await _resume_scheduler(proc, metrics_session)
            except Exception:  # noqa: BLE001 - preserve acquisition error
                pass
            pending = [
                task for group in groups for task in group if not task.done()
            ]
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        await _wait_for_engine_idle(proc, metrics_session)

    for group in groups:
        results = [task.result() for task in group]
        if {
            int(timing["n_input_tokens"]) for _text, timing in results
        } != {int(input_tokens)} or {
            int(timing["n_output_tokens"]) for _text, timing in results
        } != {int(output_tokens)}:
            raise ValueError("shape-proof server usage differs from workload")

    deadline = time.monotonic() + 30.0
    last_error: Exception | None = None
    target_cycle_ids: list[str] = []
    while time.monotonic() < deadline:
        try:
            records = _telemetry_records(
                telemetry_path,
                primary_schema=_STAGE_TELEMETRY_SCHEMA,
                expected_run_id=run_id,
                start_offset=resume_offset,
                allow_incomplete_tail=True,
            )
            _previous_ids, next_sequence = _exact_target_cycle_ids(
                records,
                target_batch=batch,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                require_publishable_service=False,
            )
            target_cycle_ids, _ = _exact_target_cycle_ids(
                records,
                target_batch=batch,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                minimum_work_sequence=next_sequence,
                require_publishable_service=False,
            )
            break
        except ValueError as exc:
            last_error = exc
            await asyncio.sleep(0.01)
    if not target_cycle_ids:
        raise RuntimeError(
            f"timed out proving second exact B{batch} cohort"
        ) from last_error
    dispatch_count = len(_dispatch_ids_from_suffix(
        telemetry_path,
        start_offset=dispatch_offset,
        run_id=run_id,
    ))
    if dispatch_count != 2 * batch:
        raise ValueError("shape proof did not freeze exactly two cohorts")
    return target_cycle_ids, {
        "target_cohort_size": batch,
        "observed_prompt_tokens": int(input_tokens),
        "observed_output_tokens": int(output_tokens),
        "exact_prefill_cycles": 1,
        "exact_decode_cycles": max(0, int(output_tokens) - 1),
        "engine_batch_cap": batch,
        "queued_proof_cohorts": 2,
        "dispatch_spans": dispatch_count,
        "discarded_warmup_cohorts": 1,
    }


def measure(key, shapes: List[Tuple[int, int, int]], repeats: int = 1) -> Dict[Tuple[int, int, int], Tuple[float, float]]:
    """Deploy the production vLLM server for ``key`` ONCE, measure every (concurrency, in_tok,
    out_tok) shape in one event loop (``repeats`` median-pooled passes per shape), then shut
    down. Returns results keyed by the full (concurrency, in_tok, out_tok) shape. Raises on
    deploy failure."""
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import VllmSubprocess
    proc = VllmSubprocess(key)   # launches the production server subprocess (waits for health)
    try:
        return asyncio.run(_measure_async(proc, shapes, repeats=repeats))
    finally:
        proc.shutdown()


def measure_component_grid(
    key,
    shapes: List[Tuple[int, int, int]],
    *,
    telemetry_path: str | os.PathLike[str],
    request_contracts: Mapping[str, Any],
    repeats: int = _COMPONENT_TOTAL_COHORT_COUNT,
) -> Dict[Tuple[int, int, int], Tuple[float, float, float]]:
    """Measure exact offline component cohorts in an isolated child process.

    The child uses ``VLLM_ENABLE_V1_MULTIPROCESSING=0``.  This detail is
    essential: with vLLM's default background EngineCore, even ``LLM.generate``
    can execute the first request while the rest of the Python list is still
    crossing ZMQ, reproducing the same split-batch bug as the OpenAI frontend.
    """

    if int(key.tensor_parallel_size) != 1 or int(key.pipeline_parallel_size) != 1:
        raise ValueError(
            "component telemetry calibration requires the one TP1/PP1 "
            "reference deployment"
        )
    if int(getattr(key, "api_server_count", 0)) != 1:
        raise ValueError(
            "component telemetry calibration requires api_server_count=1"
        )
    if int(getattr(key, "max_model_len", -1)) != 4096:
        raise ValueError(
            "component telemetry calibration requires max_model_len=4096"
        )
    if getattr(key, "enable_prefix_caching", None) is not False:
        raise ValueError(
            "component telemetry calibration requires prefix caching disabled"
        )
    if int(repeats) != _COMPONENT_TOTAL_COHORT_COUNT:
        raise ValueError(
            "component telemetry calibration requires exactly two warmup and "
            "three measurement cohorts per shape"
        )
    largest = max(int(shape[0]) for shape in shapes)
    if int(key.max_num_seqs) < largest:
        raise ValueError(
            f"component calibration cap {key.max_num_seqs} < batch {largest}"
        )
    path = Path(telemetry_path)
    if path.exists():
        raise FileExistsError(f"refusing to append component telemetry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return _measure_component_offline_subprocess(
        key,
        shapes,
        telemetry_path=path,
        repeats=repeats,
        request_contracts=request_contracts,
    )


def _offline_component_timeout_s() -> float:
    raw = os.environ.get("RAG_STACK_V1_OFFLINE_COMPONENT_TIMEOUT_S", "7200")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(
            "RAG_STACK_V1_OFFLINE_COMPONENT_TIMEOUT_S must be numeric"
        ) from exc
    if not math.isfinite(value) or value < 60.0:
        raise ValueError(
            "RAG_STACK_V1_OFFLINE_COMPONENT_TIMEOUT_S must be finite and >=60"
        )
    return value


def _terminate_owned_process_group(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        pass
    try:
        process.wait(timeout=30.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        pass


def _positive_finite_component_values(
    value: object,
    *,
    expected_count: int,
    label: str,
) -> tuple[float, ...]:
    if (
        not isinstance(value, list)
        or len(value) != int(expected_count)
    ):
        raise ValueError(f"{label} has the wrong cohort count")
    normalized: list[float] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or float(item) <= 0.0
        ):
            raise ValueError(f"{label} contains invalid component service")
        normalized.append(float(item))
    return tuple(normalized)


def _validate_offline_component_result(
    result: object,
    *,
    result_schema: str,
    expected_run_id: str,
    requested_shapes: Sequence[Tuple[int, int, int]],
    telemetry_path: str | os.PathLike[str],
    expected_request_contracts: Mapping[str, Any],
    warmup_cohorts: int = _COMPONENT_WARMUP_COHORT_COUNT,
    measurement_cohorts: int = _COMPONENT_MEASUREMENT_COHORT_COUNT,
) -> Dict[Tuple[int, int, int], Tuple[float, float, float]]:
    """Validate child JSON against a fresh parent reconstruction of raw.

    The result file is convenience IPC, never primary evidence.  The parent
    independently proves the requested shape order and all five cohorts from
    append-only telemetry, then requires every child-reported cohort sum and
    selected median to match exactly.
    """

    normalized_shapes = [
        tuple(int(value) for value in shape) for shape in requested_shapes
    ]
    if not isinstance(result, dict):
        raise ValueError("offline component result must be an object")
    expected_result_fields = {
        "schema",
        "run_id",
        "acquisition",
        "component_boundary",
        "request_contracts",
        "warmup_cohorts",
        "measurement_cohorts",
        "total_cohorts",
        "points",
    }
    if set(result) != expected_result_fields:
        raise ValueError("offline component result fields differ from the schema")
    if (
        result.get("schema") != result_schema
        or result.get("run_id") != expected_run_id
        or result.get("acquisition")
        != "offline_inprocess_engine_full_cohort_enqueue"
        or result.get("component_boundary") != COMPONENT_SERVICE_BOUNDARY
        or result.get("warmup_cohorts") != int(warmup_cohorts)
        or result.get("measurement_cohorts") != int(measurement_cohorts)
        or result.get("total_cohorts")
        != int(warmup_cohorts) + int(measurement_cohorts)
    ):
        raise ValueError("offline component result identity/counts are invalid")
    normalized_request_contracts = validate_generator_request_contracts(
        expected_request_contracts,
        expected_prompt_tokens=[shape[1] for shape in normalized_shapes],
    )
    if validate_generator_request_contracts(
        result.get("request_contracts"),
        expected_prompt_tokens=[shape[1] for shape in normalized_shapes],
    ) != normalized_request_contracts:
        raise ValueError(
            "offline component result request contracts differ from parent spec"
        )
    if (
        int(warmup_cohorts) != _COMPONENT_WARMUP_COHORT_COUNT
        or int(measurement_cohorts) != _COMPONENT_MEASUREMENT_COHORT_COUNT
    ):
        raise ValueError(
            "offline component result requires exactly two warmup and three "
            "measurement cohorts"
        )
    points = result.get("points")
    if not isinstance(points, list):
        raise ValueError("offline component result points must be a list")
    if len(points) != len(normalized_shapes):
        raise ValueError("offline component result does not cover the requested grid")

    raw_points = _audit_offline_component_raw(
        telemetry_path,
        requested_shapes=normalized_shapes,
        expected_run_id=expected_run_id,
        warmup_cohorts=warmup_cohorts,
        measurement_cohorts=measurement_cohorts,
    )
    output: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
    point_fields = {
        "batch",
        "input_tokens",
        "output_tokens",
        "offline_cohort_qps_diagnostic",
        "offline_cohort_wall_s_diagnostic",
        "warmup_component_services_s",
        "measurement_component_services_s",
        "component_service_s",
    }
    for expected_shape, point in zip(normalized_shapes, points, strict=True):
        if not isinstance(point, dict) or set(point) != point_fields:
            raise ValueError("offline component point fields differ from the schema")
        raw_shape_values = (
            point["batch"], point["input_tokens"], point["output_tokens"],
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_shape_values
        ):
            raise ValueError("offline component result shape is not exact integer")
        shape = tuple(int(value) for value in raw_shape_values)
        if shape != expected_shape or shape in output:
            raise ValueError(
                "offline component result shape order differs from the request: "
                f"observed={shape!r}, expected={expected_shape!r}"
            )
        diagnostics = (
            point["offline_cohort_qps_diagnostic"],
            point["offline_cohort_wall_s_diagnostic"],
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in diagnostics
        ):
            raise ValueError(f"offline component diagnostics are invalid for {shape!r}")
        child_warmup = _positive_finite_component_values(
            point["warmup_component_services_s"],
            expected_count=warmup_cohorts,
            label=f"offline component warmup {shape!r}",
        )
        child_measurement = _positive_finite_component_values(
            point["measurement_component_services_s"],
            expected_count=measurement_cohorts,
            label=f"offline component measurement {shape!r}",
        )
        child_service_value = point["component_service_s"]
        if (
            isinstance(child_service_value, bool)
            or not isinstance(child_service_value, (int, float))
            or not math.isfinite(float(child_service_value))
            or float(child_service_value) <= 0.0
        ):
            raise ValueError(f"offline component selected service is invalid for {shape!r}")
        raw = raw_points[shape]
        if (
            child_warmup != raw.warmup_component_services_s
            or child_measurement != raw.measurement_component_services_s
            or float(child_service_value) != raw.component_service_s
            or float(child_service_value)
            != float(statistics.median(child_measurement))
        ):
            raise ValueError(
                "offline component child result disagrees with parent raw "
                f"reconstruction for {shape!r}"
            )
        output[shape] = (
            float(diagnostics[0]),
            float(diagnostics[1]),
            raw.component_service_s,
        )
    return output


def _measure_component_offline_subprocess(
    key,
    shapes: List[Tuple[int, int, int]],
    *,
    telemetry_path: Path,
    repeats: int,
    request_contracts: Mapping[str, Any] | None = None,
) -> Dict[Tuple[int, int, int], Tuple[float, float, float]]:
    """Run the CUDA probe behind a process boundary and validate its result."""

    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_env import (
        configure_vllm_worker_env,
    )
    from .offline_vllm_component import (
        MEASUREMENT_COHORTS,
        RESULT_SCHEMA,
        SPEC_SCHEMA,
        TOTAL_COHORTS,
        WARMUP_COHORTS,
    )

    if (
        WARMUP_COHORTS != _COMPONENT_WARMUP_COHORT_COUNT
        or MEASUREMENT_COHORTS != _COMPONENT_MEASUREMENT_COHORT_COUNT
        or TOTAL_COHORTS != _COMPONENT_TOTAL_COHORT_COUNT
        or int(repeats) != TOTAL_COHORTS
    ):
        raise ValueError(
            "offline component calibration requires exactly two warmup and "
            "three measurement cohorts per shape"
        )

    devices = [value.strip() for value in str(key.device).split(",") if value.strip()]
    if len(devices) != 1:
        raise ValueError(
            "TP1/PP1 offline component calibration requires exactly one device"
        )
    cuda_id = (
        devices[0][len("cuda:") :]
        if devices[0].startswith("cuda:")
        else devices[0]
    )
    if not cuda_id:
        raise ValueError("offline component CUDA device is empty")
    if key.max_num_batched_tokens is None:
        raise ValueError(
            "offline component calibration requires an explicit token cap"
        )
    normalized_shapes = [tuple(int(value) for value in shape) for shape in shapes]
    normalized_request_contracts = validate_generator_request_contracts(
        (
            request_contracts
            if request_contracts is not None
            else build_generator_request_contracts(
                str(key.model),
                [shape[1] for shape in normalized_shapes],
            )
        ),
        expected_prompt_tokens=[shape[1] for shape in normalized_shapes],
    )
    spec = {
        "schema": SPEC_SCHEMA,
        "startup": {
            "model": str(key.model),
            "dtype": str(key.dtype),
            "gpu_memory_utilization": float(key.gpu_memory_utilization),
            "max_model_len": int(key.max_model_len),
            "max_num_seqs": int(key.max_num_seqs),
            "max_num_batched_tokens": int(key.max_num_batched_tokens),
            "kv_cache_dtype": str(key.kv_cache_dtype),
            "enable_prefix_caching": key.enable_prefix_caching,
            "tensor_parallel_size": int(key.tensor_parallel_size),
            "pipeline_parallel_size": int(key.pipeline_parallel_size),
        },
        "shapes": [list(shape) for shape in normalized_shapes],
        "request_contracts": normalized_request_contracts,
        "warmup_cohorts": WARMUP_COHORTS,
        "measurement_cohorts": MEASUREMENT_COHORTS,
    }
    run_id = uuid.uuid4().hex
    result_path = telemetry_path.with_name(
        f".{telemetry_path.name}.{run_id}.result.json"
    )
    if result_path.exists():
        raise FileExistsError(f"refusing to replace offline result: {result_path}")
    env = os.environ.copy()
    configure_vllm_worker_env(env=env)
    env.pop(_STAGE_TELEMETRY_ENV, None)
    env[_COMPONENT_TELEMETRY_ENV] = str(telemetry_path.resolve())
    env[_CALIBRATION_RUN_ID_ENV] = run_id
    env["CUDA_VISIBLE_DEVICES"] = cuda_id
    # Must be frozen before importing vLLM in the child.  InprocClient queues
    # every request locally and cannot race the first EngineCore step.
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")
    command = [
        sys.executable,
        "-c",
        (
            "from rag_stack.cost_model.llm_sim.calibration.main_llm."
            "offline_vllm_component import main; main()"
        ),
        "--spec-json",
        json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False),
        "--result-path",
        str(result_path),
    ]
    process = subprocess.Popen(
        command,
        env=env,
        start_new_session=True,
    )
    try:
        try:
            return_code = process.wait(timeout=_offline_component_timeout_s())
        except subprocess.TimeoutExpired as exc:
            _terminate_owned_process_group(process)
            raise RuntimeError(
                "offline vLLM component calibration exceeded its timeout"
            ) from exc
        if return_code != 0:
            raise RuntimeError(
                "offline vLLM component subprocess failed with return code "
                f"{return_code}; raw={telemetry_path}"
            )
        try:
            result = json.loads(result_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "offline vLLM component subprocess produced no valid result"
            ) from exc
    finally:
        _terminate_owned_process_group(process)

    output = _validate_offline_component_result(
        result,
        result_schema=RESULT_SCHEMA,
        expected_run_id=run_id,
        requested_shapes=normalized_shapes,
        telemetry_path=telemetry_path,
        expected_request_contracts=normalized_request_contracts,
        warmup_cohorts=WARMUP_COHORTS,
        measurement_cohorts=MEASUREMENT_COHORTS,
    )
    try:
        result_path.unlink()
    except FileNotFoundError:
        pass
    return output


def measure_ladder(key, concurrencies: List[int], in_tok: int, out_tok: int) -> Dict[int, Tuple[float, float]]:
    """Single-shape convenience: a concurrency ladder at one (in_tok, out_tok) → {concurrency: (qps, lat)}."""
    res = measure(key, [(c, in_tok, out_tok) for c in concurrencies])
    return {c: res[(c, in_tok, out_tok)] for c in concurrencies if (c, in_tok, out_tok) in res}


def _measure_production_timing_on_owned_server(
    key,
    *,
    target_batch: int,
    in_tok: int,
    out_tok: int,
    request_contract: Mapping[str, Any],
) -> tuple[CollocatedEngineTimingEvidence, str, int, tuple[float, ...]]:
    """Run the no-telemetry production timing half of one batch point."""

    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
        VllmSubprocess,
    )
    if int(key.max_num_seqs) != int(target_batch):
        raise ValueError("production timing server cap must equal target batch")
    timing_run_id = uuid.uuid4().hex
    clock = capture_clock_domain_proof()
    with _scheduler_control_environment():
        proc = VllmSubprocess(key)
        try:
            served_len = int(getattr(proc, "_served_max_model_len", 0) or 0)
            if served_len != int(key.max_model_len):
                raise RuntimeError(
                    "production timing served max model length differs from "
                    "the requested calibration contract"
                )

            async def acquire():
                try:
                    short = await _measure_production_interdeparture_async(
                        proc,
                        target_batch=int(target_batch),
                        input_tokens=int(in_tok),
                        output_tokens=1,
                        request_contract=request_contract,
                    )
                    full = await _measure_production_interdeparture_async(
                        proc,
                        target_batch=int(target_batch),
                        input_tokens=int(in_tok),
                        output_tokens=int(out_tok),
                        request_contract=request_contract,
                    )
                    return short, full
                finally:
                    await proc.aclose_http()

            (short_groups, short_dispatch_counts), (
                full_groups, full_dispatch_counts,
            ) = asyncio.run(acquire())
        finally:
            proc.shutdown()

    short_evidence = build_production_cohort_service_evidence(
        timing_run_id=timing_run_id,
        engine_batch_cap=int(target_batch),
        target_batch=int(target_batch),
        input_tokens=int(in_tok),
        output_tokens=1,
        cohort_request_timings=short_groups,
        observed_dispatch_counts=short_dispatch_counts,
        enginecore_fence_ack_count=_PRODUCTION_TIMING_COHORTS,
        clock=clock,
    )
    full_evidence = build_production_cohort_service_evidence(
        timing_run_id=timing_run_id,
        engine_batch_cap=int(target_batch),
        target_batch=int(target_batch),
        input_tokens=int(in_tok),
        output_tokens=int(out_tok),
        cohort_request_timings=full_groups,
        observed_dispatch_counts=full_dispatch_counts,
        enginecore_fence_ack_count=_PRODUCTION_TIMING_COHORTS,
        clock=clock,
    )
    batch_evidence = build_collocated_engine_timing_evidence(
        full_evidence,
        out_one_diagnostic=short_evidence,
    )
    target_latencies = tuple(
        float(item["last_token_ts"]) - float(item["request_send_ts"])
        for group in full_groups[1:]
        for item in group
    )
    return batch_evidence, timing_run_id, served_len, target_latencies


def _measure_stage_shape_proof_on_owned_server(
    key,
    *,
    target_batch: int,
    in_tok: int,
    out_tok: int,
    request_contract: Mapping[str, Any],
) -> tuple[int, list[str], list[str], dict[str, int]]:
    """Run the telemetry-only exact-admission proof half of one point."""

    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
        VllmSubprocess,
    )

    if int(key.max_num_seqs) != int(target_batch):
        raise ValueError("shape-proof server cap must equal target batch")
    proc = VllmSubprocess(key)
    try:
        served_len = int(getattr(proc, "_served_max_model_len", 0) or 0)
        if served_len != int(key.max_model_len):
            raise RuntimeError(
                "shape-proof served max model length differs from contract"
            )

        async def acquire():
            try:
                short_ids, short_proof = (
                    await _acquire_stage_shape_proof_async(
                        proc,
                        target_batch=int(target_batch),
                        input_tokens=int(in_tok),
                        output_tokens=1,
                        request_contract=request_contract,
                    )
                )
                full_ids, full_proof = (
                    await _acquire_stage_shape_proof_async(
                        proc,
                        target_batch=int(target_batch),
                        input_tokens=int(in_tok),
                        output_tokens=int(out_tok),
                        request_contract=request_contract,
                    )
                )
                return short_ids, short_proof, full_ids, full_proof
            finally:
                await proc.aclose_http()

        short_ids, short_proof, full_ids, full_proof = asyncio.run(acquire())
    finally:
        proc.shutdown()
    if short_proof["engine_batch_cap"] != int(target_batch) or full_proof[
        "engine_batch_cap"
    ] != int(target_batch):
        raise ValueError("shape proof lost the per-batch engine cap")
    proof = {
        "target_cohort_size": int(target_batch),
        "observed_prompt_tokens": int(in_tok),
        "observed_output_tokens": int(out_tok),
        "exact_prefill_cycles": 1,
        "exact_decode_cycles": int(out_tok) - 1,
        "engine_batch_cap": int(target_batch),
        "queued_proof_cohorts": 4,
        "dispatch_spans": int(short_proof["dispatch_spans"])
        + int(full_proof["dispatch_spans"]),
        "discarded_warmup_cohorts": 2,
        "out_one_diagnostic_output_tokens": 1,
        "out_one_diagnostic_cycle_count": 1,
    }
    return served_len, full_ids, short_ids, proof


_RUN_CONTRACT_SCHEMA = "rag_stack.vllm_stage_component_run_contract"
_OFFLINE_COMPONENT_ACQUISITION = (
    "offline_inprocess_engine_full_cohort_enqueue"
)
_PRODUCTION_STAGE_ACQUISITION = (
    "production_chat_http_full_collocated_engine_parent_interdeparture"
)
_COMMON_BATCH_PROOF_FIELDS = {
    "target_cohort_size",
    "observed_prompt_tokens",
    "observed_output_tokens",
    "exact_prefill_cycles",
    "exact_decode_cycles",
}
_COMPONENT_BATCH_PROOF_FIELDS = _COMMON_BATCH_PROOF_FIELDS | {
    "queued_proof_cohorts",
    "discarded_warmup_cohorts",
    "selected_measurement_cohorts",
}
_STAGE_BATCH_PROOF_FIELDS = _COMMON_BATCH_PROOF_FIELDS | {
    "engine_batch_cap",
    "queued_proof_cohorts",
    "dispatch_spans",
    "discarded_warmup_cohorts",
    "out_one_diagnostic_output_tokens",
    "out_one_diagnostic_cycle_count",
}
_STAGE_TELEMETRY_ISOLATION = "tmpfs_then_frozen_copy"
_STARTUP_CONTRACT_FIELDS = {
    "model",
    "device",
    "max_num_seqs",
    "max_num_batched_tokens",
    "enable_prefix_caching",
    "dtype",
    "kv_cache_dtype",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "api_server_count",
    "requested_max_model_len",
    "served_max_model_len",
}


def _run_contract_path(raw_path: str | os.PathLike[str]) -> Path:
    path = Path(raw_path)
    return path.with_suffix(path.suffix + ".contract.json")


def _freeze_jsonl_parts(parts: list[Path], destination: Path) -> None:
    """Atomically concatenate complete isolated subruns on the destination FS."""

    if not parts:
        raise ValueError("cannot freeze an empty calibration campaign")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("wb") as output:
            for part in parts:
                payload = part.read_bytes()
                if not payload or not payload.endswith(b"\n"):
                    raise ValueError(
                        f"calibration subrun is empty or incomplete: {part}"
                    )
                output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _startup_contract(key, served_max_model_len: int) -> dict[str, object]:
    return {
        "model": str(key.model),
        "device": str(key.device),
        "max_num_seqs": int(key.max_num_seqs),
        "max_num_batched_tokens": (
            None
            if key.max_num_batched_tokens is None
            else int(key.max_num_batched_tokens)
        ),
        "enable_prefix_caching": key.enable_prefix_caching,
        "dtype": str(key.dtype),
        "kv_cache_dtype": str(key.kv_cache_dtype),
        "tensor_parallel_size": int(key.tensor_parallel_size),
        "pipeline_parallel_size": int(key.pipeline_parallel_size),
        "api_server_count": int(key.api_server_count),
        "requested_max_model_len": int(key.max_model_len),
        "served_max_model_len": int(served_max_model_len),
    }


def _subrun_contract(
    *,
    run_id: str,
    key,
    served_max_model_len: int,
) -> dict[str, object]:
    """Freeze one role-specific server identity for a single batch point."""

    return {
        "run_id": str(run_id),
        "startup": _startup_contract(key, served_max_model_len),
    }


def _contract_int(
    value: object,
    *,
    field: str,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"telemetry run contract {field} is invalid")
    return int(value)


def _validate_startup_contract(
    startup: object,
    *,
    target_batch: int,
    field: str,
) -> dict:
    if not isinstance(startup, dict) or set(startup) != _STARTUP_CONTRACT_FIELDS:
        raise ValueError(f"telemetry run contract {field} fields are invalid")
    for name in ("model", "device", "dtype", "kv_cache_dtype"):
        if not isinstance(startup.get(name), str) or not startup[name]:
            raise ValueError(f"telemetry run contract {field}.{name} is invalid")
    if _contract_int(
        startup.get("max_num_seqs"),
        field=f"{field}.max_num_seqs",
        minimum=1,
    ) != int(target_batch):
        raise ValueError(
            f"telemetry run contract {field}.max_num_seqs must equal "
            f"B{target_batch}"
        )
    max_tokens = startup.get("max_num_batched_tokens")
    if max_tokens is not None:
        _contract_int(
            max_tokens,
            field=f"{field}.max_num_batched_tokens",
            minimum=1,
        )
    if startup.get("enable_prefix_caching") is not False:
        raise ValueError(
            f"telemetry run contract {field} must disable prefix caching"
        )
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "api_server_count",
    ):
        if _contract_int(
            startup.get(name), field=f"{field}.{name}", minimum=1,
        ) != 1:
            raise ValueError(
                f"telemetry run contract requires {field}.{name}=1"
            )
    requested_len = _contract_int(
        startup.get("requested_max_model_len"),
        field=f"{field}.requested_max_model_len",
        minimum=1,
    )
    served_len = _contract_int(
        startup.get("served_max_model_len"),
        field=f"{field}.served_max_model_len",
        minimum=1,
    )
    if requested_len != 4096 or served_len != requested_len:
        raise ValueError(
            "telemetry run contract requires requested/served "
            "max_model_len=4096"
        )
    return startup


def _validate_subrun_contract(
    value: object,
    *,
    role: str,
    target_batch: int,
) -> dict:
    field = f"subruns.{target_batch}.{role}"
    if not isinstance(value, dict) or set(value) != {"run_id", "startup"}:
        raise ValueError(f"telemetry run contract {field} is invalid")
    run_id = value.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError(f"telemetry run contract {field}.run_id is invalid")
    _validate_startup_contract(
        value.get("startup"), target_batch=target_batch, field=f"{field}.startup",
    )
    return value


def _validate_run_contract_document(
    document: object,
    *,
    expected_scope: str,
) -> dict:
    if not isinstance(document, dict):
        raise ValueError("telemetry run contract must be an object")
    if (
        document.get("schema") != _RUN_CONTRACT_SCHEMA
        or document.get("scope") != expected_scope
    ):
        raise ValueError("unexpected telemetry run contract schema/scope")
    common_fields = {
        "schema",
        "scope",
        "raw_telemetry_path",
        "raw_telemetry_sha256",
        "acquisition",
        "workload",
        "subruns",
        "batch_proofs",
        "target_cycle_ids",
    }
    expected_fields = (
        common_fields
        if expected_scope == "component_only"
        else common_fields
        | {
            "contract_origin",
            "telemetry_write_isolation",
            "out_one_diagnostic_cycle_ids",
            "collocated_engine_timing_evidence",
        }
    )
    if set(document) != expected_fields:
        raise ValueError(
            "telemetry run contract top-level fields differ from exact "
            "contract"
        )

    if expected_scope == "stage_only":
        origin = document.get("contract_origin")
        if not isinstance(origin, dict):
            raise ValueError("stage run contract origin is invalid")
        if origin != {"kind": "direct_acquisition"}:
            raise ValueError("stage run contract origin is invalid")

    raw_path = document.get("raw_telemetry_path")
    raw_sha = document.get("raw_telemetry_sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("telemetry run contract raw path is invalid")
    if (
        not isinstance(raw_sha, str)
        or len(raw_sha) != 64
        or any(character not in "0123456789abcdef" for character in raw_sha)
    ):
        raise ValueError("telemetry run contract raw SHA256 is invalid")
    workload = document.get("workload")
    expected_workload_fields = {
        "target_batches",
        "prompt_tokens",
        "output_tokens",
        "request_contracts",
    } | ({
        "prefill_probe_output_tokens",
    } if expected_scope == "component_only" else {
        "out_one_diagnostic_output_tokens",
    })
    if not isinstance(workload, dict) or set(workload) != expected_workload_fields:
        raise ValueError("telemetry run contract workload is invalid")
    targets_raw = workload.get("target_batches")
    if not isinstance(targets_raw, list) or not targets_raw:
        raise ValueError("telemetry run contract target batches are missing")
    targets = [
        _contract_int(value, field="workload.target_batches", minimum=1)
        for value in targets_raw
    ]
    if targets != sorted(set(targets)):
        raise ValueError("telemetry run contract target batches are invalid")
    prompt_tokens = _contract_int(
        workload.get("prompt_tokens"),
        field="workload.prompt_tokens",
        minimum=1,
    )
    output_tokens = _contract_int(
        workload.get("output_tokens"),
        field="workload.output_tokens",
        minimum=1,
    )
    diagnostic_output_tokens = _contract_int(
        workload.get(
            "prefill_probe_output_tokens"
            if expected_scope == "component_only"
            else "out_one_diagnostic_output_tokens"
        ),
        field=(
            "workload.prefill_probe_output_tokens"
            if expected_scope == "component_only"
            else "workload.out_one_diagnostic_output_tokens"
        ),
        minimum=0,
    )
    if diagnostic_output_tokens not in {0, 1}:
        raise ValueError("out=1 diagnostic token count must be zero or one")
    request_contracts = validate_generator_request_contracts(
        workload.get("request_contracts"),
        expected_prompt_tokens=[prompt_tokens],
    )
    if expected_scope == "stage_only" and output_tokens < 2:
        raise ValueError(
            "full collocated-engine contract requires output_tokens >= 2"
        )

    expected_acquisition = (
        _OFFLINE_COMPONENT_ACQUISITION
        if expected_scope == "component_only"
        else _PRODUCTION_STAGE_ACQUISITION
    )
    if document.get("acquisition") != expected_acquisition:
        raise ValueError("telemetry run contract acquisition is invalid")
    if (
        expected_scope == "stage_only"
        and document.get("telemetry_write_isolation")
        != _STAGE_TELEMETRY_ISOLATION
    ):
        raise ValueError(
            "stage proof telemetry was not acquired on isolated tmpfs"
        )

    subruns = document.get("subruns")
    batch_proofs = document.get("batch_proofs")
    target_cycle_ids = document.get("target_cycle_ids")
    out_one_diagnostic_cycle_ids = document.get(
        "out_one_diagnostic_cycle_ids"
    )
    collocated_engine_timing_evidence = document.get(
        "collocated_engine_timing_evidence"
    )
    expected_target_keys = {str(target) for target in targets}
    for name, value in (
        ("subruns", subruns),
        ("batch proofs", batch_proofs),
        ("target cycle IDs", target_cycle_ids),
    ):
        if not isinstance(value, dict) or set(value) != expected_target_keys:
            raise ValueError(
                f"telemetry run contract {name} do not cover target batches"
            )
    if expected_scope == "stage_only":
        for name, value in (
            ("out=1 diagnostic cycle IDs", out_one_diagnostic_cycle_ids),
            (
                "collocated-engine timing evidence",
                collocated_engine_timing_evidence,
            ),
        ):
            if not isinstance(value, dict) or set(value) != expected_target_keys:
                raise ValueError(
                    f"stage telemetry contract {name} do not cover target batches"
                )

    all_run_ids: set[str] = set()
    all_cycle_ids: set[str] = set()
    campaign_startup: dict[str, object] | None = None
    for target in targets:
        subrun = subruns[str(target)]
        expected_roles = (
            {"component"}
            if expected_scope == "component_only"
            else {"proof", "timing"}
        )
        if not isinstance(subrun, dict) or set(subrun) != expected_roles:
            raise ValueError(
                f"telemetry run contract B{target} subrun roles are invalid"
            )
        descriptors = {
            role: _validate_subrun_contract(
                subrun[role], role=role, target_batch=target,
            )
            for role in expected_roles
        }
        for descriptor in descriptors.values():
            normalized_startup = dict(descriptor["startup"])
            normalized_startup.pop("max_num_seqs")
            if campaign_startup is None:
                campaign_startup = normalized_startup
            elif normalized_startup != campaign_startup:
                raise ValueError(
                    "telemetry campaign startup changes across batch subruns"
                )
        run_ids = {
            role: descriptor["run_id"]
            for role, descriptor in descriptors.items()
        }
        if len(set(run_ids.values())) != len(run_ids):
            raise ValueError(
                f"telemetry run contract B{target} roles must use independent runs"
            )
        if all_run_ids.intersection(run_ids.values()):
            raise ValueError(
                "telemetry run contract reuses a run across target batches"
            )
        all_run_ids.update(run_ids.values())
        if expected_scope == "stage_only" and (
            descriptors["proof"]["startup"]
            != descriptors["timing"]["startup"]
        ):
            raise ValueError(
                f"stage B{target} proof/timing startup contracts differ"
            )

        proof = batch_proofs[str(target)]
        expected_proof_fields = (
            _COMPONENT_BATCH_PROOF_FIELDS
            if expected_scope == "component_only"
            else _STAGE_BATCH_PROOF_FIELDS
        )
        if not isinstance(proof, dict) or set(proof) != expected_proof_fields:
            raise ValueError(
                f"telemetry run contract B{target} batch proof is invalid"
            )
        values = {
            name: _contract_int(
                proof.get(name),
                field=f"batch_proofs.{target}.{name}",
            )
            for name in expected_proof_fields
        }
        if (
            values["target_cohort_size"] != target
            or values["observed_prompt_tokens"] != prompt_tokens
            or values["observed_output_tokens"] != output_tokens
            or values["exact_prefill_cycles"] != 1
            or values["exact_decode_cycles"] != output_tokens - 1
        ):
            raise ValueError(
                f"telemetry run contract B{target} does not prove its exact "
                "target phase window"
            )
        if expected_scope == "component_only" and (
            values["queued_proof_cohorts"]
            != _COMPONENT_TOTAL_COHORT_COUNT
            or values["discarded_warmup_cohorts"]
            != _COMPONENT_WARMUP_COHORT_COUNT
            or values["selected_measurement_cohorts"]
            != _COMPONENT_MEASUREMENT_COHORT_COUNT
        ):
            raise ValueError(
                f"telemetry run contract B{target} lacks its discarded "
                "offline component warmups and repeated measurements"
            )
        if expected_scope == "stage_only" and (
            values["engine_batch_cap"] != target
            or values["queued_proof_cohorts"] != 4
            or values["dispatch_spans"] != 4 * target
            or values["discarded_warmup_cohorts"] != 2
            or values["out_one_diagnostic_output_tokens"] not in {0, 1}
            or values["out_one_diagnostic_cycle_count"] not in {0, 1}
        ):
            raise ValueError(
                f"telemetry run contract B{target} lacks its exact-cap "
                "full-engine shape proof"
            )

        cycle_run_id = (
            run_ids["component"]
            if expected_scope == "component_only"
            else run_ids["proof"]
        )
        cycle_ids = target_cycle_ids[str(target)]
        expected_cycle_id_count = output_tokens * (
            _COMPONENT_MEASUREMENT_COHORT_COUNT
            if expected_scope == "component_only"
            else 1
        )
        if (
            not isinstance(cycle_ids, list)
            or len(cycle_ids) != expected_cycle_id_count
            or any(
                not isinstance(cycle_id, str)
                or not cycle_id.startswith(f"{cycle_run_id}:")
                for cycle_id in cycle_ids
            )
            or len(set(cycle_ids)) != len(cycle_ids)
        ):
            raise ValueError(
                f"telemetry run contract B{target} target cycle IDs are invalid"
            )
        if all_cycle_ids.intersection(cycle_ids):
            raise ValueError("telemetry run contract reuses a target cycle ID")
        all_cycle_ids.update(cycle_ids)

        if expected_scope == "stage_only":
            diagnostic_ids = out_one_diagnostic_cycle_ids[str(target)]
            if diagnostic_ids is not None and (
                not isinstance(diagnostic_ids, list)
                or len(diagnostic_ids) != 1
                or not isinstance(diagnostic_ids[0], str)
                or not diagnostic_ids[0].startswith(f"{run_ids['proof']}:")
                or diagnostic_ids[0] in all_cycle_ids
            ):
                raise ValueError(
                    f"telemetry run contract B{target} out=1 diagnostic cycle "
                    "ID is invalid"
                )
            if diagnostic_ids is not None:
                all_cycle_ids.add(diagnostic_ids[0])
            try:
                timing = CollocatedEngineTimingEvidence.from_dict(
                    collocated_engine_timing_evidence[str(target)]
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"telemetry run contract B{target} timing evidence is invalid"
                ) from exc
            if (
                timing.target_batch != target
                or timing.input_tokens != prompt_tokens
                or timing.output_tokens != output_tokens
                or timing.full_engine_interdeparture.timing_run_id
                != run_ids["timing"]
                or timing.full_engine_interdeparture.engine_batch_cap != target
            ):
                raise ValueError(
                    f"telemetry run contract B{target} timing evidence differs "
                    "from its timing subrun/workload identity"
                )
            diagnostic = timing.out_one_diagnostic
            has_diagnostic = diagnostic is not None
            if (
                has_diagnostic != (diagnostic_ids is not None)
                or values["out_one_diagnostic_output_tokens"]
                != (1 if has_diagnostic else 0)
                or values["out_one_diagnostic_cycle_count"]
                != (1 if has_diagnostic else 0)
                or diagnostic_output_tokens != (1 if has_diagnostic else 0)
                or (
                    diagnostic is not None
                    and (
                        diagnostic.timing_run_id != run_ids["timing"]
                        or diagnostic.engine_batch_cap != target
                    )
                )
            ):
                raise ValueError(
                    f"telemetry run contract B{target} out=1 diagnostic "
                    "identity is inconsistent"
                )
    return document


def _write_run_contract(
    raw_path: str | os.PathLike[str],
    *,
    scope: str,
    targets: list[int],
    in_tok: int,
    out_tok: int,
    acquisition: str,
    subruns: dict[int, dict[str, dict[str, object]]],
    batch_proofs: dict[int, dict[str, int]],
    target_cycle_ids: dict[int, list[str]],
    request_contracts: Mapping[str, Any],
    out_one_diagnostic_cycle_ids: dict[int, list[str] | None] | None = None,
    collocated_engine_timing_evidence: (
        dict[int, CollocatedEngineTimingEvidence] | None
    ) = None,
    contract_origin: Mapping[str, Any] | None = None,
) -> Path:
    from rag_stack.calibration_paths import atomic_write_json

    path = Path(raw_path)
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{scope} telemetry is empty or incomplete")
    document = {
        "schema": _RUN_CONTRACT_SCHEMA,
        "scope": scope,
        "raw_telemetry_path": str(path.resolve()),
        "raw_telemetry_sha256": hashlib.sha256(raw).hexdigest(),
        "acquisition": str(acquisition),
        "workload": {
            "target_batches": [int(value) for value in targets],
            "prompt_tokens": int(in_tok),
            "output_tokens": int(out_tok),
            "request_contracts": validate_generator_request_contracts(
                request_contracts,
                expected_prompt_tokens=[int(in_tok)],
            ),
        },
        "subruns": {
            str(batch): {
                str(role): {
                    "run_id": str(descriptor["run_id"]),
                    "startup": dict(descriptor["startup"]),
                }
                for role, descriptor in sorted(roles.items())
            }
            for batch, roles in sorted(subruns.items())
        },
        "batch_proofs": {
            str(batch): {
                str(name): int(value)
                for name, value in sorted(proof.items())
            }
            for batch, proof in sorted(batch_proofs.items())
        },
        "target_cycle_ids": {
            str(batch): [str(value) for value in values]
            for batch, values in sorted(target_cycle_ids.items())
        },
    }
    document["workload"][
        "prefill_probe_output_tokens"
        if scope == "component_only"
        else "out_one_diagnostic_output_tokens"
    ] = (
        1
        if scope == "component_only"
        or any(
            value is not None
            for value in (out_one_diagnostic_cycle_ids or {}).values()
        )
        else 0
    )
    if scope == "stage_only":
        if (
            out_one_diagnostic_cycle_ids is None
            or collocated_engine_timing_evidence is None
        ):
            raise ValueError(
                "stage telemetry contract requires full engine timing and an "
                "explicit optional-diagnostic map"
            )
        document["contract_origin"] = dict(
            contract_origin or {"kind": "direct_acquisition"}
        )
        document["telemetry_write_isolation"] = _STAGE_TELEMETRY_ISOLATION
        document["out_one_diagnostic_cycle_ids"] = {
            str(batch): (
                None if values is None else [str(value) for value in values]
            )
            for batch, values in sorted(
                out_one_diagnostic_cycle_ids.items()
            )
        }
        document["collocated_engine_timing_evidence"] = {
            str(batch): evidence.to_dict()
            for batch, evidence in sorted(
                collocated_engine_timing_evidence.items()
            )
        }
    _validate_run_contract_document(document, expected_scope=scope)
    return atomic_write_json(_run_contract_path(path), document)


def _load_run_contract(
    raw_path: str | os.PathLike[str],
    *,
    expected_scope: str,
) -> dict:
    path = Path(raw_path)
    contract_path = _run_contract_path(path)
    try:
        document = json.loads(contract_path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(
            f"missing telemetry run contract: {contract_path}"
        ) from exc
    document = _validate_run_contract_document(
        document, expected_scope=expected_scope,
    )
    if document.get("raw_telemetry_path") != str(path.resolve()):
        raise ValueError(
            "telemetry raw path disagrees with its run contract"
        )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != document.get("raw_telemetry_sha256"):
        raise ValueError("telemetry raw SHA256 disagrees with its run contract")
    if expected_scope == "component_only":
        _validate_component_warmup_raw(path, document)
    return document


def _validate_component_warmup_raw(path: Path, contract: dict) -> None:
    """Prove warmup exclusion and stable repeated component measurement.

    The contract's counters are not accepted as assertions on their own.  Each
    independent batch run must contain the fixed number of consecutive exact
    cohorts; only the final repeated measurements may appear in
    ``target_cycle_ids``, and their aligned per-shape GPU services must agree.
    """

    records_by_run: dict[str, list[dict]] = {}
    physical_run_ids: list[str] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(
                    "component warmup telemetry has an incomplete line at "
                    f"{line_number}"
                )
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "invalid component warmup telemetry JSON at line "
                    f"{line_number}"
                ) from exc
            if (
                not isinstance(record, dict)
                or record.get("schema") != _COMPONENT_TELEMETRY_SCHEMA
            ):
                raise ValueError(
                    "component warmup telemetry has an unexpected schema/version"
                )
            run_id = record.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                raise ValueError("component warmup telemetry run_id is missing")
            records_by_run.setdefault(run_id, []).append(record)
            physical_run_ids.append(run_id)

    prompt_tokens = int(contract["workload"]["prompt_tokens"])
    output_tokens = int(contract["workload"]["output_tokens"])
    expected_run_ids = {
        roles["component"]["run_id"]
        for roles in contract["subruns"].values()
    }
    if set(records_by_run) != expected_run_ids:
        raise ValueError(
            "component warmup telemetry runs disagree with the run contract"
        )

    expected_physical_run_ids: list[str] = []
    for batch in contract["workload"]["target_batches"]:
        roles = contract["subruns"][str(batch)]
        expected_physical_run_ids.extend(
            [roles["component"]["run_id"]]
            * (_COMPONENT_TOTAL_COHORT_COUNT * output_tokens)
        )
    if physical_run_ids != expected_physical_run_ids:
        raise ValueError(
            "component warmup telemetry run blocks are not in contracted "
            "physical file order"
        )

    for batch_text, roles in contract["subruns"].items():
        batch = int(batch_text)
        run_id = roles["component"]["run_id"]
        records = records_by_run[run_id]
        if len(records) != _COMPONENT_TOTAL_COHORT_COUNT * output_tokens:
            raise ValueError(
                f"component B{batch} raw has the wrong cohort count"
            )
        sequences = [record.get("work_sequence") for record in records]
        if (
            any(isinstance(value, bool) or not isinstance(value, int)
                for value in sequences)
            or sequences != list(
                range(_COMPONENT_TOTAL_COHORT_COUNT * output_tokens)
            )
        ):
            raise ValueError(
                f"component B{batch} raw physical work sequence is not consecutive"
            )
        for expected_sequence, record in enumerate(records):
            if record.get("cycle_sequence") != expected_sequence:
                raise ValueError(
                    f"component B{batch} raw cycle_sequence disagrees with "
                    "work_sequence"
                )
            cycle_id = record.get("cycle_id")
            if (
                not isinstance(cycle_id, str)
                or not cycle_id.startswith(
                    f"{run_id}:{expected_sequence}:"
                )
            ):
                raise ValueError(
                    f"component B{batch} raw cycle_id is not bound to its "
                    "run/sequence prefix"
                )
        cohort_ids: list[list[str]] = []
        next_sequence = 0
        for cohort_index in range(_COMPONENT_TOTAL_COHORT_COUNT):
            ids, next_sequence = _exact_target_cycle_ids(
                records,
                target_batch=batch,
                input_tokens=prompt_tokens,
                output_tokens=output_tokens,
                minimum_work_sequence=next_sequence,
            )
            if next_sequence != (cohort_index + 1) * output_tokens:
                raise ValueError(
                    f"component B{batch} exact cohorts are not consecutive"
                )
            cohort_ids.append(ids)
        selected_cohorts = cohort_ids[_COMPONENT_WARMUP_COHORT_COUNT:]
        selected_ids = [
            cycle_id
            for cohort in selected_cohorts
            for cycle_id in cohort
        ]
        all_ids = [record.get("cycle_id") for record in records]
        if (
            next_sequence
            != _COMPONENT_TOTAL_COHORT_COUNT * output_tokens
            or any(not isinstance(value, str) or not value for value in all_ids)
            or len(set(all_ids)) != len(all_ids)
            or set(all_ids)
            != {cycle_id for cohort in cohort_ids for cycle_id in cohort}
            or selected_ids != contract["target_cycle_ids"][batch_text]
        ):
            raise ValueError(
                f"component B{batch} contract does not select only its stable "
                "measured cohorts"
            )
        total_spread, _prefill_spread = _component_cohort_relative_spreads(
            records, selected_cohorts,
        )
        if total_spread > _COMPONENT_MAX_RELATIVE_SPREAD:
            raise ValueError(
                f"component B{batch} warmed cohort totals are unstable: "
                f"relative_spread={total_spread:.6f}, "
                f"limit={_COMPONENT_MAX_RELATIVE_SPREAD:.6f}"
            )
        # A frozen full-cohort run contract proves complete scheduler-cycle
        # totals.  It intentionally does not impose a phase-only prefill
        # stability gate: neither the collocated engine curve nor the
        # zero-intercept component fit publishes a prefill residual.


def read_stage_active_service_measurements(
    raw_path: str | os.PathLike[str],
) -> dict[int, dict[str, object]]:
    """Return frozen production interdeparture service by resident batch.

    Values are trusted only after the stage contract validates its raw SHA,
    exact-cap subruns, EngineCore fences, clock domain and decomposition.
    Individual stat-logger ``active_service_s`` values are deliberately absent:
    under vLLM async scheduling they are callback cadence, not GPU-complete
    cycle service.
    """

    contract = _load_run_contract(raw_path, expected_scope="stage_only")
    result: dict[int, dict[str, object]] = {}
    for batch, document in contract[
        "collocated_engine_timing_evidence"
    ].items():
        evidence = CollocatedEngineTimingEvidence.from_dict(document)
        full = evidence.full_engine_interdeparture
        diagnostic = evidence.out_one_diagnostic
        result[int(batch)] = {
            # The publishable collocated boundary is the complete out=N
            # engine service normalized by the exact number of scheduler
            # cycles it contains.  Optional out=1 evidence remains raw
            # acquisition diagnostics and never enters this service.
            "collocated_engine_cycle_service_s": (
                full.median_service_s / evidence.output_tokens
            ),
            "collocated_engine_cycle_count": evidence.output_tokens,
            "full_probe_active_wall_s": full.median_service_s,
            "sample_count": len(full.interdeparture_s),
            "full_interdeparture_s": list(full.interdeparture_s),
            "full_interval_mad_s": full.interval_mad_s,
            "full_relative_interval_mad": full.relative_interval_mad,
            "out_one_diagnostic_interdeparture_s": (
                None
                if diagnostic is None
                else list(diagnostic.interdeparture_s)
            ),
        }
    return result


def _normalized_contract_workload(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize only the scope-specific name of optional out=1 diagnostics."""

    workload = dict(contract["workload"])
    if contract.get("scope") == "component_only":
        diagnostic_tokens = workload.pop("prefill_probe_output_tokens")
    elif contract.get("scope") == "stage_only":
        diagnostic_tokens = workload.pop("out_one_diagnostic_output_tokens")
    else:
        raise ValueError("telemetry run contract scope is invalid")
    workload["out_one_diagnostic_output_tokens"] = diagnostic_tokens
    return workload


def _validate_comparable_run_contracts(component: dict, stage: dict) -> None:
    if _normalized_contract_workload(component) != _normalized_contract_workload(
        stage
    ):
        raise ValueError(
            "component/stage telemetry workload contracts differ"
        )
    if set(component["target_cycle_ids"]) != set(stage["target_cycle_ids"]):
        raise ValueError(
            "component/stage target batch coverage differs"
        )
    component_run_ids = {
        roles["component"]["run_id"]
        for roles in component["subruns"].values()
    }
    stage_run_ids = {
        descriptor["run_id"]
        for roles in stage["subruns"].values()
        for descriptor in roles.values()
    }
    if component_run_ids.intersection(stage_run_ids):
        raise ValueError(
            "component and stage campaigns must use independent runs with "
            "disjoint run IDs"
        )
    for batch in component["target_cycle_ids"]:
        component_subrun = component["subruns"][batch]["component"]
        proof_subrun = stage["subruns"][batch]["proof"]
        timing_subrun = stage["subruns"][batch]["timing"]
        if not (
            component_subrun["startup"]
            == proof_subrun["startup"]
            == timing_subrun["startup"]
        ):
            raise ValueError(
                f"component/stage B{batch} startup contracts differ"
            )
        if len({
            component_subrun["run_id"],
            proof_subrun["run_id"],
            timing_subrun["run_id"],
        }) != 3:
            raise ValueError(
                f"component/proof/timing B{batch} must use independent runs"
            )


def measure_stage_ladder(
    key,
    concurrencies: List[int],
    in_tok: int,
    out_tok: int,
    *,
    telemetry_path: str | os.PathLike[str],
    component_telemetry_path: str | os.PathLike[str],
) -> tuple[
    Dict[int, Tuple[float, float]],
    tuple[StageCycleObservation, ...],
    dict[tuple[str, str], float],
]:
    """Measure one exact-cap component/proof/timing triplet per target batch.

    Each target B launches three independent servers with max_num_seqs=B: an
    offline component run, a telemetry-only production shape proof, and a
    telemetry-disabled parent-interdeparture timing run. All three share one
    exact chat-completions/chat-template/logprobs workload contract. The
    offline run discards two exact-shape warmup cohorts, then freezes three
    aligned cohorts whose per-shape median is the narrow worker-CUDA component
    reference. Their startup and request contracts must match within B.
    """

    targets = sorted({int(value) for value in concurrencies})
    if not targets or targets[0] < 1:
        raise ValueError("stage calibration batches must be positive")
    if int(in_tok) < 1 or int(out_tok) < 2:
        raise ValueError(
            "stage calibration requires positive input tokens and output_tokens >= 2"
        )
    if int(key.tensor_parallel_size) != 1 or int(key.pipeline_parallel_size) != 1:
        raise ValueError(
            "stage/component telemetry requires the TP1/PP1 reference"
        )
    if int(getattr(key, "api_server_count", 0)) != 1:
        raise ValueError(
            "stage/component telemetry requires api_server_count=1"
        )
    if int(getattr(key, "max_model_len", -1)) != 4096:
        raise ValueError(
            "stage/component telemetry requires max_model_len=4096"
        )
    if getattr(key, "enable_prefix_caching", None) is not False:
        raise ValueError(
            "stage/component telemetry requires prefix caching disabled"
        )
    if int(key.max_num_seqs) < targets[-1]:
        raise ValueError(
            "stage calibration engine cap is smaller than its largest batch: "
            f"cap={key.max_num_seqs}, target={targets[-1]}"
        )
    request_contracts = build_generator_request_contracts(
        str(key.model),
        [int(in_tok)],
    )
    request_contract = request_contracts[str(int(in_tok))]

    stage_path = Path(telemetry_path)
    component_path = Path(component_telemetry_path)
    if stage_path.resolve() == component_path.resolve():
        raise ValueError("stage and component telemetry paths must be distinct")
    owned_outputs = (
        stage_path,
        component_path,
        _run_contract_path(stage_path),
        _run_contract_path(component_path),
    )
    for path in owned_outputs:
        if path.exists():
            raise FileExistsError(
                f"refusing to replace stage calibration artifact: {path}"
            )
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    component_path.parent.mkdir(parents=True, exist_ok=True)
    tmpfs_root = Path("/dev/shm")
    if not tmpfs_root.is_dir() or not os.access(tmpfs_root, os.W_OK):
        raise RuntimeError(
            "stage calibration requires writable /dev/shm for proof telemetry"
        )

    component_parts: list[Path] = []
    stage_parts: list[Path] = []
    component_subruns: dict[int, dict[str, dict[str, object]]] = {}
    stage_subruns: dict[int, dict[str, dict[str, object]]] = {}
    component_batch_proofs: dict[int, dict[str, int]] = {}
    stage_batch_proofs: dict[int, dict[str, int]] = {}
    component_target_cycle_ids: dict[int, list[str]] = {}
    stage_target_cycle_ids: dict[int, list[str]] = {}
    stage_out_one_diagnostic_cycle_ids: dict[int, list[str] | None] = {}
    collocated_engine_timing_evidence: dict[
        int,
        CollocatedEngineTimingEvidence,
    ] = {}
    ladder: Dict[int, Tuple[float, float]] = {}
    campaign_complete = False
    frozen_cohort_written = False

    try:
        for target in targets:
            batch_key = dataclasses.replace(key, max_num_seqs=int(target))

            component_part = component_path.with_name(
                f".{component_path.name}.B{target}."
                f"{uuid.uuid4().hex}.component.jsonl"
            )
            component_parts.append(component_part)
            component_grid = _measure_component_offline_subprocess(
                batch_key,
                [(int(target), int(in_tok), int(out_tok))],
                telemetry_path=component_part,
                repeats=_COMPONENT_TOTAL_COHORT_COUNT,
                request_contracts=request_contracts,
            )
            expected_shape = (int(target), int(in_tok), int(out_tok))
            if set(component_grid) != {expected_shape}:
                raise ValueError(
                    f"offline component B{target} grid is incomplete"
                )
            component_run_id = _primary_telemetry_run_id(
                component_part,
                primary_schema=_COMPONENT_TELEMETRY_SCHEMA,
            )
            component_records = _telemetry_records(
                component_part,
                primary_schema=_COMPONENT_TELEMETRY_SCHEMA,
                expected_run_id=component_run_id,
            )
            component_cohort_cycle_ids: list[list[str]] = []
            next_component_sequence = 0
            for cohort_index in range(_COMPONENT_TOTAL_COHORT_COUNT):
                cycle_ids, next_component_sequence = _exact_target_cycle_ids(
                    component_records,
                    target_batch=int(target),
                    input_tokens=int(in_tok),
                    output_tokens=int(out_tok),
                    minimum_work_sequence=next_component_sequence,
                )
                if next_component_sequence != (
                    (cohort_index + 1) * int(out_tok)
                ):
                    raise ValueError(
                        f"offline component B{target} cohorts are not consecutive"
                    )
                component_cohort_cycle_ids.append(cycle_ids)
            measured_component_cohorts = component_cohort_cycle_ids[
                _COMPONENT_WARMUP_COHORT_COUNT:
            ]
            component_cycle_ids = [
                cycle_id
                for cohort in measured_component_cohorts
                for cycle_id in cohort
            ]
            observed_component_cycle_ids = {
                str(record.get("cycle_id")) for record in component_records
            }
            expected_component_cycle_ids = {
                cycle_id
                for cohort in component_cohort_cycle_ids
                for cycle_id in cohort
            }
            if (
                len(component_records)
                != _COMPONENT_TOTAL_COHORT_COUNT * int(out_tok)
                or len(observed_component_cycle_ids) != len(component_records)
                or observed_component_cycle_ids != expected_component_cycle_ids
                or next_component_sequence
                != _COMPONENT_TOTAL_COHORT_COUNT * int(out_tok)
            ):
                raise ValueError(
                    f"offline component B{target} did not produce its exact "
                    "warmup and measurement cohorts"
                )
            total_spread, _prefill_spread = _component_cohort_relative_spreads(
                component_records, measured_component_cohorts,
            )
            if total_spread > _COMPONENT_MAX_RELATIVE_SPREAD:
                raise ValueError(
                    f"offline component B{target} warmed cohort totals are unstable: "
                    f"relative_spread={total_spread:.6f}, "
                    f"limit={_COMPONENT_MAX_RELATIVE_SPREAD:.6f}"
                )
            # The collocated stage profile consumes only the complete
            # scheduler-cycle cohort total.  A single prefill sample is kept
            # as a consistency diagnostic, but must not revive the retired
            # prefill/decode residual split by becoming a publication gate.
            # Whole-cohort stability above remains mandatory.
            component_target_cycle_ids[target] = component_cycle_ids
            component_batch_proofs[target] = {
                "target_cohort_size": int(target),
                "observed_prompt_tokens": int(in_tok),
                "observed_output_tokens": int(out_tok),
                "exact_prefill_cycles": 1,
                "exact_decode_cycles": int(out_tok) - 1,
                "queued_proof_cohorts": _COMPONENT_TOTAL_COHORT_COUNT,
                "discarded_warmup_cohorts": _COMPONENT_WARMUP_COHORT_COUNT,
                "selected_measurement_cohorts": (
                    _COMPONENT_MEASUREMENT_COHORT_COUNT
                ),
            }
            component_subruns[target] = {
                "component": _subrun_contract(
                    run_id=component_run_id,
                    key=batch_key,
                    served_max_model_len=int(batch_key.max_model_len),
                )
            }

            proof_part = tmpfs_root / (
                f"rag_stack_v1_stage_B{target}_{uuid.uuid4().hex}.jsonl"
            )
            stage_parts.append(proof_part)
            with _stage_telemetry_environment(proof_part) as proof_run_id:
                (
                    proof_served_len,
                    full_cycle_ids,
                    prefill_cycle_ids,
                    batch_proof,
                ) = _measure_stage_shape_proof_on_owned_server(
                    batch_key,
                    target_batch=int(target),
                    in_tok=int(in_tok),
                    out_tok=int(out_tok),
                    request_contract=request_contract,
                )
            (
                timing_evidence,
                timing_run_id,
                timing_served_len,
                target_latencies,
            ) = _measure_production_timing_on_owned_server(
                batch_key,
                target_batch=int(target),
                in_tok=int(in_tok),
                out_tok=int(out_tok),
                request_contract=request_contract,
            )
            if proof_served_len != timing_served_len:
                raise ValueError(
                    f"stage B{target} proof/timing served max lengths differ"
                )
            if (
                not target_latencies
                or any(
                    not math.isfinite(value) or value <= 0.0
                    for value in target_latencies
                )
            ):
                raise ValueError(
                    f"stage B{target} timing lacks positive request latencies"
                )

            stage_target_cycle_ids[target] = list(full_cycle_ids)
            stage_out_one_diagnostic_cycle_ids[target] = list(
                prefill_cycle_ids
            )
            stage_batch_proofs[target] = dict(batch_proof)
            collocated_engine_timing_evidence[target] = timing_evidence
            stage_subruns[target] = {
                "proof": _subrun_contract(
                    run_id=proof_run_id,
                    key=batch_key,
                    served_max_model_len=proof_served_len,
                ),
                "timing": _subrun_contract(
                    run_id=timing_run_id,
                    key=batch_key,
                    served_max_model_len=timing_served_len,
                ),
            }
            ladder[target] = (
                float(target)
                / float(
                    timing_evidence.full_engine_interdeparture.median_service_s
                ),
                float(statistics.median(target_latencies)),
            )

        _freeze_jsonl_parts(component_parts, component_path)
        _freeze_jsonl_parts(stage_parts, stage_path)
        _write_run_contract(
            component_path,
            scope="component_only",
            targets=targets,
            in_tok=in_tok,
            out_tok=out_tok,
            acquisition=_OFFLINE_COMPONENT_ACQUISITION,
            subruns=component_subruns,
            batch_proofs=component_batch_proofs,
            target_cycle_ids=component_target_cycle_ids,
            request_contracts=request_contracts,
        )
        _write_run_contract(
            stage_path,
            scope="stage_only",
            targets=targets,
            in_tok=in_tok,
            out_tok=out_tok,
            acquisition=_PRODUCTION_STAGE_ACQUISITION,
            subruns=stage_subruns,
            batch_proofs=stage_batch_proofs,
            target_cycle_ids=stage_target_cycle_ids,
            request_contracts=request_contracts,
            out_one_diagnostic_cycle_ids=(
                stage_out_one_diagnostic_cycle_ids
            ),
            collocated_engine_timing_evidence=(
                collocated_engine_timing_evidence
            ),
        )
        frozen_cohort_written = True
        if set(ladder) != set(targets):
            raise ValueError(
                "production stage campaign did not complete every target batch"
            )
        observations = read_stage_cycle_observations(stage_path)
        component_services = read_component_cycle_services(
            component_path,
            observations,
            stage_telemetry_path=stage_path,
        )
        missing = [
            (phase, batch)
            for phase in ("vllm_prefill", "vllm_decode")
            for batch in targets
            if not any(
                item.performance_stage == phase and item.batch_size == batch
                for item in observations
            )
        ]
        if missing:
            raise ValueError(
                "stage calibration did not achieve exact resident phase/batch(es): "
                + ", ".join(f"{phase}/B{batch}" for phase, batch in missing)
            )
        campaign_complete = True
        return ladder, observations, component_services
    finally:
        for part in (*component_parts, *stage_parts):
            try:
                part.unlink()
            except FileNotFoundError:
                pass
        if not campaign_complete:
            if frozen_cohort_written:
                quarantine = stage_path.parent / (
                    f"failed_stage_campaign_{uuid.uuid4().hex}"
                )
                try:
                    quarantine.mkdir(parents=False, exist_ok=False)
                    for path in owned_outputs:
                        if path.exists():
                            os.replace(path, quarantine / path.name)
                except OSError as exc:
                    print(
                        "[generator-stage] WARNING: could not fully quarantine "
                        f"the failed frozen cohort: {exc}",
                        flush=True,
                    )
                else:
                    print(
                        "[generator-stage] preserved failed frozen cohort for "
                        f"CPU diagnosis at {quarantine}",
                        flush=True,
                    )
            else:
                for path in owned_outputs:
                    try:
                        path.unlink()
                    except FileNotFoundError:
                        pass


def parallel_measure(items: List[dict], sysd: dict, fn) -> list:
    """Run ``fn(item, devices) -> rows`` for each item CONCURRENTLY across the box's GPUs, to keep
    the full calibration fast. Items are grouped by GPU cost (``item['n_gpus']``) and run **big-first**
    in waves of ``num_gpus // n`` (each wave = contiguous GPU slices) — deadlock-free, full utilisation.
    Concurrent vLLM servers isolate via per-subprocess CUDA_VISIBLE_DEVICES (vllm_subprocess.py). Returns
    the flattened rows from all items."""
    import concurrent.futures as _cf
    from collections import defaultdict
    gpus = _available_gpus(sysd)
    ngpu = max(1, len(gpus))
    groups: Dict[int, list] = defaultdict(list)
    for it in items:
        groups[int(it["n_gpus"])].append(it)
    out: list = []
    for n in sorted(groups, reverse=True):           # 4-GPU configs before 2- before 1-
        per_wave = max(1, ngpu // n)
        g = groups[n]
        for i in range(0, len(g), per_wave):
            wave = g[i:i + per_wave]
            with _cf.ThreadPoolExecutor(max_workers=len(wave)) as ex:
                futs = {ex.submit(fn, it, gpus[j * n:(j + 1) * n]): it for j, it in enumerate(wave)}
                for f in _cf.as_completed(futs):
                    r = f.result()
                    if r:
                        out.extend(r if isinstance(r, list) else [r])
    return out
