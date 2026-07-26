"""Direct calibration for independent vLLM prefill/decode engines.

This campaign deliberately measures a *role boundary*, not a deployment.  One
cheap TP1/PP1 P/D pair supplies hardware x role x resident-batch evidence; the
analytical component model remains responsible for model and parallelism
physics.  Collocated timing is never used to infer either P or D service.

Production service comes only from broad monotonic active windows:

* P: four exact-B prefill cohorts are queued behind a paused producer.  The
  elapsed parent window divided by four is one completed-P-frontier sample.
  The deliberately large acquisition token budget makes each B-resident
  cohort complete in one pure scheduler cycle, but that is a proof mechanism,
  not the runtime charging boundary or a profile dimension.  Production may
  chunk the same logical prefill across many cycles and still charges its
  frontier only once.  A separate warmup window and three independent measured
  windows are run.
* D: one exact-B cohort is queued behind a paused consumer.  A fixed 256-token
  acquisition guard gives asynchronous NIXL arrivals time to fill the resident
  cap; it does not change the canonical 128-token production/component shape.
  The immutable scheduler log supplies the first 33 consecutive pure exact-B
  work cycles whose aggregate context advances by exactly B.  The first cycle
  is the CLOCK_MONOTONIC boundary and the following 32 cycles are timed.  A
  separate warmup cohort and three independent measured cohorts are run; no
  client-SSE clock, 2B feeder, retry, or timing-based sample selection enters
  the campaign.

Scheduler ``active_service_s`` is never read as a fit value.  Role-separated
telemetry proves only that every selected cycle is phase-pure, has the exact
resident batch, and has a concrete token shape.  NIXL before/after deltas are
stored as independent physical evidence and never folded into a role curve.

The sparse curve contract is fixed at anchors 1/8/32/64/256 with B128 held
out.  Component references come from the existing independent TP1/PP1 raw
campaign.  Decode shapes that fall between its uniform context points use
strict bracketing piecewise-linear interpolation (never extrapolation); this
introduces no fitted coefficient or runtime knob.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
import uuid

from rag_stack.calibration_paths import (
    atomic_write_json,
    calibration_data_root,
)
from rag_stack.cost_model.llm_sim.calibration.main_llm.generator_request_contract import (
    generator_chat_payload,
    validate_generator_request_contract,
)
from rag_stack.cost_model.llm_sim.calibration.main_llm.nixl_physical_evidence import (
    derive_nixl_physical_evidence_from_text,
)


ROLE_TIMING_SCHEMA = "rag_stack.direct_disagg_role_timing"
ROLE_CONTRACT_SCHEMA = "rag_stack.direct_disagg_role_run_contract"
NIXL_CAMPAIGN_SCHEMA = "rag_stack.direct_disagg_nixl_physical_campaign"

PROMPT_TOKENS = 112
# Canonical production/component workload shape.  Direct-D acquisition uses a
# longer fixed guard below only to expose a full-resident plateau after NIXL's
# asynchronous arrival ramp; it is not a calibration dimension.
OUTPUT_TOKENS = 128
DECODE_ACQUISITION_OUTPUT_TOKENS = 256
SERVED_MAX_MODEL_LEN = 4096
TARGET_BATCHES: tuple[int, ...] = (1, 8, 32, 64, 128, 256)
ACQUISITION_BATCH_ORDER: tuple[int, ...] = (256, 128, 64, 32, 8, 1)
ANCHOR_BATCHES: tuple[int, ...] = (1, 8, 32, 64, 256)
HELD_OUT_BATCH = 128
MEASURED_SAMPLE_COUNT = 3
PREFILL_CYCLES_PER_WINDOW = 4
MIN_DECODE_CYCLES_PER_WINDOW = 32
DECODE_CONSECUTIVE_CYCLES_PER_WINDOW = MIN_DECODE_CYCLES_PER_WINDOW + 1
MAX_NUM_BATCHED_TOKENS = 65_536

if (
    len(ACQUISITION_BATCH_ORDER) != len(TARGET_BATCHES)
    or set(ACQUISITION_BATCH_ORDER) != set(TARGET_BATCHES)
):
    raise RuntimeError(
        "direct P/D acquisition order must cover every canonical batch once"
    )

_SCHEDULER_SCHEMA = "rag_stack.vllm.scheduler_cycle_stage_service"
_FRONTEND_SCHEMA = "rag_stack.vllm.frontend_stage_boundary_span"
_PHASE_FIELDS = (
    "scheduled_sequences",
    "scheduled_tokens",
    "scheduled_context_tokens",
    "token_context_product",
)


def _sha256_path(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timing_path(system_hardware: str, role: str) -> Path:
    return (
        calibration_data_root()
        / "generator_stage_batch"
        / f"{system_hardware}.disagg_{role}.json"
    )


def _contract_path(system_hardware: str, role: str) -> Path:
    return Path(f"{_timing_path(system_hardware, role)}.contract.json")


def _nixl_path(system_hardware: str) -> Path:
    return (
        calibration_data_root()
        / "generator_stage_batch"
        / f"{system_hardware}.disagg_nixl_physical.json"
    )


def _component_path(system_hardware: str) -> Path:
    return (
        calibration_data_root()
        / "generator_stage_batch"
        / f"{system_hardware}.component.jsonl"
    )


def _telemetry_path(
    system_hardware: str,
    role: str,
    batch: int,
    campaign_id: str,
) -> Path:
    return (
        calibration_data_root()
        / "generator_stage_batch"
        / (
            f"{system_hardware}.disagg_{role}.b{int(batch)}."
            f"{campaign_id}.telemetry.jsonl"
        )
    )


def _positive_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _strict_shape(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(_PHASE_FIELDS):
        raise ValueError(f"{label} does not have the exact phase-shape fields")
    result: dict[str, int] = {}
    for field in _PHASE_FIELDS:
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{label}.{field} must be a non-negative integer")
        result[field] = int(item)
    return result


def _strict_backlog(value: Any, *, label: str) -> dict[str, int]:
    fields = {"running_sequences", "waiting_sequences"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} does not have the exact backlog fields")
    result: dict[str, int] = {}
    for field in fields:
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{label}.{field} must be a non-negative integer")
        result[field] = int(item)
    return result


def _scheduler_records(
    path: Path,
    *,
    run_id: str,
    start_offset: int = 0,
) -> list[dict[str, Any]]:
    """Read one immutable role log without interpreting callback service."""

    if isinstance(start_offset, bool) or not isinstance(start_offset, int):
        raise ValueError("role telemetry start_offset must be an integer")
    if start_offset < 0:
        raise ValueError("role telemetry start_offset must be non-negative")
    records: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        handle.seek(start_offset)
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.endswith(b"\n"):
                raise ValueError(
                    f"incomplete role telemetry at {path}:{line_number}"
                )
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                raise ValueError("role telemetry records must be objects")
            if record.get("run_id") != run_id:
                raise ValueError("role telemetry contains a foreign run_id")
            schema = record.get("schema")
            if schema == _FRONTEND_SCHEMA:
                # Frontend spans are retained in the immutable artifact but do
                # not participate in scheduler-shape proof.
                continue
            if schema != _SCHEDULER_SCHEMA:
                continue
            records.append(record)
    if not records:
        raise ValueError("role telemetry contains no scheduler work")
    return records


def _pure_cycle(
    record: Mapping[str, Any],
    *,
    role: str,
    batch: int,
    require_publishable: bool = False,
) -> dict[str, Any] | None:
    phase = "prefill" if role == "prefill" else "decode"
    inactive_phase = "decode" if phase == "prefill" else "prefill"
    shape = record.get("shape")
    if not isinstance(shape, Mapping):
        return None
    # ``publishable`` describes whether callback ``active_service_s`` is a
    # valid interval. Direct P/D production service never consumes that field,
    # so callback publishability must not filter shape/count evidence.
    del require_publishable
    if shape.get("terminal_performance_stage") != phase:
        return None
    active = _strict_shape(shape.get(phase), label=f"{phase} shape")
    inactive = _strict_shape(
        shape.get(inactive_phase), label=f"{inactive_phase} shape"
    )
    if any(inactive.values()):
        return None
    if active["scheduled_sequences"] != int(batch):
        return None
    expected_tokens = (
        int(batch) * PROMPT_TOKENS if role == "prefill" else int(batch)
    )
    if active["scheduled_tokens"] != expected_tokens:
        return None
    if role == "prefill" and active != {
        "scheduled_sequences": int(batch),
        "scheduled_tokens": int(batch) * PROMPT_TOKENS,
        "scheduled_context_tokens": int(batch) * PROMPT_TOKENS,
        "token_context_product": int(batch) * PROMPT_TOKENS * PROMPT_TOKENS,
    }:
        return None
    if role == "decode" and (
        active["scheduled_context_tokens"] <= 0
        or active["token_context_product"]
        != active["scheduled_context_tokens"]
    ):
        return None
    timestamp = record.get("monotonic_timestamp_s")
    cycle_id = record.get("cycle_id")
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not math.isfinite(float(timestamp))
        or not isinstance(cycle_id, str)
        or not cycle_id
    ):
        raise ValueError("selected scheduler work lacks time/cycle identity")
    # active_service_s and even the callback timestamp are intentionally
    # absent from the persisted proof. The timestamp selects the parent-clock
    # interval here; production service remains role_window/count.
    return {
        "cycle_id": cycle_id,
        "monotonic_timestamp_s": float(timestamp),
        "actual_shape": active,
    }


def select_window_cycles(
    records: Sequence[Mapping[str, Any]],
    *,
    role: str,
    batch: int,
    started_monotonic_s: float,
    completed_monotonic_s: float,
    require_publishable: bool = False,
) -> list[dict[str, Any]]:
    """Prove every scheduler record enclosed by a parent active window.

    A work record in the interval that is not the exact requested pure phase
    invalidates the window; silently selecting only convenient rows would make
    telemetry a fit filter rather than a proof.
    """

    started = _positive_float(started_monotonic_s, label="window start")
    completed = _positive_float(completed_monotonic_s, label="window end")
    if completed <= started:
        raise ValueError("parent active window is not positive")
    enclosed: list[Mapping[str, Any]] = []
    for record in records:
        timestamp = record.get("monotonic_timestamp_s")
        shape = record.get("shape")
        if (
            isinstance(timestamp, (int, float))
            and not isinstance(timestamp, bool)
            and math.isfinite(float(timestamp))
            and isinstance(shape, Mapping)
            and float(timestamp) > started
            and float(timestamp) <= completed
        ):
            # Idle callbacks are part of the parent wall (including any
            # frontend/feed bubble) but are not scheduler work and therefore
            # do not contribute a divisor cycle.  A row with work in either
            # phase is retained and must pass the exact role/B proof below.
            prefill = _strict_shape(
                shape.get("prefill"), label="window prefill shape"
            )
            decode = _strict_shape(
                shape.get("decode"), label="window decode shape"
            )
            if not any(prefill.values()) and not any(decode.values()):
                continue
            enclosed.append(record)
    selected: list[dict[str, Any]] = []
    for record in enclosed:
        normalized = _pure_cycle(
            record,
            role=role,
            batch=batch,
            require_publishable=require_publishable,
        )
        if normalized is None:
            raise ValueError(
                f"{role} parent window contains non-pure/non-B{batch} work"
            )
        selected.append({
            "cycle_id": normalized["cycle_id"],
            "actual_shape": normalized["actual_shape"],
        })
    expected = (
        PREFILL_CYCLES_PER_WINDOW
        if role == "prefill"
        else MIN_DECODE_CYCLES_PER_WINDOW
    )
    if role == "prefill" and len(selected) != expected:
        raise ValueError(
            f"prefill window must prove exactly {expected} pure B{batch} cycles; "
            f"observed={len(selected)}"
        )
    if role == "decode" and len(selected) < expected:
        raise ValueError(
            f"decode frontier window must prove at least {expected} pure B{batch} "
            f"cycles; observed={len(selected)}"
        )
    if role == "decode":
        contexts = [
            cycle["actual_shape"]["scheduled_context_tokens"]
            for cycle in selected
        ]
        if any(right <= left for left, right in zip(contexts, contexts[1:])):
            raise ValueError(
                "decode parent window contexts are not strictly increasing"
            )
    return selected


def build_window_sample(
    *,
    sample_index: int,
    role: str,
    batch: int,
    started_monotonic_s: float,
    completed_monotonic_s: float,
    scheduler_records: Sequence[Mapping[str, Any]],
    require_publishable: bool = False,
) -> dict[str, Any]:
    selected = select_window_cycles(
        scheduler_records,
        role=role,
        batch=batch,
        started_monotonic_s=started_monotonic_s,
        completed_monotonic_s=completed_monotonic_s,
        require_publishable=require_publishable,
    )
    window = float(completed_monotonic_s) - float(started_monotonic_s)
    return {
        "sample_index": int(sample_index),
        "window_source": "parent_resume_to_all_completion_four_exact_cycles",
        "window_started_monotonic_s": float(started_monotonic_s),
        "window_completed_monotonic_s": float(completed_monotonic_s),
        "role_window_s": window,
        "scheduler_cycle_count": len(selected),
        "observed_service_per_cycle_s": window / len(selected),
        "selected_cycles": selected,
        "window_proof": {},
    }


def build_decode_context_window_sample(
    *,
    sample_index: int,
    batch: int,
    scheduler_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure the first saturated exact-B direct-D scheduler plateau.

    SSE delivery is not an engine clock, and asynchronous NIXL arrivals can
    spend many scheduler cycles ramping to the resident cap.  The acquisition
    request therefore uses a fixed 256-token guard while this selector consumes
    only immutable scheduler shapes and ``CLOCK_MONOTONIC`` timestamps.
    Callback ``active_service_s`` and ``raw_interval_s`` are never consumed.

    The first 33 consecutive *work* cycles that are pure decode, exact B, have
    running=B/waiting=0 both before and after, and advance aggregate context by
    exactly B define the sample.  Cycle zero is a boundary; the 32 following
    interdeparture intervals are timed and persisted as selected cycles.  Idle
    callbacks remain in wall time but do not break work-cycle consecutiveness.
    Any non-pure/partial work, admission/finish transition, or context jump
    resets the candidate run.  There is no retry or timing-based choice.
    """

    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise ValueError("decode sample_index must be an integer")
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError("decode target batch must be a positive integer")

    work: list[tuple[Mapping[str, Any], dict[str, Any] | None]] = []
    previous_timestamp: float | None = None
    seen_cycle_ids: set[str] = set()
    for record_index, record in enumerate(scheduler_records):
        timestamp = record.get("monotonic_timestamp_s")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(float(timestamp))
        ):
            raise ValueError(
                f"decode scheduler row {record_index} lacks CLOCK_MONOTONIC time"
            )
        timestamp = float(timestamp)
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise ValueError("decode scheduler timestamps are not strictly increasing")
        previous_timestamp = timestamp
        shape = record.get("shape")
        if not isinstance(shape, Mapping):
            raise ValueError("decode scheduler row lacks an exact cycle shape")
        prefill = _strict_shape(
            shape.get("prefill"), label="decode-window prefill shape"
        )
        decode = _strict_shape(
            shape.get("decode"), label="decode-window decode shape"
        )
        if not any(prefill.values()) and not any(decode.values()):
            # Idle callbacks are intentionally retained in the wall clock.
            continue
        normalized = _pure_cycle(record, role="decode", batch=batch)
        if normalized is not None:
            before = _strict_backlog(
                record.get("backlog_before_interval"),
                label="decode-window backlog_before_interval",
            )
            after = _strict_backlog(
                record.get("backlog_after_cycle"),
                label="decode-window backlog_after_cycle",
            )
            saturated = {
                "running_sequences": int(batch),
                "waiting_sequences": 0,
            }
            if before != saturated or after != saturated:
                # Exact scheduled shape alone is insufficient: this excludes
                # a first-token NIXL admission cycle and any finish/refill
                # cycle from the post-transfer resident plateau.
                normalized = None
        cycle_id = record.get("cycle_id")
        if not isinstance(cycle_id, str) or not cycle_id:
            raise ValueError("decode scheduler work lacks a cycle identity")
        if cycle_id in seen_cycle_ids:
            raise ValueError("decode scheduler work repeats a cycle identity")
        seen_cycle_ids.add(cycle_id)
        work.append((record, normalized))

    if not work:
        raise ValueError("decode cohort contains no scheduler work")
    candidate: list[dict[str, Any]] = []
    selected_run: list[dict[str, Any]] | None = None
    longest_run = 0
    for _record, normalized in work:
        if normalized is None:
            candidate = []
            continue
        context = normalized["actual_shape"]["scheduled_context_tokens"]
        if candidate:
            previous_context = candidate[-1]["actual_shape"][
                "scheduled_context_tokens"
            ]
            if context == previous_context + int(batch):
                candidate.append(normalized)
            else:
                candidate = [normalized]
        else:
            candidate = [normalized]
        longest_run = max(longest_run, len(candidate))
        if len(candidate) == DECODE_CONSECUTIVE_CYCLES_PER_WINDOW:
            selected_run = list(candidate)
            break
    if selected_run is None:
        raise ValueError(
            "decode cohort has no consecutive saturated exact-B plateau: "
            f"required={DECODE_CONSECUTIVE_CYCLES_PER_WINDOW} pure B{batch} "
            f"work cycles with context step +{batch}; longest={longest_run}"
        )

    start_boundary = selected_run[0]
    selected_full = selected_run[1:]
    if len(selected_full) != MIN_DECODE_CYCLES_PER_WINDOW:
        raise AssertionError("decode plateau selector returned the wrong interval count")
    started = float(start_boundary["monotonic_timestamp_s"])
    completed = float(selected_full[-1]["monotonic_timestamp_s"])
    window = completed - started
    if not math.isfinite(window) or window <= 0.0:
        raise ValueError("decode scheduler context window is not positive")
    selected = [
        {
            "cycle_id": item["cycle_id"],
            "actual_shape": item["actual_shape"],
        }
        for item in selected_full
    ]
    return {
        "sample_index": int(sample_index),
        "window_source": (
            "scheduler_first_saturated_exact_batch_clock_monotonic"
        ),
        "window_started_monotonic_s": started,
        "window_completed_monotonic_s": completed,
        "role_window_s": window,
        "scheduler_cycle_count": len(selected),
        "observed_service_per_cycle_s": window / len(selected),
        "selected_cycles": selected,
        "window_proof": {
            "boundary_start_cycle_id": start_boundary["cycle_id"],
            "boundary_start_shape": start_boundary["actual_shape"],
            "boundary_end_cycle_id": selected[-1]["cycle_id"],
            "consecutive_exact_batch_cycle_count": (
                DECODE_CONSECUTIVE_CYCLES_PER_WINDOW
            ),
            "context_step_per_interval": int(batch),
        },
    }


def _role_proxy(port: int) -> SimpleNamespace:
    return SimpleNamespace(base_url=f"http://127.0.0.1:{int(port)}/v1")


def _nixl_metric_payload(payload: str) -> str:
    lines = [
        line
        for line in payload.splitlines()
        if line and not line.startswith("#") and "nixl_" in line.lower()
    ]
    return "\n".join(lines) + "\n"


async def _fetch_metrics(port: int, session: Any) -> str:
    import aiohttp

    async with session.get(
        f"http://127.0.0.1:{int(port)}/metrics",
        timeout=aiohttp.ClientTimeout(total=10.0),
    ) as response:
        payload = await response.text()
        if response.status != 200:
            raise RuntimeError(
                f"vLLM metrics on port {port} returned HTTP {response.status}"
            )
        return payload


def _load_component_workload(
    system_hardware: str,
) -> tuple[Path, Path, dict[str, Any], dict[str, Any], str]:
    component_path = _component_path(system_hardware).resolve()
    component_contract_path = Path(f"{component_path}.contract.json").resolve()
    document = json.loads(component_contract_path.read_text())
    if document.get("scope") != "component_only":
        raise ValueError("generator component contract has the wrong scope")
    if document.get("raw_telemetry_path") != str(component_path):
        raise ValueError("component contract points at a different raw artifact")
    if document.get("raw_telemetry_sha256") != _sha256_path(component_path):
        raise ValueError("component raw SHA-256 disagrees with its contract")
    workload = document.get("workload")
    if not isinstance(workload, dict):
        raise ValueError("component contract lacks workload")
    if (
        workload.get("prompt_tokens") != PROMPT_TOKENS
        or workload.get("output_tokens") != OUTPUT_TOKENS
        or workload.get("target_batches") != list(TARGET_BATCHES)
    ):
        raise ValueError("component workload is not the canonical P/D shape")
    contracts = workload.get("request_contracts")
    if not isinstance(contracts, dict):
        raise ValueError("component workload lacks request contracts")
    request_contract = validate_generator_request_contract(
        contracts.get(str(PROMPT_TOKENS)),
        expected_prompt_tokens=PROMPT_TOKENS,
    )
    subruns = document.get("subruns")
    if not isinstance(subruns, dict):
        raise ValueError("component contract lacks exact batch subruns")
    models = {
        value.get("component", {}).get("startup", {}).get("model")
        for value in subruns.values()
        if isinstance(value, dict)
    }
    if len(models) != 1 or not isinstance(next(iter(models)), str):
        raise ValueError("component campaign must use one reference model")
    return (
        component_path,
        component_contract_path,
        document,
        request_contract,
        next(iter(models)),
    )


def _system_hardware_from_config(config: str) -> tuple[str, dict[str, Any]]:
    from rag_stack.cost_model.llm_sim.calibration import hardware_from_config
    from rag_stack.cost_model.stage_batch_curve import validate_system_hardware

    hardware, system = hardware_from_config(config)
    return (
        validate_system_hardware(getattr(hardware, "system_hardware", None)),
        system,
    )


async def _prefill_request(
    pair: Any,
    *,
    payload: Mapping[str, Any],
    request_id: str,
) -> tuple[dict[str, Any], float]:
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
        REQUEST_FORMAT_CHAT_COMPLETIONS,
    )

    response = await pair._raw_post(
        pair.prefill_port,
        dict(payload),
        request_id,
        request_format=REQUEST_FORMAT_CHAT_COMPLETIONS,
    )
    completed = time.monotonic()
    handle = response.get("kv_transfer_params")
    if not isinstance(handle, dict) or not handle:
        raise RuntimeError(
            f"prefill returned no NIXL handle (request_id={request_id})"
        )
    return handle, completed


async def _consume_handle(
    pair: Any,
    *,
    decode_payload: Mapping[str, Any],
    handle: Mapping[str, Any],
    request_id: str,
    expected_completion_tokens: int,
) -> dict[str, Any]:
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
        REQUEST_FORMAT_CHAT_COMPLETIONS,
    )

    if (
        isinstance(expected_completion_tokens, bool)
        or not isinstance(expected_completion_tokens, int)
        or expected_completion_tokens < 1
    ):
        raise ValueError("expected completion tokens must be a positive integer")
    requested_completion_tokens = decode_payload.get("max_tokens")
    if (
        isinstance(requested_completion_tokens, bool)
        or not isinstance(requested_completion_tokens, int)
        or requested_completion_tokens != expected_completion_tokens
    ):
        raise ValueError(
            "decode payload max_tokens disagrees with its completion guard "
            f"(expected={expected_completion_tokens}, "
            f"payload={requested_completion_tokens!r})"
        )
    payload = dict(decode_payload)
    payload["kv_transfer_params"] = dict(handle)
    response = await pair._raw_post(
        pair.decode_port,
        payload,
        request_id,
        request_format=REQUEST_FORMAT_CHAT_COMPLETIONS,
    )
    if not isinstance(response, Mapping):
        raise RuntimeError(
            f"decode returned no JSON object (request_id={request_id})"
        )
    usage = response.get("usage")
    completion_tokens = (
        usage.get("completion_tokens") if isinstance(usage, Mapping) else None
    )
    if (
        isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens != expected_completion_tokens
    ):
        raise RuntimeError(
            "decode completion-token guard failed "
            f"(request_id={request_id}, expected={expected_completion_tokens}, "
            f"observed={completion_tokens!r})"
        )
    return dict(response)


async def _run_prefill_window(
    pair: Any,
    *,
    batch: int,
    payload: Mapping[str, Any],
    drain_payload: Mapping[str, Any],
    telemetry_path: Path,
    telemetry_run_id: str,
    sample_index: int,
    control_session: Any,
) -> dict[str, Any]:
    from rag_stack.cost_model.llm_sim.calibration.main_llm.vllm_probe import (
        _is_scheduler_paused,
        _pause_scheduler,
        _resume_scheduler,
        _wait_for_dispatch_fence,
        _wait_for_engine_idle,
    )

    proxy = _role_proxy(pair.prefill_port)
    await _wait_for_engine_idle(proxy, control_session)
    await _pause_scheduler(proxy, control_session)
    offset = telemetry_path.stat().st_size if telemetry_path.exists() else 0
    tasks = [
        asyncio.create_task(_prefill_request(
            pair,
            payload=payload,
            request_id=(
                f"pd-cal-p-b{batch}-s{sample_index}-r{request}-"
                f"{uuid.uuid4().hex}"
            ),
        ))
        for request in range(PREFILL_CYCLES_PER_WINDOW * int(batch))
    ]
    try:
        await _wait_for_dispatch_fence(
            telemetry_path,
            start_offset=offset,
            run_id=telemetry_run_id,
            expected=PREFILL_CYCLES_PER_WINDOW * int(batch),
            timeout_s=180.0,
        )
        started = time.monotonic()
        await _resume_scheduler(proxy, control_session)
        results = await asyncio.gather(*tasks)
        completed = max(item[1] for item in results)
    finally:
        if await _is_scheduler_paused(proxy, control_session):
            await _resume_scheduler(proxy, control_session)
        pending = [task for task in tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    # NIXL handles are lifecycle resources, not P timing.  Drain them only
    # after the parent window has closed, so consumer work cannot define the P
    # service value.
    await asyncio.gather(*(
        _consume_handle(
            pair,
            decode_payload=drain_payload,
            handle=handle,
            request_id=(
                f"pd-cal-p-drain-b{batch}-s{sample_index}-r{index}-"
                f"{uuid.uuid4().hex}"
            ),
            expected_completion_tokens=1,
        )
        for index, (handle, _finished) in enumerate(results)
    ))
    await _wait_for_engine_idle(proxy, control_session)
    records = _scheduler_records(telemetry_path, run_id=telemetry_run_id)
    return build_window_sample(
        sample_index=sample_index,
        role="prefill",
        batch=batch,
        started_monotonic_s=started,
        completed_monotonic_s=completed,
        scheduler_records=records,
        require_publishable=False,
    )


async def _produce_handles(
    pair: Any,
    *,
    batch: int,
    payload: Mapping[str, Any],
    cohort_index: int,
) -> list[dict[str, Any]]:
    results = await asyncio.gather(*(
        _prefill_request(
            pair,
            payload=payload,
            request_id=(
                f"pd-cal-d-handle-b{batch}-c{cohort_index}-r{request}-"
                f"{uuid.uuid4().hex}"
            ),
        )
        for request in range(int(batch))
    ))
    return [handle for handle, _completed in results]


async def _run_decode_window(
    pair: Any,
    *,
    batch: int,
    prefill_payload: Mapping[str, Any],
    decode_payload: Mapping[str, Any],
    telemetry_path: Path,
    telemetry_run_id: str,
    sample_index: int,
    cohort_index: int,
    warmup: bool,
    control_session: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    from rag_stack.cost_model.llm_sim.calibration.main_llm.vllm_probe import (
        _is_scheduler_paused,
        _pause_scheduler,
        _resume_scheduler,
        _wait_for_dispatch_fence,
        _wait_for_engine_idle,
    )

    decode_proxy = _role_proxy(pair.decode_port)
    handles = await _produce_handles(
        pair,
        batch=batch,
        payload=prefill_payload,
        cohort_index=cohort_index,
    )
    await _wait_for_engine_idle(decode_proxy, control_session)
    await _pause_scheduler(decode_proxy, control_session)
    before_full = await _fetch_metrics(pair.decode_port, control_session)
    offset = telemetry_path.stat().st_size if telemetry_path.exists() else 0
    tasks = []
    for request, handle in enumerate(handles):
        request_id = (
            f"pd-cal-d-b{batch}-c{cohort_index}-r{request}-"
            f"{uuid.uuid4().hex}"
        )
        tasks.append(asyncio.create_task(_consume_handle(
            pair,
            decode_payload=decode_payload,
            handle=handle,
            request_id=request_id,
            expected_completion_tokens=DECODE_ACQUISITION_OUTPUT_TOKENS,
        )))
    try:
        await _wait_for_dispatch_fence(
            telemetry_path,
            start_offset=offset,
            run_id=telemetry_run_id,
            expected=int(batch),
            timeout_s=180.0,
        )
        await _resume_scheduler(decode_proxy, control_session)
        await asyncio.gather(*tasks)
    finally:
        if await _is_scheduler_paused(decode_proxy, control_session):
            await _resume_scheduler(decode_proxy, control_session)
        pending = [task for task in tasks if not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    await _wait_for_engine_idle(decode_proxy, control_session)
    # Freeze the cumulative transfer counters only after the role reports
    # idle, so an asynchronous final NIXL notification cannot land after the
    # physical-evidence snapshot.
    after_full = await _fetch_metrics(pair.decode_port, control_session)

    records = _scheduler_records(
        telemetry_path,
        run_id=telemetry_run_id,
        start_offset=offset,
    )
    window_sample = build_decode_context_window_sample(
        sample_index=sample_index,
        batch=batch,
        scheduler_records=records,
    )
    # Warmup is subjected to the identical deterministic shape/window proof but
    # is structurally absent from the three fit samples.
    sample: dict[str, Any] | None = None if warmup else window_sample

    before = _nixl_metric_payload(before_full)
    after = _nixl_metric_payload(after_full)
    physical = derive_nixl_physical_evidence_from_text(
        before,
        after,
        expected_transfer_count=int(batch),
    ).to_dict()
    evidence_id = f"b{batch}-c{cohort_index}"
    nixl = {
        "evidence_id": evidence_id,
        "target_batch": int(batch),
        "cohort_index": int(cohort_index),
        "fit_eligible": not warmup,
        "selected_sample_index": (
            int(sample_index) if sample is not None else None
        ),
        "role_window": {
            "window_source": str(window_sample["window_source"]),
            "window_started_monotonic_s": float(
                window_sample["window_started_monotonic_s"]
            ),
            "window_completed_monotonic_s": float(
                window_sample["window_completed_monotonic_s"]
            ),
            "role_window_s": float(window_sample["role_window_s"]),
        },
        "selected_cycle_ids": (
            [] if warmup else [
                cycle["cycle_id"]
                for cycle in window_sample["selected_cycles"]
            ]
        ),
        "before_prometheus": before,
        "before_prometheus_sha256": hashlib.sha256(
            before.encode("utf-8")
        ).hexdigest(),
        "after_prometheus": after,
        "after_prometheus_sha256": hashlib.sha256(
            after.encode("utf-8")
        ).hexdigest(),
        "physical_evidence": physical,
        "cost_model_disposition": "physical_only_never_role_service_correction",
    }
    return sample, nixl


async def _measure_pair_batch(
    pair: Any,
    *,
    batch: int,
    request_contract: Mapping[str, Any],
    prefill_telemetry_path: Path,
    decode_telemetry_path: Path,
    prefill_run_id: str,
    decode_run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    import aiohttp
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_pd_pair import (
        _empty_remote_params,
    )

    prefill_payload = generator_chat_payload(
        request_contract,
        model=pair.key.model,
        output_tokens=1,
        served_max_model_len=SERVED_MAX_MODEL_LEN,
    )
    prefill_payload["kv_transfer_params"] = _empty_remote_params()
    drain_payload = generator_chat_payload(
        request_contract,
        model=pair.key.model,
        output_tokens=1,
        served_max_model_len=SERVED_MAX_MODEL_LEN,
    )
    decode_payload = generator_chat_payload(
        request_contract,
        model=pair.key.model,
        # Acquisition-only horizon: the production/component shape remains
        # OUTPUT_TOKENS=128 in the frozen workload contract below.
        output_tokens=DECODE_ACQUISITION_OUTPUT_TOKENS,
        served_max_model_len=SERVED_MAX_MODEL_LEN,
    )
    timeout = aiohttp.ClientTimeout(total=1800.0)
    connector = aiohttp.TCPConnector(limit=0, keepalive_timeout=4.0)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector
    ) as control_session:
        p_warmup = await _run_prefill_window(
            pair,
            batch=batch,
            payload=prefill_payload,
            drain_payload=drain_payload,
            telemetry_path=prefill_telemetry_path,
            telemetry_run_id=prefill_run_id,
            sample_index=-1,
            control_session=control_session,
        )
        p_samples = []
        for sample_index in range(MEASURED_SAMPLE_COUNT):
            p_samples.append(await _run_prefill_window(
                pair,
                batch=batch,
                payload=prefill_payload,
                drain_payload=drain_payload,
                telemetry_path=prefill_telemetry_path,
                telemetry_run_id=prefill_run_id,
                sample_index=sample_index,
                control_session=control_session,
            ))

        d_samples = []
        nixl_entries = []
        _warmup_sample, warmup_nixl = await _run_decode_window(
            pair,
            batch=batch,
            prefill_payload=prefill_payload,
            decode_payload=decode_payload,
            telemetry_path=decode_telemetry_path,
            telemetry_run_id=decode_run_id,
            sample_index=-1,
            cohort_index=0,
            warmup=True,
            control_session=control_session,
        )
        nixl_entries.append(warmup_nixl)
        for cohort_index in range(1, MEASURED_SAMPLE_COUNT + 1):
            sample, nixl = await _run_decode_window(
                pair,
                batch=batch,
                prefill_payload=prefill_payload,
                decode_payload=decode_payload,
                telemetry_path=decode_telemetry_path,
                telemetry_run_id=decode_run_id,
                sample_index=cohort_index - 1,
                cohort_index=cohort_index,
                warmup=False,
                control_session=control_session,
            )
            nixl_entries.append(nixl)
            if sample is None:
                raise RuntimeError("measured decode cohort produced no sample")
            d_samples.append(sample)
    await pair.aclose_http()
    return (
        {"warmup": p_warmup, "samples": p_samples},
        {"warmup": None, "samples": d_samples},
        nixl_entries,
    )


def _role_contract_document(
    *,
    system_hardware: str,
    role: str,
    source_path: Path,
    component_source_path: Path,
    component_contract_path: Path,
    nixl_path: Path | None,
    request_contract: Mapping[str, Any],
    batches: Mapping[str, Mapping[str, Any]],
    component_joins: Mapping[str, Any],
) -> dict[str, Any]:
    from .generator_disagg_evidence import (
        REQUIRED_BATCHES,
        ROLE_AGGREGATION_CONTRACT,
        ROLE_RUN_CONTRACT_SCHEMA,
        SAMPLES_PER_BATCH,
        _selection_contract,
    )
    from .generator_request_contract import generator_request_contract_sha256

    selection = _selection_contract(role)
    return {
        "schema": ROLE_RUN_CONTRACT_SCHEMA,
        "system_hardware": system_hardware,
        "role": role,
        "source_path": str(source_path.resolve()),
        "source_sha256": _sha256_path(source_path),
        "component_source_path": str(component_source_path.resolve()),
        "component_source_sha256": _sha256_path(component_source_path),
        "component_contract_path": str(component_contract_path.resolve()),
        "component_contract_sha256": _sha256_path(component_contract_path),
        "nixl_physical_path": (
            None if nixl_path is None else str(nixl_path.resolve())
        ),
        "nixl_physical_sha256": (
            None if nixl_path is None else _sha256_path(nixl_path)
        ),
        "request_contract_sha256": generator_request_contract_sha256(
            request_contract
        ),
        "runs": {
            str(batch): {
                field: batches[str(batch)][field]
                for field in (
                    "telemetry_path",
                    "telemetry_sha256",
                    "telemetry_run_id",
                    "timing_run_id",
                )
            }
            for batch in REQUIRED_BATCHES
        },
        "workload": {
            "target_batches": list(REQUIRED_BATCHES),
            "samples_per_batch": SAMPLES_PER_BATCH,
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "request_contract": dict(request_contract),
        },
        "cycle_selection": selection,
        "aggregation": dict(ROLE_AGGREGATION_CONTRACT),
        "selected_cycle_ids": {
            str(batch): [
                [
                    cycle["cycle_id"]
                    for cycle in sample["selected_cycles"]
                ]
                for sample in batches[str(batch)]["samples"]
            ]
            for batch in REQUIRED_BATCHES
        },
        "component_joins": dict(component_joins),
    }


def _measure_campaign(config: str, *, force: bool) -> None:
    """Launch one exact-cap TP1/PP1 pair per sparse resident batch."""

    from rag_stack.cost_model.llm_sim.calibration.main_llm.generator_parallel import (
        _ensure_stage_nofile_limit,
    )
    from rag_stack.cost_model.llm_sim.calibration.main_llm.vllm_probe import (
        _available_gpus,
    )
    from rag_stack.performance_context import system_design_space
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_pd_pair import (
        VllmPdCalibrationTelemetry,
        VllmPdPair,
        VllmPdPairKey,
        VllmPdRoleTelemetry,
    )
    from .generator_disagg_evidence import (
        REQUIRED_BATCHES,
        ROLE_TIMING_SCHEMA,
        read_direct_role_evidence,
    )

    system_hardware, system = _system_hardware_from_config(config)
    (
        component_source_path,
        component_contract_path,
        _component_contract,
        request_contract,
        model,
    ) = _load_component_workload(system_hardware)
    role_paths = {
        role: _timing_path(system_hardware, role).resolve()
        for role in ("prefill", "decode")
    }
    contract_paths = {
        role: _contract_path(system_hardware, role).resolve()
        for role in ("prefill", "decode")
    }
    nixl_path = _nixl_path(system_hardware).resolve()
    completed_artifacts = (
        *role_paths.values(),
        *contract_paths.values(),
        nixl_path,
    )
    if all(path.exists() for path in completed_artifacts) and not force:
        # Frozen artifacts are still parsed before treating them as complete.
        for role in ("prefill", "decode"):
            read_direct_role_evidence(role_paths[role], contract_paths[role])
        print(
            "[generator-disagg] complete frozen direct P/D evidence exists; "
            "use --force to remeasure",
            flush=True,
        )
        return
    if any(path.exists() for path in completed_artifacts) and not force:
        raise FileExistsError(
            "incomplete direct P/D evidence exists; refuse to mix generations "
            "without --force"
        )

    gpus = [str(value) for value in _available_gpus(system)]
    if len(gpus) < 2:
        raise ValueError(
            "direct P/D calibration requires two declared GPUs for independent "
            "TP1/PP1 engines"
        )
    design = system_design_space(system)
    gpu_utilization = float(design.get("vllm_gpu_memory_utilization", 0.85))
    kv_cache_dtype = str(design.get("vllm_kv_cache_dtype", "auto"))
    dtype = str(design.get("vllm_dtype", "bfloat16"))
    _ensure_stage_nofile_limit()
    role_batches: dict[str, dict[str, Any]] = {
        "prefill": {},
        "decode": {},
    }
    nixl_entries: list[dict[str, Any]] = []
    campaign_id = uuid.uuid4().hex
    telemetry_paths_created: list[Path] = []

    print(
        f"[generator-disagg] hardware={system_hardware} reference={model} "
        f"P={gpus[0]} D={gpus[1]} "
        f"acquisition_batches={list(ACQUISITION_BATCH_ORDER)}; "
        f"production/component_output={OUTPUT_TOKENS}, "
        f"D_acquisition_guard={DECODE_ACQUISITION_OUTPUT_TOKENS}; "
        "warmup+3 direct windows per role; D window=first saturated "
        "exact-B boundary + 32 interdeparture intervals",
        flush=True,
    )
    for batch in ACQUISITION_BATCH_ORDER:
        p_path = _telemetry_path(
            system_hardware, "prefill", batch, campaign_id
        ).resolve()
        d_path = _telemetry_path(
            system_hardware, "decode", batch, campaign_id
        ).resolve()
        p_path.parent.mkdir(parents=True, exist_ok=True)
        if p_path.exists() or d_path.exists():
            raise FileExistsError("refusing to append direct P/D telemetry")
        p_run_id = f"pd-prefill-b{batch}-{uuid.uuid4().hex}"
        d_run_id = f"pd-decode-b{batch}-{uuid.uuid4().hex}"
        telemetry = VllmPdCalibrationTelemetry(
            prefill=VllmPdRoleTelemetry(str(p_path), p_run_id),
            decode=VllmPdRoleTelemetry(str(d_path), d_run_id),
        )
        pair = None
        try:
            pair = VllmPdPair(
                VllmPdPairKey(
                    model=model,
                    prefill_device=gpus[0],
                    decode_device=gpus[1],
                    max_num_seqs=int(batch),
                    prefill_max_num_seqs=int(batch),
                    decode_max_num_seqs=int(batch),
                    max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
                    enable_prefix_caching=False,
                    max_model_len=SERVED_MAX_MODEL_LEN,
                    kv_cache_dtype=kv_cache_dtype,
                    gpu_memory_utilization=gpu_utilization,
                    prefill_tensor_parallel_size=1,
                    prefill_pipeline_parallel_size=1,
                    decode_tensor_parallel_size=1,
                    decode_pipeline_parallel_size=1,
                    dtype=dtype,
                ),
                calibration_telemetry=telemetry,
            )
            prefill_result, decode_result, batch_nixl = asyncio.run(
                _measure_pair_batch(
                    pair,
                    batch=batch,
                    request_contract=request_contract,
                    prefill_telemetry_path=p_path,
                    decode_telemetry_path=d_path,
                    prefill_run_id=p_run_id,
                    decode_run_id=d_run_id,
                )
            )
        finally:
            if pair is not None:
                pair.shutdown()
        if not p_path.exists() or not d_path.exists():
            raise RuntimeError(f"B{batch} did not freeze both role telemetry logs")
        telemetry_paths_created.extend((p_path, d_path))
        for role, result, telemetry_path, telemetry_run_id in (
            ("prefill", prefill_result, p_path, p_run_id),
            ("decode", decode_result, d_path, d_run_id),
        ):
            role_batches[role][str(batch)] = {
                "target_batch": int(batch),
                "engine_max_num_seqs": int(batch),
                "telemetry_path": str(telemetry_path),
                "telemetry_sha256": _sha256_path(telemetry_path),
                "telemetry_run_id": telemetry_run_id,
                "timing_run_id": (
                    f"pd-{role}-timing-b{batch}-{uuid.uuid4().hex}"
                ),
                # The independent warmup was validated above but is omitted
                # structurally so fit code cannot accidentally select it.
                "samples": result["samples"],
            }
        nixl_entries.extend(batch_nixl)
        print(
            f"[generator-disagg] B{batch} complete: "
            f"P={[round(x['observed_service_per_cycle_s']*1e3, 3) for x in prefill_result['samples']]} ms, "
            f"D={[round(x['observed_service_per_cycle_s']*1e3, 3) for x in decode_result['samples']]} "
            "ms/cycle (first saturated exact-B 32-interval window)",
            flush=True,
        )

    atomic_write_json(nixl_path, {
        "schema": NIXL_CAMPAIGN_SCHEMA,
        "system_hardware": system_hardware,
        "evidence_scope": "physical_only_no_role_service_fit",
        "entries": sorted(
            nixl_entries,
            key=lambda item: (
                TARGET_BATCHES.index(int(item["target_batch"])),
                int(item["cohort_index"]),
            ),
        ),
    })
    timing_documents = {}
    for role in ("prefill", "decode"):
        timing_documents[role] = {
            "schema": ROLE_TIMING_SCHEMA,
            "system_hardware": system_hardware,
            "role": role,
            "batches": {
                str(batch): role_batches[role][str(batch)]
                for batch in REQUIRED_BATCHES
            },
        }
        atomic_write_json(role_paths[role], timing_documents[role])
    from .generator_disagg_evidence import build_component_joins

    for role in ("prefill", "decode"):
        component_joins = build_component_joins(
            timing_documents[role],
            component_source_path,
            component_contract_path,
        )
        contract = _role_contract_document(
            system_hardware=system_hardware,
            role=role,
            source_path=role_paths[role],
            component_source_path=component_source_path,
            component_contract_path=component_contract_path,
            nixl_path=(None if role == "prefill" else nixl_path),
            request_contract=request_contract,
            batches=role_batches[role],
            component_joins=component_joins,
        )
        atomic_write_json(contract_paths[role], contract)

    # CPU-parse both documents before announcing a frozen generation.
    for role in ("prefill", "decode"):
        read_direct_role_evidence(role_paths[role], contract_paths[role])
    print(
        f"[generator-disagg] froze direct P/D timing + contracts at "
        f"{role_paths['prefill']} / {role_paths['decode']}; NIXL physical "
        f"evidence={nixl_path}",
        flush=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rag_stack calibrate generator-disagg"
    )
    parser.add_argument("verb", choices=("measure", "fit"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.verb == "measure":
        measure(args.config, force=args.force)
    else:
        fit(args.config)


def measure(config: str, *, force: bool = False) -> None:
    """Acquire a complete direct P/D campaign (GPU entrypoint)."""

    # Implemented below the pure proof helpers so importing this module for
    # CPU fit/tests never imports vLLM or initializes CUDA.
    _measure_campaign(config, force=force)


def fit(config: str) -> None:
    """Build role curves and atomically publish one complete six-profile set."""

    from .generator_disagg_evidence import fit_and_publish_direct_disagg

    fit_and_publish_direct_disagg(config)


if __name__ == "__main__":
    main()
