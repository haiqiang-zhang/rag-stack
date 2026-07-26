"""CPU-only acquisition contracts for direct P/D calibration."""

from __future__ import annotations

import asyncio

import pytest

from rag_stack.cost_model.llm_sim.calibration.main_llm.generator_disagg import (
    ACQUISITION_BATCH_ORDER,
    DECODE_ACQUISITION_OUTPUT_TOKENS,
    DECODE_CONSECUTIVE_CYCLES_PER_WINDOW,
    MIN_DECODE_CYCLES_PER_WINDOW,
    OUTPUT_TOKENS,
    TARGET_BATCHES,
    _consume_handle,
    build_decode_context_window_sample,
    build_window_sample,
    select_window_cycles,
)


def test_large_first_acquisition_order_covers_canonical_batches_once():
    assert ACQUISITION_BATCH_ORDER == (256, 128, 64, 32, 8, 1)
    assert len(ACQUISITION_BATCH_ORDER) == len(TARGET_BATCHES)
    assert set(ACQUISITION_BATCH_ORDER) == set(TARGET_BATCHES)


def test_decode_acquisition_guard_does_not_change_component_workload_shape():
    assert OUTPUT_TOKENS == 128
    assert DECODE_ACQUISITION_OUTPUT_TOKENS == 256
    assert DECODE_CONSECUTIVE_CYCLES_PER_WINDOW == 33
    assert MIN_DECODE_CYCLES_PER_WINDOW == 32


def _record(
    *,
    run: str,
    sequence: int,
    timestamp: float,
    role: str,
    batch: int,
    context: int,
    publishable: bool = True,
    inactive: bool = False,
    active_service_s: float = 999.0,
    running_before: int | None = None,
    waiting_before: int = 0,
    running_after: int | None = None,
    waiting_after: int = 0,
):
    if role == "prefill":
        active = {
            "scheduled_sequences": batch,
            "scheduled_tokens": batch * 112,
            "scheduled_context_tokens": batch * 112,
            "token_context_product": batch * 112 * 112,
        }
    else:
        active = {
            "scheduled_sequences": batch,
            "scheduled_tokens": batch,
            "scheduled_context_tokens": context,
            "token_context_product": context,
        }
    zero = {
        "scheduled_sequences": 0,
        "scheduled_tokens": 0,
        "scheduled_context_tokens": 0,
        "token_context_product": 0,
    }
    if inactive:
        zero = dict(zero, scheduled_sequences=1, scheduled_tokens=1)
    other = "decode" if role == "prefill" else "prefill"
    return {
        "schema": "rag_stack.vllm.scheduler_cycle_stage_service",
        "run_id": run,
        "cycle_id": f"{run}:{sequence}:shape",
        "monotonic_timestamp_s": timestamp,
        "publishable": publishable,
        # Deliberately absurd: acquisition must never consume this as service.
        "active_service_s": active_service_s,
        "backlog_before_interval": {
            "running_sequences": batch if running_before is None else running_before,
            "waiting_sequences": waiting_before,
        },
        "backlog_after_cycle": {
            "running_sequences": batch if running_after is None else running_after,
            "waiting_sequences": waiting_after,
        },
        "shape": {
            "terminal_performance_stage": role,
            role: active,
            other: zero,
        },
    }


def _idle_record(*, run: str, sequence: int, timestamp: float):
    zero = {
        "scheduled_sequences": 0,
        "scheduled_tokens": 0,
        "scheduled_context_tokens": 0,
        "token_context_product": 0,
    }
    return {
        "schema": "rag_stack.vllm.scheduler_cycle_stage_service",
        "run_id": run,
        "cycle_id": None,
        "monotonic_timestamp_s": timestamp,
        "publishable": False,
        "active_service_s": None,
        "reason": "current_cycle_has_no_scheduled_work",
        "shape": {
            "terminal_performance_stage": None,
            "prefill": dict(zero),
            "decode": dict(zero),
        },
    }


def test_prefill_frontier_uses_wall_divided_by_four_not_telemetry_service():
    records = [
        _record(
            run="p-run",
            sequence=index,
            timestamp=10.2 + index * 0.2,
            role="prefill",
            batch=8,
            context=8 * 112,
            active_service_s=10_000.0 + index,
        )
        for index in range(4)
    ]
    sample = build_window_sample(
        sample_index=0,
        role="prefill",
        batch=8,
        started_monotonic_s=10.0,
        completed_monotonic_s=11.0,
        scheduler_records=records,
    )

    assert sample["scheduler_cycle_count"] == 4
    assert sample["observed_service_per_cycle_s"] == pytest.approx(0.25)
    assert all(
        set(cycle) == {"cycle_id", "actual_shape"}
        for cycle in sample["selected_cycles"]
    )
    assert "active_service_s" not in repr(sample)


def test_idle_scheduler_callbacks_stay_in_wall_but_not_cycle_divisor():
    records = []
    for index in range(4):
        records.append(_record(
            run="p-run",
            sequence=index,
            timestamp=10.1 + index * 0.2,
            role="prefill",
            batch=8,
            context=8 * 112,
        ))
        if index < 3:
            records.append(_idle_record(
                run="p-run",
                sequence=100 + index,
                timestamp=10.2 + index * 0.2,
            ))
    sample = build_window_sample(
        sample_index=0,
        role="prefill",
        batch=8,
        started_monotonic_s=10.0,
        completed_monotonic_s=11.0,
        scheduler_records=records,
    )
    assert sample["scheduler_cycle_count"] == 4
    assert sample["observed_service_per_cycle_s"] == pytest.approx(0.25)


def test_prefill_window_requires_every_one_of_four_exact_pure_cycles():
    valid = [
        _record(
            run="p-run",
            sequence=index,
            timestamp=1.1 + index * 0.1,
            role="prefill",
            batch=8,
            context=8 * 112,
        )
        for index in range(4)
    ]
    with pytest.raises(ValueError, match="exactly 4"):
        select_window_cycles(
            valid[:3],
            role="prefill",
            batch=8,
            started_monotonic_s=1.0,
            completed_monotonic_s=2.0,
        )

    contaminated = list(valid)
    contaminated[2] = _record(
        run="p-run",
        sequence=2,
        timestamp=1.3,
        role="prefill",
        batch=8,
        context=8 * 112,
        inactive=True,
    )
    with pytest.raises(ValueError, match="non-pure"):
        select_window_cycles(
            contaminated,
            role="prefill",
            batch=8,
            started_monotonic_s=1.0,
            completed_monotonic_s=2.0,
        )


def test_decode_window_accepts_staggered_aggregate_context_and_counts_cycles():
    # NIXL readiness may make aggregate context non-divisible by B.  It is
    # still an exact B scheduler cycle and must retain its real shape for the
    # strict component bracketing interpolation done by the CPU fitter.
    contexts = [1_155 + index for index in range(40)]
    records = [
        _record(
            run="d-run",
            sequence=index,
            timestamp=20.01 + index * 0.01,
            role="decode",
            batch=8,
            context=context,
        )
        for index, context in enumerate(contexts)
    ]
    sample = build_window_sample(
        sample_index=1,
        role="decode",
        batch=8,
        started_monotonic_s=20.0,
        completed_monotonic_s=20.5,
        scheduler_records=records,
    )

    assert sample["scheduler_cycle_count"] == 40
    assert sample["observed_service_per_cycle_s"] == pytest.approx(0.0125)
    assert sample["selected_cycles"][0]["actual_shape"][
        "scheduled_context_tokens"
    ] == 1_155


def test_decode_window_rejects_partial_batch_work_inside_parent_frontiers():
    records = [
        _record(
            run="d-run",
            sequence=index,
            timestamp=3.01 + index * 0.01,
            role="decode",
            batch=(7 if index == 12 else 8),
            context=1_200 + index,
        )
        for index in range(40)
    ]
    with pytest.raises(ValueError, match="non-pure/non-B8"):
        build_window_sample(
            sample_index=0,
            role="decode",
            batch=8,
            started_monotonic_s=3.0,
            completed_monotonic_s=3.5,
            scheduler_records=records,
        )


def test_callback_publishability_never_filters_parent_clock_shape_proof():
    records = [
        _record(
            run="p-run",
            sequence=index,
            timestamp=5.1 + index * 0.1,
            role="prefill",
            batch=1,
            context=112,
            publishable=index != 0,
        )
        for index in range(4)
    ]
    sample = build_window_sample(
        sample_index=0,
        role="prefill",
        batch=1,
        started_monotonic_s=5.0,
        completed_monotonic_s=5.6,
        scheduler_records=records,
    )
    assert sample["scheduler_cycle_count"] == 4


def test_decode_first_saturated_plateau_uses_32_interdeparture_intervals():
    batch = 8
    boundary_context = 1_336
    records = [
        _record(
            run="d-run",
            sequence=index,
            timestamp=30.0 + index * 0.004,
            role="decode",
            batch=batch,
            context=boundary_context + index * batch,
            publishable=index % 3 != 0,
            active_service_s=10_000.0 + index,
        )
        for index in range(DECODE_CONSECUTIVE_CYCLES_PER_WINDOW + 1)
    ]
    records.insert(20, _idle_record(
        run="d-run", sequence=500, timestamp=30.0 + 19.5 * 0.004,
    ))
    records.sort(key=lambda record: record["monotonic_timestamp_s"])

    sample = build_decode_context_window_sample(
        sample_index=0,
        batch=batch,
        scheduler_records=records,
    )

    assert sample["window_source"] == (
        "scheduler_first_saturated_exact_batch_clock_monotonic"
    )
    assert sample["scheduler_cycle_count"] == 32
    assert sample["window_proof"]["boundary_start_shape"][
        "scheduled_context_tokens"
    ] == boundary_context
    assert sample["window_proof"][
        "consecutive_exact_batch_cycle_count"
    ] == 33
    assert sample["window_proof"]["context_step_per_interval"] == batch
    assert sample["selected_cycles"][0]["actual_shape"][
        "scheduled_context_tokens"
    ] == boundary_context + batch
    assert sample["selected_cycles"][-1]["actual_shape"][
        "scheduled_context_tokens"
    ] == boundary_context + 32 * batch
    assert sample["observed_service_per_cycle_s"] == pytest.approx(0.004)
    assert "active_service_s" not in repr(sample)


def test_decode_saturated_plateau_rejects_partial_work_before_33_cycles():
    batch = 8
    records = [
        _record(
            run="d-run",
            sequence=index,
            timestamp=40.0 + index * 0.004,
            role="decode",
            batch=(7 if index == 16 else batch),
            context=1_200 + index * batch,
        )
        for index in range(DECODE_CONSECUTIVE_CYCLES_PER_WINDOW)
    ]
    with pytest.raises(ValueError, match="no consecutive saturated exact-B"):
        build_decode_context_window_sample(
            sample_index=0,
            batch=batch,
            scheduler_records=records,
        )


def test_decode_saturated_plateau_requires_exact_context_step_per_interval():
    batch = 8
    contexts = [1_200 + index * batch for index in range(33)]
    contexts[20] += 1
    records = [
        _record(
            run="d-run",
            sequence=index,
            timestamp=50.0 + index * 0.004,
            role="decode",
            batch=batch,
            context=context,
        )
        for index, context in enumerate(contexts)
    ]
    with pytest.raises(ValueError, match=r"context step \+8"):
        build_decode_context_window_sample(
            sample_index=0,
            batch=batch,
            scheduler_records=records,
        )


@pytest.mark.parametrize(
    "transition",
    (
        {"waiting_before": 1},
        {"waiting_after": 1},
        {"running_before": 7},
        {"running_after": 7},
    ),
)
def test_decode_saturated_plateau_rejects_admission_or_finish_transition(
    transition,
):
    batch = 8
    records = [
        _record(
            run="d-run",
            sequence=index,
            timestamp=60.0 + index * 0.004,
            role="decode",
            batch=batch,
            context=1_200 + index * batch,
            **(transition if index == 16 else {}),
        )
        for index in range(DECODE_CONSECUTIVE_CYCLES_PER_WINDOW)
    ]
    with pytest.raises(ValueError, match="no consecutive saturated exact-B"):
        build_decode_context_window_sample(
            sample_index=0,
            batch=batch,
            scheduler_records=records,
        )


class _FakeDecodePair:
    decode_port = 9001

    def __init__(self, response):
        self.response = response
        self.calls = []

    async def _raw_post(self, port, payload, request_id, *, request_format):
        self.calls.append((port, payload, request_id, request_format))
        return self.response


@pytest.mark.parametrize("expected_completion_tokens", (1, 256))
def test_consume_handle_returns_json_and_enforces_exact_usage(
    expected_completion_tokens,
):
    response = {
        "id": "chatcmpl-test",
        "usage": {"completion_tokens": expected_completion_tokens},
    }
    pair = _FakeDecodePair(response)
    observed = asyncio.run(_consume_handle(
        pair,
        decode_payload={"max_tokens": expected_completion_tokens},
        handle={"remote_block_ids": [1]},
        request_id="request-1",
        expected_completion_tokens=expected_completion_tokens,
    ))

    assert observed == response
    assert pair.calls[0][0] == pair.decode_port
    assert pair.calls[0][1]["kv_transfer_params"] == {
        "remote_block_ids": [1]
    }


@pytest.mark.parametrize(
    "response",
    (
        {},
        {"usage": {}},
        {"usage": {"completion_tokens": True}},
        {"usage": {"completion_tokens": 255}},
    ),
)
def test_consume_handle_fails_closed_on_missing_or_wrong_completion_usage(
    response,
):
    pair = _FakeDecodePair(response)
    with pytest.raises(RuntimeError, match="completion-token guard failed"):
        asyncio.run(_consume_handle(
            pair,
            decode_payload={"max_tokens": 256},
            handle={"remote_block_ids": [1]},
            request_id="request-1",
            expected_completion_tokens=256,
        ))


def test_consume_handle_rejects_payload_guard_disagreement_before_dispatch():
    pair = _FakeDecodePair({"usage": {"completion_tokens": 256}})
    with pytest.raises(ValueError, match="max_tokens disagrees"):
        asyncio.run(_consume_handle(
            pair,
            decode_payload={"max_tokens": 128},
            handle={"remote_block_ids": [1]},
            request_id="request-1",
            expected_completion_tokens=256,
        ))
    assert pair.calls == []
