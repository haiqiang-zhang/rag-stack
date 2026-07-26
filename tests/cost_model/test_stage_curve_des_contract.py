"""CPU-only contracts for scheduler-cycle non-negative stage overhead."""

from __future__ import annotations

import pytest

from rag_stack.cost_model.closed_loop_sim import (
    SimCall,
    SimDecodeItem,
    SimEngine,
    SimPrefillItem,
    _Engine,
    build_sim_inputs,
    _engine_cycle_profile_stage,
)
from rag_stack.cost_model.engines import CallLatency
from rag_stack.types import SystemConfig


def _hooks(calls):
    def corrected(
        resident_batch: int,
        component_service_s: float,
    ) -> float:
        calls.append((resident_batch, component_service_s))
        return component_service_s + 10.0 + resident_batch

    def prefill(_items) -> float:
        return 2.0

    def decode(_items) -> float:
        return 3.0

    prefill._rag_cm_cycle_corrected_service_s = corrected
    decode._rag_cm_cycle_corrected_service_s = corrected
    return prefill, decode


def _prefill_frontier_hook(calls):
    def corrected(
        resident_batch: int,
        completed_prefills: int,
        component_service_s: float,
    ) -> float:
        calls.append((
            resident_batch,
            completed_prefills,
            component_service_s,
        ))
        frontier_overhead = 10.0 + resident_batch
        return component_service_s + frontier_overhead * (
            completed_prefills / resident_batch
        )

    def prefill(_items) -> float:
        return 2.0

    prefill._rag_cm_prefill_frontier_corrected_service_s = corrected
    return prefill


def test_mixed_cycle_has_one_engine_overhead_at_resident_batch() -> None:
    calls = []
    prefill, decode = _hooks(calls)
    engine = _Engine(
        SimEngine(
            name="generator.unified",
            cards=(),
            depth=8,
            step_s=lambda _decode, _tokens: 99.0,
            prefill_batch_s=prefill,
            decode_step_s=decode,
        ),
        [],
    )
    for rid in range(5):
        assert not engine.admit(
            rid,
            SimCall(engine=0, eff_in=2, full_in=2, out=1),
        )

    service = engine._step_s(
        (
            SimDecodeItem(8, 0, 4),
            SimDecodeItem(9, 1, 3),
        ),
        (
            SimPrefillItem(0, 2),
            SimPrefillItem(0, 2),
            SimPrefillItem(0, 2),
        ),
    )

    assert service == pytest.approx(3.0 + 2.0 + 15.0)
    assert calls == [(5, 5.0)]


def test_direct_prefill_frontier_uses_residents_and_completed_logical_calls() -> None:
    calls = []
    prefill = _prefill_frontier_hook(calls)
    engine = _Engine(
        SimEngine(
            name="generator.prefill",
            cards=(),
            depth=2,
            budget=1,
            role="prefill",
            step_s=lambda _decode, _tokens: 99.0,
            prefill_batch_s=prefill,
        ),
        [],
    )
    for rid in range(2):
        assert not engine.admit(
            rid,
            SimCall(engine=0, eff_in=2, full_in=2, out=1),
        )

    intermediate = engine.build_step()

    assert intermediate is not None
    assert intermediate.decode_count == 0
    assert len(intermediate.prefill_work) == 1
    # One 8192-token-style intermediate chunk completes no logical P call, so
    # the completion-frontier residual is not repeated on this scheduler step.
    assert intermediate.service_s == pytest.approx(2.0)
    assert calls == [(2, 0, 2.0)]
    assert engine.complete_step(intermediate) == []

    completion = engine.build_step()

    assert completion is not None
    # One of two residents crosses its logical completion frontier: exactly
    # half of the calibrated B2 cohort overhead is charged.
    assert completion.service_s == pytest.approx(2.0 + 12.0 / 2.0)
    assert calls[-1] == (2, 1, 2.0)


def test_lazy_mixed_cycle_does_not_reintroduce_double_overhead() -> None:
    calls = []
    prefill, decode = _hooks(calls)

    def aggregate(batch: int, _input_sum: float, _generated_sum: float) -> float:
        assert batch == 1
        return 3.0

    decode._rag_cm_decode_aggregate_s = aggregate
    decode._rag_cm_decode_lazy_safe = True
    engine = _Engine(
        SimEngine(
            name="generator.unified",
            cards=(),
            depth=4,
            budget=8,
            role="unified",
            step_s=lambda _decode, _tokens: 99.0,
            prefill_batch_s=prefill,
            decode_step_s=decode,
        ),
        [],
    )
    assert not engine.admit(
        1, SimCall(engine=0, eff_in=0, full_in=8, out=2),
    )
    assert not engine.admit(
        2, SimCall(engine=0, eff_in=2, full_in=2, out=1),
    )

    plan = engine.build_step()

    assert plan is not None
    assert plan.service_s == pytest.approx(3.0 + 2.0 + 12.0)
    assert calls == [(2, 5.0)]


def test_unified_pure_prefill_and_decode_use_the_same_engine_curve() -> None:
    calls = []
    prefill, decode = _hooks(calls)
    engine = _Engine(
        SimEngine(
            name="generator.unified",
            cards=(),
            depth=8,
            role="unified",
            step_s=lambda _decode, _tokens: 99.0,
            prefill_batch_s=prefill,
            decode_step_s=decode,
        ),
        [],
    )
    assert not engine.admit(
        1, SimCall(engine=0, eff_in=2, full_in=2, out=1),
    )

    prefill_service = engine._step_s((), (SimPrefillItem(0, 2),))
    decode_engine = _Engine(engine.spec, [])
    assert not decode_engine.admit(
        2, SimCall(engine=0, eff_in=0, full_in=8, out=4),
    )
    decode_service = decode_engine._step_s((SimDecodeItem(8, 0, 4),), ())

    assert prefill_service == pytest.approx(2.0 + 11.0)
    assert decode_service == pytest.approx(3.0 + 11.0)
    assert calls == [(1, 2.0), (1, 3.0)]


def test_profile_stage_resolution_never_reuses_collocated_for_disagg() -> None:
    kwargs = {
        "prefill_stage": "generator_prefill",
        "decode_stage": "generator_decode",
    }
    assert _engine_cycle_profile_stage(
        "generator", "unified", **kwargs,
    ) == "generator"
    assert _engine_cycle_profile_stage(
        "generator", "prefill", **kwargs,
    ) == "generator_prefill"
    assert _engine_cycle_profile_stage(
        "generator", "decode", **kwargs,
    ) == "generator_decode"


def test_des_rejects_nonpositive_corrected_cycle_service() -> None:
    def prefill(_items) -> float:
        return 2.0

    def invalid_correction(
        _resident_batch: int,
        _component_service_s: float,
    ) -> float:
        return -1.0

    prefill._rag_cm_cycle_corrected_service_s = invalid_correction
    engine = _Engine(
        SimEngine(
            name="generator.unified",
            cards=(),
            depth=4,
            budget=8,
            role="unified",
            step_s=lambda _decode, _tokens: 99.0,
            prefill_batch_s=prefill,
        ),
        [],
    )
    assert not engine.admit(
        1,
        SimCall(engine=0, eff_in=0, full_in=2, out=1),
    )

    with pytest.raises(ValueError, match="non-positive/non-finite"):
        engine.build_step()


@pytest.mark.parametrize("invalid_overhead", (-1.0, float("nan"), "invalid"))
def test_des_rejects_raw_component_call_overhead_without_clamping(
    invalid_overhead,
) -> None:
    latency = CallLatency(
        prefill_s=lambda tokens, batch, tp=1, pp=1: tokens / 100_000.0,
        decode_s=lambda inp, out, batch, tp=1, pp=1: batch / 1_000_000.0,
        call_overhead_by_engine={"generator": invalid_overhead},
    )

    inputs, reason = build_sim_inputs(
        {"system": {"batching": {"dynamic_timeout_s": 0.0}}},
        SystemConfig(num_cores=8),
        [[{"stage": "generator", "input_tokens": 8, "output_tokens": 2}]],
        [["generator_prefill", "generator_decode"]],
        batch_size_request=2,
        batch_size_decode=2,
        latency=latency,
        prefix_caching=False,
    )

    assert inputs is None
    assert reason is not None
    assert "component call overhead" in reason
