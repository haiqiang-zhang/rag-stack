"""Output-preserving contracts for closed-loop DES hot-path shortcuts."""

import rag_stack.cost_model.closed_loop_sim as closed_loop_sim
import pytest

from rag_stack.cost_model.closed_loop_sim import (
    SimCall,
    SimEngine,
    SimRound,
    SimStage,
    SimTransfer,
    _Card,
    _Engine,
    simulate,
)


def _flat_step(decoding: int, chunk: float) -> float:
    return (decoding + chunk) / 10_000.0


def test_complete_step_consumes_only_the_frozen_fcfs_prefill_prefix():
    """A partial final slice stays at the head; completed heads keep order."""
    engine = _Engine(
        SimEngine(
            name="prefill",
            cards=(0,),
            depth=4,
            budget=3,
            role="prefill",
            step_s=_flat_step,
        ),
        [_Card()],
    )
    for rid, tokens in ((10, 5), (20, 2), (30, 1)):
        assert not engine.admit(
            rid,
            SimCall(engine=0, eff_in=tokens, full_in=tokens, out=1),
        )

    first = engine.build_step()
    assert first is not None
    assert first.prefill_work == ((10, 3.0),)
    assert engine.complete_step(first) == []
    assert [(state.rid, state.remaining) for state in engine.prefill] == [
        (10, 2.0), (20, 2.0), (30, 1.0),
    ]

    second = engine.build_step()
    assert second is not None
    assert second.prefill_work == ((10, 2.0), (20, 1.0))
    assert engine.complete_step(second) == [10]
    assert [(state.rid, state.remaining) for state in engine.prefill] == [
        (20, 1.0), (30, 1.0),
    ]


def test_scheduler_step_freezes_prefill_and_decode_work_before_completion():
    engine = _Engine(
        SimEngine(
            name="generator.unified",
            cards=(0,),
            depth=4,
            budget=8,
            role="unified",
            step_s=_flat_step,
        ),
        [_Card()],
    )
    assert not engine.admit(
        10, SimCall(engine=0, eff_in=1, full_in=1, out=3),
    )

    prefill_only = engine.build_step()
    assert prefill_only is not None
    assert prefill_only.decode_count == 0
    assert prefill_only.prefill_work == ((10, 1.0),)
    assert engine.complete_step(prefill_only) == []

    assert not engine.admit(
        20, SimCall(engine=0, eff_in=2, full_in=2, out=1),
    )
    mixed = engine.build_step()
    assert mixed is not None
    assert mixed.decode_count == 1
    assert mixed.prefill_work == ((20, 2.0),)
    assert mixed.service_s == pytest.approx(3 / 10_000)
    assert engine.complete_step(mixed) == []
    assert engine.decoding[10].generated == 1
    assert engine.decoding[20].generated == 0


def test_advertised_aggregate_decode_hook_preserves_production_statistics():
    """The fast hook receives the same ordered sums without item allocation."""
    aggregate_calls = []

    def item_hook(_items):
        raise AssertionError("advertised aggregate hook must bypass item hook")

    def aggregate_hook(batch, input_sum, generated_sum):
        aggregate_calls.append((batch, input_sum, generated_sum))
        return input_sum / 1_000_000.0 + generated_sum / 10_000_000.0

    item_hook._rag_cm_decode_aggregate_s = aggregate_hook
    engine = _Engine(
        SimEngine(
            name="decode",
            cards=(0,),
            depth=3,
            role="decode",
            step_s=_flat_step,
            decode_step_s=item_hook,
        ),
        [_Card()],
    )
    for rid, full_input in ((10, 101), (20, 202), (30, 303)):
        assert not engine.admit(
            rid,
            SimCall(engine=0, eff_in=0, full_in=full_input, out=2),
        )

    first = engine.build_step()
    assert first is not None
    assert aggregate_calls == [(3, 606.0, 0.0)]
    assert first.service_s == 606.0 / 1_000_000.0
    assert engine.complete_step(first) == []

    second = engine.build_step()
    assert second is not None
    assert aggregate_calls[-1] == (3, 606.0, 3.0)
    assert engine.complete_step(second) == [10, 20, 30]


def test_aggregate_decode_state_stays_exact_across_inflight_admission_and_exit():
    """Maintained sums preserve the frozen-prefix continuous-batching rules."""
    aggregate_calls = []

    def item_hook(_items):
        raise AssertionError("production aggregate path built decode items")

    def aggregate_hook(batch, input_sum, generated_sum):
        aggregate_calls.append((batch, input_sum, generated_sum))
        return 1e-3

    item_hook._rag_cm_decode_aggregate_s = aggregate_hook
    engine = _Engine(
        SimEngine(
            name="decode",
            cards=(0,),
            depth=4,
            role="decode",
            step_s=_flat_step,
            decode_step_s=item_hook,
        ),
        [_Card()],
    )
    assert not engine.admit(10, SimCall(0, 0, 2, full_in=101))
    assert not engine.admit(20, SimCall(0, 0, 3, full_in=202))

    frozen = engine.build_step()
    assert frozen is not None
    assert aggregate_calls[-1] == (2, 303.0, 0.0)

    # This request arrives while the preceding scheduler step is in flight.
    # It must contribute to the next price, but not receive that frozen token.
    assert not engine.admit(30, SimCall(0, 0, 5, full_in=303))
    assert engine.complete_step(frozen) == []
    second = engine.build_step()
    assert second is not None
    assert aggregate_calls[-1] == (3, 606.0, 2.0)

    # rid=10 exits in this step.  The ordered aggregate rebuild must retain the
    # generated state of rid=20 and the later admission exactly.
    assert engine.complete_step(second) == [10]
    third = engine.build_step()
    assert third is not None
    assert aggregate_calls[-1] == (2, 505.0, 3.0)


def test_aggregate_decode_build_does_not_scan_resident_values():
    """Production pricing reads maintained sums instead of a second scan."""

    class NoValuesDict(dict):
        def values(self):
            raise AssertionError("aggregate build rescanned decode values")

    def item_hook(_items):
        raise AssertionError("aggregate build allocated decode items")

    item_hook._rag_cm_decode_aggregate_s = lambda batch, _ins, _gens: (
        float(batch) * 1e-3
    )
    engine = _Engine(
        SimEngine(
            name="decode",
            cards=(0,),
            depth=4,
            role="decode",
            step_s=_flat_step,
            decode_step_s=item_hook,
        ),
        [_Card()],
    )
    for rid in range(4):
        assert not engine.admit(rid, SimCall(0, 0, 2, full_in=100 + rid))
    engine.decoding = NoValuesDict(engine.decoding)

    plan = engine.build_step()
    assert plan is not None
    assert plan.decode_count == 4


def test_lazy_decode_epoch_matches_eager_with_inflight_suffix_and_exits():
    """The production shortcut is numerically identical to eager residents."""
    eager_calls = []
    lazy_calls = []

    def make_hook(calls, *, lazy):
        def item_hook(_items):
            raise AssertionError("aggregate hook received decode items")

        def aggregate_hook(batch, input_sum, generated_sum):
            calls.append((batch, input_sum, generated_sum))
            return (batch + input_sum + generated_sum) / 1_000_000.0

        item_hook._rag_cm_decode_aggregate_s = aggregate_hook
        if lazy:
            item_hook._rag_cm_decode_lazy_safe = True
        return item_hook

    def make_engine(hook):
        return _Engine(
            SimEngine(
                name="decode",
                cards=(0,),
                depth=8,
                role="decode",
                step_s=_flat_step,
                decode_step_s=hook,
            ),
            [_Card()],
        )

    eager = make_engine(make_hook(eager_calls, lazy=False))
    lazy = make_engine(make_hook(lazy_calls, lazy=True))
    for engine in (eager, lazy):
        assert not engine.admit(10, SimCall(0, 0, 2, full_in=101))
        assert not engine.admit(20, SimCall(0, 0, 4, full_in=202))

    for step in range(6):
        eager_plan = eager.build_step()
        lazy_plan = lazy.build_step()
        assert eager_plan is not None
        assert lazy_plan is not None
        assert lazy_plan.service_s == eager_plan.service_s
        if step == 0:
            # This suffix arrives after the two plans have frozen.  It must not
            # receive the already-running token in either implementation.
            for engine in (eager, lazy):
                assert not engine.admit(30, SimCall(0, 0, 3, full_in=303))
        assert lazy.complete_step(lazy_plan) == eager.complete_step(eager_plan)
        assert lazy_calls == eager_calls
        if not eager.decoding:
            break

    assert not lazy.decoding
    assert not eager.decoding
    assert lazy._decode_input_sum == eager._decode_input_sum == 0.0
    assert lazy._decode_generated_sum == eager._decode_generated_sum == 0.0


def test_lazy_decode_step_does_not_iterate_residents(monkeypatch):
    """A no-exit token step is O(1), including completion."""

    class NoIterationDict(dict):
        def items(self):
            raise AssertionError("lazy decode completion iterated residents")

        def values(self):
            raise AssertionError("lazy decode pricing iterated residents")

    def item_hook(_items):
        raise AssertionError("lazy production hook received decode items")

    item_hook._rag_cm_decode_aggregate_s = lambda batch, _ins, _gens: (
        float(batch) * 1e-3
    )
    item_hook._rag_cm_decode_lazy_safe = True
    engine = _Engine(
        SimEngine(
            name="decode",
            cards=(0,),
            depth=64,
            role="decode",
            step_s=_flat_step,
            decode_step_s=item_hook,
        ),
        [_Card()],
    )
    for rid in range(64):
        assert not engine.admit(rid, SimCall(0, 0, 8, full_in=100 + rid))
    engine.decoding = NoIterationDict(engine.decoding)

    plan = engine.build_step()
    assert plan is not None
    assert engine.complete_step(plan) == []
    assert engine._decode_generated_sum == 64.0


def test_lazy_decode_audit_environment_forces_eager(monkeypatch):
    """Spawned-worker A/B can disable the shortcut without changing code."""
    monkeypatch.setenv("CM_DES_DISABLE_LAZY_DECODE", "1")

    def hook(_items):
        return 1e-3

    hook._rag_cm_decode_aggregate_s = lambda *_args: 1e-3
    hook._rag_cm_decode_lazy_safe = True
    engine = _Engine(
        SimEngine(
            name="decode",
            cards=(),
            depth=1,
            role="decode",
            step_s=_flat_step,
            decode_step_s=hook,
        ),
        [],
    )

    assert not engine._lazy_decode


def test_custom_decode_hook_still_receives_per_sequence_items():
    """Unmarked callbacks keep the public shape-aware callback contract."""
    observed = []

    def item_hook(items):
        observed.append(tuple(
            (item.input_tokens, item.generated_tokens, item.remaining_tokens)
            for item in items
        ))
        return 1e-3

    engine = _Engine(
        SimEngine(
            name="custom",
            cards=(0,),
            depth=2,
            role="decode",
            step_s=_flat_step,
            decode_step_s=item_hook,
        ),
        [_Card()],
    )
    for rid, full_input, output in ((1, 11, 2), (2, 22, 3)):
        assert not engine.admit(
            rid,
            SimCall(engine=0, eff_in=0, full_in=full_input, out=output),
        )

    plan = engine.build_step()
    assert plan is not None
    assert observed == [((11.0, 0.0, 2.0), (22.0, 0.0, 3.0))]


def test_same_time_adjacent_engine_recomputes_are_coalesced(monkeypatch):
    """A wave of admissions needs one rescan, not one no-op per request."""
    original_heappush = closed_loop_sim.heapq.heappush
    recompute_pushes = 0

    def counting_heappush(heap, item):
        nonlocal recompute_pushes
        if item[2] == "engine_recompute":
            recompute_pushes += 1
        return original_heappush(heap, item)

    monkeypatch.setattr(closed_loop_sim.heapq, "heappush", counting_heappush)
    engine = SimEngine(
        name="generator",
        cards=(0,),
        depth=32,
        step_s=_flat_step,
    )
    result = simulate(
        [],
        [engine],
        [[SimRound(stage_idxs=(), call=SimCall(0, 0, 1))]],
        n_cards=1,
        population=32,
        warmup_turnovers=1,
        window_turnovers=1,
    )

    assert result.converged
    # Initial admission plus the two completion/recycle waves.  The exact
    # final wave can stop before its queued rescan is consumed, but it is still
    # enqueued once.  Without coalescing this is 96 events.
    assert recompute_pushes == 3


def test_coalesced_engine_recomputes_still_consume_logical_event_budget(
    monkeypatch,
):
    """The safety cap must match the uncoalesced reference event stream."""
    population = 32
    monkeypatch.setattr(closed_loop_sim, "_SIM_MAX_EVENTS", population)
    result = simulate(
        [],
        [SimEngine(
            name="generator",
            cards=(0,),
            depth=population,
            step_s=_flat_step,
        )],
        [[SimRound(stage_idxs=(), call=SimCall(0, 0, 1))]],
        n_cards=1,
        population=population,
        warmup_turnovers=0,
        window_turnovers=1,
    )

    # Prime-time admission logically emits one recompute per request.  They
    # are physically one retained event, but the reference budget is exhausted
    # before the first GPU completion can run.
    assert not result.converged
    assert result.completions == 0


def test_coalesced_stage_kicks_still_consume_logical_event_budget(monkeypatch):
    """Count-trigger duplicate kicks retain their reference safety cost."""
    population = 16
    monkeypatch.setattr(closed_loop_sim, "_SIM_MAX_EVENTS", population)
    result = simulate(
        [SimStage(
            name="stage",
            service_s=lambda _batch: 1e-3,
            gpu_share=0.0,
            card=None,
            batch_cap=1,
            timeout_s=0.0,
        )],
        [],
        [[SimRound(stage_idxs=(0,), call=None)]],
        n_cards=0,
        population=population,
        warmup_turnovers=0,
        window_turnovers=1,
    )

    # The first kick starts the worker, while the adjacent retained duplicates
    # consume the remaining logical cap before stage_done can complete a row.
    assert not result.converged
    assert result.completions == 0


def test_full_batch_timeout_kick_can_change_same_time_gpu_fcfs_order():
    """No partial timeout batches is insufficient for timeout-result reuse."""
    transfer = SimTransfer(
        name="delay",
        resource="link",
        payload_bytes=1.0,
        bandwidth_bytes_s=1.0,
    )
    queries = [
        [SimRound(
            stage_idxs=(0,),
            call=None,
            boundary_transfers={0: (transfer,)},
        )],
        [SimRound(stage_idxs=(0,), call=None)],
        [SimRound(stage_idxs=(), call=SimCall(0, 0, 2))],
    ]

    def run(timeout_s):
        return simulate(
            [SimStage(
                name="stage",
                service_s=lambda _batch: 0.25,
                gpu_share=1.0,
                card=0,
                batch_cap=2,
                timeout_s=timeout_s,
            )],
            [SimEngine(
                name="engine",
                cards=(0,),
                depth=4,
                step_s=lambda _decoding, _chunk: 1.0,
            )],
            queries,
            n_cards=1,
            population=3,
            warmup_turnovers=0,
            window_turnovers=1,
        )

    short = run(1.0)
    long = run(2.0)

    # Every observed stage batch is full in both runs.  Even so, at t=1 the
    # short timeout's earlier heap sequence queues the stage GPU op before the
    # engine step releases its card; the long timeout lets the next engine step
    # reacquire first.  Reusing the short result would therefore be lossy.
    assert short.stage_batch_mean == long.stage_batch_mean == {"stage": 2.0}
    assert short.qps != long.qps
    assert short.latency_mean_s != long.latency_mean_s


def test_one_engine_batch_promotes_waiters_once(monkeypatch):
    """All vacancies from one atomic step are filled by one FCFS promotion."""
    original_promote = _Engine.promote_and_sync
    original_complete = _Engine.complete_step
    promotion_calls = 0
    completed_step_calls = 0

    def counting_promote(self):
        nonlocal promotion_calls
        promotion_calls += 1
        return original_promote(self)

    def counting_complete(self, plan):
        nonlocal completed_step_calls
        done = original_complete(self, plan)
        if done:
            completed_step_calls += 1
        return done

    monkeypatch.setattr(_Engine, "promote_and_sync", counting_promote)
    monkeypatch.setattr(_Engine, "complete_step", counting_complete)
    engine = SimEngine(
        name="generator",
        cards=(0,),
        depth=4,
        step_s=_flat_step,
    )
    result = simulate(
        [],
        [engine],
        [[SimRound(stage_idxs=(), call=SimCall(0, 0, 1))]],
        n_cards=1,
        population=16,
        warmup_turnovers=1,
        window_turnovers=1,
    )

    assert result.converged
    assert promotion_calls == completed_step_calls
    assert promotion_calls > 1


def test_multiple_boundary_transfers_retain_exact_fcfs_link_accounting():
    """Immediate progress retirement must not skip or repeat an edge."""
    first = SimTransfer(
        name="first",
        resource="shared_link",
        payload_bytes=2.0,
        bandwidth_bytes_s=10.0,
    )
    second = SimTransfer(
        name="second",
        resource="shared_link",
        payload_bytes=3.0,
        bandwidth_bytes_s=10.0,
    )
    engine = SimEngine(
        name="generator",
        cards=(0,),
        depth=8,
        step_s=lambda decoding, chunk: 1e-9,
    )
    result = simulate(
        [],
        [engine],
        [[SimRound(
            stage_idxs=(),
            call=SimCall(0, 0, 1),
            boundary_transfers={0: (first, second)},
        )]],
        n_cards=1,
        population=1,
        warmup_turnovers=1,
        window_turnovers=2,
    )

    assert result.converged
    # The sole request serially traverses 0.2s then 0.3s each turnover.
    assert abs(result.qps - 2.0) < 1e-6
    assert result.transfer_latency_mean == pytest.approx(
        {"first": 0.2, "second": 0.3},
    )
    assert result.transfer_queue_wait_mean == pytest.approx(
        {"first": 0.0, "second": 0.0},
    )
