"""Exact FIFO contracts for closed-loop simulator hot queues."""

from rag_stack.cost_model.closed_loop_sim import (
    SimCall,
    SimEngine,
    SimRound,
    SimStage,
    SimStageItem,
    SimStageOp,
    _Card,
    _Engine,
    simulate,
)


def test_engine_waiting_and_prefill_promotion_remain_fcfs():
    engine = _Engine(
        SimEngine(
            name="prefill",
            cards=(0,),
            depth=1,
            role="prefill",
            step_s=lambda decoding, chunk: 1e-3,
            prefill_batch_s=lambda items: 1e-3,
        ),
        [_Card()],
    )
    calls = [
        SimCall(engine=0, eff_in=tokens, full_in=tokens, out=1)
        for tokens in (11, 22, 33)
    ]

    assert not engine.admit(10, calls[0])
    assert not engine.admit(20, calls[1])
    assert not engine.admit(30, calls[2])

    completed = []
    resident_order = []
    for expected_rid in (10, 20, 30):
        plan = engine.build_step()
        assert plan is not None
        resident_order.append(plan.prefill_work[0][0])
        completed.extend(engine.complete_step(plan))
        engine.promote_and_sync()

    assert resident_order == [10, 20, 30]
    assert completed == [10, 20, 30]


def test_stage_batch_queue_remains_fcfs_across_front_removals():
    observed_batches = []

    def batch_ops(items):
        observed_batches.append(tuple(item.rid for item in items))
        return (SimStageOp(resource="cpu", service_s=1e-3),)

    stage = SimStage(
        name="retrieval",
        service_s=lambda batch: 1e-3,
        gpu_share=0.0,
        card=None,
        batch_cap=3,
        timeout_s=0.0,
        batch_ops=batch_ops,
    )
    queries = [[SimRound(
        stage_idxs=(0,),
        call=None,
        stage_items=(SimStageItem(stage="retrieval"),),
    )]]

    result = simulate(
        [stage],
        [],
        queries,
        n_cards=0,
        population=6,
        warmup_turnovers=1,
        window_turnovers=1,
    )

    assert result.converged
    assert observed_batches[:2] == [(0, 1, 2), (3, 4, 5)]
