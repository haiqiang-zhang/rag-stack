"""DES contracts for measured P/D role-local client admission."""

import pytest

from rag_stack.cost_model.closed_loop_sim import (
    SimCall,
    SimEngine,
    SimRound,
    SimTransfer,
    _pd_decode_admission_capacity,
    _pd_prefill_admission_capacity,
    _structural_start_population,
    simulate,
)


def _pd_call(*, full_in: int, transfer: SimTransfer) -> SimCall:
    return SimCall(
        engine=0,
        decode_engine=1,
        eff_in=1,
        full_in=full_in,
        out=1,
        transfer=transfer,
        prefill_admission_gate="generator:prefill",
        prefill_admission_capacity=3,
        decode_admission_gate="generator:decode",
        decode_admission_capacity=1,
    )


def test_pd_admission_capacity_uses_runtime_feeder_policy(monkeypatch):
    from rag_stack_evaluator.static_rag_evaluator.measured import vllm_pd_pair

    assert _pd_prefill_admission_capacity(2) == 3
    assert _pd_decode_admission_capacity(256) == 256

    monkeypatch.setattr(vllm_pd_pair, "pd_prefill_feeder_slots", lambda: 4)

    # Two independent P replicas each have engine cap 2 plus four frontend
    # feeder slots. D stays strict at its role-local engine residency.
    assert _pd_prefill_admission_capacity(2, replicas=2) == 12
    assert _pd_decode_admission_capacity(7, replicas=2) == 14


def test_pd_structural_population_fills_both_role_gates():
    transfer = SimTransfer(
        name="generator_kv",
        resource="gpu_gpu",
        payload_bytes=1.0,
        bandwidth_bytes_s=1.0,
    )
    call = _pd_call(full_in=16, transfer=transfer)
    call.decode_admission_capacity = 256
    queries = [[SimRound(stage_idxs=(), call=call)]]

    assert _structural_start_population(
        batch_size_request=2,
        batch_size_decode=256,
        queries=queries,
    ) == 259


def test_pd_decode_is_reserved_before_physical_transfer():
    prefill = SimEngine(
        name="generator.prefill",
        cards=(0,),
        depth=2,
        role="prefill",
        step_s=lambda decoding, chunk: 0.1,
        prefill_batch_s=lambda items: 0.1,
    )
    decode = SimEngine(
        name="generator.decode",
        cards=(1,),
        depth=1,
        role="decode",
        step_s=lambda decoding, chunk: 1.0,
        decode_step_s=lambda items: 1.0,
    )
    transfer = SimTransfer(
        name="generator_kv",
        resource="gpu_gpu",
        payload_bytes=1.0,
        bandwidth_bytes_s=1.0,
    )
    queries = [[SimRound(
        stage_idxs=(),
        call=_pd_call(full_in=16, transfer=transfer),
    )]]

    result = simulate(
        [],
        [prefill, decode],
        queries,
        n_cards=2,
        population=8,
        warmup_turnovers=2,
        window_turnovers=5,
    )

    assert result.converged
    # Strict D=1 covers both the 1s NIXL transfer and 1s decode. A second
    # transfer cannot enter the physical link while that D permit is held.
    assert result.qps == pytest.approx(0.5)
    assert result.transfer_queue_wait_mean[transfer.name] == pytest.approx(0.0)
    assert result.transfer_latency_mean[transfer.name] == pytest.approx(1.0)


def test_pd_prefill_permit_is_held_while_waiting_for_decode():
    prefill_batches = []

    def prefill_service(items):
        prefill_batches.append(tuple(
            item.context_tokens + item.new_tokens for item in items
        ))
        return 0.1

    prefill = SimEngine(
        name="generator.prefill",
        cards=(0,),
        depth=2,
        role="prefill",
        step_s=lambda decoding, chunk: 0.1,
        prefill_batch_s=prefill_service,
    )
    decode = SimEngine(
        name="generator.decode",
        cards=(1,),
        depth=1,
        role="decode",
        step_s=lambda decoding, chunk: 10.0,
        decode_step_s=lambda items: 10.0,
    )
    transfer = SimTransfer(
        name="generator_kv",
        resource="gpu_gpu",
        payload_bytes=10.0,
        bandwidth_bytes_s=1.0,
    )
    queries = [[SimRound(
        stage_idxs=(),
        call=_pd_call(full_in=10 + index, transfer=transfer),
    )] for index in range(8)]

    result = simulate(
        [],
        [prefill, decode],
        queries,
        n_cards=2,
        population=8,
        warmup_turnovers=1,
        window_turnovers=1,
    )

    assert result.converged
    # P cap=2 plus one feeder first supplies two full scheduler batches. Once
    # D=1 blocks, completed handles retain all three P permits, so subsequent
    # D releases can admit only one new P request at a time.
    assert [len(batch) for batch in prefill_batches[:5]] == [2, 2, 1, 1, 1]
    assert result.transfer_queue_wait_mean[transfer.name] == pytest.approx(0.0)
