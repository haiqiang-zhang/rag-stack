"""Mechanism tests for replica-local automatic prefix caching."""

from types import SimpleNamespace

import pytest

from rag_stack.cost_model.closed_loop_sim import (
    SimCall,
    SimEngine,
    SimPrefillItem,
    SimRound,
    _BlockLRU,
    _Card,
    _Engine,
    _apc_block_keys,
    _dense_decoder_weight_bytes,
    _kv_tp_shard_degree,
    _prefix_cache_capacity_blocks,
    build_sim_inputs,
    simulate,
)
from rag_stack.cost_model.engines import CallLatency
from rag_stack.types import SystemConfig


def _cached_call(*, engine=0, engine_pool=(), full_in=32, out=1):
    blocks = tuple(f"prompt-{index}" for index in range(full_in // 16))
    return SimCall(
        engine=engine,
        engine_pool=engine_pool,
        eff_in=full_in,
        full_in=full_in,
        out=out,
        cache_identity=(0, "generator", 0, full_in),
        cache_blocks=blocks,
        result_cache_blocks=blocks,
    )


def _engine(*, name="generator.unified.r0", cards=(0,), capacity=8, prefill=None):
    return SimEngine(
        name=name,
        cards=cards,
        depth=1,
        role="unified",
        step_s=lambda decoding, chunk: 1e-3,
        prefill_batch_s=prefill or (lambda items: 1e-3),
        decode_step_s=lambda items: 1e-3,
        call_overhead_s=0.0,
        prefix_cache_block_tokens=16,
        prefix_cache_capacity_blocks=capacity,
    )


def test_block_lru_refreshes_hits_and_evicts_oldest_block():
    cache = _BlockLRU(2)
    cache.publish(("a", "b"))

    assert cache.longest_prefix(("a", "missing")) == 1
    cache.publish(("c",))

    assert "a" in cache
    assert "c" in cache
    assert "b" not in cache


def test_shared_kv_pool_debits_resident_blocks_and_never_evicts_pinned():
    pool = _BlockLRU(4)
    pool.publish(("a", "b", "c", "d"))
    pinned = pool.acquire_longest_prefix(("a",))

    assert pinned == ("a",)
    assert pool.reserve_anonymous(3)
    assert pool.physical_blocks == 4
    assert pool.anonymous_blocks == 3
    assert pool.pinned_blocks == 1
    assert "a" in pool
    assert not pool.reserve_anonymous(1)

    pool.release(pinned)
    assert pool.reserve_anonymous(1)
    assert pool.physical_blocks == 4
    assert len(pool) == 0


def test_two_residents_share_and_pin_one_cached_prefix_block():
    pool = _BlockLRU(2)
    pool.publish(("shared",))

    first = pool.acquire_longest_prefix(("shared",))
    second = pool.acquire_longest_prefix(("shared",))

    assert first == second == ("shared",)
    assert pool.physical_blocks == 1
    assert pool.pinned_blocks == 1
    pool.release(first)
    assert not pool.reserve_anonymous(2)
    pool.release(second)
    assert pool.reserve_anonymous(2)
    assert pool.physical_blocks == 2


def test_failed_resident_reservation_does_not_partially_evict_cache():
    pool = _BlockLRU(3)
    pool.publish(("pinned", "free-0", "free-1"))
    held = pool.acquire_longest_prefix(("pinned",))

    assert not pool.reserve_anonymous(3)
    assert pool.physical_blocks == 3
    assert "pinned" in pool
    assert "free-0" in pool
    assert "free-1" in pool
    pool.release(held)


def test_e23_like_resident_pressure_cannot_retain_cache_beyond_pool():
    pool = _BlockLRU(48)
    working_set = tuple(f"prompt-{index}" for index in range(15))
    pool.publish(working_set)

    assert pool.reserve_anonymous(40)
    assert pool.physical_blocks == 48
    assert pool.anonymous_blocks == 40
    assert len(pool) == 8
    assert pool.longest_prefix(working_set) == 0


def test_engine_resident_kv_and_reusable_cache_share_one_capacity():
    engine = _Engine(_engine(capacity=4), [_Card()])
    engine.prefix_cache.publish(("old-0", "old-1", "old-2", "old-3"))
    call = SimCall(
        engine=0,
        eff_in=32,
        full_in=32,
        out=1,
        cache_blocks=("new-0", "new-1"),
        result_cache_blocks=("new-0", "new-1"),
    )

    assert not engine.admit(0, call)
    assert engine.prefix_cache.physical_blocks == 4
    assert engine.prefix_cache.anonymous_blocks == 3
    assert len(engine.prefix_cache) == 1


def test_engine_kv_lifecycle_promotes_prompt_then_releases_result():
    engine = _Engine(_engine(capacity=5), [_Card()])
    call = SimCall(
        engine=0,
        eff_in=16,
        full_in=16,
        out=16,
        cache_blocks=("prompt",),
        result_cache_blocks=("prompt", "result"),
    )

    assert not engine.admit(0, call)
    assert engine.prefix_cache.anonymous_blocks == 2

    prefill = engine.build_step()
    assert prefill is not None
    assert engine.complete_step(prefill) == []
    assert engine.prefix_cache.pinned_blocks == 1
    assert engine.prefix_cache.anonymous_blocks == 1
    assert "prompt" in engine.prefix_cache

    done = []
    while not done:
        decode = engine.build_step()
        assert decode is not None
        done = engine.complete_step(decode)
    assert done == [0]
    assert engine.prefix_cache.pinned_blocks == 0
    assert engine.prefix_cache.anonymous_blocks == 0
    assert engine.prefix_cache.physical_blocks == 2
    assert "prompt" in engine.prefix_cache
    assert "result" in engine.prefix_cache


def test_engine_memory_pressure_queues_until_resident_retires():
    spec = _engine(capacity=4)
    spec.depth = 2
    engine = _Engine(spec, [_Card()])

    def call(prefix):
        return SimCall(
            engine=0,
            eff_in=32,
            full_in=32,
            out=1,
            cache_blocks=(f"{prefix}-0", f"{prefix}-1"),
            result_cache_blocks=(f"{prefix}-0", f"{prefix}-1"),
        )

    assert not engine.admit(0, call("first"))
    assert not engine.admit(1, call("second"))
    assert engine.residents() == 1
    assert len(engine.waiting) == 1

    done = []
    while not done:
        plan = engine.build_step()
        assert plan is not None
        done = engine.complete_step(plan)
    assert done == [0]

    assert engine.promote_and_sync() == []
    assert engine.residents() == 1
    assert len(engine.waiting) == 0
    assert engine.prefix_cache.physical_blocks == 4
    assert engine.prefix_cache.anonymous_blocks == 3
    assert len(engine.prefix_cache) == 1


def test_repeated_closed_loop_turnover_hits_warm_prefix_cache():
    prefill_tokens = []

    def prefill(items):
        prefill_tokens.append(sum(item.new_tokens for item in items))
        return 1e-3

    result = simulate(
        [],
        [_engine(prefill=prefill)],
        [[SimRound(stage_idxs=(), call=_cached_call())]],
        n_cards=1,
        population=1,
        warmup_turnovers=1,
        window_turnovers=2,
    )

    assert result.converged
    # vLLM 0.18.1 caps APC reuse at num_tokens-1.  Since 32 is exactly block
    # aligned, every hit recomputes the final complete 16-token block.
    assert prefill_tokens == [32.0, 16.0, 16.0]


def test_prefix_cache_is_isolated_between_unified_replicas():
    prefill_tokens = []

    def prefill(items):
        prefill_tokens.append(sum(item.new_tokens for item in items))
        return 1e-3

    engines = [
        _engine(name="generator.unified.r0", cards=(0,), prefill=prefill),
        _engine(name="generator.unified.r1", cards=(1,), prefill=prefill),
    ]
    result = simulate(
        [],
        engines,
        [[SimRound(
            stage_idxs=(),
            call=_cached_call(engine_pool=(0, 1)),
        )]],
        n_cards=2,
        population=1,
        warmup_turnovers=2,
        window_turnovers=2,
    )

    assert result.converged
    # Least-load tie breaking alternates replicas. Each concrete replica must
    # compute and publish the prompt once; neither sees the other's cache.
    # Later exact-boundary hits recompute each replica's final block.
    assert prefill_tokens == [32.0, 32.0, 16.0, 16.0]


def test_exact_block_multiple_apc_hit_recomputes_final_complete_block():
    engine = _Engine(_engine(), [_Card()])
    engine.prefix_cache.publish(("prompt-0", "prompt-1"))

    assert not engine.admit(0, _cached_call(full_in=32))
    assert engine.prefill[-1].context == 16.0
    assert engine.prefill[-1].remaining == 16.0


def test_pd_prefill_cache_does_not_claim_decode_generated_blocks():
    spec = SimEngine(
        name="generator.prefill.r0",
        cards=(0,),
        depth=2,
        role="prefill",
        step_s=lambda decoding, chunk: 1e-3,
        prefill_batch_s=lambda items: 1e-3,
        call_overhead_s=0.0,
        prefix_cache_block_tokens=16,
        prefix_cache_capacity_blocks=8,
    )
    engine = _Engine(spec, [_Card()])
    first = SimCall(
        engine=0,
        eff_in=16,
        full_in=16,
        out=16,
        cache_blocks=("prompt",),
        result_cache_blocks=("prompt", "generated"),
    )
    assert not engine.admit(0, first)
    plan = engine.build_step()
    assert plan is not None
    assert engine.complete_step(plan) == [0]

    assert "prompt" in engine.prefix_cache
    assert "generated" not in engine.prefix_cache

    second = SimCall(
        engine=0,
        eff_in=32,
        full_in=32,
        out=1,
        cache_blocks=("prompt", "generated"),
        result_cache_blocks=("prompt", "generated"),
    )
    assert not engine.admit(1, second)
    # Only the first prompt block belongs to the prefill replica. The block
    # produced by PD decode is a miss and must be recomputed here.
    assert engine.prefill[-1].remaining == 16.0
    assert engine.prefill[-1].context == 16.0


def test_capacity_is_derived_from_memory_weight_and_sharded_kv_shape():
    capacity = _prefix_cache_capacity_blocks(
        gpu_memory_bytes=1000,
        gpu_memory_utilization=1.0,
        model_weight_bytes=200,
        kv_bytes_per_token=10,
        tp=2,
        pp=1,
        block_tokens=10,
    )
    assert capacity == 18
    assert _prefix_cache_capacity_blocks(
        gpu_memory_bytes=100,
        gpu_memory_utilization=0.5,
        model_weight_bytes=200,
        kv_bytes_per_token=10,
        tp=1,
        pp=1,
        block_tokens=16,
    ) == 0

    # With two KV heads and TP=4, vLLM has two distinct KV shards replicated
    # across four ranks.  Weights still shard four ways, but KV does not.
    assert _kv_tp_shard_degree(4, 2) == 2
    assert _prefix_cache_capacity_blocks(
        gpu_memory_bytes=1000,
        gpu_memory_utilization=1.0,
        model_weight_bytes=200,
        kv_bytes_per_token=10,
        tp=4,
        pp=1,
        block_tokens=10,
        kv_head_count=2,
    ) == 19

    cfg = SimpleNamespace(
        hidden_size=4,
        num_decoder_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        intermediate_size=8,
        num_ffi=2,
        vocab_size=10,
        is_moe=False,
        is_mamba=False,
    )
    # embedding 40 + 2 layers * (QKVO 48 + FFN 96) = 328 params.
    assert _dense_decoder_weight_bytes(cfg, 2) == 656.0


def test_apc_keys_preserve_shared_deterministic_turn_prefix():
    first_prompt = (((0, "generator", 0, "prompt"), 16),)
    first_result = first_prompt + (((0, "generator", 0, "output"), 16),)

    first_keys = _apc_block_keys(first_result, 32, 16)
    second_keys = _apc_block_keys(
        first_result + (((0, "generator", 1, "prompt_suffix"), 16),),
        48,
        16,
    )

    assert second_keys[:2] == first_keys
    assert second_keys[2] not in first_keys


def test_two_role_llm_rejects_absorbable_component_call_overhead():
    """A two-role engine has one stage-curve owner for framework overhead."""
    pipeline = {
        "system": {"communication": {"include_non_pd_comm": False}},
        "node_lines": [{"nodes": [{
            "stage": "generator",
            "modules": [{
                "component": "vllm",
                "model": "Qwen/Qwen2.5-1.5B-Instruct",
                "dtype": "bfloat16",
                "gpu_memory_utilization": 0.4,
                "temperature": 0.0,
            }],
        }]}],
    }

    def kernel(tokens, batch, tp=1, pp=1):
        return float(tokens) * batch / 100_000.0

    latency = CallLatency(
        # The analytical closure contains the affine term.
        prefill_s=lambda tokens, batch, tp=1, pp=1: (
            kernel(tokens, batch, tp, pp) + 0.025
        ),
        decode_s=lambda inp, out, batch, tp=1, pp=1: 1e-3,
        prefill_kernel_by_engine={"generator": kernel},
        call_overhead_by_engine={"generator": 0.025},
    )
    inputs, reason = build_sim_inputs(
        pipeline,
        SystemConfig(num_cores=1, gpu={
            "hardware_key": "unit_test_gpu",
            "Memory_size": 80.0,
            "Memory_BW": 1.0,
            "Flops": 1.0,
            "ICN": 1.0,
        }),
        [[{"stage": "generator", "input_tokens": 32, "output_tokens": 1}]],
        [["generator_prefill", "generator_decode"]],
        batch_size_request=1,
        batch_size_decode=1,
        latency=latency,
        stage_batch_profiles={},
    )

    assert inputs is None
    assert reason is not None
    assert "forbidden component call overhead 0.025" in reason
    assert "one non-absorbable owner" in reason


@pytest.mark.parametrize(
    ("qe_temperature", "generator_is_cacheable"),
    [(0.0, True), (1.0, False)],
)
def test_stochastic_upstream_output_never_becomes_exact_generator_cache_key(
    qe_temperature,
    generator_is_cacheable,
):
    pipeline = {
        "system": {"communication": {"include_non_pd_comm": False}},
        "node_lines": [{
            "nodes": [{
                "stage": "query_expansion",
                "modules": [{
                        "component": "hyde",
                    "model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "dtype": "bfloat16",
                    "gpu_memory_utilization": 0.4,
                    "temperature": qe_temperature,
                }],
            }, {
                "stage": "generator",
                "modules": [{
                    "component": "vllm",
                    "model": "Qwen/Qwen2.5-1.5B-Instruct",
                    "dtype": "bfloat16",
                    "gpu_memory_utilization": 0.4,
                    "temperature": 0.0,
                }],
            }],
        }],
    }
    latency = CallLatency(
        prefill_s=lambda tokens, batch, tp=1, pp=1: 1e-3,
        decode_s=lambda inp, out, batch, tp=1, pp=1: 1e-3,
        prefill_by_engine={
            "query_expansion": lambda tokens, batch, tp=1, pp=1: 1e-3,
        },
        decode_by_engine={
            "query_expansion": lambda inp, out, batch, tp=1, pp=1: 1e-3,
        },
    )
    logical_calls = [
        {
            "stage": "query_expansion",
            "input_tokens": 32,
            "output_tokens": 16,
        },
        {
            "stage": "generator",
            "input_tokens": 64,
            "output_tokens": 1,
        },
    ]
    # Two completion-sample rows came from the same stable dataset question.
    # Their reservoir positions differ but their logical prompt lineage does
    # not, so APC keys must be identical.
    trace = [
        [dict(call) for call in logical_calls],
        [dict(call) for call in logical_calls],
    ]

    inputs, reason = build_sim_inputs(
        pipeline,
        SystemConfig(num_cores=1, gpu={
            "hardware_key": "unit_test_gpu",
            "Memory_size": 80.0,
            "Memory_BW": 1.0,
            "Flops": 1.0,
            "ICN": 1.0,
        }),
        trace,
        [
            ["query_expansion_prefill", "query_expansion_decode"],
            ["generator_prefill", "generator_decode"],
        ],
        trace_source_question_ids=["dataset-5", "dataset-5"],
        batch_size_request=1,
        batch_size_decode=1,
        latency=latency,
        stage_batch_profiles={},
    )

    assert reason is None
    _stages, engines, queries, *_ = inputs
    assert all(
        engine.prefix_cache_capacity_blocks > 0
        for engine in engines
        if engine.role in {"unified", "prefill"}
    )
    qe_call = queries[0][0].call
    generator_call = queries[0][1].call
    assert qe_call is not None and generator_call is not None
    assert qe_call.cache_identity == ("dataset-5", "query_expansion", 0, 32)
    assert qe_call.cache_blocks
    assert queries[1][0].call.cache_blocks == qe_call.cache_blocks
    if qe_temperature > 0:
        assert qe_call.result_cache_blocks == qe_call.cache_blocks
    assert bool(generator_call.cache_identity) is generator_is_cacheable
    assert bool(generator_call.cache_blocks) is generator_is_cacheable

    unknown_inputs, unknown_reason = build_sim_inputs(
        pipeline,
        SystemConfig(num_cores=1, gpu={
            "hardware_key": "unit_test_gpu",
            "Memory_size": 80.0,
            "Memory_BW": 1.0,
            "Flops": 1.0,
            "ICN": 1.0,
        }),
        [trace[0]],
        [
            ["query_expansion_prefill", "query_expansion_decode"],
            ["generator_prefill", "generator_decode"],
        ],
        batch_size_request=1,
        batch_size_decode=1,
        latency=latency,
        stage_batch_profiles={},
    )
    assert unknown_reason is None
    _stages, _engines, unknown_queries, *_ = unknown_inputs
    assert unknown_queries[0][0].call.cache_identity is None
    assert unknown_queries[0][0].call.cache_blocks == ()
