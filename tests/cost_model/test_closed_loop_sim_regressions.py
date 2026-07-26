"""Regression contracts for the shape-aware closed-loop simulator.

These tests deliberately exercise mechanism and state, not fitted end-to-end
throughput constants.  Simulator-only state must also remain behind the frozen
canonical quality-trace boundary.
"""

from copy import deepcopy
import inspect
import pytest

from rag_stack.rag_ir import (
    make_quality_trace_envelope,
    normalize_quality_trace,
    validate_quality_trace_envelope,
)
from rag_stack.cost_model.assembly import RAGCMAssembly
from rag_stack.cost_model.closed_loop_sim import (
    SimCall,
    SimDecodeItem,
    SimEngine,
    SimPrefillItem,
    SimRound,
    SimStage,
    SimStageItem,
    SimStageOp,
    SimTransfer,
    _Card,
    _Engine,
    _GpuArbiter,
    _coprime_trace_stride,
    _llm_admission_capacity,
    build_sim_inputs,
    simulate,
)
from rag_stack.cost_model.engines import CallLatency
from rag_stack.types import SystemConfig


def _generator_call(*, input_tokens: int, output_tokens: int, step_idx: int = 0):
    return {
        "stage": "generator",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "step_idx": step_idx,
    }


def test_simulator_shape_state_stays_outside_frozen_v2_envelope():
    """Internal context/batch state must not extend or rewrite CM input v2."""
    envelope = make_quality_trace_envelope(
        [[_generator_call(input_tokens=113, output_tokens=7)]],
        question_ids=["q0"],
    )
    before = deepcopy(envelope)

    normalized = normalize_quality_trace(envelope)

    assert envelope == before
    assert normalized.traces == [before["queries"][0]["calls"]]
    assert set(normalized.traces[0][0]) == {
        "stage", "input_tokens", "output_tokens", "step_idx",
    }

    # ``full_in``/decode position and stage-batch membership are simulator
    # state.  In particular, they must not become a second way to feed runtime
    # observations through the public quality-trace contract.
    leaked = deepcopy(envelope)
    leaked["queries"][0]["calls"][0]["full_in"] = 113
    with pytest.raises(ValueError, match="unsupported fields"):
        validate_quality_trace_envelope(leaked)


def test_shape_aware_simulation_does_not_expand_public_dynamic_api():
    """Shape-aware DES inputs are derived after the existing CM boundary."""
    params = inspect.signature(RAGCMAssembly.evaluate_dynamic).parameters

    assert params["pipeline_config"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert params["workflow_schema"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["trace"].kind is inspect.Parameter.KEYWORD_ONLY
    assert not {
        "full_in",
        "decode_position",
        "stage_items",
        "prefill_batch",
    } & set(params)


@pytest.mark.parametrize("query_count", [2, 3, 75, 78, 100])
def test_trace_rotation_stride_visits_every_query(query_count):
    stride = _coprime_trace_stride(query_count)
    visited = []
    index = 0
    for _ in range(query_count):
        visited.append(index)
        index = (index + stride) % query_count

    assert len(set(visited)) == query_count
    assert index == 0


def test_des_builds_latency_from_stage_service_contract(monkeypatch):
    """Physical service construction uses the normal stage-service path."""
    observed: dict[str, bool] = {}

    def fake_build_call_latency(pipeline_config, hardware):
        del pipeline_config, hardware
        observed["called"] = True
        return CallLatency(
            prefill_s=lambda tokens, batch, tp=1, pp=1: 1e-4,
            decode_s=lambda inp, out, batch, tp=1, pp=1: 1e-4,
        )

    monkeypatch.setattr(
        "rag_stack.cost_model.engines.build_call_latency",
        fake_build_call_latency,
    )
    pipeline = {
        "system": {
            "batching": {"dynamic_timeout_s": 0.0},
            "communication": {"include_non_pd_comm": False},
        },
    }
    inputs, reason = build_sim_inputs(
        pipeline,
        SystemConfig(num_cores=16),
        [[_generator_call(input_tokens=16, output_tokens=1)]],
        [["generator_prefill", "generator_decode"]],
        stage_batch_profiles={},  # Explicit analytical-only fixture.
        batch_size_request=2,
        batch_size_decode=2,
    )

    assert reason is None
    assert inputs is not None
    assert observed["called"] is True


def test_decode_step_sees_each_sequence_context_and_generated_position():
    """Decode work is per-resident KV shape, not ``mean_input, position=1``."""
    observed: list[tuple[tuple[float, float, float], ...]] = []

    def decode_step(items: list[SimDecodeItem]) -> float:
        observed.append(tuple(
            (item.input_tokens, item.generated_tokens, item.remaining_tokens)
            for item in items
        ))
        return 1e-3

    engine = SimEngine(
        name="generator.decode",
        cards=(0,),
        depth=2,
        role="decode",
        step_s=lambda decoding, chunk: 1e-3,
        decode_step_s=decode_step,
        call_overhead_s=0.0,
    )
    queries = [
        [SimRound(
            stage_idxs=(),
            call=SimCall(engine=0, eff_in=0, out=3, full_in=11),
        )],
        [SimRound(
            stage_idxs=(),
            call=SimCall(engine=0, eff_in=0, out=3, full_in=101),
        )],
    ]

    result = simulate(
        [], [engine], queries,
        n_cards=1, population=2,
        warmup_turnovers=1, window_turnovers=1,
    )

    assert result.converged
    flattened = [item for step in observed for item in step]
    assert {item[0] for item in flattened} == {11.0, 101.0}
    assert {item[1] for item in flattened} >= {0.0, 1.0, 2.0}
    assert any(
        {(item[0], item[1]) for item in step}
        == {(11.0, 0.0), (101.0, 0.0)}
        for step in observed
    )


def test_pd_prefill_prices_one_heterogeneous_multi_prompt_batch():
    """A PD prefill scheduler step retains prompt boundaries and APC suffixes."""
    observed: list[tuple[tuple[float, float], ...]] = []

    def prefill_batch(items: list[SimPrefillItem]) -> float:
        observed.append(tuple(
            (item.context_tokens, item.new_tokens) for item in items
        ))
        return 1e-3

    prefill = SimEngine(
        name="generator.prefill",
        cards=(0,),
        depth=2,
        role="prefill",
        step_s=lambda decoding, chunk: 1e-3,
        prefill_batch_s=prefill_batch,
        call_overhead_s=0.0,
    )
    decode = SimEngine(
        name="generator.decode",
        cards=(1,),
        depth=2,
        role="decode",
        step_s=lambda decoding, chunk: 1e-6,
        decode_step_s=lambda items: 1e-6,
        call_overhead_s=0.0,
    )
    queries = [
        [SimRound(
            stage_idxs=(),
            call=SimCall(
                engine=0, decode_engine=1,
                eff_in=5, full_in=20, out=1,
            ),
        )],
        [SimRound(
            stage_idxs=(),
            call=SimCall(
                engine=0, decode_engine=1,
                eff_in=9, full_in=40, out=1,
            ),
        )],
    ]

    result = simulate(
        [], [prefill, decode], queries,
        n_cards=2, population=2,
        warmup_turnovers=1, window_turnovers=1,
    )

    assert result.converged
    assert any(
        set(batch) == {(15.0, 5.0), (31.0, 9.0)}
        for batch in observed
    )


def test_builder_wires_shape_aware_engine_hooks_and_prices_short_apc_suffix():
    """Builder engines expose both hooks and never make a cached suffix free."""
    pipeline = {
        "system": {
            "batching": {"dynamic_timeout_s": 0.0},
            "communication": {"include_non_pd_comm": False},
        },
    }

    def bucketed_prefill(tokens, batch, tp=1, pp=1):
        del batch, tp, pp
        return ((max(1, int(tokens)) + 127) // 128) * 1e-3

    latency = CallLatency(
        prefill_s=bucketed_prefill,
        decode_s=lambda inp, out, batch, tp=1, pp=1: 2e-4,
    )
    trace = [[_generator_call(input_tokens=130, output_tokens=3)]]

    inputs, reason = build_sim_inputs(
        pipeline,
        SystemConfig(num_cores=16),
        trace,
        [["generator_prefill", "generator_decode"]],
        stage_batch_profiles={},  # Explicit analytical-only fixture.
        batch_size_request=2,
        batch_size_decode=2,
        latency=latency,
    )

    assert reason is None
    _stages, engines, _queries, *_ = inputs
    assert len(engines) == 1
    engine = engines[0]
    assert engine.prefill_batch_s is not None
    assert engine.decode_step_s is not None

    # 129 and 130 occupy the same public 128-token GenZ bucket.  The builder
    # must use the adjacent-bucket model slope for this one-token APC suffix,
    # rather than turning the scheduler step into zero-time work.
    suffix_s = engine.prefill_batch_s([
        SimPrefillItem(context_tokens=129, new_tokens=1),
    ])
    assert suffix_s > 0.0
    assert suffix_s == pytest.approx(1e-3 / 128)


def test_logical_engine_admission_capacity_matches_public_runtime_policy():
    assert _llm_admission_capacity(1) == 2
    assert _llm_admission_capacity(32) == 39
    assert _llm_admission_capacity(128) == 154


def test_pd_logical_gate_is_fcfs_and_held_through_transfer_decode_overhead():
    """A PD permit is not released when prefill or KV transfer finishes."""
    observed_prefill_contexts = []

    def prefill_batch(items):
        observed_prefill_contexts.append(tuple(
            item.context_tokens for item in items
        ))
        return 1.0

    prefill = SimEngine(
        name="generator.prefill",
        cards=(0,),
        depth=2,
        role="prefill",
        step_s=lambda decoding, chunk: 1.0,
        prefill_batch_s=prefill_batch,
        call_overhead_s=0.0,
    )
    decode = SimEngine(
        name="generator.decode",
        cards=(1,),
        depth=2,
        role="decode",
        step_s=lambda decoding, chunk: 1.0,
        decode_step_s=lambda items: 1.0,
        call_overhead_s=1.0,
    )
    transfer = SimTransfer(
        name="generator_kv",
        resource="gpu_gpu",
        payload_bytes=1.0,
        bandwidth_bytes_s=1.0,
    )
    queries = [
        [SimRound(
            stage_idxs=(),
            call=SimCall(
                engine=0,
                decode_engine=1,
                eff_in=1,
                full_in=full_input,
                out=1,
                transfer=transfer,
                admission_gate="generator",
                admission_capacity=1,
            ),
        )]
        for full_input in (11, 21)
    ]

    result = simulate(
        [], [prefill, decode], queries,
        n_cards=2,
        population=2,
        warmup_turnovers=1,
        window_turnovers=3,
    )

    assert result.converged
    # One permit serializes the complete 1s prefill + 1s transfer + 1s decode
    # + 1s completion overhead lifecycle.
    assert result.qps == pytest.approx(0.25)
    # Both calls arrive at t=0; strict FCFS promotion alternates the two
    # closed-loop slots rather than allowing the just-completed slot to barge.
    # Each slot also advances through the coprime trace rotation, so r0 sees
    # 10,20,10 while r1 sees 20,10,20.
    assert observed_prefill_contexts[:6] == [
        (10.0,), (20.0,), (20.0,), (10.0,), (10.0,), (20.0,),
    ]


def test_builder_reranker_ops_preserve_heterogeneous_member_tokens():
    """The builder passes each quality row to the public reranker tile policy."""
    observed: list[tuple[int, ...]] = []

    def forward_tiles(input_tokens, request_batch_size=None):
        assert request_batch_size == 2
        members = tuple(int(tokens) for tokens in input_tokens)
        observed.append(members)
        return (sum(tokens * tokens for tokens in members) / 1e6,)

    latency = CallLatency(
        prefill_s=lambda tokens, batch, tp=1, pp=1: 1e-3,
        decode_s=lambda inp, out, batch, tp=1, pp=1: 1e-4,
        rerank_s=lambda input_tokens, batch: 1e-2,
        rerank_forward_tiles_s=forward_tiles,
    )
    pipeline = {
        "system": {
            "batching": {"dynamic_timeout_s": 0.0},
            "communication": {"include_non_pd_comm": False},
        },
    }
    trace = [[
        {
            "stage": "passage_reranker",
            "input_tokens": 11,
            "output_tokens": 0,
        },
        _generator_call(input_tokens=100, output_tokens=3),
    ]]

    inputs, reason = build_sim_inputs(
        pipeline,
        SystemConfig(num_cores=16),
        trace,
        [
            ["passage_reranker"],
            ["generator_prefill", "generator_decode"],
        ],
        stage_batch_profiles={},  # Explicit analytical-only fixture.
        batch_size_request=2,
        batch_size_decode=2,
        latency=latency,
    )

    assert reason is None
    stages, _engines, _queries, *_ = inputs
    reranker = next(stage for stage in stages if stage.name == "passage_reranker")
    assert reranker.batch_ops is not None
    ops = tuple(reranker.batch_ops([
        SimStageItem(stage="passage_reranker", input_tokens=11),
        SimStageItem(stage="passage_reranker", input_tokens=101),
    ]))

    assert observed == [(11, 101)]
    assert (56, 56) not in observed
    assert [(op.resource, op.name) for op in ops] == [
        ("gpu", "passage_reranker.forward[0]"),
    ]
    assert ops[0].service_s == pytest.approx((11**2 + 101**2) / 1e6)


def test_builder_reranker_dp_uses_contiguous_balanced_row_shards():
    """Two replicas receive adjacent rows, matching flattened runtime order."""
    observed: list[tuple[tuple[int, ...], int]] = []

    def forward_tiles(input_tokens, request_batch_size=None):
        shard = tuple(int(tokens) for tokens in input_tokens)
        observed.append((shard, int(request_batch_size)))
        return (max(shard) / 1e6,)

    latency = CallLatency(
        prefill_s=lambda tokens, batch, tp=1, pp=1: 1e-3,
        decode_s=lambda inp, out, batch, tp=1, pp=1: 1e-4,
        rerank_s=lambda input_tokens, batch: 1e-2,
        rerank_forward_tiles_s=forward_tiles,
    )
    layout = {
        "available_devices": ["cuda:0", "cuda:1", "cuda:2"],
        "resource_groups": [
            {
                "devices": ["cuda:0", "cuda:1"],
                "performance_stages": ["passage_reranker"],
            },
            {
                "devices": ["cuda:2"],
                "performance_stages": [
                    "generator_prefill", "generator_decode",
                ],
            },
        ],
        "performance_stages": {
            "passage_reranker": {
                "devices": ["cuda:0", "cuda:1"],
                "num_chips": 2,
            },
            "generator_prefill": {
                "devices": ["cuda:2"], "num_chips": 1,
                "engine": "generator", "role": "prefill",
                "tp": 1, "pp": 1, "replicas": 1,
                "replica_devices": [["cuda:2"]],
            },
            "generator_decode": {
                "devices": ["cuda:2"], "num_chips": 1,
                "engine": "generator", "role": "decode",
                "tp": 1, "pp": 1, "replicas": 1,
                "replica_devices": [["cuda:2"]],
            },
        },
        "engines": {
            "generator": {
                "pd_serving": "collocated_pd",
                "devices": ["cuda:2"],
                "tp": 1, "pp": 1, "replicas": 1,
                "replica_devices": [["cuda:2"]],
            },
        },
    }
    pipeline = {
        "system": {
            "batching": {"dynamic_timeout_s": 0.0},
            "communication": {"include_non_pd_comm": False},
        },
    }
    trace = [[
        {
            "stage": "passage_reranker",
            "input_tokens": 10,
            "output_tokens": 0,
        },
        _generator_call(input_tokens=100, output_tokens=3),
    ]]

    inputs, reason = build_sim_inputs(
        pipeline,
        SystemConfig(num_cores=16),
        trace,
        [
            ["passage_reranker"],
            ["generator_prefill", "generator_decode"],
        ],
        stage_batch_profiles={},  # Explicit analytical-only fixture.
        batch_size_request=4,
        batch_size_decode=4,
        candidate_row={"candidate_system_layout": layout},
        latency=latency,
    )

    assert reason is None
    stages, _engines, _queries, *_ = inputs
    reranker = next(stage for stage in stages if stage.name == "passage_reranker")
    assert reranker.devices == ("cuda:0", "cuda:1")
    assert reranker.batch_ops is not None
    reranker.batch_ops([
        SimStageItem(stage="passage_reranker", input_tokens=tokens)
        for tokens in (10, 11, 100, 101)
    ])

    assert observed == [
        ((10, 11), 4),
        ((100, 101), 4),
    ]
    assert ((10, 100), 4) not in observed
    assert ((11, 101), 4) not in observed


def test_qe_fanout_runs_k_ordered_retrieval_kernels_then_one_tail():
    """One worker visit executes K × (encode, search), then one fused tail."""
    emitted: list[tuple[tuple[str, str], ...]] = []

    def retrieval_ops(items: list[SimStageItem]):
        assert len(items) == 1
        assert items[0].multiplicity == 4
        ops = []
        for index in range(items[0].multiplicity):
            ops.extend((
                SimStageOp(
                    resource="gpu", service_s=0.01,
                    name=f"semantic_retrieval_encode[{index}]",
                ),
                SimStageOp(
                    resource="cpu", service_s=0.02,
                    name=f"semantic_retrieval_vectorsearch[{index}]",
                ),
            ))
        ops.append(SimStageOp(
            resource="cpu", service_s=0.03,
            name="semantic_retrieval_fused_tail",
        ))
        emitted.append(tuple((op.resource, op.name) for op in ops))
        return ops

    retrieval = SimStage(
        name="semantic_retrieval",
        service_s=lambda batch: 0.0,
        gpu_share=0.0,
        card=0,
        cards=(0,),
        batch_cap=1,
        timeout_s=0.0,
        batch_ops=retrieval_ops,
    )
    engine = SimEngine(
        name="generator.decode",
        cards=(1,),
        depth=1,
        role="decode",
        step_s=lambda decoding, chunk: 1e-6,
        call_overhead_s=0.0,
    )
    query = [SimRound(
        stage_idxs=(0,),
        stage_items=(SimStageItem(
            stage="semantic_retrieval", multiplicity=4,
        ),),
        call=SimCall(engine=0, eff_in=0, full_in=1, out=1),
    )]

    result = simulate(
        [retrieval], [engine], [query],
        n_cards=2, population=1,
        warmup_turnovers=1, window_turnovers=3,
        n_cpu_cores=8,
    )

    expected_names = tuple(
        item
        for index in range(4)
        for item in (
            ("gpu", f"semantic_retrieval_encode[{index}]"),
            ("cpu", f"semantic_retrieval_vectorsearch[{index}]"),
        )
    ) + (("cpu", "semantic_retrieval_fused_tail"),)
    assert emitted
    assert set(emitted) == {expected_names}
    assert result.converged
    # 4 * (10ms encode + 20ms search) + one 30ms tail.  A collapsed
    # single-retrieval visit would be far above this 6.67 qps mechanism bound.
    assert result.qps == pytest.approx(1.0 / 0.150001, rel=2e-3)


def _observed_rrf_call():
    return {
        "stage": "retrieval_fusion", "model_id": "rrf_pure",
        "input_tokens": 0, "output_tokens": 0,
        "input_profile": {"items": 4, "list_sizes": [2, 2], "unique_items": 3},
        "output_profile": {"items": 2, "unique_items": 2},
    }


def test_builder_requires_rrf_service_before_backend_chunking():
    """An observed fusion call cannot silently omit its separate service."""
    pipeline = {
        "system": {
            "batching": {"dynamic_timeout_s": 0.0},
            "retrieval": {"faiss_num_threads": 8},
            "communication": {"include_non_pd_comm": False},
        },
        "_token_stats": {
            "avg_chunk_tokens": 64,
            "prompt_template_tokens": 8,
            "avg_output_tokens": 8,
            "avg_query_tokens": 8,
            "bytes_per_token": 4.0,
        },
        "vectordb": [{"name": "vdb", "embedding_dim": 768}],
        "node_lines": [{
            "nodes": [{
                "stage": "query_expansion",
                "modules": [{"component": "multi_query_expansion"}],
            }, {
                "stage": "semantic_retrieval",
                "modules": [{"component": "vectordb", "vectordb": "vdb"}],
                "top_k": 4,
            }],
        }],
    }
    latency = CallLatency(
        encode_s=lambda batch: 0.01,
        retrieve_s=lambda batch: 0.05,
        retrieve_kernel_s=lambda batch: 0.02,
        retrieve_fused_overhead_s=lambda batch: 0.03,
        prefill_s=lambda tokens, batch, tp=1, pp=1: 1e-4,
        decode_s=lambda inp, out, batch, tp=1, pp=1: 1e-4,
    )
    trace = [[
        {
            "stage": "semantic_retrieval_encode",
            "input_tokens": 7,
            "output_tokens": 0,
        },
        {
            "stage": "semantic_retrieval_vectorsearch",
            "input_tokens": 7,
            "output_tokens": 0,
        },
        _observed_rrf_call(),
        {
            "stage": "generator",
            "input_tokens": 100,
            "output_tokens": 3,
        },
    ]]

    with pytest.raises(ValueError, match="requires an RRF service model"):
        build_sim_inputs(
            pipeline,
            SystemConfig(num_cores=16),
            trace,
            [
                ["semantic_retrieval_encode"],
                ["semantic_retrieval_vectorsearch"],
                ["generator_prefill", "generator_decode"],
            ],
            stage_batch_profiles={},  # Explicit analytical-only fixture.
            batch_size_request=2,
            batch_size_decode=2,
            latency=latency,
        )


def test_retrieval_partial_batch_pricing_requires_rrf_service():
    """An injected retrieval pricer must also supply the fusion service."""
    pipeline = {
        "system": {
            "batching": {"dynamic_timeout_s": 0.0},
            "retrieval": {"faiss_num_threads": 8},
            "communication": {"include_non_pd_comm": False},
        },
        "_token_stats": {
            "avg_chunk_tokens": 64,
            "prompt_template_tokens": 8,
            "avg_output_tokens": 8,
            "avg_query_tokens": 8,
            "bytes_per_token": 4.0,
        },
        "vectordb": [{"name": "vdb", "embedding_dim": 768}],
        "node_lines": [{
            "nodes": [{
                "stage": "query_expansion",
                "modules": [{"component": "multi_query_expansion"}],
            }, {
                "stage": "semantic_retrieval",
                "modules": [{"component": "vectordb", "vectordb": "vdb"}],
                "top_k": 4,
            }],
        }],
    }
    encode_batches: list[int] = []
    search_batches: list[int] = []
    tail_batches: list[int] = []

    def encode_s(batch: int) -> float:
        encode_batches.append(batch)
        return 0.01

    def search_s(batch: int) -> float:
        search_batches.append(batch)
        return 0.02

    def tail_s(batch: int) -> float:
        tail_batches.append(batch)
        return 0.03

    latency = CallLatency(
        encode_s=encode_s,
        retrieve_s=lambda batch: search_s(batch) + tail_s(batch),
        retrieve_kernel_s=search_s,
        retrieve_fused_overhead_s=tail_s,
        prefill_s=lambda tokens, batch, tp=1, pp=1: 1e-4,
        decode_s=lambda inp, out, batch, tp=1, pp=1: 1e-4,
    )
    trace = [[
        {"stage": "semantic_retrieval_encode", "input_tokens": 7},
        {
            "stage": "semantic_retrieval_vectorsearch",
            "input_tokens": 7,
        },
        _observed_rrf_call(),
        _generator_call(input_tokens=100, output_tokens=1, step_idx=3),
    ]]

    with pytest.raises(ValueError, match="requires an RRF service model"):
        build_sim_inputs(
            pipeline,
            SystemConfig(num_cores=16),
            trace,
            [
                ["semantic_retrieval_encode"],
                ["semantic_retrieval_vectorsearch"],
                ["generator_prefill", "generator_decode"],
            ],
            stage_batch_profiles={},  # Explicit analytical-only fixture.
            batch_size_request=8,
            batch_size_decode=8,
            latency=latency,
        )


def test_retrieval_pricing_requires_query_decompose_fusion_service():
    """Observed fusion requires a service regardless of the expansion method."""
    pipeline = {
        "system": {
            "batching": {"dynamic_timeout_s": 0.0},
            "retrieval": {"faiss_num_threads": 16},
            "communication": {"include_non_pd_comm": False},
        },
        "_token_stats": {
            "avg_chunk_tokens": 64,
            "prompt_template_tokens": 8,
            "avg_output_tokens": 8,
            "avg_query_tokens": 8,
            "bytes_per_token": 4.0,
        },
        "vectordb": [{"name": "vdb", "embedding_dim": 768}],
        "node_lines": [{
            "nodes": [{
                "stage": "query_expansion",
                "modules": [{"component": "query_decompose"}],
            }, {
                "stage": "semantic_retrieval",
                "modules": [{"component": "vectordb", "vectordb": "vdb"}],
                "top_k": 2,
            }],
        }],
    }
    seen = {"encode": [], "search": [], "backend": [], "outer": []}

    def encode_s(batch: int) -> float:
        seen["encode"].append(batch)
        return 0.01

    def search_s(batch: int) -> float:
        seen["search"].append(batch)
        return 0.02

    def backend_s(batch: int) -> float:
        seen["backend"].append(batch)
        return 0.05 + 0.003 * batch

    def outer_s(batch: int) -> float:
        seen["outer"].append(batch)
        return 0.001 * batch

    latency = CallLatency(
        encode_s=encode_s,
        retrieve_s=lambda batch: 0.02 + 0.05 + 0.004 * batch,
        retrieve_kernel_s=search_s,
        retrieve_backend_overhead_s=backend_s,
        retrieve_outer_overhead_s=outer_s,
        retrieve_fused_overhead_s=lambda batch: (
            backend_s(batch) + outer_s(batch)
        ),
        retrieval_fanout=2,
        prefill_s=lambda tokens, batch, tp=1, pp=1: 1e-4,
        decode_s=lambda inp, out, batch, tp=1, pp=1: 1e-4,
    )
    trace = [[
        {"stage": "semantic_retrieval_encode", "input_tokens": 7},
        {"stage": "semantic_retrieval_vectorsearch", "input_tokens": 7},
        _observed_rrf_call(),
        _generator_call(input_tokens=100, output_tokens=1, step_idx=3),
    ]]

    with pytest.raises(ValueError, match="requires an RRF service model"):
        build_sim_inputs(
            pipeline,
            SystemConfig(num_cores=16),
            trace,
            [
                ["semantic_retrieval_encode"],
                ["semantic_retrieval_vectorsearch"],
                ["generator_prefill", "generator_decode"],
            ],
            stage_batch_profiles={},  # Explicit analytical-only fixture.
            batch_size_request=16,
            batch_size_decode=16,
            latency=latency,
        )


def test_adjacent_retrieval_pairs_remain_two_worker_visits():
    """encode/search/encode/search must preserve R=2, not coalesce by name."""
    pipeline = {
        "system": {
            "batching": {"dynamic_timeout_s": 0.0},
            "communication": {"include_non_pd_comm": False},
        },
        "_token_stats": {
            "avg_chunk_tokens": 64,
            "prompt_template_tokens": 8,
            "avg_output_tokens": 8,
            "avg_query_tokens": 8,
            "bytes_per_token": 4.0,
        },
        "vectordb": [{"name": "vdb", "embedding_dim": 768}],
    }
    latency = CallLatency(
        encode_s=lambda batch: 0.01,
        retrieve_s=lambda batch: 0.02,
        retrieve_kernel_s=lambda batch: 0.02,
        retrieve_fused_overhead_s=lambda batch: 0.01,
        prefill_s=lambda tokens, batch, tp=1, pp=1: 1e-4,
        decode_s=lambda inp, out, batch, tp=1, pp=1: 1e-4,
    )
    trace = [[
        {"stage": "semantic_retrieval_encode", "input_tokens": 7},
        {
            "stage": "semantic_retrieval_vectorsearch",
            "input_tokens": 7,
        },
        {"stage": "semantic_retrieval_encode", "input_tokens": 9},
        {
            "stage": "semantic_retrieval_vectorsearch",
            "input_tokens": 9,
        },
        _generator_call(input_tokens=100, output_tokens=1, step_idx=4),
    ]]

    inputs, reason = build_sim_inputs(
        pipeline,
        SystemConfig(num_cores=16),
        trace,
        [
            ["semantic_retrieval_encode"],
            ["semantic_retrieval_vectorsearch"],
            ["generator_prefill", "generator_decode"],
        ],
        stage_batch_profiles={},  # Explicit analytical-only fixture.
        batch_size_request=8,
        batch_size_decode=8,
        latency=latency,
    )

    assert reason is None
    _stages, _engines, queries, *_ = inputs
    first_round = queries[0][0]
    assert len(first_round.stage_idxs) == 2
    assert len(first_round.stage_items) == 2
    assert [
        [call["stage"] for call in item.calls]
        for item in first_round.stage_items
    ] == [
        ["semantic_retrieval_encode", "semantic_retrieval_vectorsearch"],
        ["semantic_retrieval_encode", "semantic_retrieval_vectorsearch"],
    ]


def test_retrieval_search_without_immediate_encode_is_rejected():
    """The internal fused-worker pair is structural, not a stage-name guess."""
    latency = CallLatency(
        encode_s=lambda batch: 0.01,
        retrieve_s=lambda batch: 0.02,
        prefill_s=lambda tokens, batch, tp=1, pp=1: 1e-4,
        decode_s=lambda inp, out, batch, tp=1, pp=1: 1e-4,
    )
    trace = [[
        {
            "stage": "semantic_retrieval_vectorsearch",
            "input_tokens": 7,
        },
        _generator_call(input_tokens=100, output_tokens=1, step_idx=1),
    ]]

    with pytest.raises(ValueError, match="no immediately preceding encode"):
        build_sim_inputs(
            {
                "system": {
                    "batching": {"dynamic_timeout_s": 0.0},
                    "communication": {"include_non_pd_comm": False},
                },
            },
            SystemConfig(num_cores=16),
            trace,
            [
                ["semantic_retrieval_encode"],
                ["semantic_retrieval_vectorsearch"],
                ["generator_prefill", "generator_decode"],
            ],
            stage_batch_profiles={},  # Explicit analytical-only fixture.
            batch_size_request=8,
            batch_size_decode=8,
            latency=latency,
        )


def test_two_light_stage_gpu_ops_on_one_card_are_serialized():
    """Concrete cards are exclusive lanes for non-preemptive stage kernels."""

    def stage(name: str) -> SimStage:
        return SimStage(
            name=name,
            service_s=lambda batch: 0.0,
            gpu_share=0.0,
            card=0,
            cards=(0,),
            batch_cap=1,
            timeout_s=0.0,
            batch_ops=lambda items, _name=name: [SimStageOp(
                resource="gpu", service_s=0.1, name=f"{_name}.kernel",
            )],
        )

    engine = SimEngine(
        name="generator.decode",
        cards=(1,),
        depth=2,
        role="decode",
        step_s=lambda decoding, chunk: 1e-6,
        call_overhead_s=0.0,
    )
    queries = [
        [SimRound(
            stage_idxs=(stage_index,),
            stage_items=(SimStageItem(stage=f"stage-{stage_index}"),),
            call=SimCall(engine=0, eff_in=0, full_in=1, out=1),
        )]
        for stage_index in (0, 1)
    ]

    result = simulate(
        [stage("stage-0"), stage("stage-1")],
        [engine],
        queries,
        n_cards=2, population=2,
        warmup_turnovers=1, window_turnovers=5,
    )

    assert result.converged
    # Each completed query consumes one 100ms kernel on the same physical
    # card.  If the two workers launched concurrently, this would approach
    # 20 qps; a single exclusive lane is bounded at 10 qps.
    assert result.qps == pytest.approx(10.0, rel=2e-3)


def test_resident_engine_makes_no_progress_during_light_stage_gpu_burst():
    """A non-preemptive stage kernel owns the card for its whole burst."""
    card = _Card()
    spec = SimEngine(
        name="generator.decode",
        cards=(0,),
        depth=1,
        role="decode",
        step_s=lambda decoding, chunk: 1.0,
        call_overhead_s=0.0,
    )
    engine = _Engine(spec, [card])
    engine.admit(
        7,
        SimCall(engine=0, eff_in=0, full_in=100, out=3),
    )
    before = deepcopy(engine.decoding)

    card.stage_busy = 1
    assert engine.share() == 0.0
    assert engine.advance(5.0) == []
    assert engine.decoding == before

    card.stage_busy = 0
    assert engine.share() == 1.0
    engine.advance(1.0)
    assert engine.decoding != before


def test_gpu_arbiter_does_not_preempt_an_inflight_engine_step():
    """A stage arriving mid-step waits for that non-preemptive step to end."""
    arbiter = _GpuArbiter(1)

    engine, started = arbiter.submit(
        cards=(0,), service_s=1.0, owner="engine",
        now=0.0, name="decode-step-0",
    )
    assert started == (engine,)
    assert engine.started_at == 0.0

    stage, started = arbiter.submit(
        cards=(0,), service_s=0.25, owner="stage",
        now=0.2, name="reranker-kernel",
    )
    assert started == ()
    assert stage.started_at is None

    assert arbiter.release(engine, now=1.0) == (stage,)
    assert stage.started_at == 1.0


def test_gpu_arbiter_runs_queued_stage_before_next_engine_step():
    """Per-card FCFS tickets prevent an engine from step-by-step barging."""
    arbiter = _GpuArbiter(1)
    engine_0, started = arbiter.submit(
        cards=(0,), service_s=1.0, owner="engine",
        now=0.0, name="decode-step-0",
    )
    assert started == (engine_0,)
    stage, started = arbiter.submit(
        cards=(0,), service_s=0.5, owner="stage",
        now=0.2, name="encode-kernel",
    )
    assert started == ()
    engine_1, started = arbiter.submit(
        cards=(0,), service_s=1.0, owner="engine",
        now=0.8, name="decode-step-1",
    )
    assert started == ()

    assert arbiter.release(engine_0, now=1.0) == (stage,)
    assert stage.started_at == 1.0
    assert engine_1.started_at is None
    assert arbiter.release(stage, now=1.5) == (engine_1,)
    assert engine_1.started_at == 1.5


def test_gpu_arbiter_acquires_tp2_cards_atomically():
    """A TP op cannot occupy one rank while another requested rank is busy."""
    arbiter = _GpuArbiter(2)
    blocker, started = arbiter.submit(
        cards=(1,), service_s=1.0, owner="stage",
        now=0.0, name="card-1-kernel",
    )
    assert started == (blocker,)

    tp2, started = arbiter.submit(
        cards=(0, 1), service_s=2.0, owner="engine",
        now=0.1, name="tp2-decode-step",
    )
    assert started == ()
    assert tp2.started_at is None

    # The TP2 ticket is already at card 0's head, so a later card-0-only op
    # cannot use that rank and violate atomic model-parallel scheduling.
    follower, started = arbiter.submit(
        cards=(0,), service_s=0.1, owner="stage",
        now=0.2, name="card-0-follower",
    )
    assert started == ()
    assert follower.started_at is None

    assert arbiter.release(blocker, now=1.0) == (tp2,)
    assert tp2.started_at == 1.0
    assert tp2.cards == (0, 1)
    assert follower.started_at is None


def test_gpu_arbiter_starts_disjoint_card_ops_concurrently():
    """The arbiter serializes conflicts, not the entire GPU box."""
    arbiter = _GpuArbiter(2)
    left, started = arbiter.submit(
        cards=(0,), service_s=1.0, owner="engine",
        now=0.0, name="left-engine-step",
    )
    assert started == (left,)
    right, started = arbiter.submit(
        cards=(1,), service_s=1.0, owner="stage",
        now=0.0, name="right-stage-kernel",
    )
    assert started == (right,)
    assert left.started_at == right.started_at == 0.0


def test_light_stage_batch_receives_each_member_quality_token_shape():
    """Dynamic batching must not replace heterogeneous calls with one mean."""
    observed: list[tuple[int, ...]] = []

    def shaped_service(items: list[SimStageItem]) -> float:
        tokens = tuple(sorted(item.input_tokens for item in items))
        observed.append(tokens)
        # A padding-sensitive kernel is governed by the largest batch member.
        return max(tokens) / 1000.0

    reranker = SimStage(
        name="passage_reranker",
        service_s=lambda batch: 0.0,
        gpu_share=1.0,
        card=0,
        cards=(0,),
        batch_cap=2,
        timeout_s=0.0,
        batch_service_s=shaped_service,
        batch_gpu_service_s=shaped_service,
    )
    engine = SimEngine(
        name="generator.decode",
        cards=(1,),
        depth=2,
        role="decode",
        step_s=lambda decoding, chunk: 1e-6,
        call_overhead_s=0.0,
    )
    queries = [
        [SimRound(
            stage_idxs=(0,),
            stage_items=(SimStageItem(
                stage="passage_reranker", input_tokens=tokens,
            ),),
            call=SimCall(engine=0, eff_in=0, full_in=1, out=1),
        )]
        for tokens in (11, 101)
    ]

    result = simulate(
        [reranker], [engine], queries,
        n_cards=2, population=2,
        warmup_turnovers=1, window_turnovers=3,
    )

    assert result.converged
    assert observed
    assert set(observed) == {(11, 101)}
    assert (56, 56) not in observed
