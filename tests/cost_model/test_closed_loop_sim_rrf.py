"""Explicit logical fusion retains the runtime's serialized retrieval worker."""

from copy import deepcopy

import pytest

from rag_stack.cost_model.closed_loop_sim import (
    SimRound,
    SimStage,
    SimStageItem,
    build_sim_inputs,
    simulate,
)
from rag_stack.cost_model.engines import CallLatency
from rag_stack.types import SystemConfig
from rag_stack.rag_ir.execution_path import compile_execution_paths


def _pipeline(*, expanded=True, communication=False):
    nodes = [{
        "stage": "semantic_retrieval",
        "top_k": 4,
        "modules": [{"component": "vectordb", "vectordb": "vdb"}],
    }]
    if expanded:
        nodes.insert(0, {
            "stage": "query_expansion",
            "modules": [{"component": "multi_query_expansion"}],
        })
    return {
        "node_lines": [{"nodes": nodes}],
        "vectordb": [{"name": "vdb", "embedding_dim": 8}],
        "_token_stats": {
            "avg_chunk_tokens": 16,
            "avg_query_tokens": 4,
            "avg_output_tokens": 2,
            "prompt_template_tokens": 8,
            "bytes_per_token": 4.0,
        },
        "system": {
            "batching": {"dynamic_timeout_s": 0.0},
            "retrieval": {"faiss_num_threads": 1},
            "communication": {"include_non_pd_comm": communication},
        },
    }


def _call(stage, **kwargs):
    return {
        "stage": stage,
        "input_tokens": 8,
        "output_tokens": 1 if stage == "generator" else 0,
        **kwargs,
    }


def _trace(*, visits=1, reranker=False, fusion=True):
    calls = []
    for _ in range(visits):
        calls.extend([
            _call("semantic_retrieval_encode"),
            _call("semantic_retrieval_vectorsearch"),
        ])
        if fusion:
            calls.append(_call(
                "retrieval_fusion", model_id="rrf_pure",
                input_profile={"items": 16, "list_sizes": [4, 4, 4, 4]},
                output_profile={"items": 4, "unique_items": 4},
            ))
    if reranker:
        calls.append(_call("passage_reranker", input_bytes=1234))
    calls.append(_call("generator", input_bytes=512))
    return [[{**call, "step_idx": i} for i, call in enumerate(calls)]]


def _latency(*, rrf_s=None, reranker=False):
    return CallLatency(
        encode_s=lambda batch: 0.001,
        retrieve_s=lambda batch: 0.002,
        retrieve_kernel_s=lambda batch: 0.002,
        rrf_s=rrf_s,
        rerank_s=(lambda tokens, batch: 0.003) if reranker else None,
        prefill_s=lambda tokens, batch, tp=1, pp=1: 0.0001,
        decode_s=lambda inp, out, batch, tp=1, pp=1: 0.0001,
    )


def _build(pipeline, trace, latency, *, servers=1):
    inputs, reason = build_sim_inputs(
        pipeline,
        SystemConfig(num_cores=2),
        trace,
        [
            ["semantic_retrieval_encode", "passage_reranker"],
            ["generator_prefill", "generator_decode"],
        ],
        batch_size_request=2,
        batch_size_decode=2,
        prefix_caching=False,
        latency=latency,
        candidate_row={"num_retrieval_servers": servers},
        stage_batch_profiles={},
    )
    assert reason is None
    assert inputs is not None
    return inputs


def test_fusion_is_one_cpu_operator_inside_each_retrieval_worker_visit():
    observed_batches = []

    def rrf_s(batch):
        observed_batches.append(batch)
        return 0.004 * batch

    trace = _trace(visits=2)
    before = deepcopy(trace)
    stages, _, queries, *_ = _build(_pipeline(), trace, _latency(rrf_s=rrf_s))

    assert trace == before
    assert [stages[i].name for i in queries[0][0].stage_idxs] == [
        "semantic_retrieval", "semantic_retrieval",
    ]
    assert "retrieval_fusion" not in {stage.name for stage in stages}
    items = queries[0][0].stage_items
    assert len(items) == 2
    assert all([c["stage"] for c in item.calls] == [
        "semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "retrieval_fusion",
    ] for item in items)

    retrieval = next(stage for stage in stages if stage.name == "semantic_retrieval")
    ops = retrieval.batch_ops(items)
    assert len(ops) == 9  # Four encode/search chunks, then one row-local fusion batch.
    fusion = ops[-1]
    assert fusion.name == "retrieval_fusion"
    assert fusion.resource == "cpu" and fusion.host_width == 1
    assert fusion.service_s == pytest.approx(0.008)
    assert observed_batches == [2, 2]  # Representative cost, then this exact batch.
    assert sum(op.service_s for op in ops) == pytest.approx(0.020)
    assert retrieval.service_s(2) == pytest.approx(0.020)
    assert retrieval.gpu_service_s(2) == pytest.approx(0.004)


def test_single_list_retrieval_does_not_add_fusion():
    stages, _, queries, *_ = _build(
        _pipeline(expanded=False), _trace(fusion=False), _latency(),
    )
    assert "retrieval_fusion" not in {stage.name for stage in stages}
    assert [stages[i].name for i in queries[0][0].stage_idxs] == [
        "semantic_retrieval",
    ]


def test_declared_single_list_and_heterogeneous_fusion_use_observed_profiles():
    trace = [_trace()[0], _trace()[0]]
    for calls, sizes, output in ((trace[0], [5], 3), (trace[1], [2, 4], 4)):
        fusion = next(call for call in calls if call["stage"] == "retrieval_fusion")
        fusion["input_profile"] = {"items": sum(sizes), "list_sizes": sizes}
        fusion["output_profile"] = {"items": output, "unique_items": output}
    seen = []
    latency = _latency()
    def service(call, batch):
        seen.append((tuple(call["input_profile"]["list_sizes"]), batch))
        return 0.001 * call["input_profile"]["items"] * batch
    latency.rrf_call_s = service
    stages, _, queries, *_ = _build(_pipeline(expanded=False), trace, latency)
    seen.clear()
    items = [query[0].stage_items[0] for query in queries]
    fusion_ops = [op for op in stages[0].batch_ops(items) if op.name == "retrieval_fusion"]
    assert len(fusion_ops) == 1
    assert fusion_ops[0].service_s == pytest.approx(0.011)
    assert seen == [((5,), 1), ((2, 4), 1)]


def test_configured_expansion_without_recorded_fusion_never_charges_rrf():
    latency = _latency()
    def forbidden(*args):
        raise AssertionError("unobserved fusion was charged")
    latency.rrf_call_s = forbidden
    stages, _, queries, *_ = _build(_pipeline(), _trace(fusion=False), latency)
    assert all(op.name != "retrieval_fusion"
               for op in stages[0].batch_ops(queries[0][0].stage_items))


def test_nonlocal_fusion_requires_an_explicit_physical_worker_policy():
    trace = _trace()
    trace[0][2:2] = [
        _call("retrieval_fusion", model_id="rrf"),
        _call("retrieval_fusion", model_id="rrf"),
    ]
    for index, call in enumerate(trace[0]):
        call["step_idx"] = index
    with pytest.raises(ValueError, match="nonlocal fusion"):
        _build(_pipeline(), trace, _latency(rrf_s=lambda batch: 0.004 * batch))


def test_precompiled_fusion_is_folded_once_without_a_second_queue():
    pipeline = _pipeline()
    pipeline["system"]["batching"]["dynamic_timeout_s"] = 0.037
    trace = compile_execution_paths(_trace(visits=2), retrieval_fanout=4)
    stages, _, queries, *_ = _build(
        pipeline, trace, _latency(rrf_s=lambda batch: 0.004 * batch),
    )
    assert len(stages) == 1
    retrieval = stages[0]
    assert retrieval.name == "semantic_retrieval"
    assert retrieval.timeout_s == pytest.approx(0.037)
    assert queries[0][0].stage_idxs == (0, 0)
    for item in queries[0][0].stage_items:
        assert sum(c["stage"] == "retrieval_fusion" for c in item.calls) == 1
        assert sum(op.name == "retrieval_fusion" for op in retrieval.batch_ops([item])) == 1


def test_search_server_count_does_not_divide_local_fusion_service():
    def costs(servers):
        stages, _, queries, *_ = _build(
            _pipeline(), _trace(), _latency(rrf_s=lambda batch: 0.004 * batch),
            servers=servers,
        )
        return stages[0].batch_ops(queries[0][0].stage_items)

    one, four = costs(1), costs(4)
    assert one[-1].name == four[-1].name == "retrieval_fusion"
    assert one[-1].service_s == four[-1].service_s == pytest.approx(0.004)
    search_one = sum(op.service_s for op in one if "vectorsearch" in op.name)
    search_four = sum(op.service_s for op in four if "vectorsearch" in op.name)
    assert search_four == pytest.approx(search_one / 4)


def test_fusion_without_service_model_fails_instead_of_becoming_free():
    with pytest.raises(ValueError, match="requires an RRF service model"):
        _build(_pipeline(), _trace(), _latency())


def test_fusion_does_not_add_host_transfer_after_cpu_search():
    stages, _, queries, *_ = _build(
        _pipeline(communication=True),
        _trace(reranker=True),
        _latency(rrf_s=lambda batch: 0.004 * batch, reranker=True),
    )
    route = [stages[i].name for i in queries[0][0].stage_idxs]
    assert route == ["semantic_retrieval", "passage_reranker"]
    transfers = [
        transfer
        for round_ in queries[0]
        for group in round_.boundary_transfers.values()
        for transfer in group
    ]
    assert not any("vectorsearch_to_retrieval_fusion" in t.name for t in transfers)
    outgoing = [t for t in transfers if t.name == "retrieval_fusion_to_passage_reranker"]
    assert len(outgoing) == 1
    assert outgoing[0].payload_bytes == 1234


def test_worker_stays_busy_through_fusion_even_when_more_cpu_cores_are_available():
    stages, _, queries, n_cards, *_ = _build(
        _pipeline(), _trace(), _latency(rrf_s=lambda batch: 0.004 * batch),
    )
    assert len(stages) == 1
    worker_only = [[SimRound(
        stage_idxs=(0,), call=None, stage_items=queries[0][0].stage_items,
    )]]

    def run(cores):
        return simulate(
            stages, [], worker_only,
            n_cards=n_cards,
            population=8,
            n_cpu_cores=cores,
            warmup_turnovers=2,
            window_turnovers=8,
        )

    one_core, many_cores = run(1), run(8)
    assert one_core.converged and many_cores.converged
    # Batch two: four encode/search chunks (.012 s), then two RRF rows (.008 s).
    # One physical worker cannot overlap its next search batch with this fusion.
    expected_qps = 2 / 0.020
    assert one_core.qps == pytest.approx(expected_qps, rel=0.02)
    assert many_cores.qps == pytest.approx(one_core.qps)
    assert one_core.stage_batch_mean == {"semantic_retrieval": 2}
