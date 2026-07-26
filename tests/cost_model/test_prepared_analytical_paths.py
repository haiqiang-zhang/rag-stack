"""Validated cohort ownership and reuse across analytical candidate walks."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.cost_model import engines, workload


def _traces():
    calls = [
        {"stage": "query_expansion", "input_tokens": 8, "output_tokens": 3},
        {"stage": "semantic_retrieval_encode", "input_tokens": 8, "output_tokens": 0},
        {"stage": "semantic_retrieval_vectorsearch", "input_tokens": 8, "output_tokens": 80},
        {"stage": "retrieval_fusion", "model_id": "rrf_pure", "input_tokens": 0,
         "output_tokens": 0, "input_profile": {"items": 9, "list_sizes": [5, 4]},
         "output_profile": {"items": 3, "unique_items": 3}},
        {"stage": "passage_reranker", "input_tokens": 80, "output_tokens": 40},
        {"stage": "generator", "input_tokens": 40, "output_tokens": 4},
        {"stage": "generator", "input_tokens": 52, "output_tokens": 2},
    ]
    first = [{**call, "step_idx": 3 * index + 1} for index, call in enumerate(calls)]
    second = deepcopy(first)
    second[3]["input_profile"] = {"items": 5, "list_sizes": [3, 2]}
    second[-1]["input_tokens"] = 68
    # The compatibility path retains partial queries for reranker padding,
    # even though completed-query metrics subsequently exclude this row.
    partial = [{"stage": "passage_reranker", "input_tokens": 400,
                "output_tokens": 0, "step_idx": 12}]
    return [first, second, partial]


def _latency():
    return engines.CallLatency(
        prefill_s=lambda tokens, batch, tp, pp: tokens * batch * 0.0001 / (tp * pp),
        decode_s=lambda tokens, position, batch, tp, pp: (tokens + position) * batch * 0.00001 / (tp * pp),
        encode_s=lambda batch: batch * 0.001,
        retrieve_s=lambda batch: batch * 0.003,
        retrieve_kernel_s=lambda batch: batch * 0.002,
        retrieve_backend_overhead_s=lambda batch: batch * 0.001,
        retrieval_fanout=4,
        rrf_call_s=lambda call, batch: call["input_profile"]["items"] * batch * 0.00001,
        rerank_s=lambda tokens, batch: tokens * batch * 0.0002,
        rerank_fanout=2,
        rerank_forward_batch=lambda batch: batch * 2,
    )


@pytest.mark.parametrize("batch,decode_batch", [(1, 2), (3, 6)])
@pytest.mark.parametrize("overlap,prefix,collocation", [
    (False, False, False), (False, True, True), (True, True, True),
])
def test_prepared_walk_preserves_all_metrics_and_does_not_mutate_calls(
    batch, decode_batch, overlap, prefix, collocation,
):
    traces = _traces()
    original = deepcopy(traces)
    prepared = workload.PreparedExecutionPaths(traces)
    prepared_original = deepcopy(prepared.paths)
    options = dict(batch=batch, decode_batch=decode_batch,
                   overlap_aux_generates=overlap, prefix_caching=prefix,
                   gpu_collocated_with_gen=collocation)

    expected = engines._compute_chain_latency(traces, _latency(), **options)
    actual = engines._compute_chain_latency(
        prepared.paths, _latency(), prepared_paths=prepared, **options,
    )
    assert actual == expected
    assert actual.svc_fusion > 0
    assert actual.retrieval_calls_per_query == 1
    assert traces == original
    assert prepared.paths == prepared_original

    traces[0][3]["input_profile"]["list_sizes"].append(99)
    traces[0][-1]["input_tokens"] = 100000
    assert prepared.paths == prepared_original
    assert engines._compute_chain_latency(
        prepared.paths, _latency(), prepared_paths=prepared, **options,
    ) == actual


@pytest.mark.parametrize("invalid", [
    [[{"input_tokens": 1}]],
    [[{"stage": "retrieval_fusion", "input_profile": {"items": -1}}]],
])
def test_raw_walk_still_rejects_invalid_calls(invalid):
    with pytest.raises((ValueError, TypeError)):
        engines._compute_chain_latency(invalid, _latency())


def test_prepared_walk_rejects_an_unrelated_trace_or_untyped_carrier():
    prepared = workload.PreparedExecutionPaths(_traces())
    with pytest.raises((ValueError, TypeError)):
        engines._compute_chain_latency(deepcopy(prepared.paths), _latency(), prepared_paths=prepared)
    with pytest.raises((ValueError, TypeError)):
        engines._compute_chain_latency(prepared.paths, _latency(), prepared_paths=True)


def _hardware():
    return SimpleNamespace(system_hardware="unit_prepared_analysis", gpu="unit_gpu",
                           rag_cm={"calibration_mode": "uncalibrated_spec"})


def _candidate_frame():
    return pd.DataFrame({
        "batch_size_request": [1, 2, 4, 4],
        "batch_size_decode": [2, 4, 8, 8],
        "dynamic_batch_timeout_s": [0.001, 0.001, 0.001, 0.01],
        "num_chips_generator_prefill": [1, 1, 1, 1],
        "num_chips_generator_decode": [1, 1, 1, 1],
    })


def test_analysis_prepares_once_for_distinct_walks_and_borrows_explicit_preparation(monkeypatch):
    prepare_calls = []
    compile_paths = workload.compile_execution_paths

    def prepare(paths, **kwargs):
        prepare_calls.append(paths)
        return compile_paths(paths, **kwargs)

    monkeypatch.setattr(workload, "compile_execution_paths", prepare)
    monkeypatch.setattr(engines, "build_call_latency", lambda *_args: _latency())
    walks = []
    real_walk = engines._compute_chain_latency

    def walk(paths, latency, **kwargs):
        walks.append((paths, kwargs["prepared_paths"]))
        return real_walk(paths, latency, **kwargs)

    monkeypatch.setattr(engines, "_compute_chain_latency", walk)
    frame = _candidate_frame()
    original_frame = frame.copy(deep=True)
    traces = _traces()
    original_traces = deepcopy(traces)
    config = {"system": {"rag_cm": {"calibration_mode": "uncalibrated_spec"}}}
    original_config = deepcopy(config)
    engine = engines.DynamicEngine(SimpleNamespace(is_agentic=True))
    options = dict(pipeline_config=config, hardware=_hardware(),
                   fixed_placement=SimpleNamespace(collocation_strategy=[]))

    raw_result = engine.analyze(frame, {}, traces=traces, **options)
    assert len(prepare_calls) == 1
    assert len(walks) == 3
    assert all(paths is carrier.paths for paths, carrier in walks)
    assert len({id(carrier) for _, carrier in walks}) == 1
    assert traces == original_traces
    assert config == original_config
    pd.testing.assert_frame_equal(frame, original_frame)

    prepared = workload.PreparedExecutionPaths(traces)
    prepare_calls.clear()
    walks.clear()
    prepared_result = engine.analyze(
        frame, {}, traces=prepared.paths, prepared_paths=prepared, **options,
    )
    assert prepare_calls == []
    assert len(walks) == 3
    assert all(carrier is prepared for _, carrier in walks)
    pd.testing.assert_frame_equal(prepared_result, raw_result)


def test_compiled_analysis_prepares_the_graph_projection_once(monkeypatch):
    projection = _traces()
    projection_calls = []
    prepares = []
    compile_paths = workload.compile_execution_paths

    def prepare(paths, **kwargs):
        prepares.append(paths)
        return compile_paths(paths, **kwargs)

    def analysis_paths():
        projection_calls.append(True)
        return projection

    monkeypatch.setattr(workload, "compile_execution_paths", prepare)
    config = {"system": {"rag_cm": {"calibration_mode": "uncalibrated_spec"}}}
    plan = SimpleNamespace(analysis_paths=analysis_paths,
                           validate_candidate=lambda _row: None,
                           pipeline_config=config, hardware=_hardware(), latency=_latency())
    engine = engines.DynamicEngine(SimpleNamespace(is_agentic=True))
    result = engine.analyze(
        _candidate_frame().iloc[:1], {}, traces=[[]], pipeline_config={},
        hardware=_hardware(), compiled_plan=plan,
    )
    assert projection_calls == [True]
    assert len(prepares) == 1 and prepares[0] is projection
    assert result["svc_fusion_s"].iloc[0] > 0
    assert result["gen_out_tokens_per_query"].iloc[0] == 6
