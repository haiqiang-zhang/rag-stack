"""Reject unsupported downstream passage shapes at cost-model binding only."""

from copy import deepcopy

import pytest

from rag_stack.evaluation import EvaluationResult
from rag_stack.evaluator_contract import normalize_backend_result
from rag_stack.rag_ir import make_quality_trace_envelope
from rag_stack.rag_ir.normalize import dispatch_manifest


def _config(downstream):
    nodes = [{"stage": "semantic_retrieval", "top_k": 3,
              "modules": [{"component": "vectordb", "vectordb": "dense"}]}]
    if downstream:
        component = {"passage_reranker": "sentence_transformer_reranker",
                     "passage_compressor": "longllmlingua"}[downstream]
        nodes.append({"stage": downstream, "modules": [{"component": component}]})
    nodes.append({"stage": "generator", "modules": [{"component": "vllm"}]})
    return {"node_lines": [{"nodes": nodes}],
            "vectordb": [{"name": "dense", "db_type": "faiss_ivf"}]}


def _bind(config, output_counts):
    calls = []
    for count in output_counts:
        calls.append([
            {"stage": "semantic_retrieval_encode", "input_tokens": 1, "output_tokens": 0},
            {"stage": "semantic_retrieval_vectorsearch", "input_tokens": 0, "output_tokens": 3},
            {"stage": "retrieval_fusion", "model_id": "rrf_pure",
             "input_tokens": 0, "output_tokens": 0,
             "input_profile": {"items": 3, "list_sizes": [3], "unique_items": 3},
             "output_profile": {"items": count, "unique_items": count}},
            {"stage": "generator", "input_tokens": 3, "output_tokens": 1},
        ])
    for query in calls:
        for index, call in enumerate(query):
            call["step_idx"] = index
    trace = make_quality_trace_envelope(calls, question_ids=[str(i) for i in range(len(calls))])
    normalize_backend_result(
        EvaluationResult(metrics={"quality": 1.0}, trace=trace),
        backend="custom", pipeline_config=config,
    )
    return trace


@pytest.mark.parametrize("downstream", ["passage_reranker", "passage_compressor"])
@pytest.mark.parametrize("counts", [[2], [3, 2], [0]])
def test_downstream_shape_mismatch_is_valid_trace_but_rejected_by_cm(downstream, counts):
    config = _config(downstream)
    trace = _bind(config, counts)
    original = deepcopy(trace)
    with pytest.raises(ValueError, match="fixed-passage-count cost-model adapter"):
        dispatch_manifest(config)
    assert trace == original


@pytest.mark.parametrize("counts", [[2], [3, 2], [0]])
def test_fusion_to_generator_accepts_observed_short_variable_and_empty_output(counts):
    config = _config(None)
    _bind(config, counts)
    assert "retrieval_fusion" in [entry.stage for entry in dispatch_manifest(config)]


@pytest.mark.parametrize("downstream", ["passage_reranker", "passage_compressor"])
def test_matching_output_count_keeps_existing_downstream_binding(downstream):
    config = _config(downstream)
    _bind(config, [3, 3])
    assert downstream in [entry.stage for entry in dispatch_manifest(config)]


def test_missing_top_k_uses_current_downstream_default_count():
    config = _config("passage_reranker")
    config["node_lines"][0]["nodes"][0].pop("top_k")
    _bind(config, [3])
    with pytest.raises(ValueError, match=r"semantic_retrieval.top_k=16"):
        dispatch_manifest(config)
