"""Cross-layer contract tests: executable hybrid IR is not an assumed CM model."""

from copy import deepcopy
import json
import subprocess
import sys

import pytest

from rag_stack.evaluation import (
    EvaluationResult, capture_trace, trace_query, trace_stage,
    summarize_items, summarize_lists, normalize_evaluation_result,
)
from rag_stack.evaluator_contract import normalize_backend_result, bind_trace_workflow
from rag_stack.rag_ir import (
    WorkflowSchema, build_workflow_schema, compile_execution_paths,
    validate_quality_trace_envelope, validate_workflow_trace,
)


def hybrid_config():
    return {"node_lines": [{"node_line_name": "rag", "nodes": [
        {"stage": "semantic_retrieval", "top_k": 2,
         "modules": [{"component": "vectordb", "vectordb": "dense"}]},
        {"stage": "lexical_retrieval", "top_k": 2,
         "modules": [{"component": "bm25"}]},
        {"stage": "hybrid_retrieval", "top_k": 2,
         "modules": [{"component": "hybrid_rrf", "weight": 60}]},
        {"stage": "generator", "modules": [{"component": "vllm"}]},
    ]}], "vectordb": [{"name": "dense", "db_type": "faiss_ivf"}]}


def hybrid_trace():
    with capture_trace() as session:
        for qid in ("q1", "q2"):
            with trace_query(qid):
                with trace_stage("semantic_retrieval_encode", model_id="embedding", node_id="semantic_retrieval") as encode:
                    encode.input(token_count=3)
                    encode.output(token_count=0)
                with trace_stage("semantic_retrieval_vectorsearch", model_id="faiss_ivf", node_id="semantic_retrieval",
                                 input_refs={"embedding": [encode.call_id]}) as search:
                    search.input(token_count=0)
                    search.output(profile=summarize_lists([["a", "b"], ["b", "c"]]))
                with trace_stage("retrieval_fusion", model_id="rrf_pure", node_id="semantic_retrieval",
                                 input_refs={"candidates": [search.call_id]}) as dense:
                    dense.input(profile=summarize_lists([["a", "b"], ["b", "c"]]))
                    dense.output(profile=summarize_items(["b", "a"]))
                with trace_stage("lexical_retrieval", model_id="rank_bm25.BM25Okapi", node_id="lexical_retrieval") as lexical:
                    lexical.input(token_count=3)
                    lexical.output(profile=summarize_lists([["b", "c"]]))
                with trace_stage("retrieval_fusion", model_id="rrf_pure", node_id="lexical_retrieval",
                                 input_refs={"candidates": [lexical.call_id]}) as bm25:
                    bm25.input(profile=summarize_lists([["b", "c"]]))
                    bm25.output(profile=summarize_items(["b", "c"]))
                with trace_stage("retrieval_fusion", model_id="rrf_pure", node_id="hybrid_retrieval",
                                 input_refs={"dense": [dense.call_id], "lexical": [bm25.call_id]}) as fusion:
                    fusion.input(profile=summarize_lists([["b", "a"], ["b", "c"]]))
                    fusion.output(profile=summarize_items(["b", "a"]))
                with trace_stage("generator", model_id="generator", node_id="generator",
                                 input_refs={"candidates": [fusion.call_id]}) as generate:
                    generate.input(token_count=8)
                    generate.output(token_count=2)
    return session.finish(expected_question_ids=["q1", "q2"])


def test_hybrid_roundtrips_through_evaluator_ir_cache_and_path_without_pricing():
    config, trace = hybrid_config(), hybrid_trace()
    original = deepcopy(config)
    schema = build_workflow_schema(config, trace=trace)
    assert config == original
    assert schema.execution_order == ["semantic_retrieval", "lexical_retrieval", "hybrid_retrieval", "generator"]
    assert schema.nodes[2]["inputs"] == {"dense": ["semantic_retrieval"], "lexical": ["lexical_retrieval"]}
    assert schema.nodes[0]["operations"][-1] == "retrieval_fusion"
    payload = normalize_backend_result(
        EvaluationResult(metrics={"quality": 0.7}, trace=trace, workflow=schema),
        backend="static_gt", pipeline_config=config, question_ids=["q1", "q2"],
    )
    assert payload["__execution_dag__"] is trace
    saved = json.loads(json.dumps(trace))
    bind_trace_workflow(config, saved)
    assert build_workflow_schema(config).to_dict() == schema.to_dict()
    paths = compile_execution_paths([query["calls"] for query in saved["queries"]])
    assert paths == [query["calls"] for query in saved["queries"]]
    assert [call["node_id"] for call in paths[0] if call["stage"] == "retrieval_fusion"] == [
        "semantic_retrieval", "lexical_retrieval", "hybrid_retrieval",
    ]


@pytest.mark.parametrize("mutation,match", [
    (lambda calls: calls[5]["input_refs"].update(dense=["call_4"]), "current result"),
    (lambda calls: calls[5]["input_refs"].update(dense=["call_6"]), "earlier call"),
    (lambda calls: calls[5].update(call_id="call_4"), "duplicate call_id"),
    (lambda calls: calls[5].pop("input_refs"), "explicit dense and lexical"),
    (lambda calls: calls[5].pop("input_profile"), "actual input and output profiles"),
    (lambda calls: calls.pop(5), "earlier call|observed fusion"),
    (lambda calls: calls[5].update(node_id="invented"), "absent from workflow"),
])
def test_hybrid_contract_rejects_wrong_missing_or_future_producers(mutation, match):
    trace = hybrid_trace()
    schema = build_workflow_schema(hybrid_config(), trace=trace)
    mutation(trace["queries"][0]["calls"])
    with pytest.raises(ValueError, match=match):
        validate_workflow_trace(schema, trace)


def test_hybrid_input_counts_must_match_the_actual_referenced_outputs():
    trace = hybrid_trace()
    schema = build_workflow_schema(hybrid_config(), trace=trace)
    call = trace["queries"][0]["calls"][5]
    call["input_profile"] = summarize_lists([["b"], ["b", "c", "a"]])
    with pytest.raises(ValueError, match="disagree with referenced producer"):
        validate_workflow_trace(schema, trace)


def test_hybrid_cannot_omit_the_actual_lexical_operation():
    trace = hybrid_trace()
    schema = build_workflow_schema(hybrid_config(), trace=trace)
    trace["queries"][0]["calls"][3].pop("node_id")
    with pytest.raises(ValueError, match="observed retrieval operations"):
        validate_workflow_trace(schema, trace)


@pytest.mark.parametrize("inputs,match", [
    ({"candidates": ["semantic_retrieval"]}, "dense and lexical ports"),
    ({"dense": ["lexical_retrieval"], "lexical": ["semantic_retrieval"]}, "semantic_retrieval producer"),
])
def test_hybrid_schema_requires_typed_producer_ports(inputs, match):
    schema = build_workflow_schema(hybrid_config()).to_dict()
    schema["nodes"][2]["inputs"] = inputs
    with pytest.raises(ValueError, match=match):
        WorkflowSchema.from_dict(schema)


def test_missing_identity_is_not_reconstructed_for_legacy_hybrid_trace():
    trace = hybrid_trace()
    schema = build_workflow_schema(hybrid_config(), trace=trace)
    for query in trace["queries"]:
        for call in query["calls"]:
            for key in ("call_id", "node_id", "input_refs"):
                call.pop(key, None)
    validate_quality_trace_envelope(trace)  # Old traces remain structurally readable.
    with pytest.raises(ValueError, match="recollect"):
        validate_workflow_trace(schema, trace)


def test_hybrid_cannot_replay_aggregate_statistics_as_observed_execution():
    config = hybrid_config()
    aggregate = {"provenance": {"workflow_schema": build_workflow_schema(config).to_dict()}}
    with pytest.raises(ValueError, match="canonical quality trace"):
        bind_trace_workflow(config, aggregate)


def test_sequential_trace_cannot_return_to_an_earlier_pipeline_node():
    trace = hybrid_trace()
    schema = build_workflow_schema(hybrid_config(), trace=trace)
    calls = trace["queries"][0]["calls"]
    calls[3]["stage"] = "semantic_retrieval_encode"
    calls[3]["node_id"] = "semantic_retrieval"
    calls[2]["node_id"] = "lexical_retrieval"
    with pytest.raises(ValueError, match="sequential execution_order"):
        validate_workflow_trace(schema, trace)


def test_cm_cannot_turn_a_valid_hybrid_trace_into_zero_cost_lexical_work():
    from rag_stack.cost_model.support import require_cm_pipeline_support
    from rag_stack.cost_model.workload import prepare_trace_workload
    from rag_stack.cost_model.sweep_inputs import build_performance_stage_cost_models
    for operation in (
        lambda: require_cm_pipeline_support(hybrid_config()),
        lambda: prepare_trace_workload(hybrid_trace()),
        lambda: build_performance_stage_cost_models(hybrid_config()),
    ):
        with pytest.raises(NotImplementedError, match="BM25 analytical cost"):
            operation()


def test_workflow_rejects_parallel_policy_and_forward_node_edges():
    schema = build_workflow_schema(hybrid_config()).to_dict()
    invalid = deepcopy(schema)
    invalid["execution_policy"] = "parallel"
    with pytest.raises(ValueError, match="sequential"):
        WorkflowSchema.from_dict(invalid)
    invalid = deepcopy(schema)
    invalid["nodes"][0]["inputs"] = {"queries": ["generator"]}
    with pytest.raises(ValueError, match="preceding nodes"):
        WorkflowSchema.from_dict(invalid)


def test_planning_and_capture_do_not_import_evaluator_optimizer_or_cost_runtime():
    code = '''
import sys
from rag_stack.rag_ir import build_workflow_schema
from rag_stack.evaluation import capture_trace
build_workflow_schema({"node_lines": []})
assert not any(name.startswith(("rag_stack.cost_model", "rag_stack_evaluator", "ax", "torch")) for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
