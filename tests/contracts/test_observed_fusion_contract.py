"""Observed component I/O flows through the public evaluator/CM boundary."""

from copy import deepcopy
import json

import pytest

from rag_stack.evaluation import (
    EvaluationResult, capture_trace, summarize_items, summarize_lists,
    trace_query, trace_stage,
)
from rag_stack.evaluator_contract import bind_trace_workflow, normalize_backend_result
from rag_stack.rag_ir import build_workflow_schema, normalize_quality_trace
from rag_stack.rag_ir.execution_path import compile_execution_paths
from rag_stack.rag_ir.normalize import dispatch_manifest, with_observed_trace_components


def _config():
    return {
        "node_lines": [{"nodes": [
            {"stage": "query_expansion", "modules": [{"component": "multi_query_expansion"}]},
            {"stage": "semantic_retrieval", "top_k": 99,
             "modules": [{"component": "vectordb", "vectordb": "dense"}]},
            {"stage": "generator", "modules": [{"component": "vllm"}]},
        ]}],
        "vectordb": [{"name": "dense", "db_type": "faiss_ivf"}],
        "system": {"batch_size_request": 1},
    }


def _trace(*, fusion=True, implementation="rrf_pure"):
    """Capture unequal observed shapes, independent of configured top-k/fanout."""
    with capture_trace() as session:
        for qid, incoming, outgoing in (
            ("q1", [["a", "a", "b"], ["b", "c"]], ["b", "c"]),
            ("q2", [["x", "x", "y"]], ["y"]),
        ):
            with trace_query(qid):
                with trace_stage("query_expansion", model_id="expander") as call:
                    call.input(token_count=2)
                    call.output(token_count=3)
                with trace_stage("semantic_retrieval_encode", model_id="embedder") as call:
                    call.input(token_count=5)
                    call.output(token_count=0)
                with trace_stage("semantic_retrieval_vectorsearch", model_id="faiss_ivf") as call:
                    call.input(token_count=0)
                    call.output(profile=summarize_lists(incoming))
                if fusion:
                    with trace_stage("retrieval_fusion", model_id=implementation) as call:
                        call.input(profile=summarize_lists(incoming))
                        call.output(profile=summarize_items(outgoing))
                with trace_stage("generator", model_id="generator") as call:
                    call.input(token_count=12)
                    call.output(token_count=2)
    return session.finish(expected_question_ids=["q1", "q2"])


def test_public_capture_reaches_registered_models_and_exact_work_without_mutating_candidate():
    from rag_stack.cost_model.rrf import RRFCostModel
    from rag_stack.cost_model.sweep_inputs import build_performance_stage_cost_models

    candidate = _config()
    original = deepcopy(candidate)
    trace = _trace()
    trace_before = deepcopy(trace)
    workflow = build_workflow_schema(candidate, trace=trace)
    assert candidate == original
    assert trace == trace_before
    assert workflow.stages == [
        "query_expansion_prefill", "query_expansion_decode",
        "semantic_retrieval_encode", "semantic_retrieval_vectorsearch",
        "retrieval_fusion", "generator_prefill", "generator_decode",
    ]

    # The host deliberately binds transient metadata onto its resolved copy.
    resolved = deepcopy(candidate)
    payload = normalize_backend_result(
        EvaluationResult(metrics={"quality": 0.8}, trace=trace, workflow=workflow),
        backend="custom_eval", pipeline_config=resolved, question_ids=["q1", "q2"],
    )
    assert payload["__execution_dag__"] is trace
    assert trace["provenance"]["evaluator_backend"] == "custom_eval"
    assert candidate == original
    assert all(resolved[key] == value for key, value in original.items())

    models = build_performance_stage_cost_models(resolved)
    assert isinstance(models["retrieval_fusion"], RRFCostModel)
    assert list(models) == workflow.stages
    fusion_entries = [entry for entry in dispatch_manifest(resolved)
                      if entry.stage == "retrieval_fusion"]
    assert len(fusion_entries) == 1 and fusion_entries[0].component == "rrf"

    raw = normalize_quality_trace(trace).traces
    paths = compile_execution_paths(raw, retrieval_fanout=4)
    assert paths == raw
    assert compile_execution_paths(paths, retrieval_fanout=4) == paths
    observed = [call for calls in paths for call in calls if call["stage"] == "retrieval_fusion"]
    work = [models["retrieval_fusion"].build_work_profile(call) for call in observed]
    assert [profile.metadata["input_items"] for profile in work] == [5, 3]
    assert [profile.metadata["input_lists"] for profile in work] == [2, 1]
    assert [profile.metadata["union_items"] for profile in work] == [3, 2]
    assert [profile.outputs["candidates"].items for profile in work] == [2, 1]
    assert [profile.metadata["implementation_path"] for profile in work] == [
        "multi_list_pandas", "single_list_python",
    ]
    assert not any("duplicates" in assumption or "union_items" in assumption
                   for profile in work for assumption in profile.metadata["assumptions"])
    assert trace["queries"] == trace_before["queries"]


def test_persisted_trace_rebinds_fusion_without_aliasing_the_saved_profiles():
    trace = _trace()
    normalize_backend_result(
        EvaluationResult(metrics={"quality": 1.0}, trace=trace,
                         workflow=build_workflow_schema(_config(), trace=trace)),
        backend="custom_eval", pipeline_config=_config(), question_ids=["q1", "q2"],
    )
    saved = json.loads(json.dumps(trace))
    replay = _config()
    bind_trace_workflow(replay, saved)
    assert "retrieval_fusion" in build_workflow_schema(replay).stages
    replay["_evaluator_fusion_calls"][0]["input_profile"]["list_sizes"][0] = 999
    actual = next(call for call in saved["queries"][0]["calls"]
                  if call["stage"] == "retrieval_fusion")
    assert actual["input_profile"]["list_sizes"] == [3, 2]


@pytest.mark.parametrize("next_trace", ["none", "plain", "backend"])
def test_next_trace_clears_stale_observed_fusion_and_workflow(next_trace):
    config = _config()
    first = _trace()
    normalize_backend_result(
        EvaluationResult(metrics={"quality": 1.0}, trace=first,
                         workflow=build_workflow_schema(config, trace=first)),
        backend="custom_eval", pipeline_config=config,
    )
    assert config["_evaluator_fusion_calls"]
    if next_trace == "backend":
        normalize_backend_result(
            EvaluationResult(metrics={"quality": 0.5}, trace=_trace(fusion=False)),
            backend="custom_eval", pipeline_config=config,
        )
    else:
        bind_trace_workflow(config, None if next_trace == "none" else _trace(fusion=False))
    assert not config.get("_evaluator_fusion_calls")
    assert not config.get("_evaluator_fusion_calls_per_retrieval")
    assert "_evaluator_workflow_schema" not in config
    assert "retrieval_fusion" not in build_workflow_schema(config).stages
    assert all(entry.stage != "retrieval_fusion" for entry in dispatch_manifest(config))


def test_multilists_without_observed_merge_do_not_create_fusion_work():
    from rag_stack.cost_model.sweep_inputs import build_performance_stage_cost_models

    config, trace = _config(), _trace(fusion=False)
    original = deepcopy(config)
    normalized = normalize_quality_trace(trace).traces
    bound = with_observed_trace_components(config, normalized)
    assert config == original
    assert bound["_evaluator_fusion_calls"] == []
    assert "retrieval_fusion" not in build_workflow_schema(config, trace=trace).stages
    assert "retrieval_fusion" not in build_performance_stage_cost_models(bound)
    assert all(entry.stage != "retrieval_fusion" for entry in dispatch_manifest(config))
    paths = compile_execution_paths(normalized, retrieval_fanout=4)
    assert paths == normalized
    assert all(call["stage"] != "retrieval_fusion" for calls in paths for call in calls)


def test_unknown_fusion_implementation_is_observable_but_fails_cost_binding():
    trace = _trace(implementation="custom_hash_rrf")
    config = _config()
    normalize_backend_result(
        EvaluationResult(metrics={"quality": 1.0}, trace=trace),
        backend="custom_eval", pipeline_config=config,
    )
    assert config["_evaluator_fusion_calls"][0]["model_id"] == "custom_hash_rrf"
    with pytest.raises(ValueError, match="No retrieval_fusion cost-model binding"):
        dispatch_manifest(config)
    # Describing valid execution is independent of an available cost binding.
    assert "retrieval_fusion" in build_workflow_schema(_config(), trace=trace).stages


def test_schema_cannot_omit_an_observed_fusion_stage():
    config, trace = _config(), _trace()
    incomplete_stages = build_workflow_schema(config).stages
    with pytest.raises(ValueError, match="retrieval_fusion.*no declared workflow"):
        build_workflow_schema(config, stages=incomplete_stages, trace=trace)


@pytest.mark.parametrize("missing", ["workflow", "models"])
def test_assembly_rejects_missing_fusion_before_any_candidate_sweep(monkeypatch, missing):
    from rag_stack.cost_model.assembly import RAGCMAssembly

    config, trace = _config(), _trace()
    workflow = build_workflow_schema(config, trace=trace if missing == "models" else None)
    assembly = RAGCMAssembly(object())

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid fusion binding reached candidate evaluation")

    monkeypatch.setattr(assembly, "_run_sweep", forbidden)
    with pytest.raises(ValueError, match="omits it" if missing == "workflow" else "no supplied component model"):
        assembly.evaluate_dynamic(
            config, workflow_schema=workflow, trace=trace,
            stage_cost_models={} if missing == "models" else None,
        )
