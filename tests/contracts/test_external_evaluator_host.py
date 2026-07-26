"""CPU-only host contract, plugin dispatch, and quality/trace integration tests."""

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from rag_stack.evaluator_contract import bind_trace_workflow, normalize_backend_result
from rag_stack.rag_ir import WorkflowSchema, build_workflow_schema, make_quality_trace_envelope
from rag_stack.evaluation.api import EvaluationResult, publish_trace


STAGES = ["generator_prefill", "generator_decode"]


def _trace(qids=("q1",)):
    return make_quality_trace_envelope(
        [[{
            "stage": "generator", "input_tokens": 5, "output_tokens": 2,
            "step_idx": 0, "model_id": "external-model",
        }] for _ in qids], question_ids=list(qids),
    )


@pytest.fixture
def plugin(monkeypatch):
    from rag_stack.evaluation import registry as factory

    monkeypatch.setattr(factory, "_BACKENDS", dict(factory._BACKENDS))
    monkeypatch.setattr(factory, "_entry_points", lambda: ())
    constructed = []
    evaluator = SimpleNamespace(evaluate=lambda config, **kwargs: None)

    def create(context):
        constructed.append(context)
        return evaluator

    spec = factory.BackendSpec(
        name="host_fixture", factory=create,
        metrics=("external_score",), pipeline_modes=("external_loop", "sequential"),
        pipeline_style="methods",
    )
    factory.register_backend(spec)
    return SimpleNamespace(name=spec.name, evaluator=evaluator, contexts=constructed)


def test_external_result_persists_workflow_but_removes_optimizer_metadata():
    trace = _trace()
    workflow = WorkflowSchema(dataflow_type="agentic", stages=STAGES)
    config = {}
    payload = normalize_backend_result(
        EvaluationResult(metrics={"external_score": 0.75}, trace=trace, workflow=workflow),
        backend="host_fixture", pipeline_config=config, question_ids=["q1"],
    )
    assert payload == {"external_score": 0.75, "__execution_dag__": trace}
    assert payload["__execution_dag__"] is trace
    assert config["_evaluator_workflow_schema"] == workflow.to_dict()
    assert config["_evaluator_workflow_schema"] is not trace["provenance"]["workflow_schema"]
    assert trace["provenance"]["evaluator_backend"] == "host_fixture"
    assert config["_evaluator_trace_stages"] == ["generator"]

    # A persisted trace is sufficient to recover the same structural IR on replay.
    saved_trace = json.loads(json.dumps(trace))
    replay_config = {"pipeline_runtime": {"mode": "sequential"}}
    bind_trace_workflow(replay_config, saved_trace)
    replay_schema = build_workflow_schema(replay_config, stages=STAGES)
    assert replay_schema.dataflow_type == "agentic"
    assert replay_schema.stages == STAGES
    bind_trace_workflow(replay_config, _trace())
    assert "_evaluator_workflow_schema" not in replay_config
    assert "_evaluator_trace_stages" not in replay_config


def test_external_backend_provenance_accepts_explicit_null_provenance():
    trace = _trace()
    trace["provenance"] = None
    config = {}
    normalize_backend_result(
        EvaluationResult(metrics={"external_score": 0.75}, trace=trace),
        backend=" HOST_FIXTURE ", pipeline_config=config, question_ids=["q1"],
    )
    assert trace["provenance"] == {"evaluator_backend": "host_fixture"}
    assert config["_evaluator_trace_stages"] == ["generator"]


@pytest.mark.parametrize("actual, expected", [
    (("q1",), ["q1", "q2"]),
    (("q1", "q2"), ["q1"]),
    (("wrong",), ["q1"]),
])
def test_external_result_requires_exact_question_id_coverage(actual, expected):
    with pytest.raises(ValueError, match="exactly the evaluated question IDs"):
        normalize_backend_result(
            EvaluationResult(metrics={"external_score": 0.75}, trace=_trace(actual)),
            backend="host_fixture", pipeline_config={}, question_ids=expected,
        )


def test_duplicate_dataset_ids_and_trace_free_external_results_are_rejected():
    with pytest.raises(ValueError, match="duplicate question IDs"):
        normalize_backend_result(
            EvaluationResult(metrics={"external_score": 0.75}, trace=_trace()),
            backend="host_fixture", pipeline_config={}, question_ids=["q1", "q1"],
        )
    with pytest.raises(ValueError, match="canonical quality execution trace"):
        normalize_backend_result(
            EvaluationResult(metrics={"external_score": 0.75}),
            backend="host_fixture", pipeline_config={}, question_ids=["q1"],
        )


@pytest.mark.parametrize("backend", ["static_gt", "flashrag", "host_fixture"])
def test_all_backends_share_static_and_dynamic_result_validation(backend):
    result = EvaluationResult(
        metrics={"score": 0.5}, runtime_stats={"shape": "aggregate"},
        metadata={"__fp_hit__": "donor"},
    )
    payload = normalize_backend_result(
        result, backend=backend,
        pipeline_config={"_runtime": {"rag_ir_mode": "static"}},
        question_ids=["unused"],
    )
    assert payload["score"] == 0.5
    assert payload["__fp_hit__"] == "donor"
    assert payload["__execution_dag__"] == {"shape": "aggregate"}
    with pytest.raises(ValueError, match="canonical quality execution trace"):
        normalize_backend_result(result, backend=backend, pipeline_config={})


@pytest.mark.parametrize("backend", ["static_gt", "flashrag", "host_fixture"])
def test_all_backend_names_require_the_same_canonical_cohort(backend):
    with pytest.raises(ValueError, match="exactly the evaluated question IDs"):
        normalize_backend_result(
            EvaluationResult({"score": 0.5}, trace=_trace()),
            backend=backend, pipeline_config={}, question_ids=["different"],
        )


def test_static_provenance_does_not_require_per_query_calls():
    config = {"_runtime": {"rag_ir_mode": "static"}, "_evaluator_trace_stages": ["old"]}
    normalize_backend_result(
        EvaluationResult({"score": 0.5}, runtime_stats={
            "provenance": {"evaluator_backend": "host_fixture"},
        }),
        backend="host_fixture", pipeline_config=config,
    )
    assert "_evaluator_trace_stages" not in config


def test_quality_store_identity_separates_backends_with_the_same_metric():
    from rag_stack.quality_store import _canonical_quality_spec, _quality_spec

    settings = {"metrics": [{"metric_name": "shared_score"}]}
    builtin = _quality_spec({"eval_backend_setting": settings})
    external = _quality_spec({
        "global": {"eval_backend": "host_fixture"},
        "eval_backend_setting": settings,
    })
    assert builtin != external
    assert builtin["_evaluator_backend"] == "static_gt"
    assert external["_evaluator_backend"] == "host_fixture"
    assert _canonical_quality_spec(settings) == builtin
    assert "_evaluator_backend" not in settings


@pytest.mark.parametrize("change, message", [
    ({"stages": ["generator_prefill"]}, "stages must match"),
    ({"stages": STAGES + ["generator_decode"]}, "stages must match"),
    ({"has_encode": True}, "has_encode conflicts"),
    ({"has_reranker": True}, "has_reranker conflicts"),
    ({"has_query_expansion": 1}, "has_query_expansion conflicts"),
    ({"dataflow_type": "parallel"}, "sequential or agentic"),
])
def test_workflow_cannot_change_supported_cost_components(change, message):
    schema = WorkflowSchema(dataflow_type="sequential", stages=STAGES).to_dict()
    schema.update(change)
    config = {"_evaluator_workflow_schema": schema}
    with pytest.raises(ValueError, match=message):
        build_workflow_schema(config, stages=STAGES)


def test_external_workflow_inherits_host_corpus_stats_when_omitted():
    config = {
        "_evaluator_workflow_schema": WorkflowSchema(
            dataflow_type="agentic", stages=STAGES,
        ).to_dict(),
        "_token_stats": {
            "avg_chunk_tokens": 100, "prompt_template_tokens": 5,
            "avg_output_tokens": 20, "avg_query_tokens": 10,
        },
    }
    schema = build_workflow_schema(config, stages=STAGES)
    assert schema.corpus_stats.avg_chunk_tokens == 100
    assert schema.dataflow_type == "agentic"


@pytest.mark.parametrize("observed, resolved", [
    ("generator", []),
    ("generator", ["generator"]),
    ("generator", ["generator_prefill"]),
    ("generator", ["generator_decode"]),
    ("query_expansion", STAGES),
    ("query_expansion", STAGES + ["query_expansion"]),
    ("query_expansion", STAGES + ["query_expansion_prefill"]),
    ("semantic_retrieval_encode", STAGES),
    ("semantic_retrieval_vectorsearch", STAGES),
    ("lexical_retrieval", STAGES),
    ("passage_reranker", STAGES),
    ("passage_compressor", STAGES),
])
def test_observed_trace_stage_requires_a_declared_workflow_operation(observed, resolved):
    with pytest.raises(ValueError, match="no declared workflow operation"):
        build_workflow_schema({"_evaluator_trace_stages": [observed]}, stages=resolved)


@pytest.mark.parametrize("observed, resolved", [
    ("generator", STAGES),
    ("query_expansion", STAGES + ["query_expansion_prefill", "query_expansion_decode"]),
    ("semantic_retrieval_encode", STAGES + ["semantic_retrieval_encode"]),
    ("semantic_retrieval_vectorsearch", STAGES + ["semantic_retrieval_vectorsearch"]),
    ("lexical_retrieval", STAGES + ["lexical_retrieval"]),
    ("passage_reranker", STAGES + ["passage_reranker"]),
    ("passage_reranker", STAGES + ["passage_reranker_prefill", "passage_reranker_decode"]),
    ("passage_compressor", STAGES + ["passage_compressor"]),
    ("passage_compressor", STAGES + ["passage_compressor_prefill", "passage_compressor_decode"]),
])
def test_supported_observed_stages_match_bare_or_split_cost_components(observed, resolved):
    schema = build_workflow_schema({"_evaluator_trace_stages": [observed]}, stages=resolved)
    assert schema.stages == resolved


def test_conditional_configured_stage_may_be_unobserved():
    config = {"_evaluator_trace_stages": ["generator"]}
    schema = build_workflow_schema(
        config, stages=STAGES + ["semantic_retrieval_encode", "semantic_retrieval_vectorsearch"],
    )
    assert schema.has_encode is True


def test_replay_binds_persisted_agentic_workflow_before_schema_construction(monkeypatch):
    from rag_stack.controller import Controller

    source_trace = _trace()
    workflow = WorkflowSchema(dataflow_type="agentic", stages=STAGES)
    normalize_backend_result(
        EvaluationResult(metrics={"external_score": 0.75}, trace=source_trace, workflow=workflow),
        backend="host_fixture", pipeline_config={}, question_ids=["q1"],
    )
    persisted_trace = json.loads(json.dumps(source_trace))
    store = SimpleNamespace(get_execution_dag=Mock(return_value=persisted_trace))
    controller = Controller.__new__(Controller)
    controller._target_replay_pipeline = Mock(return_value={
        "pipeline_runtime": {"mode": "sequential"},
    })
    monkeypatch.setattr(
        "rag_stack.controller.build_workflow_schema",
        lambda config: build_workflow_schema(config, stages=STAGES),
    )
    result = SimpleNamespace(performance_score=42.0)
    assembly = SimpleNamespace(evaluate_dynamic=Mock(return_value=result))
    pipeline, trace, returned_result = controller._price_stored_quality_record(
        quality_store=store, config_hash_value="stored-row", assembly=assembly,
    )
    assert trace is persisted_trace and returned_result is result
    assert pipeline["_evaluator_trace_stages"] == ["generator"]
    schema = assembly.evaluate_dynamic.call_args.kwargs["workflow_schema"]
    assert schema.dataflow_type == "agentic"
    assert schema.stages == STAGES


def test_controller_forwards_backend_context_and_options(plugin, tmp_path):
    from rag_stack.controller import Controller

    controller = Controller.__new__(Controller)
    controller.project_dir = str(tmp_path)
    controller._data = object()
    dataset = object()
    options = {"runner": "example:run", "judge_revision": "pinned"}
    controller.config = {
        "global": {"eval_backend": plugin.name},
        "eval_backend_setting": {"backend_options": options},
    }
    assert controller._make_quality_evaluator(dataset) is plugin.evaluator
    context, = plugin.contexts
    assert context.dataset_manager is controller._data
    assert context.dataset is dataset
    assert context.project_dir == str(tmp_path)
    assert context.options == options
    assert context.options is not options


def _config(backend):
    return {
        "global": {"eval_backend": backend},
        "dataset": {"dataset_name": "dragonball_en"},
        "system": {"cpu": {"hardware_key": "fixture_cpu"}, "cm_search_space": {}},
        "optimizer": {"type": "smac_mobo", "params": {}},
        "eval_backend_setting": {"metrics": [{"metric_name": "external_score"}]},
        "algo_search_space": {
            "vectordb": [],
            "pipeline": {"methods": [{
                "mode": "external_loop",
                "node_lines": [{
                    "node_line_name": "external",
                    "nodes": [{"stage": "generator", "modules": [{
                        "component": "vllm", "model": "external-model",
                    }]}],
                }],
            }]},
        },
    }


def test_config_validation_reads_registered_modes_and_metrics(plugin):
    from rag_stack.config_validator import ConfigValidator

    config = _config(plugin.name)
    validator = ConfigValidator(config, check_model_cache=False)
    validator._check_pipeline_schema()
    validator._check_pipeline_methods()
    validator._check_gt_evaluation()
    assert validator.errors == []
    assert plugin.contexts == [], "metadata validation must not construct the evaluator"

    wrong_mode = deepcopy(config)
    wrong_mode["algo_search_space"]["pipeline"]["methods"][0]["mode"] = "unregistered"
    validator = ConfigValidator(wrong_mode, check_model_cache=False)
    validator._check_pipeline_methods()
    assert any("unregistered" in error and "external_loop" in error for error in validator.errors)

    wrong_metric = deepcopy(config)
    wrong_metric["eval_backend_setting"]["metrics"] = [{"metric_name": "unregistered"}]
    validator = ConfigValidator(wrong_metric, check_model_cache=False)
    validator._check_gt_evaluation()
    assert any("Unknown metric" in error and "unregistered" in error for error in validator.errors)


def test_typed_result_and_early_callback_reach_host_cm_and_plain_quality_scores(
    plugin, tmp_path, monkeypatch,
):
    from rag_stack.controller import Controller
    from rag_stack.project_config import freeze_project_config, write_project_config

    trace = _trace()
    workflow = WorkflowSchema(dataflow_type="agentic", stages=STAGES)
    controller = Controller.__new__(Controller)
    controller.project_dir = str(tmp_path)
    controller.config = {
        "global": {"eval_backend": plugin.name, "rag_ir_mode": "dynamic"},
        "eval_backend_setting": {"metrics": [{"metric_name": "external_score"}]},
    }
    write_project_config(controller.project_dir, json.dumps(controller.config))
    controller._project_config_raw = freeze_project_config(controller.project_dir)
    controller._project_runtime_config = deepcopy(controller.config)
    controller._combined_quality_spec = None
    controller.optimizer = SimpleNamespace(
        objective_names=["performance", "quality"],
        objective_costs={"performance": 1, "quality": 10},
        _pending_full_metrics=[],
    )
    controller._quality_memo_enabled = False
    controller._quality_memo = {}
    controller._gt_memo_hits = 0
    controller._external_quality_store = None
    controller._eval_counter = 0
    controller._measured_provider = None
    controller._cm_fixed_provider = None
    controller._assembly_results = []
    controller._data = SimpleNamespace(
        qa_data=pd.DataFrame({"qid": ["q1"]}),
        resolve_corpus=lambda params: SimpleNamespace(chunk_hash="fixture"),
        apply_corpus_view=lambda config, view: None,
    )
    controller._is_known_invalid = lambda config: False
    controller._resolve_config = lambda config: {
        "_runtime": {"rag_ir_mode": "dynamic"}, "corpus_runtime": {"chunker": {}},
    }
    controller._configure_rag_ir_mode = lambda config: None
    controller._record_invalid = Mock()
    controller._save_eval_record = Mock()
    controller._update_summary = Mock()

    def evaluate(config, *, on_trace_ready, **kwargs):
        publish_trace(trace, workflow=workflow, on_trace_ready=on_trace_ready)
        return EvaluationResult({"external_score": 0.75}, trace, workflow)

    controller.rag_quality_evaluator = SimpleNamespace(evaluate=evaluate)
    controller.assembly = SimpleNamespace(evaluate_dynamic=Mock(
        return_value=SimpleNamespace(performance_score=42.0, best_hw_params=None),
    ))

    def build_schema(config):
        assert config["_evaluator_workflow_schema"] == workflow.to_dict()
        assert config["_evaluator_trace_stages"] == ["generator"]
        assert trace["provenance"]["evaluator_backend"] == plugin.name
        return workflow

    monkeypatch.setattr("rag_stack.controller.build_workflow_schema", build_schema)
    result, valid, successes, _ = controller._evaluate_objectives(
        [{}], ["performance", "quality"], obj_columns=[],
    )
    assert valid == [{}] and successes == 1
    assert result["quality"] == [{"combined": 0.75, "external_score": 0.75}]
    assert result["performance"] == [42.0]
    controller.assembly.evaluate_dynamic.assert_called_once()
    assert controller.assembly.evaluate_dynamic.call_args.kwargs["trace"] is trace
    record = controller._save_eval_record.call_args.kwargs
    assert record["execution_dag"] is trace
    assert record["quality_metrics"] == {"external_score": 0.75}
