"""CPU-only public contract and external backend discovery tests."""

from __future__ import annotations

from tests import REPO_ROOT

from dataclasses import replace
from pathlib import Path
import subprocess
import os
import json
import logging
from copy import deepcopy
from importlib import metadata
import sys
from types import SimpleNamespace

import pytest

from rag_stack.evaluation.api import (
    EvaluationResult, Evaluator, normalize_evaluation_result, publish_trace,
)
from rag_stack.evaluation import registry as factory
from rag_stack.evaluation.registry import BackendSpec, EvaluatorContext


class ExternalEvaluator:
    def __init__(self, context):
        self.context = context

    def evaluate(self, config, run_dir=None, metrics_override=None, on_trace_ready=None):
        trace = make_trace()
        publish_trace(trace, on_trace_ready=on_trace_ready)
        return EvaluationResult({"quality": 0.75}, trace)


def make_trace():
    return {"queries": [{"question_id": "q1", "calls": [{
        "stage": "generator", "input_tokens": 8, "output_tokens": 3,
        "step_idx": 0, "model_id": "test-model",
    }]}]}


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(factory, "_BACKENDS", {})
    monkeypatch.syspath_prepend(str(REPO_ROOT / "FlashRAG"))
    # Real importlib discovery using temporary distribution metadata. Do not
    # reinstall an editable package in the shared benchmark environment.
    dist = tmp_path / "fixture_evaluator-0.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: fixture-evaluator\nVersion: 0.0.0\n")
    (dist / "entry_points.txt").write_text(
        "[rag_stack.evaluators]\n"
        "static_gt = rag_stack_evaluator.backends:STATIC_GT\n"
        "flashrag = flashrag.integrations.rag_stack:BACKEND\n"
    )
    entries = tuple(metadata.Distribution.at(dist).entry_points)
    monkeypatch.setattr(factory, "_entry_points", lambda: entries)


def spec(name="external", **kwargs):
    return BackendSpec(name=name, factory=ExternalEvaluator, metrics=("quality",), **kwargs)


def test_structural_backend_receives_context_and_can_publish_trace():
    factory.register_backend(spec())
    manager = object()
    options = {"judge": "cpu-fixture"}
    evaluator = factory.create_quality_evaluator(
        dataset_manager=manager, project_dir="project", backend="external",
        backend_options=options,
    )
    assert isinstance(evaluator, Evaluator)
    assert evaluator.context.dataset_manager is manager
    assert evaluator.context.project_dir == "project"
    assert evaluator.context.options == options
    assert evaluator.context.options is not options
    observed = []
    result = normalize_evaluation_result(
        evaluator.evaluate({}, on_trace_ready=observed.append), require_trace=True,
    )
    assert result["quality"] == 0.75
    assert result["__execution_dag__"] is observed[0]
    assert factory.supported_metrics("external") == ("quality",)
    assert factory.supported_pipeline_modes("external") == ("sequential",)
    assert not factory.get_backend_spec("external").supports_measured


def test_builtin_metadata_and_lazy_capabilities():
    assert factory.get_backend_spec("static_gt").pipeline_style == "node_lines"
    assert factory.get_backend_spec("static_gt").supports_measured
    assert factory.get_backend_spec("flashrag").pipeline_style == "methods"
    assert "react" in factory.supported_pipeline_modes("flashrag")
    assert "rouge_l" in factory.supported_metrics("flashrag")
    called = []
    factory.register_backend(replace(spec(), metrics=lambda: called.append(True) or ("quality",)))
    assert not called
    assert factory.supported_metrics("external") == ("quality",)
    assert called == [True]


def test_discovery_does_not_import_unselected_entry_points(monkeypatch):
    loaded = []
    entries = (
        SimpleNamespace(name="external", load=lambda: loaded.append("external") or spec()),
        SimpleNamespace(name="unused", load=lambda: pytest.fail("loaded unused plugin")),
    )
    monkeypatch.setattr(factory, "_entry_points", lambda: entries)
    assert factory.available_backends() == ("external", "unused")
    assert not loaded
    assert factory.get_backend_spec("external").name == "external"
    assert factory.get_backend_spec("external").name == "external"
    assert loaded == ["external"]


def test_duplicate_names_cannot_replace_builtin_or_registered_backend():
    with pytest.raises(ValueError, match="already registered or installed"):
        factory.register_backend(spec("static_gt"))
    factory.register_backend(spec())
    with pytest.raises(ValueError, match="already registered"):
        factory.register_backend(spec())


def test_duplicate_installed_names_and_invalid_entry_points_fail(monkeypatch):
    entry = SimpleNamespace(name="external", load=lambda: spec())
    monkeypatch.setattr(factory, "_entry_points", lambda: (entry, entry))
    with pytest.raises(ValueError, match="Multiple installed"):
        factory.get_backend_spec("external")
    with pytest.raises(ValueError, match="already registered or installed"):
        factory.register_backend(spec())
    monkeypatch.setattr(factory, "_entry_points", lambda: (
        SimpleNamespace(name="external", load=lambda: ExternalEvaluator),
    ))
    with pytest.raises(TypeError, match="must resolve to a BackendSpec"):
        factory.get_backend_spec("external")
    monkeypatch.setattr(factory, "_entry_points", lambda: (
        SimpleNamespace(name="external", load=lambda: spec("different")),
    ))
    with pytest.raises(ValueError, match="declares backend name"):
        factory.get_backend_spec("external")


def test_unknown_backend_and_retired_alias_are_actionable():
    with pytest.raises(ValueError, match="Supported: flashrag, static_gt"):
        factory.get_backend_spec("unknown")
    with pytest.raises(ValueError, match="renamed to 'static_gt'"):
        factory.get_backend_spec("static")


def test_factory_rejects_missing_evaluate_and_non_dictionary_options():
    factory.register_backend(replace(spec(), factory=lambda context: object()))
    with pytest.raises(TypeError, match="evaluate method"):
        factory.create_quality_evaluator(backend="external")
    with pytest.raises(TypeError, match="must be a dictionary"):
        factory.create_quality_evaluator(backend="external", backend_options=[])
    with pytest.raises(TypeError, match="unexpected keyword"):
        factory.create_quality_evaluator(backend="external", flashrag_options={})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, None, "0.8"])
def test_strict_results_reject_invalid_scores(value):
    with pytest.raises(ValueError, match="finite number"):
        normalize_evaluation_result(EvaluationResult({"quality": value}))
    with pytest.raises(ValueError, match="finite number"):
        normalize_evaluation_result({"quality": value})


def test_strict_result_requires_trace_when_used_for_cost_model():
    assert normalize_evaluation_result(EvaluationResult({"quality": 1})) == {"quality": 1.0}
    with pytest.raises(ValueError, match="requires a canonical"):
        normalize_evaluation_result(EvaluationResult({"quality": 1}), require_trace=True)
    with pytest.raises(TypeError, match="envelope must be a dict"):
        normalize_evaluation_result(EvaluationResult({"quality": 1}, []))


def test_legacy_flat_results_retain_private_metadata_and_static_fanout():
    fanout = {"retrieval": {"input_tokens": 8}}
    result = {"quality": None, "__fp_hit__": "donor", "__execution_dag__": fanout}
    normalized = normalize_evaluation_result(result, allow_legacy=True)
    assert normalized == result
    assert normalized["__execution_dag__"] is fanout
    with pytest.raises(ValueError, match="requires a canonical"):
        normalize_evaluation_result(result, require_trace=True, allow_legacy=True)
    with pytest.raises(ValueError, match="Unsupported evaluator result metadata"):
        normalize_evaluation_result(result)


def test_typed_results_never_enable_legacy_score_validation():
    with pytest.raises(ValueError, match="finite number"):
        normalize_evaluation_result(EvaluationResult({"quality": None}), allow_legacy=True)


def test_workflow_published_before_callback_and_preserved_with_trace_identity():
    trace = make_trace()
    trace["provenance"] = None
    workflow = {"dataflow_type": "sequential", "stages": ["generator_prefill", "generator_decode"]}
    observed = []
    published = publish_trace(trace, workflow=workflow, on_trace_ready=observed.append)
    assert published is trace is observed[0]
    assert observed[0]["provenance"]["workflow_schema"]["dataflow_type"] == "sequential"
    result = normalize_evaluation_result(EvaluationResult({"quality": 0.75}, trace, workflow))
    assert result["__execution_dag__"] is trace
    assert result["__workflow_schema__"] == trace["provenance"]["workflow_schema"]
    replay = normalize_evaluation_result({"quality": 0.75, "__execution_dag__": trace})
    assert replay["__workflow_schema__"] == result["__workflow_schema__"]
    with pytest.raises(ValueError, match="conflicts"):
        normalize_evaluation_result(EvaluationResult(
            {"quality": 0.75}, trace, dict(workflow, dataflow_type="agentic"),
        ))


def test_trace_callback_failure_is_advisory_but_bad_trace_is_rejected(caplog, monkeypatch):
    def broken_callback(trace):
        raise RuntimeError("CM fixture failure")

    # Applications may disable or reroute loggers during configuration. Own
    # this capture locally so the contract does not depend on suite order.
    logger = logging.getLogger(publish_trace.__module__)
    monkeypatch.setattr(logger, "disabled", False)
    monkeypatch.setattr(logger, "propagate", False)
    monkeypatch.setattr(logger, "handlers", [caplog.handler])
    trace = make_trace()
    with caplog.at_level(logging.WARNING, logger=logger.name):
        assert publish_trace(trace, on_trace_ready=broken_callback) is trace
    assert "continuing quality evaluation" in caplog.text
    trace["queries"][0]["calls"][-1]["stage"] = "semantic_retrieval_encode"
    with pytest.raises(ValueError, match="incomplete"):
        publish_trace(trace, on_trace_ready=lambda trace: pytest.fail("invalid trace published"))


def test_shared_interface_and_custom_backend_do_not_import_evaluator_or_models(tmp_path):
    code = r'''
import sys
class RejectRuntimeImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"rag_stack_evaluator", "torch", "vllm", "flashrag", "transformers"}:
            raise AssertionError("Unexpected runtime dependency: " + fullname)
sys.meta_path.insert(0, RejectRuntimeImports())
from rag_stack.evaluation import (
    EvaluationResult, Evaluator, BaseEvaluator, TraceSession,
    BackendSpec, register_backend, create_quality_evaluator, available_backends,
)
from rag_stack.evaluation.api import normalize_evaluation_result
class Plugin:
    def __init__(self, context):
        self.context = context
    def evaluate(self, config, run_dir=None, metrics_override=None, on_trace_ready=None):
        session = TraceSession()
        with session.query("q1"):
            with session.stage("generator", model_id="cpu-fixture") as call:
                call.input(token_count=5)
                call.output(token_count=2)
        return EvaluationResult({"quality": 0.5}, session.finish(expected_question_ids=["q1"]))
from types import ModuleType
plugin = ModuleType("independent_plugin")
plugin.BACKEND = BackendSpec("independent", Plugin, metrics=("quality",))
sys.modules["independent_plugin"] = plugin
assert "independent" in available_backends()
result = create_quality_evaluator(backend="independent").evaluate({})
assert isinstance(result, EvaluationResult)
assert normalize_evaluation_result(result, require_trace=True)["quality"] == 0.5
assert "rag_stack_evaluator" not in sys.modules
assert "torch" not in sys.modules
'''
    dist = tmp_path / "independent_plugin-0.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: independent-plugin\nVersion: 0.0.0\n")
    (dist / "entry_points.txt").write_text(
        "[rag_stack.evaluators]\nindependent = independent_plugin:BACKEND\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(REPO_ROOT)))
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_empty_registry_has_no_implicit_backend(monkeypatch):
    monkeypatch.setattr(factory, "_entry_points", lambda: ())
    assert factory.available_backends() == ()
    with pytest.raises(ValueError, match="Unknown global.eval_backend"):
        factory.get_backend_spec("static_gt")


def test_runtime_stats_and_metadata_are_separate_from_canonical_trace():
    aggregate = {"retrieval": {"input_tokens": 8}}
    result = EvaluationResult(
        {"quality": 0.75}, runtime_stats=aggregate, metadata={"__fp_hit__": "donor"},
    )
    normalized = normalize_evaluation_result(result)
    assert normalized["__execution_dag__"] is aggregate
    assert normalized["__fp_hit__"] == "donor"
    with pytest.raises(ValueError, match="requires a canonical"):
        normalize_evaluation_result(result, require_trace=True)
    with pytest.raises(ValueError, match="unsupported fields"):
        normalize_evaluation_result(EvaluationResult({"quality": 0.75}, aggregate))
    with pytest.raises(ValueError, match="Invalid evaluator metadata"):
        normalize_evaluation_result(EvaluationResult({}, metadata={"__execution_dag__": aggregate}))


def test_static_ir_capability_defaults_to_false():
    assert not spec().supports_static_ir
    assert factory.get_backend_spec("static_gt").supports_static_ir
    assert not factory.get_backend_spec("flashrag").supports_static_ir
    with pytest.raises(TypeError, match="supports_static_ir must be a boolean"):
        replace(spec(), supports_static_ir="yes")


@pytest.mark.parametrize("aggregate", [
    None,
    {"retrieval": {"input_tokens": 8}},
    {"retrieval": {"input_tokens": 8}, "provenance": None},
    {"provenance": {"source": "static-fixture"}},
])
def test_static_workflow_survives_host_normalization_and_serialized_replay(aggregate):
    from rag_stack.evaluator_contract import bind_trace_workflow, normalize_backend_result
    from rag_stack.rag_ir import WorkflowSchema, build_workflow_schema

    stages = ["generator_prefill", "generator_decode"]
    workflow = WorkflowSchema(dataflow_type="agentic", stages=stages)
    original = deepcopy(aggregate)
    config = {"_runtime": {"rag_ir_mode": "static"}}
    payload = normalize_backend_result(
        EvaluationResult({"quality": 0.75}, workflow=workflow, runtime_stats=aggregate),
        backend="static_fixture", pipeline_config=config,
    )
    carrier = payload["__execution_dag__"]
    assert carrier is not aggregate
    assert aggregate == original
    assert carrier["provenance"]["workflow_schema"] == workflow.to_dict()
    assert "__workflow_schema__" not in payload
    assert config["_evaluator_workflow_schema"] == workflow.to_dict()
    replay_config = {"pipeline_runtime": {"mode": "sequential"}}
    bind_trace_workflow(replay_config, json.loads(json.dumps(carrier)))
    assert build_workflow_schema(replay_config, stages=stages).to_dict() == workflow.to_dict()


def test_static_embedded_workflow_is_validated_without_mutating_statistics():
    from rag_stack.rag_ir import WorkflowSchema

    workflow = WorkflowSchema(dataflow_type="sequential", stages=[]).to_dict()
    aggregate = {"provenance": {"workflow_schema": workflow, "source": "fixture"}}
    original = deepcopy(aggregate)
    payload = normalize_evaluation_result(EvaluationResult({}, runtime_stats=aggregate))
    assert payload["__workflow_schema__"] == workflow
    assert payload["__execution_dag__"]["provenance"]["source"] == "fixture"
    assert payload["__execution_dag__"] is not aggregate
    assert aggregate == original
    with pytest.raises(ValueError, match="conflicts"):
        normalize_evaluation_result(EvaluationResult(
            {}, workflow=dict(workflow, dataflow_type="agentic"), runtime_stats=aggregate,
        ))
    with pytest.raises(TypeError, match="provenance must be a dictionary"):
        normalize_evaluation_result(EvaluationResult({}, runtime_stats={"provenance": []}))
    with pytest.raises(ValueError, match="dataflow_type"):
        normalize_evaluation_result(EvaluationResult({}, runtime_stats={
            "provenance": {"workflow_schema": dict(workflow, dataflow_type="invalid")},
        }))
