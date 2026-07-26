"""Host configuration and optimizer integration with the evaluator backend."""

from __future__ import annotations

from tests import REPO_ROOT

from copy import deepcopy
import json
from pathlib import Path
from importlib import metadata
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def provider_entry_points(monkeypatch, tmp_path):
    """Discover provider metadata without reinstalling the shared environment."""
    from rag_stack.evaluation import registry

    monkeypatch.syspath_prepend(str(REPO_ROOT / "FlashRAG"))
    monkeypatch.setattr(registry, "_BACKENDS", {})
    dist = tmp_path / "fixture_flashrag-0.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Name: fixture-flashrag\nVersion: 0.0.0\n")
    (dist / "entry_points.txt").write_text(
        "[rag_stack.evaluators]\n"
        "flashrag = flashrag.integrations.rag_stack:BACKEND\n"
        "static_gt = rag_stack_evaluator.backends:STATIC_GT\n"
    )
    entries = tuple(metadata.Distribution.at(dist).entry_points)
    monkeypatch.setattr(registry, "_entry_points", lambda: entries)


def test_static_backend_does_not_import_flashrag(tmp_path):
    """Host startup and static evaluator selection keep FlashRAG optional."""
    code = r'''
import sys
from types import SimpleNamespace

class RejectFlashRAG:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "flashrag" or fullname.startswith("flashrag."):
            raise AssertionError("Static backend imported " + fullname)

sys.meta_path.insert(0, RejectFlashRAG())
from rag_stack.controller import Controller
from rag_stack_evaluator.static_rag_evaluator import StaticRAGEvaluatorQualityOnly
controller = Controller.__new__(Controller)
controller.project_dir = sys.argv[1]
controller.config = {"global": {"eval_backend": "static_gt"}}
controller._data = SimpleNamespace(
    project_dir=controller.project_dir, qa_data=[], corpus_data=[],
)
evaluator = controller._make_quality_evaluator(None)
assert isinstance(evaluator._engine, StaticRAGEvaluatorQualityOnly)
assert evaluator._dataset is controller._data
'''
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "static")],
        cwd=REPO_ROOT, capture_output=True,
        text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_controller_selects_external_flashrag_evaluator(tmp_path, monkeypatch):
    """The host forwards its manager and backend options to the evaluator."""
    from rag_stack.controller import Controller
    from flashrag.integrations.rag_stack import FlashRAGQualityEvaluator

    controller = Controller.__new__(Controller)
    controller.project_dir = str(tmp_path / "flashrag")
    options = {"monitor": "full", "generator_model": "fixture-model"}
    controller.config = {
        "global": {"eval_backend": "flashrag"},
        "eval_backend_setting": {"flashrag": options},
    }
    controller._data = SimpleNamespace(
        project_dir=controller.project_dir,
        qa_data=pd.DataFrame({"qid": ["q0"], "query": ["Question?"]}),
        corpus_data=pd.DataFrame({"doc_id": ["d0"], "contents": ["Answer"]}),
    )
    evaluator = controller._make_quality_evaluator(None)
    assert isinstance(evaluator, FlashRAGQualityEvaluator)
    assert evaluator._dataset is controller._data
    assert evaluator.qa_data is controller._data.qa_data
    assert evaluator.corpus_data is controller._data.corpus_data
    assert evaluator.flashrag_options == options


def _make_method_sweep_config():
    """Minimal config with two method blocks for tree tests."""
    return {
        "global": {"eval_backend": "flashrag", "n_iterations": 2},
        "dataset": {},
        "system": {"cpu": {}, "cm_search_space": {}},
        "optimizer": {"type": "smac_mobo", "params": {}},
        "eval_backend_setting": {"metrics": [{"metric_name": "em"}]},
        "algo_search_space": {
            "vectordb": [{"name": "faiss_idx", "db_type": "faiss_hnsw"}],
            "pipeline": {"methods": [
                {
                    "mode": "sequential",
                    "node_lines": [{
                        "node_line_name": "seq",
                        "nodes": [
                            {"stage": "semantic_retrieval", "top_k": [4, 8],
                             "modules": [{"component": "vectordb", "vectordb": "faiss_idx"}]},
                            {"stage": "generator", "modules": [{"component": "vllm", "model": "qwen"}]},
                        ],
                    }],
                },
                {
                    "mode": "corag",
                    "max_iter_num": [2, 3],
                    "node_lines": [{
                        "node_line_name": "co",
                        "nodes": [
                            {"stage": "semantic_retrieval", "top_k": 8,
                             "modules": [{"component": "vectordb", "vectordb": "faiss_idx"}]},
                            {"stage": "generator", "modules": [{"component": "vllm", "model": "qwen"}]},
                        ],
                    }],
                },
            ]},
        },
    }



def test_pipeline_mode_has_dependents():
    """Top-level pipeline__mode ChoiceParameter must carry the method
    dependents so Ax auto-deactivates the losing branches."""
    from rag_stack.search_space import RagSearchSpace, SEP
    from ax.core.parameter import ChoiceParameter

    cfg = _make_method_sweep_config()
    ss = RagSearchSpace.from_yaml(cfg)

    mode_param = ss.ss.parameters.get(f"pipeline{SEP}mode")
    assert isinstance(mode_param, ChoiceParameter)
    assert set(mode_param.values) == {"sequential", "corag"}
    deps = mode_param.dependents
    assert deps is not None
    assert set(deps.keys()) == {"sequential", "corag"}

    # sequential branch should include semantic_retrieval__top_k under its prefix.
    seq_params = set(deps["sequential"])
    assert any("sequential" in p and "top_k" in p for p in seq_params)
    # corag branch should include max_iter_num under its prefix; should NOT
    # include any "sequential__" param.
    corag_params = set(deps["corag"])
    assert any("corag" in p and "max_iter_num" in p for p in corag_params)
    assert all("sequential" not in p for p in corag_params)



def test_enumerate_has_no_spurious_cross_method_combos():
    """exhaustive enumeration of a pipeline.mode tree must NOT pair
    method=sequential with corag-only knobs (and vice-versa)."""
    from rag_stack.search_space import RagSearchSpace

    cfg = _make_method_sweep_config()
    ss = RagSearchSpace.from_yaml(cfg)
    configs = ss.enumerate_all_configs()
    assert configs and isinstance(configs, list)

    for c in configs:
        pipeline = c.get("pipeline") or {}
        mode = pipeline.get("mode")
        # When mode is set, the OTHER method's subtree should not be populated.
        if mode == "sequential":
            corag_sub = pipeline.get("corag")
            # Allowed: missing/empty dict. Not allowed: any actual sub-knob.
            assert not corag_sub or "max_iter_num" not in corag_sub, (
                f"sequential trial leaked corag knobs: {pipeline!r}"
            )
        elif mode == "corag":
            seq_sub = pipeline.get("sequential", {}) or {}
            # The sequential branch shouldn't be sampled when mode=corag.
            assert not seq_sub.get("semantic_retrieval"), (
                f"corag trial leaked sequential knobs: {pipeline!r}"
            )



def test_validator_rejects_old_pipeline_shape():
    """Top-level node_lines + old backend 'static' must be rejected."""
    from rag_stack.config_validator import ConfigValidator

    cfg = {
        "system": {"cpu": {}, "search_space": {}},
        "gt_evaluation": {"backend": "static"},
        "pipeline": {"mode": "sequential"},     # old shape: dict, not list
        "node_lines": [{                         # old shape: top-level
            "node_line_name": "x",
            "nodes": [{"stage": "generator", "modules": [{"component": "vllm", "model": "q"}]}],
        }],
        "optimizer": {"type": "smac_mobo", "params": {}},
    }
    errors = ConfigValidator(cfg).validate()
    joined = "\n".join(errors)
    assert "Top-level 'node_lines:' is no longer supported" in joined
    assert "Top-level 'pipeline:' is no longer supported" in joined
    assert "Missing 'algo_search_space:'" in joined



def test_validator_rejects_uninstrumented_method():
    """Unknown methods fail validation with the evaluator's supported set."""
    from rag_stack.config_validator import ConfigValidator
    from rag_stack.evaluation.registry import supported_pipeline_modes

    # Fixture must be schema-valid apart from the uninstrumented method, or the
    # validator's earlier required-field errors (dataset.dataset_name,
    # vectordb.embedding_model — both added in June) mask the method check.
    cfg = {
        "global": {"eval_backend": "flashrag"},
        "dataset": {"dataset_name": "dragonball_en"},
        "system": {"cpu": {"hardware_key": "fixture_cpu"}, "cm_search_space": {}},
        "optimizer": {"type": "smac_mobo", "params": {}},
        "eval_backend_setting": {},
        "algo_search_space": {
            "vectordb": [{"name": "v", "db_type": "faiss_flat",
                          "embedding_model": "sentence-transformers/all-mpnet-base-v2"}],
            "pipeline": {"methods": [
            {"mode": "deepsearcher", "node_lines": [{
                "node_line_name": "x",
                "nodes": [
                    {"stage": "semantic_retrieval", "top_k": 4,
                     "modules": [{"component": "vectordb", "vectordb": "v"}]},
                    {"stage": "generator", "modules": [{"component": "vllm", "model": "q"}]},
                ],
            }]}
        ]}},
    }
    errors = ConfigValidator(cfg).validate()
    joined = "\n".join(errors)
    assert "deepsearcher" in joined
    assert "PIPELINE_MAP" in joined
    # Every supported method should appear in the error so the user knows
    # what's available.
    for m in supported_pipeline_modes("flashrag"):
        assert m in joined



def test_derive_composite_objectives_from_descriptions():
    """derive_composite_objectives turns each metric into a sub-metric carrying
    its description. (param_deps was removed as dead: computed, never consumed.)"""
    from rag_stack.search_space import RagSearchSpace

    cfg = _make_method_sweep_config()
    cfg["eval_backend_setting"] = {
        "backend": "flashrag",
        "metrics": [
            {"metric_name": "em", "description": "exact match"},
            {"metric_name": "global_one"},
        ],
    }
    ss = RagSearchSpace.from_yaml(cfg)
    co = ss.derive_composite_objectives(cfg["eval_backend_setting"])
    sm = co["quality"]["sub_metrics"]
    assert sm["em"]["description"] == "exact match"
    assert sm["global_one"]["description"] == ""
    assert "param_deps" not in sm["em"]


def test_validator_accepts_flashrag_metric_alias():
    """Host validation consumes aliases published by evaluator metadata."""
    from rag_stack.config_validator import ConfigValidator

    config = _make_method_sweep_config()
    config["eval_backend_setting"]["metrics"] = [{"metric_name": "rouge_l"}]
    validator = ConfigValidator(config)
    validator._check_gt_evaluation()
    assert not any("Unknown metric" in error for error in validator.errors)


def test_flashrag_trace_stays_out_of_optimizer_quality_scores(tmp_path, monkeypatch):
    """The host forwards canonical traces to the CM and persists plain metrics."""
    from rag_stack.controller import Controller
    from rag_stack.rag_ir import make_quality_trace_envelope

    trace = make_quality_trace_envelope(
        [[{
            "stage": "generator", "input_tokens": 5, "output_tokens": 3,
            "step_idx": 0, "model_id": "fixture-model",
        }]], question_ids=["0"],
    )
    controller = Controller.__new__(Controller)
    controller.project_dir = str(tmp_path / "project")
    controller.config = {
        "global": {"eval_backend": "flashrag", "rag_ir_mode": "dynamic"},
        "eval_backend_setting": {"metrics": [{"metric_name": "em"}]},
    }
    from rag_stack.project_config import freeze_project_config, write_project_config

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
        resolve_corpus=lambda _params: SimpleNamespace(chunk_hash="fixture"),
        apply_corpus_view=lambda _config, _view: None,
    )
    controller._is_known_invalid = lambda _config: False
    controller._resolve_config = lambda _config: {
        "_runtime": {"rag_ir_mode": "dynamic"},
        "corpus_runtime": {"chunker": {}},
    }
    controller._configure_rag_ir_mode = lambda _config: None
    controller._record_invalid = Mock()
    controller._save_eval_record = Mock()
    controller._update_summary = Mock()
    controller.rag_quality_evaluator = SimpleNamespace(evaluate=Mock(
        return_value={"em": 0.5, "__execution_dag__": trace},
    ))
    controller.assembly = SimpleNamespace(evaluate_dynamic=Mock(
        return_value=SimpleNamespace(performance_score=42.0, best_hw_params=None),
    ))
    monkeypatch.setattr("rag_stack.controller.build_workflow_schema", lambda _: object())

    result, valid, successes, _ = controller._evaluate_objectives(
        [{}], ["performance", "quality"], obj_columns=[],
    )

    assert valid == [{}]
    assert successes == 1
    assert result["quality"] == [{"combined": 0.5, "em": 0.5}]
    assert result["performance"] == [42.0]
    assert controller.assembly.evaluate_dynamic.call_args.kwargs["trace"] is trace
    record = controller._save_eval_record.call_args.kwargs
    assert record["execution_dag"] is trace
    assert record["quality_metrics"] == {"em": 0.5}
