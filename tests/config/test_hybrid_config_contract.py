"""Hybrid structure, optimizer projection, and read-only API contracts (CPU)."""

from tests import REPO_ROOT

from copy import deepcopy
import json
import subprocess
import sys

import pytest

from rag_stack.pipeline_contract import (
    hybrid_capability_errors,
    hybrid_structure_errors,
    normalize_retrieval_aliases,
)


def hybrid_config(source="measured"):
    return {
        "global": {"eval_backend": "static_gt"},
        "system": {"performance_source": source},
        "algo_search_space": {"pipeline": {"node_lines": [{
            "node_line_name": "retrieval",
            "nodes": [
                {"stage": "semantic_retrieval", "top_k": [16, 32],
                 "modules": [{"component": "vectordb", "vectordb": "dense"}]},
                {"stage": "lexical_retrieval", "top_k": [17, 33],
                 "modules": [{"component": "bm25", "bm25_tokenizer": "porter_stemmer"}]},
                {"stage": "hybrid_retrieval", "top_k": [5, 9],
                 "modules": [{"component": "hybrid_rrf", "weight": [30, 60]}]},
                {"stage": "prompt_maker", "modules": [{"component": "fstring", "prompt": "{query}"}]},
                {"stage": "generator", "modules": [{"component": "vllm", "model": "test-model"}]},
            ],
        }]}},
    }


def _block(config):
    return config["algo_search_space"]["pipeline"]


def _nodes(config):
    return _block(config)["node_lines"][0]["nodes"]


def test_hybrid_structure_and_cost_capability_are_independent():
    config = hybrid_config()
    assert hybrid_structure_errors(_block(config)) == []
    assert hybrid_capability_errors(_block(config), performance_source="measured") == []
    errors = hybrid_capability_errors(_block(config), performance_source="cost_model")
    assert len(errors) == 2 and "not zero cost" in errors[0]
    assert "analytical scheduling/cost binding" in errors[1]


@pytest.mark.parametrize("mutation, message", [
    (lambda c: _nodes(c).pop(1), "exactly one lexical_retrieval"),
    (lambda c: _nodes(c)[1].update(optional=True), "enabled and non-optional"),
    (lambda c: _block(c).update(rag_dataflow=["sequential", {"react": {"max_iter": [3]}}]), "sequential execution only"),
    (lambda c: _nodes(c)[2]["modules"][0].update(component="hybrid_cc"), "hybrid_cc is not supported"),
    (lambda c: _nodes(c)[2]["modules"][0].update(weight=[0, 60]), "rank constant"),
    (lambda c: _nodes(c)[1].update(top_k=[1.5]), "positive integers"),
    (lambda c: _nodes(c)[2]["modules"][0].update(rrf_k=[30]), "not both"),
    (lambda c: _nodes(c).insert(0, _nodes(c).pop(2)), "node order"),
])
def test_invalid_hybrid_workflows_fail_before_execution(mutation, message):
    config = hybrid_config()
    mutation(config)
    assert any(message in error for error in hybrid_structure_errors(_block(config)))


def test_alias_normalization_preserves_input_and_parameter_specs():
    config = hybrid_config()
    _nodes(config)[0]["stage"] = "dense_retrieval"
    _nodes(config)[1]["stage"] = "sparse_retrieval"
    _nodes(config)[2]["modules"][0]["component"] = "HybridRRF"
    original = deepcopy(config)
    normalized = normalize_retrieval_aliases(config)
    assert [node["stage"] for node in _nodes(normalized)[:3]] == [
        "semantic_retrieval", "lexical_retrieval", "hybrid_retrieval",
    ]
    assert _nodes(normalized)[2]["modules"][0] == {"component": "hybrid_rrf", "weight": [30, 60]}
    assert config == original


def test_resolved_singular_module_nodes_share_the_hybrid_contract():
    resolved = deepcopy(_block(hybrid_config()))
    for node in resolved["node_lines"][0]["nodes"]:
        node["module"] = node.pop("modules")[0]
    assert hybrid_structure_errors(resolved) == []
    normalized = normalize_retrieval_aliases(resolved)
    assert normalized["node_lines"][0]["nodes"][2]["modules"][0]["component"] == "hybrid_rrf"
    assert "module" in resolved["node_lines"][0]["nodes"][2]


def test_optimizer_sweeps_and_resolves_three_independent_top_ks(monkeypatch):
    from rag_stack.search_space import rag_search_space as module

    # This test concerns algorithm parameters, not measured GPU allocation.
    monkeypatch.setattr(module, "extract_system_params", lambda *args: [])
    config = hybrid_config()
    adapter = module.RagSearchSpace.from_yaml(config)
    names = adapter.ss.parameters
    for stage in ("semantic_retrieval", "lexical_retrieval", "hybrid_retrieval"):
        assert f"pipeline__sequential__{stage}__top_k" in names
    assert "pipeline__sequential__hybrid_retrieval__hybrid_rrf__weight" in names
    _, lines = adapter.resolve_pipeline_block(config, {"pipeline": {"sequential": {
        "semantic_retrieval": {"top_k": 32},
        "lexical_retrieval": {"top_k": 17},
        "hybrid_retrieval": {"top_k": 5, "weight": 60},
    }}})
    assert [node["top_k"] for node in lines[0]["nodes"][:3]] == [32, 17, 5]
    assert lines[0]["nodes"][2]["modules"][0]["weight"] == 60
    with pytest.raises(NotImplementedError, match="BM25 analytical"):
        module.RagSearchSpace.from_yaml(hybrid_config("cost_model"))


def test_preview_can_resolve_before_cm_support_without_changing_the_target(monkeypatch):
    from rag_stack.search_space import rag_search_space as module

    config = hybrid_config("cost_model")
    original = deepcopy(config)
    observed = []
    monkeypatch.setattr(module, "extract_system_params", lambda c, *args: observed.append(c["system"]) or [])
    adapter = module.RagSearchSpace.from_yaml(config, check_performance_support=False)
    assert "pipeline__sequential__lexical_retrieval__top_k" in adapter.ss.parameters
    assert observed and all(system["performance_source"] == "cost_model" for system in observed)
    assert config == original
    # Deferring pricing does not relax the serving workflow's structure.
    _nodes(config).pop(1)
    with pytest.raises(ValueError, match="exactly one lexical_retrieval"):
        module.RagSearchSpace.from_yaml(config, check_performance_support=False)


def test_validator_reports_bm25_model_gap_but_allows_measured_non_power_two(monkeypatch):
    from rag_stack.config_validator import ConfigValidator

    for source in ("measured", "cost_model"):
        validator = ConfigValidator.__new__(ConfigValidator)
        validator.config = hybrid_config(source)
        validator.errors = []
        validator.gt_backend = "static_gt"
        validator.performance_source = source
        monkeypatch.setattr(validator, "_check_generator_node", lambda *args: None)
        monkeypatch.setattr(validator, "_check_vllm_chat_contract", lambda *args: None)
        monkeypatch.setattr(validator, "_check_cost_model_mapping", lambda *args: None)
        validator.config["algo_search_space"]["vectordb"] = [{"name": "dense"}]
        validator._check_pipeline_blocks()
        if source == "measured":
            assert validator.errors == []
        else:
            assert len(validator.errors) == 2
            assert "BM25 analytical" in validator.errors[0]


def test_plan_preserves_workflow_when_cost_model_unavailable():
    from rag_stack.server.routers.rag_project import plan_pipeline, ValidateConfigRequest

    result = plan_pipeline(ValidateConfigRequest(config=hybrid_config("cost_model")))
    assert result["valid"] is False
    assert result["capabilities"]["execution_supported"] is True
    assert result["capabilities"]["measured_supported"] is True
    assert result["capabilities"]["cost_model_supported"] is False
    assert result["workflow"]["nodes"][2]["inputs"] == {
        "dense": ["semantic_retrieval"], "lexical": ["lexical_retrieval"],
    }
    assert "lexical_retrieval" in result["workflow"]["stages"]
    assert "retrieval_fusion" in result["workflow"]["stages"]


def test_unknown_measured_capability_is_not_advertised_as_supported():
    from rag_stack.server.routers.rag_project import plan_pipeline, ValidateConfigRequest

    config = hybrid_config()
    _nodes(config)[-1]["modules"][0]["component"] = "openai_llm"
    assert plan_pipeline(ValidateConfigRequest(config=config))["capabilities"]["measured_supported"] is None


def test_measured_layout_excludes_lexical_and_fusion_cpu_operations():
    from rag_stack.search_space.placement import measured_gpu_stages

    assert measured_gpu_stages(hybrid_config()) == [
        "semantic_retrieval_encode", "generator_prefill", "generator_decode",
    ]


@pytest.mark.parametrize("observed", [False, True])
def test_measured_layout_accepts_hybrid_candidate_reduction_without_cm_dispatch(monkeypatch, observed):
    from rag_stack.performance_context import PerformanceContext
    from rag_stack.rag_ir import normalize
    from rag_stack import system_layout
    from rag_stack.types import SystemConfig

    config = hybrid_config()
    pipeline = deepcopy(_block(config))
    nodes = pipeline["node_lines"][0]["nodes"]
    for node, top_k in zip(nodes[:3], (32, 24, 8)):
        node["top_k"] = top_k
    nodes[2]["modules"][0]["weight"] = 60
    nodes.insert(3, {"stage": "passage_reranker", "top_k": 4,
                     "modules": [{"component": "sentence_transformer_reranker"}]})
    if observed:
        pipeline["_evaluator_fusion_calls"] = [{
            "stage": "retrieval_fusion", "model_id": "rrf_pure",
            "input_profile": {"list_sizes": [32, 24]},
            "output_profile": {"items": 8},
        }]

    def forbidden(*args, **kwargs):
        pytest.fail("Measured layout must not consult CM dispatch or its shape constraints")

    monkeypatch.setattr(normalize, "dispatch_manifest", forbidden)
    monkeypatch.setattr(system_layout, "cm_stage_names_from_pipeline", forbidden)
    context = PerformanceContext.__new__(PerformanceContext)
    context.hardware = SystemConfig(num_cores=8)
    context.config = {"system": {
        "performance_source": "measured",
        "system_design_space": {"available_gpus": ["cuda:0"],
                                "batch_size_request": 1, "batch_size_decode": 1},
        "cm_search_space": {"min_num_gpus": 1, "max_num_gpus": 1},
        "retrieval": {"faiss_num_threads": 1},
    }}
    system = context.resolve_system_config({}, resolved_config=pipeline)
    stages = system["layout"]["performance_stages"]
    assert stages["retrieval_fusion"]["kind"] == "cpu_retrieval"
    assert stages["retrieval_fusion"]["num_chips"] == 0
    lexical = stages["lexical_retrieval"]
    assert lexical["kind"] == "cpu_retrieval"
    assert lexical["num_chips"] == 0 and lexical["devices"] == []
    assert not {"tp", "pp", "num_servers", "replicas", "cost"} & lexical.keys()
    lexical_group = next(group for group in system["layout"]["resource_groups"]
                         if "lexical_retrieval" in group["performance_stages"])
    assert lexical_group["devices"] == [] and "num_servers" not in lexical_group
    assert system_layout.performance_stage_devices(system, "lexical_retrieval") == []
    assert system_layout.performance_stage_num_chips(system, "lexical_retrieval") == 0
    order = system["layout"]["performance_stage_order"]
    assert order.index("semantic_retrieval_vectorsearch") < order.index("lexical_retrieval") < order.index("retrieval_fusion") < order.index("passage_reranker")
    assert stages["passage_reranker"]["devices"] == ["cuda:0"]
    assert "lexical_retrieval" not in system["layout"]["engines"]
    assert all("lexical_retrieval" not in occupants
               for occupants in system["layout"]["gpu_occupants"].values())
    _, chips, _ = system_layout.collocation_and_chips(system, order)
    assert "lexical_retrieval" not in chips


def test_documented_hybrid_config_validates_without_model_downloads():
    from pathlib import Path
    import yaml
    from rag_stack.config_validator import ConfigValidator
    from rag_stack.search_space.rag_search_space import RagSearchSpace

    path = REPO_ROOT / "configs/rag_stack/hybrid_sequential.yaml"
    config = yaml.safe_load(path.read_text())
    assert ConfigValidator(config, check_model_cache=False).validate() == []
    space = RagSearchSpace.from_yaml(config)
    assert "pipeline__sequential__hybrid_retrieval__hybrid_rrf__weight" in space.ss.parameters


def test_invalid_hybrid_cost_model_config_never_discovers_model_cache(monkeypatch):
    from pathlib import Path
    import yaml
    from rag_stack.config_validator import ConfigValidator

    path = REPO_ROOT / "configs/rag_stack/hybrid_sequential.yaml"
    config = yaml.safe_load(path.read_text())
    config["system"]["performance_source"] = "cost_model"
    validator = ConfigValidator(config)
    monkeypatch.setattr(validator, "_check_hf_model_cache", lambda: pytest.fail("Invalid config must not download models"))
    assert any("BM25 analytical" in error for error in validator.validate())


def test_registry_exposes_canonical_names_aliases_and_retrieval_capabilities():
    from rag_stack.server.routers.rag_project import get_node_types

    stages = get_node_types()["stages"]
    assert "dense_retrieval" not in stages
    assert "sparse_retrieval" not in stages
    assert stages["semantic_retrieval"]["aliases"] == ["dense_retrieval"]
    assert stages["lexical_retrieval"]["aliases"] == ["sparse_retrieval"]
    bm25 = stages["lexical_retrieval"]["component_capabilities"]["bm25"]
    assert bm25["measured_supported"] and not bm25["cost_model_supported"]
    rrf = stages["hybrid_retrieval"]["component_parameters"]["hybrid_rrf"]["weight"]
    assert rrf["default"] == 60 and rrf["label"] == "RRF rank constant"
    fusion = stages["hybrid_retrieval"]["component_capabilities"]["hybrid_rrf"]
    assert fusion["measured_supported"] and not fusion["cost_model_supported"]
    assert "analytical scheduling/cost binding" in fusion["reason"]


def test_planning_does_not_import_optimizer_evaluator_or_cost_model():
    script = """
import sys
from rag_stack.server.routers.rag_project import plan_pipeline, ValidateConfigRequest
result = plan_pipeline(ValidateConfigRequest(config=CONFIG))
assert result['workflow']['nodes']
for prefix in ('ax', 'rag_stack_evaluator', 'rag_stack.cost_model'):
    assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules), prefix
""".replace("CONFIG", repr(hybrid_config()))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_trace_api_reads_only_valid_saved_quality_traces(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from rag_stack.server.routers import rag_project
    from rag_stack.rag_ir.plan import describe_pipeline

    project = tmp_path / "demo"
    artifacts = project / "evaluations" / "eval_0001"
    artifacts.mkdir(parents=True)
    monkeypatch.setattr(rag_project, "_get_project_dir", lambda: tmp_path)
    assert rag_project.evaluation_trace(1, "demo")["available"] is False
    trace = {"queries": [{"question_id": "q1", "calls": [{
        "stage": "generator", "input_tokens": 8, "output_tokens": 2, "step_idx": 0,
    }]}]}
    path = artifacts / "execution_dag.json"
    path.write_text(json.dumps(trace))
    assert rag_project.evaluation_trace(1, "demo")["trace"] == trace
    trace["provenance"] = {"workflow_schema": describe_pipeline(hybrid_config())}
    path.write_text(json.dumps(trace))
    with pytest.raises(HTTPException) as error:
        rag_project.evaluation_trace(1, "demo")
    assert error.value.status_code == 422
    assert "requires observed fusion" in error.value.detail
    trace.pop("provenance")
    trace["queries"][0]["calls"][0]["stage"] = "lexical_retrieval"
    path.write_text(json.dumps(trace))
    with pytest.raises(HTTPException) as error:
        rag_project.evaluation_trace(1, "demo")
    assert error.value.status_code == 422
    with pytest.raises(HTTPException) as error:
        rag_project.evaluation_trace(1, "../demo")
    assert error.value.status_code == 400


def test_trace_api_rejects_artifact_symlinks_outside_project(tmp_path, monkeypatch):
    from fastapi import HTTPException
    from rag_stack.server.routers import rag_project

    artifacts = tmp_path / "demo" / "evaluations" / "eval_0001"
    artifacts.mkdir(parents=True)
    outside = tmp_path / "other.json"
    outside.write_text("{}")
    (artifacts / "execution_dag.json").symlink_to(outside)
    monkeypatch.setattr(rag_project, "_get_project_dir", lambda: tmp_path)
    with pytest.raises(HTTPException) as error:
        rag_project.evaluation_trace(1, "demo")
    assert error.value.status_code == 400
