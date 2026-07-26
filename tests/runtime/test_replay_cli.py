"""Replay uses two resolved configs and an independently selected dataset."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from rag_stack import replay_cli
from rag_stack.evaluation import EvaluationResult
from rag_stack.rag_ir import make_quality_trace_envelope


@pytest.fixture
def inputs(tmp_path):
    algo = {
        "node_lines": [{"node_line_name": "generation", "nodes": [
            {"stage": "generator", "modules": [{
                "component": "vllm_api", "model": "test/chosen-model",
                "max_tokens": 19, "temperature": 0.2,
            }]},
        ]}],
        "vectordb": [], "corpus_runtime": {"chunker": {}},
        "eval_backend_setting": {"metrics": ["retrieval_token_f1"]},
    }
    system = {"layout": {"available_devices": ["cuda:0"]}}
    paths = [tmp_path / "algo_config.json", tmp_path / "system_config.json"]
    for path, payload in zip(paths, (algo, system)):
        path.write_text(json.dumps(payload))
    qa, corpus = tmp_path / "qa.parquet", tmp_path / "corpus.parquet"
    qa.touch()
    corpus.touch()
    return algo, system, *paths, qa, corpus


def test_dry_run_keeps_dataset_external_and_does_not_import_models(inputs, tmp_path):
    _, _, algo_path, system_path, qa, corpus = inputs
    script = """
import sys
from rag_stack.replay_cli import main
assert main(sys.argv[1:]) == 0
assert 'torch' not in sys.modules
assert 'rag_stack_evaluator.static_rag_evaluator' not in sys.modules
"""
    output = tmp_path / "output"
    completed = subprocess.run([
        sys.executable, "-c", script, str(algo_path), str(system_path),
        "--qa", str(qa), "--corpus", str(corpus), "--dry-run", "--output-dir", str(output),
    ], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert not output.exists()
    assert "dataset" not in json.loads(algo_path.read_text())


@pytest.mark.parametrize("mutation, message", [
    (lambda algo: algo.update(dataset={"qa": "old-training.parquet"}), "must not bind a dataset"),
    (lambda algo: algo.update(algo_search_space={}), "not optimizer settings"),
    (lambda algo: algo.update(cm_search_space={}), "not optimizer settings"),
    (lambda algo: algo.update(system={"cpu": "epyc_9124"}), "belong in system_config"),
    (lambda algo: algo["node_lines"][0]["nodes"][0]["modules"].append({"component": "vllm"}),
     "exactly one module"),
])
def test_replay_rejects_ambiguous_inputs(inputs, mutation, message):
    algo, _, algo_path, system_path, _, _ = inputs
    mutation(algo)
    algo_path.write_text(json.dumps(algo))
    with pytest.raises(ValueError, match=message):
        replay_cli.load_replay_inputs(algo_path, system_path)


@pytest.mark.parametrize("field", ["dataset", "cm_search_space", "deployment_presets"])
def test_replay_rejects_dataset_or_search_settings_in_system(inputs, field):
    _, system, algo_path, system_path, _, _ = inputs
    system[field] = {}
    system_path.write_text(json.dumps(system))
    with pytest.raises(ValueError):
        replay_cli.load_replay_inputs(algo_path, system_path)


def test_runtime_projects_selected_choices_for_dataset_owner(inputs, tmp_path, monkeypatch):
    algo, _, _, _, qa, corpus = inputs
    algo["corpus_runtime"] = {"chunker": {"component": "character", "chunk_size": 256}}
    original = deepcopy(algo)
    seen = {}

    class Manager:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self.config = kwargs["config"]

        def inject_corpus_size(self):
            pass

        def inject_token_stats(self):
            self.config["_token_stats"] = {"avg_chunk_tokens": 42}

    fake_module = ModuleType("rag_stack.dataset_manager")
    fake_module.DatasetManager = Manager
    monkeypatch.setitem(sys.modules, "rag_stack.dataset_manager", fake_module)
    monkeypatch.setattr("rag_stack.evaluation.create_quality_evaluator", lambda **kwargs: kwargs)
    manager, evaluator, backend = replay_cli._runtime(
        algo, tmp_path, qa_path=qa, corpus_path=corpus, dataset_name="caller-corpus",
    )
    from rag_stack.cost_model.token_stats import TokenStats
    from rag_stack.search_space.rag_search_space import algo_vectordb_blocks
    assert TokenStats._find_generator_model(seen["config"]) == "test/chosen-model"
    assert algo_vectordb_blocks(seen["config"]) == []
    assert seen["config"]["algo_search_space"]["corpus"] == original["corpus_runtime"]
    assert seen["qa_data_path"] == str(qa)
    assert seen["corpus_data_path"] == str(corpus)
    assert seen["config"]["dataset"] == {"dataset_name": "caller-corpus"}
    assert "dataset" not in algo and "algo_search_space" not in algo
    assert algo["_token_stats"] == {"avg_chunk_tokens": 42}
    assert evaluator["dataset_manager"] is manager and backend == "static_gt"


@pytest.mark.parametrize("stage", ["generator", "query_expansion"])
@pytest.mark.parametrize("backend", ["vllm_api", "vllm"])
@pytest.mark.parametrize("sampling", [{}, {"max_tokens": 19, "temperature": 0.2}])
def test_measured_provider_defaults_preserve_resolved_sampling(stage, backend, sampling):
    from rag_stack_evaluator.static_rag_evaluator.measured.evaluator import (
        apply_measured_generation_defaults,
    )

    module = {"model": "fixture", **sampling}
    if stage == "generator":
        module["component"] = backend
    else:
        module.update(component="query_decompose", generator_backend=backend)
    if backend == "vllm_api":
        module["uri"] = "http://source-gateway"
    source = {"node_lines": [{"nodes": [{"stage": stage, "modules": [module]}]}]}
    original = deepcopy(source)

    projected = replay_cli._project_measured_algo(source)
    # Exercise the real provider's final policy, not merely the host projection.
    effective = apply_measured_generation_defaults(projected)
    actual = effective["node_lines"][0]["nodes"][0]["modules"][0]
    expected_cap = sampling.get("max_tokens", 4096 if backend == "vllm_api" else 512)
    expected_temperature = sampling.get("temperature", 0.4 if backend == "vllm_api" else 1.0)
    assert actual["max_tokens"] == expected_cap
    assert actual["temperature"] == expected_temperature
    assert actual.get("generator_backend", actual["component"]) == "vllm"
    assert "uri" not in actual and "_measured_source_max_tokens" not in actual
    assert source == original


@pytest.mark.parametrize("mode", ["quality", "cm", "measured"])
def test_real_boundaries_receive_resolved_configs_and_fresh_trace(inputs, tmp_path, monkeypatch, mode):
    algo, system, _, _, qa, corpus = inputs
    if mode == "quality":
        algo["eval_backend_setting"]["performance_only"] = True
    originals = deepcopy((algo, system))
    observed = {}
    trace = make_quality_trace_envelope([[{
        "stage": "generator", "model_id": "test/chosen-model",
        "input_tokens": 12, "output_tokens": 3, "step_idx": 0,
    }]], question_ids=["q1"])

    class Evaluator:
        def evaluate(self, config, *, run_dir, metrics_override):
            observed.update(config=config, metrics_override=metrics_override)
            score_file = Path(run_dir) / "judge_scores.json"
            score_file.parent.mkdir(parents=True)
            score_file.write_text(json.dumps({"scores": [0.75]}))
            return EvaluationResult(metrics={"answer_quality": 0.75}, trace=trace)

    def runtime(config, project, *, qa_path, corpus_path, dataset_name=None):
        observed.update(qa=qa_path, corpus=corpus_path)
        return SimpleNamespace(qa_data={"qid": SimpleNamespace(tolist=lambda: ["q1"])}), Evaluator(), "static_gt"

    def price(config, deployment, observed_trace, *, dataset_name=None):
        observed.update(price_config=config, system=deployment, trace=observed_trace)
        return {"status": "ok", "performance_score": 2.0}

    def measure(config, deployment, evaluator, project):
        observed.update(config=config, system=deployment)
        return {"status": "ok", "performance_score": 3.0}

    monkeypatch.setattr(replay_cli, "_runtime", runtime)
    monkeypatch.setattr(replay_cli, "_price", price)
    monkeypatch.setattr(replay_cli, "_measure", measure)
    result = replay_cli.evaluate_pair(algo, system, mode, tmp_path, qa_path=qa, corpus_path=corpus)
    assert (algo, system) == originals
    assert observed["qa"] == qa and observed["corpus"] == corpus
    assert "dataset" not in observed["config"]
    assert observed["config"]["system"] == system
    if mode == "cm":
        assert observed["trace"] is trace
        assert observed["metrics_override"] == []
        assert observed["price_config"]["_runtime"]["rag_ir_mode"] == "dynamic"
    elif mode == "quality":
        assert "performance_only" not in observed["config"]["eval_backend_setting"]
        assert observed["metrics_override"] is None
        assert result["metrics"] == {"answer_quality": 0.75}
        assert result["per_query_scores"] == {"judge_scores.json": [0.75]}
        assert result["question_ids"] == ["q1"]
    else:
        module = observed["config"]["node_lines"][0]["nodes"][0]["modules"][0]
        assert module["component"] == "vllm" and module["max_tokens"] == 19
        assert observed["config"]["eval_backend_setting"] == {"performance_only": True}


@pytest.mark.parametrize("fail", [False, True])
def test_replay_publishes_result_and_events_and_cleans_scratch(inputs, tmp_path, monkeypatch, fail):
    _, _, algo_path, system_path, qa, corpus = inputs
    monkeypatch.setenv("PROJECT_DIR", "/original/project")
    scratch_paths = []

    def evaluate(algo, system, mode, project, *, qa_path, corpus_path, dataset_name=None):
        assert qa_path == qa and corpus_path == corpus
        assert dataset_name == "caller-corpus"
        assert os.environ["PROJECT_DIR"] == str(project)
        scratch_paths.append(project)
        (project / "intermediate.txt").write_text("temporary")
        if fail:
            raise RuntimeError("test failure")
        return {"status": "ok", "metrics": {"quality": 0.8}}

    monkeypatch.setattr(replay_cli, "evaluate_pair", evaluate)
    output = tmp_path / "published"
    status = replay_cli.run_replay(algo_path, system_path, qa_path=qa, corpus_path=corpus,
                                   mode="quality", output=output, dataset_name="caller-corpus")
    assert status == int(fail)
    assert os.environ["PROJECT_DIR"] == "/original/project"
    assert all(not path.exists() for path in scratch_paths)
    assert {p.name for p in output.iterdir()} == {"quality_result.json", "RUN.md"}
    result = json.loads((output / "quality_result.json").read_text())
    assert result["status"] == ("error" if fail else "ok")
    assert f"--qa {qa}" in (output / "RUN.md").read_text()
    assert f"--corpus {corpus}" in (output / "RUN.md").read_text()
    assert "--dataset-name caller-corpus" in (output / "RUN.md").read_text()
    assert not any("sha" in key for key in result)


def test_missing_external_dataset_fails_before_creating_output(inputs, tmp_path):
    _, _, algo_path, system_path, qa, corpus = inputs
    qa.unlink()
    output = tmp_path / "published"
    with pytest.raises(ValueError, match="Dataset file does not exist"):
        replay_cli.run_replay(algo_path, system_path, qa_path=qa, corpus_path=corpus,
                              mode="quality", output=output)
    assert not output.exists()


def test_quality_without_scoring_protocol_fails_before_model_loading(inputs, tmp_path, monkeypatch):
    algo, system, _, _, qa, corpus = inputs
    algo["eval_backend_setting"] = {"performance_only": True}
    monkeypatch.setattr(replay_cli, "_runtime", lambda *args, **kwargs: pytest.fail("models loaded"))
    with pytest.raises(ValueError, match="contains no scoring protocol"):
        replay_cli.evaluate_pair(algo, system, "quality", tmp_path, qa_path=qa, corpus_path=corpus)


def test_cm_requires_explicit_faiss_corpus_identity_before_loading_models(inputs, tmp_path, monkeypatch):
    algo, system, _, _, qa, corpus = inputs
    algo["vectordb"] = [{"name": "index", "db_type": "faiss_hnsw"}]
    monkeypatch.setattr(replay_cli, "_runtime", lambda *args, **kwargs: pytest.fail("models loaded"))
    with pytest.raises(ValueError, match="requires --dataset-name"):
        replay_cli.evaluate_pair(algo, system, "cm", tmp_path, qa_path=qa, corpus_path=corpus)


def test_cm_binds_corpus_identity_only_in_pricing_context(inputs, monkeypatch):
    algo, system, *_ = inputs
    observed = {}
    trace = {"queries": []}

    class Assembly:
        def __init__(self, hardware):
            observed["hardware"] = hardware

        def evaluate_fixed(self, pipeline, deployment, *, trace):
            observed.update(pipeline=pipeline, deployment=deployment, trace=trace)
            return SimpleNamespace(performance_score=8.0, best_hw_params={"batch": 4})

    monkeypatch.setattr("rag_stack.cost_model.assembly.RAGCMAssembly", Assembly)
    monkeypatch.setattr("rag_stack.types.SystemConfig.from_dict", lambda config: config)
    result = replay_cli._price(algo, system, trace, dataset_name="caller-corpus")
    assert observed["pipeline"]["dataset"] == {"dataset_name": "caller-corpus"}
    assert "dataset" not in algo
    assert observed["trace"] is trace and observed["deployment"] is system
    assert result["performance_score"] == 8.0
