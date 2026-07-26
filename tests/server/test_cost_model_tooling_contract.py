"""Apps tooling must expose the production component/calibration contracts."""

import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from rag_stack.cost_model.base import CostModelModule
from rag_stack.cost_model.types import CostModelResult
from rag_stack.server import cost_model_api as api
from rag_stack.server.dependencies import get_cost_model
from rag_stack.server.schemas import AppSettings


class Component(CostModelModule):
    def estimate(self, config, hardware):
        if "error" in config:
            raise ValueError("Missing exact calibration for selected hardware")
        if "unsupported" in config:
            raise NotImplementedError("No component estimate for this operation")
        return CostModelResult(latency_s=config.get("latency", 0.2), throughput_qps=5)


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(api.base_router, prefix="/api/cost-model")
    app.dependency_overrides[get_cost_model] = Component
    monkeypatch.setattr(api, "parse_hardware", lambda hardware: hardware)
    return TestClient(app)


def test_component_results_never_claim_compiled_simulation(client):
    result = client.post("/api/cost-model/example/estimate", json={"hardware": {}, "pipeline_config": {}})
    assert result.status_code == 200
    assert result.json()["scope"] == "component_estimate"
    assert result.json()["latency_s"] == 0.2


@pytest.mark.parametrize("config,status,detail", [
    ({"error": True}, 422, "Missing exact calibration"),
    ({"unsupported": True}, 501, "No component estimate"),
    ({"latency": 0}, 422, "finite positive"),
])
def test_component_errors_are_explicit_and_never_zero_fallbacks(client, config, status, detail):
    result = client.post("/api/cost-model/example/estimate", json={"hardware": {}, "pipeline_config": config})
    assert result.status_code == status
    assert detail in result.json()["detail"]
    assert "latency_s" not in result.json()


def test_failed_batch_does_not_return_partial_success(client):
    result = client.post("/api/cost-model/example/estimate-batch", json={"hardware": {}, "configs": [{}, {"error": True}]})
    assert result.status_code == 422
    assert "results" not in result.json()


def test_profile_browser_reads_runtime_root_and_current_envelope(monkeypatch, tmp_path):
    runtime = tmp_path / "runtime"
    unrelated = tmp_path / "apps-setting"
    folder = runtime / "llm_sim"
    folder.mkdir(parents=True)
    document = {"profiles": {
        "agnostic__embedding__bf16": {"component": "agnostic", "model_key": "embedding", "compute_efficiency": 0.5},
        "tart__t5__bf16": {"component": "tart", "model_key": "t5", "compute_efficiency": 0.6},
    }}
    path = folder / "gpu.json"
    path.write_text(json.dumps(document))
    monkeypatch.setenv("RAG_STACK_CALIBRATION_PROFILE_DIR", str(runtime))
    monkeypatch.setenv("RAG_STACK_CALIBRATION_DATA_DIR", str(tmp_path / "frozen"))
    monkeypatch.setattr(api, "load_settings", lambda: AppSettings(calibration_profiles_dir=str(unrelated)))

    rows = api.list_cost_model_profiles("retrieval_encode_sim")["profiles"]
    assert len(rows) == 1
    assert rows[0]["record_key"] == "agnostic__embedding__bf16"
    assert rows[0]["match_scope"] == "shared_model_candidate"
    assert rows[0]["can_delete"] is False
    assert api.get_cost_model_profile("retrieval_encode_sim", "gpu.json") == document
    status = api.get_unified_calibration_status()
    assert status["scope"] == "artifact_inventory"
    assert status["profiles_root"] == str(runtime)
    assert status["data_root"] == str(tmp_path / "frozen")
    assert status["profiles"][0]["record_count"] == 2
    assert status["configured_profiles_root"] == str(unrelated)

    with pytest.raises(HTTPException) as caught:
        api.delete_cost_model_profile("retrieval_encode_sim", "gpu.json")
    assert caught.value.status_code == 409
    assert json.loads(path.read_text()) == document


def test_legacy_llm_profile_shape_is_reported_not_rendered(monkeypatch, tmp_path):
    folder = tmp_path / "llm_sim"
    folder.mkdir()
    (folder / "gpu.json").write_text(json.dumps({"agnostic__old__bf16": {"component": "agnostic"}}))
    monkeypatch.setenv("RAG_STACK_CALIBRATION_PROFILE_DIR", str(tmp_path))
    rows = api.list_cost_model_profiles("llm_sim")["profiles"]
    assert len(rows) == 1
    assert "envelope fields" in rows[0]["error"]


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "system.json"
    path.write_text("{}")
    return str(path)


@pytest.mark.parametrize("command", [
    "status", "plan", "all-faiss", "llmlingua2", "generator-disagg measure", "generator-disagg fit",
])
def test_current_dispatch_commands_and_staging_roots_are_forwarded(command, config, tmp_path):
    args = api._build_unified_calibration_args(api.UnifiedCalibrationRunRequest(
        command=command, config_path=config,
        data_dir=str(tmp_path / "measurements"), profile_dir=str(tmp_path / "profiles"),
    ))
    assert args[:len(command.split())] == command.split()
    assert args[args.index("--config") + 1] == config
    assert args[args.index("--data-dir") + 1] == str(tmp_path / "measurements")
    assert args[args.index("--profile-dir") + 1] == str(tmp_path / "profiles")


def test_workload_requires_external_project_and_passes_data_paths(config, tmp_path):
    with pytest.raises(HTTPException, match="project_dir"):
        api._build_unified_calibration_args(api.UnifiedCalibrationRunRequest(command="workload", config_path=config))
    qa = tmp_path / "qa.parquet"
    corpus = tmp_path / "corpus.parquet"
    qa.touch()
    corpus.touch()
    args = api._build_unified_calibration_args(api.UnifiedCalibrationRunRequest(
        command="workload", config_path=config, project_dir=str(tmp_path / "cache"),
        qa_data=str(qa), corpus_data=str(corpus),
    ))
    assert args[args.index("--project-dir") + 1] == str(tmp_path / "cache")
    assert args[args.index("--qa-data") + 1] == str(qa)
    assert args[args.index("--corpus-data") + 1] == str(corpus)


@pytest.mark.parametrize("command,fields,detail", [
    ("fit ivf", {"force": True}, "force"),
    ("llmlingua2", {"qa_data": "unused"}, "qa_data"),
    ("generator-disagg", {}, "measure or fit"),
    ("pd", {}, "Unsupported calibration command"),
    ("e2e", {}, "Unsupported calibration command"),
    ("ivf", {"maxiter": 0}, "positive"),
])
def test_invalid_launcher_fields_fail_before_a_job_is_spawned(command, fields, detail, config, monkeypatch):
    monkeypatch.setattr(api, "launch_job", lambda **kwargs: pytest.fail("Invalid request launched a job"))
    with pytest.raises(HTTPException) as caught:
        api.run_unified_calibration_job(api.UnifiedCalibrationRunRequest(command=command, config_path=config, **fields))
    assert caught.value.status_code == 400
    assert detail in caught.value.detail


def test_generic_llm_calibration_without_runner_fails_before_spawn(config, monkeypatch):
    monkeypatch.setattr(api, "launch_job", lambda **kwargs: pytest.fail("Unsupported generic calibration launched"))
    with pytest.raises(HTTPException) as caught:
        api.run_cost_model_job("llm_sim", api.CalibrationRunRequest(kind="calibrate", module="vllm", config_path=config))
    assert caught.value.status_code == 501
    assert "dedicated" in caught.value.detail


def test_command_descriptors_match_launcher_fields(config, tmp_path):
    for command in api.calibration_commands():
        args = api._build_unified_calibration_args(api.UnifiedCalibrationRunRequest(
            command=command["value"],
            config_path=config if command["needs_config"] else "",
            force=command["supports_force"],
            maxiter=3 if command["supports_maxiter"] else None,
            project_dir=str(tmp_path / "cache") if command["requires_project_dir"] else "",
            extra_args=["--cost-model", "passage_reranker_sim", "--module", "tart"] if command["required_extra_args"] else [],
        ))
        assert args[0] == command["value"].split()[0]


def test_llm_launcher_requires_runner_identity_before_spawn(config, monkeypatch):
    monkeypatch.setattr(api, "launch_job", lambda **kwargs: pytest.fail("Missing runner identity launched a job"))
    with pytest.raises(HTTPException, match="--cost-model"):
        api.run_unified_calibration_job(api.UnifiedCalibrationRunRequest(command="llm", config_path=config))


def test_inline_fields_obey_same_contract_as_form_fields(config):
    for command, extra in [("fit ivf", ["--force"]), ("ivf", ["--maxiter", "0"]), ("ivf", ["--profile-dir"] )]:
        with pytest.raises(HTTPException) as caught:
            api._build_unified_calibration_args(api.UnifiedCalibrationRunRequest(command=command, config_path=config, extra_args=extra))
        assert caught.value.status_code == 400
