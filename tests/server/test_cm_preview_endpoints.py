"""Project preview HTTP/path contracts; no model or subprocess execution."""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import yaml

from rag_stack import cm_preview
from rag_stack.server.routers import cm_preview as routes, rag_project
from rag_stack.rag_ir import make_quality_trace_envelope


@pytest.fixture
def client(tmp_path, monkeypatch):
    project = tmp_path / "projects" / "chosen"
    project.mkdir(parents=True)
    dataset = tmp_path / "datasets"
    dataset.mkdir()
    (dataset / "qa.parquet").touch()
    (dataset / "corpus.parquet").touch()
    (project / "rag_stack_config.yaml").write_text(yaml.safe_dump({
        "dataset": {"dataset_name": "fixture", "qa": "qa.parquet", "corpus": "corpus.parquet"}}))
    evaluation = project / "evaluations" / "eval_0000"
    evaluation.mkdir(parents=True)
    (evaluation / "pipeline_config.json").write_text(json.dumps({"node_lines": [{"nodes": []}]}))
    (evaluation / "system_config.json").write_text(json.dumps({"layout": {"available_devices": ["cuda:0"]}}))
    (evaluation / "execution_dag.json").write_text(json.dumps(make_quality_trace_envelope(
        [[{"stage": "generator", "input_tokens": 1, "output_tokens": 1, "step_idx": 0}]], question_ids=["q1"])))
    monkeypatch.setattr(rag_project, "_get_project_dir", lambda: project.parent)
    monkeypatch.setattr(rag_project, "_get_datasets_dir", lambda: dataset)
    monkeypatch.setattr(cm_preview, "launch_worker", lambda path: None)
    app = FastAPI()
    app.include_router(routes.router, prefix="/api/rag-project")
    return TestClient(app), project


BASE = "/api/rag-project/projects/chosen/cm-previews"


def test_create_list_reopen_cancel_and_download(client):
    api, project = client
    response = api.post(BASE, json={"eval_id": 0, "reuse_trace": True})
    assert response.status_code == 202, response.text
    result = response.json()
    assert result["status"] == "queued" and result["skip_judge"] is True
    assert result["cohort"]["requested_count"] == 8
    path = f"{BASE}/{result['preview_id']}"
    assert api.get(BASE).json()["previews"][0]["preview_id"] == result["preview_id"]
    assert api.get(path).json()["run_id"] == result["run_id"]
    assert api.get(path + "/artifacts/algo_config.json").status_code == 200
    assert api.get(path + "/artifacts/source_trace.json").json()["queries"][0]["question_id"] == "q1"
    assert api.get(path + "/artifacts/unlisted.json").status_code == 400
    assert api.post(path + "/cancel").json()["status"] == "cancelled"
    assert not (project / "summary.json").exists()


def test_saved_preview_description_is_read_only_and_executable_artifact_is_not_exposed(client, monkeypatch):
    api, project = client
    result = api.post(BASE, json={"eval_id": 0, "reuse_trace": True}).json()
    path = f"{BASE}/{result['preview_id']}"
    directory = project / "cm_previews" / result["preview_id"]
    description = {"schema_version": "rag-cm.compiled-plan.v1", "status": "compiled",
                   "queries": [{"question_id": "q1", "operations": []}], "prediction": None}
    cm_preview.write_json(directory / "compiled_description.json", description)
    # An obsolete executable-plan artifact must not become a preview input.
    cm_preview.write_json(directory / "compiled_plan.json", {"restore_inputs": {"obsolete": True}})
    cm_preview.update_state(directory, status="succeeded", phase="complete")
    def must_not_execute(*args, **kwargs):
        pytest.fail("Viewing a saved description must not execute the evaluator or compiler")
    monkeypatch.setattr(cm_preview, "_compile", must_not_execute)
    monkeypatch.setattr(cm_preview, "run_worker", must_not_execute)
    before = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    opened = api.get(path)
    assert opened.status_code == 200, opened.text
    assert opened.json()["compiled_plan"] == description
    assert opened.json()["compiled_path_status"] == "available"
    names = [item["name"] for item in opened.json()["artifacts"]]
    assert "compiled_description.json" in names
    assert "compiled_plan.json" not in names
    download = api.get(path + "/artifacts/compiled_description.json")
    assert download.status_code == 200, download.text
    assert download.json() == description
    assert api.get(path + "/artifacts/compiled_plan.json").status_code == 400
    assert {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()} == before


@pytest.mark.parametrize("body", [
    {}, {"eval_id": 0, "parameters": {}}, {"eval_id": -1},
    {"eval_id": 0, "sample_count": 0}, {"eval_id": 0, "sample_count": 129},
    {"parameters": {}, "reuse_trace": True}, {"eval_id": 0, "question_ids": [1, "1"]},
    {"eval_id": 0, "reuse_trace": True, "qa_data": "qa.parquet"},
])
def test_invalid_selection_is_rejected_before_job_creation(client, body):
    api, project = client
    assert api.post(BASE, json=body).status_code == 422
    assert not (project / "cm_previews").exists()


def test_dataset_and_project_symlink_escape_are_rejected(client, tmp_path):
    api, project = client
    outside = tmp_path / "private.parquet"
    outside.touch()
    response = api.post(BASE, json={"eval_id": 0, "qa_data": str(outside), "corpus_data": "corpus.parquet"})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_input"
    (project.parent / "escape").symlink_to(tmp_path)
    response = api.get("/api/rag-project/projects/escape/cm-previews")
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "invalid_project"


def test_new_project_missing_dataset_has_structured_error(client):
    api, project = client
    (project / "rag_stack_config.yaml").write_text('dataset: {}\n')
    response = api.post(BASE, json={"parameters": {}})
    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "dataset_required"
    assert response.json()["detail"]["details"]["field"] == "qa_data"


def test_options_reports_each_dataset_independently(client):
    api, project = client
    (project / "rag_stack_config.yaml").write_text('dataset: {qa: qa.parquet}\n')
    response = api.get(BASE + "/options")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["dataset"] == {"qa_available": True, "corpus_available": False, "dataset_name": None}
