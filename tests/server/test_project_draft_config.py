"""Draft configuration edits are allowed only before a project's first run."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from rag_stack.project_summary import (
    CONFIG_FILENAME, ProjectStatus, create_project_summary, save_project_summary,
)
from rag_stack.server.routers import rag_project
from rag_stack.project_config import CONFIG_FREEZE_FILENAME, freeze_project_config


@pytest.fixture
def draft(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "projects"
    project = root / "draft"
    project.mkdir(parents=True)
    config = rag_project.scaffold_config(rag_project.ScaffoldRequest(optimizer_type="sobol"))
    (project / CONFIG_FILENAME).write_text(config["content"])
    save_project_summary(str(project), create_project_summary())
    monkeypatch.setattr(rag_project, "_get_project_dir", lambda: root)
    return project, config


def test_http_draft_roundtrip_preserves_raw_yaml_and_preview_snapshot(draft):
    project, config = draft
    app = FastAPI()
    app.include_router(rag_project.router, prefix="/api/rag-project")
    client = TestClient(app)
    before = client.get("/api/rag-project/config/draft")
    assert before.status_code == 200
    assert before.json()["editable"] is True
    snapshot = project / "cm_previews" / "preview" / "source_config.yaml"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text(config["content"])
    content = "# My first experiment\n" + config["content"]
    response = client.put("/api/rag-project/config", json={
        "project_name": "draft", "content": content,
    })
    assert response.status_code == 200, response.text
    assert response.json() == {
        "ok": True, "project_name": "draft", "content": config["config"],
        "raw_content": content, "editable": True, "read_only_reason": None,
    }
    assert (project / CONFIG_FILENAME).read_text() == content
    assert snapshot.read_text() == config["content"]
    assert not list(project.glob(".config.*.tmp"))


def test_created_project_can_change_real_config_fields_before_first_eval(draft):
    project, config = draft
    replacement = dict(config["config"])
    replacement["global"] = {**replacement["global"], "n_iterations": 17}
    document = yaml.safe_dump(replacement, sort_keys=False)
    result = rag_project.save_project_config(rag_project.SaveProjectConfigRequest(
        project_name="draft", content=document,
    ))
    assert result["editable"] is True
    assert result["content"]["global"]["n_iterations"] == 17
    assert (project / CONFIG_FILENAME).read_text() == document
    assert not (project / CONFIG_FREEZE_FILENAME).exists()


@pytest.mark.parametrize("status", [
    ProjectStatus.RUNNING, ProjectStatus.FAILED, ProjectStatus.COMPLETED,
    ProjectStatus.QUALITY_COMPLETE,
])
def test_zero_evaluations_does_not_unlock_started_project(draft, status):
    project, config = draft
    save_project_summary(str(project), create_project_summary(status))
    assert rag_project.get_project_config("draft")["editable"] is False
    with pytest.raises(HTTPException) as exc:
        rag_project.save_project_config(rag_project.SaveProjectConfigRequest(
            project_name="draft", content=config["content"],
        ))
    assert exc.value.status_code == 409
    assert (project / CONFIG_FILENAME).read_text() == config["content"]


@pytest.mark.parametrize("artifact", [
    "evaluations/0/result.json", "optimizer/checkpoint.json",
    "logs/optimization_old.log", "all_evaluations.csv",
    "data/qa.parquet", "system_transfer.json",
])
def test_created_summary_cannot_hide_prior_run_artifacts(draft, artifact):
    project, _ = draft
    path = project / artifact
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("previous run")
    document = rag_project.get_project_config("draft")
    assert document["editable"] is False
    assert document["read_only_reason"]


@pytest.mark.parametrize("field,value", [("pid", 123), ("n_evaluations", 1), ("n_invalid_configs", 1)])
def test_created_summary_with_process_or_attempt_history_is_read_only(draft, field, value):
    project, _ = draft
    summary = create_project_summary()
    setattr(summary, field, value)
    save_project_summary(str(project), summary)
    assert rag_project.get_project_config("draft")["editable"] is False


@pytest.mark.parametrize("content", ["global: [broken", "[]\n", "unexpected: true\n"])
def test_invalid_yaml_or_schema_never_replaces_draft(draft, content):
    project, config = draft
    with pytest.raises(HTTPException) as exc:
        rag_project.save_project_config(rag_project.SaveProjectConfigRequest(
            project_name="draft", content=content,
        ))
    assert exc.value.status_code == 400
    assert "Invalid configuration" in exc.value.detail
    assert (project / CONFIG_FILENAME).read_text() == config["content"]


def test_save_uses_canonical_validator_without_model_downloads(draft, monkeypatch):
    _, config = draft
    recorded = {}

    class Validator:
        def __init__(self, config, *, check_model_cache):
            recorded.update(config=config, check_model_cache=check_model_cache)

        def validate(self):
            return []

    monkeypatch.setattr("rag_stack.config_validator.ConfigValidator", Validator)
    rag_project.save_project_config(rag_project.SaveProjectConfigRequest(
        project_name="draft", content=config["content"],
    ))
    assert recorded == {"config": config["config"], "check_model_cache": False}


def test_launch_waits_for_atomic_config_save(draft, tmp_path, monkeypatch):
    project, config = draft
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    for filename in ("qa.parquet", "corpus.parquet"):
        (datasets / filename).write_bytes(b"dataset")
    monkeypatch.setattr(rag_project, "_get_datasets_dir", lambda: datasets)
    replacement_started, allow_replacement, launch_started = Event(), Event(), Event()
    original_replace = rag_project.os.replace
    new_content = "# Saved before launch\n" + config["content"]

    def replace(source, target):
        if Path(target) == project / CONFIG_FILENAME:
            replacement_started.set()
            assert allow_replacement.wait(5)
        return original_replace(source, target)

    def launch(*args, **kwargs):
        assert (project / CONFIG_FILENAME).read_text() == new_content
        launch_started.set()
        return SimpleNamespace(pid=987654, wait=lambda: 0, terminate=lambda: None)

    monkeypatch.setattr(rag_project.os, "replace", replace)
    monkeypatch.setattr(rag_project.subprocess, "Popen", launch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        save = pool.submit(rag_project.save_project_config, rag_project.SaveProjectConfigRequest(
            project_name="draft", content=new_content,
        ))
        assert replacement_started.wait(5)
        run = pool.submit(rag_project.run_project, rag_project.RunProjectRequest(
            project_name="draft", qa_data="qa.parquet", corpus_data="corpus.parquet",
        ))
        try:
            assert not launch_started.wait(0.1)
        finally:
            allow_replacement.set()
        assert save.result(timeout=5)["ok"]
        assert run.result(timeout=5)["ok"]
    assert rag_project.get_project_config("draft")["editable"] is False


def test_run_started_during_validation_prevents_late_save(draft, monkeypatch):
    project, config = draft

    def validate(_request):
        save_project_summary(str(project), create_project_summary(ProjectStatus.RUNNING))
        return {"valid": True, "errors": []}

    monkeypatch.setattr(rag_project, "validate_config", validate)
    with pytest.raises(HTTPException) as exc:
        rag_project.save_project_config(rag_project.SaveProjectConfigRequest(
            project_name="draft", content="# late draft\n" + config["content"],
        ))
    assert exc.value.status_code == 409
    assert (project / CONFIG_FILENAME).read_text() == config["content"]


def test_recreate_does_not_bypass_history_guard(draft, tmp_path, monkeypatch):
    project, config = draft
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "replacement.yaml").write_text("replacement: true\n")
    monkeypatch.setattr(rag_project, "_get_configs_dir", lambda: configs)
    (project / "optimizer").mkdir()
    with pytest.raises(HTTPException) as exc:
        rag_project.create_project(rag_project.CreateProjectRequest(
            project_name="draft", config_filename="replacement.yaml",
        ))
    assert exc.value.status_code == 409
    assert (project / CONFIG_FILENAME).read_text() == config["content"]


def test_config_endpoint_rejects_path_escape(draft):
    with pytest.raises(HTTPException) as exc:
        rag_project.get_project_config("../draft")
    assert exc.value.status_code == 400


def _launch_inputs(tmp_path, monkeypatch):
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    for name in ("qa.parquet", "corpus.parquet"):
        (datasets / name).write_bytes(b"fixture")
    monkeypatch.setattr(rag_project, "_get_datasets_dir", lambda: datasets)
    return rag_project.RunProjectRequest(
        project_name="draft", qa_data="qa.parquet", corpus_data="corpus.parquet",
    )


def test_api_freezes_exact_config_before_spawning_worker(draft, tmp_path, monkeypatch):
    project, config = draft
    request = _launch_inputs(tmp_path, monkeypatch)

    def launch(*args, **kwargs):
        snapshot = json.loads((project / CONFIG_FREEZE_FILENAME).read_text())
        assert snapshot["raw_content"] == config["content"]
        return SimpleNamespace(pid=987654, wait=lambda: 0, terminate=lambda: None)

    monkeypatch.setattr(rag_project.subprocess, "Popen", launch)
    assert rag_project.run_project(request)["ok"]
    assert rag_project.get_project_config("draft")["editable"] is False


def test_failed_spawn_does_not_unlock_config(draft, tmp_path, monkeypatch):
    project, config = draft
    request = _launch_inputs(tmp_path, monkeypatch)

    def launch(*args, **kwargs):
        raise OSError("worker failed to start")

    monkeypatch.setattr(rag_project.subprocess, "Popen", launch)
    with pytest.raises(OSError, match="worker failed"):
        rag_project.run_project(request)
    assert (project / CONFIG_FREEZE_FILENAME).is_file()
    assert rag_project.get_project_status("draft")["status"] == "failed"
    assert rag_project.get_project_config("draft")["editable"] is False
    with pytest.raises(HTTPException) as exc:
        rag_project.save_project_config(rag_project.SaveProjectConfigRequest(
            project_name="draft", content=config["content"],
        ))
    assert exc.value.status_code == 409


def test_reset_and_created_summary_do_not_unlock_frozen_config(draft):
    project, config = draft
    freeze_project_config(project)
    frozen = (project / CONFIG_FREEZE_FILENAME).read_bytes()
    # Reset may remove all observations and replace status with CREATED.
    (project / "evaluations").mkdir()
    rag_project._wipe_project_state(project)
    assert (project / CONFIG_FREEZE_FILENAME).read_bytes() == frozen
    assert rag_project.get_project_config("draft")["editable"] is False
    with pytest.raises(HTTPException) as exc:
        rag_project.save_project_config(rag_project.SaveProjectConfigRequest(
            project_name="draft", content="# replacement\n" + config["content"],
        ))
    assert exc.value.status_code == 409


def test_external_change_is_rejected_before_reset_or_launch(draft, monkeypatch):
    project, config = draft
    freeze_project_config(project)
    (project / CONFIG_FILENAME).write_text("# changed\n" + config["content"])
    result = project / "all_evaluations.csv"
    result.write_text("existing results")
    launches = []
    monkeypatch.setattr(rag_project.subprocess, "Popen", lambda *a, **kw: launches.append(a))
    with pytest.raises(HTTPException) as exc:
        rag_project.run_project(rag_project.RunProjectRequest(project_name="draft", reset=True))
    assert exc.value.status_code == 409
    assert "changed" in exc.value.detail
    assert launches == []
    assert result.read_text() == "existing results"


def test_recreate_cannot_replace_deleted_frozen_config(draft, tmp_path, monkeypatch):
    project, config = draft
    freeze_project_config(project)
    (project / CONFIG_FILENAME).unlink()
    (project / "summary.json").unlink()
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "replacement.yaml").write_text(config["content"])
    monkeypatch.setattr(rag_project, "_get_configs_dir", lambda: configs)
    with pytest.raises(HTTPException) as exc:
        rag_project.create_project(rag_project.CreateProjectRequest(
            project_name="draft", config_filename="replacement.yaml",
        ))
    assert exc.value.status_code == 409
    assert not (project / CONFIG_FILENAME).exists()
