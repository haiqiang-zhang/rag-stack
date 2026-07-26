"""Configuration ownership holds across API/CLI helpers and process boundaries."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from threading import Event

import pytest

from rag_stack import project_config
from rag_stack.project_config import (
    CONFIG_FILENAME, CONFIG_FREEZE_FILENAME, ProjectConfigError,
    freeze_project_config, project_config_lock, project_config_read_only_reason,
    verify_project_config, write_project_config,
)


@pytest.fixture
def draft(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    write_project_config(project, "dataset: {}\n")
    return project


def test_first_launch_retains_raw_input_without_environment_expansion(draft, monkeypatch):
    monkeypatch.setenv("PROJECT_DIR", "/expanded/path")
    content = '# Original input\r\ndataset: {qa: "${PROJECT_DIR}/qa.parquet"}\r\n'
    write_project_config(draft, content)
    assert freeze_project_config(draft) == content
    snapshot = json.loads((draft / CONFIG_FREEZE_FILENAME).read_text())
    assert snapshot == {
        "schema_version": "rag-stack.project-config-freeze.v1",
        "frozen_at": snapshot["frozen_at"],
        "raw_content": content,
    }
    assert snapshot["frozen_at"]
    assert (draft / CONFIG_FILENAME).read_bytes() == content.encode()
    assert verify_project_config(draft) == content


def test_frozen_config_accepts_only_verified_noop(draft):
    freeze_project_config(draft)
    original = (draft / CONFIG_FREEZE_FILENAME).read_bytes()
    assert write_project_config(draft, "dataset: {}\n") == "dataset: {}\n"
    with pytest.raises(ProjectConfigError, match="read-only"):
        write_project_config(draft, "dataset: {dataset_name: changed}\n")
    assert freeze_project_config(draft) == "dataset: {}\n"
    assert (draft / CONFIG_FREEZE_FILENAME).read_bytes() == original


@pytest.mark.parametrize("operation", [
    verify_project_config, freeze_project_config,
    lambda project: write_project_config(project, "dataset: {dataset_name: edited}\n"),
    lambda project: write_project_config(project, "dataset: {}\n"),
])
def test_out_of_band_edit_is_rejected_even_when_write_matches_edited_file(draft, operation):
    freeze_project_config(draft)
    (draft / CONFIG_FILENAME).write_text("dataset: {dataset_name: edited}\n")
    with pytest.raises(ProjectConfigError, match="changed since its first run"):
        operation(draft)


def test_reset_and_empty_results_do_not_unlock_configuration(draft):
    freeze_project_config(draft)
    (draft / "summary.json").write_text(json.dumps({"status": "created", "n_evaluations": 0}))
    assert project_config_read_only_reason(draft)
    with pytest.raises(ProjectConfigError, match="read-only"):
        write_project_config(draft, "dataset: {dataset_name: changed}\n")


@pytest.mark.parametrize("status", ["running", "failed", "completed", "quality_complete"])
def test_legacy_started_project_adopts_retained_config_but_cannot_edit(draft, status):
    (draft / "summary.json").write_text(json.dumps({"status": status, "n_evaluations": 0}))
    assert project_config_read_only_reason(draft)
    with pytest.raises(ProjectConfigError, match="read-only"):
        write_project_config(draft, "dataset: {dataset_name: changed}\n")
    assert freeze_project_config(draft) == "dataset: {}\n"
    assert (draft / CONFIG_FREEZE_FILENAME).is_file()


@pytest.mark.parametrize("history", [
    "evaluations/eval_0000/result.json", "optimizer/checkpoint.json",
    "all_evaluations.csv", "logs/startup.log", "data/qa.parquet",
    "system_transfer.json",
])
def test_missing_summary_cannot_hide_legacy_run_history(draft, history):
    path = draft / history
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("previous run")
    assert project_config_read_only_reason(draft)
    with pytest.raises(ProjectConfigError, match="read-only"):
        write_project_config(draft, "dataset: {dataset_name: changed}\n")


@pytest.mark.parametrize("freeze", [True, False])
def test_missing_started_configuration_cannot_be_recreated(draft, freeze):
    if freeze:
        freeze_project_config(draft)
    else:
        (draft / "summary.json").write_text('{"status": "failed"}')
    (draft / CONFIG_FILENAME).unlink()
    with pytest.raises(ProjectConfigError):
        write_project_config(draft, "dataset: {}\n")
    with pytest.raises(ProjectConfigError, match="unavailable"):
        freeze_project_config(draft)
    assert not (draft / CONFIG_FILENAME).exists()


@pytest.mark.parametrize("raw", ["[]", "dataset: [broken", ""])
def test_invalid_draft_save_and_freeze_do_not_publish_snapshot(draft, raw):
    with pytest.raises(ProjectConfigError):
        write_project_config(draft, raw)
    assert (draft / CONFIG_FILENAME).read_text() == "dataset: {}\n"
    (draft / CONFIG_FILENAME).write_text(raw)
    with pytest.raises(ProjectConfigError):
        freeze_project_config(draft)
    assert not (draft / CONFIG_FREEZE_FILENAME).exists()
    write_project_config(draft, "dataset: {}\n")


@pytest.mark.parametrize("raw", ["not json", "[]", "{}", '{"schema_version": "future"}'])
def test_corrupt_freeze_fails_closed(draft, raw):
    (draft / CONFIG_FREEZE_FILENAME).write_text(raw)
    assert project_config_read_only_reason(draft)
    for operation in (verify_project_config, freeze_project_config):
        with pytest.raises(ProjectConfigError, match="Frozen project configuration"):
            operation(draft)
    with pytest.raises(ProjectConfigError):
        write_project_config(draft, "dataset: {}\n")
    assert (draft / CONFIG_FREEZE_FILENAME).read_text() == raw


def test_preview_snapshots_do_not_freeze_a_draft(draft):
    preview = draft / "cm_previews" / "sample" / "source_config.yaml"
    preview.parent.mkdir(parents=True)
    preview.write_text("dataset: {}\n")
    assert project_config_read_only_reason(draft) is None
    write_project_config(draft, "dataset: {dataset_name: changed}\n")
    assert preview.read_text() == "dataset: {}\n"


def test_failed_atomic_save_preserves_existing_config(draft, monkeypatch):
    def fail_replace(*_args):
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr(project_config.os, "replace", fail_replace)
    with pytest.raises(OSError, match="filesystem failure"):
        write_project_config(draft, "dataset: {dataset_name: changed}\n")
    assert (draft / CONFIG_FILENAME).read_text() == "dataset: {}\n"
    assert not list(draft.glob(".config.*.tmp"))


def test_nested_lock_serializes_freeze_before_late_thread_save(draft):
    freezer_ready, release_freezer, writer_ready = Event(), Event(), Event()

    def freeze():
        with project_config_lock(draft):
            freezer_ready.set()
            assert release_freezer.wait(5)
            return freeze_project_config(draft)

    def write():
        writer_ready.set()
        return write_project_config(draft, "dataset: {dataset_name: late}\n")

    with ThreadPoolExecutor(max_workers=2) as pool:
        freezing = pool.submit(freeze)
        assert freezer_ready.wait(5)
        writing = pool.submit(write)
        assert writer_ready.wait(5)
        try:
            assert not writing.done()
        finally:
            release_freezer.set()
        assert freezing.result(timeout=5) == "dataset: {}\n"
        with pytest.raises(ProjectConfigError, match="read-only"):
            writing.result(timeout=5)


def test_process_lock_serializes_cli_style_save_against_api_style_launch(draft):
    script = """
import sys
from rag_stack.project_config import ProjectConfigError, write_project_config
print('ready', flush=True)
try:
    write_project_config(sys.argv[1], 'dataset: {dataset_name: late}\\n')
except ProjectConfigError:
    print('frozen', flush=True)
else:
    raise RuntimeError('A concurrent launch allowed a late configuration change')
"""
    with project_config_lock(draft):
        child = subprocess.Popen(
            [sys.executable, "-c", script, str(draft)], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        try:
            assert child.stdout.readline().strip() == "ready"
            assert child.poll() is None
            freeze_project_config(draft)
        except BaseException:
            child.kill()
            child.communicate(timeout=5)
            raise
    stdout, stderr = child.communicate(timeout=10)
    assert child.returncode == 0, stderr
    assert stdout.strip() == "frozen"
    assert verify_project_config(draft) == "dataset: {}\n"
