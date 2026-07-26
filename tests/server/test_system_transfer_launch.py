"""CPU-only launch contracts for full system transfer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml
from fastapi import HTTPException

import rag_stack.cli as cli
import rag_stack.controller as controller_module
import rag_stack.project_summary as project_summary_module
from rag_stack.project_summary import (
    ProjectStatus,
    create_project_summary,
    load_project_summary,
    save_project_summary,
)
from rag_stack.quality_store import QualityStore
from rag_stack.server.routers import rag_project


QWEN_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]


def test_selective_transfer_cli_flags_are_repeatable_nonnegative() -> None:
    parser = argparse.ArgumentParser()
    cli._add_optimize_args(parser)
    args = parser.parse_args([
        "--mode", "hardware_replay",
        "--reuse-target-project", "/prior-target",
        "--retransfer-source-index", "4",
        "--retransfer-source-index", "0",
    ])
    assert args.reuse_target_project == "/prior-target"
    assert args.retransfer_source_index == [4, 0]

    with pytest.raises(SystemExit):
        parser.parse_args([
            "--retransfer-source-index", "-1",
        ])


def _write_vllm_config(path: Path, uri: str) -> None:
    path.write_text(yaml.safe_dump({
        "algo_search_space": {
            "pipeline": {
                "node_lines": [{
                    "nodes": [{
                        "stage": "query_expansion",
                        "modules": [{
                            "component": "query_decompose",
                            "generator_backend": "vllm_api",
                            "uri": uri,
                            "model": {"ordered": QWEN_MODELS},
                        }],
                    }, {
                        "stage": "generator",
                        "modules": [{
                            "component": "vllm_api",
                            "uri": uri,
                            "model": {"ordered": QWEN_MODELS},
                        }],
                    }],
                }],
            },
        },
    }, sort_keys=False))


def _models_response(models: list[str]) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [{"id": model, "max_model_len": 32768} for model in models],
    }
    return response


def test_refinement_preflight_requires_env_for_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    _write_vllm_config(config, "${RAG_LLM_BASE_URL}")
    monkeypatch.delenv("RAG_LLM_BASE_URL", raising=False)
    request = Mock()
    monkeypatch.setattr("requests.get", request)

    with pytest.raises(ValueError, match="RAG_LLM_BASE_URL.*empty"):
        cli._preflight_hardware_replay_llm(str(config))

    request.assert_not_called()


def test_refinement_preflight_supports_fixed_uri_and_checks_all_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    _write_vllm_config(config, "http://fixed-qwen:8000")
    monkeypatch.delenv("RAG_LLM_BASE_URL", raising=False)
    request = Mock(return_value=_models_response(QWEN_MODELS))
    monkeypatch.setattr("requests.get", request)

    cli._preflight_hardware_replay_llm(str(config))

    request.assert_called_once_with(
        "http://fixed-qwen:8000/v1/models",
        timeout=cli._RAG_LLM_PREFLIGHT_TIMEOUT_S,
    )


def test_refinement_preflight_resolves_env_uri(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    _write_vllm_config(config, "${RAG_LLM_BASE_URL}")
    monkeypatch.setenv("RAG_LLM_BASE_URL", "http://env-qwen:8000/")
    request = Mock(return_value=_models_response(QWEN_MODELS))
    monkeypatch.setattr("requests.get", request)

    cli._preflight_hardware_replay_llm(str(config))

    request.assert_called_once_with(
        "http://env-qwen:8000/v1/models",
        timeout=cli._RAG_LLM_PREFLIGHT_TIMEOUT_S,
    )


def test_refinement_preflight_rejects_unreachable_or_incomplete_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    _write_vllm_config(config, "http://fixed-qwen:8000")

    monkeypatch.setattr(
        "requests.get", Mock(side_effect=OSError("connection refused")),
    )
    with pytest.raises(ValueError, match="connection refused"):
        cli._preflight_hardware_replay_llm(str(config))

    monkeypatch.setattr(
        "requests.get", Mock(return_value=_models_response(QWEN_MODELS[:-1])),
    )
    with pytest.raises(ValueError, match="missing required model.*14B"):
        cli._preflight_hardware_replay_llm(str(config))


def test_refinement_preflight_local_vllm_chat_never_calls_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({
        "algo_search_space": {
            "pipeline": {
                "node_lines": [{
                    "nodes": [{
                        "stage": "query_expansion",
                        "modules": [{
                            "component": "query_decompose",
                            "generator_backend": "vllm",
                            "model": {"ordered": QWEN_MODELS},
                            "use_chat_template": True,
                        }],
                    }, {
                        "stage": "generator",
                        "modules": [{
                            "component": "vllm",
                            "model": {"ordered": QWEN_MODELS},
                            "use_chat_template": True,
                        }],
                    }],
                }],
            },
        },
    }, sort_keys=False))
    monkeypatch.delenv("RAG_LLM_BASE_URL", raising=False)
    request = Mock(side_effect=AssertionError("local chat must not use HTTP"))
    monkeypatch.setattr("requests.get", request)

    cli._preflight_hardware_replay_llm(str(config))

    request.assert_not_called()


def test_preflight_runs_before_quality_store_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "target"
    source = tmp_path / "source"
    project.mkdir()
    source.mkdir()
    _write_vllm_config(project / "rag_stack_config.yaml", "${RAG_LLM_BASE_URL}")
    save_project_summary(
        str(project), create_project_summary(ProjectStatus.RUNNING),
    )

    preflight = Mock(side_effect=ValueError("endpoint unavailable"))
    importer = Mock()
    monkeypatch.setattr(cli, "_preflight_hardware_replay_llm", preflight)
    monkeypatch.setattr(
        "rag_stack.quality_store.QualityStore.from_project", importer,
    )
    monkeypatch.setattr(cli, "_setup_logging", lambda: None)
    monkeypatch.setattr(cli, "_install_subproc_reaper", lambda: None)

    args = SimpleNamespace(
        project_dir=str(project),
        config=None,
        performance_source=None,
        cm_calibration_mode=None,
        mode="hardware_replay",
        quality_store=None,
        from_project=str(source),
        qa_data=None,
        corpus_data=None,
        n_iterations=None,
        hardware_config=None,
        n_refine_iterations=10,
    )
    with pytest.raises(SystemExit) as error:
        cli.cmd_optimize(args)

    assert error.value.code == 1
    preflight.assert_called_once_with(str(project / "rag_stack_config.yaml"))
    importer.assert_not_called()
    assert load_project_summary(str(project)).status is ProjectStatus.FAILED


class _FakeProcess:
    def __init__(self, *, pid: int = 4242, return_code: int = 0):
        self.pid = pid
        self.return_code = return_code
        self.terminated = False

    def wait(self) -> int:
        return self.return_code

    def terminate(self) -> None:
        self.terminated = True


class _CapturedThread:
    created: list["_CapturedThread"] = []

    def __init__(self, *, target, daemon: bool):
        self.target = target
        self.daemon = daemon
        self.started = False
        self.created.append(self)

    def start(self) -> None:
        self.started = True


def _prepare_api_project(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    transfer: bool,
) -> tuple[Path, Path]:
    root = tmp_path / "projects"
    project = root / "target"
    data = project / "data"
    data.mkdir(parents=True)
    (data / "qa.parquet").write_bytes(b"qa")
    (data / "corpus.parquet").write_bytes(b"corpus")
    (project / "rag_stack_config.yaml").write_text("dataset: {}\n")
    save_project_summary(
        str(project), create_project_summary(ProjectStatus.CREATED),
    )
    if transfer:
        (root / "source").mkdir()
        (project / "system_transfer.json").write_text(json.dumps(
            _transfer_launch(),
        ))
    monkeypatch.setattr(rag_project, "_get_project_dir", lambda: root)
    return root, project


def _transfer_launch(**overrides) -> dict:
    payload = {
        "kind": "system_transfer_launch",
        "source_project": "source",
        "system_yaml": "fixture.yaml",
        "n_refine_iterations": 10,
        "n_refine_iterations_source": "server_default",
        "generator_backend": "inherit",
        "vllm_chat_projection": {},
    }
    payload.update(overrides)
    return payload


def _transfer_state(budget: int) -> dict:
    return {
        "kind": "system_transfer_state",
        "status": "running",
        "cost_model_version": 1,
        "target_config_sha256": "a" * 64,
        "target_system_sha256": "b" * 64,
        "quality_store_path": "/fixture/quality-store",
        "refinement_iterations_requested": budget,
        "source_records": [{
            "source_index": 0,
            "algo_hash": "source-algo",
            "record_marker_sha256": "c" * 64,
        }],
    }


def _capture_api_launch(
    monkeypatch: pytest.MonkeyPatch,
    process: _FakeProcess,
) -> tuple[dict, list[_CapturedThread]]:
    captured: dict = {}

    def _popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return process

    _CapturedThread.created = []
    monkeypatch.setattr(rag_project.subprocess, "Popen", _popen)
    monkeypatch.setattr(rag_project.threading, "Thread", _CapturedThread)
    return captured, _CapturedThread.created


def test_api_claims_pid_immediately_and_removes_transfer_worker_caps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, project = _prepare_api_project(tmp_path, monkeypatch, transfer=True)
    process = _FakeProcess()
    captured, threads = _capture_api_launch(monkeypatch, process)
    monkeypatch.setenv("CM_SWEEP_WORKERS", "8")
    monkeypatch.setenv("CM_DES_WORKERS", "8")

    response = rag_project.run_project(rag_project.RunProjectRequest(
        project_name="target",
        mode="hardware_replay",
        n_refine_iterations=10,
    ))

    assert response["pid"] == process.pid
    summary = load_project_summary(str(project))
    assert summary.status is ProjectStatus.RUNNING
    assert summary.pid == process.pid
    assert "CM_SWEEP_WORKERS" not in captured["env"]
    assert "CM_DES_WORKERS" not in captured["env"]
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert len(threads) == 1 and threads[0].started
    threads[0].target()


def test_general_api_run_preserves_ambient_worker_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_api_project(tmp_path, monkeypatch, transfer=False)
    process = _FakeProcess()
    captured, threads = _capture_api_launch(monkeypatch, process)
    monkeypatch.setenv("CM_SWEEP_WORKERS", "17")
    monkeypatch.setenv("CM_DES_WORKERS", "19")

    rag_project.run_project(rag_project.RunProjectRequest(
        project_name="target", mode="full",
    ))

    assert captured["env"]["CM_SWEEP_WORKERS"] == "17"
    assert captured["env"]["CM_DES_WORKERS"] == "19"
    threads[0].target()


def test_api_resume_infers_persisted_system_transfer_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, project = _prepare_api_project(tmp_path, monkeypatch, transfer=True)
    source = root / "source"
    (project / "system_transfer.json").write_text(json.dumps(
        _transfer_launch(),
    ))
    process = _FakeProcess()
    captured, threads = _capture_api_launch(monkeypatch, process)
    monkeypatch.setenv("CM_SWEEP_WORKERS", "8")
    monkeypatch.setenv("CM_DES_WORKERS", "8")

    rag_project.run_project(rag_project.RunProjectRequest(
        project_name="target",
        # UI resume defaults must not turn a transfer into normal full mode.
        mode="full",
        n_refine_iterations=0,
    ))

    command = captured["command"]
    assert command[command.index("--mode") + 1] == "hardware_replay"
    assert command[command.index("--from-project") + 1] == str(source.resolve())
    assert command[command.index("--n-refine-iterations") + 1] == "10"
    assert "--quality-store" not in command
    assert "CM_SWEEP_WORKERS" not in captured["env"]
    assert "CM_DES_WORKERS" not in captured["env"]
    assert captured["env"]["HF_HUB_OFFLINE"] == "1"
    assert captured["env"]["TRANSFORMERS_OFFLINE"] == "1"
    threads[0].target()


@pytest.mark.parametrize(
    ("state_budget", "requested_budget"),
    [(10, 20), (20, 0)],
)
def test_api_resume_uses_monotonic_or_already_extended_transfer_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state_budget: int,
    requested_budget: int,
) -> None:
    root, project = _prepare_api_project(tmp_path, monkeypatch, transfer=True)
    source = root / "source"
    (project / "system_transfer.json").write_text(json.dumps(
        _transfer_launch(),
    ))
    (project / "system_transfer_state.json").write_text(json.dumps(
        _transfer_state(state_budget),
    ))
    process = _FakeProcess()
    captured, threads = _capture_api_launch(monkeypatch, process)

    rag_project.run_project(rag_project.RunProjectRequest(
        project_name="target",
        n_refine_iterations=requested_budget,
    ))

    command = captured["command"]
    assert command[command.index("--n-refine-iterations") + 1] == "20"
    threads[0].target()


def test_api_resume_rejects_transfer_budget_decrease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, project = _prepare_api_project(tmp_path, monkeypatch, transfer=True)
    (project / "system_transfer.json").write_text(json.dumps(
        _transfer_launch(),
    ))
    (project / "system_transfer_state.json").write_text(json.dumps(
        _transfer_state(20),
    ))

    with pytest.raises(HTTPException, match="budget cannot be decreased"):
        rag_project.run_project(rag_project.RunProjectRequest(
            project_name="target",
            n_refine_iterations=15,
        ))


def test_api_reaper_marks_only_its_failed_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, project = _prepare_api_project(tmp_path, monkeypatch, transfer=True)
    process = _FakeProcess(return_code=7)
    _, threads = _capture_api_launch(monkeypatch, process)

    rag_project.run_project(rag_project.RunProjectRequest(
        project_name="target",
        mode="hardware_replay",
        n_refine_iterations=10,
    ))
    threads[0].target()

    summary = load_project_summary(str(project))
    assert summary.status is ProjectStatus.FAILED
    assert summary.pid is None


def test_api_reaper_does_not_clobber_a_new_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, project = _prepare_api_project(tmp_path, monkeypatch, transfer=True)
    process = _FakeProcess(return_code=7)
    _, threads = _capture_api_launch(monkeypatch, process)

    rag_project.run_project(rag_project.RunProjectRequest(
        project_name="target",
        mode="hardware_replay",
        n_refine_iterations=10,
    ))
    replacement = load_project_summary(str(project))
    replacement.pid = process.pid + 1
    save_project_summary(str(project), replacement)
    threads[0].target()

    summary = load_project_summary(str(project))
    assert summary.status is ProjectStatus.RUNNING
    assert summary.pid == process.pid + 1


def test_reset_removes_stale_system_transfer_result(
    tmp_path: Path,
) -> None:
    project = tmp_path / "target"
    project.mkdir()
    (project / "rag_stack_config.yaml").write_text("dataset: {}\n")
    save_project_summary(
        str(project), create_project_summary(ProjectStatus.COMPLETED),
    )
    result = project / "system_transfer_result.json"
    result.write_text('{"status": "completed"}\n')
    (project / "system_transfer_state.json").write_text('{"status": "running"}\n')
    (project / "quality_store").mkdir()
    (project / "quality_store/manifest.json").write_text("{}\n")

    report = rag_project._wipe_project_state(project)

    assert "system_transfer_result.json" in report["files"]
    assert "system_transfer_state.json" in report["files"]
    assert "quality_store" in report["dirs"]
    assert not result.exists()
    assert not (project / "system_transfer_state.json").exists()
    assert not (project / "quality_store").exists()
    assert load_project_summary(str(project)).status is ProjectStatus.CREATED


def test_cli_reuses_matching_from_project_quality_store_without_reimport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    store_path = tmp_path / "target/quality_store"
    eval_dir = source / "evaluations/eval_0007"
    eval_dir.mkdir(parents=True)
    store_path.mkdir(parents=True)
    source_config = source / "rag_stack_config.yaml"
    store_config = store_path / "rag_stack_config.yaml"
    source_config.write_text("global: {seed: 45}\n")
    store_config.write_text(source_config.read_text())
    artifacts = []
    for name in (
        "quality_metrics.json", "pipeline_config.json", "execution_dag.json",
    ):
        path = eval_dir / name
        path.write_text("{}\n")
        artifacts.append(path)
    source_mtime_ns = max(path.stat().st_mtime_ns for path in artifacts)
    (store_path / "manifest.json").write_text("{}\n")
    record_hash = "source-hash"
    fake_store = SimpleNamespace(
        store_path=str(store_path),
        _manifest={
            "source_project_dirs": [str(source.resolve())],
            "last_imported_hashes": [record_hash],
        },
        get_record=lambda value: ({
            "record": {
                "source": {"eval_id": 7},
                "source_mtime_ns": source_mtime_ns,
            },
        } if value == record_hash else None),
    )
    load = Mock(return_value=fake_store)
    importer = Mock()
    monkeypatch.setattr(QualityStore, "load", load)
    monkeypatch.setattr(QualityStore, "from_project", importer)

    resolved, reused = cli._quality_store_for_project_resume(
        source_project=str(source), store_path=str(store_path),
    )

    assert resolved == str(store_path.resolve())
    assert reused is True
    load.assert_called_once_with(str(store_path.resolve()))
    importer.assert_not_called()


def test_cli_quality_store_resume_fails_closed_on_source_config_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    store_path = tmp_path / "target/quality_store"
    source.mkdir(parents=True)
    store_path.mkdir(parents=True)
    (source / "rag_stack_config.yaml").write_text("global: {seed: 44}\n")
    (store_path / "rag_stack_config.yaml").write_text("global: {seed: 45}\n")
    (store_path / "manifest.json").write_text("{}\n")
    fake_store = SimpleNamespace(
        store_path=str(store_path),
        _manifest={
            "source_project_dirs": [str(source.resolve())],
            "last_imported_hashes": ["source-hash"],
        },
    )
    monkeypatch.setattr(QualityStore, "load", Mock(return_value=fake_store))
    importer = Mock()
    monkeypatch.setattr(QualityStore, "from_project", importer)

    with pytest.raises(RuntimeError, match="source config no longer matches"):
        cli._quality_store_for_project_resume(
            source_project=str(source), store_path=str(store_path),
        )
    importer.assert_not_called()


def test_project_summary_commit_uses_atomic_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_replace = os.replace
    replace_spy = Mock(side_effect=real_replace)
    monkeypatch.setattr(project_summary_module.os, "replace", replace_spy)

    save_project_summary(
        str(tmp_path), create_project_summary(ProjectStatus.RUNNING),
    )

    replace_spy.assert_called_once()
    temp_path, final_path = replace_spy.call_args.args
    assert temp_path != str(tmp_path / "summary.json")
    assert final_path == str(tmp_path / "summary.json")
    assert not list(tmp_path.glob(".summary.*.tmp"))


def test_controller_accepts_parent_claim_owned_by_its_own_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "rag_stack_config.yaml").write_text("dataset: {}\n")
    summary = create_project_summary(ProjectStatus.RUNNING)
    summary.pid = os.getpid()
    save_project_summary(str(tmp_path), summary)

    class _ReachedContext(RuntimeError):
        pass

    context = Mock(side_effect=_ReachedContext("context reached"))
    monkeypatch.setattr(controller_module, "PerformanceContext", context)
    monkeypatch.setattr(controller_module, "pid_alive", lambda _pid: True)

    with pytest.raises(_ReachedContext, match="context reached"):
        controller_module.Controller(
            config_path=str(tmp_path / "rag_stack_config.yaml"),
            project_dir=str(tmp_path),
        )
    context.assert_called_once()
