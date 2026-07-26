"""Config ownership at CLI and direct Python boundaries, without model work."""

from pathlib import Path
from copy import deepcopy
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from rag_stack import cli
from rag_stack.controller import Controller
from rag_stack.project_config import (
    CONFIG_FREEZE_FILENAME,
    ProjectConfigError,
    freeze_project_config,
    verify_project_config,
)
from rag_stack.project_summary import (
    CONFIG_FILENAME,
    ProjectStatus,
    create_project_summary,
    load_project_summary,
    save_project_summary,
)


RAW = "# original input\nglobal:\n  n_iterations: 8\nsystem:\n  performance_source: cost_model\n"


def _args(**kwargs):
    return SimpleNamespace(
        **dict(config=None, performance_source=None, cm_calibration_mode=None,
               n_iterations=None) | kwargs,
    )


def _project(tmp_path, raw=RAW):
    project = tmp_path / "project"
    project.mkdir()
    (project / CONFIG_FILENAME).write_text(raw)
    return project


def _controller(project):
    controller = Controller.__new__(Controller)
    controller.project_dir = str(project)
    controller._project_config_raw = freeze_project_config(project)
    controller.config = yaml.safe_load(controller._project_config_raw)
    controller._project_runtime_config = deepcopy(controller.config)
    return controller


def test_cli_initialization_freezes_original_input_before_controller(tmp_path):
    source = tmp_path / "source.yaml"
    source.write_bytes(RAW.replace("\n", "\r\n").encode())
    project = tmp_path / "new-project"

    config_path = cli._prepare_optimize_project_config(
        str(project), _args(config=str(source)),
    )

    assert Path(config_path).read_bytes() == source.read_bytes()
    assert (project / CONFIG_FREEZE_FILENAME).is_file()
    assert load_project_summary(str(project)).status is ProjectStatus.CREATED


def test_cli_prepares_new_config_before_creating_run_logs(tmp_path, monkeypatch):
    import rag_stack.controller as module

    source = tmp_path / "source.yaml"
    source.write_text(RAW)
    project = tmp_path / "project"
    launched = Mock(return_value=SimpleNamespace(get_pareto=lambda **_: {}))
    monkeypatch.setattr(module, "Controller", launched)
    monkeypatch.setattr(cli, "_setup_logging", lambda: None)
    monkeypatch.setattr(cli, "_install_subproc_reaper", lambda: None)
    monkeypatch.delenv(cli.PROJECT_LOG_OWNED_BY_PARENT_ENV, raising=False)

    def attach(output_dir, command):
        assert output_dir == str(project)
        assert command == "optimize"
        assert verify_project_config(project) == RAW
        assert (project / CONFIG_FREEZE_FILENAME).is_file()
        logs = project / "logs"
        logs.mkdir()
        (logs / "run.log").write_text("started")

    monkeypatch.setattr(cli, "_attach_project_log", attach)
    monkeypatch.setattr(sys, "argv", [
        "rag-stack", "optimize", "--project-dir", str(project),
        "--config", str(source),
    ])
    cli.main()
    launched.assert_called_once()


def test_cli_resume_never_overwrites_project_or_checkpoint(tmp_path):
    project = _project(tmp_path)
    freeze_project_config(project)
    summary = create_project_summary(ProjectStatus.FAILED)
    summary.n_evaluations = 4
    save_project_summary(str(project), summary)
    source = tmp_path / "other.yaml"
    source.write_text(RAW.replace("8", "12"))

    with pytest.raises(ProjectConfigError, match="read-only"):
        cli._prepare_optimize_project_config(
            str(project), _args(config=str(source)),
        )

    assert verify_project_config(project) == RAW
    retained = load_project_summary(str(project))
    assert retained.status is ProjectStatus.FAILED
    assert retained.n_evaluations == 4
    source.write_text(RAW)
    cli._prepare_optimize_project_config(str(project), _args(config=str(source)))
    assert load_project_summary(str(project)).n_evaluations == 4


@pytest.mark.parametrize("override", [
    {"performance_source": "measured"},
    {"cm_calibration_mode": "uncalibrated_spec"},
    {"n_iterations": 12},
])
def test_cli_policy_overrides_are_draft_only(tmp_path, override):
    project = _project(tmp_path)
    cli._prepare_optimize_project_config(str(project), _args(**override))
    frozen = verify_project_config(project)
    assert frozen != RAW
    original_mtime = (project / CONFIG_FILENAME).stat().st_mtime_ns
    cli._prepare_optimize_project_config(str(project), _args(**override))
    assert (project / CONFIG_FILENAME).stat().st_mtime_ns == original_mtime
    reset = dict(performance_source="cost_model", cm_calibration_mode="strict",
                 n_iterations=8)
    with pytest.raises(ProjectConfigError, match="read-only"):
        cli._prepare_optimize_project_config(str(project), _args(**reset))
    assert verify_project_config(project) == frozen


def test_cli_identical_effective_options_keep_raw_formatting(tmp_path):
    project = _project(tmp_path)
    freeze_project_config(project)
    cli._prepare_optimize_project_config(str(project), _args(
        performance_source="cost_model", cm_calibration_mode="strict",
        n_iterations=8,
    ))
    assert verify_project_config(project) == RAW


def test_controller_binds_raw_input_before_initializing_resources(tmp_path, monkeypatch):
    import rag_stack.controller as module

    source = tmp_path / "external.yaml"
    source.write_text(RAW + "dataset:\n  qa: ${PROJECT_DIR}/data/qa.parquet\n")
    project = tmp_path / "new-project"

    class ContextReached(Exception):
        pass

    def context(**kwargs):
        assert kwargs["config_path"] == str(project / CONFIG_FILENAME)
        assert verify_project_config(project) == source.read_text()
        assert (project / CONFIG_FREEZE_FILENAME).is_file()
        raise ContextReached

    monkeypatch.setattr(module, "PerformanceContext", context)
    with pytest.raises(ContextReached):
        Controller(config_path=str(source), project_dir=str(project))
    assert "${PROJECT_DIR}" in (project / CONFIG_FILENAME).read_text()


def test_controller_rejects_alternate_config_before_resources(tmp_path, monkeypatch):
    import rag_stack.controller as module

    project = _project(tmp_path)
    freeze_project_config(project)
    source = tmp_path / "different.yaml"
    source.write_text(RAW.replace("cost_model", "measured"))
    context = Mock()
    monkeypatch.setattr(module, "PerformanceContext", context)

    with pytest.raises(ProjectConfigError, match="read-only"):
        Controller(config_path=str(source), project_dir=str(project))
    context.assert_not_called()
    assert verify_project_config(project) == RAW


@pytest.mark.parametrize("mutation", ["replace", "delete"])
def test_controller_detects_file_changes_before_candidate_execution(tmp_path, mutation):
    project = _project(tmp_path)
    controller = _controller(project)
    config = project / CONFIG_FILENAME
    if mutation == "replace":
        config.write_text(RAW.replace("8", "12"))
    else:
        config.unlink()

    with pytest.raises(ProjectConfigError):
        controller._evaluate_objectives([{}], ["performance", "quality"])
    if mutation == "delete":
        assert not config.exists()


def test_python_iteration_override_fails_before_measured_provider(tmp_path):
    controller = _controller(_project(tmp_path))
    controller._measured_provider = Mock()
    with pytest.raises(ProjectConfigError, match="n_iterations"):
        controller.get_pareto(n_iterations=12)
    controller._measured_provider.assert_not_called()


def test_python_config_mutation_cannot_change_frozen_budget(tmp_path):
    controller = _controller(_project(tmp_path))
    controller.config["global"]["n_iterations"] = 12
    with pytest.raises(ProjectConfigError, match="after initialization"):
        controller.get_pareto(n_iterations=12)


def test_identical_python_budget_runs_with_initialized_runtime_enrichment(tmp_path):
    controller = _controller(_project(tmp_path))
    controller.config["_token_stats"] = {"avg_output_tokens": 32}
    controller._project_runtime_config = deepcopy(controller.config)
    controller._measured_provider = None
    controller._get_pareto_impl = Mock(return_value={"gt_count": 1})
    assert controller.get_pareto(n_iterations=8) == {"gt_count": 1}
    controller._get_pareto_impl.assert_called_once_with(8)
    assert verify_project_config(controller.project_dir) == RAW


def test_python_selection_override_fails_before_quality_store(tmp_path, monkeypatch):
    from rag_stack.types import SystemConfig
    import rag_stack.controller as module

    controller = _controller(_project(tmp_path))
    loader = Mock()
    monkeypatch.setattr(module.QualityStore, "load", loader)
    with pytest.raises(ProjectConfigError, match="performance_selection"):
        controller.get_pareto_for_hardware(
            SystemConfig.from_dict(controller.config["system"]),
            quality_store_path="unused", performance_selection={"selection": "max_throughput"},
        )
    loader.assert_not_called()
