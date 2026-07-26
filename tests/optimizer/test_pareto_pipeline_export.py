"""CPU-only contracts for exact Pareto pipeline export."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from rag_stack.controller import Controller


def test_pareto_yaml_uses_exact_persisted_eval_pipeline(tmp_path: Path) -> None:
    controller = Controller.__new__(Controller)
    controller.project_dir = str(tmp_path)
    controller._invalid_configs = []
    controller._resolve_config = Mock(
        side_effect=AssertionError(
            "Pareto export must not reinterpret an enriched optimizer config"
        )
    )
    controller._selected_system_design = Mock(return_value=None)
    controller._ssops = SimpleNamespace(
        flatten_2deep=lambda config: {
            "pipeline__prompt": config["pipeline"]["prompt"],
        }
    )

    eval_dir = tmp_path / "evaluations" / "eval_0007"
    eval_dir.mkdir(parents=True)
    exact_pipeline = {
        "pipeline": {
            "component": "long_context_reorder",
            "prompt": "Question: {query}\n Context: {retrieved_contents}\n Answer:",
        },
        "vectordb": [{"name": "fixture", "N": 572}],
        "_token_stats": {"avg_chunk_tokens": 326},
    }
    (eval_dir / "pipeline_config.json").write_text(
        json.dumps(exact_pipeline),
        encoding="utf-8",
    )
    exact_system = {"hardware": {"available_gpu_count": 4}}
    # Historical measured evals persisted the exact prelaunch deployment
    # under this name.  Pareto publication must remain replayable without a
    # GPU remeasurement.
    (eval_dir / "system_config_resolved.json").write_text(
        json.dumps(exact_system),
        encoding="utf-8",
    )

    controller._save_pareto_frontier(
        {
            "pareto_configs": [
                {
                    "pipeline": {
                        "component": "long_context_reorder",
                        "prompt": "stale first-module prompt",
                    }
                }
            ],
            "pareto_objectives": [{"performance": 12.0, "quality": 0.5}],
            "pareto_eval_ids": [7],
        }
    )

    exported = yaml.safe_load(
        (tmp_path / "pareto_configs" / "pareto_eval_0007.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert exported["pipeline"] == exact_pipeline["pipeline"]
    assert exported["vectordb"] == exact_pipeline["vectordb"]
    assert exported["_token_stats"] == exact_pipeline["_token_stats"]
    assert exported["_eval_id"] == 7
    assert exported["_pareto_objectives"] == {
        "performance": 12.0,
        "quality": 0.5,
    }
    assert json.loads(
        (
            tmp_path
            / "pareto_configs"
            / "pareto_eval_0007_system_config.json"
        ).read_text(encoding="utf-8")
    ) == exact_system
    assert json.loads(
        (
            tmp_path
            / "pareto_configs"
            / "pareto_eval_0007_pipeline_config.json"
        ).read_text(encoding="utf-8")
    ) == exact_pipeline
    controller._resolve_config.assert_not_called()


def test_pareto_pipeline_export_fails_closed_without_exact_eval_config(
    tmp_path: Path,
) -> None:
    controller = Controller.__new__(Controller)
    controller.project_dir = str(tmp_path)

    with pytest.raises(RuntimeError, match="exact persisted config.*eval_0003"):
        controller._save_pareto_frontier(
            {
                "pareto_configs": [{"pipeline": {"component": "fixture"}}],
                "pareto_objectives": [
                    {"performance": 12.0, "quality": 0.5},
                ],
                "pareto_eval_ids": [3],
            },
        )

    assert not (tmp_path / "pareto_frontier.csv").exists()
    assert not (tmp_path / "pareto_configs").exists()


def _publication_controller(tmp_path: Path) -> Controller:
    controller = Controller.__new__(Controller)
    controller.project_dir = str(tmp_path)
    controller._invalid_configs = []
    controller._selected_system_design = Mock(return_value=None)
    controller._ssops = SimpleNamespace(
        flatten_2deep=lambda config: {
            "pipeline__component": config["pipeline"]["component"],
        }
    )
    return controller


def _write_exact_eval(tmp_path: Path, eval_id: int) -> None:
    eval_dir = tmp_path / "evaluations" / f"eval_{eval_id:04d}"
    eval_dir.mkdir(parents=True)
    (eval_dir / "pipeline_config.json").write_text(
        json.dumps({"pipeline": {"component": f"component-{eval_id}"}}),
        encoding="utf-8",
    )
    (eval_dir / "system_config_resolved.json").write_text(
        json.dumps({"layout": {"eval_id": eval_id}}),
        encoding="utf-8",
    )


def _publication(eval_ids: list[int]) -> dict:
    return {
        "pareto_configs": [
            {"pipeline": {"component": f"component-{eval_id}"}}
            for eval_id in eval_ids
        ],
        "pareto_objectives": [
            {"performance": float(eval_id + 1), "quality": 0.5}
            for eval_id in eval_ids
        ],
        "pareto_eval_ids": eval_ids,
    }


def test_pareto_republication_replaces_complete_config_set(
    tmp_path: Path,
) -> None:
    controller = _publication_controller(tmp_path)
    _write_exact_eval(tmp_path, 7)
    _write_exact_eval(tmp_path, 8)

    controller._save_pareto_frontier(_publication([7, 8]))
    # Anything left by an older publication must disappear with the directory
    # swap; consumers must see exactly the eval IDs in the current CSV.
    (tmp_path / "pareto_configs" / "stale-from-previous-run.txt").write_text(
        "stale",
        encoding="utf-8",
    )
    controller._save_pareto_frontier(_publication([7]))

    with (tmp_path / "pareto_frontier.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["eval_id"]) for row in rows] == [7]
    assert {
        path.name for path in (tmp_path / "pareto_configs").iterdir()
    } == {
        "pareto_eval_0007.yaml",
        "pareto_eval_0007_pipeline_config.json",
        "pareto_eval_0007_system_config.json",
    }


def test_empty_pareto_publication_clears_previous_artifacts(
    tmp_path: Path,
) -> None:
    controller = _publication_controller(tmp_path)
    _write_exact_eval(tmp_path, 7)
    controller._save_pareto_frontier(_publication([7]))

    controller._save_pareto_frontier(_publication([]))

    assert not (tmp_path / "pareto_frontier.csv").exists()
    assert not (tmp_path / "pareto_configs").exists()
    assert not list(tmp_path.glob(".pareto-publication-*"))


def test_pareto_staging_failure_preserves_previous_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _publication_controller(tmp_path)
    _write_exact_eval(tmp_path, 7)
    _write_exact_eval(tmp_path, 8)
    controller._save_pareto_frontier(_publication([7]))

    csv_before = (tmp_path / "pareto_frontier.csv").read_bytes()
    configs_before = {
        path.name: path.read_bytes()
        for path in (tmp_path / "pareto_configs").iterdir()
    }

    def _fail_yaml(*_args, **_kwargs):
        raise OSError("injected staging failure")

    monkeypatch.setattr("rag_stack.controller.safe_yaml_dump", _fail_yaml)
    with pytest.raises(OSError, match="injected staging failure"):
        controller._save_pareto_frontier(_publication([8]))

    assert (tmp_path / "pareto_frontier.csv").read_bytes() == csv_before
    assert {
        path.name: path.read_bytes()
        for path in (tmp_path / "pareto_configs").iterdir()
    } == configs_before
    assert not list(tmp_path.glob(".pareto-publication-*"))


def test_pareto_commit_failure_rolls_back_previous_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _publication_controller(tmp_path)
    _write_exact_eval(tmp_path, 7)
    _write_exact_eval(tmp_path, 8)
    controller._save_pareto_frontier(_publication([7]))

    csv_before = (tmp_path / "pareto_frontier.csv").read_bytes()
    configs_before = {
        path.name: path.read_bytes()
        for path in (tmp_path / "pareto_configs").iterdir()
    }
    real_replace = os.replace
    injected = False

    def _fail_new_csv_install(source, destination):
        nonlocal injected
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            not injected
            and source_path.name == "pareto_frontier.csv"
            and source_path.parent.name.startswith(".pareto-publication-")
            and destination_path == tmp_path / "pareto_frontier.csv"
        ):
            injected = True
            raise OSError("injected commit failure")
        return real_replace(source, destination)

    monkeypatch.setattr("rag_stack.controller.os.replace", _fail_new_csv_install)
    with pytest.raises(OSError, match="injected commit failure"):
        controller._save_pareto_frontier(_publication([8]))

    assert injected
    assert (tmp_path / "pareto_frontier.csv").read_bytes() == csv_before
    assert {
        path.name: path.read_bytes()
        for path in (tmp_path / "pareto_configs").iterdir()
    } == configs_before
    assert not list(tmp_path.glob(".pareto-publication-*"))


def test_pareto_publication_fails_closed_without_exact_system_config(
    tmp_path: Path,
) -> None:
    controller = _publication_controller(tmp_path)
    eval_dir = tmp_path / "evaluations" / "eval_0003"
    eval_dir.mkdir(parents=True)
    (eval_dir / "pipeline_config.json").write_text(
        json.dumps({"pipeline": {"component": "component-3"}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="exact persisted system config"):
        controller._save_pareto_frontier(_publication([3]))

    assert not (tmp_path / "pareto_frontier.csv").exists()
    assert not (tmp_path / "pareto_configs").exists()
