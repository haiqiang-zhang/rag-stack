"""Contracts separating calibration specs from system-transfer targets."""

from __future__ import annotations

from tests import REPO_ROOT

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml
from fastapi import HTTPException

from rag_stack.server.routers import rag_project


def _quality_setting(
    *,
    answer_model: str = "openai/deepseek/deepseek-v4-flash",
    combined_metrics: list[str] | None = None,
    subset: int = 15,
) -> dict:
    return {
        "combined_quality": {
            "mode": "mean",
            "metrics": combined_metrics or ["deepeval_answer_correctness"],
        },
        "metrics": [{
            "metric_name": "deepeval_answer_correctness",
            "model": answer_model,
            "description": "Display text may change.",
        }, {
            "metric_name": "deepeval_faithfulness",
            "model": "openai/deepseek/deepseek-v4-flash",
            "description": "Diagnostic.",
        }],
        "submetric_query_subset": subset,
    }


def _materialized_config_shape() -> dict:
    return {
        "global": {},
        "dataset": {},
        "system": {},
        "optimizer": {"type": "agent_qnehvi", "params": {}},
        "eval_backend_setting": {"metrics": []},
        "algo_search_space": {},
    }


def test_system_transfer_target_separates_metadata_from_system(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.yaml"
    target.write_text(yaml.safe_dump({
        "system_transfer": {
            "n_refine_iterations": 20,
            "generator_backend": "vllm",
        },
        "system": {
            "performance_source": "cost_model",
            "gpu": "fixture_gpu",
        },
    }))

    system, metadata = rag_project._load_system_transfer_target(target)

    assert system == {
        "performance_source": "cost_model",
        "gpu": "fixture_gpu",
    }
    assert metadata == {
        "n_refine_iterations": 20,
        "generator_backend": "vllm",
    }


def test_system_transfer_target_accepts_only_mapping_rag_config_overrides(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.yaml"
    overrides = {
        "optimizer": {
            "type": "agent_qnehvi",
            "params": {"seed": 43},
        },
        "eval_backend_setting": {
            "combined_quality": {
                "mode": "mean",
                "metrics": ["deepeval_answer_correctness"],
            },
            "metrics": [{
                "metric_name": "deepeval_answer_correctness",
                "model": "openai/deepseek/deepseek-v4-flash",
            }],
        },
    }
    target.write_text(yaml.safe_dump({
        "system_transfer": {"rag_config_overrides": overrides},
        "system": {
            "performance_source": "cost_model",
            "gpu": "fixture_gpu",
        },
    }))

    _, metadata = rag_project._load_system_transfer_target(target)

    assert metadata["rag_config_overrides"] == overrides


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ([], "rag_config_overrides.*must be a mapping"),
        ({"global": {}}, "Unsupported.*rag_config_overrides sections"),
        ({"optimizer": "agent_qnehvi"}, "optimizer.*must be a mapping"),
        ({"optimizer": {}}, "optimizer.*type must be a non-empty string"),
        ({"optimizer": {"type": ""}},
         "optimizer.*type must be a non-empty string"),
        ({"optimizer": {"type": 7}},
         "optimizer.*type must be a non-empty string"),
        ({"optimizer": {"type": "agent_qnehvi", "params": []}},
         "optimizer.*params must be a mapping"),
        ({"optimizer": {"type": "agent_qnehvi", "paramz": {}}},
         "optimizer.*unsupported fields.*paramz"),
        ({"eval_backend_setting": []},
         "eval_backend_setting.*must be a mapping"),
        ({"eval_backend_setting": {"metrics": {}}},
         "eval_backend_setting.*metrics must be a list"),
    ],
)
def test_system_transfer_target_rejects_invalid_rag_config_overrides(
    tmp_path: Path, overrides: object, message: str,
) -> None:
    target = tmp_path / "target.yaml"
    target.write_text(yaml.safe_dump({
        "system_transfer": {"rag_config_overrides": overrides},
        "system": {
            "performance_source": "cost_model",
            "gpu": "fixture_gpu",
        },
    }))

    with pytest.raises(HTTPException, match=message) as error:
        rag_project._load_system_transfer_target(target)

    assert error.value.status_code == 400


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        (["answer", {"metric_name": "faithfulness"}], "homogeneous"),
        ([{"metric_name": "answer"}, "faithfulness"], "homogeneous"),
        ([{"metric": "answer"}], "unsupported 'metric' alias"),
        ([{"metric_name": "answer", "metric": "answer"}],
         "unsupported 'metric' alias"),
        ([{"metric_name": 3}], "metric_name must be a non-empty string"),
        ([""], "must be a non-empty string"),
        (["answer", "answer"], "duplicate metric 'answer'"),
        ([{"metric_name": "answer"}, {"metric_name": "answer"}],
         "duplicate metric 'answer'"),
    ],
)
def test_system_transfer_target_rejects_invalid_static_gt_metric_shapes(
    tmp_path: Path, metrics: object, message: str,
) -> None:
    target = tmp_path / "target.yaml"
    target.write_text(yaml.safe_dump({
        "system_transfer": {
            "rag_config_overrides": {
                "eval_backend_setting": {"metrics": metrics},
            },
        },
        "system": {
            "performance_source": "cost_model",
            "gpu": "fixture_gpu",
        },
    }))

    system, metadata = rag_project._load_system_transfer_target(target)

    with pytest.raises(HTTPException, match=message) as error:
        rag_project._materialize_system_transfer_config(
            _materialized_config_shape(), system, metadata,
        )

    assert error.value.status_code == 400


def test_flashrag_transfer_preserves_mixed_metric_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _materialized_config_shape()
    source["global"] = {"eval_backend": "flashrag"}
    setting = {
        "combined_quality": {"mode": "mean", "metrics": ["em"]},
        "metrics": ["em", {"metric_name": "f1"}],
    }
    source["eval_backend_setting"] = deepcopy(setting)

    target = rag_project._materialize_system_transfer_config(
        source,
        {"performance_source": "cost_model", "gpu": "fixture_gpu"},
        {"rag_config_overrides": {
            "eval_backend_setting": deepcopy(setting),
        }},
    )
    validator = Mock()
    validator.return_value.validate.return_value = []
    monkeypatch.setattr(
        "rag_stack.config_validator.ConfigValidator", validator,
    )

    rag_project._validate_materialized_system_transfer_config(target)

    assert target["eval_backend_setting"]["metrics"] == [
        "em", {"metric_name": "f1"},
    ]
    validator.assert_called_once_with(target, check_model_cache=False)


def test_profiled_a100_transfer_declares_s43_optimizer_and_four_metrics() -> None:
    root = REPO_ROOT
    path = root / "configs/system_transfer/transfer_sgs_a100_8x_profiled.yaml"

    system, metadata = rag_project._load_system_transfer_target(path)
    overrides = metadata["rag_config_overrides"]
    quality_reference = yaml.safe_load(
        (root / "configs/rag_stack/paper_bench_agent_s45.yaml").read_text()
    )
    s43_reference = yaml.safe_load(
        (root / "configs/rag_stack/paper_bench_msmarco_agent_v2_s43.yaml")
        .read_text()
    )
    expected_optimizer = deepcopy(s43_reference["optimizer"])
    expected_optimizer["params"].update({
        "agent_strict": True,
        "agent_wall_timeout_s": 150,
    })

    assert system["rag_cm"]["calibration_mode"] == "strict"
    assert metadata["n_refine_iterations"] == 20
    assert metadata["generator_backend"] == "vllm"
    assert overrides["optimizer"] == expected_optimizer
    assert (
        overrides["eval_backend_setting"]
        == quality_reference["eval_backend_setting"]
    )
    eval_setting = overrides["eval_backend_setting"]
    assert eval_setting["combined_quality"] == {
        "mode": "mean",
        "metrics": ["deepeval_answer_correctness"],
    }
    assert [metric["metric_name"] for metric in eval_setting["metrics"]] == [
        "deepeval_answer_correctness",
        "deepeval_faithfulness",
        "deepeval_context_precision",
        "deepeval_context_recall",
    ]
    assert {
        metric["model"] for metric in eval_setting["metrics"]
    } == {"openai/deepseek/deepseek-v4-flash"}


def test_system_transfer_picker_is_scoped_to_configured_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "systems"
    root.mkdir()
    (root / "target.yaml").write_text(yaml.safe_dump({
        "system_transfer": {
            "n_refine_iterations": 20,
            "generator_backend": "vllm",
        },
        "system": {
            "performance_source": "cost_model",
            "gpu": "fixture_gpu",
        },
    }))
    (root / "not-a-target.yaml").write_text("unrelated: true\n")
    monkeypatch.setattr(rag_project, "_get_system_yaml_root", lambda: root)

    listing = rag_project.list_system_yamls()
    paths = {item["path"] for item in listing["system_yamls"]}

    assert listing["root"] == str(root)
    assert paths == {"target.yaml"}
    assert listing["system_yamls"][0]["n_refine_iterations"] == 20
    assert listing["system_yamls"][0]["generator_backend"] == "vllm"


def test_system_transfer_path_cannot_escape_configured_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "systems"
    root.mkdir()
    monkeypatch.setattr(rag_project, "_get_system_yaml_root", lambda: root)

    with pytest.raises(HTTPException, match="Invalid system YAML path") as error:
        rag_project._system_yaml_path("../outside.yaml")

    assert error.value.status_code == 400


@pytest.mark.parametrize(
    "optimizer_override",
    [
        {},
        {"type": ""},
        {"type": 3},
        {"type": "agent_qnehvi", "params": []},
        {"type": "agent_qnehvi", "paramz": {"seed": 43}},
    ],
)
def test_invalid_optimizer_override_fails_before_target_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    optimizer_override: object,
) -> None:
    projects = tmp_path / "projects"
    systems = tmp_path / "systems"
    source = projects / "source"
    source.mkdir(parents=True)
    systems.mkdir()
    (source / "rag_stack_config.yaml").write_text(yaml.safe_dump(
        _materialized_config_shape(), sort_keys=False,
    ))
    (systems / "a100.yaml").write_text(yaml.safe_dump({
        "system_transfer": {
            "rag_config_overrides": {"optimizer": optimizer_override},
        },
        "system": {
            "performance_source": "cost_model",
            "gpu": "A100_80GB_GPU",
        },
    }, sort_keys=False))
    launch = Mock()
    monkeypatch.setattr(rag_project, "_get_project_dir", lambda: projects)
    monkeypatch.setattr(rag_project, "_get_system_yaml_root", lambda: systems)
    monkeypatch.setattr(rag_project, "run_project", launch)

    with pytest.raises(HTTPException) as error:
        rag_project.system_transfer(rag_project.SystemTransferRequest(
            project_name="target",
            source_project="source",
            system_yaml="a100.yaml",
        ))

    assert error.value.status_code == 400
    assert not (projects / "target").exists()
    launch.assert_not_called()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda setting: setting["combined_quality"].update({
            "metrics": ["deepeval_faithfulness"],
        }),
        lambda setting: setting["metrics"][0].update({
            "model": "openai/a-different-judge",
        }),
        lambda setting: setting.update({"submetric_query_subset": 10}),
    ],
    ids=["combined-input", "objective-judge", "sampling-setting"],
)
def test_system_transfer_rejects_quality_objective_contract_changes(
    mutate,
) -> None:
    source_setting = _quality_setting()
    target_setting = deepcopy(source_setting)
    mutate(target_setting)
    source_config = {
        "global": {},
        "dataset": {},
        "system": {},
        "optimizer": {"type": "agent_qnehvi"},
        "eval_backend_setting": source_setting,
        "algo_search_space": {},
    }

    with pytest.raises(
        HTTPException, match="must not change the combined quality objective",
    ) as error:
        rag_project._materialize_system_transfer_config(
            source_config,
            {"performance_source": "cost_model", "gpu": "A100_80GB_GPU"},
            {"rag_config_overrides": {
                "eval_backend_setting": target_setting,
            }},
        )

    assert error.value.status_code == 400


@pytest.mark.parametrize("target_subset", [None, 0])
def test_system_transfer_quality_contract_canonicalizes_runtime_defaults(
    target_subset: int | None,
) -> None:
    source_setting = _quality_setting()
    source_setting["combined_quality"].pop("mode")
    source_setting.pop("submetric_query_subset")
    target_setting = deepcopy(source_setting)
    target_setting["combined_quality"]["mode"] = "MEAN"
    target_setting["strategy"] = "mean"
    target_setting["submetric_query_subset"] = target_subset
    source_config = _materialized_config_shape()
    source_config["eval_backend_setting"] = source_setting

    target = rag_project._materialize_system_transfer_config(
        source_config,
        {"performance_source": "cost_model", "gpu": "A100_80GB_GPU"},
        {"rag_config_overrides": {
            "eval_backend_setting": target_setting,
        }},
    )

    assert target["eval_backend_setting"] == target_setting


@pytest.mark.parametrize(
    "mutate",
    [
        lambda setting: setting.update({
            "metrics": list(reversed(setting["metrics"])),
        }),
        lambda setting: setting["combined_quality"].update({
            "metrics": list(reversed(
                setting["combined_quality"]["metrics"],
            )),
        }),
    ],
    ids=["metric-order", "combined-metric-order"],
)
def test_system_transfer_quality_contract_preserves_metric_order(
    mutate,
) -> None:
    source_setting = _quality_setting(combined_metrics=[
        "deepeval_answer_correctness", "deepeval_faithfulness",
    ])
    target_setting = deepcopy(source_setting)
    mutate(target_setting)
    source_config = _materialized_config_shape()
    source_config["eval_backend_setting"] = source_setting

    with pytest.raises(
        HTTPException, match="must not change the combined quality objective",
    ):
        rag_project._materialize_system_transfer_config(
            source_config,
            {"performance_source": "cost_model", "gpu": "A100_80GB_GPU"},
            {"rag_config_overrides": {
                "eval_backend_setting": target_setting,
            }},
        )


@pytest.mark.parametrize("duplicate_location", ["metrics", "combined"])
def test_system_transfer_quality_contract_rejects_duplicate_metrics(
    duplicate_location: str,
) -> None:
    target_setting = _quality_setting(combined_metrics=[
        "deepeval_answer_correctness", "deepeval_faithfulness",
    ])
    source_config = _materialized_config_shape()
    source_config["eval_backend_setting"] = _quality_setting(combined_metrics=[
        "deepeval_answer_correctness", "deepeval_faithfulness",
    ])
    if duplicate_location == "metrics":
        target_setting["metrics"] = [
            target_setting["metrics"][0], *target_setting["metrics"],
        ]
    else:
        target_setting["combined_quality"]["metrics"] = [
            "deepeval_answer_correctness",
            "deepeval_answer_correctness",
        ]

    with pytest.raises(HTTPException, match="duplicate metric"):
        rag_project._materialize_system_transfer_config(
            source_config,
            {"performance_source": "cost_model", "gpu": "A100_80GB_GPU"},
            {"rag_config_overrides": {
                "eval_backend_setting": target_setting,
            }},
        )


@pytest.mark.parametrize(
    ("combined", "message"),
    [
        ({"mode": "mean", "metrics": [3]}, "must be a non-empty string"),
        ({"mode": "mean", "metrics": [{}]}, "must be a non-empty string"),
        ({"mode": "mean", "metrics": [""]}, "must be a non-empty string"),
        ({"mode": "weighted", "weights": {3: 1.0}},
         "weights keys must be non-empty strings"),
    ],
)
def test_system_transfer_quality_contract_rejects_invalid_combined_names(
    combined: dict, message: str,
) -> None:
    target_setting = _quality_setting()
    target_setting["combined_quality"] = combined
    source_config = _materialized_config_shape()
    source_config["eval_backend_setting"] = _quality_setting()

    with pytest.raises(HTTPException, match=message):
        rag_project._materialize_system_transfer_config(
            source_config,
            {"performance_source": "cost_model", "gpu": "A100_80GB_GPU"},
            {"rag_config_overrides": {
                "eval_backend_setting": target_setting,
            }},
        )


def test_system_transfer_quality_contract_failure_precedes_target_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    systems = tmp_path / "systems"
    source = projects / "source"
    source.mkdir(parents=True)
    systems.mkdir()
    source_config = {
        "global": {"eval_backend": "static_gt"},
        "dataset": {},
        "system": {"performance_source": "cost_model", "gpu": "H100_GPU"},
        "optimizer": {"type": "agent_qnehvi"},
        "eval_backend_setting": _quality_setting(),
        "algo_search_space": {},
    }
    (source / "rag_stack_config.yaml").write_text(
        yaml.safe_dump(source_config, sort_keys=False),
    )
    changed = _quality_setting(
        combined_metrics=["deepeval_faithfulness"],
    )
    (systems / "a100.yaml").write_text(yaml.safe_dump({
        "system_transfer": {
            "rag_config_overrides": {
                "eval_backend_setting": changed,
            },
        },
        "system": {
            "performance_source": "cost_model",
            "gpu": "A100_80GB_GPU",
        },
    }, sort_keys=False))
    launch = Mock()
    validate = Mock()
    monkeypatch.setattr(rag_project, "_get_project_dir", lambda: projects)
    monkeypatch.setattr(rag_project, "_get_system_yaml_root", lambda: systems)
    monkeypatch.setattr(rag_project, "run_project", launch)
    monkeypatch.setattr(
        rag_project, "_validate_materialized_system_transfer_config", validate,
    )

    with pytest.raises(
        HTTPException, match="must not change the combined quality objective",
    ) as error:
        rag_project.system_transfer(rag_project.SystemTransferRequest(
            project_name="target",
            source_project="source",
            system_yaml="a100.yaml",
        ))

    assert error.value.status_code == 400
    assert not (projects / "target").exists()
    validate.assert_not_called()
    launch.assert_not_called()


def test_system_transfer_materialized_config_validation_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = Mock()
    validator.return_value.validate.return_value = ["broken final config"]
    monkeypatch.setattr(
        "rag_stack.config_validator.ConfigValidator", validator,
    )
    config = _materialized_config_shape()

    with pytest.raises(
        HTTPException, match="broken final config",
    ) as error:
        rag_project._validate_materialized_system_transfer_config(config)

    assert error.value.status_code == 400
    validator.assert_called_once_with(config, check_model_cache=False)


def test_system_transfer_materialized_config_shape_exception_is_http_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = Mock()
    validator.return_value.validate.side_effect = AttributeError(
        "nested config has no attribute get",
    )
    monkeypatch.setattr(
        "rag_stack.config_validator.ConfigValidator", validator,
    )

    with pytest.raises(
        HTTPException, match="nested config has no attribute get",
    ) as error:
        rag_project._validate_materialized_system_transfer_config(
            _materialized_config_shape(),
        )

    assert error.value.status_code == 400


def test_system_transfer_materialized_config_does_not_hide_system_bug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = Mock()
    validator.return_value.validate.side_effect = RuntimeError(
        "validator implementation bug",
    )
    monkeypatch.setattr(
        "rag_stack.config_validator.ConfigValidator", validator,
    )

    with pytest.raises(RuntimeError, match="validator implementation bug"):
        rag_project._validate_materialized_system_transfer_config(
            _materialized_config_shape(),
        )


@pytest.mark.parametrize(
    ("request_override", "expected_iterations", "expected_source"),
    [
        (None, 10, "system_yaml"),
        (3, 3, "request_override"),
    ],
)
def test_system_transfer_resolves_yaml_default_or_request_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request_override: int | None,
    expected_iterations: int,
    expected_source: str,
) -> None:
    projects = tmp_path / "projects"
    systems = tmp_path / "systems"
    datasets = tmp_path / "datasets"
    source = projects / "source"
    source.mkdir(parents=True)
    systems.mkdir()
    msmarco = datasets / "msmarco_full"
    msmarco.mkdir(parents=True)
    (msmarco / "corpus.parquet").write_bytes(b"test corpus placeholder")
    (msmarco / "qa_test.parquet").write_bytes(b"test qa placeholder")

    source_config = {
        "global": {"eval_backend": "static_gt", "rag_ir_mode": "dynamic"},
        "dataset": {
            "dataset_name": "msmarco_full",
            "corpus": "datasets/msmarco_full/corpus.parquet",
            "qa": "datasets/msmarco_full/qa_test.parquet",
        },
        "system": {"performance_source": "cost_model", "gpu": "H100_GPU"},
        "optimizer": {
            "type": "agent_qnehvi",
            "params": {"seed": 1, "retired_source_param": True},
        },
        "eval_backend_setting": {
            "combined_quality": {
                "mode": "mean",
                "metrics": ["deepeval_answer_correctness"],
            },
            "metrics": [{
                "metric_name": "deepeval_answer_correctness",
                "model": "openai/deepseek/deepseek-v4-flash",
                "description": "Old display text.",
            }, {
                "metric_name": "retrieval_token_recall",
                "description": "Retired source diagnostic.",
            }],
            "submetric_query_subset": 15,
        },
        "algo_search_space": {},
        # Neither source nor target transfer metadata belongs in the RAG config.
        "system_transfer": {"n_refine_iterations": 99},
        "legacy_metadata": {"must_not_survive": True},
    }
    (source / "rag_stack_config.yaml").write_text(
        yaml.safe_dump(source_config, sort_keys=False),
    )
    (systems / "a100.yaml").write_text(yaml.safe_dump({
        "system_transfer": {
            "n_refine_iterations": 10,
            "generator_backend": "vllm",
            "rag_config_overrides": {
                "optimizer": {
                    "type": "agent_qnehvi",
                    "params": {"seed": 43, "agent_strict": True},
                },
                "eval_backend_setting": {
                    "combined_quality": {
                        "mode": "mean",
                        "metrics": ["deepeval_answer_correctness"],
                    },
                    "metrics": [{
                        "metric_name": "deepeval_answer_correctness",
                        "model": "openai/deepseek/deepseek-v4-flash",
                        "description": "New display text.",
                    }, {
                        "metric_name": "deepeval_faithfulness",
                        "model": "openai/deepseek/deepseek-v4-flash",
                        "description": "New target diagnostic.",
                    }],
                    "submetric_query_subset": 15,
                },
            },
        },
        "system": {
            "performance_source": "cost_model",
            "gpu": "A100_80GB_GPU",
            "cpu": {"hardware_key": "epyc_7742_2s"},
            "cm_search_space": {"max_batch_size_request": 256},
        },
    }, sort_keys=False))

    launched = []

    def _fake_run(request):
        launched.append(request)
        return {"status": "running"}

    monkeypatch.setattr(rag_project, "_get_project_dir", lambda: projects)
    monkeypatch.setattr(rag_project, "_get_system_yaml_root", lambda: systems)
    monkeypatch.setattr(rag_project, "_get_datasets_dir", lambda: datasets)
    monkeypatch.setattr(rag_project, "run_project", _fake_run)
    validate = Mock()
    monkeypatch.setattr(
        rag_project, "_validate_materialized_system_transfer_config", validate,
    )
    response = rag_project.system_transfer(rag_project.SystemTransferRequest(
        project_name="target",
        source_project="source",
        system_yaml="a100.yaml",
        n_refine_iterations=request_override,
    ))

    assert len(launched) == 1
    assert launched[0].n_refine_iterations == expected_iterations
    assert launched[0].corpus_data == "msmarco_full/corpus.parquet"
    assert launched[0].qa_data == "msmarco_full/qa_test.parquet"
    assert response["n_refine_iterations"] == expected_iterations
    assert response["n_refine_iterations_source"] == expected_source
    assert response["generator_backend"] == "vllm"
    target = projects / "target"
    target_config = yaml.safe_load(
        (target / "rag_stack_config.yaml").read_text(),
    )
    assert set(target_config) == {
        "global",
        "dataset",
        "system",
        "optimizer",
        "eval_backend_setting",
        "algo_search_space",
    }
    assert target_config["system"]["gpu"] == "A100_80GB_GPU"
    assert target_config["optimizer"] == {
        "type": "agent_qnehvi",
        "params": {"seed": 43, "agent_strict": True},
    }
    target_metrics = target_config["eval_backend_setting"]["metrics"]
    assert [metric["metric_name"] for metric in target_metrics] == [
        "deepeval_answer_correctness", "deepeval_faithfulness",
    ]
    assert "retired_source_param" not in str(target_config["optimizer"])
    assert "retrieval_token_recall" not in str(
        target_config["eval_backend_setting"],
    )
    validate.assert_called_once_with(target_config)
    # The API passes dataset paths through to the worker instead of copying the
    # 8.84M-passage corpus while assembling a converted transfer project.
    assert not (target / "data").exists()
    provenance = json.loads((target / "system_transfer.json").read_text())
    assert provenance["kind"] == "system_transfer_launch"
    assert provenance["n_refine_iterations"] == expected_iterations
    assert provenance["n_refine_iterations_source"] == expected_source
    assert provenance["generator_backend"] == "vllm"
    assert provenance["vllm_chat_projection"]["backend"] == "vllm_chat"
    assert "cm_worker_env" not in provenance
    assert response["system_transfer_provenance"] == provenance


def test_system_transfer_projects_api_modules_to_local_vllm_chat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    projects = tmp_path / "projects"
    systems = tmp_path / "systems"
    source = projects / "source"
    source.mkdir(parents=True)
    source_data = source / "data"
    source_data.mkdir()
    (source_data / "qa.parquet").write_bytes(b"qa")
    (source_data / "corpus.parquet").write_bytes(b"corpus")
    systems.mkdir()
    source_config = {
        "global": {"eval_backend": "static_gt", "rag_ir_mode": "dynamic"},
        "dataset": {},
        "system": {"performance_source": "cost_model", "gpu": "H100_GPU"},
        "optimizer": {"type": "agent_qnehvi"},
        "eval_backend_setting": {},
        "algo_search_space": {
            "pipeline": {
                "node_lines": [{
                    "nodes": [{
                        "stage": "query_expansion",
                        "modules": [
                            {
                                "component": component,
                                "generator_backend": "vllm_api",
                                "uri": "${RAG_LLM_BASE_URL}",
                                "model": {"ordered": ["Qwen/test"]},
                                "temperature": [0.0, 0.7],
                            }
                            for component in (
                                "query_decompose", "hyde",
                                "multi_query_expansion",
                            )
                        ],
                    }, {
                        "stage": "generator",
                        "modules": [{
                            "component": "vllm_api",
                            "uri": "${RAG_LLM_BASE_URL}",
                            "batch": 100,
                            "request_timeout": 600,
                            "model": {"ordered": ["Qwen/test"]},
                            "max_tokens": 512,
                            "temperature": [0.0, 0.7],
                        }],
                    }],
                }],
            },
        },
    }
    source_config_path = source / "rag_stack_config.yaml"
    source_config_path.write_text(yaml.safe_dump(source_config, sort_keys=False))
    source_bytes = source_config_path.read_bytes()
    (systems / "a100.yaml").write_text(yaml.safe_dump({
        "system_transfer": {
            "n_refine_iterations": 10,
            "generator_backend": "vllm",
        },
        "system": {
            "performance_source": "cost_model",
            "gpu": "A100_80GB_GPU",
        },
    }))
    monkeypatch.setattr(rag_project, "_get_project_dir", lambda: projects)
    monkeypatch.setattr(rag_project, "_get_system_yaml_root", lambda: systems)
    monkeypatch.delenv("RAG_LLM_BASE_URL", raising=False)
    preflight = Mock(side_effect=ValueError("gateway unavailable"))
    monkeypatch.setattr(
        "rag_stack.cli._preflight_hardware_replay_llm_config",
        preflight,
    )
    launch = Mock(return_value={"status": "running"})
    monkeypatch.setattr(rag_project, "run_project", launch)
    validate = Mock()
    monkeypatch.setattr(
        rag_project, "_validate_materialized_system_transfer_config", validate,
    )

    rag_project.system_transfer(rag_project.SystemTransferRequest(
        project_name="target",
        source_project="source",
        system_yaml="a100.yaml",
    ))

    launch.assert_called_once()
    preflight.assert_not_called()
    assert source_config_path.read_bytes() == source_bytes
    target_path = projects / "target"
    target = yaml.safe_load((target_path / "rag_stack_config.yaml").read_text())
    modules = [
        module
        for node_line in target["algo_search_space"]["pipeline"]["node_lines"]
        for node in node_line["nodes"]
        for module in node["modules"]
    ]
    local_modules = [
        module for module in modules
        if module.get("component") == "vllm"
        or module.get("generator_backend") == "vllm"
    ]
    assert len(local_modules) == 4
    assert all("uri" not in module for module in local_modules)
    assert all("batch" not in module for module in local_modules)
    assert all("request_timeout" not in module for module in local_modules)
    assert all(module["use_chat_template"] is True for module in local_modules)
    assert all(module["gpu_memory_utilization"] == 0.4 for module in local_modules)
    assert all(module["dtype"] == "bfloat16" for module in local_modules)
    query_modules = [
        module for module in local_modules
        if module.get("generator_backend") == "vllm"
    ]
    assert len(query_modules) == 3
    assert all(module["max_tokens"] == 4096 for module in query_modules)
    assert "vllm_api" not in (target_path / "rag_stack_config.yaml").read_text()
    provenance = json.loads((target_path / "system_transfer.json").read_text())
    assert provenance["generator_backend"] == "vllm"
    assert provenance["vllm_chat_projection"] == {
        "backend": "vllm_chat",
        "component_conversions": 1,
        "generator_backend_conversions": 3,
        "api_default_max_tokens_preserved": 3,
        "transport_fields_removed": 6,
        "local_vllm_modules": 4,
    }
    validate.assert_called_once_with(target)


def test_system_transfer_dataset_path_cannot_escape_datasets_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"not a dataset")
    monkeypatch.setattr(rag_project, "_get_datasets_dir", lambda: datasets)

    with pytest.raises(HTTPException, match="inside datasets_dir") as error:
        rag_project._resolve_dataset_file("../outside.parquet", "dataset.corpus")

    assert error.value.status_code == 400


@pytest.mark.parametrize("invalid", [-1, True, 1.5, "10"])
def test_system_transfer_yaml_rejects_invalid_iteration_metadata(
    tmp_path: Path,
    invalid: object,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump({
        "system_transfer": {"n_refine_iterations": invalid},
        "system": {
            "performance_source": "cost_model",
            "gpu": "A100_80GB_GPU",
        },
    }))
    with pytest.raises(HTTPException, match="non-negative integer") as error:
        rag_project._load_system_transfer_target(path)
    assert error.value.status_code == 400


@pytest.mark.parametrize("invalid", [None, True, 1, "local", "openai"])
def test_system_transfer_yaml_rejects_invalid_generator_backend(
    tmp_path: Path,
    invalid: object,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump({
        "system_transfer": {"generator_backend": invalid},
        "system": {
            "performance_source": "cost_model",
            "gpu": "A100_80GB_GPU",
        },
    }))
    with pytest.raises(HTTPException, match="generator_backend.*one of") as error:
        rag_project._load_system_transfer_target(path)
    assert error.value.status_code == 400


@pytest.mark.parametrize("backend", ["inherit", "vllm", "vllm_api"])
def test_system_transfer_yaml_accepts_supported_generator_backend(
    tmp_path: Path,
    backend: str,
) -> None:
    path = tmp_path / "valid.yaml"
    path.write_text(yaml.safe_dump({
        "system_transfer": {"generator_backend": backend},
        "system": {
            "performance_source": "cost_model",
            "gpu": "A100_80GB_GPU",
        },
    }))
    _, metadata = rag_project._load_system_transfer_target(path)
    assert metadata["generator_backend"] == backend
