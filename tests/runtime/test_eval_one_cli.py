from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_stack import eval_one_cli


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    qa_path = tmp_path / "qa.parquet"
    corpus_path = tmp_path / "corpus.parquet"
    qa_path.write_bytes(b"qa")
    corpus_path.write_bytes(b"corpus")
    pipeline_path = tmp_path / "pipeline_config.json"
    system_path = tmp_path / "system_config.json"
    pipeline_path.write_text(
        json.dumps(
            {
                "dataset": {
                    "dataset_name": "unit",
                    "qa": str(qa_path),
                    "corpus": str(corpus_path),
                },
                "eval_backend_setting": {
                    "metrics": [{"metric_name": "deepeval_answer_correctness"}]
                },
                "system": {
                    "performance_objectives": {"selection": "max_throughput"}
                },
                "node_lines": [
                    {
                        "nodes": [
                            {
                                "modules": [
                                    {
                                        "component": "vllm_api",
                                        "model": "test/model",
                                        "uri": "http://quality-server:8000/v1",
                                        "batch": 20,
                                        "request_timeout": 60,
                                    },
                                    {
                                        "component": "query_decompose",
                                        "generator_backend": "vllm_api",
                                        "model": "test/query-model",
                                        "uri": "http://quality-server:8001/v1",
                                    },
                                ]
                            }
                        ]
                    }
                ],
            }
        )
    )
    system_path.write_text(
        json.dumps(
            {
                "layout": {
                    "available_devices": ["cuda:0", "cuda:1"],
                }
            }
        )
    )
    return pipeline_path, system_path


def _fake_runtime(monkeypatch, *, admissible: bool = True):
    observations = {}

    class FakeTrialInvalid(Exception):
        pass

    class FakeDatasetManager:
        def __init__(self, **kwargs):
            observations["dataset_manager"] = kwargs

    class FakeEvaluator:
        def __init__(self, **kwargs):
            observations["evaluator"] = kwargs

    class FakeProvider:
        def __init__(self, evaluator, gpus, *, n_queries, selection):
            observations["provider"] = {
                "evaluator": evaluator,
                "gpus": gpus,
                "n_queries": n_queries,
                "selection": selection,
            }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def evaluate(
            self, pipeline_config, system_config, *, run_dir,
            require_admissible=True,
        ):
            observations["evaluate"] = {
                "pipeline_config": pipeline_config,
                "system_config": system_config,
                "run_dir": run_dir,
                "require_admissible": require_admissible,
            }
            reasons = [] if admissible else ["population_not_saturated"]
            check_names = (
                "driver_errors_window_zero",
                "measurement_rate_stability_waves_ge_2",
                "population_saturated",
                "qps_subwindow_cv_le_0_15",
                "warmup_rate_stable_not_cap_forced",
            )
            return SimpleNamespace(
                performance_score=12.5,
                raw_performance={
                    "qps": 12.5,
                    "population_saturated": admissible,
                    "measured_gt_admissibility": {
                        "admissible": admissible,
                        "reasons": reasons,
                        "checks": {
                            name: {
                                "observed": None,
                                "passed": admissible,
                                "required": True,
                            }
                            for name in check_names
                        },
                    },
                },
                performance_execution_trace={"kind": "performance_trace"},
            )

    monkeypatch.setattr(
        eval_one_cli,
        "_load_runtime",
        lambda: (
            FakeDatasetManager,
            FakeEvaluator,
            FakeProvider,
            FakeTrialInvalid,
        ),
    )
    return observations


def test_eval_one_runs_full_performance_only_and_persists_trace(
    tmp_path, monkeypatch
):
    pipeline_path, system_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "output"
    observations = _fake_runtime(monkeypatch)

    exit_code = eval_one_cli.run_eval_one(
        str(pipeline_path),
        str(system_path),
        output_dir_arg=str(output_dir),
    )

    assert exit_code == 0
    assert observations["provider"]["gpus"] == ["cuda:0", "cuda:1"]
    assert observations["provider"]["n_queries"] is None
    assert observations["provider"]["selection"] == "max_throughput"
    assert observations["evaluate"]["require_admissible"] is False
    assert observations["evaluate"]["pipeline_config"][
        "eval_backend_setting"
    ] == {"performance_only": True}
    modules = observations["evaluate"]["pipeline_config"]["node_lines"][0][
        "nodes"
    ][0]["modules"]
    assert modules[0]["component"] == "vllm"
    assert modules[1]["generator_backend"] == "vllm"
    for module in modules:
        assert module["use_chat_template"] is True
        assert module["max_tokens"] == 4096
        assert "uri" not in module
        assert "batch" not in module
        assert "request_timeout" not in module
    # The canonical input is copied byte-for-byte, without the overlay.
    assert (
        output_dir / "inputs" / "pipeline_config.json"
    ).read_bytes() == pipeline_path.read_bytes()

    record = json.loads((output_dir / "result.json").read_text())
    assert record["status"] == "ok"
    assert record["performance_only"] is True
    assert record["quality_mode"] == "preserve_existing"
    assert record["performance_mode"] == "measured"
    assert record["qps"] == pytest.approx(12.5)
    trace_path = Path(record["performance_execution_trace"])
    assert json.loads(trace_path.read_text()) == {"kind": "performance_trace"}


def test_eval_one_accepts_optimizer_resolved_hardware_device_inventory(
    tmp_path, monkeypatch
):
    pipeline_path, system_path = _write_inputs(tmp_path)
    system_config = json.loads(system_path.read_text())
    del system_config["layout"]["available_devices"]
    system_config["hardware"] = {
        "available_gpus": ["cuda:0", "cuda:1"],
        "available_gpu_count": 2,
    }
    system_path.write_text(json.dumps(system_config))
    observations = _fake_runtime(monkeypatch)

    exit_code = eval_one_cli.run_eval_one(
        str(pipeline_path),
        str(system_path),
        output_dir_arg=str(tmp_path / "output"),
    )

    assert exit_code == 0
    assert observations["provider"]["gpus"] == ["cuda:0", "cuda:1"]


def test_eval_one_rejects_conflicting_concrete_device_inventories(
    tmp_path, monkeypatch
):
    pipeline_path, system_path = _write_inputs(tmp_path)
    system_config = json.loads(system_path.read_text())
    system_config["hardware"] = {"available_gpus": ["cuda:1", "cuda:0"]}
    system_path.write_text(json.dumps(system_config))
    _fake_runtime(monkeypatch)

    exit_code = eval_one_cli.run_eval_one(
        str(pipeline_path),
        str(system_path),
        output_dir_arg=str(tmp_path / "output"),
    )

    assert exit_code == 1
    record = json.loads((tmp_path / "output" / "result.json").read_text())
    assert record["status"] == "error"
    assert "inventories disagree" in record["error"]


def test_eval_one_rejects_positive_qps_from_inadmissible_window(
    tmp_path, monkeypatch
):
    pipeline_path, system_path = _write_inputs(tmp_path)
    output_dir = tmp_path / "output"
    _fake_runtime(monkeypatch, admissible=False)

    exit_code = eval_one_cli.run_eval_one(
        str(pipeline_path),
        str(system_path),
        output_dir_arg=str(output_dir),
    )

    assert exit_code == 2
    record = json.loads((output_dir / "result.json").read_text())
    assert record["status"] == "inadmissible"
    assert record["qps"] == pytest.approx(12.5)
    assert "population_not_saturated" in record["error"]


def test_eval_one_parser_requires_only_the_two_json_paths():
    args = eval_one_cli.build_parser().parse_args(
        ["pipeline_config.json", "system_config.json"]
    )

    assert args.pipeline_config == "pipeline_config.json"
    assert args.system_config == "system_config.json"
    assert args.performance_only is True
    assert args.output_dir is None


@pytest.mark.parametrize("flag", ["--performance-only", "--perf-only"])
def test_eval_one_parser_accepts_explicit_performance_only_mode(flag):
    args = eval_one_cli.build_parser().parse_args(
        [flag, "pipeline_config.json", "system_config.json"]
    )

    assert args.performance_only is True


def test_default_output_dir_is_stable_and_pair_specific(tmp_path, monkeypatch):
    pipeline_dir = tmp_path / "row_001"
    pipeline_dir.mkdir()
    pipeline_path = pipeline_dir / "pipeline_config.json"
    monkeypatch.setattr(eval_one_cli, "_repository_root", lambda: tmp_path)

    first = eval_one_cli._default_output_dir(pipeline_path, b"pipeline", b"system")
    second = eval_one_cli._default_output_dir(pipeline_path, b"pipeline", b"system")
    changed = eval_one_cli._default_output_dir(
        pipeline_path, b"pipeline", b"other-system"
    )

    assert first == second
    assert first != changed
    assert first.parent == tmp_path / "tmp_outputs" / "rag_stack_projects"
