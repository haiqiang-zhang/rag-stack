from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rag_stack import calibration_cli
from rag_stack.calibration_inventory import build_calibration_inventory
from rag_stack.cost_model.llm_sim.cli import _build_pipeline_config
from rag_stack.cost_model.retrieval_encode import RetrievalEncodeCostModel
from rag_stack.cost_model.workload_profile import build_workload_search_space


def test_retrieval_encode_calibration_publishes_agnostic_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RetrievalEncodeCostModel()
    pipeline = _build_pipeline_config(
        "huggingface_all_mpnet_base_v2", "retrieval_encode_sim"
    )

    class Runner:
        def run(self, *_args):
            return {(1, 128): 0.01, (8, 256): 0.02}

        def detect_bits(self):
            return "bf16"

    monkeypatch.setattr(model, "_get_benchmark_runner", lambda _cfg: Runner())
    from rag_stack.cost_model.llm_sim import calibration

    monkeypatch.setattr(
        calibration,
        "split_fit_validation_evidence",
        lambda measured: (measured, {}),
    )
    captured: dict[str, object] = {}

    def fake_fit_and_save(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(component=kwargs["component"])

    monkeypatch.setattr(calibration, "fit_and_save", fake_fit_and_save)

    record = model.calibrate(
        hardware=object(),
        pipeline_config=pipeline,
        batch_sizes=(1, 8),
        seq_lens=(128, 256),
        warmup=0,
        measurements=1,
    )

    assert record.component == "agnostic"
    assert captured["component"] == "agnostic"
    assert captured["model_key"] == "huggingface_all_mpnet_base_v2"


def test_workload_cli_uses_lightweight_builder(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rag_stack.cost_model import workload_profile

    captured: dict[str, object] = {}

    def fake_build(**kwargs):
        captured.update(kwargs)
        return {
            "corpus_key": "fixture",
            "embedding_models": ["embedding-a"],
            "project_dir": "/tmp/workload-project",
        }

    monkeypatch.setattr(workload_profile, "build_workload_profiles", fake_build)
    calibration_cli._run_workload(SimpleNamespace(
        config="full.yaml",
        project_dir="/tmp/workload-project",
        qa_data="qa.parquet",
        corpus_data="corpus.parquet",
    ))

    assert captured == {
        "config_path": "full.yaml",
        "project_dir": "/tmp/workload-project",
        "qa_data_path": "qa.parquet",
        "corpus_data_path": "corpus.parquet",
    }
    assert "corpus=fixture" in capsys.readouterr().out


def test_workload_builder_reads_real_ax_search_space_parameters() -> None:
    from rag_stack.search_space import SEP

    search_space = SimpleNamespace(parameters={
        f"vectordb{SEP}embedding_model": SimpleNamespace(
            values=["embedding-a", "embedding-b"]
        ),
        f"vectordb{SEP}nlist_factor": SimpleNamespace(values=[1.0, 2.0]),
        f"vectordb{SEP}nprobe": SimpleNamespace(values=[8, 32]),
    })
    config = {
        "algo_search_space": {
            "vectordb": [{
                "name": "ivf",
                "db_type": "faiss_ivf",
                "embedding_model": "embedding-fixed",
            }],
        },
    }

    result = build_workload_search_space(config, search_space, range(100))

    assert result == {
        "embedding_models": [
            "embedding-a",
            "embedding-b",
            "embedding-fixed",
        ],
        "nlist": [10, 20],
        "nprobe": [8, 32],
    }


def test_public_calibration_surface_has_direct_pd_and_no_dead_stage_route() -> None:
    assert "generator-disagg" in calibration_cli.PASSTHROUGH
    assert "stage-serving" not in calibration_cli.PASSTHROUGH
    help_text = calibration_cli.build_parser().format_help()
    assert "direct P/D prefill+decode stage campaign" in help_text
    assert "V2" not in help_text


def test_status_json_is_parseable_and_returns_two_when_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from rag_stack import calibration_inventory

    inventory = SimpleNamespace(
        deploy_ready=False,
        pending_commands=(),
        to_dict=lambda: {"deploy_ready": False, "items": []},
    )
    monkeypatch.setattr(
        calibration_inventory,
        "build_calibration_inventory",
        lambda *_args, **_kwargs: inventory,
    )

    with pytest.raises(SystemExit) as exc:
        calibration_cli._run_status(
            "status",
            ["--config", "fixture.yaml", "--json"],
            data_dir="/tmp/data",
            profile_dir="/tmp/profiles",
        )

    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out) == {
        "deploy_ready": False,
        "items": [],
    }


def test_custom_cpu_hnsw_plan_never_emits_an_unusable_fit(
    tmp_path: Path,
) -> None:
    config = tmp_path / "custom-cpu.yaml"
    config.write_text(yaml.safe_dump({
        "system": {
            "cpu": {"hardware_key": "custom_cpu"},
        },
    }), encoding="utf-8")
    data_root = tmp_path / "data"
    profile_root = tmp_path / "profiles"

    missing = build_calibration_inventory(
        config,
        data_dir=data_root,
        profile_dir=profile_root,
        scopes=("hnsw",),
    ).items[0]
    assert missing.state == "needs_measure"
    assert len(missing.commands) == 1
    assert "measure hnsw" in missing.commands[0]
    assert "fit hnsw" not in missing.commands[0]

    frozen = data_root / "hnsw" / "custom_cpu.parquet"
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(b"PAR1dataPAR1")
    blocked = build_calibration_inventory(
        config,
        data_dir=data_root,
        profile_dir=profile_root,
        scopes=("hnsw",),
    ).items[0]
    assert blocked.state == "blocked"
    assert not blocked.commands
    assert "all five --hnsw-*" in blocked.detail
