"""File-only tests for publication-export project conversion."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

import rag_stack.optimizer_export as optimizer_export_module
from rag_stack.optimizer_export import convert_optimizer_export


def _trace(*, token_offset: int = 0) -> dict:
    return {
        "queries": [
            {
                "question_id": str(index),
                "calls": [
                    {
                        "stage": "generator",
                        "input_tokens": 10 + token_offset + index,
                        "output_tokens": 2,
                        "step_idx": 0,
                        "model_id": "fixture-generator",
                    }
                ],
            }
            for index in range(2)
        ],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_export(root: Path, *, combined_override: str | None = None) -> Path:
    root.mkdir(parents=True)
    config = {
        "global": {"eval_backend": "static_gt", "rag_ir_mode": "dynamic"},
        "dataset": {
            "dataset_name": "fixture",
            "qa": "datasets/fixture/qa.parquet",
            "corpus": "datasets/fixture/corpus.parquet",
        },
        "eval_backend_setting": {
            "combined_quality": {
                "mode": "mean",
                "metrics": ["answer_correctness", "retrieval_recall"],
            },
            "metrics": [
                {"metric_name": "answer_correctness", "model": "mock"},
                {"metric_name": "retrieval_recall"},
            ],
        },
        "algo_search_space": {"vectordb": [], "pipeline": {"node_lines": []}},
    }
    (root / "rag_stack_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (root / "summary.json").write_text(
        json.dumps({"status": "completed", "n_evaluations": 3}),
        encoding="utf-8",
    )
    (root / "pareto_frontier.csv").write_text(
        "eval_id,quality,performance\n0,0.300000,1.000000\n",
        encoding="utf-8",
    )

    rows = [
        (0, "GT", "", "0.200000", "0.400000"),
        (1, "GT_FP", "", "0.400000", "0.600000"),
        (2, "OOM", "CM: source hardware has no feasible deployment", "0.600000", "0.800000"),
    ]
    with (root / "all_evaluations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pipeline__sequential",
                "performance",
                "quality__combined",
                "quality__answer_correctness",
                "quality__retrieval_recall",
                "eval_type",
                "eval_error",
                "eval_id",
            ],
        )
        writer.writeheader()
        for eval_id, eval_type, error, answer, recall in rows:
            combined = (float(answer) + float(recall)) / 2
            writer.writerow(
                {
                    "pipeline__sequential": repr({"rag_dataflow": "sequential"}),
                    "performance": "0.000000" if eval_type == "OOM" else "1.000000",
                    "quality__combined": (
                        combined_override
                        if combined_override is not None and eval_id == 0
                        else f"{combined:.6f}"
                    ),
                    "quality__answer_correctness": answer,
                    "quality__retrieval_recall": recall,
                    "eval_type": eval_type,
                    "eval_error": error,
                    "eval_id": str(eval_id),
                }
            )

    for eval_id, *_ in rows:
        config_dir = root / "eval_configs" / f"eval_{eval_id:04d}"
        trace_dir = root / "eval_traces" / f"eval_{eval_id:04d}"
        config_dir.mkdir(parents=True)
        trace_dir.mkdir(parents=True)
        (config_dir / "pipeline_config.json").write_text(
            json.dumps({"pipeline_runtime": {"mode": "sequential"}, "id": eval_id}),
            encoding="utf-8",
        )
        if eval_id != 2:
            (config_dir / "system_config.json").write_text(
                json.dumps({"source_gpu": "H100", "eval_id": eval_id}),
                encoding="utf-8",
            )
        (trace_dir / "execution_dag.json").write_text(
            json.dumps(_trace(token_offset=eval_id)), encoding="utf-8"
        )
    return root


def test_convert_export_preserves_every_quality_bearing_row(tmp_path: Path):
    source = _write_export(tmp_path / "export")
    output = tmp_path / "derived-project"
    source_hashes = {
        path.relative_to(source): _sha256(path)
        for path in source.rglob("*")
        if path.is_file()
    }

    manifest = convert_optimizer_export(source, output)

    assert manifest["input_row_count"] == 3
    assert manifest["output_evaluation_count"] == 3
    assert manifest["source_type_counts"] == {"GT": 1, "GT_FP": 1, "OOM": 1}
    assert manifest["trace_query_count"] == 2
    assert (output / "rag_stack_config.yaml").read_bytes() == (
        source / "rag_stack_config.yaml"
    ).read_bytes()
    assert (output / "all_evaluations.csv").read_bytes() == (
        source / "all_evaluations.csv"
    ).read_bytes()
    assert json.loads(
        (output / "evaluations/eval_0002/quality_metrics.json").read_text()
    ) == {"answer_correctness": 0.6, "retrieval_recall": 0.8}
    assert not (output / "evaluations/eval_0002/system_config.json").exists()
    assert (output / "evaluations/eval_0001/system_config.json").is_file()

    marker = json.loads((output / "conversion_manifest.json").read_text())
    oom = next(record for record in marker["records"] if record["eval_id"] == 2)
    assert oom["source_eval_type"] == "OOM"
    assert oom["source_eval_error"].startswith("CM:")
    assert oom["quality_reconstruction"]["kind"] == (
        "all_evaluations_csv_six_decimal"
    )
    assert source_hashes == {
        path.relative_to(source): _sha256(path)
        for path in source.rglob("*")
        if path.is_file()
    }


def test_convert_export_fails_atomically_on_combined_quality_mismatch(
    tmp_path: Path,
):
    source = _write_export(tmp_path / "export", combined_override="0.999999")
    output = tmp_path / "derived-project"

    with pytest.raises(ValueError, match="combined quality mismatch"):
        convert_optimizer_export(source, output)

    assert not output.exists()
    assert list(tmp_path.glob(".derived-project.convert-*")) == []


def test_convert_export_fails_atomically_if_source_changes_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    source = _write_export(tmp_path / "export")
    output = tmp_path / "derived-project"
    trace_path = source / "eval_traces/eval_0001/execution_dag.json"
    original_write_json = optimizer_export_module._write_json

    def mutate_after_manifest_write(path: Path, payload):
        identity = original_write_json(path, payload)
        if path.name == "conversion_manifest.json":
            trace_path.write_text(
                json.dumps(_trace(token_offset=999)), encoding="utf-8"
            )
        return identity

    monkeypatch.setattr(
        optimizer_export_module, "_write_json", mutate_after_manifest_write
    )

    with pytest.raises(RuntimeError, match="source artifact changed during conversion"):
        convert_optimizer_export(source, output)

    assert not output.exists()
    assert list(tmp_path.glob(".derived-project.convert-*")) == []


def test_convert_export_rejects_unknown_trace_header(tmp_path: Path):
    source = _write_export(tmp_path / "export")
    trace_path = source / "eval_traces/eval_0001/execution_dag.json"
    trace = _trace()
    trace["obsolete_header"] = 1
    trace_path.write_text(json.dumps(trace), encoding="utf-8")

    with pytest.raises(ValueError, match="strict frozen quality trace"):
        convert_optimizer_export(source, tmp_path / "derived-project")


def test_convert_export_projects_exact_retired_v2_trace_header(tmp_path: Path):
    source = _write_export(tmp_path / "export")
    trace_path = source / "eval_traces/eval_0001/execution_dag.json"
    versioned_trace = {"schema_version": 2, **_trace(token_offset=1)}
    trace_path.write_text(json.dumps(versioned_trace), encoding="utf-8")
    source_hash = _sha256(trace_path)

    output = tmp_path / "derived-project"
    manifest = convert_optimizer_export(source, output)

    output_trace_path = output / "evaluations/eval_0001/execution_dag.json"
    assert json.loads(output_trace_path.read_text()) == _trace(token_offset=1)
    assert _sha256(trace_path) == source_hash
    record = next(item for item in manifest["records"] if item["eval_id"] == 1)
    assert record["trace_projection"] == {
        "kind": "remove_retired_schema_header",
        "source_schema_version": 2,
        "removed_fields": ["schema_version"],
    }
    assert record["source_artifacts"]["execution_dag.json"]["sha256"] == (
        source_hash
    )
    assert record["output_artifacts"]["execution_dag.json"]["sha256"] == (
        _sha256(output_trace_path)
    )


@pytest.mark.parametrize("invalid_version", [True, 1, 3, "2", 2.0])
def test_convert_export_rejects_non_exact_retired_trace_version(
    tmp_path: Path, invalid_version,
):
    source = _write_export(tmp_path / "export")
    trace_path = source / "eval_traces/eval_0001/execution_dag.json"
    trace_path.write_text(
        json.dumps({"schema_version": invalid_version, **_trace()}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exact integer 2"):
        convert_optimizer_export(source, tmp_path / "derived-project")


def test_convert_export_rejects_non_cm_oom(tmp_path: Path):
    source = _write_export(tmp_path / "export")
    csv_path = source / "all_evaluations.csv"
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    rows[2]["eval_error"] = "CUDA out of memory in quality harness"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="not an authenticated source-CM"):
        convert_optimizer_export(source, tmp_path / "derived-project")
