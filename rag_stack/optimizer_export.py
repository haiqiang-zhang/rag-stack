"""Convert a publication optimizer export into a standard project.

Publication exports intentionally keep resolved pipeline configs and quality
traces in separate ``eval_configs`` / ``eval_traces`` trees.  Hardware replay
and :class:`rag_stack.quality_store.QualityStore` consume the normal project
layout instead::

    evaluations/eval_NNNN/{pipeline_config,execution_dag,quality_metrics}.json

This module performs that conversion without modifying the source export.
The conversion is deliberately fail-closed: every CSV row must have one
pipeline config, one frozen trace, and every configured quality metric;
the combined quality value is checked against the config's aggregation rule.
The destination appears only after the complete derived project and its
provenance manifest have been written successfully.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import yaml

from rag_stack.rag_ir.trace import (
    validate_quality_trace_envelope,
)


CONVERSION_FORMAT = "rag_stack_optimizer_export_conversion"
CONVERSION_FORMAT_VERSION = 1
_EVAL_DIR_RE = re.compile(r"eval_(\d{4})$")
_SUPPORTED_SOURCE_TYPES = frozenset({"GT", "GT_FP", "OOM"})
_ROOT_FILES = (
    "rag_stack_config.yaml",
    "all_evaluations.csv",
    "summary.json",
)
_OPTIONAL_ROOT_FILES = (
    "pareto_frontier.csv",
    "pareto_quality_measured_performance.csv",
)
_CONVERSION_MANIFEST_FIELDS = frozenset({
    "configured_quality_metrics",
    "created_at_utc",
    "format",
    "format_version",
    "input_row_count",
    "output_evaluation_count",
    "output_project_dir",
    "quality_origin",
    "quality_precision",
    "records",
    "root_artifacts",
    "source_files_unchanged",
    "source_project_dir",
    "source_type_counts",
    "trace_query_count",
})
_RETIRED_TRACE_SCHEMA_VERSION = 2


def _validate_conversion_manifest(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise TypeError("conversion manifest must be an object")
    if set(payload) != _CONVERSION_MANIFEST_FIELDS:
        raise ValueError(
            "conversion manifest fields do not match the current format: "
            f"{sorted(set(payload) ^ _CONVERSION_MANIFEST_FIELDS)}"
        )
    if (
        payload["format"] != CONVERSION_FORMAT
        or payload["format_version"] != CONVERSION_FORMAT_VERSION
    ):
        raise ValueError("unsupported conversion manifest format")
    if not isinstance(payload["records"], list):
        raise TypeError("conversion manifest records must be a list")
    if payload["output_evaluation_count"] != len(payload["records"]):
        raise ValueError("conversion manifest output count is inconsistent")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    return {
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _bytes_identity(payload: bytes) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _require_regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required regular file is missing or a symlink: {path}")


def _capture_regular_file(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Capture one source artifact and bind its identity to those exact bytes."""

    _require_regular_file(path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"could not read source artifact {path}: {exc}") from exc
    identity = _bytes_identity(payload)
    try:
        current_identity = _file_identity(path)
    except OSError as exc:
        raise RuntimeError(
            f"source artifact changed while it was captured: {path}"
        ) from exc
    if current_identity != identity:
        raise RuntimeError(f"source artifact changed while it was captured: {path}")
    return payload, identity


def _json_object_from_bytes(path: Path, payload: bytes) -> dict:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def _load_json_object(path: Path) -> dict:
    payload, _ = _capture_regular_file(path)
    return _json_object_from_bytes(path, payload)


def _write_json(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return _file_identity(path)


def _write_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    identity = _file_identity(path)
    expected = _bytes_identity(payload)
    if identity != expected:
        raise RuntimeError(f"captured-byte write integrity check failed: {path}")
    return identity


def _project_export_trace(trace: dict) -> tuple[dict, dict[str, Any]]:
    """Project the one frozen versioned export shape to today's trace contract.

    Publication optimizer exports produced before the trace contract became
    versionless carry a single top-level ``schema_version: 2`` header.  That
    header never changed trace semantics.  Accept only that exact historical
    integer and otherwise keep the current validator fail-closed; the source
    artifact remains untouched and the conversion manifest records both file
    identities.
    """

    if "schema_version" not in trace:
        validate_quality_trace_envelope(trace)
        return trace, {"kind": "identity"}

    source_version = trace["schema_version"]
    if (
        isinstance(source_version, bool)
        or not isinstance(source_version, int)
        or source_version != _RETIRED_TRACE_SCHEMA_VERSION
    ):
        raise ValueError(
            "unsupported optimizer-export trace schema_version="
            f"{source_version!r}; expected the exact integer "
            f"{_RETIRED_TRACE_SCHEMA_VERSION}"
        )

    projected = {
        key: value
        for key, value in trace.items()
        if key != "schema_version"
    }
    validate_quality_trace_envelope(projected)
    return projected, {
        "kind": "remove_retired_schema_header",
        "source_schema_version": _RETIRED_TRACE_SCHEMA_VERSION,
        "removed_fields": ["schema_version"],
    }


def _configured_metric_names(config: dict) -> list[str]:
    metrics = (config.get("eval_backend_setting") or {}).get("metrics") or []
    names: list[str] = []
    for position, metric in enumerate(metrics):
        if not isinstance(metric, dict):
            raise ValueError(
                f"eval_backend_setting.metrics[{position}] must be an object"
            )
        name = metric.get("metric_name") or metric.get("metric")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"eval_backend_setting.metrics[{position}] has no metric name"
            )
        names.append(name)
    if not names:
        raise ValueError("rag_stack_config.yaml configures no quality metrics")
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate configured quality metric names: {names}")
    return names


def _finite_float(value: Any, *, field: str) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field} is empty")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} is not finite: {value!r}")
    return parsed


def _aggregate_quality_metrics(metrics: dict[str, float], spec: dict | None) -> float:
    """Mirror the runtime quality aggregation without importing ML/data deps."""

    if spec is None:
        return sum(metrics.values()) / len(metrics)
    mode = str(spec.get("mode", "mean")).lower()
    if mode == "mean":
        names = list(spec.get("metrics") or metrics)
        if not names:
            raise ValueError("combined_quality mean has no metrics")
        missing = set(names) - set(metrics)
        if missing:
            raise ValueError(
                f"combined_quality references missing metrics: {sorted(missing)}"
            )
        return sum(metrics[name] for name in names) / len(names)
    if mode == "weighted":
        weights = dict(spec.get("weights") or {})
        if not weights:
            raise ValueError("combined_quality weighted mode has no weights")
        missing = set(weights) - set(metrics)
        if missing:
            raise ValueError(
                f"combined_quality references missing metrics: {sorted(missing)}"
            )
        denominator = sum(float(weight) for weight in weights.values())
        if denominator <= 0:
            raise ValueError("combined_quality weights must have a positive sum")
        return sum(
            float(weights[name]) * metrics[name] for name in weights
        ) / denominator
    raise ValueError(f"unsupported combined_quality mode: {mode!r}")


def _csv_rows(
    csv_path: Path, payload: bytes
) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"invalid UTF-8 CSV {csv_path}: {exc}") from exc
    handle = io.StringIO(text, newline="")
    reader = csv.DictReader(handle)
    if reader.fieldnames is None:
        raise ValueError(f"CSV has no header: {csv_path}")
    rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no evaluation rows: {csv_path}")
    if any(None in row for row in rows):
        raise ValueError(f"CSV contains rows wider than its header: {csv_path}")
    return list(reader.fieldnames), rows


def _evaluation_ids(root: Path, directory: str, filename: str) -> set[int]:
    parent = root / directory
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"missing regular directory: {parent}")
    result: set[int] = set()
    for child in parent.iterdir():
        if not child.is_dir() or child.is_symlink():
            raise ValueError(f"unexpected non-directory or symlink in {parent}: {child}")
        match = _EVAL_DIR_RE.fullmatch(child.name)
        if match is None:
            raise ValueError(f"unexpected evaluation directory name: {child}")
        eval_id = int(match.group(1))
        artifact = child / filename
        _require_regular_file(artifact)
        result.add(eval_id)
    return result


def _validate_source_type(row: dict[str, str], *, eval_id: int) -> None:
    eval_type = (row.get("eval_type") or "").strip()
    if eval_type not in _SUPPORTED_SOURCE_TYPES:
        raise ValueError(
            f"eval_id={eval_id} has unsupported source eval_type={eval_type!r}; "
            f"expected one of {sorted(_SUPPORTED_SOURCE_TYPES)}"
        )
    if eval_type == "OOM" and not (row.get("eval_error") or "").lstrip().startswith("CM:"):
        raise ValueError(
            f"eval_id={eval_id} OOM is not an authenticated source-CM failure"
        )


def convert_optimizer_export(
    source_project_dir: str | os.PathLike[str],
    output_project_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Convert one seed export to a standard, trace-aware project directory.

    The source is never changed.  ``output_project_dir`` must not already
    exist; conversion happens in a sibling temporary directory which is
    atomically renamed into place only after all integrity checks pass.
    """

    source = Path(source_project_dir).expanduser().resolve()
    output = Path(output_project_dir).expanduser().resolve()
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"source project is not a regular directory: {source}")
    if output.exists():
        raise FileExistsError(f"output project already exists: {output}")
    if output == source or output.is_relative_to(source):
        raise ValueError("output project must not be the source or live inside it")

    root_payloads: dict[str, bytes] = {}
    root_source_identities: dict[str, dict[str, Any]] = {}
    source_snapshots: dict[Path, dict[str, Any]] = {}
    for name in _ROOT_FILES + _OPTIONAL_ROOT_FILES:
        source_path = source / name
        if name in _OPTIONAL_ROOT_FILES and not source_path.exists():
            continue
        payload, identity = _capture_regular_file(source_path)
        root_payloads[name] = payload
        root_source_identities[name] = identity
        source_snapshots[source_path] = identity

    try:
        config = yaml.safe_load(
            root_payloads["rag_stack_config.yaml"].decode("utf-8")
        ) or {}
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid rag_stack_config.yaml in {source}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("rag_stack_config.yaml must contain an object")
    if ((config.get("global") or {}).get("rag_ir_mode") or "dynamic") != "dynamic":
        raise ValueError("optimizer export conversion requires global.rag_ir_mode=dynamic")

    metric_names = _configured_metric_names(config)
    combined_spec = (config.get("eval_backend_setting") or {}).get(
        "combined_quality"
    )
    fieldnames, rows = _csv_rows(
        source / "all_evaluations.csv", root_payloads["all_evaluations.csv"]
    )
    required_columns = {
        "eval_id",
        "eval_type",
        "eval_error",
        "quality__combined",
        *(f"quality__{name}" for name in metric_names),
    }
    missing_columns = required_columns - set(fieldnames)
    if missing_columns:
        raise ValueError(
            f"all_evaluations.csv is missing columns: {sorted(missing_columns)}"
        )
    exported_metric_columns = {
        name.removeprefix("quality__")
        for name in fieldnames
        if name.startswith("quality__") and name != "quality__combined"
    }
    if exported_metric_columns != set(metric_names):
        raise ValueError(
            "CSV/config quality metric mismatch: "
            f"csv={sorted(exported_metric_columns)}, config={sorted(metric_names)}"
        )

    rows_by_id: dict[int, dict[str, str]] = {}
    for position, row in enumerate(rows, start=2):
        try:
            eval_id = int(row.get("eval_id") or "")
        except ValueError as exc:
            raise ValueError(
                f"all_evaluations.csv:{position} has invalid eval_id"
            ) from exc
        if eval_id < 0 or eval_id > 9999:
            raise ValueError(f"eval_id must be in [0, 9999], got {eval_id}")
        if eval_id in rows_by_id:
            raise ValueError(f"duplicate eval_id={eval_id} in all_evaluations.csv")
        _validate_source_type(row, eval_id=eval_id)
        rows_by_id[eval_id] = row

    csv_ids = set(rows_by_id)
    pipeline_ids = _evaluation_ids(
        source, "eval_configs", "pipeline_config.json"
    )
    trace_ids = _evaluation_ids(source, "eval_traces", "execution_dag.json")
    if pipeline_ids != csv_ids:
        raise ValueError(
            "CSV/pipeline evaluation IDs differ: "
            f"missing={sorted(csv_ids - pipeline_ids)}, "
            f"extra={sorted(pipeline_ids - csv_ids)}"
        )
    if trace_ids != csv_ids:
        raise ValueError(
            "CSV/trace evaluation IDs differ: "
            f"missing={sorted(csv_ids - trace_ids)}, "
            f"extra={sorted(trace_ids - csv_ids)}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.convert-", dir=output.parent)
    )
    try:
        root_artifacts: dict[str, dict[str, Any]] = {}
        for name, payload in root_payloads.items():
            root_artifacts[name] = {
                "source": root_source_identities[name],
                "output": _write_bytes(temp_path / name, payload),
            }

        records: list[dict[str, Any]] = []
        query_count: int | None = None
        type_counts: dict[str, int] = {}
        for eval_id in sorted(rows_by_id):
            row = rows_by_id[eval_id]
            eval_type = row["eval_type"].strip()
            type_counts[eval_type] = type_counts.get(eval_type, 0) + 1
            source_config_dir = source / "eval_configs" / f"eval_{eval_id:04d}"
            source_trace_dir = source / "eval_traces" / f"eval_{eval_id:04d}"
            pipeline_source = source_config_dir / "pipeline_config.json"
            trace_source = source_trace_dir / "execution_dag.json"
            pipeline_bytes, pipeline_identity = _capture_regular_file(
                pipeline_source
            )
            source_snapshots[pipeline_source] = pipeline_identity
            _json_object_from_bytes(pipeline_source, pipeline_bytes)
            trace_bytes, trace_identity = _capture_regular_file(trace_source)
            source_snapshots[trace_source] = trace_identity
            source_trace = _json_object_from_bytes(trace_source, trace_bytes)
            try:
                trace, trace_projection = _project_export_trace(source_trace)
            except Exception as exc:
                raise ValueError(
                    f"eval_id={eval_id} is not a strict frozen quality trace: {exc}"
                ) from exc
            current_query_count = len(trace["queries"])
            if query_count is None:
                query_count = current_query_count
            elif current_query_count != query_count:
                raise ValueError(
                    f"eval_id={eval_id} trace cardinality {current_query_count} "
                    f"differs from {query_count}"
                )

            quality_metrics = {
                name: _finite_float(
                    row[f"quality__{name}"],
                    field=f"eval_id={eval_id} quality__{name}",
                )
                for name in metric_names
            }
            source_combined = _finite_float(
                row["quality__combined"],
                field=f"eval_id={eval_id} quality__combined",
            )
            recomputed_combined = _aggregate_quality_metrics(
                quality_metrics, combined_spec
            )
            if f"{source_combined:.6f}" != f"{recomputed_combined:.6f}":
                raise ValueError(
                    f"eval_id={eval_id} combined quality mismatch: "
                    f"CSV={source_combined:.6f}, recomputed={recomputed_combined:.6f}"
                )

            destination = temp_path / "evaluations" / f"eval_{eval_id:04d}"
            pipeline_output = destination / "pipeline_config.json"
            trace_output = destination / "execution_dag.json"
            quality_output = destination / "quality_metrics.json"
            output_artifacts = {
                "pipeline_config.json": _write_bytes(
                    pipeline_output, pipeline_bytes
                ),
                "execution_dag.json": (
                    _write_bytes(trace_output, trace_bytes)
                    if trace_projection["kind"] == "identity"
                    else _write_json(trace_output, trace)
                ),
                "quality_metrics.json": _write_json(quality_output, quality_metrics),
            }
            source_artifacts = {
                "pipeline_config.json": pipeline_identity,
                "execution_dag.json": trace_identity,
                "all_evaluations.csv": root_artifacts[
                    "all_evaluations.csv"
                ]["source"],
            }
            for optional_config in (
                "system_config.json",
                "system_config_resolved.json",
            ):
                optional_source = source_config_dir / optional_config
                if not optional_source.exists():
                    continue
                optional_bytes, optional_identity = _capture_regular_file(
                    optional_source
                )
                source_snapshots[optional_source] = optional_identity
                _json_object_from_bytes(optional_source, optional_bytes)
                source_artifacts[optional_config] = optional_identity
                output_artifacts[optional_config] = _write_bytes(
                    destination / optional_config, optional_bytes
                )

            records.append(
                {
                    "eval_id": eval_id,
                    "source_eval_type": eval_type,
                    "source_eval_error": row.get("eval_error") or "",
                    "trace_projection": trace_projection,
                    "quality_reconstruction": {
                        "kind": "all_evaluations_csv_six_decimal",
                        "metric_columns": [f"quality__{name}" for name in metric_names],
                        "combined_column": "quality__combined",
                        "combined_csv": source_combined,
                        "combined_recomputed": recomputed_combined,
                    },
                    "source_artifacts": source_artifacts,
                    "output_artifacts": output_artifacts,
                }
            )

        manifest = {
            "format": CONVERSION_FORMAT,
            "format_version": CONVERSION_FORMAT_VERSION,
            "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_project_dir": str(source),
            "output_project_dir": str(output),
            "source_files_unchanged": True,
            "quality_origin": "reconstructed_from_all_evaluations_csv",
            "quality_precision": "six_decimal_export",
            "trace_query_count": query_count,
            "configured_quality_metrics": metric_names,
            "source_type_counts": type_counts,
            "input_row_count": len(rows_by_id),
            "output_evaluation_count": len(records),
            "root_artifacts": root_artifacts,
            "records": records,
        }
        _validate_conversion_manifest(manifest)
        _write_json(temp_path / "conversion_manifest.json", manifest)

        # Final readback verifies the manifest and every declared output before
        # the atomic publish rename makes the project visible to another process.
        readback = _validate_conversion_manifest(
            _load_json_object(temp_path / "conversion_manifest.json")
        )
        if readback.get("output_evaluation_count") != len(rows_by_id):
            raise RuntimeError("conversion manifest readback count mismatch")
        for record in readback.get("records") or []:
            eval_dir = temp_path / "evaluations" / f"eval_{record['eval_id']:04d}"
            for name, identity in record["output_artifacts"].items():
                if _file_identity(eval_dir / name) != identity:
                    raise RuntimeError(
                        f"conversion output integrity mismatch: {eval_dir / name}"
                    )

        # Parse, output, and manifest identities all derive from the same byte
        # snapshots. Recheck every source just before publication so a source
        # mutation cannot be silently attributed to a different snapshot.
        for source_path, expected_identity in source_snapshots.items():
            try:
                _, current_identity = _capture_regular_file(source_path)
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f"source artifact changed during conversion: {source_path}"
                ) from exc
            if current_identity != expected_identity:
                raise RuntimeError(
                    f"source artifact changed during conversion: {source_path}"
                )

        os.replace(temp_path, output)
        return manifest
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag_stack convert-optimizer-export",
        description="Convert one optimizer export seed into a standard project",
    )
    parser.add_argument(
        "--source-project",
        required=True,
        help="Exported seed directory containing eval_configs/eval_traces",
    )
    parser.add_argument(
        "--output-project",
        required=True,
        help="New standard project directory (must not already exist)",
    )
    args = parser.parse_args(argv)
    manifest = convert_optimizer_export(args.source_project, args.output_project)
    print(
        f"Converted {manifest['output_evaluation_count']} evaluations -> "
        f"{manifest['output_project_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
