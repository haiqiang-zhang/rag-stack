"""Run one saved resolved deployment through measured evaluation.

The public command intentionally has only two required inputs::

    python -m rag_stack eval-one --performance-only \\
        pipeline_config.json system_config.json

Everything that defines measured-performance truth (performance-only scoring,
full QA population, concrete devices, trace persistence, and admissibility
gating) is owned here. Performance-only is the default and only supported
evaluation mode; the explicit flag makes replay intent auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import time
import traceback
from typing import Any, Sequence

from rag_stack.logging_utils import (
    attach_project_log,
    detach_project_log,
    setup_logging,
)
from rag_stack.system_transfer_backend import project_to_vllm_chat


logger = logging.getLogger("RAG-Stack")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_stack eval-one",
        description=(
            "Measure one fully resolved RAG pipeline and concrete system "
            "deployment on the local hardware."
        ),
    )
    parser.add_argument(
        "pipeline_config",
        help="Path to the resolved pipeline_config.json",
    )
    parser.add_argument(
        "system_config",
        help="Path to the concrete system_config.json",
    )
    parser.add_argument(
        "--performance-only",
        "--perf-only",
        action="store_true",
        default=True,
        help=(
            "Measure performance without recomputing quality or invoking an "
            "LLM judge. This is the default and only eval-one mode."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Result/cache directory. By default, a deterministic directory is "
            "created under tmp_outputs/rag_stack_projects/."
        ),
    )
    return parser


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return value, payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _available_gpus(system_config: dict[str, Any]) -> list[str]:
    """Resolve the concrete device inventory from a saved system artifact.

    Optimizer evaluation artifacts store the inventory under
    ``hardware.available_gpus``; exported replay bundles may additionally put
    the same inventory under ``layout.available_devices``. Both are canonical
    concrete-system representations, but conflicting values are ambiguous and
    must fail closed.
    """
    layout = system_config.get("layout")
    if not isinstance(layout, dict):
        raise ValueError("system_config.layout must be an object")
    hardware = system_config.get("hardware")
    if hardware is not None and not isinstance(hardware, dict):
        raise ValueError("system_config.hardware must be an object")

    layout_devices = layout.get("available_devices")
    hardware_devices = (hardware or {}).get("available_gpus")
    inventories = [
        (name, value)
        for name, value in (
            ("layout.available_devices", layout_devices),
            ("hardware.available_gpus", hardware_devices),
        )
        if value is not None
    ]
    for name, value in inventories:
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(device, str) or not device for device in value)
        ):
            raise ValueError(
                f"system_config.{name} must be a non-empty list of device names"
            )

    if not inventories:
        raise ValueError(
            "system_config must define a non-empty concrete device list at "
            "layout.available_devices or hardware.available_gpus"
        )
    if len(inventories) == 2 and inventories[0][1] != inventories[1][1]:
        raise ValueError(
            "system_config concrete device inventories disagree: "
            "layout.available_devices != hardware.available_gpus"
        )
    return list(inventories[0][1])


def _default_output_dir(
    pipeline_path: Path,
    pipeline_payload: bytes,
    system_payload: bytes,
) -> Path:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", pipeline_path.parent.name)
    label = label.strip("._-") or "config"
    identity = hashlib.sha256(
        pipeline_payload + b"\0" + system_payload
    ).hexdigest()[:12]
    return (
        _repository_root()
        / "tmp_outputs"
        / "rag_stack_projects"
        / f"eval_one_{label}_{identity}"
    )


def _resolve_dataset_path(value: Any, *, pipeline_path: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("pipeline_config.dataset.qa/corpus must be a path string")
    expanded = Path(os.path.expandvars(value)).expanduser()
    if expanded.is_absolute():
        candidates = [expanded]
    else:
        candidates = [
            Path.cwd() / expanded,
            _repository_root() / expanded,
            pipeline_path.parent / expanded,
        ]
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            return resolved
    attempted = ", ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        f"dataset path {value!r} was not found; attempted: {attempted}"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_runtime():
    """Load GPU/runtime dependencies only after CLI and input validation."""
    from rag_stack_evaluator.static_rag_evaluator import StaticRAGEvaluatorQualityOnly
    from rag_stack.dataset_manager import DatasetManager
    from rag_stack_evaluator.static_rag_evaluator.measured.provider import MeasuredProvider
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import TrialInvalid

    return DatasetManager, StaticRAGEvaluatorQualityOnly, MeasuredProvider, TrialInvalid


def _publication_rejections(
    performance_score: Any,
    performance: dict[str, Any],
    performance_trace: Any,
) -> list[str]:
    reasons: list[str] = []
    verdict = performance.get("measured_gt_admissibility")
    try:
        from rag_stack_evaluator.static_rag_evaluator.measured.artifact_contract import (
            validate_measured_gt_admissibility,
        )
        verdict = validate_measured_gt_admissibility(verdict)
    except ValueError as exc:
        reasons.append(f"invalid_measured_gt_admissibility: {exc}")
    else:
        reported = verdict.get("reasons")
        if verdict["admissible"] is not True and reported:
            reasons.extend(str(reason) for reason in reported)
        elif verdict["admissible"] is not True:
            reasons.append("measured_gt_inadmissible")

    if performance.get("population_saturated") is not True:
        reasons.append("population_not_saturated")

    try:
        qps = float(performance["qps"])
    except (KeyError, TypeError, ValueError):
        reasons.append("missing_or_invalid_qps")
    else:
        if not math.isfinite(qps) or qps <= 0.0:
            reasons.append("non_positive_or_non_finite_qps")

    try:
        score = float(performance_score)
    except (TypeError, ValueError):
        reasons.append("invalid_performance_score")
    else:
        if not math.isfinite(score):
            reasons.append("non_finite_performance_score")

    if not isinstance(performance_trace, dict) or not performance_trace:
        reasons.append("missing_performance_execution_trace")
    return list(dict.fromkeys(reasons))


def run_eval_one(
    pipeline_config_arg: str,
    system_config_arg: str,
    *,
    output_dir_arg: str | None = None,
) -> int:
    pipeline_path = Path(pipeline_config_arg).expanduser().resolve()
    system_path = Path(system_config_arg).expanduser().resolve()
    pipeline_config, pipeline_payload = _read_json_object(
        pipeline_path, label="pipeline config"
    )
    system_config, system_payload = _read_json_object(
        system_path, label="system config"
    )

    output_dir = (
        Path(output_dir_arg).expanduser().resolve()
        if output_dir_arg
        else _default_output_dir(
            pipeline_path, pipeline_payload, system_payload
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging()
    log_state = attach_project_log(str(output_dir), "eval_one")

    pipeline_sha256 = _sha256(pipeline_payload)
    system_sha256 = _sha256(system_payload)
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pipeline_path, inputs_dir / "pipeline_config.json")
    shutil.copy2(system_path, inputs_dir / "system_config.json")

    run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
    invocation_dir = output_dir / "runs" / run_id
    invocation_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "result.json"
    trial_invalid_type: Any = ()
    base_record: dict[str, Any] = {
        "pipeline_config": str(pipeline_path),
        "pipeline_config_sha256": pipeline_sha256,
        "system_config": str(system_path),
        "system_config_sha256": system_sha256,
        "output_dir": str(output_dir),
        "run_dir": str(invocation_dir),
        "performance_only": True,
        "quality_mode": "preserve_existing",
        "performance_mode": "measured",
    }

    try:
        # Saved quality pipelines may still point at an external vllm_api
        # endpoint.  Measured replay owns the deployment, so project every
        # such module onto local vLLM chat without modifying the source JSON.
        pipeline_config, _ = project_to_vllm_chat(pipeline_config)

        dataset = pipeline_config.get("dataset")
        if not isinstance(dataset, dict):
            raise ValueError("pipeline_config.dataset must be an object")
        qa_path = _resolve_dataset_path(
            dataset.get("qa"), pipeline_path=pipeline_path
        )
        corpus_path = _resolve_dataset_path(
            dataset.get("corpus"), pipeline_path=pipeline_path
        )

        available_gpus = _available_gpus(system_config)

        # The source JSON remains untouched. A one-point measured replay owns
        # no quality axis and must never spend judge/API calls.
        pipeline_config["eval_backend_setting"] = {"performance_only": True}
        selection = (
            (
                (pipeline_config.get("system") or {}).get(
                    "performance_objectives"
                )
                or {}
            ).get("selection")
            or "max_throughput"
        )
        os.environ["PROJECT_DIR"] = str(output_dir)

        (
            DatasetManager,
            StaticRAGEvaluatorQualityOnly,
            MeasuredProvider,
            TrialInvalid,
        ) = _load_runtime()
        trial_invalid_type = TrialInvalid

        dataset_manager = DatasetManager(
            project_dir=str(output_dir),
            config=pipeline_config,
            qa_data_path=str(qa_path),
            corpus_data_path=str(corpus_path),
        )
        evaluator = StaticRAGEvaluatorQualityOnly(
            dataset_manager=dataset_manager,
            project_dir=str(output_dir),
        )
        measured_run_dir = invocation_dir / "measured_run"
        with MeasuredProvider(
            evaluator,
            available_gpus,
            n_queries=None,
            selection=selection,
        ) as provider:
            result = provider.evaluate(
                pipeline_config,
                system_config,
                run_dir=str(measured_run_dir),
                require_admissible=False,
            )

        performance = dict(result.raw_performance)
        performance_trace = result.performance_execution_trace
        trace_path = invocation_dir / "performance_execution_trace.json"
        _write_json(trace_path, performance_trace)
        rejections = _publication_rejections(
            result.performance_score,
            performance,
            performance_trace,
        )
        status = "ok" if not rejections else "inadmissible"
        record = {
            **base_record,
            "status": status,
            "performance_score": float(result.performance_score),
            "qps": performance.get("qps"),
            "performance": performance,
            "performance_execution_trace": str(trace_path),
        }
        if rejections:
            record["error"] = ", ".join(rejections)
        _write_json(invocation_dir / "result.json", record)
        _write_json(result_path, record)

        if rejections:
            logger.error(
                "Measured window is not publishable: %s", ", ".join(rejections)
            )
            logger.error("Diagnostic result: %s", result_path)
            return 2

        logger.info("Measured QPS: %.6f", float(performance["qps"]))
        logger.info("Result: %s", result_path)
        logger.info("Performance trace: %s", trace_path)
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:  # noqa: BLE001 - persist one-point failures
        is_invalid = bool(trial_invalid_type) and isinstance(
            exc, trial_invalid_type
        )
        record = {
            **base_record,
            "status": "invalid" if is_invalid else "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        _write_json(invocation_dir / "result.json", record)
        _write_json(result_path, record)
        logger.error("eval-one failed: %s", record["error"])
        logger.error("Failure record: %s", result_path)
        return 2 if is_invalid else 1
    finally:
        detach_project_log(log_state)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_eval_one(
            args.pipeline_config,
            args.system_config,
            output_dir_arg=args.output_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(1, f"rag_stack eval-one: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
