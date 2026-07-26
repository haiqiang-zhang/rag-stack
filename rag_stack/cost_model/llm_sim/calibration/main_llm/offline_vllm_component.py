"""Isolated fixed-cohort vLLM component measurement for CM.

This module is a child-process entry point.  It deliberately uses vLLM's
offline ``LLM`` API with the in-process EngineCore so every request in a
cohort is enqueued before the first scheduler step.  That is the only reliable
way to measure an exact resident component cohort: an OpenAI frontend (and
the default ZMQ-backed offline EngineCore) may start executing after receiving
the first request and split one intended cohort across scheduler cycles.

Only the audited component worker's CUDA-event service enters the result.
Tokenisation, enqueue time, scheduler time and result materialisation remain
diagnostics and cannot be absorbed by ceff/meff.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any, Mapping, Sequence

from .generator_request_contract import (
    COMPONENT_SERVICE_BOUNDARY,
    render_generator_chat_prompt_token_ids,
    validate_generator_request_contracts,
)

SPEC_SCHEMA = "rag_stack.offline_vllm_component_spec"
RESULT_SCHEMA = "rag_stack.offline_vllm_component_result"
WARMUP_COHORTS = 2
MEASUREMENT_COHORTS = 3
TOTAL_COHORTS = WARMUP_COHORTS + MEASUREMENT_COHORTS
COMPONENT_WORKER = (
    "rag_stack_evaluator.vllm_instrumentation.llm_sim.calibration.component_telemetry."
    "VllmComponentTelemetryWorker"
)
_COMPONENT_TELEMETRY_ENV = "RAG_STACK_LLM_COMPONENT_TELEMETRY_PATH"
_CALIBRATION_RUN_ID_ENV = "RAG_STACK_VLLM_CALIBRATION_RUN_ID"


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields differ from the exact schema: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _finite_fraction(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number in (0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result <= 1.0:
        raise ValueError(f"{label} must be a finite number in (0, 1]")
    return result


def validate_spec(value: Any) -> dict[str, Any]:
    """Validate the complete parent/child contract without importing vLLM."""

    if not isinstance(value, Mapping):
        raise ValueError("offline component spec must be an object")
    _exact_fields(
        value,
        {
            "schema",
            "startup",
            "shapes",
            "request_contracts",
            "warmup_cohorts",
            "measurement_cohorts",
        },
        "offline component spec",
    )
    if value["schema"] != SPEC_SCHEMA:
        raise ValueError("unexpected offline component spec kind")
    startup = value["startup"]
    if not isinstance(startup, Mapping):
        raise ValueError("offline component startup must be an object")
    startup_fields = {
        "model",
        "dtype",
        "gpu_memory_utilization",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "kv_cache_dtype",
        "enable_prefix_caching",
        "tensor_parallel_size",
        "pipeline_parallel_size",
    }
    _exact_fields(startup, startup_fields, "offline component startup")
    model = startup["model"]
    dtype = startup["dtype"]
    kv_cache_dtype = startup["kv_cache_dtype"]
    if not isinstance(model, str) or not model.strip():
        raise ValueError("offline component model must be non-empty")
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError("offline component dtype must be non-empty")
    if not isinstance(kv_cache_dtype, str) or not kv_cache_dtype.strip():
        raise ValueError("offline component kv_cache_dtype must be non-empty")
    if startup["enable_prefix_caching"] is not False:
        raise ValueError("offline component calibration requires prefix caching off")
    tp = _positive_int(startup["tensor_parallel_size"], "tensor_parallel_size")
    pp = _positive_int(startup["pipeline_parallel_size"], "pipeline_parallel_size")
    if (tp, pp) != (1, 1):
        raise ValueError("offline component calibration requires TP1/PP1")
    normalized_startup = {
        "model": model,
        "dtype": dtype,
        "gpu_memory_utilization": _finite_fraction(
            startup["gpu_memory_utilization"], "gpu_memory_utilization"
        ),
        "max_model_len": _positive_int(startup["max_model_len"], "max_model_len"),
        "max_num_seqs": _positive_int(startup["max_num_seqs"], "max_num_seqs"),
        "max_num_batched_tokens": _positive_int(
            startup["max_num_batched_tokens"], "max_num_batched_tokens"
        ),
        "kv_cache_dtype": kv_cache_dtype,
        "enable_prefix_caching": False,
        "tensor_parallel_size": tp,
        "pipeline_parallel_size": pp,
    }
    raw_shapes = value["shapes"]
    if not isinstance(raw_shapes, Sequence) or isinstance(raw_shapes, (str, bytes)):
        raise ValueError("offline component shapes must be a non-empty sequence")
    shapes: list[tuple[int, int, int]] = []
    for index, raw_shape in enumerate(raw_shapes):
        if (
            not isinstance(raw_shape, Sequence)
            or isinstance(raw_shape, (str, bytes))
            or len(raw_shape) != 3
        ):
            raise ValueError(f"offline component shape {index} must be [batch,in,out]")
        shape = tuple(
            _positive_int(item, f"shapes[{index}][{axis}]")
            for axis, item in enumerate(raw_shape)
        )
        if shape[0] > normalized_startup["max_num_seqs"]:
            raise ValueError(
                f"shape B{shape[0]} exceeds max_num_seqs="
                f"{normalized_startup['max_num_seqs']}"
            )
        if shape[0] * shape[1] > normalized_startup["max_num_batched_tokens"]:
            raise ValueError(
                f"shape B{shape[0]}x{shape[1]} exceeds max_num_batched_tokens="
                f"{normalized_startup['max_num_batched_tokens']}"
            )
        if shape[1] + shape[2] > normalized_startup["max_model_len"]:
            raise ValueError(
                f"shape {shape!r} exceeds max_model_len="
                f"{normalized_startup['max_model_len']}"
            )
        shapes.append(shape)
    if not shapes or len(set(shapes)) != len(shapes):
        raise ValueError("offline component shapes must be non-empty and unique")
    request_contracts = validate_generator_request_contracts(
        value["request_contracts"],
        expected_prompt_tokens=[shape[1] for shape in shapes],
    )
    warmup_cohorts = _positive_int(
        value["warmup_cohorts"], "warmup_cohorts",
    )
    measurement_cohorts = _positive_int(
        value["measurement_cohorts"], "measurement_cohorts",
    )
    if (
        warmup_cohorts != WARMUP_COHORTS
        or measurement_cohorts != MEASUREMENT_COHORTS
    ):
        raise ValueError(
            "offline component calibration requires exactly "
            f"{WARMUP_COHORTS} warmup + {MEASUREMENT_COHORTS} measurement "
            "cohorts per shape"
        )
    return {
        "schema": SPEC_SCHEMA,
        "startup": normalized_startup,
        "shapes": shapes,
        "request_contracts": request_contracts,
        "warmup_cohorts": warmup_cohorts,
        "measurement_cohorts": measurement_cohorts,
    }


def _read_segment(path: Path, start_offset: int) -> bytes:
    if not path.exists():
        raise RuntimeError("component worker produced no telemetry file")
    with path.open("rb") as handle:
        handle.seek(start_offset)
        payload = handle.read()
    if not payload or not payload.endswith(b"\n"):
        raise RuntimeError("component worker produced empty/incomplete telemetry")
    return payload


def _flush_component_telemetry(
    llm: object,
    *,
    expected_work_sequences: Sequence[int],
) -> None:
    """Publish one exact completed cohort through the worker control plane."""

    expected = list(expected_work_sequences)
    if (
        not expected
        or any(
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
            for sequence in expected_work_sequences
        )
        or any(
            current != previous + 1
            for previous, current in zip(expected, expected[1:])
        )
    ):
        raise RuntimeError("component telemetry expected sequence is invalid")
    collective_rpc = getattr(llm, "collective_rpc", None)
    if not callable(collective_rpc):
        raise RuntimeError("offline vLLM has no component telemetry flush RPC")
    replies = collective_rpc("flush_component_telemetry")
    if (
        not isinstance(replies, Sequence)
        or isinstance(replies, (str, bytes))
        or len(replies) != 1
        or not isinstance(replies[0], Mapping)
    ):
        raise RuntimeError("component telemetry flush returned an invalid reply")
    reply = replies[0]
    if set(reply) != {"flushed_count", "work_sequences", "pending_count"}:
        raise RuntimeError("component telemetry flush reply fields are invalid")
    flushed_count = reply["flushed_count"]
    pending_count = reply["pending_count"]
    work_sequences = reply["work_sequences"]
    if (
        isinstance(flushed_count, bool)
        or not isinstance(flushed_count, int)
        or flushed_count != len(expected)
        or isinstance(pending_count, bool)
        or not isinstance(pending_count, int)
        or pending_count != 0
        or not isinstance(work_sequences, list)
        or any(
            isinstance(sequence, bool) or not isinstance(sequence, int)
            for sequence in work_sequences
        )
        or work_sequences != expected
    ):
        raise RuntimeError(
            "component telemetry flush does not match the exact cohort cycles"
        )


def _shutdown_llm(llm: object) -> None:
    """Best-effort vLLM 0.18.1 teardown before the process exits."""

    renderer = getattr(llm, "renderer", None)
    shutdown_renderer = getattr(renderer, "shutdown", None)
    if callable(shutdown_renderer):
        try:
            shutdown_renderer()
        except Exception:  # noqa: BLE001 - process exit is the hard boundary
            pass
    engine = getattr(llm, "llm_engine", None)
    core = getattr(engine, "engine_core", None)
    shutdown_core = getattr(core, "shutdown", None)
    if callable(shutdown_core):
        try:
            shutdown_core(timeout=30.0)
        except TypeError:
            shutdown_core()


def measure(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Run one fixed-cohort component campaign inside this isolated process."""

    normalized = validate_spec(spec)
    if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
        raise RuntimeError(
            "exact offline cohorts require VLLM_ENABLE_V1_MULTIPROCESSING=0 "
            "before vLLM import"
        )
    telemetry_value = os.environ.get(_COMPONENT_TELEMETRY_ENV)
    run_id = os.environ.get(_CALIBRATION_RUN_ID_ENV)
    if not telemetry_value or not run_id:
        raise RuntimeError("offline component telemetry path/run id is missing")
    telemetry_path = Path(telemetry_value).expanduser().resolve()
    if telemetry_path.exists():
        raise FileExistsError(
            f"refusing to append offline component telemetry: {telemetry_path}"
        )
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy imports are essential: CUDA visibility and the in-process EngineCore
    # switch must be frozen by the parent before importing vLLM.
    from vllm import LLM, SamplingParams

    from .vllm_probe import _exact_component_cohort_service_from_segment

    startup = normalized["startup"]
    llm = LLM(
        model=startup["model"],
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        dtype=startup["dtype"],
        gpu_memory_utilization=startup["gpu_memory_utilization"],
        max_model_len=startup["max_model_len"],
        max_num_seqs=startup["max_num_seqs"],
        max_num_batched_tokens=startup["max_num_batched_tokens"],
        enable_prefix_caching=False,
        kv_cache_dtype=startup["kv_cache_dtype"],
        worker_cls=COMPONENT_WORKER,
        disable_log_stats=True,
    )
    try:
        tokenizer = llm.get_tokenizer()
        points: list[dict[str, Any]] = []
        next_work_sequence = 0
        for batch, input_tokens, output_tokens in normalized["shapes"]:
            request_contract = normalized["request_contracts"][str(input_tokens)]
            rendered_prompt_token_ids = render_generator_chat_prompt_token_ids(
                tokenizer,
                request_contract,
            )
            qps_values: list[float] = []
            wall_values: list[float] = []
            component_values: list[float] = []
            for _ in range(TOTAL_COHORTS):
                start_offset = (
                    telemetry_path.stat().st_size if telemetry_path.exists() else 0
                )
                prompts = [list(rendered_prompt_token_ids) for _ in range(batch)]
                sampling = SamplingParams(
                    max_tokens=output_tokens,
                    temperature=request_contract["temperature"],
                    seed=request_contract["seed"],
                    ignore_eos=True,
                    logprobs=request_contract["top_logprobs"],
                )
                started = time.perf_counter()
                outputs = llm.generate(prompts, sampling, use_tqdm=False)
                wall_s = time.perf_counter() - started
                expected_work_sequences = range(
                    next_work_sequence,
                    next_work_sequence + output_tokens,
                )
                _flush_component_telemetry(
                    llm,
                    expected_work_sequences=expected_work_sequences,
                )
                next_work_sequence += output_tokens
                if len(outputs) != batch:
                    raise RuntimeError(
                        f"offline component returned {len(outputs)} requests for B{batch}"
                    )
                prompt_lengths = {
                    len(output.prompt_token_ids or ()) for output in outputs
                }
                generated_lengths = {
                    len(output.outputs[0].token_ids) for output in outputs
                }
                if prompt_lengths != {input_tokens}:
                    raise RuntimeError(
                        "offline component prompt shape mismatch: "
                        f"expected={input_tokens}, observed={sorted(prompt_lengths)}"
                    )
                if generated_lengths != {output_tokens}:
                    raise RuntimeError(
                        "offline component output shape mismatch: "
                        f"expected={output_tokens}, observed={sorted(generated_lengths)}"
                    )
                segment = _read_segment(telemetry_path, start_offset)
                component_s = _exact_component_cohort_service_from_segment(
                    segment,
                    target_batch=batch,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
                wall_values.append(float(wall_s))
                qps_values.append(float(batch) / float(wall_s))
                component_values.append(float(component_s))
            selected_qps = qps_values[WARMUP_COHORTS:]
            selected_wall = wall_values[WARMUP_COHORTS:]
            warmup_component = component_values[:WARMUP_COHORTS]
            measurement_component = component_values[WARMUP_COHORTS:]
            if (
                len(warmup_component) != WARMUP_COHORTS
                or len(measurement_component) != MEASUREMENT_COHORTS
            ):
                raise RuntimeError(
                    "offline component child lost its fixed cohort contract"
                )
            point = {
                "batch": batch,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "offline_cohort_qps_diagnostic": float(
                    statistics.median(selected_qps)
                ),
                "offline_cohort_wall_s_diagnostic": float(
                    statistics.median(selected_wall)
                ),
                "warmup_component_services_s": [
                    float(value) for value in warmup_component
                ],
                "measurement_component_services_s": [
                    float(value) for value in measurement_component
                ],
                "component_service_s": float(
                    statistics.median(measurement_component)
                ),
            }
            points.append(point)
            print(
                f"  offline component B{batch:>4} in{input_tokens:>5} "
                f"out{output_tokens:>5} service "
                f"{point['component_service_s'] * 1000:>9.1f} ms "
                f"({WARMUP_COHORTS} warmup + "
                f"{MEASUREMENT_COHORTS} measurement)",
                flush=True,
            )
        return {
            "schema": RESULT_SCHEMA,
            "run_id": run_id,
            "acquisition": "offline_inprocess_engine_full_cohort_enqueue",
            "component_boundary": COMPONENT_SERVICE_BOUNDARY,
            "request_contracts": normalized["request_contracts"],
            "warmup_cohorts": WARMUP_COHORTS,
            "measurement_cohorts": MEASUREMENT_COHORTS,
            "total_cohorts": TOTAL_COHORTS,
            "points": points,
        }
    finally:
        _shutdown_llm(llm)


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="offline-vllm-component")
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--result-path", required=True)
    args = parser.parse_args(argv)
    result_path = Path(args.result_path).expanduser().resolve()
    if result_path.exists():
        raise FileExistsError(f"refusing to replace offline result: {result_path}")
    try:
        spec = json.loads(args.spec_json)
    except json.JSONDecodeError as exc:
        raise ValueError("--spec-json is not valid JSON") from exc
    _atomic_write(result_path, measure(spec))


if __name__ == "__main__":
    main()
