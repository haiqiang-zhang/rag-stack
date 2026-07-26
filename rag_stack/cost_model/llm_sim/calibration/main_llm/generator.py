"""Generator component calibrator — pure GPU ceff/meff from vLLM telemetry.

The component boundary is intentionally narrower than serving.  An isolated
offline vLLM EngineCore enqueues each complete cohort before its first step,
while the audited worker times only transformer-forward + LM-head CUDA work.
This prevents OpenAI arrival jitter, dynamic waiting, scheduler/IPC work and
result materialisation from being absorbed by ceff/meff.  Those costs belong
to the separate ``hardware × PerformanceStage × batch`` curves.
It measures real prefill+decode latency on a MINIMAL identifiable point set (default: 4
fit + 2 held-out validation points spanning prefill-dominated / deep-batch-decode / mixed
regimes, with B128 fitted and B256 strictly held out; densify via
--batch-sizes/--shapes), then fits
ceff/meff with ``fit_and_save`` → an ``agnostic__<model>`` record in
``calibration_profiles/llm_sim/<hw>.json``.  The held-out points never enter the fit.

The component reference deployment is always TP1/PP1 with ``max_num_seqs`` equal to the
largest requested resident cohort. TP/PP are
deployment physics layered elsewhere; measuring every deployment here would make calibration
unbounded and contaminate the shared component coefficients. A model that cannot fit the
single-GPU reference fails/skips instead of silently switching topology.

    python -m rag_stack.cost_model.llm_sim.calibration.main_llm.generator \
        --config <cfg>                                                          # whole family
    python -m rag_stack.cost_model.llm_sim.calibration.main_llm.generator \
        --model Qwen/Qwen2.5-7B-Instruct --config <cfg>                         # one model
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import threading
import time
import uuid

# DEFAULT: minimal identifiable point set.  The component fit has exactly two
# physical coefficients (ceff, meff); serving overhead is calibrated elsewhere
# on hardware x stage x batch and is fixed to zero here.  Four fit points plus
# two strictly held-out points keep the campaign bounded while making
# publication at the deployed B256 decode cap verifiable:
#   * prefill-dominated anchor  (moderate batch, long context, short output)
#   * B128 deep-batch DECODE anchor (short context, long generation — the ONE
#     expensive point; without a decode-dominated regime every fitted point
#     is <=30% decode and the deep-batch decode step is under-predicted
#     ~2.4x: r25, eval_0009's b128 step 10.6 ms real vs 4.4 ms at old fit)
#   * a mixed third fit point, so neither regime rests on one noisy sample
#   * a B128 control at the exact B256 held-out shape
#   * one mixed out-of-sample validation point and one same-shape B256
#     extrapolation check that proves the maximum deployed decode batch
# Densify via --batch-sizes/--shapes (their cross product replaces POINTS)
# when diagnosing fit shape, or give explicit points via --points.
POINTS = [
    (32, 512, 128),    # prefill-dominated anchor
    (128, 64, 512),    # deep-batch decode anchor (the expensive point)
    (4, 64, 256),      # mixed third fit point
    (128, 64, 128),    # same-shape control for the B256 batch holdout
]
VALIDATION_POINTS = [
    (16, 128, 128),    # held out: mixed, mid concurrency
    # The shorter output keeps the 14B TP1 reference resident at B256.  Its
    # exact B128 fit control above isolates batch extrapolation rather than
    # allowing token shape to hide it.
    (256, 64, 128),
]
# Density-mode grid, used only when --batch-sizes / --shapes are given:
# client concurrency ladder (capped server-side at max_num_seqs; r12) ×
# (seq_tokens, out_tokens) shape pairs including decode-dominated ones (r25).
# Keep B256 out of the optional fit grid because it is the publication
# holdout.  Include its short-output control shape so an explicit density-mode
# campaign can retain the same B128/B256 paired validation contract.
BATCH = [1, 4, 16, 64, 128]
SHAPES = [(128, 128), (512, 128), (64, 128), (64, 256), (64, 512)]
# The component contract is fixed: two cold/clock warmups are retained in raw
# evidence but excluded, then exactly three complete cohorts are median-pooled.
# This must not be weakened from the CLI.
REPEATS = 5
CALIBRATION_MAX_MODEL_LEN = 4096
CALIBRATION_MIN_NUM_BATCHED_TOKENS = 65_536
COMPONENT_COVERAGE_SCHEMA = (
    "rag_stack.generator_component_resident_batch_coverage"
)
REQUIRED_RESIDENT_BATCHES = (128, 256)
_SAVE_LOCK = threading.Lock()       # serialize fit_and_save (concurrent profile writes race)


class GeneratorComponentCoverageError(RuntimeError):
    """The TP1/PP1 reference cannot prove required resident-batch coverage."""


def _resident_batch_coverage_contract(
    fit_points,
    held_out_points,
    *,
    reference_max_num_seqs: int,
) -> dict:
    """Build the exact batch-only coverage proof before any GPU is started."""

    def _normalize_points(
        points,
        *,
        label: str,
    ) -> list[tuple[int, int, int]]:
        normalized: list[tuple[int, int, int]] = []
        for point in points:
            if not isinstance(point, (tuple, list)) or len(point) != 3:
                raise GeneratorComponentCoverageError(
                    f"{label} points must be batch:seq:out triples"
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in point
            ):
                raise GeneratorComponentCoverageError(
                    f"{label} batch:seq:out values must be positive integers"
                )
            normalized.append(tuple(int(value) for value in point))
        return normalized

    normalized_fit = _normalize_points(fit_points, label="fit")
    normalized_held_out = _normalize_points(
        held_out_points,
        label="held-out",
    )
    fit_batches = sorted({point[0] for point in normalized_fit})
    held_out_batches = sorted({point[0] for point in normalized_held_out})
    if not fit_batches or not held_out_batches:
        raise GeneratorComponentCoverageError(
            "component coverage requires non-empty fit and held-out batches"
        )
    overlap = set(fit_batches).intersection(held_out_batches)
    if overlap:
        raise GeneratorComponentCoverageError(
            "component fit/held-out resident-batch roles are ambiguous: "
            f"overlap={sorted(overlap)}"
        )
    missing = set(REQUIRED_RESIDENT_BATCHES).difference(
        (*fit_batches, *held_out_batches)
    )
    if missing:
        raise GeneratorComponentCoverageError(
            "component calibration does not cover required resident batches: "
            f"missing={sorted(missing)}"
        )
    if 128 not in fit_batches or 256 not in held_out_batches:
        raise GeneratorComponentCoverageError(
            "component coverage requires B128 as a fit anchor and B256 as "
            "a held-out batch validation point"
        )
    b128_shapes = {
        (seq_tokens, output_tokens)
        for batch, seq_tokens, output_tokens in normalized_fit
        if batch == 128
    }
    b256_shapes = {
        (seq_tokens, output_tokens)
        for batch, seq_tokens, output_tokens in normalized_held_out
        if batch == 256
    }
    if not b128_shapes.intersection(b256_shapes):
        raise GeneratorComponentCoverageError(
            "component B128 fit and B256 held-out coverage requires an exact "
            "same-token-shape pair so validation isolates the batch axis"
        )
    if (
        isinstance(reference_max_num_seqs, bool)
        or not isinstance(reference_max_num_seqs, int)
        or reference_max_num_seqs < max(REQUIRED_RESIDENT_BATCHES)
    ):
        raise GeneratorComponentCoverageError(
            "TP1/PP1 reference max_num_seqs cannot cover required resident "
            f"batch {max(REQUIRED_RESIDENT_BATCHES)}"
        )
    return {
        "schema": COMPONENT_COVERAGE_SCHEMA,
        "batch_unit": "resident_sequences",
        "required_batches": list(REQUIRED_RESIDENT_BATCHES),
        "fit_batches": fit_batches,
        "held_out_batches": held_out_batches,
        "reference_max_num_seqs": int(reference_max_num_seqs),
        "status": "complete",
    }


def _gen_cfg(model: str, tp: int = 1, pp: int = 1) -> dict:
    return {"node_lines": [{"node_line_name": "main", "nodes": [
        {"stage": "semantic_retrieval", "top_k": 4,
         "modules": [{"component": "vectordb", "vectordb": "v", "batch_size": 1}]},
        {"stage": "generator",
         "modules": [{"component": "vllm", "model": model, "batch_size": 1,
                      "tensor_parallel": int(tp), "pipeline_parallel": int(pp)}]}]}],
        "_token_stats": {"avg_chunk_tokens": 128, "prompt_template_tokens": 32,
                         "avg_output_tokens": 128, "avg_query_tokens": 16}}


def _telemetry_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-")
    return slug or "unknown"


def _component_telemetry_path(
    hardware_key: str,
    model: str,
    max_num_seqs: int,
):
    """Return a unique, durable raw-evidence path under calibration_data."""

    from rag_stack.calibration_paths import calibration_data_root

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    suffix = uuid.uuid4().hex[:12]
    filename = (
        f"{_telemetry_slug(model)}__tp1_pp1_cap{int(max_num_seqs)}__"
        f"{timestamp}_{suffix}.jsonl"
    )
    return (
        calibration_data_root()
        / "generator_component"
        / _telemetry_slug(hardware_key)
        / filename
    )


def _freeze_component_measurement_manifest(
    telemetry_path,
    *,
    hardware_key: str,
    model: str,
    max_num_seqs: int,
    startup_key,
    repeats: int,
    results: dict,
    requested_shapes: list[tuple[int, int, int]],
    calibration_key_of: dict,
    validation_specs: set,
    resident_batch_coverage: dict,
    request_contracts: dict,
):
    """Persist the raw-file identity plus HTTP diagnostics and fit source.

    The JSONL remains the original append-only worker evidence.  This manifest
    makes the exact cohort-to-fit mapping auditable without relabelling HTTP
    latency as component service.
    """

    from rag_stack.calibration_paths import atomic_write_json
    from rag_stack.cost_model.llm_sim.calibration import (
        GENERATOR_COMPONENT_FIT_LATENCY_SOURCE,
    )
    from .vllm_probe import (
        _COMPONENT_MEASUREMENT_COHORT_COUNT,
        _COMPONENT_MAX_RELATIVE_SPREAD,
        _COMPONENT_TOTAL_COHORT_COUNT,
        _COMPONENT_WARMUP_COHORT_COUNT,
        _audit_offline_component_raw,
    )
    from .generator_request_contract import (
        generator_request_contracts_sha256,
        validate_generator_request_contracts,
    )

    if int(repeats) != _COMPONENT_TOTAL_COHORT_COUNT:
        raise ValueError(
            "generator component manifest requires two warmup and three "
            "measurement cohorts"
        )
    source_fit_points = []
    source_validation_points = []
    for measurement_key in requested_shapes:
        batch_size, token_shape = calibration_key_of[measurement_key]
        source_spec = (
            int(batch_size), int(token_shape[0]), int(token_shape[1])
        )
        (
            source_validation_points
            if source_spec in validation_specs
            else source_fit_points
        ).append(source_spec)
    expected_coverage = _resident_batch_coverage_contract(
        source_fit_points,
        source_validation_points,
        reference_max_num_seqs=int(max_num_seqs),
    )
    if resident_batch_coverage != expected_coverage:
        raise GeneratorComponentCoverageError(
            "component manifest coverage differs from the measured recipe"
        )
    normalized_request_contracts = validate_generator_request_contracts(
        request_contracts,
        expected_prompt_tokens=[shape[1] for shape in requested_shapes],
    )
    request_contracts_sha256 = generator_request_contracts_sha256(
        normalized_request_contracts
    )
    raw_points = _audit_offline_component_raw(
        telemetry_path,
        requested_shapes=requested_shapes,
    )
    run_ids = {point.run_id for point in raw_points.values()}
    if len(run_ids) != 1:
        raise ValueError("generator component raw must contain exactly one run")

    raw = telemetry_path.read_bytes()
    rows = []
    for measurement_key in requested_shapes:
        qps, cohort_wall_s, component_service_s = results[measurement_key]
        raw_point = raw_points[measurement_key]
        if float(component_service_s) != raw_point.component_service_s:
            raise ValueError(
                "generator component fit service disagrees with raw evidence"
            )
        calibration_key = calibration_key_of[measurement_key]
        batch_size, token_shape = calibration_key
        source_spec = (
            int(batch_size), int(token_shape[0]), int(token_shape[1])
        )
        rows.append({
            "split": "validation" if source_spec in validation_specs else "fit",
            "client_batch": int(batch_size),
            "input_tokens": int(measurement_key[1]),
            "output_tokens": int(measurement_key[2]),
            "offline_cohort_qps_diagnostic": float(qps),
            "offline_cohort_wall_s_diagnostic": float(cohort_wall_s),
            "component_service_s": float(component_service_s),
            "measurement_relative_spread": float(
                raw_point.measurement_relative_spread
            ),
            "cohorts": [
                {
                    "cohort_index": cohort_index,
                    "role": (
                        "warmup"
                        if cohort_index < _COMPONENT_WARMUP_COHORT_COUNT
                        else "selected_measurement"
                    ),
                    "component_service_s": float(service),
                    "cycle_ids": list(
                        raw_point.cohort_cycle_ids[cohort_index]
                    ),
                }
                for cohort_index, service in enumerate(
                    raw_point.warmup_component_services_s
                    + raw_point.measurement_component_services_s
                )
            ],
        })
    manifest_path = telemetry_path.with_suffix(".manifest.json")
    atomic_write_json(manifest_path, {
        "schema": "rag_stack.generator_component_measurement",
        "hardware_key": str(hardware_key),
        "model": str(model),
        "run_id": next(iter(run_ids)),
        "reference_execution": {
            "mode": "offline_inprocess_engine_full_cohort_enqueue",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "max_num_seqs": int(max_num_seqs),
            "max_num_batched_tokens": int(startup_key.max_num_batched_tokens),
            "max_model_len": int(startup_key.max_model_len),
            "enable_prefix_caching": startup_key.enable_prefix_caching,
            "dtype": str(startup_key.dtype),
            "kv_cache_dtype": str(startup_key.kv_cache_dtype),
        },
        "cohort_contract": {
            "warmup_cohorts": _COMPONENT_WARMUP_COHORT_COUNT,
            "selected_measurement_cohorts": (
                _COMPONENT_MEASUREMENT_COHORT_COUNT
            ),
            "total_cohorts": _COMPONENT_TOTAL_COHORT_COUNT,
            "selection": "median_of_selected_measurement_cohort_totals",
            "max_selected_relative_spread": _COMPONENT_MAX_RELATIVE_SPREAD,
        },
        "resident_batch_coverage": resident_batch_coverage,
        "request_contracts": normalized_request_contracts,
        "request_contracts_sha256": request_contracts_sha256,
        "raw_telemetry_path": str(telemetry_path.resolve()),
        "raw_telemetry_sha256": hashlib.sha256(raw).hexdigest(),
        "fit_latency_source": GENERATOR_COMPONENT_FIT_LATENCY_SOURCE,
        "token_shape_proof": (
            "chat_template_rendered_prompt_ids_and_ignore_eos_exact_shape; "
            "ignore_eos_is_calibration_shape_control_not_production_stopping"
        ),
        "points": rows,
    })
    return manifest_path


def calibrate(model: str, config: str, tp: int = 1, pp: int = 1, devices=None) -> int:
    """Measure one TP1/PP1 GPU component reference, validate, fit, and save."""
    if int(tp) != 1 or int(pp) != 1:
        raise ValueError(
            "generator component calibration is topology-independent and "
            "requires the TP1/PP1 reference deployment"
        )
    from rag_stack.cost_model import COST_MODEL_REGISTRY
    from rag_stack.cost_model.llm_sim.calibration import (
        fit_and_save,
        _inject_shape,
        detect_hardware_key,
        hardware_from_config,
    )
    from .vllm_probe import build_key, measure_component_grid
    from .generator_request_contract import build_generator_request_contracts

    hw, sysd = hardware_from_config(config)
    hardware_key = detect_hardware_key(hw)
    if not isinstance(hardware_key, str) or not hardware_key.strip():
        raise ValueError("generator component calibration requires a hardware key")
    cm = COST_MODEL_REGISTRY["llm_sim"]()
    cfg = _gen_cfg(model, 1, 1)

    overlap = set(POINTS).intersection(VALIDATION_POINTS)
    if not VALIDATION_POINTS:
        raise ValueError(
            "generator component calibration requires held-out validation points"
        )
    if overlap:
        raise ValueError(
            "generator validation points must be disjoint from fit points: "
            f"{sorted(overlap)!r}"
        )
    shapes, pair_of, calibration_key_of = [], {}, {}
    validation_specs = set(VALIDATION_POINTS)
    all_points = list(POINTS) + list(VALIDATION_POINTS)
    if not all_points:
        raise ValueError("generator component calibration has no measurement points")
    calibration_cap = max(int(point[0]) for point in all_points)
    resident_batch_coverage = _resident_batch_coverage_contract(
        POINTS,
        VALIDATION_POINTS,
        reference_max_num_seqs=calibration_cap,
    )
    for (c, sl, out) in all_points:
        pair = (int(sl), int(out))
        p = cm._get_params(_inject_shape(cfg, 1, pair))
        in_tok, out_tok = int(p["input_tokens"]), int(p["output_tokens"])
        if pair_of.get((in_tok, out_tok), pair) != pair:
            raise ValueError(f"ambiguous grid: shapes {pair_of[(in_tok, out_tok)]} and {pair} "
                             f"both map to (in={in_tok}, out={out_tok})")
        pair_of[(in_tok, out_tok)] = pair
        measurement_key = (int(c), in_tok, out_tok)
        if measurement_key in calibration_key_of:
            raise ValueError(
                "duplicate generator measurement point after token-shape "
                f"normalization: {measurement_key!r}"
            )
        calibration_key_of[measurement_key] = (int(c), pair)
        shapes.append(measurement_key)

    if any(
        int(input_tokens) + int(output_tokens) > CALIBRATION_MAX_MODEL_LEN
        for _, input_tokens, output_tokens in shapes
    ):
        raise ValueError(
            "generator component calibration shape exceeds max_model_len="
            f"{CALIBRATION_MAX_MODEL_LEN}"
        )
    required_prefill_tokens = max(
        int(batch) * int(input_tokens)
        for batch, input_tokens, _ in shapes
    )
    # vLLM accepts an exact token cap.  Rounding 67k/134k to the next power of
    # two needlessly reserves very large compile/cudagraph buffers and can
    # reduce the 14B TP1 KV cache below the intended resident cohort.
    token_cap = max(
        CALIBRATION_MIN_NUM_BATCHED_TOKENS,
        required_prefill_tokens,
    )
    reference_devices = None
    if devices is not None:
        reference_devices = (
            [devices] if isinstance(devices, str) else list(devices)[:1]
        )
        if not reference_devices:
            raise ValueError("TP1/PP1 component calibration requires one device")
    key = build_key(
        model,
        1,
        1,
        sysd,
        reference_devices,
        max_num_seqs=calibration_cap,
        max_num_batched_tokens=token_cap,
        enable_prefix_caching=False,
        api_server_count=1,
        max_model_len=CALIBRATION_MAX_MODEL_LEN,
    )
    if (
        int(key.tensor_parallel_size) != 1
        or int(key.pipeline_parallel_size) != 1
        or int(key.max_num_seqs) < calibration_cap
        or int(key.max_num_batched_tokens) != token_cap
        or int(key.max_num_batched_tokens) < required_prefill_tokens
        or int(key.max_model_len) != CALIBRATION_MAX_MODEL_LEN
        or int(key.api_server_count) != 1
        or key.enable_prefix_caching is not False
    ):
        raise ValueError(
            "invalid generator component reference key: expected TP1/PP1 "
            f"with sequence cap >= {calibration_cap} and prefill token cap "
            f">= {required_prefill_tokens}"
        )
    telemetry_path = _component_telemetry_path(
        hardware_key, model, int(key.max_num_seqs),
    )
    request_contracts = build_generator_request_contracts(
        model,
        [shape[1] for shape in shapes],
    )
    print(
        f"=== ceff/meff component {model} tp1pp1 on {key.device} @ "
        f"{hardware_key} max_num_seqs={key.max_num_seqs} "
        f"max_num_batched_tokens={key.max_num_batched_tokens} "
        f"raw={telemetry_path} ===",
        flush=True,
    )
    try:
        # Launch one isolated offline engine and retain append-only worker
        # telemetry. Only tuple[2] GPU service enters fit.
        results = measure_component_grid(
            key,
            shapes,
            telemetry_path=telemetry_path,
            repeats=REPEATS,
            request_contracts=request_contracts,
        )
    except Exception as e:  # noqa: BLE001
        print(
            f"[generator] {model} TP1/PP1 REQUIRED COVERAGE FAILED: "
            f"required resident batches={list(REQUIRED_RESIDENT_BATCHES)}, "
            f"reference max_num_seqs={key.max_num_seqs}; "
            f"{type(e).__name__}: {str(e)[:120]}; no component profile was "
            f"published; raw={telemetry_path}",
            flush=True,
        )
        return 1

    missing = set(shapes).difference(results)
    unexpected = set(results).difference(shapes)
    if missing or unexpected:
        raise GeneratorComponentCoverageError(
            "component measurement coverage grid mismatch: "
            f"missing={sorted(missing)!r}, unexpected={sorted(unexpected)!r}"
        )
    measured_batches = {int(measurement_key[0]) for measurement_key in results}
    missing_required_batches = set(REQUIRED_RESIDENT_BATCHES).difference(
        measured_batches
    )
    if missing_required_batches:
        raise GeneratorComponentCoverageError(
            "component measurement did not produce required resident batches: "
            f"missing={sorted(missing_required_batches)}"
        )
    manifest_path = _freeze_component_measurement_manifest(
        telemetry_path,
        hardware_key=hardware_key,
        model=model,
        max_num_seqs=int(key.max_num_seqs),
        startup_key=key,
        repeats=REPEATS,
        results=results,
        requested_shapes=shapes,
        calibration_key_of=calibration_key_of,
        validation_specs=validation_specs,
        resident_batch_coverage=resident_batch_coverage,
        request_contracts=request_contracts,
    )
    measured, validation = {}, {}
    for measurement_key, (_qps, _cohort_wall_s, component_service_s) in results.items():
        calibration_key = calibration_key_of[measurement_key]
        c, pair = calibration_key
        target = (
            validation
            if (c, int(pair[0]), int(pair[1])) in validation_specs
            else measured
        )
        target[calibration_key] = component_service_s
    with _SAVE_LOCK:   # concurrent calibrations share llm_sim/<hw>.json — serialize the merge-write
        rec = fit_and_save(cost_model=cm, hardware=hw, pipeline_config=cfg, measured=measured,
                           component="agnostic", model_key=None, bits="bf16",
                           validation_points=validation)
    validation_summary = (
        f"{rec.validation_mape * 100:.1f}%"
        if rec.validation_mape is not None
        else "not_provided"
    )
    print(f"SAVED {model}: ceff={rec.compute_efficiency:.4f} meff={rec.memory_efficiency:.4f} "
          f"mape={rec.fit_mape*100:.1f}% validation_mape="
          f"{validation_summary} corr={rec.correlation:.3f} "
          f"evidence={manifest_path}", flush=True)
    return 0


def measure_all(config: str) -> None:
    """Calibrate every model that fits the one-GPU TP1/PP1 reference."""
    from .generator_parallel import MODELS, _fits, _size_b, _hw_from_config
    from rag_stack.cost_model.llm_sim.calibration import hardware_from_config
    from .vllm_probe import parallel_measure
    hw_key, num_gpus, mem_gb = _hw_from_config(config)
    _, sysd = hardware_from_config(config)
    print(f"[generator] adaptive ceff/meff on {hw_key}: {num_gpus} GPU × {mem_gb:.0f} GB (GPU-parallel)",
          flush=True)
    items = []
    for model in MODELS:
        if not _fits(_size_b(model), 1, 1, mem_gb):
            print(
                f"[generator] SKIP {model}: TP1/PP1 component reference does "
                f"not fit one {mem_gb:.0f} GB GPU",
                flush=True,
            )
            continue
        items.append({"model": model, "tp": 1, "pp": 1, "n_gpus": 1})
    failures = []
    failure_lock = threading.Lock()

    def run_one(item, devices):
        result = calibrate(
            item["model"], config, item["tp"], item["pp"], devices,
        )
        if result:
            with failure_lock:
                failures.append(item["model"])
        return []

    parallel_measure(items, sysd, run_one)
    if failures:
        raise RuntimeError(
            "generator component calibration failed for: "
            + ", ".join(sorted(failures))
        )


def main(argv=None) -> None:
    global POINTS, VALIDATION_POINTS, BATCH, SHAPES, REPEATS
    ap = argparse.ArgumentParser(prog="rag_stack calibrate generator")
    ap.add_argument("--model", default=None,
                    help="one HF model id; omit to calibrate the whole family (GPU-aware)")
    ap.add_argument("--config", required=True,
                    help="standard rag-stack config; deployment hardware is read from its `system` block")
    ap.add_argument("--points", default=None,
                    help="explicit batch:seq:out measurement points, comma-separated "
                         "(default: the 4-point minimal fit recipe)")
    ap.add_argument("--validation-points", default=None,
                    help="strictly held-out batch:seq:out points, comma-separated "
                         "(default includes the B256 maximum-batch check; "
                         "required with custom fit grids)")
    ap.add_argument("--batch-sizes", default=None,
                    help="density mode: comma-separated client-concurrency ladder; measured "
                         "points become this ladder × --shapes (default ladder: module BATCH)")
    ap.add_argument("--shapes", default=None,
                    help="density mode: comma-separated seq:out token-shape pairs, e.g. "
                         "'128:128,512:128,64:256,64:512' (default: module SHAPES)")
    ap.add_argument(
        "--repeats", type=int, default=None,
        help=(
            "fixed total cohorts per point; only 5 is accepted "
            "(2 warmup + 3 selected measurement)"
        ),
    )
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))
    if args.repeats is not None and int(args.repeats) != REPEATS:
        raise SystemExit(
            "--repeats is fixed at 5: two warmup + three selected "
            "measurement cohorts"
        )
    if args.points and (args.batch_sizes or args.shapes):
        raise SystemExit("--points and --batch-sizes/--shapes are mutually exclusive")
    custom_fit_grid = bool(args.points or args.batch_sizes or args.shapes)
    if args.points:
        POINTS = [tuple(int(t) for t in x.split(":")) for x in args.points.split(",") if x]
        if any(len(p) != 3 for p in POINTS):
            raise SystemExit("--points expects batch:seq:out triples, e.g. 256:64:512")
    elif args.batch_sizes or args.shapes:   # density mode: full cross product
        if args.batch_sizes:
            BATCH = [int(x) for x in args.batch_sizes.split(",") if x]
        if args.shapes:
            SHAPES = [tuple(int(t) for t in x.split(":")) for x in args.shapes.split(",") if x]
            if any(len(p) != 2 for p in SHAPES):
                raise SystemExit("--shapes expects seq:out pairs, e.g. 128:128,64:512")
        POINTS = [(c, sl, out) for (sl, out) in SHAPES for c in BATCH]
    if args.validation_points:
        VALIDATION_POINTS = [
            tuple(int(t) for t in x.split(":"))
            for x in args.validation_points.split(",") if x
        ]
        if any(len(p) != 3 for p in VALIDATION_POINTS):
            raise SystemExit(
                "--validation-points expects batch:seq:out triples, e.g. "
                "16:128:128,64:64:256"
            )
    elif custom_fit_grid:
        raise SystemExit(
            "custom generator fit grids require explicit --validation-points; "
            "held-out evidence cannot be inferred from training points"
        )

    if args.model:
        from .generator_parallel import _fits, _size_b, _hw_from_config
        _, num_gpus, mem_gb = _hw_from_config(args.config)
        if num_gpus < 1 or not _fits(_size_b(args.model), 1, 1, mem_gb):
            raise SystemExit(
                f"[generator] {args.model} cannot use the required TP1/PP1 "
                f"component reference on a {mem_gb:.0f} GB GPU"
            )
        raise SystemExit(calibrate(args.model, args.config, 1, 1))
    measure_all(args.config)


if __name__ == "__main__":
    main()
