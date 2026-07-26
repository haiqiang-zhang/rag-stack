"""Sparse vLLM serving-stage calibration.

This command measures one inexpensive TP1/PP1 reference deployment per hardware,
not a model×TP×PP deployment grid.  TP communication stays in GenZ and PP uses
a deterministic full-resident-batch scheduler-step law.  The only published
empirical serving dimension is ``hardware × collocated stage × resident batch``.

The fixed anchors are 1/8/32/64/256; batch 128 is measured only as an
out-of-sample validation point.  The production vLLM scheduler provides the
active/resident batch evidence, so HTTP client concurrency is never relabelled
as a calibration batch.  Each resident batch uses three fresh, exact-cap
subruns under one identical startup/workload contract: component-only GPU
service (two exact warmups discarded, then three cohorts median-pooled), shape proof, and
telemetry-disabled production timing.  The complete out=N production service
and complete independent component cohort are normalized by the same exact N
scheduler cycles.  Their non-negative difference is one collocated engine-cycle
overhead; the short probe is validation-only and never creates separate P/D
knobs.  Any negative independent-run difference fails the complete fit; it is
never omitted, interpolated over, or clamped into an apparently valid anchor.

    python -m rag_stack.cost_model.llm_sim.calibration.main_llm.generator_parallel \
        measure --config <cfg> [--force]
    python -m rag_stack.cost_model.llm_sim.calibration.main_llm.generator_parallel \
        fit --config <cfg>

"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import statistics

from rag_stack.calibration_inventory import CANONICAL_GENERATOR_MODELS
from rag_stack.calibration_paths import (
    atomic_write_json,
    calibration_data_root,
    calibration_profile_root,
)

from .generator import _gen_cfg
from .generator_request_contract import (
    validate_generator_request_contracts,
)
from .stage_timing_contract import PRODUCTION_COHORT_COUNT
from .vllm_probe import build_key, measure_stage_ladder

# ---- sparse hardware × stage × resident-batch calibration contract ----
MODELS = list(CANONICAL_GENERATOR_MODELS)
# Five mandatory anchors define one non-negative piecewise-linear stage
# overhead.  Any negative raw anchor fails the complete fit. B128 is
# measured but held out from publication so interpolation is validated out of
# sample.  These values are *scheduler-observed resident batches*, never HTTP
# client concurrency labels.
STAGE_BATCH_ANCHORS = (1, 8, 32, 64, 256)
STAGE_BATCH_VALIDATION = (128,)
BATCH_LADDER = sorted((*STAGE_BATCH_ANCHORS, *STAGE_BATCH_VALIDATION))
MAX_STAGE_BATCH = max(STAGE_BATCH_ANCHORS)
_STAGE_NOFILE_HEADROOM = 1_024
_STAGE_NOFILE_MINIMUM = (
    PRODUCTION_COHORT_COUNT * MAX_STAGE_BATCH + _STAGE_NOFILE_HEADROOM
)
_STAGE_NOFILE_TARGET = 65_536
# A short reference prompt keeps a real B256 prefill feasible.  Token/context
# physics remain in GenZ; the stage overhead has only a batch axis.
SEQ = 16
MAX_NUM_BATCHED_TOKENS = 65_536
# Measurement is NON-STREAMING (fixed-batch llm.generate) → gpu_memory_utilization is NOT set
# (it's a KV/continuous-batching budget, irrelevant to per-batch latency). vLLM uses its own
# default; a batch that doesn't fit OOMs and stops the ladder. _FIT_MEM_FRAC below is ONLY the
# offline feasibility estimate (≈ vLLM's default), never passed to vLLM.
_FIT_MEM_FRAC = 0.90
_KV_HEADROOM_GB = 2.0  # min room (KV+activations+overhead) left after weights, at batch 1


def _ensure_stage_nofile_limit() -> tuple[int, int]:
    """Raise and verify the FD budget before deleting frozen evidence.

    The B256 timing run deliberately holds eight paused cohorts, or 2,048
    concurrent HTTP sockets.  The common shell default of 1,024 therefore
    fails deterministically before the saturation evidence can be formed.
    Both the calibration parent and every subsequently spawned server inherit
    this limit.
    """

    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    desired = min(int(hard), max(int(soft), _STAGE_NOFILE_TARGET))
    if desired > int(soft):
        resource.setrlimit(resource.RLIMIT_NOFILE, (desired, hard))
    verified_soft, verified_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if int(verified_soft) < _STAGE_NOFILE_MINIMUM:
        raise RuntimeError(
            "generator stage calibration needs at least "
            f"{_STAGE_NOFILE_MINIMUM} open files for "
            f"{PRODUCTION_COHORT_COUNT}xB{MAX_STAGE_BATCH} paused timing "
            f"requests; observed soft={verified_soft}, hard={verified_hard}"
        )
    return int(verified_soft), int(verified_hard)


# ----------------------------------------------------------------------
# GPU-AWARE case selection. Unlike FAISS (device-independent cases), LLM
# calibration cases depend on the box: how many GPUs (which tp/pp are even
# possible) and how much memory (which models fit at each parallelism). The
# grid is DERIVED from the config's `system` block — num_gpus from
# available_gpus, per-GPU mem from gpu.Memory_size (preset → genz_llm/Systems).
# ----------------------------------------------------------------------
def _parallelism_configs(num_gpus: int):
    """All power-of-2 (tp, pp) with tp·pp ≤ num_gpus, ordered by chips then tp-heavy."""
    pows = [p for p in (1, 2, 4, 8, 16, 32) if p <= max(1, num_gpus)]
    cfgs = {(tp, pp) for tp in pows for pp in pows if tp * pp <= num_gpus}
    return sorted(cfgs, key=lambda x: (x[0] * x[1], -x[0]))


def _fits(size_b: float, tp: int, pp: int, mem_gb: float, util: float = _FIT_MEM_FRAC) -> bool:
    """Weights (bf16, sharded across tp·pp) must fit with KV headroom left on each GPU."""
    per_gpu_weights = 2.0 * size_b / (tp * pp)
    return per_gpu_weights <= mem_gb * util - _KV_HEADROOM_GB


def select_cases(models, num_gpus: int, mem_gb: float):
    """GPU-aware (model, tp, pp) grid: every feasible parallelism for each model on this box."""
    cases = []
    for model in models:
        sz = _size_b(model)
        for (tp, pp) in _parallelism_configs(num_gpus):
            if sz and _fits(sz, tp, pp, mem_gb):
                cases.append((model, tp, pp))
    return cases


def _size_b(model: str):
    m = re.search(r"([0-9.]+)\s*B", model)
    return float(m.group(1)) if m else None


def _hw_from_config(config: str):
    """(hw_key, num_gpus, mem_per_gpu_GB) read from the config's `system` block — the single
    hardware input (the GPU key/count/mem; no torch.cuda, no nvidia-smi)."""
    from rag_stack.cost_model.llm_sim.calibration import (
        hardware_from_config, detect_hardware_key, available_gpu_count, gpu_memory_gb)
    hw, sysd = hardware_from_config(config)
    return detect_hardware_key(hw), available_gpu_count(sysd), gpu_memory_gb(sysd)


def _system_hardware_from_config(config: str) -> str:
    from rag_stack.cost_model.llm_sim.calibration import hardware_from_config
    from rag_stack.cost_model.stage_batch_curve import validate_system_hardware

    hardware, _ = hardware_from_config(config)
    try:
        return validate_system_hardware(
            getattr(hardware, "system_hardware", None)
        )
    except ValueError as exc:
        raise ValueError(
            "stage-batch calibration requires a path-safe "
            "config.system.system_hardware; a GPU SKU alone cannot identify "
            "host-side serving overhead"
        ) from exc


def _data_path(hw_key: str) -> str:
    return str(
        calibration_data_root()
        / "generator_stage_batch"
        / f"{hw_key}.jsonl"
    )


def _component_data_path(hw_key: str) -> str:
    return str(
        calibration_data_root()
        / "generator_stage_batch"
        / f"{hw_key}.component.jsonl"
    )


def _label(tp: int, pp: int) -> str:
    return f"tp{tp}" + (f"pp{pp}" if pp > 1 else "")


def _shape():
    """(in_tok, out_tok) for the representative generator shape — system-free
    (derived from the pipeline _token_stats, not from any hardware)."""
    from rag_stack.cost_model import COST_MODEL_REGISTRY
    from rag_stack.cost_model.llm_sim.calibration import _inject_shape
    cm = COST_MODEL_REGISTRY["llm_sim"]()
    p = cm._get_params(_inject_shape(_gen_cfg(MODELS[0]), 1, SEQ))
    return int(p["input_tokens"]), int(p["output_tokens"])


# ----------------------------------------------------------------------
# Measure one hardware reference campaign.  Each batch B launches exact-cap
# component/proof/timing subprocesses with max_num_seqs=B.
# ----------------------------------------------------------------------
def _measure_config(
    model: str,
    tp: int,
    pp: int,
    sysd: dict,
    in_tok: int,
    out_tok: int,
    stage_telemetry_path: str,
    component_telemetry_path: str,
    devices=None,
) -> None:
    if int(tp) != 1 or int(pp) != 1:
        raise ValueError(
            "stage-batch calibration uses exactly one TP1/PP1 reference "
            "deployment; TP/PP candidates are analytical inputs, not "
            "calibration identities"
        )
    key = build_key(
        model,
        tp,
        pp,
        sysd,
        devices,
        max_num_seqs=MAX_STAGE_BATCH,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enable_prefix_caching=False,
        api_server_count=1,
        max_model_len=4096,
    )
    print(
        f"=== generator stage-batch {model} TP1/PP1 on {key.device} "
        f"in{in_tok} out{out_tok} max_num_seqs={key.max_num_seqs} "
        f"anchors={list(STAGE_BATCH_ANCHORS)} "
        f"heldout={list(STAGE_BATCH_VALIDATION)} ===",
        flush=True,
    )
    measure_stage_ladder(
        key,
        BATCH_LADDER,
        in_tok,
        out_tok,
        telemetry_path=stage_telemetry_path,
        component_telemetry_path=component_telemetry_path,
    )


def _resolve_cases(config, models, parallels):
    """The GPU-aware (model, tp, pp) case list from the config's hardware (or a manual override)."""
    hw_key, num_gpus, mem_gb = _hw_from_config(config)
    if parallels:  # manual override (testing): model × given parallelism
        cases = [(m, tp, pp) for m in models for (tp, pp) in parallels]
    else:
        cases = select_cases(models, num_gpus, mem_gb)
    return cases, hw_key, num_gpus, mem_gb


def _print_case_selection(config, cases, hw_key: str, num_gpus: int, mem_gb: float, models=None) -> None:
    """Show the GPU-aware case selection derived from the config."""
    models = models or MODELS
    print(f"[gen-tp] hw={hw_key} (from {config})  {num_gpus} GPU × {mem_gb:.0f} GB  "
          f"(fit-fraction {_FIT_MEM_FRAC}, KV headroom {_KV_HEADROOM_GB} GB)", flush=True)
    print(f"[gen-tp] parallelism on {num_gpus} GPUs: "
          f"{[_label(tp, pp) for tp, pp in _parallelism_configs(num_gpus)]}", flush=True)
    print(f"[gen-tp] selected {len(cases)} cases → {_data_path(hw_key)}:", flush=True)
    by_model = {}
    for m, tp, pp in cases:
        by_model.setdefault(m, []).append(_label(tp, pp))
    for m in models:
        feasible = by_model.get(m, [])
        dropped = [_label(tp, pp) for tp, pp in _parallelism_configs(num_gpus)
                   if _label(tp, pp) not in feasible]
        sz = _size_b(m)
        print(f"  {m.split('/')[-1]:28s} ({2*sz:.0f}GB bf16)  selected: {feasible or '—'}"
              + (f"   [skip: {dropped}]" if dropped else ""), flush=True)


def _reference_case(cases):
    """Choose one cheap TP1/PP1 evidence deployment for this hardware.

    The selected model is acquisition-only. No runtime lookup or fitted
    coefficient may branch on it.
    """

    reference = [case for case in cases if case[1:] == (1, 1)]
    if not reference:
        raise ValueError(
            "generator stage-batch calibration requires one feasible TP1/PP1 "
            "reference model"
        )
    return min(reference, key=lambda case: (_size_b(case[0]) or float("inf"), case[0]))


def measure_grid(config, force: bool, models=None, parallels=None, timeout_s: int = 1800,
                 full: bool = False) -> None:
    from rag_stack.cost_model.llm_sim.calibration import hardware_from_config

    models = models or MODELS
    cases, hw_key, num_gpus, mem_gb = _resolve_cases(config, models, parallels)
    if full or parallels:
        raise ValueError(
            "model×TP×PP calibration grids are retired; stage-batch calibration "
            "uses one hardware reference deployment"
        )
    reference = _reference_case(cases)
    stage_hardware = _system_hardware_from_config(config)
    path = _data_path(stage_hardware)
    component_path = _component_data_path(stage_hardware)
    artifacts = (
        path,
        component_path,
        f"{path}.contract.json",
        f"{component_path}.contract.json",
    )
    if any(os.path.exists(artifact) for artifact in artifacts) and not force:
        print(
            f"[generator-stage] frozen telemetry exists: {path} / "
            f"{component_path} (use --force to re-measure)",
            flush=True,
        )
        return
    # Complete every non-destructive preflight before replacing a frozen
    # campaign.  In particular, B256 cannot run under the usual 1024-FD shell
    # limit, and an invalid config/shape must not erase prior evidence.
    _, sysd = hardware_from_config(config)
    in_tok, out_tok = _shape()
    nofile_soft, _ = _ensure_stage_nofile_limit()
    print(
        f"[generator-stage] verified nofile soft={nofile_soft} "
        f"for {PRODUCTION_COHORT_COUNT}xB{MAX_STAGE_BATCH} timing",
        flush=True,
    )
    for artifact in artifacts:
        if os.path.exists(artifact):
            os.unlink(artifact)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model, tp, pp = reference
    print(
        f"[generator-stage] system_hardware={stage_hardware} gpu={hw_key} "
        f"{num_gpus} GPU × {mem_gb:.0f} GB; "
        f"one reference={model} TP1/PP1; six batch points → {path}",
        flush=True,
    )
    _measure_config(
        model,
        tp,
        pp,
        sysd,
        in_tok,
        out_tok,
        path,
        component_path,
    )
    print(
        f"[generator-stage] froze isolated stage/component telemetry → "
        f"{path} / {component_path}",
        flush=True,
    )


def _remove_retired_component_blocks(hardware) -> None:
    """Remove the two superseded deployment-calibration payloads first.

    This write precedes stage publication.  If publication subsequently fails,
    runtime either keeps using an older valid stage generation or fails closed;
    it can never see a newly published stage generation alongside the old
    deployment-specific overhead source.
    """

    from rag_stack.cost_model.llm_sim.calibration import detect_hardware_key

    gpu_key = detect_hardware_key(hardware)
    path = calibration_profile_root() / "llm_sim" / f"{gpu_key}.json"
    if not path.exists():
        return
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot sanitize component profile {path}") from exc
    from rag_stack.llm_sim_profile_schema import (
        unwrap_llm_sim_profile_document,
        wrap_llm_sim_profiles,
    )

    profiles = unwrap_llm_sim_profile_document(document, source=str(path))
    changed = False
    for retired in ("_tp_pp_efficiency", "_vllm_scheduler_step_residual"):
        changed = profiles.pop(retired, None) is not None or changed
    if changed:
        atomic_write_json(path, wrap_llm_sim_profiles(profiles))


def _measurements_for_collocated_engine(
    *,
    component_cohort_measurements: dict[int, dict[str, object]],
    stage_service_measurements: dict[int, dict[str, object]],
    system_hardware: str,
    stage: str,
):
    from rag_stack.cost_model.stage_batch_curve import (
        ANCHOR_BATCHES,
        HELD_OUT_BATCH,
        StageBatchCurveKey,
        StageBatchMeasurement,
    )

    key = StageBatchCurveKey(system_hardware, stage)
    required = (*ANCHOR_BATCHES, HELD_OUT_BATCH)
    measurements = []
    for batch in required:
        component = component_cohort_measurements.get(int(batch))
        if not isinstance(component, dict):
            raise ValueError(f"no exact full component cohort for B{batch}")
        cohort_totals = [
            float(value) for value in component["cohort_total_service_s"]
        ]
        if not cohort_totals:
            raise ValueError(f"no full component cohort totals for B{batch}")
        service = stage_service_measurements.get(int(batch))
        if not isinstance(service, dict):
            raise ValueError(
                f"no frozen short/long stage service for B{batch}"
            )
        scheduler_cycles = int(service["collocated_engine_cycle_count"])
        component_scheduler_cycles = int(component["scheduler_cycle_count"])
        if scheduler_cycles != component_scheduler_cycles:
            raise ValueError(
                f"B{batch} full production/component cohorts disagree on "
                f"scheduler cycles: production={scheduler_cycles}, "
                f"component={component_scheduler_cycles}"
            )
        observed_total = float(service[
            "collocated_engine_cycle_service_s"
        ])
        sample_count = min(int(service["sample_count"]), len(cohort_totals))
        # The full out=N production cohort is one real collocated engine
        # interval containing exactly N scheduler cycles: one pure-prefill
        # cycle followed by N-1 decode cycles.  Compare it with the complete
        # independent component cohort using the same arithmetic
        # normalization.  The out=1 probe is intentionally not used to split
        # or fit phase-specific residuals.
        reference_component = (
            float(statistics.median(cohort_totals)) / scheduler_cycles
        )
        overhead = observed_total - reference_component
        if overhead < 0.0:
            raise ValueError(
                f"B{batch} collocated engine-cycle overhead is negative: "
                f"observed_total_service_s={observed_total:.17g}, "
                "reference_component_service_s="
                f"{reference_component:.17g}, raw_overhead_s={overhead:.17g}; "
                "the complete fit must fail so a measurement/contract bug "
                "cannot be hidden by interpolation (never clamp)"
            )
        measurements.append(StageBatchMeasurement(
            key=key,
            target_batch=int(batch),
            engine_max_num_seqs=int(batch),
            observed_resident_batch=int(batch),
            observed_total_service_s=observed_total,
            reference_component_service_s=reference_component,
            overhead_s=overhead,
            sample_count=sample_count,
        ))
    return key, measurements


def fit(config) -> None:
    """Publish sparse collocated hardware×stage×batch curves from telemetry."""

    from rag_stack.cost_model.llm_sim.calibration import hardware_from_config
    from rag_stack.cost_model.stage_batch_curve import (
        build_stage_batch_curve,
        publish_stage_batch_curves,
    )
    from .vllm_probe import (
        _normalized_contract_workload,
        read_component_full_cohort_service_measurements,
        read_stage_active_service_measurements,
        read_stage_cycle_observations,
    )

    hardware, _ = hardware_from_config(config)
    system_hardware = _system_hardware_from_config(config)
    source_path = _data_path(system_hardware)
    component_source_path = _component_data_path(system_hardware)
    stage_contract_path = f"{source_path}.contract.json"
    component_contract_path = f"{component_source_path}.contract.json"
    evidence_paths = (
        source_path,
        component_source_path,
        stage_contract_path,
        component_contract_path,
    )
    if any(not os.path.exists(path) for path in evidence_paths):
        raise SystemExit(
            f"[generator-stage] incomplete frozen telemetry at {source_path} / "
            f"{component_source_path} and their run contracts; "
            "run `measure` first"
        )
    observations = read_stage_cycle_observations(source_path)
    models = {item.model for item in observations}
    if len(models) != 1:
        raise ValueError("stage telemetry must contain one reference model")
    component_cohort_measurements = (
        read_component_full_cohort_service_measurements(
        component_source_path,
        observations,
        stage_telemetry_path=source_path,
        )
    )
    stage_service_measurements = read_stage_active_service_measurements(
        source_path
    )
    stage_contract_document = json.loads(Path(stage_contract_path).read_text())
    component_contract_document = json.loads(
        Path(component_contract_path).read_text()
    )
    if _normalized_contract_workload(
        stage_contract_document
    ) != _normalized_contract_workload(component_contract_document):
        raise ValueError("stage/component request contracts differ")
    request_contracts = validate_generator_request_contracts(
        stage_contract_document["workload"].get("request_contracts"),
        expected_prompt_tokens=[
            int(stage_contract_document["workload"]["prompt_tokens"])
        ],
    )
    if len(request_contracts) != 1:
        raise ValueError("stage calibration requires one request contract")
    curves = []
    # Generator and query expansion use the same complete collocated vLLM
    # engine-cycle boundary.  This campaign deliberately cannot emit
    # role-specific P/D keys; those require independent disaggregated-engine
    # measurements.
    for target_stage in ("generator", "query_expansion"):
        key, measurements = (
            _measurements_for_collocated_engine(
                component_cohort_measurements=component_cohort_measurements,
                stage_service_measurements=stage_service_measurements,
                system_hardware=system_hardware,
                stage=target_stage,
            )
        )
        curves.append(build_stage_batch_curve(
            key=key,
            measurements=measurements,
        ))
    profile_path = (
        calibration_profile_root()
        / "stage_batch"
        / f"{system_hardware}.json"
    )
    _remove_retired_component_blocks(hardware)
    publish_stage_batch_curves(profile_path, curves)
    published_anchors = {
        curve.key.stage: [
            anchor.batch_size for anchor in curve.anchors
        ]
        for curve in curves
    }
    print(
        f"[generator-stage] published {len(curves)} hardware×stage curves "
        f"with anchors={published_anchors}, heldout="
        f"{list(STAGE_BATCH_VALIDATION)} → {profile_path}",
        flush=True,
    )


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="rag_stack calibrate generator-tp")
    ap.add_argument("verb", nargs="?", default="measure", choices=["measure", "fit"])
    ap.add_argument("--config", default=None,
                    help="standard rag-stack config; hardware (spec + key) is read from its `system` block")
    ap.add_argument("--force", action="store_true", help="(measure) re-measure even if frozen data exists")
    ap.add_argument("--models", default=None, help="comma-substr filter on the model grid (testing)")
    ap.add_argument("--parallel", default=None, help="comma tp:pp filter, e.g. 1:1,4:1 (testing)")
    ap.add_argument("--full", action="store_true",
                    help="(measure) the COMPLETE grid (all pp configs × all sizes); default = fast subset")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args(list(sys.argv[1:] if argv is None else argv))

    if not args.config:
        raise SystemExit("[gen-tp] --config <rag-stack config> required (reads hardware from system block)")
    models = [m for m in MODELS if not args.models or any(k in m for k in args.models.split(","))]
    parallels = (None if not args.parallel
                 else [tuple(int(x) for x in p.split(":")) for p in args.parallel.split(",")])
    if args.verb == "fit":
        fit(args.config)
        return
    measure_grid(args.config, args.force, models, parallels, args.timeout, args.full)


if __name__ == "__main__":
    main()
