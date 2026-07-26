"""CPU-only contracts for the unified calibration inventory.

These tests deliberately build isolated YAML/JSON fixtures instead of reading
the repository's mutable calibration roots.  In particular, the historical
TP/PP and scheduler-residual blocks must have no bearing on current
readiness: analytical TP/PP plus the two collocated hardware x stage x batch
curves are
the complete serving contract.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import rag_stack.calibration_inventory as inventory
from rag_stack.cost_model.llm_sim.calibration.main_llm import (
    generator_request_contract,
)
from rag_stack.cost_model.llm_sim.calibration.main_llm import (
    stage_timing_contract,
)
from rag_stack.cost_model.stage_batch_curve import (
    ANCHOR_BATCHES,
    HELD_OUT_BATCH,
    StageBatchCurveKey,
    StageBatchMeasurement,
    build_stage_batch_curve,
)


_GPU_KEY = "H100_GPU"
_CPU_KEY = "epyc_9124"
_SYSTEM_HARDWARE = "sgs_h100x4_epyc9124"
_MODEL = "Qwen/Qwen2.5-7B-Instruct"
_MODEL_KEY = inventory.generator_record_model_key(_MODEL)
_REQUIRED_STAGES = (
    "generator",
    "query_expansion",
)


def test_inventory_reads_embedding_registry_from_production_cost_model() -> None:
    assert {
        "huggingface_all_mpnet_base_v2",
        "huggingface_baai_bge_small",
        "huggingface_bge_m3",
    }.issubset(inventory._genz_model_map_keys())


def _request_contracts(prompt_tokens: int) -> dict[str, dict[str, object]]:
    template = "fixture-chat-template"
    token_ids = list(range(prompt_tokens))
    return {
        str(prompt_tokens): {
            "schema": generator_request_contract.REQUEST_CONTRACT_SCHEMA,
            "endpoint": generator_request_contract.ENDPOINT,
            "request_format": generator_request_contract.REQUEST_FORMAT,
            "messages": [{"role": "user", "content": "fixture question"}],
            "chat_template": {
                "source": generator_request_contract.CHAT_TEMPLATE_SOURCE,
                "tokenize": True,
                "add_generation_prompt": True,
                "chat_template_sha256": hashlib.sha256(
                    template.encode("utf-8")
                ).hexdigest(),
            },
            "rendered_prompt_tokens": prompt_tokens,
            "rendered_prompt_token_ids_sha256": (
                generator_request_contract._sha256(token_ids)
            ),
            "temperature": (
                generator_request_contract.CALIBRATION_TEMPERATURE
            ),
            "seed": generator_request_contract.CALIBRATION_SEED,
            "stream": False,
            "ignore_eos": True,
            "ignore_eos_purpose": (
                generator_request_contract.IGNORE_EOS_PURPOSE
            ),
            "logprobs": True,
            "top_logprobs": 0,
            "n": 1,
            "max_tokens_policy": (
                generator_request_contract.MAX_TOKENS_POLICY
            ),
            "truncate_prompt_tokens_policy": (
                generator_request_contract.TRUNCATE_PROMPT_TOKENS_POLICY
            ),
            "truncation_side": generator_request_contract.TRUNCATION_SIDE,
        }
    }


def _point(batch: int, seq_len: tuple[int, int]) -> dict[str, object]:
    return {
        "batch_size": batch,
        "seq_len": list(seq_len),
        "measured_s": 0.01,
        "predicted_s": 0.01,
        "rel_err": 0.0,
    }


def _roofline_record() -> dict[str, object]:
    """A boundary-valid generator component record.

    Condition=100 and held-out MAPE=20% intentionally exercise the inclusive
    publication limits.  B128 is a sparse fit anchor and B256 is an independent
    same-token-shape holdout, matching the component coverage contract.
    """

    return {
        "component": "agnostic",
        "model_key": _MODEL_KEY,
        "hardware_key": _GPU_KEY,
        "calibration_contract": "standalone_ceff_meff_zero_intercept_v1",
        "compute_efficiency": 0.5,
        "memory_efficiency": 0.6,
        "fit_mape": 0.20,
        "per_call_overhead_s": 0.0,
        "correlation": 0.95,
        "bits": "bf16",
        "grid": [
            _point(32, (512, 128)),
            _point(128, (64, 512)),
            _point(4, (64, 256)),
            _point(128, (64, 128)),
        ],
        "identifiability_rank": 2,
        "identifiability_condition": 100.0,
        "validation_mape": 0.20,
        "validation_grid": [
            _point(16, (128, 128)),
            _point(256, (64, 128)),
        ],
    }


def _stage_profile(
    stage: str,
    *,
    overhead_delta_s: float = 0.0,
) -> dict[str, object]:
    key = StageBatchCurveKey(_SYSTEM_HARDWARE, stage)
    # A simple affine curve makes B128 exactly interpolable while retaining a
    # real, excluded validation observation.
    measurements = [
        StageBatchMeasurement(
            key=key,
            target_batch=batch,
            engine_max_num_seqs=256,
            observed_resident_batch=batch,
            observed_total_service_s=(
                0.1 + 0.0001 * batch + 0.001 + 0.00001 * batch
                + overhead_delta_s
            ),
            reference_component_service_s=0.1 + 0.0001 * batch,
            overhead_s=0.001 + 0.00001 * batch + overhead_delta_s,
            sample_count=3,
        )
        for batch in (*ANCHOR_BATCHES, HELD_OUT_BATCH)
    ]
    return build_stage_batch_curve(
        key=key,
        measurements=measurements,
    ).to_dict()


def _stage_block(
    *,
    missing: str | None = None,
) -> dict[str, object]:
    return {
        "system_hardware": _SYSTEM_HARDWARE,
        "profiles": {
            stage: _stage_profile(stage)
            for stage in _REQUIRED_STAGES
            if stage != missing
        },
    }


def _config(path: Path, *, pipeline: bool = False) -> Path:
    document: dict[str, object] = {
        "system": {
            "system_hardware": _SYSTEM_HARDWARE,
            "cpu": _CPU_KEY,
            "gpu": _GPU_KEY,
            "cm_search_space": {"placement_policy": ["collocated"]},
            "available_gpus": [
                "cuda:0",
                "cuda:1",
                "cuda:2",
                "cuda:3",
            ],
        },
    }
    if pipeline:
        # The pipeline inventory will also report the intentionally absent
        # retrieval path.  This fixture targets its independently actionable
        # generator and scheduler-vllm items only.
        document.update({
            "dataset": {
                "dataset_name": "fixture",
                "qa": "fixture-qa.parquet",
                "corpus": "fixture-corpus.parquet",
            },
            "algo_search_space": {
                "vectordb": [],
                "pipeline": {
                    "node_lines": [{
                        "node_line_name": "generation",
                        "nodes": [{
                            "stage": "generator",
                            "modules": [{
                                "component": "vllm",
                                "model": _MODEL,
                                "bits": "bf16",
                            }],
                        }],
                    }],
                },
            },
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _write_component_profile(root: Path, document: dict[str, object]) -> None:
    path = root / "llm_sim" / f"{_GPU_KEY}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "profiles": document,
    }), encoding="utf-8")


@pytest.mark.parametrize(
    "field",
    ["component_calibration_store", "calibration_store"],
)
def test_inventory_rejects_retired_per_config_profile_store_fields(
    tmp_path: Path,
    field: str,
) -> None:
    config = _config(tmp_path / "invalid-store.yaml")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["system"]["rag_cm"] = {field: "/tmp/forbidden-profile-root"}
    config.write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported by cost_model"):
        inventory.build_calibration_inventory(
            config,
            data_dir=tmp_path / "data",
            profile_dir=tmp_path / "profiles",
            scopes=("generator",),
            generator_models=(_MODEL,),
        )


def _llmlingua2_inventory_profile() -> dict[str, object]:
    return {
        "profile_kind": "compressor_llmlingua2",
        "service_unit": "milliseconds_per_fused_compressor_call",
        "demand_axes": ["n_chunks", "chunk_tokens"],
        "model": "llmlingua2",
        "hardware_key": _GPU_KEY,
        "cpu_hardware_key": _CPU_KEY,
        "fixed_ms": 1.0,
        "per_chunk_ms": 2.0,
        "per_token_ms": 0.01,
        "fit_mape": 0.02,
        "correlation": 0.99,
        "grid": [{
            "n_chunks": 1,
            "chunk_tokens": 64,
            "measured_ms": 3.64,
            "predicted_ms": 3.64,
            "rel_err": 0.0,
        }],
        "note": "fixture",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("retired_version_marker", 2),
        ("evidence_scope", "standalone_component_only_no_rag_e2e_gt"),
        ("timestamp", "2026-07-16T00:00:00Z"),
        ("provenance", {"source_path": "/calibration/raw.jsonl"}),
    ],
)
def test_llmlingua2_inventory_rejects_unknown_contract_fields(
    field: str,
    value: object,
) -> None:
    profile = _llmlingua2_inventory_profile()
    profile[field] = value

    error = inventory.llmlingua2_profile_validation_error(
        profile,
        gpu_key=_GPU_KEY,
        cpu_key=_CPU_KEY,
    )

    assert error is not None
    assert "unknown LLMLingua2 profile fields" in error


def _write_stage_profile(root: Path, document: dict[str, object]) -> None:
    path = root / "stage_batch" / f"{_SYSTEM_HARDWARE}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def _stage_timing_evidence(
    *,
    timing_run_id: str,
    batch: int,
    input_tokens: int,
    output_tokens: int,
) -> stage_timing_contract.CollocatedEngineTimingEvidence:
    clock = stage_timing_contract.ClockDomainProof(
        domain=stage_timing_contract.CLOCK_DOMAIN,
        parent_response_source=(
            stage_timing_contract.PARENT_RESPONSE_CLOCK_SOURCE
        ),
        implementation="clock_gettime(CLOCK_MONOTONIC)",
        monotonic=True,
        adjustable=False,
        resolution_s=1e-9,
    )

    def interdeparture(
        *, output_count: int, first_endpoint_s: float, service_s: float,
    ) -> stage_timing_contract.ProductionInterdeparture:
        groups: list[list[dict[str, float]]] = []
        for group_index in range(
            stage_timing_contract.PRODUCTION_COHORT_COUNT
        ):
            endpoint_s = first_endpoint_s + group_index * service_s
            groups.append([
                {
                    "request_send_ts": (
                        first_endpoint_s - 10.0
                        + group_index * 0.001
                        + index * 1e-7
                    ),
                    "last_token_ts": (
                        endpoint_s - (batch - 1 - index) * 1e-7
                    ),
                }
                for index in range(batch)
            ])
        return stage_timing_contract.build_production_cohort_service_evidence(
            timing_run_id=timing_run_id,
            engine_batch_cap=batch,
            target_batch=batch,
            input_tokens=input_tokens,
            output_tokens=output_count,
            cohort_request_timings=groups,
            observed_dispatch_counts=[
                batch * (index + 1)
                for index in range(
                    stage_timing_contract.PRODUCTION_COHORT_COUNT
                )
            ],
            enginecore_fence_ack_count=(
                stage_timing_contract.PRODUCTION_COHORT_COUNT
            ),
            clock=clock,
        )

    out_one_diagnostic = interdeparture(
        output_count=1,
        first_endpoint_s=100.0,
        service_s=0.01,
    )
    full_engine = interdeparture(
        output_count=output_tokens,
        first_endpoint_s=200.0,
        service_s=0.01 * output_tokens,
    )
    return stage_timing_contract.build_collocated_engine_timing_evidence(
        full_engine,
        out_one_diagnostic=out_one_diagnostic,
    )


def _write_frozen_stage_cohort(
    root: Path,
    *,
    same_run_id: bool = False,
    drop_stage_cycle: bool = False,
    stage_shape_mismatch: bool = False,
    component_shape_mismatch: bool = False,
    target_batches: tuple[int, ...] = (1, 8, 32, 64, 128, 256),
) -> Path:
    """Write the four-file multi-run contract consumed by offline stage fitting."""

    from rag_stack.cost_model.llm_sim.calibration.main_llm.vllm_probe import (
        _OFFLINE_COMPONENT_ACQUISITION,
        _PRODUCTION_STAGE_ACQUISITION,
        _subrun_contract,
        _write_run_contract,
    )

    stage_raw = (
        root / "generator_stage_batch" / f"{_SYSTEM_HARDWARE}.jsonl"
    )
    component_raw = stage_raw.with_name(
        f"{_SYSTEM_HARDWARE}.component.jsonl"
    )
    stage_raw.parent.mkdir(parents=True, exist_ok=True)
    base_key = SimpleNamespace(
        model=_MODEL,
        device="cuda:0",
        max_num_seqs=256,
        max_num_batched_tokens=65536,
        enable_prefix_caching=False,
        dtype="bfloat16",
        kv_cache_dtype="auto",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        api_server_count=1,
        max_model_len=4096,
    )
    targets = list(target_batches)
    prompt_tokens = 10
    output_tokens = 2
    zero = {
        "scheduled_sequences": 0,
        "scheduled_tokens": 0,
        "scheduled_context_tokens": 0,
        "token_context_product": 0,
    }

    def shape(batch: int, index: int) -> dict[str, object]:
        phase = "prefill" if index == 0 else "decode"
        if phase == "prefill":
            active = {
                "scheduled_sequences": batch,
                "scheduled_tokens": batch * prompt_tokens,
                "scheduled_context_tokens": batch * prompt_tokens,
                "token_context_product": (
                    batch * prompt_tokens * prompt_tokens
                ),
            }
        else:
            context = batch * (prompt_tokens + index)
            active = {
                "scheduled_sequences": batch,
                "scheduled_tokens": batch,
                "scheduled_context_tokens": context,
                "token_context_product": context,
            }
        return {
            "terminal_performance_stage": phase,
            "prefill": active if phase == "prefill" else dict(zero),
            "decode": active if phase == "decode" else dict(zero),
        }

    component_records: list[dict[str, object]] = []
    stage_records: list[dict[str, object]] = []
    component_target_cycle_ids: dict[int, list[str]] = {}
    stage_target_cycle_ids: dict[int, list[str]] = {}
    stage_out_one_diagnostic_cycle_ids: dict[int, list[str] | None] = {}
    component_subruns: dict[int, dict[str, dict[str, object]]] = {}
    stage_subruns: dict[int, dict[str, dict[str, object]]] = {}
    timing_evidence: dict[
        int, stage_timing_contract.CollocatedEngineTimingEvidence
    ] = {}

    for target in targets:
        component_run_id = f"component-run-B{target}"
        proof_run_id = (
            component_run_id
            if same_run_id
            else f"stage-proof-run-B{target}"
        )
        timing_run_id = f"stage-timing-run-B{target}"
        batch_key = SimpleNamespace(
            **{**vars(base_key), "max_num_seqs": target}
        )
        component_ids: list[str] = []
        stage_ids: list[str] = []
        # Freeze five exact component cohorts per run.  The first two remain
        # in SHA-bound raw evidence as warmup; target_cycle_ids concatenate
        # every cycle from the final three measurement cohorts.
        labels = (
            "warmup-0", "warmup-1",
            "selected-0", "selected-1", "selected-2",
        )
        for cohort_index, label in enumerate(labels):
            for index in range(output_tokens):
                work_sequence = cohort_index * output_tokens + index
                component_id = (
                    f"{component_run_id}:{work_sequence}:{label}-{index}"
                )
                component_shape = shape(target, index)
                if (
                    component_shape_mismatch
                    and cohort_index == 2
                    and target == 8
                    and index == 1
                ):
                    component_shape["decode"]["scheduled_context_tokens"] += 1  # type: ignore[index,operator]
                    component_shape["decode"]["token_context_product"] += 1  # type: ignore[index,operator]
                component_records.append({
                    "schema": "rag_stack.vllm.genz_component_gpu_execution",
                    "run_id": component_run_id,
                    "cycle_id": component_id,
                    "work_sequence": work_sequence,
                    "cycle_sequence": work_sequence,
                    "world_size": 1,
                    "tensor_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                    "rank": 0,
                    "gpu_execution_s": (
                        0.4 + 0.4 * cohort_index
                        if cohort_index < 2
                        else 0.0039 + 0.0001 * (cohort_index - 2)
                    ),
                    "shape_digest": f"B{target}:step{index}",
                    "shape": component_shape,
                    "model_identity": {"model": _MODEL},
                    "publishable": True,
                    "reason": "publishable",
                })
                if cohort_index >= 2:
                    component_ids.append(component_id)

        for index in range(output_tokens):
            stage_id = f"{proof_run_id}:{index}"
            stage_shape = shape(target, index)
            if stage_shape_mismatch and target == 8 and index == 0:
                stage_shape["prefill"]["scheduled_tokens"] += 1  # type: ignore[index,operator]
            stage_records.append({
                "schema": "rag_stack.vllm.scheduler_cycle_stage_service",
                "run_id": proof_run_id,
                "cycle_id": stage_id,
                "work_sequence": index,
                "active_service_s": 0.01,
                "shape": stage_shape,
                "backlog_before_interval": {
                    "running_sequences": target if index else 0,
                    "waiting_sequences": 0 if index else target,
                },
                "model_identity": {"model": _MODEL},
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "publishable": True,
                "reason": "publishable",
            })
            stage_ids.append(stage_id)

        component_target_cycle_ids[target] = component_ids
        stage_target_cycle_ids[target] = stage_ids
        stage_out_one_diagnostic_cycle_ids[target] = [
            f"{proof_run_id}:out-one-diagnostic"
        ]
        component_subruns[target] = {
            "component": _subrun_contract(
                run_id=component_run_id,
                key=batch_key,
                served_max_model_len=4096,
            )
        }
        stage_subruns[target] = {
            "proof": _subrun_contract(
                run_id=proof_run_id,
                key=batch_key,
                served_max_model_len=4096,
            ),
            "timing": _subrun_contract(
                run_id=timing_run_id,
                key=batch_key,
                served_max_model_len=4096,
            ),
        }
        timing_evidence[target] = _stage_timing_evidence(
            timing_run_id=timing_run_id,
            batch=target,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

    if drop_stage_cycle:
        missing = stage_target_cycle_ids[8][1]
        stage_records = [
            record for record in stage_records
            if record["cycle_id"] != missing
        ]
    stage_raw.write_text(
        "".join(json.dumps(record) + "\n" for record in stage_records),
        encoding="utf-8",
    )
    component_raw.write_text(
        "".join(json.dumps(record) + "\n" for record in component_records),
        encoding="utf-8",
    )

    component_batch_proofs = {
        target: {
            "target_cohort_size": target,
            "observed_prompt_tokens": prompt_tokens,
            "observed_output_tokens": output_tokens,
            "exact_prefill_cycles": 1,
            "exact_decode_cycles": output_tokens - 1,
            "queued_proof_cohorts": 5,
            "discarded_warmup_cohorts": 2,
            "selected_measurement_cohorts": 3,
        }
        for target in targets
    }
    stage_batch_proofs = {
        target: {
            "target_cohort_size": target,
            "observed_prompt_tokens": prompt_tokens,
            "observed_output_tokens": output_tokens,
            "exact_prefill_cycles": 1,
            "exact_decode_cycles": output_tokens - 1,
            "engine_batch_cap": target,
            "queued_proof_cohorts": 4,
            "dispatch_spans": 4 * target,
            "discarded_warmup_cohorts": 2,
            "out_one_diagnostic_output_tokens": 1,
            "out_one_diagnostic_cycle_count": 1,
        }
        for target in targets
    }
    _write_run_contract(
        component_raw,
        scope="component_only",
        targets=targets,
        in_tok=prompt_tokens,
        out_tok=output_tokens,
        acquisition=_OFFLINE_COMPONENT_ACQUISITION,
        subruns=component_subruns,
        batch_proofs=component_batch_proofs,
        target_cycle_ids=component_target_cycle_ids,
        request_contracts=_request_contracts(prompt_tokens),
    )
    _write_run_contract(
        stage_raw,
        scope="stage_only",
        targets=targets,
        in_tok=prompt_tokens,
        out_tok=output_tokens,
        acquisition=_PRODUCTION_STAGE_ACQUISITION,
        subruns=stage_subruns,
        batch_proofs=stage_batch_proofs,
        target_cycle_ids=stage_target_cycle_ids,
        request_contracts=_request_contracts(prompt_tokens),
        out_one_diagnostic_cycle_ids=(
            stage_out_one_diagnostic_cycle_ids
        ),
        collocated_engine_timing_evidence=timing_evidence,
    )
    return stage_raw


def _inventory_item(result, name: str):
    matches = [item for item in result.items if item.name == name]
    assert len(matches) == 1, [item.name for item in result.items]
    return matches[0]


def _generator_inventory(tmp_path: Path, record: dict[str, object]):
    config = _config(tmp_path / "generator.yaml")
    profile_root = tmp_path / "profiles"
    _write_component_profile(
        profile_root,
        {f"agnostic__{_MODEL_KEY}__bf16": deepcopy(record)},
    )
    return inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator",),
        generator_models=(_MODEL,),
    )


def _generator_record_error(record: dict[str, object]) -> str | None:
    return inventory._llm_record_validation_error(
        record,
        component="agnostic",
        model_key=_MODEL_KEY,
        gpu_key=_GPU_KEY,
        bits="bf16",
    )


def test_generator_roofline_accepts_exact_identifiability_and_holdout_limits(
    tmp_path: Path,
) -> None:
    assert _generator_record_error(_roofline_record()) is None
    item = _inventory_item(
        _generator_inventory(tmp_path, _roofline_record()),
        "generator",
    )
    assert item.state == "ready"


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"profiles": {}, "extra": {}},
        {"legacy": 2},
    ],
)
def test_generator_inventory_rejects_non_exact_envelope_fields(
    tmp_path: Path,
    document: dict[str, object],
) -> None:
    config = _config(tmp_path / "generator.yaml")
    profile_root = tmp_path / "profiles"
    path = profile_root / "llm_sim" / f"{_GPU_KEY}.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(document), encoding="utf-8")

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator",),
        generator_models=(_MODEL,),
    )

    item = _inventory_item(result, "generator")
    assert item.state == "invalid"
    assert "envelope fields must be exactly" in item.detail


def test_generator_inventory_accepts_exact_key_without_raw_calibration_data(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "deliberately-absent-calibration-data"
    assert not data_root.exists()
    config = _config(tmp_path / "generator.yaml")
    profile_root = tmp_path / "profiles"
    _write_component_profile(profile_root, {
        f"agnostic__{_MODEL_KEY}__bf16": _roofline_record(),
    })

    result = inventory.build_calibration_inventory(
        config,
        data_dir=data_root,
        profile_dir=profile_root,
        scopes=("generator",),
        generator_models=(_MODEL,),
    )

    assert _inventory_item(result, "generator").state == "ready"
    assert not data_root.exists()


def test_generator_roofline_requires_zero_intercept_contract() -> None:
    record = _roofline_record()
    record.pop("calibration_contract")

    error = _generator_record_error(record)

    assert error is not None
    assert "standalone_ceff_meff_zero_intercept_v1" in error


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("measurement_evidence", {"manifest_path": "/calibration/raw.json"}),
        ("provenance", {"source_path": "/calibration/raw.jsonl"}),
        ("timestamp", "2026-07-16T00:00:00Z"),
        ("cuda_name", "NVIDIA H100"),
    ],
)
def test_generator_roofline_rejects_legacy_provenance_fields(
    field: str,
    value: object,
) -> None:
    record = _roofline_record()
    record[field] = value

    error = _generator_record_error(record)

    assert error is not None
    assert "unknown fields" in error
    assert field in error


def test_generator_roofline_requires_same_shape_b128_b256_pair() -> None:
    record = _roofline_record()
    for point in record["grid"]:
        if point["batch_size"] == 128 and point["seq_len"] == [64, 128]:
            point["seq_len"] = [65, 128]

    error = _generator_record_error(record)

    assert error is not None
    assert "same-token-shape B128 fit/B256 held-out pair" in error


def test_generator_component_readiness_remains_keyed_by_gpu_without_system_identity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "generator.yaml")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    del document["system"]["system_hardware"]
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    profile_root = tmp_path / "profiles"
    _write_component_profile(profile_root, {
        f"agnostic__{_MODEL_KEY}__bf16": _roofline_record(),
    })

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator",),
        generator_models=(_MODEL,),
    )

    assert result.system_hardware is None
    assert result.gpu_key == _GPU_KEY
    assert _inventory_item(result, "generator").state == "ready"


def test_generator_component_ignores_invalid_irrelevant_system_identity(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "generator.yaml")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["system"]["system_hardware"] = "not:path-safe"
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    profile_root = tmp_path / "profiles"
    _write_component_profile(profile_root, {
        f"agnostic__{_MODEL_KEY}__bf16": _roofline_record(),
    })

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator",),
        generator_models=(_MODEL,),
    )

    assert result.system_hardware is None
    assert _inventory_item(result, "generator").state == "ready"


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("identifiability_rank", 1, "identifiability_rank=2"),
        ("identifiability_condition", 100.000001, "in [1, 100]"),
    ],
)
def test_generator_roofline_requires_rank_two_and_bounded_condition(
    tmp_path: Path,
    field: str,
    value: object,
    detail: str,
) -> None:
    record = _roofline_record()
    record[field] = value

    error = _generator_record_error(record)
    assert error is not None
    assert detail in error
    item = _inventory_item(_generator_inventory(tmp_path, record), "generator")

    assert item.state == "invalid"


def test_generator_roofline_requires_independent_holdout_at_or_below_20_percent(
    tmp_path: Path,
) -> None:
    too_large = _roofline_record()
    too_large["validation_mape"] = 0.200001
    error = _generator_record_error(too_large)
    assert error is not None
    assert "validation_mape must be <= 0.20" in error
    item = _inventory_item(
        _generator_inventory(tmp_path / "too-large", too_large),
        "generator",
    )
    assert item.state == "invalid"

    overlapping = _roofline_record()
    overlapping["validation_grid"] = [
        deepcopy(overlapping["grid"][0]),  # type: ignore[index]
    ]
    error = _generator_record_error(overlapping)
    assert error is not None
    assert "training and held-out grids overlap" in error
    item = _inventory_item(
        _generator_inventory(tmp_path / "overlap", overlapping),
        "generator",
    )
    assert item.state == "invalid"


def test_generator_roofline_training_mape_must_be_at_or_below_20_percent(
    tmp_path: Path,
) -> None:
    # The accepted fixture above proves that exactly 20% remains publishable.
    record = _roofline_record()
    record["fit_mape"] = 0.200001

    error = _generator_record_error(record)
    assert error == "roofline record fit_mape must be <= 0.20"
    item = _inventory_item(_generator_inventory(tmp_path, record), "generator")
    assert item.state == "invalid"


def test_generator_roofline_rejects_nonzero_component_call_overhead(
    tmp_path: Path,
) -> None:
    record = _roofline_record()
    record["per_call_overhead_s"] = 1e-6

    error = _generator_record_error(record)
    assert error is not None
    assert "zero-intercept roofline record requires zero per-call overhead" in error
    item = _inventory_item(_generator_inventory(tmp_path, record), "generator")

    assert item.state == "invalid"
    assert item.commands
    assert all(
        "rag_stack calibrate generator --model"
        in command
        for command in item.commands
    )


@pytest.mark.parametrize(
    "legacy_blocks",
    [
        {},
        {
            "_tp_pp_efficiency": {"malformed": "must be ignored"},
            "_vllm_scheduler_step_residual": ["also", "ignored"],
        },
    ],
    ids=("legacy-absent", "legacy-malformed"),
)
def test_generator_tp_readiness_depends_only_on_two_collocated_stage_curves(
    tmp_path: Path,
    legacy_blocks: dict[str, object],
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    profile_root = tmp_path / "profiles"
    _write_stage_profile(
        profile_root, {"_stage_batch_curves": _stage_block()},
    )
    if legacy_blocks:
        _write_component_profile(profile_root, legacy_blocks)

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator-tp",),
        generator_models=(_MODEL,),
    )

    item = _inventory_item(result, "generator-tp")
    assert item.state == "ready"
    assert item.profile_path.endswith(
        f"stage_batch/{_SYSTEM_HARDWARE}.json"
    )
    assert item.frozen_path is not None
    assert item.frozen_path.endswith(
        f"generator_stage_batch/{_SYSTEM_HARDWARE}.jsonl"
    )
    assert result.system_hardware == _SYSTEM_HARDWARE
    assert result.to_dict()["hardware"]["system_hardware"] == _SYSTEM_HARDWARE
    assert result.deploy_ready


def test_generator_tp_partial_frozen_cohort_requires_forced_remeasurement(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    profile_root = tmp_path / "profiles"
    raw = (
        tmp_path
        / "data"
        / "generator_stage_batch"
        / f"{_SYSTEM_HARDWARE}.jsonl"
    )
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("{}\n", encoding="utf-8")

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator-tp",),
        # Stage calibration is system-level.  An explicitly empty component
        # model selection must not affect its readiness or command recipe.
        generator_models=(),
    )
    item = _inventory_item(result, "generator-tp")

    assert item.state == "needs_measure"
    assert item.frozen_path == str(raw)
    assert item.profile_path == str(
        profile_root / "stage_batch" / f"{_SYSTEM_HARDWARE}.json"
    )
    assert len(item.commands) == 1
    assert "calibrate generator-tp measure --config" in item.commands[0]
    assert item.commands[0].endswith(" --force")
    assert "absent/incomplete" in item.detail


def test_generator_tp_complete_frozen_cohort_is_ready_to_fit(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    profile_root = tmp_path / "profiles"
    raw = _write_frozen_stage_cohort(tmp_path / "data")

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator-tp",),
        generator_models=(),
    )
    item = _inventory_item(result, "generator-tp")

    assert item.state == "ready_to_fit"
    assert item.frozen_path == str(raw)
    assert len(item.commands) == 1
    assert "calibrate generator-tp fit --config" in item.commands[0]
    assert "verified independent stage/component frozen cohort" in item.detail

    component_contract = json.loads(
        raw.with_name(
            f"{_SYSTEM_HARDWARE}.component.jsonl.contract.json"
        ).read_text()
    )
    assert component_contract["scope"] == "component_only"
    for batch in ("128", "256"):
        assert component_contract["batch_proofs"][batch][
            "selected_measurement_cohorts"
        ] == 3
        target_ids = component_contract["target_cycle_ids"][batch]
        assert len(target_ids) == 3 * 2
        assert all(":selected-" in cycle_id for cycle_id in target_ids)


def test_generator_tp_ready_profile_ignores_unbound_raw_calibration_data(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    data_root = tmp_path / "data"
    profile_root = tmp_path / "profiles"
    raw = _write_frozen_stage_cohort(data_root)
    with raw.open("a", encoding="utf-8") as stream:
        stream.write('{"late": "uncontracted"}\n')
    _write_stage_profile(
        profile_root,
        {"_stage_batch_curves": _stage_block()},
    )

    result = inventory.build_calibration_inventory(
        config,
        data_dir=data_root,
        profile_dir=profile_root,
        scopes=("generator-tp",),
    )

    assert _inventory_item(result, "generator-tp").state == "ready"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provenance", {"source_path": "/calibration/stage.jsonl"}),
        ("provenance_sha256", "1" * 64),
    ],
)
def test_generator_tp_rejects_legacy_profile_provenance_fields(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    profile_root = tmp_path / "profiles"
    block = _stage_block()
    block["profiles"]["generator"][field] = value  # type: ignore[index]
    _write_stage_profile(
        profile_root,
        {"_stage_batch_curves": block},
    )

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator-tp",),
    )
    item = _inventory_item(result, "generator-tp")

    assert item.state == "invalid"
    assert "fields do not match the exact schema" in item.detail


def test_generator_tp_inventory_requires_matching_collocated_alias_data(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    profile_root = tmp_path / "profiles"
    alias_block = _stage_block()
    alias_block["profiles"]["query_expansion"] = _stage_profile(  # type: ignore[index]
        "query_expansion", overhead_delta_s=0.001,
    )
    _write_stage_profile(profile_root, {"_stage_batch_curves": alias_block})
    mismatched_alias = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator-tp",),
    )
    alias_item = _inventory_item(mismatched_alias, "generator-tp")
    assert alias_item.state == "invalid"
    assert "anchors/validation" in alias_item.detail


def test_generator_tp_mismatched_frozen_contract_requires_force(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    profile_root = tmp_path / "profiles"
    _write_frozen_stage_cohort(tmp_path / "data", same_run_id=True)

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator-tp",),
    )
    item = _inventory_item(result, "generator-tp")

    assert item.state == "needs_measure"
    assert len(item.commands) == 1
    assert item.commands[0].endswith(" --force")
    assert "independent runs" in item.detail


def test_generator_tp_frozen_raw_sha_mismatch_requires_force(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    profile_root = tmp_path / "profiles"
    raw = _write_frozen_stage_cohort(tmp_path / "data")
    with raw.open("a", encoding="utf-8") as stream:
        stream.write('{"late": "uncontracted"}\n')

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator-tp",),
    )
    item = _inventory_item(result, "generator-tp")

    assert item.state == "needs_measure"
    assert len(item.commands) == 1
    assert item.commands[0].endswith(" --force")
    assert "raw SHA256 disagrees" in item.detail


@pytest.mark.parametrize(
    ("fixture_kwargs", "detail"),
    [
        (
            {"drop_stage_cycle": True},
            "does not cover its frozen target cycle IDs",
        ),
        (
            {"stage_shape_mismatch": True},
            "stage frozen target shape mismatch",
        ),
        (
            {"component_shape_mismatch": True},
            "telemetry target window lost exact phase/batch/context",
        ),
        (
            {"target_batches": (1, 8, 32, 64, 128)},
            "must exactly cover anchors plus held-out validation",
        ),
    ],
    ids=(
        "missing-target-cycle",
        "stage-shape",
        "component-exact-cycle",
        "missing-b256",
    ),
)
def test_generator_tp_unfittable_raw_cohort_requires_force(
    tmp_path: Path,
    fixture_kwargs: dict[str, object],
    detail: str,
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    _write_frozen_stage_cohort(tmp_path / "data", **fixture_kwargs)

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=tmp_path / "profiles",
        scopes=("generator-tp",),
    )
    item = _inventory_item(result, "generator-tp")

    assert item.state == "needs_measure"
    assert len(item.commands) == 1
    assert item.commands[0].endswith(" --force")
    assert detail in item.detail


def test_generator_tp_ready_profile_does_not_require_gpu_acquisition_config(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    del document["system"]["gpu"]
    del document["system"]["available_gpus"]
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    profile_root = tmp_path / "profiles"
    _write_stage_profile(
        profile_root, {"_stage_batch_curves": _stage_block()},
    )

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator-tp",),
    )

    assert result.gpu_key is None
    assert result.gpu_count is None
    assert _inventory_item(result, "generator-tp").state == "ready"


def test_generator_tp_missing_profile_reports_gpu_acquisition_blocker(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    del document["system"]["gpu"]
    del document["system"]["available_gpus"]
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=tmp_path / "profiles",
        scopes=("generator-tp",),
    )
    item = _inventory_item(result, "generator-tp")

    assert item.state == "needs_measure"
    assert not item.commands
    assert "stage measurement is blocked" in item.detail
    assert "config.system.gpu" in item.detail
    assert "available_gpus" in item.detail


def test_generator_tp_rejects_extra_stage_profile_keys(tmp_path: Path) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    profile_root = tmp_path / "profiles"
    block = _stage_block()
    block["profiles"]["extra_stage"] = _stage_profile(  # type: ignore[index]
        "generator"
    )
    _write_stage_profile(profile_root, {"_stage_batch_curves": block})

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator-tp",),
        generator_models=(),
    )
    item = _inventory_item(result, "generator-tp")

    assert item.state == "invalid"
    assert "extra=['extra_stage']" in item.detail


def test_generator_tp_stage_profile_root_matches_runtime_exact_schema(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    profile_root = tmp_path / "profiles"
    _write_stage_profile(profile_root, {
        "_stage_batch_curves": _stage_block(),
        "unexpected_root_key": {},
    })

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator-tp",),
        generator_models=(_MODEL,),
    )
    item = _inventory_item(result, "generator-tp")

    assert item.state == "invalid"
    assert "must contain exactly '_stage_batch_curves'" in item.detail


def test_generator_tp_requires_system_hardware_identity(tmp_path: Path) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    del document["system"]["system_hardware"]
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="system_hardware"):
        inventory.build_calibration_inventory(
            config,
            data_dir=tmp_path / "data",
            profile_dir=tmp_path / "profiles",
            scopes=("generator-tp",),
            generator_models=(_MODEL,),
        )


def test_generator_tp_reuses_runtime_path_safe_system_validator(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["system"]["system_hardware"] = "not:path-safe"
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="system_hardware is not path-safe"):
        inventory.build_calibration_inventory(
            config,
            data_dir=tmp_path / "data",
            profile_dir=tmp_path / "profiles",
            scopes=("generator-tp",),
        )


@pytest.mark.parametrize("missing", _REQUIRED_STAGES)
def test_generator_tp_legacy_blocks_cannot_replace_a_missing_stage_curve(
    tmp_path: Path,
    missing: str,
) -> None:
    config = _config(tmp_path / "generator-tp.yaml")
    profile_root = tmp_path / "profiles"
    _write_stage_profile(profile_root, {
        "_stage_batch_curves": _stage_block(missing=missing),
    })
    _write_component_profile(profile_root, {
        "_tp_pp_efficiency": {"ready": True},
        "_vllm_scheduler_step_residual": {"ready": True},
    })

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("generator-tp",),
        generator_models=(_MODEL,),
    )
    item = _inventory_item(result, "generator-tp")

    assert item.state == "invalid"
    assert missing in item.detail
    assert item.commands
    assert all(
        "rag_stack calibrate generator-tp" in command
        for command in item.commands
    )
    assert not result.deploy_ready


def test_pipeline_scheduler_item_uses_the_same_stage_curve_contract(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "pipeline.yaml", pipeline=True)
    profile_root = tmp_path / "profiles"
    record_key = f"agnostic__{_MODEL_KEY}__bf16"
    component = {
        record_key: _roofline_record(),
        # Deliberately malformed retired blocks prove that the pipeline's
        # scheduler readiness cannot consult either of them.
        "_tp_pp_efficiency": None,
        "_vllm_scheduler_step_residual": "retired",
    }
    stage = {"_stage_batch_curves": _stage_block()}
    _write_component_profile(profile_root, component)
    _write_stage_profile(profile_root, stage)

    ready = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("pipeline",),
    )
    assert _inventory_item(ready, "generator").state == "ready"
    assert _inventory_item(ready, "scheduler-vllm").state == "ready"

    missing = deepcopy(stage)
    del missing["_stage_batch_curves"]["profiles"][  # type: ignore[index]
        "query_expansion"
    ]
    _write_stage_profile(profile_root, missing)
    invalid = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("pipeline",),
    )
    item = _inventory_item(invalid, "scheduler-vllm")
    assert item.state == "invalid"
    assert "query_expansion" in item.detail


def test_pipeline_disaggregated_search_requires_direct_pd_stage_curves(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "pipeline-disagg.yaml", pipeline=True)
    document = yaml.safe_load(config.read_text(encoding="utf-8"))
    document["system"]["cm_search_space"]["placement_policy"] = [
        "collocated",
        "disaggregated",
    ]
    config.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    profile_root = tmp_path / "profiles"
    _write_component_profile(
        profile_root,
        {f"agnostic__{_MODEL_KEY}__bf16": _roofline_record()},
    )
    _write_stage_profile(
        profile_root,
        {"_stage_batch_curves": _stage_block()},
    )

    result = inventory.build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("pipeline",),
    )
    item = _inventory_item(result, "scheduler-vllm")

    assert item.state == "needs_measure"
    assert "permits disaggregated placement" in item.detail
    assert len(item.commands) == 2
    assert "calibrate generator-disagg measure" in item.commands[0]
    assert "calibrate generator-disagg fit" in item.commands[1]
    assert not result.deploy_ready
