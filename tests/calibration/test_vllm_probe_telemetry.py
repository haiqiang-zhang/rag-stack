from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from rag_stack.cost_model.llm_sim.calibration.main_llm import vllm_probe
from rag_stack.cost_model.llm_sim.calibration.main_llm import (
    generator_request_contract,
)
from rag_stack.cost_model.llm_sim.calibration.main_llm import (
    offline_vllm_component,
)
from rag_stack.cost_model.llm_sim.calibration.main_llm import (
    stage_timing_contract,
)


def _phase_shape(
    phase: str,
    *,
    batch: int,
    tokens: int,
    context: int,
    product: int,
) -> dict:
    zero = {
        "scheduled_sequences": 0,
        "scheduled_tokens": 0,
        "scheduled_context_tokens": 0,
        "token_context_product": 0,
    }
    active = {
        "scheduled_sequences": batch,
        "scheduled_tokens": tokens,
        "scheduled_context_tokens": context,
        "token_context_product": product,
    }
    return {
        "terminal_performance_stage": phase,
        "prefill": active if phase == "prefill" else dict(zero),
        "decode": active if phase == "decode" else dict(zero),
    }


def _component_record(
    *,
    run_id: str,
    cycle_id: str,
    phase: str = "decode",
    batch: int = 2,
    tokens: int = 2,
    context: int = 20,
    product: int = 20,
    service_s: float = 0.004,
    model: str = "Qwen/Test",
    work_sequence: int = 0,
) -> dict:
    return {
        "schema": "rag_stack.vllm.genz_component_gpu_execution",
        "run_id": run_id,
        "cycle_id": cycle_id,
        "work_sequence": work_sequence,
        "world_size": 1,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "rank": 0,
        "gpu_execution_s": service_s,
        "shape_digest": f"{phase}:B{batch}:T{tokens}:C{context}:P{product}",
        "shape": _phase_shape(
            phase,
            batch=batch,
            tokens=tokens,
            context=context,
            product=product,
        ),
        "model_identity": {"model": model},
        "publishable": True,
        "reason": "publishable",
    }


def _stage_record(
    *,
    run_id: str,
    cycle_id: str,
    phase: str = "decode",
    batch: int = 2,
    tokens: int = 2,
    context: int = 20,
    product: int = 20,
    active_service_s: float = 0.01,
    model: str = "Qwen/Test",
    running: int | None = None,
    waiting: int | None = None,
    work_sequence: int = 0,
) -> dict:
    if running is None:
        running = batch if phase == "decode" else 0
    if waiting is None:
        waiting = 0 if phase == "decode" else batch
    return {
        "schema": "rag_stack.vllm.scheduler_cycle_stage_service",
        "run_id": run_id,
        "cycle_id": cycle_id,
        "work_sequence": work_sequence,
        "active_service_s": active_service_s,
        "shape": _phase_shape(
            phase,
            batch=batch,
            tokens=tokens,
            context=context,
            product=product,
        ),
        "backlog_before_interval": {
            "running_sequences": running,
            "waiting_sequences": waiting,
        },
        "model_identity": {"model": model},
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "publishable": True,
        "reason": "publishable",
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _write_offline_component_campaign(
    path: Path,
    *,
    shape: tuple[int, int, int],
    cohort_services_s: tuple[float, float, float, float, float],
    run_id: str = "offline-run",
) -> None:
    batch, input_tokens, output_tokens = shape
    records = []
    sequence = 0
    for cohort_service_s in cohort_services_s:
        per_cycle_service = float(cohort_service_s) / float(output_tokens)
        for step in range(output_tokens):
            phase = "prefill" if step == 0 else "decode"
            context = (
                batch * input_tokens
                if step == 0
                else batch * (input_tokens + step)
            )
            record = _component_record(
                run_id=run_id,
                cycle_id=f"{run_id}:{sequence}:digest-{sequence}",
                phase=phase,
                batch=batch,
                tokens=batch * input_tokens if step == 0 else batch,
                context=context,
                product=(
                    batch * input_tokens * input_tokens
                    if step == 0
                    else context
                ),
                service_s=per_cycle_service,
                work_sequence=sequence,
            )
            record["cycle_sequence"] = sequence
            records.append(record)
            sequence += 1
    _write_jsonl(path, records)


def _offline_component_child_result(
    *,
    shape: tuple[int, int, int],
    warmup_services_s: tuple[float, float],
    measurement_services_s: tuple[float, float, float],
    run_id: str = "offline-run",
) -> dict:
    return {
        "schema": offline_vllm_component.RESULT_SCHEMA,
        "run_id": run_id,
        "acquisition": "offline_inprocess_engine_full_cohort_enqueue",
        "component_boundary": (
            generator_request_contract.COMPONENT_SERVICE_BOUNDARY
        ),
        "request_contracts": _request_contracts(shape[1]),
        "warmup_cohorts": 2,
        "measurement_cohorts": 3,
        "total_cohorts": 5,
        "points": [{
            "batch": shape[0],
            "input_tokens": shape[1],
            "output_tokens": shape[2],
            "offline_cohort_qps_diagnostic": 100.0,
            "offline_cohort_wall_s_diagnostic": 0.1,
            "warmup_component_services_s": list(warmup_services_s),
            "measurement_component_services_s": list(
                measurement_services_s
            ),
            "component_service_s": sorted(measurement_services_s)[1],
        }],
    }


def _key(**overrides):
    from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
        VllmStartupKey,
    )

    values = {
        "model": "Qwen/Test",
        "device": "cuda:0",
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "enable_prefix_caching": False,
        "max_model_len": 4096,
        "kv_cache_dtype": "auto",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "api_server_count": 1,
        "gpu_memory_utilization": 0.85,
    }
    values.update(overrides)
    return VllmStartupKey(**values)


def _request_contract(prompt_tokens: int = 10) -> dict:
    template = "fixture-chat-template"
    token_ids = list(range(prompt_tokens))
    return {
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
        "temperature": generator_request_contract.CALIBRATION_TEMPERATURE,
        "seed": generator_request_contract.CALIBRATION_SEED,
        "stream": False,
        "ignore_eos": True,
        "ignore_eos_purpose": generator_request_contract.IGNORE_EOS_PURPOSE,
        "logprobs": True,
        "top_logprobs": 0,
        "n": 1,
        "max_tokens_policy": generator_request_contract.MAX_TOKENS_POLICY,
        "truncate_prompt_tokens_policy": (
            generator_request_contract.TRUNCATE_PROMPT_TOKENS_POLICY
        ),
        "truncation_side": generator_request_contract.TRUNCATION_SIDE,
    }


def _request_contracts(prompt_tokens: int = 10) -> dict:
    return {str(prompt_tokens): _request_contract(prompt_tokens)}


def test_generator_chat_request_contract_matches_measured_payload() -> None:
    contract = _request_contract()
    payload = generator_request_contract.generator_chat_payload(
        contract,
        model="Qwen/Test",
        output_tokens=4,
        served_max_model_len=4096,
    )

    assert payload == {
        "model": "Qwen/Test",
        "messages": [{"role": "user", "content": "fixture question"}],
        "temperature": 0.7,
        "seed": 0,
        "max_tokens": 4,
        "stream": False,
        "ignore_eos": True,
        "logprobs": True,
        "top_logprobs": 0,
        "n": 1,
        "truncate_prompt_tokens": 4092,
        "truncation_side": "right",
    }
    assert "prompt" not in payload
    assert contract["ignore_eos_purpose"] == (
        "calibration_exact_shape_control_not_production_stopping_policy"
    )


def test_generator_request_contract_rejects_raw_or_non_logprob_workload() -> None:
    for field, value in (
        ("request_format", "completions"),
        ("logprobs", False),
        ("top_logprobs", 1),
        ("temperature", 0.0),
        ("ignore_eos_purpose", "production_stopping_policy"),
    ):
        mutated = _request_contract()
        mutated[field] = value
        with pytest.raises(ValueError, match="generator request"):
            generator_request_contract.validate_generator_request_contract(
                mutated
            )

    legacy_dual_expression = _request_contract()
    legacy_dual_expression["offline_sampling_logprobs"] = (
        legacy_dual_expression.pop("top_logprobs")
    )
    with pytest.raises(ValueError, match="fields"):
        generator_request_contract.validate_generator_request_contract(
            legacy_dual_expression
        )


def test_offline_component_flush_binds_exact_cohort_sequences() -> None:
    class FakeLlm:
        calls: list[str] = []

        def collective_rpc(self, method):
            self.calls.append(method)
            return [{
                "flushed_count": 3,
                "work_sequences": [7, 8, 9],
                "pending_count": 0,
            }]

    llm = FakeLlm()
    offline_vllm_component._flush_component_telemetry(
        llm,
        expected_work_sequences=range(7, 10),
    )
    assert llm.calls == ["flush_component_telemetry"]


@pytest.mark.parametrize(
    "reply",
    (
        {
            "flushed_count": 2,
            "work_sequences": [7, 8, 9],
            "pending_count": 0,
        },
        {
            "flushed_count": 3,
            "work_sequences": [7, 9, 10],
            "pending_count": 0,
        },
        {
            "flushed_count": 3,
            "work_sequences": [7, 8, 9],
            "pending_count": 1,
        },
    ),
)
def test_offline_component_flush_rejects_partial_or_wrong_cohort(reply) -> None:
    class FakeLlm:
        @staticmethod
        def collective_rpc(_method):
            return [reply]

    with pytest.raises(RuntimeError, match="exact cohort cycles"):
        offline_vllm_component._flush_component_telemetry(
            FakeLlm(),
            expected_work_sequences=range(7, 10),
        )


def test_offline_component_renders_exact_contracted_chat_template() -> None:
    contract = _request_contract()

    class FakeTokenizer:
        chat_template = "fixture-chat-template"

        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt):
            assert messages == contract["messages"]
            assert tokenize is True
            assert add_generation_prompt is True
            return list(range(10))

    assert generator_request_contract.render_generator_chat_prompt_token_ids(
        FakeTokenizer(),
        contract,
    ) == list(range(10))


def test_offline_component_spec_requires_request_contracts() -> None:
    spec = {
        "schema": offline_vllm_component.SPEC_SCHEMA,
        "startup": {
            "model": "Qwen/Test",
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.9,
            "max_model_len": 4096,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 4096,
            "kv_cache_dtype": "auto",
            "enable_prefix_caching": False,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
        },
        "shapes": [[2, 10, 2]],
        "request_contracts": _request_contracts(),
        "warmup_cohorts": 2,
        "measurement_cohorts": 3,
    }

    assert offline_vllm_component.validate_spec(spec)["request_contracts"] == (
        _request_contracts()
    )
    extra = dict(spec, unexpected=True)
    with pytest.raises(ValueError, match="exact schema"):
        offline_vllm_component.validate_spec(extra)
    missing = dict(spec)
    missing.pop("request_contracts")
    with pytest.raises(ValueError, match="exact schema"):
        offline_vllm_component.validate_spec(missing)


def _component_proof(batch: int, *, out_tok: int) -> dict[str, int]:
    return {
        "target_cohort_size": batch,
        "observed_prompt_tokens": 10,
        "observed_output_tokens": out_tok,
        "exact_prefill_cycles": 1,
        "exact_decode_cycles": out_tok - 1,
        "queued_proof_cohorts": 5,
        "discarded_warmup_cohorts": 2,
        "selected_measurement_cohorts": 3,
    }


def _stage_proof(batch: int, *, out_tok: int) -> dict[str, int]:
    return {
        "target_cohort_size": batch,
        "observed_prompt_tokens": 10,
        "observed_output_tokens": out_tok,
        "exact_prefill_cycles": 1,
        "exact_decode_cycles": out_tok - 1,
        "engine_batch_cap": batch,
        "queued_proof_cohorts": 4,
        "dispatch_spans": 4 * batch,
        "discarded_warmup_cohorts": 2,
        "out_one_diagnostic_output_tokens": 1,
        "out_one_diagnostic_cycle_count": 1,
    }


def _clock_proof() -> stage_timing_contract.ClockDomainProof:
    return stage_timing_contract.ClockDomainProof(
        domain=stage_timing_contract.CLOCK_DOMAIN,
        parent_response_source=(
            stage_timing_contract.PARENT_RESPONSE_CLOCK_SOURCE
        ),
        implementation="clock_gettime(CLOCK_MONOTONIC)",
        monotonic=True,
        adjustable=False,
        resolution_s=1e-9,
    )


def _interdeparture(
    *,
    timing_run_id: str,
    batch: int,
    input_tokens: int,
    output_tokens: int,
    first_endpoint_s: float,
    service_s: float,
) -> stage_timing_contract.ProductionInterdeparture:
    groups: list[list[dict[str, float]]] = []
    send_origin_s = first_endpoint_s - 10.0
    for group_index in range(
        stage_timing_contract.PRODUCTION_COHORT_COUNT
    ):
        endpoint_s = first_endpoint_s + group_index * service_s
        groups.append([
            {
                "request_send_ts": (
                    send_origin_s + group_index * 0.001 + index * 1e-7
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
        output_tokens=output_tokens,
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
        clock=_clock_proof(),
    )


def _timing_evidence(
    *,
    timing_run_id: str,
    batch: int,
    input_tokens: int,
    output_tokens: int,
    full_engine_service_s: float = 0.025,
    out_one_diagnostic_service_s: float = 0.01,
) -> stage_timing_contract.CollocatedEngineTimingEvidence:
    diagnostic = _interdeparture(
        timing_run_id=timing_run_id,
        batch=batch,
        input_tokens=input_tokens,
        output_tokens=1,
        first_endpoint_s=100.0,
        service_s=out_one_diagnostic_service_s,
    )
    full = _interdeparture(
        timing_run_id=timing_run_id,
        batch=batch,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        first_endpoint_s=200.0,
        service_s=full_engine_service_s,
    )
    return stage_timing_contract.build_collocated_engine_timing_evidence(
        full,
        out_one_diagnostic=diagnostic,
    )


def _write_contract_pair(
    component_path: Path,
    stage_path: Path,
    *,
    component_run: str,
    stage_run: str,
    component_key=None,
    stage_key=None,
    stage_shape_mismatch: bool = False,
    component_measurement_scales: tuple[float, float, float] = (
        0.98, 1.0, 1.02,
    ),
    component_measurement_prefill_services_s: (
        tuple[float, float, float] | None
    ) = None,
) -> None:
    # Component run-contract v7 keeps five consecutive exact cohorts in the
    # SHA-bound raw stream, discards the first two, and freezes all cycles from
    # the final three measurement cohorts.
    source_component_rows = [
        json.loads(line) for line in component_path.read_text().splitlines()
    ]
    stage_rows = [
        json.loads(line) for line in stage_path.read_text().splitlines()
    ]
    assert len(source_component_rows) == len(stage_rows)
    out_tok = len(stage_rows)
    assert out_tok >= 2

    # Normalize these compact matching fixtures to the contracted B2/in10
    # prefill followed by exact decode contexts.
    for index, (component_row, stage_row) in enumerate(zip(
        source_component_rows, stage_rows, strict=True,
    )):
        phase = "prefill" if index == 0 else "decode"
        tokens = 20 if index == 0 else 2
        context = 20 if index == 0 else 2 * (10 + index)
        product = 200 if index == 0 else context
        component_row["shape"] = _phase_shape(
            phase,
            batch=2,
            tokens=tokens,
            context=context,
            product=product,
        )
        component_row["shape_digest"] = (
            f"{phase}:B2:T{tokens}:C{context}:P{product}"
        )
        stage_row["shape"] = _phase_shape(
            phase,
            batch=2,
            tokens=tokens,
            context=context,
            product=product,
        )
        stage_row["work_sequence"] = index
        stage_row["backlog_before_interval"] = {
            "running_sequences": 0 if index == 0 else 2,
            "waiting_sequences": 2 if index == 0 else 0,
        }
    if stage_shape_mismatch:
        stage_rows[1]["shape"]["decode"]["scheduled_context_tokens"] += 1
        stage_rows[1]["shape"]["decode"]["token_context_product"] += 1
    _write_jsonl(stage_path, stage_rows)
    stage_ids = [row["cycle_id"] for row in stage_rows]

    component_rows = []
    component_ids = []
    for cohort_index in range(5):
        for index, source in enumerate(source_component_rows):
            row = dict(source)
            work_sequence = cohort_index * out_tok + index
            row["cycle_id"] = (
                f"{component_run}:{work_sequence}:"
                f"cohort-{cohort_index}-{index}"
            )
            row["work_sequence"] = work_sequence
            row["cycle_sequence"] = work_sequence
            if cohort_index < 2:
                row["gpu_execution_s"] = 0.4 + 0.4 * cohort_index
            else:
                measurement_index = cohort_index - 2
                if (
                    index == 0
                    and component_measurement_prefill_services_s is not None
                ):
                    row["gpu_execution_s"] = (
                        component_measurement_prefill_services_s[
                            measurement_index
                        ]
                    )
                else:
                    row["gpu_execution_s"] = (
                        float(source["gpu_execution_s"])
                        * component_measurement_scales[measurement_index]
                    )
                component_ids.append(row["cycle_id"])
            component_rows.append(row)
    _write_jsonl(component_path, component_rows)
    out_one_diagnostic_id = f"{stage_run}:out-one-diagnostic:B2"
    timing_run = f"{stage_run}:timing"
    component_key = component_key or _key(max_num_seqs=2)
    stage_key = stage_key or _key(max_num_seqs=2)
    timing_evidence = _timing_evidence(
        timing_run_id=timing_run,
        batch=2,
        input_tokens=10,
        output_tokens=out_tok,
    )
    vllm_probe._write_run_contract(
        component_path,
        scope="component_only",
        targets=[2],
        in_tok=10,
        out_tok=out_tok,
        acquisition=vllm_probe._OFFLINE_COMPONENT_ACQUISITION,
        subruns={
            2: {
                "component": vllm_probe._subrun_contract(
                    run_id=component_run,
                    key=component_key,
                    served_max_model_len=4096,
                )
            }
        },
        batch_proofs={2: _component_proof(2, out_tok=out_tok)},
        target_cycle_ids={2: component_ids},
        request_contracts=_request_contracts(),
    )
    vllm_probe._write_run_contract(
        stage_path,
        scope="stage_only",
        targets=[2],
        in_tok=10,
        out_tok=out_tok,
        acquisition=vllm_probe._PRODUCTION_STAGE_ACQUISITION,
        subruns={
            2: {
                "proof": vllm_probe._subrun_contract(
                    run_id=stage_run,
                    key=stage_key,
                    served_max_model_len=4096,
                ),
                "timing": vllm_probe._subrun_contract(
                    run_id=timing_run,
                    key=stage_key,
                    served_max_model_len=4096,
                ),
            }
        },
        batch_proofs={2: _stage_proof(2, out_tok=out_tok)},
        target_cycle_ids={2: stage_ids},
        request_contracts=_request_contracts(),
        out_one_diagnostic_cycle_ids={2: [out_one_diagnostic_id]},
        collocated_engine_timing_evidence={2: timing_evidence},
    )


def test_independent_runs_match_component_by_exact_phase_shape(tmp_path):
    component_path = tmp_path / "component.jsonl"
    stage_path = tmp_path / "stage.jsonl"
    _write_jsonl(component_path, [
        _component_record(
            run_id="component-run", cycle_id="component-run:c0",
            service_s=0.004, work_sequence=0,
        ),
        _component_record(
            run_id="component-run", cycle_id="component-run:c1",
            service_s=0.006, work_sequence=1,
        ),
    ])
    _write_jsonl(stage_path, [
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:s0", work_sequence=0,
        ),
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:s1", work_sequence=1,
        ),
    ])
    _write_contract_pair(
        component_path,
        stage_path,
        component_run="component-run",
        stage_run="stage-run",
    )

    observations = vllm_probe.read_stage_cycle_observations(stage_path)
    services = vllm_probe.read_component_cycle_services(
        component_path,
        observations,
        stage_telemetry_path=stage_path,
    )

    assert services == {
        ("stage-run", "stage-run:s0"): pytest.approx(0.004),
        ("stage-run", "stage-run:s1"): pytest.approx(0.006),
    }
    component_contract = json.loads(
        vllm_probe._run_contract_path(component_path).read_text()
    )
    stage_contract = json.loads(
        vllm_probe._run_contract_path(stage_path).read_text()
    )
    component_subrun = component_contract["subruns"]["2"]["component"]
    proof_subrun = stage_contract["subruns"]["2"]["proof"]
    timing_subrun = stage_contract["subruns"]["2"]["timing"]
    assert len({
        component_subrun["run_id"],
        proof_subrun["run_id"],
        timing_subrun["run_id"],
    }) == 3
    assert (
        component_subrun["startup"]
        == proof_subrun["startup"]
        == timing_subrun["startup"]
    )
    assert component_subrun["startup"]["max_num_seqs"] == 2
    assert "run_id" not in component_contract
    assert "startup" not in component_contract
    assert vllm_probe._normalized_contract_workload(
        component_contract
    ) == vllm_probe._normalized_contract_workload(stage_contract)


def test_component_stage_join_rejects_different_chat_request_contracts(tmp_path):
    component_path = tmp_path / "component.jsonl"
    stage_path = tmp_path / "stage.jsonl"
    _write_jsonl(component_path, [
        _component_record(
            run_id="component-run", cycle_id="component-run:c0",
            service_s=0.004, work_sequence=0,
        ),
        _component_record(
            run_id="component-run", cycle_id="component-run:c1",
            service_s=0.006, work_sequence=1,
        ),
    ])
    _write_jsonl(stage_path, [
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:s0", work_sequence=0,
        ),
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:s1", work_sequence=1,
        ),
    ])
    _write_contract_pair(
        component_path,
        stage_path,
        component_run="component-run",
        stage_run="stage-run",
    )
    stage_contract_path = vllm_probe._run_contract_path(stage_path)
    stage_contract = json.loads(stage_contract_path.read_text())
    changed = _request_contract()
    changed["messages"] = [{"role": "user", "content": "different fixture"}]
    changed["rendered_prompt_token_ids_sha256"] = "f" * 64
    stage_contract["workload"]["request_contracts"] = {"10": changed}
    stage_contract_path.write_text(
        json.dumps(stage_contract, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    observations = vllm_probe.read_stage_cycle_observations(stage_path)
    with pytest.raises(ValueError, match="workload contracts differ"):
        vllm_probe.read_component_cycle_services(
            component_path,
            observations,
            stage_telemetry_path=stage_path,
        )


def test_run_contract_fails_closed_without_request_contract_provenance(
    tmp_path,
) -> None:
    component_path = tmp_path / "component.jsonl"
    stage_path = tmp_path / "stage.jsonl"
    _write_jsonl(component_path, [
        _component_record(
            run_id="component-run", cycle_id="component-run:c0",
            service_s=0.004, work_sequence=0,
        ),
        _component_record(
            run_id="component-run", cycle_id="component-run:c1",
            service_s=0.006, work_sequence=1,
        ),
    ])
    _write_jsonl(stage_path, [
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:s0", work_sequence=0,
        ),
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:s1", work_sequence=1,
        ),
    ])
    _write_contract_pair(
        component_path,
        stage_path,
        component_run="component-run",
        stage_run="stage-run",
    )
    contract_path = vllm_probe._run_contract_path(stage_path)
    contract = json.loads(contract_path.read_text())
    contract["workload"].pop("request_contracts")
    contract_path.write_text(
        json.dumps(contract, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="workload is invalid"):
        vllm_probe.read_stage_cycle_observations(stage_path)


def test_offline_match_rejects_startup_or_exact_shape_mismatch(tmp_path):
    component_path = tmp_path / "component.jsonl"
    stage_path = tmp_path / "stage.jsonl"
    _write_jsonl(component_path, [
        _component_record(
            run_id="component-run", cycle_id="component-run:c0", tokens=3,
            work_sequence=0,
        ),
        _component_record(
            run_id="component-run", cycle_id="component-run:c1", tokens=3,
            work_sequence=1,
        ),
    ])
    _write_jsonl(stage_path, [
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:s0", tokens=2,
            work_sequence=0,
        ),
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:s1", tokens=2,
            work_sequence=1,
        ),
    ])
    _write_contract_pair(
        component_path,
        stage_path,
        component_run="component-run",
        stage_run="stage-run",
        stage_key=_key(max_num_seqs=2, max_num_batched_tokens=4096),
        stage_shape_mismatch=True,
    )
    observations = vllm_probe.read_stage_cycle_observations(stage_path)

    with pytest.raises(ValueError, match="startup contracts differ"):
        vllm_probe.read_component_cycle_services(
            component_path,
            observations,
            stage_telemetry_path=stage_path,
        )

    # Make startup contracts comparable; the exact token shape must still fail.
    stage_ids = ["stage-run:s0", "stage-run:s1"]
    out_one_diagnostic_id = "stage-run:out-one-diagnostic:B2"
    stage_key = _key(max_num_seqs=2)
    timing_run = "stage-run:timing"
    vllm_probe._write_run_contract(
        stage_path,
        scope="stage_only",
        targets=[2],
        in_tok=10,
        out_tok=2,
        acquisition=vllm_probe._PRODUCTION_STAGE_ACQUISITION,
        subruns={
            2: {
                role: vllm_probe._subrun_contract(
                    run_id=run_id,
                    key=stage_key,
                    served_max_model_len=4096,
                )
                for role, run_id in (
                    ("proof", "stage-run"),
                    ("timing", timing_run),
                )
            }
        },
        batch_proofs={2: _stage_proof(2, out_tok=2)},
        target_cycle_ids={2: stage_ids},
        request_contracts=_request_contracts(),
        out_one_diagnostic_cycle_ids={2: [out_one_diagnostic_id]},
        collocated_engine_timing_evidence={
            2: _timing_evidence(
                timing_run_id=timing_run,
                batch=2,
                input_tokens=10,
                output_tokens=2,
            )
        },
    )
    with pytest.raises(ValueError, match="no exact phase-shape match"):
        vllm_probe.read_component_cycle_services(
            component_path,
            observations,
            stage_telemetry_path=stage_path,
        )


def test_component_parser_rejects_mixed_phase(tmp_path):
    mixed = _component_record(run_id="component-run", cycle_id="mixed")
    mixed["publishable"] = False
    mixed["reason"] = "mixed_prefill_decode_not_component_identifiable"
    mixed["shape"]["prefill"] = {
        "scheduled_sequences": 1,
        "scheduled_tokens": 8,
        "scheduled_context_tokens": 8,
        "token_context_product": 64,
    }
    path = tmp_path / "component.jsonl"
    _write_jsonl(path, [mixed])

    with pytest.raises(ValueError, match="mixed prefill/decode"):
        vllm_probe.read_component_cycle_services(path)


def test_exact_component_cohort_segment_sums_complete_phase_sequence(tmp_path):
    records = [
        _component_record(
            run_id="component-run", cycle_id="prefill", phase="prefill",
            batch=2, tokens=20, context=20, product=200,
            service_s=0.01,
        ),
    ]
    records.extend(
        _component_record(
            run_id="component-run", cycle_id=f"decode-{index}",
            phase="decode", batch=2, tokens=2,
            context=22 + index * 2, product=22 + index * 2,
            service_s=0.002,
        )
        for index in range(3)
    )
    path = tmp_path / "exact-cohort.jsonl"
    _write_jsonl(path, records)

    assert vllm_probe._exact_component_cohort_service_from_segment(
        path.read_bytes(),
        target_batch=2,
        input_tokens=10,
        output_tokens=4,
    ) == pytest.approx(0.016)


def test_offline_component_parent_excludes_two_cold_warmup_cohorts(tmp_path):
    path = tmp_path / "offline-component.jsonl"
    shape = (2, 10, 2)
    warmup = (20.0, 10.0)
    measurement = (1.0, 1.02, 1.04)
    _write_offline_component_campaign(
        path,
        shape=shape,
        cohort_services_s=warmup + measurement,
    )
    result = _offline_component_child_result(
        shape=shape,
        warmup_services_s=warmup,
        measurement_services_s=measurement,
    )

    output = vllm_probe._validate_offline_component_result(
        result,
        result_schema=offline_vllm_component.RESULT_SCHEMA,
        expected_run_id="offline-run",
        requested_shapes=[shape],
        telemetry_path=path,
        expected_request_contracts=_request_contracts(shape[1]),
    )

    assert output[shape][2] == pytest.approx(1.02)
    assert output[shape][2] != pytest.approx(
        sorted(warmup + measurement)[2]
    )
    raw = vllm_probe._audit_offline_component_raw(
        path, requested_shapes=[shape], expected_run_id="offline-run",
    )[shape]
    assert raw.warmup_component_services_s == pytest.approx(warmup)
    assert raw.measurement_component_services_s == pytest.approx(measurement)
    assert raw.measurement_relative_spread == pytest.approx(
        (1.04 - 1.0) / 1.02
    )


def test_offline_component_parent_rejects_child_raw_service_mismatch(tmp_path):
    path = tmp_path / "offline-component-mismatch.jsonl"
    shape = (2, 10, 2)
    warmup = (20.0, 10.0)
    measurement = (1.0, 1.02, 1.04)
    _write_offline_component_campaign(
        path,
        shape=shape,
        cohort_services_s=warmup + measurement,
    )
    result = _offline_component_child_result(
        shape=shape,
        warmup_services_s=warmup,
        measurement_services_s=measurement,
    )
    result["points"][0]["measurement_component_services_s"][1] = 1.01

    with pytest.raises(ValueError, match="disagrees with parent raw"):
        vllm_probe._validate_offline_component_result(
            result,
            result_schema=offline_vllm_component.RESULT_SCHEMA,
            expected_run_id="offline-run",
            requested_shapes=[shape],
            telemetry_path=path,
            expected_request_contracts=_request_contracts(shape[1]),
        )


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "shuffled", "foreign_run"],
)
def test_offline_component_parent_rejects_invalid_raw_cohort_campaign(
    tmp_path, mutation,
):
    path = tmp_path / f"offline-component-{mutation}.jsonl"
    shape = (2, 10, 2)
    _write_offline_component_campaign(
        path,
        shape=shape,
        cohort_services_s=(20.0, 10.0, 1.0, 1.02, 1.04),
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "missing":
        records.pop()
    elif mutation == "extra":
        records.append(dict(records[-1]))
    elif mutation == "shuffled":
        records[4], records[5] = records[5], records[4]
    else:
        records[5]["run_id"] = "foreign-run"
    _write_jsonl(path, records)

    with pytest.raises(ValueError):
        vllm_probe._audit_offline_component_raw(
            path, requested_shapes=[shape], expected_run_id="offline-run",
        )


def test_offline_component_parent_rejects_unstable_selected_totals(tmp_path):
    path = tmp_path / "offline-component-unstable.jsonl"
    shape = (2, 10, 2)
    # Two selected cohorts remain 20% slower than the first; their median
    # cannot hide that the total-service campaign is not reproducible.
    _write_offline_component_campaign(
        path,
        shape=shape,
        cohort_services_s=(20.0, 10.0, 1.0, 1.2, 1.2),
    )

    with pytest.raises(ValueError, match="selected cohort totals are unstable"):
        vllm_probe._audit_offline_component_raw(
            path, requested_shapes=[shape], expected_run_id="offline-run",
        )


def test_component_stability_accepts_one_decode_outlier_with_stable_prefill(
    tmp_path,
):
    component_path = tmp_path / "component-stability.jsonl"
    stage_path = tmp_path / "stage-stability.jsonl"
    component_records = []
    cohort_cycle_ids = []
    for cohort_index in range(3):
        ids = []
        for step in range(4):
            phase = "prefill" if step == 0 else "decode"
            tokens = 20 if step == 0 else 2
            context = 20 if step == 0 else 2 * (10 + step)
            product = 200 if step == 0 else context
            service_s = 100.0 if step == 0 else 1.0
            if cohort_index == 1 and step == 2:
                service_s = 1.82
            cycle_id = f"component-run:c{cohort_index}:step{step}"
            ids.append(cycle_id)
            component_records.append(_component_record(
                run_id="component-run",
                cycle_id=cycle_id,
                phase=phase,
                batch=2,
                tokens=tokens,
                context=context,
                product=product,
                service_s=service_s,
                work_sequence=cohort_index * 4 + step,
            ))
        cohort_cycle_ids.append(ids)
    _write_jsonl(component_path, component_records)

    total_spread, prefill_spread = (
        vllm_probe._component_cohort_relative_spreads(
            component_records, cohort_cycle_ids,
        )
    )
    spread = vllm_probe._component_cohort_max_relative_spread(
        component_records, cohort_cycle_ids,
    )
    assert total_spread == pytest.approx(0.82 / 103.0)
    assert prefill_spread == 0.0
    assert spread == total_spread
    assert spread < vllm_probe._COMPONENT_MAX_RELATIVE_SPREAD

    stage_records = []
    for step in range(4):
        phase = "prefill" if step == 0 else "decode"
        tokens = 20 if step == 0 else 2
        context = 20 if step == 0 else 2 * (10 + step)
        product = 200 if step == 0 else context
        stage_records.append(_stage_record(
            run_id="stage-run",
            cycle_id=f"stage-run:step{step}",
            phase=phase,
            batch=2,
            tokens=tokens,
            context=context,
            product=product,
            work_sequence=step,
        ))
    _write_jsonl(stage_path, stage_records)
    observations = vllm_probe.read_stage_cycle_observations(stage_path)
    parsed_components = vllm_probe._read_component_cycle_records(
        component_path,
    )
    joined = vllm_probe._match_stage_component_services(
        observations, parsed_components,
    )
    # The isolated 82% sample is rejected by neither the cohort-total gate nor
    # the robust per-shape median consumed by the stage/component join.
    assert joined[("stage-run", "stage-run:step2")] == pytest.approx(1.0)


def test_full_cohort_contract_does_not_gate_on_retired_prefill_split(tmp_path):
    component_path = tmp_path / "component-prefill-unstable.jsonl"
    stage_path = tmp_path / "stage-prefill-unstable.jsonl"
    _write_jsonl(component_path, [
        _component_record(
            run_id="component-run", cycle_id="component-run:prefill",
            phase="prefill", service_s=0.03, work_sequence=0,
        ),
        _component_record(
            run_id="component-run", cycle_id="component-run:decode",
            service_s=1.0, work_sequence=1,
        ),
    ])
    _write_jsonl(stage_path, [
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:prefill",
            phase="prefill", work_sequence=0,
        ),
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:decode",
            work_sequence=1,
        ),
    ])
    _write_contract_pair(
        component_path,
        stage_path,
        component_run="component-run",
        stage_run="stage-run",
        component_measurement_scales=(1.0, 1.0, 1.0),
        component_measurement_prefill_services_s=(0.03, 0.03, 0.01),
    )

    contract = vllm_probe._load_run_contract(
        component_path, expected_scope="component_only",
    )
    assert contract["target_cycle_ids"]["2"]


def test_component_contract_rejects_cold_full_measurement_cohort(tmp_path):
    component_path = tmp_path / "component-cold.jsonl"
    stage_path = tmp_path / "stage-cold.jsonl"
    _write_jsonl(component_path, [
        _component_record(
            run_id="component-run", cycle_id="component-run:source0",
            service_s=0.004, work_sequence=0,
        ),
        _component_record(
            run_id="component-run", cycle_id="component-run:source1",
            service_s=0.006, work_sequence=1,
        ),
    ])
    _write_jsonl(stage_path, [
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:s0", work_sequence=0,
        ),
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:s1", work_sequence=1,
        ),
    ])
    _write_contract_pair(
        component_path,
        stage_path,
        component_run="component-run",
        stage_run="stage-run",
        component_measurement_scales=(1.0, 1.0, 1.2),
    )

    with pytest.raises(ValueError, match="warmed cohort totals are unstable"):
        vllm_probe._load_run_contract(
            component_path, expected_scope="component_only",
        )


@pytest.mark.parametrize(
    ("mutation", "detail"),
    (
        ("physical_order", "physical work sequence"),
        ("cycle_sequence", "cycle_sequence disagrees"),
        ("cycle_id", "cycle_id is not bound"),
    ),
)
def test_component_contract_binds_physical_sequence_and_cycle_identity(
    tmp_path, mutation, detail,
):
    component_path = tmp_path / f"component-{mutation}.jsonl"
    stage_path = tmp_path / f"stage-{mutation}.jsonl"
    _write_jsonl(component_path, [
        _component_record(
            run_id="component-run", cycle_id="source-prefill",
            phase="prefill", service_s=0.004, work_sequence=0,
        ),
        _component_record(
            run_id="component-run", cycle_id="source-decode",
            service_s=0.006, work_sequence=1,
        ),
    ])
    _write_jsonl(stage_path, [
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:s0", work_sequence=0,
        ),
        _stage_record(
            run_id="stage-run", cycle_id="stage-run:s1", work_sequence=1,
        ),
    ])
    _write_contract_pair(
        component_path,
        stage_path,
        component_run="component-run",
        stage_run="stage-run",
    )
    contract = json.loads(
        vllm_probe._run_contract_path(component_path).read_text()
    )
    records = [
        json.loads(line) for line in component_path.read_text().splitlines()
    ]
    if mutation == "physical_order":
        records[0], records[1] = records[1], records[0]
    elif mutation == "cycle_sequence":
        records[0]["cycle_sequence"] = 9
    else:
        records[0]["cycle_id"] = "component-run:not-sequence-bound"
    _write_jsonl(component_path, records)

    with pytest.raises(ValueError, match=detail):
        vllm_probe._validate_component_warmup_raw(
            component_path, contract,
        )


def test_exact_component_cohort_segment_rejects_mixed_prefill(tmp_path):
    mixed = _component_record(
        run_id="component-run", cycle_id="mixed", phase="prefill",
        batch=2, tokens=20, context=20, product=200,
    )
    mixed["publishable"] = False
    mixed["reason"] = "mixed_prefill_decode_not_component_identifiable"
    mixed["shape"]["decode"] = {
        "scheduled_sequences": 1,
        "scheduled_tokens": 1,
        "scheduled_context_tokens": 8,
        "token_context_product": 8,
    }
    path = tmp_path / "mixed-target.jsonl"
    decode_records = [
        _component_record(
            run_id="component-run", cycle_id=f"decode-{index}",
            phase="decode", batch=2, tokens=2,
            context=22 + index * 2, product=22 + index * 2,
        )
        for index in range(3)
    ]
    _write_jsonl(path, [mixed, *decode_records])

    with pytest.raises(ValueError, match="non-publishable cycle"):
        vllm_probe._exact_component_cohort_service_from_segment(
            path.read_bytes(),
            target_batch=2,
            input_tokens=10,
            output_tokens=4,
        )


def test_exact_component_cohort_segment_rejects_leading_blocker(tmp_path):
    records = [
        _component_record(
            run_id="component-run", cycle_id="blocker", phase="decode",
            batch=256, tokens=256, context=2048, product=2048,
        ),
        _component_record(
            run_id="component-run", cycle_id="prefill", phase="prefill",
            batch=2, tokens=20, context=20, product=200,
        ),
    ]
    records.extend(
        _component_record(
            run_id="component-run", cycle_id=f"decode-{index}",
            phase="decode", batch=2, tokens=2,
            context=22 + index * 2, product=22 + index * 2,
        )
        for index in range(3)
    )
    path = tmp_path / "blocker-prefixed.jsonl"
    _write_jsonl(path, records)

    with pytest.raises(ValueError, match="cycle count differs"):
        vllm_probe._exact_component_cohort_service_from_segment(
            path.read_bytes(),
            target_batch=2,
            input_tokens=10,
            output_tokens=4,
        )


def test_component_parser_rejects_non_tp1_reference(tmp_path):
    record = _component_record(run_id="component-run", cycle_id="c0")
    record["world_size"] = 2
    record["tensor_parallel_size"] = 2
    path = tmp_path / "component.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises(ValueError, match="TP1/PP1"):
        vllm_probe.read_component_cycle_services(path)


def test_queued_extra_stage_cycles_never_become_resident_evidence(tmp_path):
    path = tmp_path / "stage.jsonl"
    _write_jsonl(path, [
        _stage_record(
            run_id="stage-run",
            cycle_id="queued-prefill",
            phase="prefill",
            batch=2,
            tokens=20,
            context=20,
            product=200,
            running=0,
            waiting=3,
        ),
        _stage_record(
            run_id="stage-run",
            cycle_id="queued-decode",
            running=2,
            waiting=1,
        ),
        _stage_record(
            run_id="stage-run",
            cycle_id="resident-decode",
            running=2,
            waiting=0,
        ),
    ])

    observations = vllm_probe.read_stage_cycle_observations(path)
    assert [(item.run_id, item.cycle_id) for item in observations] == [
        ("stage-run", "resident-decode"),
    ]


def test_frozen_reader_skips_discarded_first_sample_before_backlog_gate(
    tmp_path,
):
    path = tmp_path / "stage.jsonl"
    warmup = _stage_record(
        run_id="stage-run",
        cycle_id="discarded-warmup-prefill",
        phase="prefill",
        batch=2,
        tokens=20,
        context=20,
        product=200,
    )
    warmup["active_service_s"] = None
    warmup["backlog_before_interval"] = None
    warmup["publishable"] = False
    warmup["reason"] = "first_sample_no_interval"
    target = _stage_record(
        run_id="stage-run",
        cycle_id="resident-target-decode",
        running=2,
        waiting=0,
    )
    _write_jsonl(path, [warmup, target])

    observations = vllm_probe.read_stage_cycle_observations(
        path,
        allowed_cycle_ids={"resident-target-decode"},
    )

    assert [(item.run_id, item.cycle_id) for item in observations] == [
        ("stage-run", "resident-target-decode"),
    ]


def test_measure_stage_ladder_runs_component_and_stage_in_isolation(
    monkeypatch, tmp_path,
):
    stage_path = tmp_path / "stage.jsonl"
    component_path = tmp_path / "component.jsonl"
    calls = []

    selected_component_ids = []

    def component_records(run_id):
        rows = []
        for cohort_index, (label, prefill_s, decode_s) in enumerate((
            ("warmup-0", 0.3, 0.8),
            ("warmup-1", 0.9, 0.2),
            ("selected-0", 0.0029, 0.0039),
            ("selected-1", 0.003, 0.004),
            ("selected-2", 0.0031, 0.0041),
        )):
            sequence_base = cohort_index * 4
            cohort_rows = [
                _component_record(
                    run_id=run_id,
                    cycle_id=f"{run_id}:{sequence_base}:{label}-prefill",
                    phase="prefill",
                    batch=2,
                    tokens=20,
                    context=20,
                    product=200,
                    service_s=prefill_s,
                    work_sequence=sequence_base,
                ),
            ]
            cohort_rows.extend(
                _component_record(
                    run_id=run_id,
                    cycle_id=(
                        f"{run_id}:{sequence_base + index}:"
                        f"{label}-decode-{index}"
                    ),
                    batch=2,
                    tokens=2,
                    context=2 * (10 + index),
                    product=2 * (10 + index),
                    service_s=decode_s,
                    work_sequence=sequence_base + index,
                )
                for index in range(1, 4)
            )
            for row in cohort_rows:
                row["cycle_sequence"] = row["work_sequence"]
            rows.extend(cohort_rows)
            if cohort_index >= 2:
                selected_component_ids.extend(
                    row["cycle_id"] for row in cohort_rows
                )
        return rows

    def stage_records(run_id):
        rows = [
            _stage_record(
                run_id=run_id,
                cycle_id=f"{run_id}:prefill",
                phase="prefill",
                batch=2,
                tokens=20,
                context=20,
                product=200,
                work_sequence=0,
            ),
        ]
        rows.extend(
            _stage_record(
                run_id=run_id,
                cycle_id=f"{run_id}:decode-{index}",
                batch=2,
                tokens=2,
                context=2 * (10 + index),
                product=2 * (10 + index),
                work_sequence=index,
            )
            for index in range(1, 4)
        )
        return rows

    def fake_offline(
        key, shapes, *, telemetry_path, repeats, request_contracts,
    ):
        assert key.max_num_seqs == 2
        assert shapes == [(2, 10, 4)] and repeats == 5
        assert request_contracts == _request_contracts()
        _write_jsonl(telemetry_path, component_records("component-run"))
        return {(2, 10, 4): (10.0, 0.1, 0.015)}

    def fake_shape_proof(
        key, *, target_batch, in_tok, out_tok, request_contract,
    ):
        assert key.max_num_seqs == target_batch == 2
        assert (in_tok, out_tok) == (10, 4)
        assert request_contract == _request_contract()
        stage_env = os.environ.get(vllm_probe._STAGE_TELEMETRY_ENV)
        run_id = os.environ[vllm_probe._CALIBRATION_RUN_ID_ENV]
        calls.append((stage_env, run_id, "proof", key.max_num_seqs))
        rows = stage_records(run_id)
        _write_jsonl(Path(stage_env), rows)
        ids = [row["cycle_id"] for row in rows]
        return (
            4096,
            ids,
            [f"{run_id}:prefill-probe:B2"],
            _stage_proof(2, out_tok=4),
        )

    def fake_timing(
        key, *, target_batch, in_tok, out_tok, request_contract,
    ):
        assert key.max_num_seqs == target_batch == 2
        assert (in_tok, out_tok) == (10, 4)
        assert request_contract == _request_contract()
        timing_run_id = "timing-run"
        calls.append((None, timing_run_id, "timing", key.max_num_seqs))
        return (
            _timing_evidence(
                timing_run_id=timing_run_id,
                batch=2,
                input_tokens=10,
                output_tokens=4,
            ),
            timing_run_id,
            4096,
            (0.1, 0.11),
        )
    monkeypatch.setattr(
        vllm_probe,
        "build_generator_request_contracts",
        lambda _model, _tokens: _request_contracts(),
    )
    monkeypatch.setattr(
        vllm_probe,
        "_measure_component_offline_subprocess",
        fake_offline,
    )
    monkeypatch.setattr(
        vllm_probe,
        "_measure_stage_shape_proof_on_owned_server",
        fake_shape_proof,
    )
    monkeypatch.setattr(
        vllm_probe,
        "_measure_production_timing_on_owned_server",
        fake_timing,
    )
    ladder, observations, services = vllm_probe.measure_stage_ladder(
        _key(),
        [2],
        10,
        4,
        telemetry_path=stage_path,
        component_telemetry_path=component_path,
    )

    assert ladder[2][0] == pytest.approx(80.0)
    assert ladder[2][1] == pytest.approx(0.105)
    assert len(observations) == 4
    assert services == {
        (calls[0][1], f"{calls[0][1]}:prefill"): pytest.approx(0.003),
        **{
            (calls[0][1], f"{calls[0][1]}:decode-{index}"):
                pytest.approx(0.004)
            for index in range(1, 4)
        },
    }
    assert len(calls) == 2
    assert Path(calls[0][0]).parent == Path("/dev/shm")
    assert calls[0][1] != "component-run"
    assert calls[0][2:] == ("proof", 2)
    assert calls[1] == (None, "timing-run", "timing", 2)
    assert component_path.exists() and stage_path.exists()
    component_contract = json.loads(
        vllm_probe._run_contract_path(component_path).read_text()
    )
    stage_contract = json.loads(
        vllm_probe._run_contract_path(stage_path).read_text()
    )
    assert stage_contract["contract_origin"] == {
        "kind": "direct_acquisition"
    }
    assert set(component_contract["subruns"]["2"]) == {"component"}
    assert set(stage_contract["subruns"]["2"]) == {"proof", "timing"}
    assert component_contract["target_cycle_ids"]["2"] == selected_component_ids
    assert len(component_contract["target_cycle_ids"]["2"]) == 3 * 4
    assert all(
        ":warmup-" not in cycle_id
        for cycle_id in component_contract["target_cycle_ids"]["2"]
    )
    assert component_contract["batch_proofs"]["2"] == {
        "target_cohort_size": 2,
        "observed_prompt_tokens": 10,
        "observed_output_tokens": 4,
        "exact_prefill_cycles": 1,
        "exact_decode_cycles": 3,
        "queued_proof_cohorts": 5,
        "discarded_warmup_cohorts": 2,
        "selected_measurement_cohorts": 3,
    }
    assert all(
        descriptor["startup"]["max_num_seqs"] == 2
        for descriptor in (
            component_contract["subruns"]["2"]["component"],
            stage_contract["subruns"]["2"]["proof"],
            stage_contract["subruns"]["2"]["timing"],
        )
    )
    measurements = vllm_probe.read_stage_active_service_measurements(
        stage_path
    )
    assert measurements[2]["sample_count"] == 7
    assert len(measurements[2]["full_interdeparture_s"]) == 7
    assert len(
        measurements[2]["out_one_diagnostic_interdeparture_s"]
    ) == 7


def test_production_timing_evidence_freezes_interdeparture_samples() -> None:
    evidence = _timing_evidence(
        timing_run_id="timing-run",
        batch=1,
        input_tokens=10,
        output_tokens=4,
    )

    restored = stage_timing_contract.CollocatedEngineTimingEvidence.from_dict(
        evidence.to_dict()
    )

    assert restored.full_engine_interdeparture.timing_run_id == "timing-run"
    assert len(restored.full_engine_interdeparture.interdeparture_s) == 7
    assert restored.out_one_diagnostic is not None
    assert len(restored.out_one_diagnostic.cohorts) == 8
    assert restored.out_one_diagnostic.engine_batch_cap == 1
    assert restored.out_one_diagnostic.telemetry_enabled is False
    serialized = repr(restored.to_dict()).lower()
    assert "prefill_total_service" not in serialized
    assert "decode_step_service" not in serialized


def test_build_key_honors_calibration_frontend_and_context_overrides():
    key = vllm_probe.build_key(
        "Qwen/Test",
        1,
        1,
        {"available_gpus": ["cuda:0"]},
        api_server_count=1,
        max_model_len=4096,
        enable_prefix_caching=False,
    )
    assert key.api_server_count == 1
    assert key.max_model_len == 4096
    assert key.enable_prefix_caching is False
