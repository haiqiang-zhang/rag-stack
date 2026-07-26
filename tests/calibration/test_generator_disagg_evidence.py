"""CPU-only tests for strict direct P/D component evidence assembly."""

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from rag_stack.cost_model.llm_sim.calibration.main_llm.generator_disagg_evidence import (
    ActualCycleShape,
    ComponentCycleReference,
    ComponentReferenceGrid,
    DirectDisaggEvidenceError,
    DirectRoleBatchEvidence,
    DirectRoleEvidence,
    DirectRoleWindowSample,
    DECODE_ACQUISITION_GUARD_DISPOSITION,
    DECODE_ACQUISITION_GUARD_OUTPUT_TOKENS,
    DECODE_CONSECUTIVE_EXACT_BATCH_CYCLES,
    DECODE_INTERDEPARTURE_INTERVALS_PER_SAMPLE,
    DECODE_WINDOW_SOURCE,
    REQUIRED_BATCHES,
    SelectedRoleCycle,
    _component_grid_from_selected_records,
    _selection_contract,
    _validate_selected_cycles_against_telemetry,
    merge_and_publish_complete_generation,
    measurements_for_direct_role,
    validate_nixl_physical_campaign,
)
from rag_stack.cost_model.llm_sim.calibration.main_llm.nixl_physical_evidence import (
    derive_nixl_physical_evidence_from_text,
)


def _component_grid(
    points_by_batch,
    *,
    path=Path("/component.jsonl"),
    contract_path=Path("/component.jsonl.contract.json"),
):
    return ComponentReferenceGrid(
        by_phase_batch_tokens=points_by_batch,
        source_path=path,
        source_sha256="a" * 64,
        contract_path=contract_path,
        contract_sha256="b" * 64,
        prompt_tokens=112,
        output_tokens=128,
        request_contract={},
    )


def test_component_grid_uses_only_contract_selected_cohorts() -> None:
    shape = ActualCycleShape(8, 896, 896, 100_352)
    records = [
        ComponentCycleReference("selected-0", "prefill", shape, 0.1),
        ComponentCycleReference("selected-1", "prefill", shape, 0.2),
        ComponentCycleReference("selected-2", "prefill", shape, 0.3),
        # Two discarded warmups are present in raw but absent from the run
        # contract's target_cycle_ids and must not affect the 0.2 median.
        ComponentCycleReference("warmup-0", "prefill", shape, 100.0),
        ComponentCycleReference("warmup-1", "prefill", shape, 200.0),
    ]
    selected = {"selected-0", "selected-1", "selected-2"}

    result = _component_grid_from_selected_records(
        records, selected_cycle_ids=selected
    )

    point_shape, service, cycle_ids = result[("prefill", 8, 896)][0]
    assert point_shape == shape
    assert service == pytest.approx(0.2)
    assert cycle_ids == ("selected-0", "selected-1", "selected-2")


def test_direct_role_merge_rejects_unknown_stage_profile_fields(tmp_path) -> None:
    path = tmp_path / "stage-batch.json"
    path.write_text(json.dumps({
        "_stage_batch_curves": {
            "system_hardware": "test_hw",
            "profiles": {},
            "unexpected": True,
        },
    }))

    with pytest.raises(DirectDisaggEvidenceError, match="exact schema"):
        merge_and_publish_complete_generation(path, ())


def test_decode_component_join_is_strict_bracketed_interpolation() -> None:
    left = ActualCycleShape(8, 8, 904, 904)
    right = ActualCycleShape(8, 8, 912, 912)
    target = ActualCycleShape(8, 8, 910, 910)
    grid = _component_grid({
        ("decode", 8, 8): (
            (left, 1.0, ("l0", "l1", "l2")),
            (right, 3.0, ("r0", "r1", "r2")),
        )
    })

    join = grid.join_for("decode", target, target_cycle_id="role-cycle")

    assert join["component_service_s"] == pytest.approx(2.5)
    assert [point["weight"] for point in join["bracket"]] == pytest.approx(
        [0.25, 0.75]
    )
    assert join["bracket"][0]["component_cycle_ids"] == ["l0", "l1", "l2"]
    assert join["bracket"][1]["component_cycle_ids"] == ["r0", "r1", "r2"]


def test_decode_component_join_forbids_extrapolation() -> None:
    left = ActualCycleShape(8, 8, 904, 904)
    right = ActualCycleShape(8, 8, 912, 912)
    grid = _component_grid({
        ("decode", 8, 8): (
            (left, 1.0, ("l0", "l1", "l2")),
            (right, 3.0, ("r0", "r1", "r2")),
        )
    })

    with pytest.raises(DirectDisaggEvidenceError, match="extrapolation"):
        grid.service_for("decode", ActualCycleShape(8, 8, 900, 900))


def _decode_evidence_and_grid(*, negative_sample: bool = False):
    component_path = Path("/component.jsonl")
    component_contract_path = Path("/component.jsonl.contract.json")
    batches = {}
    points = {}
    joins = {}
    for batch in REQUIRED_BATCHES:
        shapes = (
            ActualCycleShape(batch, batch, batch * 150, batch * 150),
            ActualCycleShape(batch, batch, batch * 151, batch * 151),
            ActualCycleShape(batch, batch, batch * 152, batch * 152),
        )
        component_services = (0.10, 0.40, 0.20)
        observed_services = [0.35, 0.45, 0.60]
        if negative_sample:
            observed_services[1] = 0.30
        point_values = []
        samples = []
        sample_joins = []
        for index, (shape, component_service, observed) in enumerate(zip(
            shapes, component_services, observed_services, strict=True
        )):
            component_ids = tuple(
                f"component-b{batch}-s{index}-c{cohort}"
                for cohort in range(3)
            )
            point_values.append((shape, component_service, component_ids))
            role_cycle = SelectedRoleCycle(
                f"role-b{batch}:s{index}", shape
            )
            samples.append(DirectRoleWindowSample(
                sample_index=index,
                window_source=DECODE_WINDOW_SOURCE,
                window_started_monotonic_s=float(index + 1),
                window_completed_monotonic_s=float(index + 1) + observed,
                role_window_s=observed,
                scheduler_cycle_count=1,
                observed_service_per_cycle_s=observed,
                selected_cycles=(role_cycle,),
                window_proof={},
            ))
            sample_joins.append([{
                "target_cycle_id": role_cycle.cycle_id,
                "target_shape": shape.to_dict(),
                "bracket": [{
                    "component_cycle_ids": list(component_ids),
                    "shape": shape.to_dict(),
                    "weight": 1.0,
                    "median_component_service_s": component_service,
                }],
                "component_service_s": component_service,
            }])
        points[("decode", batch, batch)] = tuple(point_values)
        joins[str(batch)] = sample_joins
        batches[batch] = DirectRoleBatchEvidence(
            target_batch=batch,
            engine_max_num_seqs=batch,
            telemetry_path=Path(f"/decode-b{batch}.jsonl"),
            telemetry_sha256="c" * 64,
            telemetry_run_id=f"decode-b{batch}",
            timing_run_id=f"timing-b{batch}",
            samples=tuple(samples),
        )
    grid = _component_grid(
        points,
        path=component_path,
        contract_path=component_contract_path,
    )
    evidence = DirectRoleEvidence(
        system_hardware="test_hw",
        role="decode",
        prompt_tokens=112,
        output_tokens=128,
        request_contract={},
        batches=batches,
        source_path=Path("/decode.json"),
        source_sha256="d" * 64,
        contract_path=Path("/decode.json.contract.json"),
        contract_sha256="e" * 64,
        component_source_path=component_path,
        component_source_sha256="a" * 64,
        component_contract_path=component_contract_path,
        component_contract_sha256="b" * 64,
        nixl_physical_path=Path("/nixl.json"),
        nixl_physical_sha256="f" * 64,
        component_joins=joins,
    )
    return evidence, grid


def test_batch_point_selects_one_paired_residual_sample() -> None:
    evidence, grid = _decode_evidence_and_grid()

    measurements = measurements_for_direct_role(
        evidence,
        grid,
        system_hardware="test_hw",
        stage="generator_decode",
    )

    # residuals are 0.25, 0.05, 0.40.  The middle paired sample is sample 0;
    # all three published fields must come from that same sample.
    assert measurements[0].observed_total_service_s == pytest.approx(0.35)
    assert measurements[0].reference_component_service_s == pytest.approx(0.10)
    assert measurements[0].overhead_s == pytest.approx(0.25)


def test_any_negative_paired_residual_fails_complete_measurement() -> None:
    evidence, grid = _decode_evidence_and_grid(negative_sample=True)

    with pytest.raises(DirectDisaggEvidenceError, match="negative"):
        measurements_for_direct_role(
            evidence,
            grid,
            system_hardware="test_hw",
            stage="generator_decode",
        )


def test_self_consistent_but_tampered_component_join_is_rejected() -> None:
    evidence, grid = _decode_evidence_and_grid()
    joins = copy.deepcopy(evidence.component_joins)
    point = joins[str(REQUIRED_BATCHES[0])][0][0]
    point["bracket"][0]["median_component_service_s"] = 0.11
    point["component_service_s"] = 0.11
    tampered = replace(evidence, component_joins=joins)

    with pytest.raises(DirectDisaggEvidenceError, match="strict recomputation"):
        measurements_for_direct_role(
            tampered,
            grid,
            system_hardware="test_hw",
            stage="generator_decode",
        )


def _decode_scheduler_window(tmp_path):
    run_id = "decode-run"
    boundary_shape = ActualCycleShape(1, 1, 143, 143)
    selected = tuple(
        SelectedRoleCycle(
            f"{run_id}:{index}:shape",
            ActualCycleShape(1, 1, context, context),
        )
        for index, context in enumerate(
            range(
                144,
                144 + DECODE_INTERDEPARTURE_INTERVALS_PER_SAMPLE,
            ),
            start=1,
        )
    )
    sample = DirectRoleWindowSample(
        sample_index=0,
        window_source=DECODE_WINDOW_SOURCE,
        window_started_monotonic_s=1.0,
        window_completed_monotonic_s=1.32,
        role_window_s=0.32,
        scheduler_cycle_count=len(selected),
        observed_service_per_cycle_s=0.01,
        selected_cycles=selected,
        window_proof={
            "boundary_start_cycle_id": f"{run_id}:0:shape",
            "boundary_start_shape": boundary_shape.to_dict(),
            "boundary_end_cycle_id": selected[-1].cycle_id,
            "consecutive_exact_batch_cycle_count": (
                DECODE_CONSECUTIVE_EXACT_BATCH_CYCLES
            ),
            "context_step_per_interval": 1,
        },
    )
    zero = {
        "scheduled_sequences": 0,
        "scheduled_tokens": 0,
        "scheduled_context_tokens": 0,
        "token_context_product": 0,
    }
    records = [{
        "schema": "rag_stack.vllm.frontend_stage_boundary_span",
        "run_id": run_id,
        "kind": "request_dispatch_ipc",
        "request_id": "chatcmpl-pd-cal-d-b1-c1-r0-" + "a" * 32,
        "started_monotonic_s": 0.8,
        "finished_monotonic_s": 0.9,
    }]
    for sequence, (cycle_id, timestamp, shape) in enumerate((
        (f"{run_id}:0:shape", 1.0, boundary_shape),
        *(
            (cycle.cycle_id, 1.0 + index * 0.01, cycle.actual_shape)
            for index, cycle in enumerate(selected, start=1)
        ),
    )):
        records.append({
            "schema": "rag_stack.vllm.scheduler_cycle_stage_service",
            "run_id": run_id,
            "cycle_id": cycle_id,
            "monotonic_timestamp_s": timestamp,
            "publishable": False,
            "reason": "callback_interval_not_a_fit_value",
            "shape": {
                "terminal_performance_stage": "decode",
                "prefill": dict(zero),
                "decode": shape.to_dict(),
            },
            "backlog_before_interval": {
                "running_sequences": 1,
                "waiting_sequences": 0,
            },
            "backlog_after_cycle": {
                "running_sequences": 1,
                "waiting_sequences": 0,
            },
            "sequence": sequence,
        })
    path = tmp_path / "decode.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path, run_id, sample, records


def test_decode_first_saturated_exact_batch_window_is_rejoined_to_raw(
    tmp_path,
) -> None:
    path, run_id, sample, _records = _decode_scheduler_window(tmp_path)

    _validate_selected_cycles_against_telemetry(
        path, telemetry_run_id=run_id, role="decode", samples=(sample,)
    )


@pytest.mark.parametrize("tamper", ("timestamp", "shape", "id"))
def test_decode_saturated_exact_batch_boundary_tampering_fails_closed(
    tmp_path,
    tamper,
) -> None:
    path, run_id, sample, records = _decode_scheduler_window(tmp_path)
    if tamper == "timestamp":
        records[1]["monotonic_timestamp_s"] = 0.99
        path.write_text("".join(
            json.dumps(record) + "\n" for record in records
        ))
        match = "start boundary does not match"
    else:
        proof = copy.deepcopy(sample.window_proof)
        if tamper == "shape":
            proof["boundary_start_shape"]["scheduled_context_tokens"] = 142
            proof["boundary_start_shape"]["token_context_product"] = 142
            match = "start boundary does not match"
        else:
            proof["boundary_start_cycle_id"] = f"{run_id}:missing:shape"
            match = "absent from raw"
        sample = replace(sample, window_proof=proof)

    with pytest.raises(DirectDisaggEvidenceError, match=match):
        _validate_selected_cycles_against_telemetry(
            path, telemetry_run_id=run_id, role="decode", samples=(sample,)
        )


def test_decode_selection_contract_keeps_guard_out_of_calibration_workload() -> None:
    selection = _selection_contract("decode")

    assert selection == {
        "phase": "decode",
        "interdeparture_intervals_per_sample": 32,
        "consecutive_exact_batch_cycle_count": 33,
        "acquisition_guard_output_tokens": 256,
        "acquisition_guard_disposition": (
            "acquisition_liveness_only_not_profile_key_or_fit_dimension"
        ),
        "window_policy": (
            "first_saturated_exact_batch_32_interdeparture_intervals"
        ),
        "clock": "CLOCK_MONOTONIC",
    }
    assert DECODE_ACQUISITION_GUARD_OUTPUT_TOKENS == 256
    assert "not_profile_key_or_fit_dimension" in (
        DECODE_ACQUISITION_GUARD_DISPOSITION
    )
    assert "output_tokens" not in selection


def test_decode_raw_first_saturated_run_and_zero_waiting_are_required(
    tmp_path,
) -> None:
    path, run_id, sample, records = _decode_scheduler_window(tmp_path)
    records[1]["backlog_before_interval"]["waiting_sequences"] = 1
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(DirectDisaggEvidenceError, match="no saturated"):
        _validate_selected_cycles_against_telemetry(
            path, telemetry_run_id=run_id, role="decode", samples=(sample,)
        )


def test_decode_raw_rejects_a_later_saturated_run(tmp_path) -> None:
    path, run_id, sample, records = _decode_scheduler_window(tmp_path)
    template = records[1]
    earlier = []
    for index in range(DECODE_CONSECUTIVE_EXACT_BATCH_CYCLES):
        record = copy.deepcopy(template)
        context = 100 + index
        record["cycle_id"] = f"{run_id}:earlier-{index}:shape"
        record["monotonic_timestamp_s"] = 0.91 + index * 0.002
        record["shape"]["decode"] = ActualCycleShape(
            1, 1, context, context
        ).to_dict()
        record["sequence"] = -100 + index
        earlier.append(record)
    records[1:1] = earlier
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    with pytest.raises(DirectDisaggEvidenceError, match="not the first raw"):
        _validate_selected_cycles_against_telemetry(
            path, telemetry_run_id=run_id, role="decode", samples=(sample,)
        )


def _nixl_metrics(*, batch: int, after: bool) -> str:
    count = batch if after else 0
    xfer = 1.0 if after else 0.0
    post = 0.1 if after else 0.0
    byte_count = count
    byte_sum = batch * 1024 if after else 0
    return "\n".join([
        f"vllm:nixl_xfer_time_seconds_sum {xfer}",
        f"vllm:nixl_xfer_time_seconds_count {count}",
        f"vllm:nixl_post_time_seconds_sum {post}",
        f"vllm:nixl_post_time_seconds_count {count}",
        f"vllm:nixl_bytes_transferred_sum {byte_sum}",
        f"vllm:nixl_bytes_transferred_count {byte_count}",
        "vllm:nixl_num_failed_transfers_total 0",
        "vllm:nixl_num_failed_notifications_total 0",
        "",
    ])


def _nixl_campaign(tmp_path):
    selected_samples = {}
    entries = []
    for batch in REQUIRED_BATCHES:
        samples = []
        for sample_index in range(3):
            started = float(batch * 100 + sample_index * 10 + 1)
            window = 0.5
            cycle = SelectedRoleCycle(
                f"decode-b{batch}:sample-{sample_index}",
                ActualCycleShape(batch, batch, batch * 150, batch * 150),
            )
            samples.append(DirectRoleWindowSample(
                sample_index=sample_index,
                window_source=DECODE_WINDOW_SOURCE,
                window_started_monotonic_s=started,
                window_completed_monotonic_s=started + window,
                role_window_s=window,
                scheduler_cycle_count=1,
                observed_service_per_cycle_s=window,
                selected_cycles=(cycle,),
                window_proof={},
            ))
        selected_samples[batch] = tuple(samples)
        for cohort in range(4):
            selected_index = None if cohort == 0 else cohort - 1
            if cohort == 0:
                started = float(batch * 100)
                window = 0.25
                cycle_ids = []
            else:
                sample = samples[selected_index]
                started = sample.window_started_monotonic_s
                window = sample.role_window_s
                cycle_ids = [
                    cycle.cycle_id for cycle in sample.selected_cycles
                ]
            before = _nixl_metrics(batch=batch, after=False)
            after = _nixl_metrics(batch=batch, after=True)
            entries.append({
                "evidence_id": f"b{batch}-c{cohort}",
                "target_batch": batch,
                "cohort_index": cohort,
                "fit_eligible": cohort > 0,
                "selected_sample_index": selected_index,
                "role_window": {
                    "window_source": DECODE_WINDOW_SOURCE,
                    "window_started_monotonic_s": started,
                    "window_completed_monotonic_s": started + window,
                    "role_window_s": window,
                },
                "selected_cycle_ids": cycle_ids,
                "before_prometheus": before,
                "before_prometheus_sha256": hashlib.sha256(
                    before.encode()
                ).hexdigest(),
                "after_prometheus": after,
                "after_prometheus_sha256": hashlib.sha256(
                    after.encode()
                ).hexdigest(),
                "physical_evidence": derive_nixl_physical_evidence_from_text(
                    before, after, expected_transfer_count=batch
                ).to_dict(),
                "cost_model_disposition": (
                    "physical_only_never_role_service_correction"
                ),
            })
    document = {
        "schema": "rag_stack.direct_disagg_nixl_physical_campaign",
        "system_hardware": "test_hw",
        "evidence_scope": "physical_only_no_role_service_fit",
        "entries": entries,
    }
    path = tmp_path / "nixl.json"
    path.write_text(json.dumps(document))
    return path, document, selected_samples


def test_nixl_campaign_binds_fixed_warmup_plus_three_role_windows(tmp_path) -> None:
    path, _document, selected_samples = _nixl_campaign(tmp_path)

    validate_nixl_physical_campaign(
        path,
        system_hardware="test_hw",
        selected_samples_by_batch=selected_samples,
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda entry: entry.update(
                cohort_index=7,
                evidence_id=f"b{entry['target_batch']}-c7",
            ),
            "unexpected batch/cohort",
        ),
        (
            lambda entry: entry["role_window"].update(role_window_s=9.0),
            "role window is internally inconsistent",
        ),
        (
            lambda entry: entry.update(selected_cycle_ids=["wrong"]),
            "does not bind its exact timing sample IDs",
        ),
    ],
)
def test_nixl_fixed_cohort_tampering_fails_closed(
    tmp_path, mutation, match
) -> None:
    path, document, selected_samples = _nixl_campaign(tmp_path)
    if "batch/cohort" in match:
        target = next(
            item for item in document["entries"]
            if item["target_batch"] == REQUIRED_BATCHES[0]
            and item["cohort_index"] == 2
        )
    elif "internally" in match:
        target = next(
            item for item in document["entries"]
            if item["cohort_index"] == 0
        )
    else:
        target = next(
            item for item in document["entries"]
            if item["fit_eligible"] is True
        )
    mutation(target)
    path.write_text(json.dumps(document))

    with pytest.raises(DirectDisaggEvidenceError, match=match):
        validate_nixl_physical_campaign(
            path,
            system_hardware="test_hw",
            selected_samples_by_batch=selected_samples,
        )
