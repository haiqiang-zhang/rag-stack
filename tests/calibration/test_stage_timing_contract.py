from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rag_stack.cost_model.llm_sim.calibration.main_llm import (
    generator_parallel,
)
from rag_stack.cost_model.llm_sim.calibration.main_llm.stage_timing_contract import (
    CLOCK_DOMAIN,
    ClockDomainProof,
    CollocatedEngineTimingEvidence,
    ProductionCohortTiming,
    ProductionInterdeparture,
    RequestTiming,
    StageTimingContractError,
    build_collocated_engine_timing_evidence,
    build_production_cohort_service_evidence,
    capture_clock_domain_proof,
)


_RUN_ID = "production-timing-run"
_BATCH = 2
_INPUT_TOKENS = 16
_SHORT_INTERVALS = (0.19, 0.20, 0.21, 0.20, 0.19, 0.21, 0.20)
_FULL_INTERVALS = (0.79, 0.80, 0.81, 0.80, 0.79, 0.81, 0.80)


def _clock() -> ClockDomainProof:
    return ClockDomainProof(
        domain=CLOCK_DOMAIN,
        parent_response_source="time.monotonic",
        implementation="clock_gettime(CLOCK_MONOTONIC)",
        monotonic=True,
        adjustable=False,
        resolution_s=1e-9,
    )


def _raw_cohorts(
    *,
    base: float,
    intervals: tuple[float, ...],
    batch: int = _BATCH,
    completion_spread_s: float = 0.002,
) -> list[list[dict[str, float]]]:
    endpoints = [base + 1.0]
    for interval in intervals:
        endpoints.append(endpoints[-1] + interval)
    return [
        [
            {
                "request_send_ts": base + group_index * 0.01 + index * 0.001,
                "last_token_ts": (
                    endpoint
                    - completion_spread_s * (batch - index - 1) / max(batch - 1, 1)
                ),
            }
            for index in range(batch)
        ]
        for group_index, endpoint in enumerate(endpoints)
    ]


def _phase(
    *,
    output_tokens: int,
    base: float,
    intervals: tuple[float, ...],
    batch: int = _BATCH,
    dispatch_counts: tuple[int, ...] | None = None,
) -> ProductionInterdeparture:
    return build_production_cohort_service_evidence(
        timing_run_id=_RUN_ID,
        engine_batch_cap=batch,
        target_batch=batch,
        input_tokens=_INPUT_TOKENS,
        output_tokens=output_tokens,
        cohort_request_timings=_raw_cohorts(
            base=base,
            intervals=intervals,
            batch=batch,
        ),
        observed_dispatch_counts=(
            dispatch_counts
            if dispatch_counts is not None
            else tuple(batch * index for index in range(1, 9))
        ),
        enginecore_fence_ack_count=8,
        clock=_clock(),
    )


def _evidence() -> CollocatedEngineTimingEvidence:
    short = _phase(
        output_tokens=1,
        base=10.0,
        intervals=_SHORT_INTERVALS,
    )
    full = _phase(
        output_tokens=4,
        base=20.0,
        intervals=_FULL_INTERVALS,
    )
    return build_collocated_engine_timing_evidence(
        full,
        out_one_diagnostic=short,
    )


def test_clock_proof_uses_only_parent_monotonic() -> None:
    proof = capture_clock_domain_proof()

    assert proof.domain == CLOCK_DOMAIN
    assert proof.parent_response_source == "time.monotonic"
    assert "CLOCK_MONOTONIC" in proof.implementation


def test_clock_proof_fails_closed_on_bad_clock_or_backwards_sample() -> None:
    bad_info = SimpleNamespace(
        implementation="QueryPerformanceCounter",
        monotonic=True,
        adjustable=False,
        resolution=1e-9,
    )
    with pytest.raises(StageTimingContractError, match="CLOCK_MONOTONIC"):
        capture_clock_domain_proof(get_clock_info=lambda _name: bad_info)

    good_info = SimpleNamespace(
        implementation="clock_gettime(CLOCK_MONOTONIC)",
        monotonic=True,
        adjustable=False,
        resolution=1e-9,
    )
    samples = iter((2.0, 1.0))
    with pytest.raises(StageTimingContractError, match="moved backwards"):
        capture_clock_domain_proof(
            get_clock_info=lambda _name: good_info,
            sample_clock=lambda: next(samples),
        )


def test_phase_uses_eight_cohorts_and_seven_interdepartures() -> None:
    phase = _phase(
        output_tokens=1,
        base=10.0,
        intervals=_SHORT_INTERVALS,
    )

    assert len(phase.cohorts) == 8
    assert len(phase.interdeparture_s) == 7
    assert phase.observed_dispatch_counts == tuple(range(2, 17, 2))
    assert phase.interdeparture_s == pytest.approx(_SHORT_INTERVALS)
    assert phase.median_service_s == pytest.approx(0.20)
    assert phase.interval_mad_s == pytest.approx(0.01)
    assert phase.relative_interval_mad == pytest.approx(0.05)
    assert all(
        cohort.completion_spread_s == pytest.approx(0.002)
        for cohort in phase.cohorts
    )


def test_cohort_zero_is_warmup_not_a_service_sample() -> None:
    raw = _raw_cohorts(base=10.0, intervals=_SHORT_INTERVALS)
    raw[0][0]["request_send_ts"] -= 100.0
    phase = build_production_cohort_service_evidence(
        timing_run_id=_RUN_ID,
        engine_batch_cap=_BATCH,
        target_batch=_BATCH,
        input_tokens=_INPUT_TOKENS,
        output_tokens=1,
        cohort_request_timings=raw,
        observed_dispatch_counts=tuple(range(2, 17, 2)),
        enginecore_fence_ack_count=8,
        clock=_clock(),
    )

    assert phase.median_service_s == pytest.approx(0.20)


def test_common_clock_shift_does_not_change_service() -> None:
    baseline = _phase(
        output_tokens=1,
        base=10.0,
        intervals=_SHORT_INTERVALS,
    )
    shifted = _phase(
        output_tokens=1,
        base=1010.0,
        intervals=_SHORT_INTERVALS,
    )

    assert shifted.interdeparture_s == pytest.approx(baseline.interdeparture_s)
    assert shifted.median_service_s == pytest.approx(baseline.median_service_s)
    assert shifted.interval_mad_s == pytest.approx(baseline.interval_mad_s)


def test_full_engine_interval_is_the_only_publishable_timing() -> None:
    evidence = _evidence()

    assert evidence.full_engine_interdeparture.median_service_s == pytest.approx(
        0.80
    )
    assert evidence.out_one_diagnostic is not None
    assert evidence.out_one_diagnostic.median_service_s == pytest.approx(0.20)


def test_stage_evidence_round_trips_exactly() -> None:
    evidence = _evidence()
    document = evidence.to_dict()

    assert CollocatedEngineTimingEvidence.from_dict(document) == evidence
    assert document["out_one_diagnostic"]["telemetry_enabled"] is False
    assert len(document["full_engine_interdeparture"]["cohorts"]) == 8
    assert len(document["full_engine_interdeparture"]["interdeparture_s"]) == 7

    document["unexpected"] = True
    with pytest.raises(StageTimingContractError, match="exact schema"):
        CollocatedEngineTimingEvidence.from_dict(document)


def test_serialized_contract_contains_no_cycle_or_instrumentation_fit_fields() -> None:
    document = _evidence().to_dict()
    serialized = repr(document).lower()

    for forbidden in (
        "cycle_id",
        "blocker",
        "active_service",
        "cumulative",
        "mfu",
        "frontend",
        "prefill_total_service",
        "decode_step_service",
        "full_minus_short",
    ):
        assert forbidden not in serialized


def test_out_one_diagnostic_is_optional_and_not_publishable_service() -> None:
    full = _phase(
        output_tokens=4,
        base=20.0,
        intervals=_FULL_INTERVALS,
    )
    evidence = build_collocated_engine_timing_evidence(full)

    assert evidence.out_one_diagnostic is None
    restored = CollocatedEngineTimingEvidence.from_dict(evidence.to_dict())
    assert restored == evidence


def test_engine_cap_must_equal_target_batch() -> None:
    with pytest.raises(StageTimingContractError, match="engine_batch_cap"):
        build_production_cohort_service_evidence(
            timing_run_id=_RUN_ID,
            engine_batch_cap=4,
            target_batch=_BATCH,
            input_tokens=_INPUT_TOKENS,
            output_tokens=1,
            cohort_request_timings=_raw_cohorts(
                base=10.0,
                intervals=_SHORT_INTERVALS,
            ),
            observed_dispatch_counts=tuple(range(2, 17, 2)),
            enginecore_fence_ack_count=8,
            clock=_clock(),
        )


def test_production_phase_must_freeze_telemetry_disabled() -> None:
    document = _phase(
        output_tokens=1,
        base=10.0,
        intervals=_SHORT_INTERVALS,
    ).to_dict()
    document["telemetry_enabled"] = True

    with pytest.raises(StageTimingContractError, match="telemetry disabled"):
        ProductionInterdeparture.from_dict(document)


def test_dispatch_counter_proof_must_be_exact_cumulative_batches() -> None:
    with pytest.raises(StageTimingContractError, match="counts"):
        _phase(
            output_tokens=1,
            base=10.0,
            intervals=_SHORT_INTERVALS,
            dispatch_counts=(2, 4, 6, 8, 10, 12, 14, 15),
        )


def test_enginecore_fifo_fence_ack_is_frozen_and_exact() -> None:
    document = _phase(
        output_tokens=1,
        base=10.0,
        intervals=_SHORT_INTERVALS,
    ).to_dict()
    assert document["enginecore_fence_ack_count"] == 8
    assert "fifo_is_paused_ack" in document["enginecore_dispatch_fence"]

    document["enginecore_fence_ack_count"] = 7
    with pytest.raises(StageTimingContractError, match="one FIFO EngineCore"):
        ProductionInterdeparture.from_dict(document)


def test_exactly_eight_cohorts_are_required() -> None:
    with pytest.raises(StageTimingContractError, match="exactly eight"):
        build_production_cohort_service_evidence(
            timing_run_id=_RUN_ID,
            engine_batch_cap=_BATCH,
            target_batch=_BATCH,
            input_tokens=_INPUT_TOKENS,
            output_tokens=1,
            cohort_request_timings=_raw_cohorts(
                base=10.0,
                intervals=_SHORT_INTERVALS,
            )[:-1],
            observed_dispatch_counts=tuple(range(2, 17, 2)),
            enginecore_fence_ack_count=8,
            clock=_clock(),
        )


def test_each_cohort_must_freeze_exactly_b_request_timings() -> None:
    raw = _raw_cohorts(base=10.0, intervals=_SHORT_INTERVALS)
    raw[3].pop()

    with pytest.raises(StageTimingContractError, match="exactly B"):
        build_production_cohort_service_evidence(
            timing_run_id=_RUN_ID,
            engine_batch_cap=_BATCH,
            target_batch=_BATCH,
            input_tokens=_INPUT_TOKENS,
            output_tokens=1,
            cohort_request_timings=raw,
            observed_dispatch_counts=tuple(range(2, 17, 2)),
            enginecore_fence_ack_count=8,
            clock=_clock(),
        )


def test_groups_must_be_dispatched_in_index_order() -> None:
    raw = _raw_cohorts(base=10.0, intervals=_SHORT_INTERVALS)
    raw[4][0]["request_send_ts"] = raw[3][0]["request_send_ts"] - 0.1

    with pytest.raises(StageTimingContractError, match="index order"):
        build_production_cohort_service_evidence(
            timing_run_id=_RUN_ID,
            engine_batch_cap=_BATCH,
            target_batch=_BATCH,
            input_tokens=_INPUT_TOKENS,
            output_tokens=1,
            cohort_request_timings=raw,
            observed_dispatch_counts=tuple(range(2, 17, 2)),
            enginecore_fence_ack_count=8,
            clock=_clock(),
        )


def test_all_groups_must_be_queued_before_first_completion() -> None:
    raw = _raw_cohorts(base=10.0, intervals=_SHORT_INTERVALS)
    raw[7][1]["request_send_ts"] = raw[0][0]["last_token_ts"] + 0.01

    with pytest.raises(StageTimingContractError, match="queued before"):
        build_production_cohort_service_evidence(
            timing_run_id=_RUN_ID,
            engine_batch_cap=_BATCH,
            target_batch=_BATCH,
            input_tokens=_INPUT_TOKENS,
            output_tokens=1,
            cohort_request_timings=raw,
            observed_dispatch_counts=tuple(range(2, 17, 2)),
            enginecore_fence_ack_count=8,
            clock=_clock(),
        )


def test_cohort_endpoints_must_be_strictly_increasing() -> None:
    bad_intervals = (0.2, 0.2, 0.0, 0.2, 0.2, 0.2, 0.2)

    with pytest.raises(StageTimingContractError, match="strictly increasing"):
        _phase(output_tokens=1, base=10.0, intervals=bad_intervals)


@pytest.mark.parametrize(
    "path, delta, message",
    (
        (("cohorts", 2, "endpoint_monotonic_s"), 0.01, "cohort endpoint"),
        (("cohorts", 2, "completion_spread_s"), 0.01, "completion spread"),
        (("interdeparture_s", 2), 0.01, "interdeparture samples"),
        (("median_service_s",), 0.01, "median steady-state"),
        (("interval_mad_s",), 0.01, "interval MAD"),
        (("relative_interval_mad",), 0.01, "relative interval MAD"),
    ),
)
def test_round_trip_recomputes_all_derived_phase_fields(
    path: tuple[object, ...],
    delta: float,
    message: str,
) -> None:
    document = _phase(
        output_tokens=1,
        base=10.0,
        intervals=_SHORT_INTERVALS,
    ).to_dict()
    cursor = document
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index,assignment]
    cursor[path[-1]] += delta  # type: ignore[index,operator]

    with pytest.raises(StageTimingContractError, match=message):
        ProductionInterdeparture.from_dict(document)


def test_relative_interval_mad_above_twenty_percent_fails_closed() -> None:
    unstable = (0.10, 0.10, 0.10, 0.20, 0.30, 0.30, 0.30)

    with pytest.raises(StageTimingContractError, match="20% stability"):
        _phase(output_tokens=1, base=10.0, intervals=unstable)


def test_out_one_diagnostic_and_full_engine_timing_must_not_overlap() -> None:
    short = _phase(
        output_tokens=1,
        base=10.0,
        intervals=_SHORT_INTERVALS,
    )
    overlapping = _phase(
        output_tokens=4,
        base=10.5,
        intervals=_FULL_INTERVALS,
    )

    with pytest.raises(StageTimingContractError, match="must not overlap"):
        build_collocated_engine_timing_evidence(
            overlapping,
            out_one_diagnostic=short,
        )


def test_out_one_diagnostic_and_full_must_share_timing_run_and_shape() -> None:
    evidence = _evidence()
    wrong_run = replace(
        evidence.full_engine_interdeparture,
        timing_run_id="different-production-run",
    )

    with pytest.raises(StageTimingContractError, match="share the full engine"):
        build_collocated_engine_timing_evidence(
            wrong_run,
            out_one_diagnostic=evidence.out_one_diagnostic,
        )


def test_out_one_diagnostic_is_never_subtracted_from_full_engine_timing() -> None:
    diagnostic = _phase(
        output_tokens=1,
        base=10.0,
        intervals=_FULL_INTERVALS,
    )
    full = _phase(
        output_tokens=4,
        base=20.0,
        intervals=_SHORT_INTERVALS,
    )

    evidence = build_collocated_engine_timing_evidence(
        full,
        out_one_diagnostic=diagnostic,
    )
    assert evidence.full_engine_interdeparture.median_service_s == pytest.approx(
        0.20
    )


def test_generator_fit_rejects_any_negative_anchor_without_interpolation() -> None:
    batches = (1, 8, 32, 64, 256, 128)
    component_cohorts = {
        batch: {
            "scheduler_cycle_count": 4,
            "cohort_total_service_s": [
                1.0 if batch == 8 else 0.4,
            ] * 3,
            "cohort_count": 3,
        }
        for batch in batches
    }
    stage_services = {
        batch: {
            "collocated_engine_cycle_count": 4,
            "collocated_engine_cycle_service_s": 0.20,
            "sample_count": 7,
        }
        for batch in batches
    }

    with pytest.raises(
        ValueError,
        match=(
            r"B8 collocated engine-cycle overhead is negative.*"
            r"complete fit must fail.*never clamp"
        ),
    ):
        generator_parallel._measurements_for_collocated_engine(
            component_cohort_measurements=component_cohorts,
            stage_service_measurements=stage_services,
            system_hardware="sgs_h100x4_epyc9124",
            stage="generator",
        )


def test_nested_contract_rejects_unknown_fields() -> None:
    document = deepcopy(
        _phase(
            output_tokens=1,
            base=10.0,
            intervals=_SHORT_INTERVALS,
        ).to_dict()
    )
    document["cohorts"][0]["request_timings"][0]["request_id"] = "invented"

    with pytest.raises(StageTimingContractError, match="exact schema"):
        ProductionInterdeparture.from_dict(document)


def test_cohort_parent_timings_have_no_fabricated_request_id_mapping() -> None:
    cohort = _phase(
        output_tokens=1,
        base=10.0,
        intervals=_SHORT_INTERVALS,
    ).cohorts[0]

    assert isinstance(cohort, ProductionCohortTiming)
    assert all(
        set(timing.to_dict())
        == {"request_index", "sent_monotonic_s", "completed_monotonic_s"}
        for timing in cohort.request_timings
    )
    assert isinstance(cohort.request_timings[0], RequestTiming)
