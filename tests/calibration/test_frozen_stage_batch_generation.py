from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_stack.calibration_paths import CALIBRATION_PROFILE_DIR_ENV
from rag_stack.cost_model.stage_batch_curve import (
    ANCHOR_BATCHES,
    COMPLETE_ENGINE_STAGE_KEYS,
    HELD_OUT_BATCH,
    REQUIRED_COLLOCATED_ENGINE_STAGES,
    EngineCycleCurveApplicator,
    FrozenStageBatchGeneration,
    StageBatchCurveKey,
    StageBatchCurveValidationError,
    StageBatchMeasurement,
    build_stage_batch_curve,
    load_stage_batch_curve,
    publish_stage_batch_curves,
    resolve_frozen_stage_batch_generation,
)


_SYSTEM_HARDWARE = "fixture_h100x4_host"


def _hardware():
    return SimpleNamespace(
        system_hardware=_SYSTEM_HARDWARE,
        gpu="H100_GPU",
    )


def _profile(stage: str):
    key = StageBatchCurveKey(_SYSTEM_HARDWARE, stage)
    overhead = {
        1: 0.001,
        8: 0.004,
        32: 0.010,
        64: 0.018,
        128: 0.034,
        256: 0.066,
    }
    measurements = [
        StageBatchMeasurement(
            key=key,
            target_batch=batch,
            engine_max_num_seqs=256,
            observed_resident_batch=batch,
            reference_component_service_s=0.5 + 0.001 * batch,
            overhead_s=overhead[batch],
            observed_total_service_s=(
                0.5 + 0.001 * batch + overhead[batch]
            ),
            sample_count=5,
        )
        for batch in (*ANCHOR_BATCHES, HELD_OUT_BATCH)
    ]
    return build_stage_batch_curve(
        key=key,
        measurements=measurements,
    )


def _pair():
    return tuple(_profile(stage) for stage in REQUIRED_COLLOCATED_ENGINE_STAGES)


def _six():
    return tuple(_profile(stage) for stage in COMPLETE_ENGINE_STAGE_KEYS)


def _profile_path(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "profiles"
    monkeypatch.setenv(CALIBRATION_PROFILE_DIR_ENV, str(root))
    path = root / "stage_batch" / f"{_SYSTEM_HARDWARE}.json"
    path.parent.mkdir(parents=True)
    return path


@pytest.mark.parametrize("complete", [False, True])
def test_strict_resolver_reads_one_whole_valid_generation_once(
    tmp_path,
    monkeypatch,
    complete,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    published = _six() if complete else _pair()
    publish_stage_batch_curves(path, published)
    original_read_text = Path.read_text
    reads = []

    def counted_read_text(self, *args, **kwargs):
        if self == path:
            reads.append(self)
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", counted_read_text)
    frozen = resolve_frozen_stage_batch_generation(
        _hardware(),
        calibration_mode="strict",
    )

    assert reads == [path]
    assert frozen.enabled is True
    assert set(frozen.materialize_profiles()) == {
        profile.key.stage for profile in published
    }
    assert frozen.reason == (
        "strict_complete_six_profile_generation"
        if complete
        else "strict_collocated_pair_generation"
    )
    document = frozen.to_dict()
    json.dumps(document, allow_nan=False)
    assert FrozenStageBatchGeneration.from_dict(document) == frozen
    assert document["profile_content_sha256"] == {
        profile.key.stage: profile.content_sha256 for profile in published
    }
    assert "profile_provenance_sha256" not in document
    assert all("provenance" not in profile
               for profile in document["profiles"].values())


@pytest.mark.parametrize("malformed", [False, True])
def test_strict_resolver_fails_closed_for_absent_or_malformed_file(
    tmp_path,
    monkeypatch,
    malformed,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    if malformed:
        path.write_text("{")

    with pytest.raises(StageBatchCurveValidationError):
        resolve_frozen_stage_batch_generation(
            _hardware(),
            calibration_mode="strict",
        )


def test_uncalibrated_spec_reuses_complete_six_or_empty_fairness(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _pair())

    pair = resolve_frozen_stage_batch_generation(
        _hardware(),
        calibration_mode="uncalibrated_spec",
    )
    assert pair.enabled is False
    assert pair.reason == "collocated_only_generation"
    assert pair.generation_sha256 is None
    assert pair.materialize_profiles() == {}

    publish_stage_batch_curves(path, _six())
    complete = resolve_frozen_stage_batch_generation(
        _hardware(),
        calibration_mode="uncalibrated_spec",
    )
    assert complete.enabled is True
    assert complete.reason == "complete_six_profile_generation"
    assert set(complete.materialize_profiles()) == set(
        COMPLETE_ENGINE_STAGE_KEYS
    )


def test_unified_applicator_adds_one_curve_after_components_are_merged(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _pair())
    generation = resolve_frozen_stage_batch_generation(
        _hardware(), calibration_mode="strict",
    )
    applicator = EngineCycleCurveApplicator(generation)
    profiles = generation.materialize_profiles()
    component = 2.0 + 3.0

    corrected = applicator.corrected_service_s(
        engine_stage="generator",
        role="unified",
        prefill_stage="generator_prefill",
        decode_stage="generator_decode",
        resident_batch=5,
        component_service_s=component,
    )

    expected = profiles["generator"].corrected_service_s(5, component)
    phase_double_count = (
        profiles["generator"].corrected_service_s(2, 2.0)
        + profiles["generator"].corrected_service_s(3, 3.0)
    )
    assert corrected == pytest.approx(expected)
    assert corrected != pytest.approx(phase_double_count)


def test_disagg_applicator_uses_direct_roles_and_never_falls_back(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _six())
    complete = resolve_frozen_stage_batch_generation(
        _hardware(), calibration_mode="strict",
    )
    applicator = EngineCycleCurveApplicator(complete)
    profiles = complete.materialize_profiles()

    prefill = applicator.corrected_service_s(
        engine_stage="generator",
        role="prefill",
        prefill_stage="generator_prefill",
        decode_stage="generator_decode",
        resident_batch=8,
        component_service_s=1.25,
    )
    decode = applicator.corrected_service_s(
        engine_stage="generator",
        role="decode",
        prefill_stage="generator_prefill",
        decode_stage="generator_decode",
        resident_batch=32,
        component_service_s=2.5,
    )
    assert prefill == pytest.approx(
        profiles["generator_prefill"].corrected_service_s(8, 1.25)
    )
    assert decode == pytest.approx(
        profiles["generator_decode"].corrected_service_s(32, 2.5)
    )
    frontier = applicator.prefill_frontier_corrected_service_s(
        engine_stage="generator",
        prefill_stage="generator_prefill",
        decode_stage="generator_decode",
        resident_batch=8,
        completed_prefills=2,
        component_service_s=1.25,
    )
    full_prefill_overhead = (
        profiles["generator_prefill"].corrected_service_s(8, 1.25) - 1.25
    )
    assert frontier == pytest.approx(
        1.25 + full_prefill_overhead * 2.0 / 8.0
    )

    publish_stage_batch_curves(path, _pair())
    pair = resolve_frozen_stage_batch_generation(
        _hardware(), calibration_mode="strict",
    )
    with pytest.raises(
        StageBatchCurveValidationError,
        match="no direct disaggregated engine curve.*collocated profiles cannot",
    ):
        EngineCycleCurveApplicator(pair).corrected_service_s(
            engine_stage="generator",
            role="prefill",
            prefill_stage="generator_prefill",
            decode_stage="generator_decode",
            resident_batch=8,
            component_service_s=1.25,
        )


def test_disabled_special_mode_applicator_is_exactly_analytical_only(
    tmp_path,
    monkeypatch,
) -> None:
    _profile_path(tmp_path, monkeypatch)
    generation = resolve_frozen_stage_batch_generation(
        _hardware(), calibration_mode="uncalibrated_spec",
    )

    assert EngineCycleCurveApplicator(generation).corrected_service_s(
        engine_stage="generator",
        role="unified",
        prefill_stage="generator_prefill",
        decode_stage="generator_decode",
        resident_batch=256,
        component_service_s=7.5,
    ) == pytest.approx(7.5)


def test_compatibility_loader_uses_strict_generation_without_role_fallback(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _pair())

    assert load_stage_batch_curve(_hardware(), "generator").key.stage == (
        "generator"
    )
    with pytest.raises(
        StageBatchCurveValidationError,
        match="no direct disaggregated engine curve.*collocated profiles cannot",
    ):
        load_stage_batch_curve(_hardware(), "generator_prefill")
