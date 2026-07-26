"""CPU-only atomic stage-batch policy tests for uncalibrated specs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.calibration_paths import CALIBRATION_PROFILE_DIR_ENV
from rag_stack.cost_model.closed_loop_sim import (
    _STAGE_BATCH_RUNTIME_SELECTION_KEY,
    _pipeline_with_frozen_stage_batch_selection,
    _stage_batch_profiles_for_pipeline,
    SimResult,
    apply_closed_loop_simulation,
    build_sim_inputs,
)
from rag_stack.cost_model.engines import CallLatency
from rag_stack.cost_model.stage_batch_curve import (
    ANCHOR_BATCHES,
    COMPLETE_ENGINE_STAGE_KEYS,
    HELD_OUT_BATCH,
    PROFILE_BLOCK_KEY,
    REQUIRED_COLLOCATED_ENGINE_STAGES,
    StageBatchCurveKey,
    StageBatchCurveValidationError,
    StageBatchMeasurement,
    build_stage_batch_curve,
    publish_stage_batch_curves,
    resolve_uncalibrated_spec_stage_batch_selection,
    stage_batch_profiles_from_runtime_selection,
)
from rag_stack.types import SystemConfig


_SYSTEM_HARDWARE = "unit_a100x8_host"


def _hardware(*, mode: str | None = "uncalibrated_spec") -> SimpleNamespace:
    return SimpleNamespace(
        system_hardware=_SYSTEM_HARDWARE,
        gpu="A100_80GB_GPU",
        rag_cm=(None if mode is None else {"calibration_mode": mode}),
    )


def _profile_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "profiles"
    monkeypatch.setenv(CALIBRATION_PROFILE_DIR_ENV, str(root))
    path = root / "stage_batch" / f"{_SYSTEM_HARDWARE}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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


def test_missing_profile_disables_all_stage_curves(tmp_path, monkeypatch) -> None:
    _profile_path(tmp_path, monkeypatch)

    selection = resolve_uncalibrated_spec_stage_batch_selection(_hardware())

    assert selection["enabled"] is False
    assert selection["reason"] == "profile_file_absent"
    assert selection["profiles"] == {}
    assert selection["generation_sha256"] is None
    assert stage_batch_profiles_from_runtime_selection(
        selection, _hardware(),
    ) == {}
    json.dumps(selection, allow_nan=False)


def test_runtime_selection_rejects_unknown_fields(tmp_path, monkeypatch) -> None:
    _profile_path(tmp_path, monkeypatch)
    selection = resolve_uncalibrated_spec_stage_batch_selection(_hardware())
    selection["unexpected"] = True

    with pytest.raises(StageBatchCurveValidationError, match="exact schema"):
        stage_batch_profiles_from_runtime_selection(selection, _hardware())


def test_valid_collocated_pair_is_insufficient_in_special_mode(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _pair())

    selection = resolve_uncalibrated_spec_stage_batch_selection(_hardware())

    assert selection["enabled"] is False
    assert selection["reason"] == "collocated_only_generation"
    assert selection["profiles"] == {}


def test_known_key_incomplete_generation_disables_atomically(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _six())
    document = json.loads(path.read_text())
    del document[PROFILE_BLOCK_KEY]["profiles"]["generator_decode"]
    path.write_text(json.dumps(document))

    selection = resolve_uncalibrated_spec_stage_batch_selection(_hardware())

    assert selection["enabled"] is False
    assert selection["reason"] == "incomplete_generation"
    assert selection["profiles"] == {}


def test_complete_six_generation_round_trips_as_json_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _six())

    selection = resolve_uncalibrated_spec_stage_batch_selection(_hardware())
    encoded = json.dumps(selection, sort_keys=True, allow_nan=False)
    restored = json.loads(encoded)
    profiles = stage_batch_profiles_from_runtime_selection(
        restored, _hardware(),
    )

    assert selection["enabled"] is True
    assert selection["reason"] == "complete_six_profile_generation"
    assert set(profiles) == set(COMPLETE_ENGINE_STAGE_KEYS)


def test_present_but_corrupt_pair_remains_an_error(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _pair())
    document = json.loads(path.read_text())
    document[PROFILE_BLOCK_KEY]["profiles"]["generator"][
        "content_sha256"
    ] = "f" * 64
    path.write_text(json.dumps(document))

    with pytest.raises(
        StageBatchCurveValidationError,
        match="content digest mismatch",
    ):
        resolve_uncalibrated_spec_stage_batch_selection(_hardware())


def test_incomplete_generation_cannot_hide_corrupt_present_profile(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _six())
    document = json.loads(path.read_text())
    profiles = document[PROFILE_BLOCK_KEY]["profiles"]
    del profiles["generator_decode"]
    profiles["generator"]["content_sha256"] = "f" * 64
    path.write_text(json.dumps(document))

    with pytest.raises(
        StageBatchCurveValidationError,
        match="content digest mismatch",
    ):
        resolve_uncalibrated_spec_stage_batch_selection(_hardware())


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("unexpected", "value", "exact schema"),
        ("system_hardware", "another_host", "hardware identity mismatch"),
    ],
)
def test_present_block_schema_and_hardware_errors_remain_errors(
    tmp_path,
    monkeypatch,
    field,
    value,
    error,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _pair())
    document = json.loads(path.read_text())
    document[PROFILE_BLOCK_KEY][field] = value
    path.write_text(json.dumps(document))

    with pytest.raises(StageBatchCurveValidationError, match=error):
        resolve_uncalibrated_spec_stage_batch_selection(_hardware())


def test_present_legacy_provenance_field_remains_an_error(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _pair())
    document = json.loads(path.read_text())
    document[PROFILE_BLOCK_KEY]["profiles"]["generator"]["provenance"] = {
        "source_path": "/tmp/raw.jsonl"
    }
    path.write_text(json.dumps(document))

    with pytest.raises(
        StageBatchCurveValidationError,
        match="exact schema",
    ):
        resolve_uncalibrated_spec_stage_batch_selection(_hardware())


def test_present_malformed_json_remains_an_error(tmp_path, monkeypatch) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    path.write_text("{")

    with pytest.raises(
        StageBatchCurveValidationError,
        match="cannot read stage-batch profile file",
    ):
        resolve_uncalibrated_spec_stage_batch_selection(_hardware())


def test_hardware_policy_wins_and_selection_is_frozen_for_run(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    hardware = _hardware(mode="uncalibrated_spec")
    source_pipeline = {
        "system": {"rag_cm": {"calibration_mode": "strict"}},
    }

    first = _pipeline_with_frozen_stage_batch_selection(
        source_pipeline, hardware,
    )
    first_selection = first["_runtime"][_STAGE_BATCH_RUNTIME_SELECTION_KEY]
    assert first_selection["enabled"] is False
    assert _stage_batch_profiles_for_pipeline(first, hardware) == {}

    # A newly published generation cannot change this run's already-frozen
    # all-or-nothing choice.
    publish_stage_batch_curves(path, _six())
    second = _pipeline_with_frozen_stage_batch_selection(
        source_pipeline, hardware,
    )
    assert second["_runtime"][_STAGE_BATCH_RUNTIME_SELECTION_KEY] == (
        first_selection
    )
    assert _stage_batch_profiles_for_pipeline(second, hardware) == {}

    # A distinct replay/run gets a distinct SystemConfig object and can see
    # the now-complete generation.
    fresh = _pipeline_with_frozen_stage_batch_selection(
        source_pipeline,
        _hardware(mode="uncalibrated_spec"),
    )
    assert fresh["_runtime"][_STAGE_BATCH_RUNTIME_SELECTION_KEY][
        "enabled"
    ] is True
    json.dumps(fresh, allow_nan=False)


def test_hardware_strict_mode_requires_one_frozen_generation(
    tmp_path,
    monkeypatch,
) -> None:
    _profile_path(tmp_path, monkeypatch)
    pipeline_special = {
        "system": {"rag_cm": {"calibration_mode": "uncalibrated_spec"}},
    }
    hardware = _hardware(mode="strict")

    with pytest.raises(
        StageBatchCurveValidationError,
        match="profile file is absent",
    ):
        _pipeline_with_frozen_stage_batch_selection(
            pipeline_special,
            hardware,
        )


def test_pipeline_mode_is_fallback_when_hardware_has_no_policy(
    tmp_path,
    monkeypatch,
) -> None:
    _profile_path(tmp_path, monkeypatch)
    pipeline_special = {
        "system": {"rag_cm": {"calibration_mode": "uncalibrated_spec"}},
    }

    resolved = _pipeline_with_frozen_stage_batch_selection(
        pipeline_special,
        _hardware(mode=None),
    )

    assert resolved["_runtime"][_STAGE_BATCH_RUNTIME_SELECTION_KEY][
        "enabled"
    ] is False


def test_disabled_special_mode_attaches_no_engine_cycle_overhead(
    tmp_path,
    monkeypatch,
) -> None:
    _profile_path(tmp_path, monkeypatch)

    def forbidden_loader(*_args, **_kwargs):
        raise AssertionError("disabled special mode must not load one stage")

    monkeypatch.setattr(
        "rag_stack.cost_model.stage_batch_curve.load_stage_batch_curve",
        forbidden_loader,
    )
    hardware = SystemConfig(
        num_cores=8,
        system_hardware=_SYSTEM_HARDWARE,
        gpu="A100_80GB_GPU",
        rag_cm={"calibration_mode": "uncalibrated_spec"},
    )
    latency = CallLatency(
        prefill_s=lambda tokens, batch, tp=1, pp=1: (
            float(tokens) * float(batch) / 100_000.0
        ),
        decode_s=lambda inp, out, batch, tp=1, pp=1: (
            float(batch) / 1_000_000.0
        ),
    )

    inputs, reason = build_sim_inputs(
        {
            "system": {
                "rag_cm": {"calibration_mode": "uncalibrated_spec"},
                "batching": {"dynamic_timeout_s": 0.0},
                "communication": {"include_non_pd_comm": False},
            },
        },
        hardware,
        [[{"stage": "generator", "input_tokens": 8, "output_tokens": 2}]],
        [["generator_prefill", "generator_decode"]],
        batch_size_request=2,
        batch_size_decode=2,
        latency=latency,
        prefix_caching=False,
    )

    assert reason is None
    assert inputs is not None
    engines = inputs[1]
    assert len(engines) == 1
    assert getattr(
        engines[0].prefill_batch_s,
        "_rag_cm_cycle_corrected_service_s",
        None,
    ) is None
    assert getattr(
        engines[0].decode_step_s,
        "_rag_cm_cycle_corrected_service_s",
        None,
    ) is None


def test_apply_closed_loop_stamps_one_frozen_policy_on_every_candidate(
    tmp_path,
    monkeypatch,
) -> None:
    _profile_path(tmp_path, monkeypatch)
    frame = pd.DataFrame({
        "qps": [100.0, 80.0],
        "latency_s": [0.1, 0.2],
        "batch_size_request": [8, 16],
        "batch_size_decode": [8, 16],
        "collocation_strategy": [
            [["generator_prefill", "generator_decode"]],
            [["generator_prefill"], ["generator_decode"]],
        ],
    })

    def fake_builder(*_args, **_kwargs):
        return ([], [], [], 1, 2, None), None

    def fake_simulate(*_args, **kwargs):
        population = int(kwargs["population"])
        return SimResult(
            qps=7.0,
            latency_mean_s=0.5,
            engine_fill_mean={},
            stage_batch_mean={},
            window_s=1.0,
            completions=population,
            population=population,
        )

    result = apply_closed_loop_simulation(
        frame,
        pipeline_config={"system": {}},
        hardware=_hardware(mode="uncalibrated_spec"),
        traces=[[{"stage": "generator"}]],
        latency=object(),
        builder_fn=fake_builder,
        simulate_fn=fake_simulate,
    )

    assert result["cm_calibration_mode"].tolist() == [
        "uncalibrated_spec",
        "uncalibrated_spec",
    ]
    assert result["stage_batch_profile_enabled"].tolist() == [False, False]
    assert result["stage_batch_profile_reason"].tolist() == [
        "profile_file_absent",
        "profile_file_absent",
    ]
    assert result["stage_batch_profile_generation_sha256"].isna().all()
    assert "stage_batch_profile_provenance_sha256" not in result.columns
