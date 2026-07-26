from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
from types import SimpleNamespace

import pytest

from rag_stack.cost_model.llm_sim.calibration.main_llm import (
    generator_parallel,
    generator_request_contract,
)
from rag_stack.calibration_paths import CALIBRATION_PROFILE_DIR_ENV
from rag_stack.cost_model.stage_batch_curve import (
    ANCHOR_BATCHES,
    COMPLETE_ENGINE_STAGE_KEYS,
    DISAGG_ROLE_STAGE_KEYS,
    HELD_OUT_BATCH,
    PROFILE_BLOCK_KEY,
    REQUIRED_COLLOCATED_ENGINE_STAGES,
    StageBatchCurveDomainError,
    StageBatchCurveKey,
    StageBatchCurveProfile,
    StageBatchCurveValidationError,
    StageBatchMeasurement,
    build_stage_batch_curve,
    load_stage_batch_curve,
    publish_stage_batch_curves,
    validate_system_hardware,
)


_COLLOCATED_ENGINE_STAGES = (
    "generator",
    "query_expansion",
)
_SYSTEM_HARDWARE = "sgs_h100x4_epyc9124"


def _key(stage: str = "generator") -> StageBatchCurveKey:
    return StageBatchCurveKey(_SYSTEM_HARDWARE, stage)


def _generator_request_contract() -> dict:
    template = "fixture-chat-template"
    token_ids = list(range(10))
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
        "rendered_prompt_tokens": 10,
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


def _measurements(
    *,
    key: StageBatchCurveKey | None = None,
    cap: int = 256,
    held_out_overhead_s: float | None = None,
    omitted_anchor_batches: frozenset[int] = frozenset(),
) -> list[StageBatchMeasurement]:
    key = key or _key()
    # Deliberately non-uniform slopes so interpolation is observable. B128 lies
    # one third of the way from B64 to B256.
    overhead = {
        1: 0.001,
        8: 0.004,
        32: 0.010,
        64: 0.018,
        256: 0.066,
    }
    predicted_128 = 0.018 + (128 - 64) / (256 - 64) * (0.066 - 0.018)
    overhead[HELD_OUT_BATCH] = (
        predicted_128
        if held_out_overhead_s is None
        else held_out_overhead_s
    )
    return [
        StageBatchMeasurement(
            key=key,
            target_batch=batch,
            engine_max_num_seqs=cap,
            observed_resident_batch=batch,
            reference_component_service_s=0.5 + 0.001 * batch,
            overhead_s=overhead[batch],
            observed_total_service_s=(
                0.5 + 0.001 * batch + overhead[batch]
            ),
            sample_count=5,
        )
        for batch in (*ANCHOR_BATCHES, HELD_OUT_BATCH)
        if batch not in omitted_anchor_batches
    ]


def _profile(**kwargs) -> StageBatchCurveProfile:
    key = kwargs.pop("key", _key())
    measurements = kwargs.pop("measurements", _measurements(key=key))
    return build_stage_batch_curve(
        key=key,
        measurements=measurements,
        **kwargs,
    )


def _refresh_content_sha256(document: dict) -> None:
    payload = {
        key: value
        for key, value in document.items()
        if key != "content_sha256"
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    document["content_sha256"] = hashlib.sha256(encoded).hexdigest()


def _collocated_stage_profiles() -> tuple[StageBatchCurveProfile, ...]:
    return tuple(_profile(key=_key(stage)) for stage in _COLLOCATED_ENGINE_STAGES)


def _complete_stage_profiles() -> tuple[StageBatchCurveProfile, ...]:
    return tuple(_profile(key=_key(stage)) for stage in COMPLETE_ENGINE_STAGE_KEYS)


def _profile_path(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "profiles"
    monkeypatch.setenv(CALIBRATION_PROFILE_DIR_ENV, str(root))
    path = root / "stage_batch" / f"{_SYSTEM_HARDWARE}.json"
    path.parent.mkdir(parents=True)
    return path


def _hardware(name: str = _SYSTEM_HARDWARE) -> SimpleNamespace:
    return SimpleNamespace(system_hardware=name, gpu="H100_GPU")


def test_key_is_exactly_system_hardware_and_stage() -> None:
    key = _key()
    assert key.to_dict() == {
        "system_hardware": _SYSTEM_HARDWARE,
        "stage": "generator",
    }

    for forbidden in ("model", "tp", "pp", "deployment"):
        document = {**key.to_dict(), forbidden: "forbidden"}
        with pytest.raises(StageBatchCurveValidationError, match="exact schema"):
            StageBatchCurveKey.from_dict(document)

    legacy = dict(key.to_dict())
    legacy["hardware_key"] = legacy.pop("system_hardware")
    with pytest.raises(StageBatchCurveValidationError, match="exact schema"):
        StageBatchCurveKey.from_dict(legacy)


def test_unknown_serving_stage_is_rejected() -> None:
    with pytest.raises(StageBatchCurveValidationError, match="unknown vLLM"):
        StageBatchCurveKey(_SYSTEM_HARDWARE, "made_up_stage")


@pytest.mark.parametrize(
    "identity",
    (
        None,
        "",
        ".",
        "..",
        "../system",
        "system/name",
        "system\\name",
        " system",
        "system ",
        "system name",
        "système",
        "x" * 256,
    ),
)
def test_system_hardware_identity_must_be_strictly_path_safe(identity) -> None:
    with pytest.raises(StageBatchCurveValidationError, match="system_hardware"):
        validate_system_hardware(identity)


def test_system_hardware_validator_preserves_one_canonical_identity() -> None:
    assert validate_system_hardware(_SYSTEM_HARDWARE) == _SYSTEM_HARDWARE


def test_curve_has_fixed_sparse_anchors_and_real_b128_holdout() -> None:
    profile = _profile()
    assert tuple(anchor.batch_size for anchor in profile.anchors) == ANCHOR_BATCHES
    assert HELD_OUT_BATCH not in {anchor.batch_size for anchor in profile.anchors}
    assert profile.validation.batch_size == HELD_OUT_BATCH
    assert profile.validation.passed is True
    assert profile.corrected_service_s(
        HELD_OUT_BATCH,
        profile.validation.reference_component_service_s,
    ) == pytest.approx(
        profile.validation.predicted_total_service_s
    )
    assert set(profile.to_dict()) == {
        "key",
        "batch_unit",
        "interpolation",
        "max_batch_size",
        "anchors",
        "validation",
        "content_sha256",
    }
    assert all(set(anchor.to_dict()) == {
        "batch_size",
        "observed_total_service_s",
        "reference_component_service_s",
        "overhead_s",
        "engine_max_num_seqs",
        "observed_resident_batch",
        "sample_count",
    } for anchor in profile.anchors)


def test_curve_is_piecewise_linear_and_fails_above_256() -> None:
    profile = _profile()
    # B16 is one third of the interval B8 -> B32.
    component = 1.0
    assert profile.corrected_service_s(16, component) == pytest.approx(
        component + 0.004 + (16 - 8) / (32 - 8) * (0.010 - 0.004)
    )
    assert profile.corrected_service_s(0, component) == component
    assert profile.corrected_service_s(256, component) == pytest.approx(
        component + 0.066
    )
    with pytest.raises(StageBatchCurveDomainError, match="exceeds calibrated maximum"):
        profile.corrected_service_s(257, component)


def test_runtime_adds_nonnegative_overhead_and_never_reduces_component() -> None:
    profile = _profile()

    assert profile.anchors[0].overhead_s == pytest.approx(0.001)
    assert profile.corrected_service_s(1, 0.5) == pytest.approx(0.501)
    for batch in range(257):
        assert profile.corrected_service_s(batch, 0.5) >= 0.5


def test_negative_anchor_evidence_is_rejected_not_clamped() -> None:
    with pytest.raises(
        StageBatchCurveValidationError,
        match="finite and non-negative",
    ):
        StageBatchMeasurement(
            key=_key(),
            target_batch=8,
            engine_max_num_seqs=256,
            observed_resident_batch=8,
            observed_total_service_s=0.49,
            reference_component_service_s=0.5,
            overhead_s=-0.01,
            sample_count=3,
        )


@pytest.mark.parametrize("component", (0.0, -1.0, float("inf"), float("nan")))
def test_runtime_requires_positive_finite_component_service(component: float) -> None:
    with pytest.raises(StageBatchCurveDomainError, match="component_service_s"):
        _profile().corrected_service_s(8, component)


def test_publication_requires_domain_endpoints_and_holdout() -> None:
    measurements = _measurements()
    with pytest.raises(StageBatchCurveValidationError, match="every required"):
        build_stage_batch_curve(
            key=_key(),
            measurements=[m for m in measurements if m.target_batch != 256],
        )

    duplicate = [*measurements, measurements[0]]
    with pytest.raises(StageBatchCurveValidationError, match="duplicate measurement"):
        build_stage_batch_curve(
            key=_key(),
            measurements=duplicate,
        )


def test_missing_interior_anchor_fails_complete_fit() -> None:
    with pytest.raises(
        StageBatchCurveValidationError,
        match=r"every required sparse anchor.*missing=\[8\]",
    ):
        _profile(
            measurements=_measurements(
                omitted_anchor_batches=frozenset({8})
            ),
        )


def test_heldout_negative_overhead_evidence_is_rejected_not_clamped() -> None:
    with pytest.raises(
        StageBatchCurveValidationError,
        match="finite and non-negative",
    ):
        _measurements(held_out_overhead_s=-0.001)


def test_measurement_rejects_cap_below_target() -> None:
    with pytest.raises(StageBatchCurveValidationError, match="cannot admit"):
        StageBatchMeasurement(
            key=_key(),
            target_batch=256,
            engine_max_num_seqs=64,
            observed_resident_batch=256,
            observed_total_service_s=0.6,
            reference_component_service_s=0.5,
            overhead_s=0.1,
            sample_count=3,
        )


def test_measurement_requires_exact_observed_resident_batch() -> None:
    with pytest.raises(StageBatchCurveValidationError, match="must exactly equal"):
        StageBatchMeasurement(
            key=_key(),
            target_batch=256,
            engine_max_num_seqs=256,
            observed_resident_batch=255,
            observed_total_service_s=0.6,
            reference_component_service_s=0.5,
            overhead_s=0.1,
            sample_count=3,
        )


def test_measurement_requires_exact_service_decomposition() -> None:
    with pytest.raises(StageBatchCurveValidationError, match="must prove"):
        StageBatchMeasurement(
            key=_key(),
            target_batch=8,
            engine_max_num_seqs=256,
            observed_resident_batch=8,
            observed_total_service_s=0.6,
            reference_component_service_s=0.5,
            overhead_s=0.05,
            sample_count=3,
        )


def test_all_measurements_must_have_the_published_hardware_stage_key() -> None:
    measurements = _measurements()
    measurements[-1] = StageBatchMeasurement(
        key=_key("query_expansion"),
        target_batch=HELD_OUT_BATCH,
        engine_max_num_seqs=256,
        observed_resident_batch=HELD_OUT_BATCH,
        observed_total_service_s=measurements[-1].observed_total_service_s,
        reference_component_service_s=(
            measurements[-1].reference_component_service_s
        ),
        overhead_s=measurements[-1].overhead_s,
        sample_count=5,
    )
    with pytest.raises(StageBatchCurveValidationError, match="exact published"):
        build_stage_batch_curve(
            key=_key(),
            measurements=measurements,
        )


def test_b128_is_a_real_publication_gate_not_a_fit_point() -> None:
    measurements = _measurements(held_out_overhead_s=0.5)
    with pytest.raises(StageBatchCurveValidationError, match="publication limit"):
        build_stage_batch_curve(
            key=_key(),
            measurements=measurements,
        )


def test_b128_prediction_is_recomputed_after_content_rehash() -> None:
    document = _profile().to_dict()
    validation = document["validation"]
    component = validation["reference_component_service_s"]
    validation["observed_total_service_s"] *= 10.0
    validation["observed_overhead_s"] = (
        validation["observed_total_service_s"] - component
    )
    validation["predicted_overhead_s"] = validation[
        "observed_overhead_s"
    ]
    validation["predicted_total_service_s"] = validation[
        "observed_total_service_s"
    ]
    validation["absolute_error_s"] = 0.0
    validation["relative_error"] = 0.0
    validation["passed"] = True
    _refresh_content_sha256(document)

    with pytest.raises(
        StageBatchCurveValidationError,
        match="fixed anchor interpolation",
    ):
        StageBatchCurveProfile.from_dict(document)


def test_b128_publication_limit_cannot_be_relaxed_after_content_rehash() -> None:
    document = _profile().to_dict()
    document["validation"]["max_relative_error"] = 1.0
    _refresh_content_sha256(document)

    with pytest.raises(
        StageBatchCurveValidationError,
        match="fixed publication limit",
    ):
        StageBatchCurveProfile.from_dict(document)


def test_b128_publication_limit_cannot_be_overridden_at_build_time() -> None:
    with pytest.raises(
        StageBatchCurveValidationError,
        match="fixed at 0.35",
    ):
        build_stage_batch_curve(
            key=_key(),
            measurements=_measurements(),
            max_held_out_relative_error=1.0,
        )


def test_content_digest_is_verified() -> None:
    profile = _profile()
    document = profile.to_dict()
    restored = StageBatchCurveProfile.from_dict(document)
    assert restored.to_dict() == document

    content_tamper = deepcopy(document)
    content_tamper["anchors"][2]["overhead_s"] += 0.001
    content_tamper["anchors"][2]["observed_total_service_s"] += 0.001
    with pytest.raises(StageBatchCurveValidationError, match="content digest mismatch"):
        StageBatchCurveProfile.from_dict(content_tamper)


def test_legacy_provenance_fields_are_rejected_without_fallback() -> None:
    document = _profile().to_dict()
    document["provenance"] = {"source_path": "/tmp/raw.jsonl"}
    document["provenance_sha256"] = "f" * 64
    with pytest.raises(StageBatchCurveValidationError, match="exact schema"):
        StageBatchCurveProfile.from_dict(document)


def test_serialized_profile_contains_overhead_but_no_fitted_coefficients() -> None:
    document = _profile().to_dict()

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    names = set(keys(document))
    assert "overhead_s" in names
    assert names.isdisjoint({
        "coefficient",
        "coefficients",
        "alpha",
        "beta",
        "tp",
        "pp",
        "model",
        "deployment",
        "provenance",
        "source_path",
    })


def test_non_integer_runtime_batch_is_rejected() -> None:
    with pytest.raises(StageBatchCurveDomainError, match="must be an integer"):
        _profile().corrected_service_s(16.0, 1.0)  # type: ignore[arg-type]


def test_legacy_signed_correction_document_is_rejected_without_fallback() -> None:
    document = _profile().to_dict()
    document["interpolation"] = (
        "piecewise_linear_signed_correction_in_resident_batch"
    )
    for anchor in document["anchors"]:
        anchor["correction_s"] = anchor.pop("overhead_s")
    validation = document["validation"]
    validation["observed_correction_s"] = validation.pop(
        "observed_overhead_s"
    )
    validation["predicted_correction_s"] = validation.pop(
        "predicted_overhead_s"
    )

    with pytest.raises(
        StageBatchCurveValidationError,
        match="fields do not match the exact schema",
    ):
        StageBatchCurveProfile.from_dict(document)


def test_profile_with_unknown_contract_field_is_rejected() -> None:
    document = _profile().to_dict()
    document["retired_version_marker"] = 3

    with pytest.raises(
        StageBatchCurveValidationError,
        match="fields do not match the exact schema",
    ):
        StageBatchCurveProfile.from_dict(document)


def test_publish_replaces_stage_file_and_loads_two_collocated_identities(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    path.write_text(json.dumps({
        "agnostic__reference__bf16": {"preserved": True},
        "_tp_pp_efficiency": {"retired": True},
        "_vllm_scheduler_step_residual": {"retired": True},
    }))
    profiles = _collocated_stage_profiles()
    assert set(_COLLOCATED_ENGINE_STAGES) == set(
        REQUIRED_COLLOCATED_ENGINE_STAGES
    )

    publish_stage_batch_curves(path, profiles)

    document = json.loads(path.read_text())
    assert set(document) == {PROFILE_BLOCK_KEY}
    assert document[PROFILE_BLOCK_KEY]["system_hardware"] == _SYSTEM_HARDWARE
    assert set(document[PROFILE_BLOCK_KEY]["profiles"]) == set(
        _COLLOCATED_ENGINE_STAGES
    )

    expected = {profile.key.stage: profile for profile in profiles}
    for stage in _COLLOCATED_ENGINE_STAGES:
        loaded = load_stage_batch_curve(_hardware(), stage)
        assert loaded.key == StageBatchCurveKey(_SYSTEM_HARDWARE, stage)
        assert loaded.content_sha256 == expected[stage].content_sha256


def test_publish_rejects_partial_collocated_or_disagg_generations(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    profiles = _collocated_stage_profiles()

    with pytest.raises(StageBatchCurveValidationError, match="exactly cover"):
        publish_stage_batch_curves(path, profiles[:-1])

    extra = _profile(key=_key("generator_prefill"))
    with pytest.raises(StageBatchCurveValidationError, match="exactly cover"):
        publish_stage_batch_curves(path, (*profiles, extra))


def test_publish_and_load_complete_six_profile_generation(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    profiles = _complete_stage_profiles()

    publish_stage_batch_curves(path, profiles)

    document = json.loads(path.read_text())
    assert set(document[PROFILE_BLOCK_KEY]["profiles"]) == set(
        COMPLETE_ENGINE_STAGE_KEYS
    )
    expected = {profile.key.stage: profile for profile in profiles}
    for stage in COMPLETE_ENGINE_STAGE_KEYS:
        loaded = load_stage_batch_curve(_hardware(), stage)
        assert loaded.key == StageBatchCurveKey(_SYSTEM_HARDWARE, stage)
        assert loaded.content_sha256 == expected[stage].content_sha256


def test_query_expansion_role_alias_must_match_generator_role_curve(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    profiles = list(_complete_stage_profiles())
    index = next(
        index for index, profile in enumerate(profiles)
        if profile.key.stage == "query_expansion_prefill"
    )
    alias = profiles[index]
    measurements = _measurements(key=alias.key)
    shifted = [
        StageBatchMeasurement(
            key=item.key,
            target_batch=item.target_batch,
            engine_max_num_seqs=item.engine_max_num_seqs,
            observed_resident_batch=item.observed_resident_batch,
            observed_total_service_s=item.observed_total_service_s + 0.001,
            reference_component_service_s=item.reference_component_service_s,
            overhead_s=item.overhead_s + 0.001,
            sample_count=item.sample_count,
        )
        for item in measurements
    ]
    profiles[index] = _profile(key=alias.key, measurements=shifted)

    with pytest.raises(
        StageBatchCurveValidationError,
        match="query-expansion prefill anchors/validation must exactly equal",
    ):
        publish_stage_batch_curves(path, profiles)


def test_publish_requires_query_expansion_to_equal_generator_measurements(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    profiles = list(_collocated_stage_profiles())
    query_index = next(
        index for index, profile in enumerate(profiles)
        if profile.key.stage == "query_expansion"
    )
    query = profiles[query_index]
    shifted = [
        StageBatchMeasurement(
            key=measurement.key,
            target_batch=measurement.target_batch,
            engine_max_num_seqs=measurement.engine_max_num_seqs,
            observed_resident_batch=measurement.observed_resident_batch,
            observed_total_service_s=(
                measurement.observed_total_service_s + 0.01
            ),
            reference_component_service_s=measurement.reference_component_service_s,
            overhead_s=measurement.overhead_s + 0.01,
            sample_count=measurement.sample_count,
        )
        for measurement in _measurements(key=query.key)
    ]
    profiles[query_index] = _profile(
        key=query.key,
        measurements=shifted,
    )

    with pytest.raises(StageBatchCurveValidationError, match="anchors/validation"):
        publish_stage_batch_curves(path, profiles)


def test_collocated_measurement_normalizes_full_cohorts_without_phase_fit() -> None:
    batches = (*ANCHOR_BATCHES, HELD_OUT_BATCH)
    component = {
        batch: {
            "scheduler_cycle_count": 4,
            # Publication must median the complete cohort totals first, then
            # divide by the exact scheduler-cycle count: median=8 -> 2 s.
            "cohort_total_service_s": [4.0, 100.0, 8.0],
            "cohort_count": 3,
        }
        for batch in batches
    }
    stage = {
        batch: {
            "collocated_engine_cycle_count": 4,
            "collocated_engine_cycle_service_s": 3.0,
            "sample_count": 7,
        }
        for batch in batches
    }

    key, measurements = (
        generator_parallel._measurements_for_collocated_engine(
            component_cohort_measurements=component,
            stage_service_measurements=stage,
            system_hardware=_SYSTEM_HARDWARE,
            stage="generator",
        )
    )

    assert key.stage == "generator"
    assert len(measurements) == len(batches)
    assert all(
        measurement.reference_component_service_s == pytest.approx(2.0)
        and measurement.observed_total_service_s == pytest.approx(3.0)
        and measurement.overhead_s == pytest.approx(1.0)
        and measurement.sample_count == 3
        for measurement in measurements
    )


def test_generator_parallel_fit_publishes_pricing_only_profiles(
    tmp_path,
    monkeypatch,
) -> None:
    import rag_stack.cost_model.llm_sim.calibration as calibration
    from rag_stack.cost_model.llm_sim.calibration.main_llm import vllm_probe

    profile_root = tmp_path / "profiles"
    monkeypatch.setenv(CALIBRATION_PROFILE_DIR_ENV, str(profile_root))
    data_root = tmp_path / "data"
    stage_raw = data_root / "stage.jsonl"
    component_raw = data_root / "stage.component.jsonl"
    stage_contract = Path(f"{stage_raw}.contract.json")
    component_contract = Path(f"{component_raw}.contract.json")
    for index, path in enumerate((stage_raw, component_raw), start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"evidence-{index}\n".encode())
    request_contracts = {"10": _generator_request_contract()}
    common_workload = {
        "target_batches": list(ANCHOR_BATCHES) + [HELD_OUT_BATCH],
        "prompt_tokens": 10,
        "output_tokens": 20,
        "request_contracts": request_contracts,
    }
    for path, scope in (
        (stage_contract, "stage_only"),
        (component_contract, "component_only"),
    ):
        path.write_text(
            json.dumps({
                "scope": scope,
                "workload": {
                    **common_workload,
                    (
                        "out_one_diagnostic_output_tokens"
                        if scope == "stage_only"
                        else "prefill_probe_output_tokens"
                    ): 1,
                },
            }) + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        generator_parallel, "_data_path", lambda _hardware: str(stage_raw),
    )
    monkeypatch.setattr(
        generator_parallel,
        "_component_data_path",
        lambda _hardware: str(component_raw),
    )
    monkeypatch.setattr(
        generator_parallel,
        "_system_hardware_from_config",
        lambda _config: _SYSTEM_HARDWARE,
    )
    monkeypatch.setattr(
        calibration,
        "hardware_from_config",
        lambda _config: (_hardware(), {}),
    )
    monkeypatch.setattr(
        vllm_probe,
        "read_stage_cycle_observations",
        lambda _path: (SimpleNamespace(model="Qwen/Test"),),
    )
    monkeypatch.setattr(
        vllm_probe,
        "read_component_full_cohort_service_measurements",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        vllm_probe,
        "read_stage_active_service_measurements",
        lambda _path: {},
    )

    def fake_measurements(
        *,
        system_hardware,
        stage,
        **_kwargs,
    ):
        key = StageBatchCurveKey(system_hardware, stage)
        return key, _measurements(key=key)

    monkeypatch.setattr(
        generator_parallel,
        "_measurements_for_collocated_engine",
        fake_measurements,
    )
    monkeypatch.setattr(
        generator_parallel, "_remove_retired_component_blocks", lambda _hw: None,
    )

    generator_parallel.fit("unused.yaml")

    profile_path = (
        profile_root / "stage_batch" / f"{_SYSTEM_HARDWARE}.json"
    )
    document = json.loads(profile_path.read_text())
    profiles = document[PROFILE_BLOCK_KEY]["profiles"]
    for profile in profiles.values():
        assert set(profile) == {
            "key",
            "batch_unit",
            "interpolation",
            "max_batch_size",
            "anchors",
            "validation",
            "content_sha256",
        }
        assert all(set(anchor) == {
            "batch_size",
            "observed_total_service_s",
            "reference_component_service_s",
            "overhead_s",
            "engine_max_num_seqs",
            "observed_resident_batch",
            "sample_count",
        } for anchor in profile["anchors"])
        encoded = json.dumps(profile, sort_keys=True)
        assert str(tmp_path) not in encoded
        assert "request_contract" not in encoded


def test_publish_rejects_mixed_system_hardware_identities(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    profiles = list(_collocated_stage_profiles())
    stage = profiles[-1].key.stage
    profiles[-1] = _profile(
        key=StageBatchCurveKey("another_system", stage)
    )

    with pytest.raises(StageBatchCurveValidationError, match="one system_hardware"):
        publish_stage_batch_curves(path, profiles)


@pytest.mark.parametrize(
    "relative_path",
    (
        "stage_batch/wrong_system.json",
        "wrong_directory/sgs_h100x4_epyc9124.json",
    ),
)
def test_publish_rejects_noncanonical_parent_or_filename(
    tmp_path,
    monkeypatch,
    relative_path: str,
) -> None:
    canonical = _profile_path(tmp_path, monkeypatch)
    wrong = canonical.parents[1] / relative_path

    with pytest.raises(StageBatchCurveValidationError, match="path must be canonical"):
        publish_stage_batch_curves(wrong, _collocated_stage_profiles())


def test_publish_document_keeps_stage_profiles_exact_and_independently_digested(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    profiles = _collocated_stage_profiles()
    publish_stage_batch_curves(path, profiles)

    document = json.loads(path.read_text())
    stored = document[PROFILE_BLOCK_KEY]["profiles"]
    assert len({item["content_sha256"] for item in stored.values()}) == 2
    for map_stage, raw in stored.items():
        assert raw["key"] == {
            "system_hardware": _SYSTEM_HARDWARE,
            "stage": map_stage,
        }
        assert StageBatchCurveProfile.from_dict(raw).key.stage == map_stage


def test_disk_load_rejects_content_digest_tampering(tmp_path, monkeypatch) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _collocated_stage_profiles())
    document = json.loads(path.read_text())
    document[PROFILE_BLOCK_KEY]["profiles"]["generator"]["anchors"][2][
        "overhead_s"
    ] += 0.001
    document[PROFILE_BLOCK_KEY]["profiles"]["generator"]["anchors"][2][
        "observed_total_service_s"
    ] += 0.001
    path.write_text(json.dumps(document))

    with pytest.raises(StageBatchCurveValidationError, match="content digest mismatch"):
        load_stage_batch_curve(_hardware(), "generator")


def test_disk_load_rejects_legacy_provenance_fields(tmp_path, monkeypatch) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _collocated_stage_profiles())
    document = json.loads(path.read_text())
    document[PROFILE_BLOCK_KEY]["profiles"]["generator"]["provenance"] = {
        "source_path": "/tmp/raw.jsonl"
    }
    path.write_text(json.dumps(document))

    with pytest.raises(StageBatchCurveValidationError, match="exact schema"):
        load_stage_batch_curve(_hardware(), "generator")


def test_load_validates_complete_generation_not_only_requested_curve(
    tmp_path, monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _collocated_stage_profiles())
    document = json.loads(path.read_text())
    document[PROFILE_BLOCK_KEY]["profiles"]["query_expansion"][
        "content_sha256"
    ] = "f" * 64
    path.write_text(json.dumps(document))

    with pytest.raises(StageBatchCurveValidationError, match="digest mismatch"):
        load_stage_batch_curve(_hardware(), "generator")


def test_load_fails_closed_when_profile_file_is_missing(tmp_path, monkeypatch) -> None:
    _profile_path(tmp_path, monkeypatch)

    with pytest.raises(StageBatchCurveValidationError, match="profile file is absent"):
        load_stage_batch_curve(_hardware(), "generator")


def test_load_never_falls_back_to_gpu_sku_for_host_stage_correction(
    tmp_path, monkeypatch,
) -> None:
    _profile_path(tmp_path, monkeypatch)
    gpu_only = SimpleNamespace(gpu="H100_GPU", system_hardware="")

    with pytest.raises(StageBatchCurveValidationError, match="system_hardware"):
        load_stage_batch_curve(gpu_only, "generator")


def test_load_fails_closed_when_stage_curve_is_missing(tmp_path, monkeypatch) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _collocated_stage_profiles())
    document = json.loads(path.read_text())
    del document[PROFILE_BLOCK_KEY]["profiles"]["query_expansion"]
    path.write_text(json.dumps(document))

    with pytest.raises(
        StageBatchCurveValidationError,
        match="exact collocated pair or the exact complete six-profile",
    ):
        load_stage_batch_curve(_hardware(), "query_expansion")


def test_disagg_role_never_falls_back_to_collocated_curve(
    tmp_path, monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _collocated_stage_profiles())

    with pytest.raises(
        StageBatchCurveValidationError,
        match="no direct disaggregated engine curve.*collocated profiles cannot",
    ):
        load_stage_batch_curve(_hardware(), "generator_prefill")


def test_load_fails_closed_when_profile_block_is_missing(tmp_path, monkeypatch) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    path.write_text(json.dumps({"agnostic__reference__bf16": {"preserved": True}}))

    with pytest.raises(StageBatchCurveValidationError, match="authoritative block"):
        load_stage_batch_curve(_hardware(), "generator")


def test_load_rejects_unrelated_payload_in_stage_profile_file(
    tmp_path, monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _collocated_stage_profiles())
    document = json.loads(path.read_text())
    document["unrelated"] = {"coefficient": 1.0}
    path.write_text(json.dumps(document))

    with pytest.raises(StageBatchCurveValidationError, match="only its authoritative"):
        load_stage_batch_curve(_hardware(), "generator")


def test_load_fails_closed_on_block_hardware_mismatch(tmp_path, monkeypatch) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _collocated_stage_profiles())
    document = json.loads(path.read_text())
    document[PROFILE_BLOCK_KEY]["system_hardware"] = "another_system"
    path.write_text(json.dumps(document))

    with pytest.raises(StageBatchCurveValidationError, match="hardware identity mismatch"):
        load_stage_batch_curve(_hardware(), "generator")


def test_load_rejects_legacy_hardware_key_block_schema(tmp_path, monkeypatch) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _collocated_stage_profiles())
    document = json.loads(path.read_text())
    block = document[PROFILE_BLOCK_KEY]
    block["hardware_key"] = block.pop("system_hardware")
    path.write_text(json.dumps(document))

    with pytest.raises(StageBatchCurveValidationError, match="exact schema"):
        load_stage_batch_curve(_hardware(), "generator")


def test_load_rejects_unknown_block_field(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _collocated_stage_profiles())
    document = json.loads(path.read_text())
    document[PROFILE_BLOCK_KEY]["retired_version_marker"] = 2
    path.write_text(json.dumps(document))

    with pytest.raises(StageBatchCurveValidationError, match="exact schema"):
        load_stage_batch_curve(_hardware(), "generator")


def test_load_fails_closed_when_map_and_embedded_stage_identity_disagree(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _collocated_stage_profiles())
    document = json.loads(path.read_text())
    stored = document[PROFILE_BLOCK_KEY]["profiles"]
    stored["query_expansion"] = deepcopy(stored["generator"])
    path.write_text(json.dumps(document))

    with pytest.raises(StageBatchCurveValidationError, match="map key disagrees"):
        load_stage_batch_curve(_hardware(), "query_expansion")


def test_loaded_curve_still_fails_closed_above_profile_domain(
    tmp_path,
    monkeypatch,
) -> None:
    path = _profile_path(tmp_path, monkeypatch)
    publish_stage_batch_curves(path, _collocated_stage_profiles())
    loaded = load_stage_batch_curve(_hardware(), "generator")

    with pytest.raises(StageBatchCurveDomainError, match="exceeds calibrated maximum"):
        loaded.corrected_service_s(257, 1.0)


def test_stage_publication_preflight_removes_retired_component_blocks(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setenv(CALIBRATION_PROFILE_DIR_ENV, str(tmp_path))
    component_path = tmp_path / "llm_sim" / "H100_GPU.json"
    component_path.parent.mkdir(parents=True)
    component_path.write_text(json.dumps({
        "profiles": {
            "agnostic__Qwen_reference__bf16": {"preserved": True},
            "_tp_pp_efficiency": {"retired": True},
            "_vllm_scheduler_step_residual": {"retired": True},
        },
    }))

    generator_parallel._remove_retired_component_blocks(
        SimpleNamespace(gpu="H100_GPU")
    )

    assert json.loads(component_path.read_text()) == {
        "profiles": {
            "agnostic__Qwen_reference__bf16": {"preserved": True},
        },
    }


def test_stage_measurement_raises_and_verifies_nofile_limit(monkeypatch) -> None:
    state = {"soft": 1_024, "hard": 1_048_576}

    monkeypatch.setattr(
        resource,
        "getrlimit",
        lambda _kind: (state["soft"], state["hard"]),
    )

    def fake_setrlimit(_kind, value):
        state["soft"], state["hard"] = value

    monkeypatch.setattr(resource, "setrlimit", fake_setrlimit)

    soft, hard = generator_parallel._ensure_stage_nofile_limit()

    assert soft == 65_536
    assert hard == 1_048_576
    assert soft >= (
        generator_parallel.PRODUCTION_COHORT_COUNT
        * generator_parallel.MAX_STAGE_BATCH
        + generator_parallel._STAGE_NOFILE_HEADROOM
    )


def test_stage_measurement_rejects_insufficient_nofile_hard_limit(
    monkeypatch,
) -> None:
    hard = generator_parallel._STAGE_NOFILE_MINIMUM - 1
    monkeypatch.setattr(
        resource,
        "getrlimit",
        lambda _kind: (hard, hard),
    )
    monkeypatch.setattr(resource, "setrlimit", lambda _kind, _value: None)

    with pytest.raises(RuntimeError, match="open files"):
        generator_parallel._ensure_stage_nofile_limit()
