from __future__ import annotations

from tests import REPO_ROOT

import json
from pathlib import Path

import pytest

from rag_stack.cost_model.llm_sim.calibration import (
    CAPACITY_HINGE_LAW_COMPONENT_CONTRACT,
    CalibrationRecord,
    invalidate_cache,
    load_record,
    save_record,
)
from rag_stack.cost_model.llm_sim.genz_adapter import GenZLLMSimBase


def _envelope(profiles: dict, **extra) -> dict:
    return {
        "profiles": profiles,
        **extra,
    }


def _linear_only_payload() -> dict:
    return {
        "component": "sentence_transformer_reranker",
        "model_key": "ms_marco_minilm_l2",
        "hardware_key": "H100_GPU",
        "bits": "f32",
        "fit_mape": 0.05,
        "linear_law": {
            "fixed_ms": 5.1,
            "per_item_ms": 0.024,
            "per_token_ms": 0.00096,
            "fit_mape": 0.05,
        },
        "max_seq_len": 512,
    }


def test_load_record_accepts_compact_linear_law_only_profile(tmp_path):
    key = "sentence_transformer_reranker__ms_marco_minilm_l2__f32"
    (tmp_path / "H100_GPU.json").write_text(
        json.dumps(_envelope({key: _linear_only_payload()}))
    )
    invalidate_cache()

    record = load_record(
        "sentence_transformer_reranker",
        "ms_marco_minilm_l2",
        "H100_GPU",
        bits="f32",
        directory=tmp_path,
    )

    assert record is not None
    assert record.compute_efficiency is None
    assert record.memory_efficiency is None
    assert record.per_call_overhead_s == 0.0
    assert record.linear_law["fixed_ms"] == 5.1


def test_load_record_rejects_ambiguous_precision_when_bits_are_omitted(tmp_path):
    base = _linear_only_payload()
    f32 = dict(base)
    bf16 = {**base, "bits": "bf16"}
    (tmp_path / "H100_GPU.json").write_text(json.dumps(_envelope({
        "sentence_transformer_reranker__ms_marco_minilm_l2__f32": f32,
        "sentence_transformer_reranker__ms_marco_minilm_l2__bf16": bf16,
    })))
    invalidate_cache()

    with pytest.raises(ValueError, match="ambiguous calibration precision"):
        load_record(
            "sentence_transformer_reranker",
            "ms_marco_minilm_l2",
            "H100_GPU",
            bits=None,
            directory=tmp_path,
        )


def test_linear_law_only_round_trip_does_not_restore_dead_coefficients(tmp_path):
    record = CalibrationRecord(**_linear_only_payload())
    path = save_record(record, tmp_path)
    payload = json.loads(path.read_text())
    assert set(payload) == {"profiles"}
    stored = payload["profiles"][
        "sentence_transformer_reranker__ms_marco_minilm_l2__f32"
    ]

    assert "compute_efficiency" not in stored
    assert "memory_efficiency" not in stored
    assert "per_call_overhead_s" not in stored


@pytest.mark.parametrize(
    "overrides",
    [
        {"linear_law": None},
        {"compute_efficiency": 0.5, "linear_law": None},
        {"linear_law": {"fixed_ms": 1.0, "per_item_ms": 2.0}},
    ],
)
def test_calibration_record_rejects_incomplete_pricing_forms(overrides):
    payload = _linear_only_payload()
    payload.update(overrides)
    with pytest.raises(ValueError):
        CalibrationRecord(**payload)


def test_calibration_record_rejects_e2e_ratio_grid():
    payload = _linear_only_payload()
    payload["grid"] = [{
        "n_e2e_points": 37,
        "ratio_median": 10.7,
        "note": "fit against RAG throughput",
    }]
    with pytest.raises(ValueError, match=r"grid\[0\] must contain exactly"):
        CalibrationRecord(**payload)


def test_calibration_record_rejects_overlapping_pricing_forms():
    payload = _linear_only_payload()
    payload.update({
        "compute_efficiency": 0.5,
        "memory_efficiency": 0.6,
    })
    with pytest.raises(ValueError, match="pricing forms are exclusive"):
        CalibrationRecord(**payload)


@pytest.mark.parametrize(
    "forbidden_field",
    [
        "measurement_evidence",
        "single_role_measurement_evidence",
        "cuda_name",
        "timestamp",
    ],
)
def test_runtime_record_rejects_provenance_fields(forbidden_field):
    payload = _linear_only_payload()
    payload[forbidden_field] = {} if "evidence" in forbidden_field else "source"
    with pytest.raises(TypeError, match=forbidden_field):
        CalibrationRecord(**payload)


def test_capacity_hinge_record_loads_without_provenance_and_preserves_equation(
    tmp_path,
):
    key = "monot5__monot5-3b__f32"
    fit_mape = 0.04
    payload = {
        "component": "monot5",
        "model_key": "monot5-3b",
        "hardware_key": "A100_80GB_GPU",
        "bits": "f32",
        "calibration_contract": CAPACITY_HINGE_LAW_COMPONENT_CONTRACT,
        "fit_mape": fit_mape,
        "correlation": 0.99,
        "grid": [{
            "batch_size": 192,
            "seq_len": 512,
            "measured_s": 37.8,
            "predicted_s": 37.8,
            "rel_err": 0.0,
        }],
        "identifiability_rank": None,
        "identifiability_condition": None,
        "validation_mape": 0.01,
        "validation_grid": [{
            "batch_size": 256,
            "seq_len": 512,
            "measured_s": 49.6,
            "predicted_s": 49.5,
            "rel_err": -0.002,
        }],
        "linear_law": {
            "fixed_ms": 60.0,
            "per_item_ms": 0.0,
            "per_token_ms": 0.168,
            "capacity_tokens": 69_632.0,
            "overload_step_ms": 15_021.0,
            "overload_per_token_ms": 0.178,
            "fit_mape": fit_mape,
        },
        "max_seq_len": 512,
    }
    (tmp_path / "A100_80GB_GPU.json").write_text(
        json.dumps(_envelope({key: payload}))
    )
    invalidate_cache()
    record = load_record(
        "monot5",
        "monot5-3b",
        "A100_80GB_GPU",
        bits="f32",
        directory=tmp_path,
    )
    assert record is not None
    batch = 192
    input_tokens = 512
    aggregate_tokens = batch * input_tokens
    expected_ms = (
        60.0
        + batch * 0.168 * input_tokens
        + 15_021.0
        + 0.178 * (aggregate_tokens - 69_632.0)
    )
    assert GenZLLMSimBase._linear_law_latency_s(
        record,
        {"input_tokens": input_tokens},
        batch,
    ) == pytest.approx(expected_ms / 1000.0)


def test_every_shipped_llm_profile_uses_the_current_exact_envelope():
    profile_dir = REPO_ROOT / "calibration_profiles" / "llm_sim"
    for profile_path in sorted(profile_dir.glob("*.json")):
        document = json.loads(profile_path.read_text())
        assert set(document) == {"profiles"}
        assert isinstance(document["profiles"], dict)
        for payload in document["profiles"].values():
            CalibrationRecord(**payload)


@pytest.mark.parametrize(
    "document",
    [
        {},
        _envelope({}, extra_field=True),
    ],
)
def test_load_record_rejects_missing_or_extra_envelope_fields(
    tmp_path, document,
):
    (tmp_path / "H100_GPU.json").write_text(json.dumps(document))
    invalidate_cache()

    with pytest.raises(ValueError, match="envelope fields must be exactly"):
        load_record(
            "sentence_transformer_reranker",
            "ms_marco_minilm_l2",
            "H100_GPU",
            bits="f32",
            directory=tmp_path,
        )


def test_save_record_rejects_invalid_existing_envelope_without_overwrite(tmp_path):
    path = tmp_path / "H100_GPU.json"
    original = json.dumps({
        "sentence_transformer_reranker__ms_marco_minilm_l2__f32": (
            _linear_only_payload()
        ),
    })
    path.write_text(original)

    with pytest.raises(ValueError, match="envelope fields must be exactly"):
        save_record(CalibrationRecord(**_linear_only_payload()), tmp_path)

    assert path.read_text() == original


def test_rrf_profile_has_cpu_identity_and_fit_diagnostics():
    root = REPO_ROOT
    profile = json.loads(
        (root / "calibration_profiles/retrieval_worker/epyc_9124.json").read_text()
    )
    rrf = profile["components"]["rrf_fusion"]

    assert profile["profile_kind"] == "retrieval_worker"
    assert profile["hardware_key"] == "epyc_9124"
    assert set(profile) == {"profile_kind", "hardware_key", "components"}
    assert rrf["demand_axis"] == "input_items"
    assert "fanout * top_k" in rrf["equation"]
    assert rrf["per_call_s"] == 0.00091323
    assert rrf["per_item_s"] == 0.0000033252
    assert len(rrf["grid"]) == 40
    assert rrf["fit_mape"] >= 0.0
