from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_stack.calibration_paths import calibration_profile_root
from rag_stack.cost_model.llm_sim.compressor_llmlingua2_sim import (
    LLMLingua2ProfileError,
    load_llmlingua2_coeffs,
)
from rag_stack.types import SystemConfig


@pytest.fixture(autouse=True)
def _isolated_profile_root(request, monkeypatch):
    if "tmp_path" in request.fixturenames:
        monkeypatch.setenv(
            "RAG_STACK_CALIBRATION_PROFILE_DIR",
            str(request.getfixturevalue("tmp_path")),
        )


def _hardware(
    profile_root: Path,
    key: str,
    cpu_key: str = "fixture_cpu",
) -> SystemConfig:
    del profile_root
    return SystemConfig(
        num_cores=1,
        gpu={"hardware_key": key},
        cpu_preset=cpu_key,
    )


def _profile(key: str, cpu_key: str = "fixture_cpu") -> dict:
    grid = []
    for n_chunks in (1, 2, 4, 8, 16, 32):
        for chunk_tokens in (64, 128, 256, 448):
            predicted_ms = 4.0 + n_chunks * (3.0 + 0.01 * chunk_tokens)
            grid.append({
                "n_chunks": n_chunks,
                "chunk_tokens": chunk_tokens,
                "measured_ms": predicted_ms,
                "predicted_ms": predicted_ms,
                "rel_err": 0.0,
            })
    return {
        "profile_kind": "compressor_llmlingua2",
        "service_unit": "milliseconds_per_fused_compressor_call",
        "demand_axes": ["n_chunks", "chunk_tokens"],
        "model": "llmlingua2",
        "hardware_key": key,
        "cpu_hardware_key": cpu_key,
        "fixed_ms": 4.0,
        "per_chunk_ms": 3.0,
        "per_token_ms": 0.01,
        "fit_mape": 0.02,
        "correlation": 0.99,
        "grid": grid,
        "note": "standalone fixture",
    }


def _write_profile(profile_root: Path, path_key: str, document: dict) -> Path:
    path = profile_root / "compressor_llmlingua2" / f"{path_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize(
    ("key", "cpu_key", "expected"),
    [
        (
            "H100_GPU",
            "epyc_9124",
            {
                "fixed_ms": 4.993947854136787,
                "per_chunk_ms": 3.036083454123027,
                "per_token_ms": 0.005727182185899613,
                "fit_mape": 0.022692080992141855,
                "correlation": 0.9998557524585674,
            },
        ),
        (
            "rtx_3090",
            "epyc_7313",
            {
                "fixed_ms": 3.6674863644028006,
                "per_chunk_ms": 6.701036126539473,
                "per_token_ms": 0.006281007694268759,
                "fit_mape": 0.011742683304025286,
                "correlation": 0.9999740158805315,
            },
        ),
    ],
)
def test_shipped_exact_profiles_keep_their_existing_predictions(
    key, cpu_key, expected,
):
    document = json.loads(
        (
            calibration_profile_root()
            / "compressor_llmlingua2"
            / f"{key}.json"
        ).read_text()
    )
    assert document["fit_mape"] == pytest.approx(expected["fit_mape"])
    assert document["correlation"] == pytest.approx(expected["correlation"])
    assert {
        (row["n_chunks"], row["chunk_tokens"])
        for row in document["grid"]
    } == {
        (n_chunks, chunk_tokens)
        for n_chunks in (1, 2, 4, 8, 16, 32)
        for chunk_tokens in (64, 128, 256, 448)
    }
    actual = load_llmlingua2_coeffs(SystemConfig(
        num_cores=1, gpu=key, cpu_preset=cpu_key,
    ))
    assert actual == pytest.approx({
        field: expected[field]
        for field in ("fixed_ms", "per_chunk_ms", "per_token_ms")
    })


def test_exact_profile_load_needs_no_raw_calibration_tree(tmp_path, monkeypatch):
    missing_raw_root = tmp_path / "raw-calibration-tree-does-not-exist"
    monkeypatch.setenv(
        "RAG_STACK_CALIBRATION_DATA_DIR",
        str(missing_raw_root),
    )
    _write_profile(tmp_path, "H100_GPU", _profile("H100_GPU"))

    assert load_llmlingua2_coeffs(
        _hardware(tmp_path, "H100_GPU"),
    ) == pytest.approx({
        "fixed_ms": 4.0,
        "per_chunk_ms": 3.0,
        "per_token_ms": 0.01,
    })
    assert not missing_raw_root.exists()


def test_missing_h100_never_falls_back_to_present_rtx_3090_profile(tmp_path):
    _write_profile(tmp_path, "rtx_3090", _profile("rtx_3090"))

    with pytest.raises(
        LLMLingua2ProfileError,
        match=r"missing .* exact GPU 'H100_GPU'.*H100_GPU\.json",
    ):
        load_llmlingua2_coeffs(_hardware(tmp_path, "H100_GPU"))


def test_profile_hardware_identity_must_match_exact_request(tmp_path):
    _write_profile(tmp_path, "H100_GPU", _profile("rtx_3090"))

    with pytest.raises(LLMLingua2ProfileError, match="hardware_key.*H100_GPU"):
        load_llmlingua2_coeffs(_hardware(tmp_path, "H100_GPU"))


def test_profile_cpu_identity_must_match_exact_request(tmp_path):
    _write_profile(tmp_path, "H100_GPU", _profile("H100_GPU", "other_cpu"))

    with pytest.raises(LLMLingua2ProfileError, match="cpu_hardware_key"):
        load_llmlingua2_coeffs(_hardware(tmp_path, "H100_GPU"))


def test_malformed_json_fails_closed_with_profile_path(tmp_path):
    path = _write_profile(tmp_path, "H100_GPU", _profile("H100_GPU"))
    path.write_text("{not-json")

    with pytest.raises(LLMLingua2ProfileError, match="invalid JSON.*H100_GPU"):
        load_llmlingua2_coeffs(_hardware(tmp_path, "H100_GPU"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fixed_ms", -1.0),
        ("per_chunk_ms", 0.0),
        ("per_token_ms", float("nan")),
        ("per_token_ms", float("inf")),
        ("per_chunk_ms", True),
    ],
)
def test_nonfinite_or_nonphysical_coefficients_fail_closed(
    tmp_path,
    field,
    value,
):
    profile = _profile("H100_GPU")
    profile[field] = value
    _write_profile(tmp_path, "H100_GPU", profile)

    with pytest.raises(LLMLingua2ProfileError, match=field):
        load_llmlingua2_coeffs(_hardware(tmp_path, "H100_GPU"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile_kind", "assembly"),
        ("service_unit", "seconds_per_stage"),
        ("demand_axes", ["batch"]),
        ("invented_e2e_coefficient", 2.0),
    ],
)
def test_schema_mismatch_fails_closed(
    tmp_path,
    field,
    value,
):
    profile = _profile("H100_GPU")
    profile[field] = value
    _write_profile(tmp_path, "H100_GPU", profile)

    with pytest.raises(LLMLingua2ProfileError, match=field):
        load_llmlingua2_coeffs(_hardware(tmp_path, "H100_GPU"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence_scope", "standalone_component_only_no_rag_e2e_gt"),
        ("cuda_name", "NVIDIA H100 80GB HBM3"),
        ("timestamp", "2026-07-11T00:00:00+00:00"),
        ("source_path", "calibration_data/llmlingua2/raw.json"),
        ("source_sha256", "a" * 64),
        ("provenance", {"source": "raw"}),
    ],
)
def test_legacy_provenance_fields_are_rejected(
    tmp_path,
    field,
    value,
):
    profile = _profile("H100_GPU")
    profile[field] = value
    _write_profile(tmp_path, "H100_GPU", profile)

    with pytest.raises(LLMLingua2ProfileError, match=field):
        load_llmlingua2_coeffs(_hardware(tmp_path, "H100_GPU"))
