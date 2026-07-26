"""CPU-only contracts for the explicit uncalibrated hardware-spec mode."""

from __future__ import annotations

import json

import pytest

from rag_stack.cost_model.calibration_mode import (
    STRICT_CALIBRATION_MODE,
    UNCALIBRATED_SPEC_MODE,
    is_uncalibrated_spec_mode,
    resolve_calibration_mode,
)
from rag_stack.types import SystemConfig


def _pipeline(*, spec: bool) -> dict:
    system = {"retrieval": {"faiss_num_threads": 3}}
    if spec:
        system["rag_cm"] = {"calibration_mode": "uncalibrated_spec"}
    return {"system": system}


def _hardware() -> SystemConfig:
    return SystemConfig(
        cpu_hardware_key="test_inline_cpu",
        num_cores=8,
        peak_flops=1.0e12,
        peak_int_ops=5.0e11,
        mem_bandwidth=1.0e11,
        mem_bandwidth_1t=2.0e10,
        gpu={
            "hardware_key": "test_inline_gpu",
            "Flops": 100.0,
            "Memory_size": 80.0,
            "Memory_BW": 2_000.0,
            "ICN": 100.0,
        },
    )


def test_calibration_mode_is_strict_by_default_and_exact_when_enabled() -> None:
    assert resolve_calibration_mode({}) == STRICT_CALIBRATION_MODE
    assert resolve_calibration_mode({"system": {}}) == STRICT_CALIBRATION_MODE
    config = _pipeline(spec=True)
    assert resolve_calibration_mode(config) == UNCALIBRATED_SPEC_MODE
    assert is_uncalibrated_spec_mode(config)

    with pytest.raises(ValueError, match="calibration_mode"):
        resolve_calibration_mode({
            "system": {"rag_cm": {"calibration_mode": "best_effort"}},
        })


def test_hardware_calibration_mode_policy_has_priority_over_pipeline() -> None:
    pipeline_spec = _pipeline(spec=True)

    # Any target-owned rag_cm block is authoritative.  An empty block therefore
    # intentionally selects its default strict policy instead of inheriting.
    hardware_strict = _hardware()
    hardware_strict.rag_cm = {}
    assert (
        resolve_calibration_mode(pipeline_spec, hardware_strict)
        == STRICT_CALIBRATION_MODE
    )

    hardware_spec = _hardware()
    hardware_spec.rag_cm = {"calibration_mode": "uncalibrated_spec"}
    assert is_uncalibrated_spec_mode(_pipeline(spec=False), hardware_spec)

    hardware_invalid = _hardware()
    hardware_invalid.rag_cm = {"calibration_mode": "best_effort"}
    with pytest.raises(ValueError, match="hardware.rag_cm.calibration_mode"):
        resolve_calibration_mode(pipeline_spec, hardware_invalid)


class _FakeGenZComponent:
    """Mixin methods installed onto GenZLLMSimBase without running GenZ."""

    def _get_model(self, pipeline_config: dict):
        return "unused"

    def _get_params(self, pipeline_config: dict) -> dict:
        return {"input_tokens": 16, "output_tokens": 1, "bits": "bf16"}

    def _find_module(self, pipeline_config: dict) -> dict:
        # A real, forward-only taxonomy component avoids the generator's
        # independent two-role evidence contract in this unit test.
        return {
            "component": "similarity_threshold_cutoff",
            "bits": "bf16",
        }

    def _get_model_name_for_df(self, pipeline_config: dict) -> str:
        return "fake_component_model"


def _fake_genz_component():
    from rag_stack.cost_model.llm_sim.genz_adapter import GenZLLMSimBase

    class FakeComponent(_FakeGenZComponent, GenZLLMSimBase):
        pass

    return FakeComponent()


def test_genz_missing_profile_uses_physical_roofline_only_in_spec_mode(
    monkeypatch,
) -> None:
    from rag_stack.cost_model.llm_sim import calibration

    monkeypatch.setattr(calibration, "load_record", lambda *args, **kwargs: None)
    physical = {"Flops": 100.0, "Memory_size": 80.0, "real_values": True}
    monkeypatch.setattr(calibration, "_hardware_to_dict", lambda hardware: physical)
    component = _fake_genz_component()

    with pytest.raises(ValueError, match="requires an exact component record"):
        component._resolve_calibration(_hardware(), _pipeline(spec=False))

    assert component._resolve_calibration(
        _hardware(), _pipeline(spec=True),
    ) == (physical, 1.0, 1.0, "bf16", 0.0)


def test_genz_present_but_malformed_profile_does_not_fallback(
    monkeypatch,
) -> None:
    from rag_stack.cost_model.llm_sim import calibration
    from rag_stack.cost_model.llm_sim.genz_adapter import GenZCalibrationError

    def malformed(*args, **kwargs):
        raise ValueError("malformed exact record")

    monkeypatch.setattr(calibration, "load_record", malformed)
    component = _fake_genz_component()
    with pytest.raises(GenZCalibrationError, match="Invalid GenZ calibration"):
        component._resolve_calibration(_hardware(), _pipeline(spec=True))


def test_genz_present_falsey_record_is_not_treated_as_missing(
    monkeypatch,
    tmp_path,
) -> None:
    from rag_stack.cost_model.llm_sim import calibration
    from rag_stack.cost_model.llm_sim.genz_adapter import GenZCalibrationError

    profile = tmp_path / "test_inline_gpu.json"
    profile.write_text(json.dumps({
        "profiles": {
            "similarity_threshold_cutoff__fake_component_model__bf16": {},
        },
    }), encoding="utf-8")
    monkeypatch.setattr(calibration, "_default_calibration_dir", lambda: tmp_path)
    calibration.invalidate_cache()

    with pytest.raises(GenZCalibrationError, match="present entry"):
        _fake_genz_component()._resolve_calibration(
            _hardware(), _pipeline(spec=True),
        )


def test_ivf_missing_profile_uses_inline_cpu_spec_only_in_spec_mode(
    monkeypatch,
    tmp_path,
) -> None:
    from rag_stack.cost_model.faiss_ivf_sim import adapter

    missing = tmp_path / "missing-ivf.json"
    monkeypatch.setattr(adapter, "system_profile_path", lambda *args: missing)
    model = adapter.FaissIVFCostModel()

    with pytest.raises(ValueError, match="missing IVF CPU profile"):
        model._resolve_hardware(_pipeline(spec=False), _hardware())

    resolved = model._resolve_hardware(_pipeline(spec=True), _hardware())
    assert resolved.name == "from_system_config"
    assert resolved.active_threads == 3
    assert resolved.peak_flops == _hardware().peak_flops


def test_ivf_present_but_malformed_profile_does_not_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    from rag_stack.cost_model.faiss_ivf_sim import adapter

    malformed = tmp_path / "malformed-ivf.json"
    malformed.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(adapter, "system_profile_path", lambda *args: malformed)

    with pytest.raises(ValueError):
        adapter.FaissIVFCostModel()._resolve_hardware(
            _pipeline(spec=True), _hardware(),
        )


def test_hnsw_missing_profile_uses_inline_cpu_spec_only_in_spec_mode(
    monkeypatch,
    tmp_path,
) -> None:
    from rag_stack.cost_model.faiss_hnsw_sim import adapter
    from rag_stack.cost_model.faiss_hnsw_sim.coefficients import (
        HNSWProfileError,
    )
    from rag_stack.cost_model.faiss_hnsw_sim.cost_model import (
        HNSWCoefficients,
    )

    missing = tmp_path / "missing-hnsw.json"
    monkeypatch.setattr(
        adapter, "profile_path_for_preset", lambda cpu_key: str(missing),
    )

    def strict_missing(cpu_key):
        raise HNSWProfileError(f"missing profile for {cpu_key}")

    monkeypatch.setattr(adapter, "load_profile_for_preset", strict_missing)
    model = adapter.FaissHNSWCostModel()
    with pytest.raises(HNSWProfileError, match="missing profile"):
        model._resolve_hardware(_pipeline(spec=False), _hardware())

    resolved, coefficients = model._resolve_hardware(
        _pipeline(spec=True), _hardware(),
    )
    assert resolved.name == "from_system_config"
    assert resolved.active_threads == 3
    assert coefficients == HNSWCoefficients()


def test_hnsw_present_but_malformed_profile_does_not_fallback(
    monkeypatch,
    tmp_path,
) -> None:
    from rag_stack.cost_model.faiss_hnsw_sim import adapter
    from rag_stack.cost_model.faiss_hnsw_sim.coefficients import (
        HNSWProfileError,
    )

    malformed = tmp_path / "malformed-hnsw.json"
    malformed.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        adapter, "profile_path_for_preset", lambda cpu_key: str(malformed),
    )

    def invalid(cpu_key):
        raise HNSWProfileError(f"malformed profile for {cpu_key}")

    monkeypatch.setattr(adapter, "load_profile_for_preset", invalid)
    with pytest.raises(HNSWProfileError, match="malformed profile"):
        adapter.FaissHNSWCostModel()._resolve_hardware(
            _pipeline(spec=True), _hardware(),
        )
