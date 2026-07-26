"""Special-mode fallbacks for components without a full spec model."""

from __future__ import annotations

import json

import pytest

from rag_stack.cost_model.llm_sim import compressor_llmlingua2_sim as llm2
from rag_stack.cost_model.retrieval_worker_profile import (
    RRFFusionSpecModel,
    RetrievalWorkerProfileError,
    resolve_rrf_fusion_profile,
)
from rag_stack.types import SystemConfig


@pytest.fixture(autouse=True)
def _isolated_profile_root(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_STACK_CALIBRATION_PROFILE_DIR", str(tmp_path))


def _mode(mode: str) -> dict:
    return {"system": {"rag_cm": {"calibration_mode": mode}}}


def _compressor_pipeline(mode: str) -> dict:
    pipeline = _mode(mode)
    pipeline.update({
        "_token_stats": {
            "avg_chunk_tokens": 86,
            "prompt_template_tokens": 17,
            "avg_output_tokens": 22,
            "avg_query_tokens": 7,
            "bytes_per_token": 4.2,
            "avg_compressor_elems_per_chunk": 1.0,
            "avg_compressor_elem_tokens": 89,
        },
        "node_lines": [{"nodes": [
            {
                "stage": "semantic_retrieval",
                "top_k": 4,
                "modules": [{"component": "vectordb"}],
            },
            {
                "stage": "passage_compressor",
                "modules": [{"component": "llmlingua2", "batch_size": 2}],
            },
        ]}],
    })
    return pipeline


def _hardware() -> SystemConfig:
    return SystemConfig(
        cpu_hardware_key="new_cpu",
        num_cores=16,
        peak_int_ops=1.6e12,
        mem_bandwidth=100e9,
        mem_bandwidth_1t=20e9,
        gpu={
            "hardware_key": "new_gpu",
            "Flops": 100.0,
            "Memory_BW": 1000.0,
            "Memory_size": 80.0,
            "ICN": 100.0,
        },
    )


def test_llmlingua_missing_profile_uses_labelled_bert_floor_only_in_spec_mode(
    monkeypatch,
):
    calls = []

    def _fake_spec(hardware, *, forward_tile_units, unit_tokens):
        calls.append((hardware, forward_tile_units, unit_tokens))
        return 0.125

    monkeypatch.setattr(llm2, "_uncalibrated_spec_forward_service_s", _fake_spec)
    model = llm2.PassageCompressorLLMLingua2CostModel()
    plan = model.build_request_operator_plan(
        _compressor_pipeline("uncalibrated_spec"),
        _hardware(),
        request_batch=2,
        forward_cap=4,
    )

    assert plan.forward_tile_units == (4.0, 4.0)
    assert plan.service_s == pytest.approx(0.125)
    assert plan.pricing_source == "uncalibrated_spec_bert_forward_lower_bound"
    assert calls and calls[0][2] == 89

    with pytest.raises(llm2.LLMLingua2ProfileError, match="missing"):
        model.build_request_operator_plan(
            _compressor_pipeline("strict"),
            _hardware(),
            request_batch=2,
        )


def test_llmlingua_present_but_malformed_profile_never_falls_back(
    monkeypatch, tmp_path,
):
    profile = tmp_path / "compressor_llmlingua2" / "new_gpu.json"
    profile.parent.mkdir(parents=True)
    profile.write_text("{not-json")
    monkeypatch.setattr(
        llm2,
        "_uncalibrated_spec_forward_service_s",
        lambda *args, **kwargs: pytest.fail("invalid profile was hidden"),
    )

    with pytest.raises(llm2.LLMLingua2ProfileError, match="invalid JSON"):
        llm2.PassageCompressorLLMLingua2CostModel().build_request_operator_plan(
            _compressor_pipeline("uncalibrated_spec"),
            _hardware(),
            request_batch=2,
        )


def test_rrf_missing_profile_uses_cpu_spec_model_and_strict_still_fails():
    hardware = _hardware()
    model = resolve_rrf_fusion_profile(
        hardware, _mode("uncalibrated_spec"),
    )

    assert isinstance(model, RRFFusionSpecModel)
    assert 0.0 < model.service_s(8) < model.service_s(256)
    with pytest.raises(RetrievalWorkerProfileError, match="missing"):
        resolve_rrf_fusion_profile(hardware, _mode("strict"))


def test_rrf_present_but_malformed_profile_never_falls_back(tmp_path):
    profile = tmp_path / "retrieval_worker" / "new_cpu.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({"unexpected": 999}))

    with pytest.raises(RetrievalWorkerProfileError):
        resolve_rrf_fusion_profile(
            _hardware(), _mode("uncalibrated_spec"),
        )
