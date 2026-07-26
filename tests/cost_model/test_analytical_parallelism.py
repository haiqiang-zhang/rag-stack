"""CPU-only contracts for analytical TP/PP pricing."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.cost_model.llm_sim import genz_adapter
from rag_stack.cost_model.llm_sim.genz_adapter import MainLLMCostModel
from rag_stack.cost_model.llm_sim.genz_adapter import GenZCalibrationError
from rag_stack.cost_model.llm_sim.calibration import (
    CalibrationRecord,
    LEGACY_INTERNAL_COMPONENT_CONTRACT,
    ZERO_INTERCEPT_ROOFLINE_CONTRACT,
)
from rag_stack.cost_model.llm_sim import tp_efficiency
from rag_stack.types import CMDeviceCapacityError


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("generator_prefill", True),
        ("generator_decode", True),
        ("query_expansion_prefill", True),
        ("query_expansion_decode", True),
        ("passage_reranker_prefill", True),
        ("passage_reranker_decode", True),
        ("passage_compressor_prefill", True),
        ("passage_compressor_decode", True),
        ("passage_reranker", False),
        ("semantic_retrieval_encode", False),
        ("passage_compressor", False),
        ("unknown_stage", False),
    ],
)
def test_two_role_llm_stage_predicate_uses_concrete_taxonomy_membership(
    stage: str,
    expected: bool,
) -> None:
    assert genz_adapter._is_prefill_decode_engine_stage(stage) is expected


@pytest.mark.parametrize(
    ("batch", "pp"),
    [
        (1, 1),
        (256, 1),
        (1, 2),
        (2, 2),
        (256, 2),
        (2, 4),
        (4, 4),
        (256, 4),
    ],
)
def test_pipeline_fill_drain_is_identity_for_one_scheduler_step(
    batch: int, pp: int,
) -> None:
    assert tp_efficiency.pipeline_fill_drain_factor(batch, pp) == pytest.approx(
        1.0
    )


@pytest.mark.parametrize("batch,pp", [(0, 1), (1, 0), (1.5, 2), (True, 2)])
def test_pipeline_fill_drain_rejects_non_positive_or_fractional_shape(
    batch, pp,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        tp_efficiency.pipeline_fill_drain_factor(batch, pp)


def test_legacy_parallelism_api_is_profile_independent(monkeypatch) -> None:
    def profile_access_is_a_bug(_hardware):
        raise AssertionError("production must not read _tp_pp_efficiency")

    monkeypatch.setattr(
        tp_efficiency, "_llm_sim_coeffs", profile_access_is_a_bug,
    )
    hardware = SimpleNamespace(
        interconnect={"gpu_gpu": {"latency_us": 1.25}},
    )

    # TP has no learned multiplier; GenZ owns its analytical communication.
    assert tp_efficiency.tp_pp_throughput_factor(
        "any-model", tp=8, pp=1, hardware=hardware,
    ) == pytest.approx(1.0)
    # Compatibility callers validate the runtime shape but remain identity.
    assert tp_efficiency.tp_pp_throughput_factor(
        "any-model", tp=8, pp=2, hardware=hardware, resident_batch=256,
    ) == pytest.approx(1.0)
    assert tp_efficiency.pp_latency_penalty(
        "any-model", 8, 2, 999.0, hardware, resident_batch=256,
    ) == pytest.approx(1.0)
    assert tp_efficiency.effective_icn_ll_us(hardware) == pytest.approx(1.25)
    assert tp_efficiency.tp_collective_overhead_us(hardware) == 0.0


def test_two_role_llm_rejects_stale_model_keyed_serving_intercept(
    monkeypatch,
) -> None:
    model = MainLLMCostModel()
    stale = SimpleNamespace(
        calibration_contract=LEGACY_INTERNAL_COMPONENT_CONTRACT,
        per_call_overhead_s=0.01,
        linear_law=None,
        max_seq_len=None,
    )
    monkeypatch.setattr(model, "_calibration_record", lambda *_a: stale)
    monkeypatch.setattr(model, "_stage_kinds", lambda _cfg: ["prefill", "decode"])
    monkeypatch.setattr(
        model,
        "_performance_stage_name",
        lambda _cfg, kind: f"generator_{kind}",
    )

    with pytest.raises(GenZCalibrationError, match="zero-intercept"):
        model._pricing_record_and_params(object(), {}, {"input_tokens": 16})


def _portable_two_role_record(*, held_out_seq_len=(64, 128)) -> CalibrationRecord:
    return CalibrationRecord(
        component="agnostic",
        model_key="unit-model",
        hardware_key="unit-hardware",
        bits="bf16",
        compute_efficiency=0.55,
        memory_efficiency=0.78,
        fit_mape=0.02,
        correlation=0.99,
        calibration_contract=ZERO_INTERCEPT_ROOFLINE_CONTRACT,
        identifiability_rank=2,
        identifiability_condition=2.0,
        validation_mape=0.01,
        grid=[{
            "batch_size": 128,
            "seq_len": [64, 128],
            "measured_s": 1.0,
            "predicted_s": 1.0,
            "rel_err": 0.0,
        }],
        validation_grid=[{
            "batch_size": 256,
            "seq_len": list(held_out_seq_len),
            "measured_s": 2.0,
            "predicted_s": 2.0,
            "rel_err": 0.0,
        }],
    )


def test_two_role_llm_accepts_self_contained_profile_grid_coverage() -> None:
    genz_adapter.validate_two_role_llm_component_record(
        _portable_two_role_record()
    )


def test_two_role_llm_rejects_missing_same_shape_b128_b256_pair() -> None:
    with pytest.raises(GenZCalibrationError, match="runtime pricing fields"):
        genz_adapter.validate_two_role_llm_component_record(
            _portable_two_role_record(held_out_seq_len=(128, 128))
        )


def test_dynamic_des_rejects_stale_component_intercept_before_use(
    monkeypatch,
) -> None:
    """The dynamic call-latency path must not bypass the shared record gate."""
    from rag_stack.cost_model import engines, sweep_inputs
    from rag_stack.cost_model.llm_sim import calibration

    model = MainLLMCostModel()
    stale = SimpleNamespace(
        component="agnostic",
        model_key="unit-model",
        hardware_key="unit-hardware",
        bits="bf16",
        compute_efficiency=0.5,
        memory_efficiency=0.5,
        calibration_contract=LEGACY_INTERNAL_COMPONENT_CONTRACT,
        per_call_overhead_s=0.01,
        linear_law=None,
    )
    monkeypatch.setattr(model, "_get_model", lambda _cfg: "unit-model")
    monkeypatch.setattr(
        model, "_get_model_name_for_df", lambda _cfg: "unit-model",
    )
    monkeypatch.setattr(
        model,
        "_find_module",
        lambda _cfg: {"component": "vllm", "bits": "bf16"},
    )
    monkeypatch.setattr(model, "_stage_kinds", lambda _cfg: ["prefill", "decode"])
    monkeypatch.setattr(
        model,
        "_performance_stage_name",
        lambda _cfg, kind: f"generator_{kind}",
    )
    monkeypatch.setattr(calibration, "load_record", lambda *_a, **_kw: stale)
    monkeypatch.setattr(
        sweep_inputs,
        "build_performance_stage_cost_models",
        lambda _cfg: {
            "generator_prefill": model,
            "generator_decode": model,
        },
    )
    hardware = SimpleNamespace(
        gpu={"hardware_key": "unit-hardware"},
        search_space={},
        interconnect={},
    )

    with pytest.raises(GenZCalibrationError, match="zero-intercept"):
        engines.build_call_latency({}, hardware)


def test_single_role_auxiliary_can_keep_legacy_internal_runtime_profile(
    monkeypatch,
) -> None:
    model = MainLLMCostModel()
    auxiliary = SimpleNamespace(
        calibration_contract=LEGACY_INTERNAL_COMPONENT_CONTRACT,
        per_call_overhead_s=0.01,
        linear_law=None,
        max_seq_len=None,
    )
    monkeypatch.setattr(model, "_calibration_record", lambda *_a: auxiliary)
    monkeypatch.setattr(model, "_stage_kinds", lambda _cfg: ["prefill"])
    monkeypatch.setattr(
        model,
        "_performance_stage_name",
        lambda _cfg, _kind: "auxiliary_prefill",
    )

    record, params = model._pricing_record_and_params(
        object(), {}, {"input_tokens": 16}
    )
    assert record is auxiliary
    assert params == {"input_tokens": 16}


class _ParallelismVerdict:
    def require(self) -> None:
        return None


def _stub_main_llm(monkeypatch) -> MainLLMCostModel:
    model = MainLLMCostModel()
    params = {"input_tokens": 32, "output_tokens": 1, "bits": "bf16"}
    monkeypatch.setattr(model, "_find_module", lambda _cfg: {"batch_size": 8})
    monkeypatch.setattr(model, "_get_model", lambda _cfg: "unit-model")
    monkeypatch.setattr(model, "_get_params", lambda _cfg: dict(params))
    monkeypatch.setattr(
        model, "_get_model_name_for_df", lambda _cfg: "unit-model",
    )
    monkeypatch.setattr(
        model,
        "_pricing_record_and_params",
        lambda _hw, _cfg, values: (object(), values),
    )
    monkeypatch.setattr(
        model,
        "_resolve_calibration",
        lambda _hw, _cfg: ({}, 1.0, 1.0, "bf16", 0.0),
    )
    monkeypatch.setattr(model, "_map_stage", lambda stage: stage)
    monkeypatch.setattr(
        model, "_linear_law_latency_s", lambda _record, _params, _batch: None,
    )
    monkeypatch.setattr(model, "_sweep_max_devices", lambda _search: 4)
    monkeypatch.setattr(
        model, "_system_config_for_parallelism", lambda *_a, **_kw: {},
    )
    return model


@pytest.mark.parametrize(
    "message",
    [
        (
            "All params would not fit on chip. System Memory Cap:93.0 GB, "
            "Weights:5.75 GB, KV Cache:102.30 GB"
        ),
        (
            "All params would not fit on the largest PP rank. System Memory "
            "Cap:93.0 GB, Max Rank Resident Memory:95.13 GB."
        ),
    ],
)
def test_exact_device_capacity_recognizes_only_genz_verdicts(message) -> None:
    assert genz_adapter._is_exact_device_capacity_failure(ValueError(message))


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError(
            "All params would not fit on the largest PP rank. System Memory "
            "Cap:93.0 GB, Max Rank Resident Memory:95.13 GB."
        ),
        ValueError("All params would not fit on the largest PP rank"),
        ValueError("tuple index out of range"),
    ],
)
def test_unrelated_exact_device_errors_remain_untyped(error) -> None:
    assert not genz_adapter._is_exact_device_capacity_failure(error)


def test_largest_pp_rank_capacity_verdict_becomes_typed(monkeypatch) -> None:
    model = _stub_main_llm(monkeypatch)
    raw = ValueError(
        "All params would not fit on the largest PP rank. System Memory "
        "Cap:93.0 GB, Max Rank Resident Memory:95.13 GB."
    )
    monkeypatch.setattr(
        genz_adapter,
        "decode_moddeling",
        lambda **_kwargs: (_ for _ in ()).throw(raw),
    )
    monkeypatch.setattr(
        genz_adapter,
        "check_model_parallelism",
        lambda *_args, **_kwargs: _ParallelismVerdict(),
    )

    with pytest.raises(CMDeviceCapacityError) as exc_info:
        model.estimate_stage_for_devices(
            {},
            SimpleNamespace(),
            stage="decode",
            batch_size=256,
            devices=["cuda:3"],
            parallelism=(1, 1),
        )

    assert isinstance(exc_info.value.__cause__, CMDeviceCapacityError)
    assert exc_info.value.__cause__.__cause__ is raw


def test_genz_pp_models_the_complete_scheduler_step_batch(monkeypatch) -> None:
    from GenZ.LLM_inference import llm_decode, llm_prefill

    observed: dict[str, list[int]] = {"prefill": [], "decode": []}

    class _System:
        @staticmethod
        def get_off_chip_mem_size() -> float:
            return 80 * 1024

    def _summary(module):
        return pd.DataFrame({
            f"Total Weights ({module.unit.unit_mem})": [1.0],
            f"KV Cache ({module.unit.unit_mem})": [1.0],
            f"Unused Weights ({module.unit.unit_mem})": [0.0],
        })

    for name, module, factory_name in (
        ("prefill", llm_prefill, "create_full_prefill_model"),
        ("decode", llm_decode, "create_full_decode_model"),
    ):
        monkeypatch.setattr(module, "get_inference_system", lambda **_kw: _System())
        monkeypatch.setattr(module, factory_name, lambda **_kw: "fixture-model")

        def _model_df(*_args, _name=name, **kwargs):
            observed[_name].append(int(kwargs["batch_size"]))
            return object()

        monkeypatch.setattr(module, "get_model_df", _model_df)
        # This test isolates the scheduler-step batch passed into GenZ.  PP
        # graph partitioning and max-rank memory ownership have dedicated
        # structural tests, so keep their input boundary stubbed here instead
        # of returning a fake dataframe to the real graph parser.
        monkeypatch.setattr(
            module,
            "get_pipeline_stage_memory_requirements",
            lambda *_args, **_kwargs: (1.0, 1.0, 1.0, 1.0),
        )
        monkeypatch.setattr(
            module,
            "get_summary_table",
            lambda _df, _unit, _module=module, **_kw: _summary(_module),
        )

    llm_prefill.prefill_moddeling(
        model="fixture",
        batch_size=1,
        input_tokens=32,
        pipeline_parallel=4,
        model_profilling=True,
    )
    llm_decode.decode_moddeling(
        model="fixture",
        batch_size=1,
        input_tokens=32,
        output_tokens=1,
        Bb=1,
        pipeline_parallel=4,
        model_profilling=True,
    )

    assert observed == {"prefill": [1], "decode": [1]}


def test_phase_level_paths_stay_analytical_and_des_gets_cycle_curve(monkeypatch) -> None:
    """Coarse phase prices never consume the scheduler-cycle overhead."""
    from rag_stack.cost_model import stage_batch_curve

    model = _stub_main_llm(monkeypatch)
    hardware = SimpleNamespace(
        search_space={"max_num_gpus_per_stage": 4},
        interconnect={},
    )
    raw = {
        "Latency": 10.0,
        "SaturatedServiceLatency": 10.0,
        "TraversalLatency": 20.0,
    }

    monkeypatch.setattr(genz_adapter, "prefill_moddeling", lambda **_kw: raw)
    monkeypatch.setattr(genz_adapter, "decode_moddeling", lambda **_kw: raw)

    def fake_stage_loader(*_a, **_kw):
        raise AssertionError("phase-level pricing loaded an engine-cycle curve")
    monkeypatch.setattr(
        stage_batch_curve,
        "load_stage_batch_curve",
        fake_stage_loader,
    )
    monkeypatch.setattr(
        genz_adapter,
        "valid_parallelism_configs",
        lambda _model, devices, batch_size=None: (
            [SimpleNamespace(parallelism=(2, 2))] if devices == 4 else []
        ),
    )
    monkeypatch.setattr(
        genz_adapter,
        "check_model_parallelism",
        lambda *_a, **_kw: _ParallelismVerdict(),
    )
    monkeypatch.setattr(genz_adapter, "_get_pareto_df", lambda frame: frame)
    monkeypatch.setattr(
        genz_adapter, "_is_prefill_decode_engine_stage", lambda _stage: True,
    )
    monkeypatch.setattr(
        genz_adapter,
        "resolve_fixed_llm_parallelism",
        lambda _cfg, stage=None: (2, 2),
    )
    monkeypatch.setattr(
        genz_adapter,
        "kv_capped_decode_batch",
        lambda _model, batch, *_a, **_kw: batch,
    )
    monkeypatch.setattr(
        model, "_performance_stage_name", lambda _cfg, kind: kind,
    )

    for stage in ("prefill", "decode"):
        monkeypatch.setattr(model, "_stage_kinds", lambda _cfg, value=stage: [value])
        single = model.estimate({}, hardware)
        assert single.latency_s == pytest.approx(0.020)

    for stage in ("prefill", "decode"):
        sweep = model.generate_sweep_df({}, hardware, stage, max_batch_size=8)
        sweep_b8 = sweep[
            (sweep["batch_size"] == 8)
            & (sweep["winning_tp"] == 2)
            & (sweep["winning_pp"] == 2)
        ].iloc[0]
        assert float(sweep_b8["time_ms"]) == pytest.approx(20.0)

        exact = model.estimate_stage_for_devices(
            {},
            hardware,
            stage=stage,
            batch_size=8,
            devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
            parallelism=(2, 2),
        )
        assert exact["latency_s"] == pytest.approx(0.020)

    from rag_stack.cost_model import engines, sweep_inputs
    from rag_stack.cost_model.llm_sim import genz_cache

    monkeypatch.setattr(
        sweep_inputs,
        "build_performance_stage_cost_models",
        lambda _cfg: {
            "generator_prefill": model,
            "generator_decode": model,
        },
    )
    monkeypatch.setattr(
        genz_cache, "cached_prefill_moddeling", lambda **_kw: raw,
    )
    monkeypatch.setattr(
        genz_cache, "cached_decode_moddeling", lambda **_kw: raw,
    )
    latency = engines.build_call_latency({}, hardware)
    assert latency.prefill_kernel_by_engine["generator"](
        32, 8, 2, 2,
    ) == pytest.approx(0.020)
    assert latency.decode_by_engine["generator"](
        32, 1, 8, 2, 2,
    ) == pytest.approx(0.020)
