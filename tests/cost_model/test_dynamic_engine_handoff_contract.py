from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.cost_model import closed_loop_sim, dynamic_pipeline, engines
from rag_stack.cost_model.candidate_layout import CANDIDATE_LAYOUT_COLUMN


def _uncalibrated_hardware():
    return SimpleNamespace(
        system_hardware="unit_dynamic_engine_handoff",
        gpu="unit_gpu",
        rag_cm={"calibration_mode": "uncalibrated_spec"},
    )


def _layout(device: str) -> dict:
    """Minimal concrete collocated layout with an exact one-rank device order."""

    stages = ("generator_prefill", "generator_decode")
    resource_id = f"gpu:{device}"
    performance_stages = {
        stage: {
            "kind": "gpu",
            "resource": resource_id,
            "devices": [device],
            "num_chips": 1,
            "engine": "generator",
            "role": "prefill" if stage.endswith("prefill") else "decode",
            "tp": 1,
            "pp": 1,
            "replicas": 1,
            "replica_devices": [[device]],
        }
        for stage in stages
    }
    return {
        "source": "unit_test",
        "available_devices": [device],
        "resource_groups": [
            {
                "id": resource_id,
                "kind": "gpu",
                "devices": [device],
                "performance_stages": list(stages),
            }
        ],
        "performance_stages": performance_stages,
        "engines": {
            "generator": {
                "pd_serving": "collocated_pd",
                "devices": [device],
                "num_chips": 1,
                "performance_stages": list(stages),
                "tp": 1,
                "pp": 1,
                "replicas": 1,
                "replica_devices": [[device]],
            }
        },
    }


def _chain_perf(*_args, **kwargs) -> engines.ChainPerf:
    return engines.ChainPerf(
        e2e=1.0,
        e2e_mean=1.0,
        ttft=0.1,
        tpot=0.01,
        svc_encode=1.0,
        decode_batch=int(kwargs.get("decode_batch") or 1),
    )


def test_cost_compatibility_entry_delegates_to_explicit_pipeline(
    monkeypatch,
) -> None:
    frame = pd.DataFrame({"candidate": [1]})
    sweep_dfs = {"generator_prefill": pd.DataFrame({"batch_size": [1]})}
    pipeline = {"system": {"batch_size_request": 1}}
    hardware = object()
    traces = [[{"stage": "generator"}]]
    placement = object()
    expected = pd.DataFrame({"authoritative": [True]})
    calls = []

    def evaluate_candidates(**kwargs):
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(dynamic_pipeline, "evaluate_candidates", evaluate_candidates)
    engine = engines.DynamicEngine(SimpleNamespace(is_agentic=False))
    result = engine.cost(
        frame,
        sweep_dfs,
        pipeline_config=pipeline,
        hardware=hardware,
        traces=traces,
        fixed_placement=placement,
    )

    assert result is expected
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["engine"] is engine
    assert kwargs["df"] is frame
    assert kwargs["sweep_dfs"] is sweep_dfs
    assert kwargs["pipeline_config"] is pipeline
    assert kwargs["hardware"] is hardware
    assert kwargs["traces"] is traces
    assert kwargs["trace_source_question_ids"] is None
    assert kwargs["fixed_placement"] is placement


def test_dynamic_engine_keeps_analytical_evidence_when_des_sets_final_qps(
    monkeypatch,
) -> None:
    analytical_seen: dict[str, list] = {}
    from rag_stack.cost_model import compiled
    # This handoff test injects analytical/simulator results. Physical lowering
    # and retained-graph ownership have independent real-compiler coverage.
    monkeypatch.setattr(compiled, "bind_plan", lambda workload, **kwargs: SimpleNamespace(
        pipeline_config=workload.pipeline_config,
        analysis_paths=lambda: workload.execution_paths,
        validate_candidate=lambda row: None, hardware=kwargs["hardware"],
        latency=engines.CallLatency(), sim_inputs=(),
    ))

    monkeypatch.setattr(
        engines,
        "build_call_latency",
        lambda _pipeline, _hardware: engines.CallLatency(),
    )
    monkeypatch.setattr(engines, "_compute_chain_latency", _chain_perf)

    def fake_des(frame, **_kwargs):
        analytical_seen["qps"] = frame["dynamic_analytical_qps"].tolist()
        analytical_seen["latency"] = frame[
            "dynamic_analytical_latency_s"
        ].tolist()
        analytical_seen["is_final"] = frame[
            "dynamic_analytical_score_is_final"
        ].tolist()
        out = frame.copy()
        out["qps"] = [101.0]
        out["latency_s"] = [202.0]
        return out

    monkeypatch.setattr(closed_loop_sim, "apply_closed_loop_simulation", fake_des)

    engine = engines.DynamicEngine(
        SimpleNamespace(is_agentic=False, dataflow_type="sequential")
    )
    result = engine.cost(
        pd.DataFrame(
            {
                "qps": [8.0],
                "latency_s": [0.25],
                "batch_size_request": [4],
                "batch_size_decode": [4],
            }
        ),
        {},
        pipeline_config={
            "system": {
                "batch_size_request": 4,
                "rag_cm": {"calibration_mode": "uncalibrated_spec"},
            },
        },
        hardware=_uncalibrated_hardware(),
        traces=[[]],
        fixed_placement=SimpleNamespace(collocation_strategy=[]),
    )

    assert analytical_seen["qps"] == [4.0]
    assert analytical_seen["latency"] == [1.0]
    assert analytical_seen["is_final"] == [False]
    assert result["qps"].tolist() == [101.0]
    assert result["latency_s"].tolist() == [202.0]
    assert result["dynamic_analytical_qps"].tolist() == [4.0]
    assert result["dynamic_analytical_latency_s"].tolist() == [1.0]
    assert result["dynamic_analytical_score_is_final"].tolist() == [False]


def test_dynamic_engine_prices_exact_layouts_and_reuses_timeout_siblings(
    monkeypatch,
) -> None:
    layout_a = _layout("cuda:0")
    layout_b = _layout("cuda:1")
    received_layouts: list[dict | None] = []

    def fake_build_call_latency(pipeline, _hardware):
        received_layouts.append(
            deepcopy((pipeline.get("system") or {}).get("layout"))
        )
        return engines.CallLatency()

    monkeypatch.setattr(engines, "build_call_latency", fake_build_call_latency)
    monkeypatch.setattr(engines, "_compute_chain_latency", _chain_perf)
    monkeypatch.setattr(
        closed_loop_sim,
        "apply_closed_loop_simulation",
        lambda frame, **_kwargs: frame,
    )

    frame = pd.DataFrame(
        {
            "qps": [8.0, 8.0, 8.0],
            "latency_s": [0.25, 0.25, 0.25],
            "batch_size_request": [4, 4, 4],
            "batch_size_decode": [4, 4, 4],
            "dynamic_batch_timeout_s": [0.001, 0.001, 0.010],
            CANDIDATE_LAYOUT_COLUMN: [layout_a, layout_b, deepcopy(layout_a)],
        }
    )
    engine = engines.DynamicEngine(
        SimpleNamespace(is_agentic=False, dataflow_type="sequential")
    )
    engine.analyze(
        frame,
        {},
        pipeline_config={
            "system": {
                "batch_size_request": 4,
                "rag_cm": {"calibration_mode": "uncalibrated_spec"},
            },
        },
        hardware=_uncalibrated_hardware(),
        traces=[[]],
        fixed_placement=SimpleNamespace(collocation_strategy=[]),
    )

    # Only concrete candidate layouts are priced. The third row differs only
    # by timeout and reuses layout A; no unused base-layout closure is built.
    assert received_layouts == [layout_a, layout_b]
    assert received_layouts[0]["engines"]["generator"]["tp"] == 1
    assert received_layouts[1]["engines"]["generator"]["tp"] == 1
    assert received_layouts[0]["engines"]["generator"]["devices"] == [
        "cuda:0"
    ]
    assert received_layouts[1]["engines"]["generator"]["devices"] == [
        "cuda:1"
    ]


@pytest.mark.parametrize("coarse_qps", [None, 0.0, -99.0, float("nan"), 1e30])
def test_analysis_ignores_coarse_scores_and_never_calls_des(monkeypatch, coarse_qps):
    monkeypatch.setattr(
        engines, "build_call_latency", lambda *_args: engines.CallLatency(),
    )
    monkeypatch.setattr(engines, "_compute_chain_latency", _chain_perf)

    def reject_des(*_args, **_kwargs):
        raise AssertionError("analytical scoring must not execute simulation")

    monkeypatch.setattr(closed_loop_sim, "apply_closed_loop_simulation", reject_des)
    frame = pd.DataFrame({"batch_size_request": [4], "batch_size_decode": [4]})
    if coarse_qps is not None:
        frame["qps"] = coarse_qps
        frame["qps_per_chip"] = coarse_qps
        frame["latency_s"] = float("nan")
    original = frame.copy(deep=True)
    engine = engines.DynamicEngine(SimpleNamespace(is_agentic=False))

    result = engine.analyze(
        frame, {}, pipeline_config={}, hardware=_uncalibrated_hardware(),
        traces=[[]], fixed_placement=SimpleNamespace(collocation_strategy=[]),
    )

    assert result["dynamic_analytical_qps"].tolist() == [4.0]
    assert result["dynamic_analytical_latency_s"].tolist() == [1.0]
    assert result["dynamic_analytical_score_is_final"].tolist() == [False]
    assert "qps_per_chip" not in result
    assert "simulator_status" not in result
    pd.testing.assert_frame_equal(frame, original)


def test_analysis_without_resource_demand_rejects_coarse_score_fallback(monkeypatch):
    monkeypatch.setattr(
        engines, "build_call_latency", lambda *_args: engines.CallLatency(),
    )
    monkeypatch.setattr(
        engines, "_compute_chain_latency",
        lambda *_args, **_kwargs: engines.ChainPerf(e2e=1.0, e2e_mean=1.0),
    )
    engine = engines.DynamicEngine(SimpleNamespace(is_agentic=False))
    frame = pd.DataFrame({"batch_size_request": [4], "qps": [100.0]})

    with pytest.raises(RuntimeError, match="no finite resource capacity"):
        engine.analyze(
            frame, {}, pipeline_config={}, hardware=_uncalibrated_hardware(),
            traces=[[]], fixed_placement=SimpleNamespace(collocation_strategy=[]),
        )


def test_simulation_executes_precomputed_plan_without_rescreening(monkeypatch):
    from rag_stack.cost_model import hybrid_screening

    frozen = SimpleNamespace(
        enabled=False, reason="test", generation_sha256="test",
        profile_content_sha256={},
    )
    monkeypatch.setattr(
        closed_loop_sim, "_pipeline_with_frozen_stage_batch_selection",
        lambda cfg, _hardware: cfg,
    )
    monkeypatch.setattr(
        closed_loop_sim, "_stage_batch_generation_for_pipeline",
        lambda *_args: frozen,
    )
    monkeypatch.setattr(
        closed_loop_sim, "_rag_cm_calibration_mode",
        lambda *_args: "uncalibrated_spec",
    )

    def reject_screening(*_args, **_kwargs):
        raise AssertionError("simulation must honor the supplied plan")

    monkeypatch.setattr(hybrid_screening, "screen_des_candidates", reject_screening)
    frame = pd.DataFrame({
        "batch_size_request": [1, 2], "batch_size_decode": [1, 2],
        "qps": [10.0, 20.0], "latency_s": [0.1, 0.2],
    }, index=[7, 7])
    plan = hybrid_screening.HybridScreeningResult(
        selected_positions=(1,),
        diagnostics=pd.DataFrame({
            "selection_reasons": [(), ("test_selected",)],
            "exclusion_reason": ["test_ranked_out", ""],
        }, index=frame.index),
        summary={"mode": "test_precomputed"},
    )
    built_batches = []

    def builder(*_args, candidate_row, **_kwargs):
        built_batches.append(int(candidate_row["batch_size_request"]))
        return ([], [], [], 0, 1, 1), None

    def simulate(*_args, population, **_kwargs):
        return closed_loop_sim.SimResult(
            qps=3.0, latency_mean_s=0.5, engine_fill_mean={},
            stage_batch_mean={}, window_s=1.0, completions=10,
            population=population,
        )

    result = closed_loop_sim.apply_closed_loop_simulation(
        frame, pipeline_config={}, hardware=object(), traces=[],
        screening=plan, builder_fn=builder, simulate_fn=simulate,
    )

    assert built_batches == [2]
    assert result["qps"].tolist() == [0.0, 3.0]
    assert result["latency_s"].tolist() == [float("inf"), 0.5]
    assert result["simulator_status"].tolist() == ["screened_out", "ok"]
    assert result["hybrid_screening_mode"].tolist() == ["test_precomputed"] * 2
