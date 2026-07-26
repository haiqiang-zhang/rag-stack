"""Targeted integration guards for analytical screening -> injected DES."""

from collections import Counter, defaultdict
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.cost_model.closed_loop_sim import (
    SimResult,
    SimStage,
    apply_closed_loop_simulation,
)
from rag_stack.cost_model.hybrid_screening import (
    HybridScreeningResult,
    _topology_identity,
    screen_des_candidates,
)
import rag_stack.cost_model.hybrid_screening as hybrid_screening
from rag_stack.types import PerformanceSelection


_TIMEOUTS = (0.002, 0.01, 0.05)
_THROUGHPUT = PerformanceSelection(selection="max_throughput")


@pytest.fixture(autouse=True)
def _freeze_cpu_test_component_policy(monkeypatch):
    from rag_stack.cost_model import closed_loop_sim

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


def _layout(topology: int) -> dict:
    device = f"test-device:{topology}"
    stage = "generator_decode"
    return {
        "device_layout_id": f"topology:{topology}",
        "available_devices": [device],
        "resource_groups": [{
            "id": "generator",
            "kind": "gpu",
            "devices": [device],
            "performance_stages": [stage],
        }],
        "performance_stages": {
            stage: {
                "engine": "generator",
                "role": "decode",
                "resource_group": "generator",
                "devices": [device],
                "tp": 1,
                "pp": 1,
                "replicas": 1,
            },
        },
        "engines": {
            "generator": {
                "pd_serving": "collocated_pd",
                "devices": [device],
                "tp": 1,
                "pp": 1,
                "replicas": 1,
            },
        },
    }


def test_topology_identity_includes_exact_tp_pp_not_only_rago_layout_id():
    tp1 = _layout(0)
    tp2 = _layout(0)
    tp2["performance_stages"]["generator_decode"].update({
        "tp": 2, "pp": 1, "replicas": 1,
    })

    assert tp1["device_layout_id"] == tp2["device_layout_id"]
    assert _topology_identity(tp1) != _topology_identity(tp2)


def _large_candidate_frame() -> pd.DataFrame:
    """Ten topologies x ten batch parents x three timeout siblings."""

    rows = []
    for topology in range(10):
        layout = _layout(topology)
        for parent in range(10):
            request = 2 ** (parent % 5)
            decode = request * (2 if parent >= 5 else 1)
            # A strict analytical order makes the expected shortlist stable,
            # while the test never relies on its exact size.
            capacity = 1_000.0 - topology * 50.0 - parent
            marker = topology * 10 + parent
            for timeout in _TIMEOUTS:
                rows.append({
                    "candidate_marker": marker,
                    "qps": capacity,
                    "qps_per_chip": capacity,
                    "qps_generator_decode": capacity,
                    "latency_s": 1.0 / capacity,
                    "dynamic_analytical_qps": capacity,
                    "dynamic_analytical_latency_s": 1.0 / capacity,
                    "batch_size_request": request,
                    "batch_size_decode": decode,
                    "batch_size_generator_decode": decode,
                    "dynamic_batch_timeout_s": timeout,
                    "num_chips_e2e": 1,
                    "num_retrieval_servers": 1,
                    "pipeline_mode": "react",
                    "collocation_strategy": [["generator_decode"]],
                    "candidate_system_layout": layout,
                })
    frame = pd.DataFrame(rows)
    # Exercise positional bookkeeping independently of index labels.
    frame.index = [index // 2 for index in range(len(frame))]
    return frame


def _injected_des_observers():
    builder_calls = []
    simulate_calls = []

    def builder(*args, candidate_row=None, **kwargs):
        del args, kwargs
        marker = int(candidate_row["candidate_marker"])
        timeout = float(candidate_row["dynamic_batch_timeout_s"])
        builder_calls.append((marker, timeout))
        stage = SimStage(
            name=f"candidate:{marker}",
            service_s=lambda batch: float(batch),
            gpu_share=0.0,
            card=None,
            batch_cap=1,
            timeout_s=timeout,
        )
        return ([stage], [], [], 1, 1, None), None

    def simulate(stages, *args, **kwargs):
        del args
        marker = int(stages[0].name.split(":", 1)[1])
        timeout = float(stages[0].timeout_s)
        population = int(kwargs["population"])
        simulate_calls.append((marker, timeout, population))
        return SimResult(
            qps=10_000.0 + marker,
            latency_mean_s=1.0 + timeout,
            engine_fill_mean={},
            stage_batch_mean={},
            window_s=1.0,
            completions=population,
            population=population,
        )

    return builder, simulate, builder_calls, simulate_calls


def _apply(frame, builder, simulate, screening):
    return apply_closed_loop_simulation(
        frame,
        pipeline_config={
            "system": {"cm_search_space": {"no_microbatching": True}},
        },
        hardware=SimpleNamespace(performance=_THROUGHPUT),
        traces=[[{"stage": "generator"}]],
        builder_fn=builder,
        simulate_fn=simulate,
        screening=screening,
    )


def test_hybrid_plan_calls_injected_des_only_for_selected_rows():
    frame = _large_candidate_frame()
    assert len(frame) == 300
    plan = screen_des_candidates(frame, selection=_THROUGHPUT, production=True)
    selected = set(plan.selected_positions)
    assert 0 < len(selected) < len(frame)

    # Every timeout-free parent is either kept with all three siblings or not
    # kept at all.
    by_parent = defaultdict(list)
    for position, row in enumerate(frame.itertuples(index=False)):
        by_parent[int(row.candidate_marker)].append(position)
    assert all(
        len(set(positions) & selected) in {0, len(_TIMEOUTS)}
        for positions in by_parent.values()
    )

    builder, simulate, builder_calls, simulate_calls = _injected_des_observers()
    out = _apply(frame, builder, simulate, plan)

    expected_calls = [
        (
            int(frame.iloc[position]["candidate_marker"]),
            float(frame.iloc[position]["dynamic_batch_timeout_s"]),
        )
        for position in sorted(selected)
    ]
    assert builder_calls == expected_calls
    # The unchanged throughput at the second concurrency rung reaches plateau.
    simulated_rows = Counter((marker, timeout) for marker, timeout, _ in simulate_calls)
    assert simulated_rows == Counter({call: 2 for call in expected_calls})

    screened = [position for position in range(len(frame)) if position not in selected]
    assert (out.iloc[screened]["qps"] == 0.0).all()
    assert out.iloc[screened]["latency_s"].map(math.isinf).all()
    assert set(out.iloc[screened]["simulator_status"]) == {"screened_out"}
    assert set(out.iloc[screened]["simulator_error"]) == {
        "dynamic_analytical_screened_out",
    }
    assert (out.iloc[screened]["qps_cap_upper_bound"] > 0.0).all()
    assert set(out.iloc[sorted(selected)]["simulator_status"]) == {"ok"}
    assert out["simulator_evaluated_deployments"].eq(len(selected)).all()
    assert out["hybrid_screened_candidates"].eq(len(screened)).all()


def test_exact_all_overrides_forced_hybrid_and_evaluates_every_row(monkeypatch):
    frame = _large_candidate_frame()
    monkeypatch.setenv("CM_DES_HYBRID_FORCE", "1")
    monkeypatch.setenv("CM_DES_EXACT_ALL", "1")
    builder, simulate, builder_calls, simulate_calls = _injected_des_observers()
    plan = screen_des_candidates(frame, selection=_THROUGHPUT, production=True)

    out = _apply(frame, builder, simulate, plan)

    assert len(builder_calls) == len(frame)
    assert len(simulate_calls) == 2 * len(frame)
    assert set(out["simulator_status"]) == {"ok"}
    assert (out["qps"] > 0.0).all()
    assert out["latency_s"].map(math.isfinite).all()
    assert out["simulator_evaluated_deployments"].eq(len(frame)).all()
    assert out["hybrid_shortlisted_candidates"].eq(len(frame)).all()
    assert out["hybrid_screened_candidates"].eq(0).all()
    assert set(out["hybrid_screening_mode"]) == {"exhaustive"}
    assert set(out["hybrid_selection_reasons"]) == {
        ("below_threshold_or_exact_mode",),
    }


def test_adapter_honors_explicit_plan_without_running_screening(monkeypatch):
    frame = _large_candidate_frame().iloc[:3].copy()
    plan = HybridScreeningResult(
        selected_positions=(2,),
        diagnostics=pd.DataFrame({
            "position": [0, 1, 2],
            "selected": [False, False, True],
            "selection_reasons": [(), (), ("test_selected",)],
            "exclusion_reason": ["ranked_out", "ranked_out", ""],
        }, index=frame.index),
        summary={"mode": "test_precomputed", "selected_rows": 1},
    )

    def reject_screening(*_args, **_kwargs):
        raise AssertionError("an explicit simulation plan must not be screened again")

    monkeypatch.setattr(hybrid_screening, "screen_des_candidates", reject_screening)
    builder, simulate, builder_calls, _simulate_calls = _injected_des_observers()

    out = _apply(frame, builder, simulate, plan)

    assert builder_calls == [(
        int(frame.iloc[2]["candidate_marker"]),
        float(frame.iloc[2]["dynamic_batch_timeout_s"]),
    )]
    assert out["simulator_error"].tolist() == [
        "dynamic_analytical_screened_out",
        "dynamic_analytical_screened_out",
        "",
    ]
    assert out["simulator_status"].tolist() == ["screened_out", "screened_out", "ok"]
