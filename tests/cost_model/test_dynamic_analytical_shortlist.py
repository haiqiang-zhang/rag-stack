"""DynamicEngine analytical-score to authoritative-DES shortlist guards."""

from __future__ import annotations

from collections import defaultdict
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.cost_model import hybrid_screening
from rag_stack.types import PerformanceSelection


_THROUGHPUT = PerformanceSelection(selection="max_throughput")
_TIMEOUTS = (0.002, 0.01, 0.05)


def _layout(
    topology: int,
    *,
    generator_pd: str = "collocated_pd",
    query_expansion_pd: str | None = None,
) -> dict:
    device = f"gpu:{topology}"
    stages = {
        "generator_prefill": {
            "engine": "generator",
            "role": "prefill",
            "resource": "generator",
            "devices": [device],
            "tp": 1,
            "pp": 1,
            "replicas": 1,
        },
        "generator_decode": {
            "engine": "generator",
            "role": "decode",
            "resource": "generator",
            "devices": [device],
            "tp": 1,
            "pp": 1,
            "replicas": 1,
        },
    }
    engines = {
        "generator": {
            "pd_serving": generator_pd,
            "devices": [device],
            "tp": 1,
            "pp": 1,
            "replicas": 1,
        },
    }
    if query_expansion_pd is not None:
        engines["query_expansion"] = {
            "pd_serving": query_expansion_pd,
            "devices": [device],
            "tp": 1,
            "pp": 1,
            "replicas": 1,
        }
    return {
        "device_layout_id": f"layout:{topology}",
        "available_devices": [device],
        "resource_groups": [{
            "id": "generator",
            "kind": "gpu",
            "devices": [device],
            "performance_stages": list(stages),
        }],
        "performance_stages": stages,
        "engines": engines,
    }


def _row(
    parent: int,
    *,
    score: float,
    layout: dict,
    timeout: float = 0.002,
    qps_cap: float = 1.0,
    dataflow: str = "react",
) -> dict:
    return {
        "parent": parent,
        "dynamic_analytical_qps": score,
        # This is deliberately present in tests which prove it cannot rank
        # the shortlist.  It remains an independent upstream diagnostic.
        "qps_cap_upper_bound": qps_cap,
        "batch_size_request": parent + 1,
        "batch_size_decode": parent + 1,
        "dynamic_batch_timeout_s": timeout,
        "pipeline_mode": dataflow,
        "candidate_system_layout": layout,
    }


def _force_screening(monkeypatch) -> None:
    monkeypatch.setattr(hybrid_screening, "_MIN_ROWS_FOR_SCREENING", 1)
    monkeypatch.setenv("CM_DES_HYBRID_FORCE", "1")
    monkeypatch.delenv("CM_DES_EXACT_ALL", raising=False)


def test_two_role_llm_uses_dynamic_score_instead_of_exhaustive_bypass(
    monkeypatch,
) -> None:
    _force_screening(monkeypatch)
    layout = _layout(0)
    frame = pd.DataFrame([
        _row(parent, score=100.0 - parent, layout=layout)
        for parent in range(20)
    ])

    result = hybrid_screening.screen_des_candidates(
        frame,
        selection=_THROUGHPUT,
        production=False,
    )

    assert 0 < len(result.selected_positions) < len(frame)
    assert result.summary["mode"] == (
        "hybrid_dynamic_engine_analytical_des"
    )
    assert result.summary["score"] == (
        "dynamic_engine_trace_analytical_qps"
    )
    assert result.summary["score_is_final"] is False
    assert not result.diagnostics["analytical_screening_is_final"].any()


def test_dataflow_and_generator_qe_pd_keep_independent_structural_seats(
    monkeypatch,
) -> None:
    _force_screening(monkeypatch)
    rows = []
    expected_strata = set()
    stratum = 0
    for dataflow in ("sequential", "react"):
        for generator_pd in ("collocated_pd", "disagg_pd"):
            for qe_pd in ("collocated_pd", "disagg_pd"):
                expected_strata.add((dataflow, generator_pd, qe_pd))
                layout = _layout(
                    stratum,
                    generator_pd=generator_pd,
                    query_expansion_pd=qe_pd,
                )
                for parent in range(20):
                    # Later strata are globally worse than earlier ones.
                    # Their survival proves these three axes own independent
                    # structural seats instead of relying on global rescue.
                    score = 10_000.0 - stratum * 1_000.0 - parent
                    rows.append(_row(
                        parent,
                        score=score,
                        layout=layout,
                        dataflow=dataflow,
                    ))
                stratum += 1
    frame = pd.DataFrame(rows)

    result = hybrid_screening.screen_des_candidates(
        frame,
        selection=_THROUGHPUT,
        production=False,
    )
    selected_diagnostics = result.diagnostics.loc[
        result.diagnostics["selected"]
    ]
    selected_strata = set(zip(
        selected_diagnostics["analytical_dataflow"],
        selected_diagnostics["analytical_generator_pd"],
        selected_diagnostics["analytical_query_expansion_pd"],
    ))

    assert selected_strata == expected_strata
    assert {
        mode
        for _dataflow, mode, _qe in selected_strata
    } == {"collocated_pd", "disagg_pd"}
    assert {
        mode
        for _dataflow, _generator, mode in selected_strata
    } == {"collocated_pd", "disagg_pd"}
    assert len(result.selected_positions) < len(frame)


def test_dynamic_score_orders_shortlist_and_rago_cap_does_not(
    monkeypatch,
) -> None:
    _force_screening(monkeypatch)
    layout = _layout(0)
    frame = pd.DataFrame([
        _row(
            parent,
            score=float(parent + 1),
            qps_cap=float(10_000 - parent),
            layout=layout,
        )
        for parent in range(20)
    ])

    result = hybrid_screening.screen_des_candidates(
        frame,
        selection=_THROUGHPUT,
        production=False,
    )
    selected = set(result.selected_positions)

    assert 19 in selected
    assert 0 not in selected
    assert result.diagnostics.iloc[19]["analytical_screening_qps"] == 20.0
    assert result.summary["score_column"] == "dynamic_analytical_qps"


def test_669_parents_with_three_timeout_siblings_stays_bounded() -> None:
    rows = []
    for parent in range(669):
        layout = _layout(parent % 12)
        for timeout in _TIMEOUTS:
            rows.append(_row(
                parent,
                score=10_000.0 - parent,
                layout=layout,
                timeout=timeout,
            ))
    frame = pd.DataFrame(rows)
    assert len(frame) == 2_007

    result = hybrid_screening.screen_des_candidates(frame, selection=_THROUGHPUT, production=True)
    selected = set(result.selected_positions)

    by_parent: dict[int, list[int]] = defaultdict(list)
    for position, parent in enumerate(frame["parent"]):
        by_parent[int(parent)].append(position)
    assert all(
        len(selected.intersection(positions)) in {0, len(_TIMEOUTS)}
        for positions in by_parent.values()
    )
    selected_topologies = {
        frame.iloc[position]["candidate_system_layout"]["device_layout_id"]
        for position in result.selected_positions
    }
    # All twelve physical layouts retain a DES seat.  Five batch parents per
    # topology plus the ten-parent global rescue is an absolute 70-parent
    # ceiling; atomic timeout expansion makes at most 210 rows.
    assert selected_topologies == {
        f"layout:{topology}" for topology in range(12)
    }
    assert 0 < len(result.selected_positions) <= 210
    assert len(result.selected_positions) < len(frame)
    assert len(result.selected_positions) % len(_TIMEOUTS) == 0
    assert result.summary["topologies_per_stratum"] == 12
    assert result.summary["score"] == (
        "dynamic_engine_trace_analytical_qps"
    )


def test_large_production_sweep_without_dynamic_score_fails_closed() -> None:
    layout = _layout(0)
    frame = pd.DataFrame([
        {
            key: value
            for key, value in _row(
                parent,
                score=1.0,
                layout=layout,
            ).items()
            if key != "dynamic_analytical_qps"
        }
        for parent in range(256)
    ])

    with pytest.raises(
        RuntimeError,
        match="missing columns: dynamic_analytical_qps",
    ):
        hybrid_screening.screen_des_candidates(frame, selection=_THROUGHPUT, production=True)


@pytest.mark.parametrize("bad_score", (0.0, -1.0, math.nan, math.inf))
def test_large_production_sweep_with_invalid_dynamic_score_fails_closed(
    bad_score: float,
) -> None:
    layout = _layout(0)
    frame = pd.DataFrame([
        _row(parent, score=1_000.0 - parent, layout=layout)
        for parent in range(256)
    ])
    frame.loc[17, "dynamic_analytical_qps"] = bad_score

    with pytest.raises(RuntimeError, match="finite positive"):
        hybrid_screening.screen_des_candidates(frame, selection=_THROUGHPUT, production=True)


def test_small_or_injected_calls_remain_exact_without_score() -> None:
    small = pd.DataFrame([{"candidate": index} for index in range(255)])
    small_result = hybrid_screening.screen_des_candidates(
        small,
        production=True,
    )
    assert small_result.selected_positions == tuple(range(len(small)))

    injected = pd.DataFrame([{"candidate": index} for index in range(300)])
    injected_result = hybrid_screening.screen_des_candidates(
        injected,
        selection=_THROUGHPUT,
        production=False,
    )
    assert injected_result.selected_positions == tuple(range(len(injected)))


def _opposing_objectives_frame() -> pd.DataFrame:
    """The lowest-latency batch has the lowest QPS, within each topology."""
    rows = []
    for topology in range(2):
        for parent in range(30):
            for timeout in _TIMEOUTS:
                row = _row(
                    parent, score=float(parent + 1), layout=_layout(topology),
                    timeout=timeout,
                )
                row["dynamic_analytical_latency_s"] = float(parent + 1)
                rows.append(row)
    # The production API promises positional selection even for duplicate labels.
    return pd.DataFrame(rows, index=[7] * len(rows))


@pytest.mark.parametrize("selection", [None, PerformanceSelection(selection="min_latency")])
def test_latency_target_preserves_low_latency_batches_and_all_timeouts(monkeypatch, selection):
    _force_screening(monkeypatch)
    frame = _opposing_objectives_frame()
    before = frame.copy(deep=True)

    result = hybrid_screening.screen_des_candidates(frame, selection=selection)
    chosen = frame.iloc[list(result.selected_positions)]

    assert 0 in set(chosen["parent"])
    assert 29 not in set(chosen["parent"])
    for topology in (0, 1):
        lowest = chosen[
            chosen["candidate_system_layout"].map(
                lambda layout: layout["device_layout_id"] == f"layout:{topology}"
            ) & chosen["parent"].eq(0)
        ]
        assert set(lowest["dynamic_batch_timeout_s"]) == set(_TIMEOUTS)
    assert result.summary["metric"] == "latency_s"
    assert result.summary["higher_is_better"] is False
    assert result.summary["score_column"] == "dynamic_analytical_latency_s"
    pd.testing.assert_frame_equal(frame, before)


def test_changing_target_reverses_screening_ranking(monkeypatch):
    _force_screening(monkeypatch)
    frame = _opposing_objectives_frame()
    result = hybrid_screening.screen_des_candidates(frame, selection=_THROUGHPUT)
    chosen = frame.iloc[list(result.selected_positions)]
    assert 29 in set(chosen["parent"])
    assert 0 not in set(chosen["parent"])
    assert result.summary["higher_is_better"] is True


def test_constrained_policy_keeps_objective_and_slo_rankings(monkeypatch):
    _force_screening(monkeypatch)
    frame = _opposing_objectives_frame()
    result = hybrid_screening.screen_des_candidates(
        frame,
        selection=PerformanceSelection(
            selection="constrained", optimize="throughput_qps",
            constraint={"latency_s": 1.0},
        ),
    )
    chosen = frame.iloc[list(result.selected_positions)]
    assert {0, 29} <= set(chosen["parent"])
    assert result.summary["screening_metrics"] == ["throughput_qps", "latency_s"]
    assert result.summary["selected_rows"] == len(chosen)
    assert result.summary["selected_parents"] * len(_TIMEOUTS) == len(chosen)
    assert (result.diagnostics[result.diagnostics["selected"]]["exclusion_reason"] == "").all()


def test_weighted_policy_scores_compromise_and_retains_metric_endpoints(monkeypatch):
    _force_screening(monkeypatch)
    frame = pd.DataFrame([
        _row(parent, score=100.0, layout=_layout(0)) for parent in range(27)
    ])
    frame["dynamic_analytical_latency_s"] = 100.0
    frame.loc[25, ["dynamic_analytical_qps", "dynamic_analytical_latency_s"]] = [80.0, 20.0]
    frame.loc[26, ["dynamic_analytical_qps", "dynamic_analytical_latency_s"]] = [1.0, 1.0]
    result = hybrid_screening.screen_des_candidates(
        frame,
        selection=PerformanceSelection(
            selection="weighted", weights={"throughput_qps": 1.0, "latency_s": 1.0},
        ),
    )
    assert {0, 25, 26} <= set(result.selected_positions)
    assert result.diagnostics.loc[25, "global_parent_rank"] == 1
    assert result.summary["screening_metrics"] == ["weighted", "throughput_qps", "latency_s"]


@pytest.mark.parametrize("bad_latency", [-1.0, math.nan, math.inf])
def test_latency_target_rejects_invalid_analytical_latency(monkeypatch, bad_latency):
    _force_screening(monkeypatch)
    frame = _opposing_objectives_frame().reset_index(drop=True)
    frame.loc[0, "dynamic_analytical_latency_s"] = bad_latency
    with pytest.raises(RuntimeError, match="finite non-negative.*dynamic_analytical_latency_s"):
        hybrid_screening.screen_des_candidates(frame)


def test_latency_target_requires_latency_column_even_when_qps_exists(monkeypatch):
    _force_screening(monkeypatch)
    frame = _opposing_objectives_frame().drop(columns="dynamic_analytical_latency_s")
    with pytest.raises(RuntimeError, match="missing columns: dynamic_analytical_latency_s"):
        hybrid_screening.screen_des_candidates(frame)


def test_zero_latency_is_valid_and_exact_mode_keeps_all_candidates(monkeypatch):
    _force_screening(monkeypatch)
    frame = _opposing_objectives_frame().reset_index(drop=True)
    frame.loc[0, "dynamic_analytical_latency_s"] = 0.0
    result = hybrid_screening.screen_des_candidates(frame)
    assert 0 in result.selected_positions

    monkeypatch.setenv("CM_DES_EXACT_ALL", "1")
    exact = hybrid_screening.screen_des_candidates(frame.drop(columns="dynamic_analytical_latency_s"))
    assert exact.selected_positions == tuple(range(len(frame)))


@pytest.mark.parametrize("metric,policy", [("ttft_s", "min_ttft"), ("tpot_s", "min_tpot")])
def test_latency_submetrics_use_their_own_analytical_columns(monkeypatch, metric, policy):
    _force_screening(monkeypatch)
    frame = _opposing_objectives_frame()
    column = "dynamic_analytical_latency_s_" + metric.removesuffix("_s")
    frame[column] = frame["dynamic_analytical_latency_s"]
    result = hybrid_screening.screen_des_candidates(
        frame, selection=PerformanceSelection(selection=policy),
    )
    assert result.summary["score_column"] == column
    assert result.summary["metric"] == metric
    assert 0 in result.selected_positions
    assert len(frame) - 1 not in result.selected_positions


@pytest.mark.parametrize("policy", ["min_latency", "max_throughput"])
def test_des_adapter_passes_target_and_final_selection_uses_simulation(monkeypatch, policy):
    from rag_stack.cost_model import closed_loop_sim as sim
    from rag_stack.cost_model.selection import apply_selection

    _force_screening(monkeypatch)
    frame = _opposing_objectives_frame().reset_index(drop=True)
    frame["qps"] = frame["dynamic_analytical_qps"]
    frame["latency_s"] = frame["dynamic_analytical_latency_s"]
    selection = PerformanceSelection(selection=policy)
    hardware = SimpleNamespace(performance=selection)
    frozen = SimpleNamespace(
        enabled=False, reason="test", generation_sha256="test",
        profile_content_sha256={},
    )
    monkeypatch.setattr(sim, "_pipeline_with_frozen_stage_batch_selection", lambda cfg, hw: cfg)
    monkeypatch.setattr(sim, "_stage_batch_generation_for_pipeline", lambda cfg, hw: frozen)
    monkeypatch.setattr(sim, "_rag_cm_calibration_mode", lambda cfg, hw: "uncalibrated_spec")
    simulated_parents = []

    def builder(_pipeline, _hardware, _traces, _groups, *, candidate_row, **kwargs):
        parent = int(candidate_row["parent"])
        simulated_parents.append(parent)
        return ([parent], [], [], 0, 1, 1), None

    def simulate(stages, engines, queries, *, population, **kwargs):
        # Deliberately different from analytical values/rankings. The actual
        # saturation driver and adapter must publish only these DES results.
        value = float(1000 - stages[0])
        return sim.SimResult(
            qps=value, latency_mean_s=value, engine_fill_mean={},
            stage_batch_mean={}, window_s=1.0, completions=10, population=population,
        )

    out = sim.apply_closed_loop_simulation(
        frame, pipeline_config={}, hardware=hardware, traces=[], latency=None,
        builder_fn=builder, simulate_fn=simulate,
    )
    if policy == "min_latency":
        assert 0 in simulated_parents and 29 not in simulated_parents
        expected_parent = max(simulated_parents)
    else:
        assert 29 in simulated_parents and 0 not in simulated_parents
        expected_parent = min(simulated_parents)
    winner, score = apply_selection(out, selection)
    assert winner["parent"] == expected_parent
    assert winner["latency_s"] == 1000 - expected_parent
    assert abs(score) == 1000 - expected_parent
    assert set(out["analytical_screening_metric"]) == {
        "latency_s" if policy == "min_latency" else "throughput_qps"
    }
    assert (out.loc[out["simulator_status"] == "screened_out", "qps"] == 0).all()
    assert out.loc[out["simulator_status"] == "screened_out", "latency_s"].map(math.isinf).all()
