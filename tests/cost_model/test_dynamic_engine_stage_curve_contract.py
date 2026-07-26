from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.cost_model import closed_loop_sim, engines, stage_batch_curve


_REQUEST_BATCH = 8
_DECODE_BATCH = 16
_PREFILL_ENGINE_DEPTH = 3
_DECODE_ENGINE_DEPTH = 17
_UNIFIED_ENGINE_DEPTH = 13
_CURVE_OVERHEAD_S = 0.02


def _chain_perf(*_args, **kwargs) -> engines.ChainPerf:
    return engines.ChainPerf(
        e2e=1.0,
        e2e_mean=1.0,
        ttft=0.1,
        tpot=0.01,
        svc_gen=0.5,
        svc_prefill=0.25,
        svc_decode=0.25,
        svc_prefill_by_stage={"generator_prefill": 0.25},
        svc_decode_by_stage={"generator_decode": 0.25},
        decode_batch=int(kwargs.get("decode_batch") or 1),
        gen_in_tokens_per_query=128.0,
        gen_out_tokens_per_query=2.0,
    )


def _install_cpu_doubles(monkeypatch, curve_calls: list[dict]) -> None:
    generation = SimpleNamespace(
        generation_sha256="frozen-generation",
        profile_content_sha256={"generator": "content"},
        enabled=True,
    )

    class SpyEngineCycleCurveApplicator:
        def __init__(self, received_generation):
            assert received_generation is generation

        def corrected_service_s(self, **kwargs):
            curve_calls.append(dict(kwargs))
            return float(kwargs["component_service_s"]) + _CURVE_OVERHEAD_S

    monkeypatch.setattr(
        closed_loop_sim,
        "_pipeline_with_frozen_stage_batch_selection",
        lambda pipeline, _hardware: pipeline,
    )
    monkeypatch.setattr(
        closed_loop_sim,
        "_stage_batch_generation_for_pipeline",
        lambda _pipeline, _hardware: generation,
    )
    monkeypatch.setattr(
        stage_batch_curve,
        "EngineCycleCurveApplicator",
        SpyEngineCycleCurveApplicator,
    )
    monkeypatch.setattr(
        engines,
        "build_call_latency",
        lambda _pipeline, _hardware: engines.CallLatency(),
    )
    monkeypatch.setattr(engines, "_compute_chain_latency", _chain_perf)


def _pipeline() -> dict:
    return {
        "system": {
            "batch_size_request": _REQUEST_BATCH,
            "batch_size_decode": _DECODE_BATCH,
            "vllm": {
                "engines": {
                    "generator": {
                        "prefill_max_num_seqs": _PREFILL_ENGINE_DEPTH,
                        "decode_max_num_seqs": _DECODE_ENGINE_DEPTH,
                        "max_num_seqs": _UNIFIED_ENGINE_DEPTH,
                    }
                }
            },
        }
    }


def _groups(pd_mode: str) -> list[list[str]]:
    if pd_mode == "collocated_pd":
        return [["generator_prefill", "generator_decode"]]
    assert pd_mode == "disagg_pd"
    return [["generator_prefill"], ["generator_decode"]]


def _run(
    *,
    pd_mode: str,
    traces: list,
    is_agentic: bool,
) -> pd.DataFrame:
    groups = _groups(pd_mode)
    frame = pd.DataFrame(
        {
            "qps": [100.0],
            "latency_s": [0.1],
            "batch_size_request": [_REQUEST_BATCH],
            "batch_size_decode": [_DECODE_BATCH],
            "collocation_strategy": [groups],
        }
    )
    return engines.DynamicEngine(
        SimpleNamespace(
            is_agentic=is_agentic,
            dataflow_type="agentic" if is_agentic else "sequential",
        )
    ).analyze(
        frame,
        {},
        pipeline_config=_pipeline(),
        hardware=object(),
        traces=traces,
        fixed_placement=SimpleNamespace(collocation_strategy=groups),
    )


def _generator_call(*, input_tokens: int, output_tokens: int) -> dict:
    return {
        "stage": "generator",
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def test_collocated_analytical_uses_one_unified_curve_at_engine_depth(
    monkeypatch,
) -> None:
    curve_calls: list[dict] = []
    _install_cpu_doubles(monkeypatch, curve_calls)

    result = _run(
        pd_mode="collocated_pd",
        traces=[[
            _generator_call(input_tokens=9_000, output_tokens=2),
        ]],
        is_agentic=False,
    )

    assert curve_calls == [
        {
            "engine_stage": "generator",
            "role": "unified",
            "prefill_stage": "generator_prefill",
            "decode_stage": "generator_decode",
            "resident_batch": _UNIFIED_ENGINE_DEPTH,
            # DynamicEngine asks for the bare curve delta and composes it with
            # component demand once, after the P+D lanes have been combined.
            "component_service_s": 1.0,
        }
    ]
    audit = result.loc[0, "analytical_engine_cycle_curve_cycles"]["generator"]
    assert audit["pd_serving"] == "collocated_pd"
    assert audit["resident_batch"] == _UNIFIED_ENGINE_DEPTH
    assert audit["mean_cycles_per_query"] == 4.0
    assert "mean_prefill_cycles_per_query" not in audit
    assert result.loc[0, "dynamic_analytical_stage_batch_enabled"]


def test_disagg_analytical_uses_direct_prefill_and_decode_role_curves(
    monkeypatch,
) -> None:
    curve_calls: list[dict] = []
    _install_cpu_doubles(monkeypatch, curve_calls)

    result = _run(
        pd_mode="disagg_pd",
        traces=[[
            _generator_call(input_tokens=9_000, output_tokens=2),
        ]],
        is_agentic=False,
    )

    assert [call["role"] for call in curve_calls] == ["prefill", "decode"]
    assert all(call["role"] != "unified" for call in curve_calls)
    assert [call["resident_batch"] for call in curve_calls] == [
        _PREFILL_ENGINE_DEPTH,
        _DECODE_ENGINE_DEPTH,
    ]
    assert all(call["engine_stage"] == "generator" for call in curve_calls)
    assert all(call["component_service_s"] == 1.0 for call in curve_calls)
    audit = result.loc[0, "analytical_engine_cycle_curve_cycles"]["generator"]
    assert audit["pd_serving"] == "disagg_pd"
    assert audit["prefill_resident_batch"] == _PREFILL_ENGINE_DEPTH
    assert audit["decode_resident_batch"] == _DECODE_ENGINE_DEPTH
    assert audit["mean_prefill_cycles_per_query"] == 4.0
    assert audit["mean_prefill_frontiers_per_query"] == 1.0
    assert audit["mean_decode_cycles_per_query"] == 2.0
    assert audit["prefill_overhead_s_per_frontier"] == pytest.approx(
        _CURVE_OVERHEAD_S
    )
    assert "prefill_overhead_s_per_cycle" not in audit
    assert "mean_cycles_per_query" not in audit


def test_output_tokens_and_agent_loop_change_cycles_and_analytical_qps(
    monkeypatch,
) -> None:
    curve_calls: list[dict] = []
    _install_cpu_doubles(monkeypatch, curve_calls)

    short = _run(
        pd_mode="collocated_pd",
        traces=[[
            _generator_call(input_tokens=128, output_tokens=2),
        ]],
        is_agentic=False,
    )
    loop = _run(
        pd_mode="collocated_pd",
        traces=[[
            _generator_call(input_tokens=128, output_tokens=2),
            _generator_call(input_tokens=256, output_tokens=2),
        ]],
        is_agentic=True,
    )
    long = _run(
        pd_mode="collocated_pd",
        traces=[[
            _generator_call(input_tokens=128, output_tokens=20),
        ]],
        is_agentic=False,
    )

    def cycles(frame: pd.DataFrame) -> float:
        return float(
            frame.loc[0, "analytical_engine_cycle_curve_cycles"]
            ["generator"]["mean_cycles_per_query"]
        )

    assert cycles(short) == 2.0
    assert cycles(loop) == 4.0
    assert cycles(long) == 20.0
    assert (
        short.loc[0, "dynamic_analytical_qps"]
        > loop.loc[0, "dynamic_analytical_qps"]
        > long.loc[0, "dynamic_analytical_qps"]
    )
