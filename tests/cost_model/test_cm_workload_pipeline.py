"""Contracts for shared trace preparation and explicit CM phase boundaries."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.cost_model import dynamic_pipeline
from rag_stack.cost_model.assembly import RAGCMAssembly
from rag_stack.cost_model.types import AssemblyResult
from rag_stack.cost_model.workload import prepare_trace_workload, summarize_trace
from rag_stack.rag_ir.trace import make_quality_trace_envelope
from rag_stack.rag_ir.workflow_schema import WorkflowSchema
from rag_stack.types import PerformanceSelection


def _calls(generation_pairs, query_lengths):
    calls = []
    for index, (input_tokens, output_tokens) in enumerate(generation_pairs):
        calls.append({"stage": "generator", "input_tokens": input_tokens,
                      "output_tokens": output_tokens, "step_idx": len(calls)})
        if index < len(query_lengths):
            for stage, inp, out in (
                ("semantic_retrieval_encode", query_lengths[index], 0),
                ("semantic_retrieval_vectorsearch", query_lengths[index], 1600),
                ("passage_reranker", 1600, 600),
            ):
                calls.append({"stage": stage, "input_tokens": inp,
                              "output_tokens": out, "step_idx": len(calls)})
    return calls


def _react_envelope():
    return make_quality_trace_envelope([
        _calls([(180, 16), (800, 14), (1400, 48)], [8, 7]),
        _calls([(200, 10), (800, 48)], [5]),
    ], question_ids=["q1", "q2"])


def test_react_summary_preserves_calls_and_separates_retrieval_axes():
    envelope = _react_envelope()
    original = deepcopy(envelope)
    workload = prepare_trace_workload(envelope, query_expansion_fanout=4)

    assert envelope == original
    assert workload.traces == [query["calls"] for query in envelope["queries"]]
    assert workload.source_question_ids == ["q1", "q2"]
    assert workload.summary["prefill_fanout"] == 2.5
    assert workload.summary["num_retrievals"] == 2.5  # Legacy generator PD key.
    assert workload.summary["retrieval_invocations_per_query"] == 1.5
    assert workload.summary["query_expansion_fanout"] == 4
    assert workload.summary["retrieval_backend_fanout"] == 6
    assert workload.summary["token_stats"] == {
        "avg_query_tokens": 6,
        "avg_retrieved_context_tokens": 1600,
        "avg_generator_input_tokens": 676,
        "avg_generator_output_tokens": 68,
        "avg_query_expansion_input_tokens": 0,
        "avg_query_expansion_output_tokens": 0,
    }


def test_sampling_keeps_question_identity_and_summary_on_the_same_cohort():
    envelope = make_quality_trace_envelope(
        [_calls([(10 * index, index)], []) for index in range(1, 8)],
        question_ids=[f"q{index}" for index in range(1, 8)],
    )
    workload = prepare_trace_workload(envelope, max_queries=3)
    assert workload.source_question_ids == ["q1", "q4", "q7"]
    assert [calls[0]["input_tokens"] for calls in workload.traces] == [10, 40, 70]
    assert workload.summary["n_user_queries"] == 3
    assert workload.summary["token_stats"]["avg_generator_input_tokens"] == 40
    assert workload.normalization.kept_queries == 7


def test_summary_is_order_independent_but_retained_traces_keep_order():
    traces = prepare_trace_workload(_react_envelope()).traces
    reordered = [list(reversed(calls)) for calls in reversed(traces)]
    assert summarize_trace(traces) == summarize_trace(reordered)
    assert traces != reordered


@pytest.mark.parametrize("kwargs", [{"max_queries": 0}, {"query_expansion_fanout": 0}])
def test_workload_rejects_invalid_preparation_limits(kwargs):
    with pytest.raises(ValueError, match="must be positive"):
        prepare_trace_workload(_react_envelope(), **kwargs)


def test_pipeline_shares_calibration_and_cohort_and_passes_the_exact_plan(monkeypatch):
    events = []
    workload = prepare_trace_workload(_react_envelope())
    config = {"system": {}}
    frozen = {"system": {"frozen_test_marker": True}}
    hardware = SimpleNamespace(performance=PerformanceSelection(selection="min_latency"))
    candidates = pd.DataFrame({"qps": [99999.0], "batch_size_request": [8]})
    analytical = pd.DataFrame({"qps": [20.0], "latency_s": [0.4]})
    final = pd.DataFrame({"qps": [12.0], "latency_s": [0.7], "simulator_status": ["ok"]})
    plan = object()

    def freeze(pipeline, hw):
        assert pipeline is config and hw is hardware
        events.append("freeze")
        return frozen

    def analyze(df, sweep_dfs, **kwargs):
        assert df is candidates
        assert kwargs["pipeline_config"] is frozen
        assert kwargs["traces"] is workload.execution_paths
        assert kwargs["prepared_paths"] is workload.prepared_paths
        assert kwargs["trace_source_question_ids"] is workload.source_question_ids
        events.append("analyze")
        return analytical

    def screen(df, **kwargs):
        assert df is analytical
        assert kwargs["selection"] is hardware.performance
        events.append("screen")
        return plan

    def simulate(df, **kwargs):
        assert df is analytical and kwargs["screening"] is plan
        assert kwargs["pipeline_config"] is frozen
        assert kwargs["traces"] is workload.execution_paths
        assert kwargs["prepared_paths"] is workload.prepared_paths
        assert kwargs["trace_source_question_ids"] is workload.source_question_ids
        assert "latency" not in kwargs  # No unused analytic service handoff.
        events.append("simulate")
        return final

    monkeypatch.setattr(dynamic_pipeline.closed_loop_sim, "_pipeline_with_frozen_stage_batch_selection", freeze)
    monkeypatch.setattr(dynamic_pipeline.hybrid_screening, "screen_des_candidates", screen)
    monkeypatch.setattr(dynamic_pipeline.closed_loop_sim, "apply_closed_loop_simulation", simulate)
    monkeypatch.setattr("rag_stack.cost_model.workload.compile_execution_paths",
                        lambda *a, **kw: pytest.fail("A prepared cohort must not be compiled again"))
    result = dynamic_pipeline.evaluate_candidates(
        SimpleNamespace(analyze=analyze), candidates, {}, pipeline_config=config,
        hardware=hardware, traces=workload.execution_paths,
        trace_source_question_ids=workload.source_question_ids,
        prepared_paths=workload.prepared_paths,
    )
    assert result is final
    assert events == ["freeze", "analyze", "screen", "simulate"]


@pytest.mark.parametrize("fixed", [False, True])
def test_fixed_and_swept_candidates_share_trace_pipeline_and_final_selection(monkeypatch, fixed):
    hardware = SimpleNamespace(performance=PerformanceSelection(selection="min_latency"))
    assembly = RAGCMAssembly(hardware)
    schema = WorkflowSchema("agentic", stages=["generator_prefill", "generator_decode"])
    config = {"system": {"batch_size_request": 8, "batch_size_decode": 16}}
    candidate = pd.DataFrame({"qps": [99999.0], "latency_s": [0.001]})
    candidate_result = AssemblyResult(0.0, {}, candidate, {})
    placement = object() if fixed else None
    visited = []

    def sweep(*args, **kwargs):
        assert not fixed
        assert kwargs["dag"]["prefill_fanout"] == 2.5
        visited.append("sweep")
        return candidate_result

    def construct_fixed(pipeline, stages, summary, actual_placement, request, decode, models):
        assert fixed and actual_placement is placement
        assert (request, decode) == (8, 16)
        assert summary["prefill_fanout"] == 2.5
        visited.append("fixed")
        return candidate_result

    def evaluate(**kwargs):
        assert kwargs["df"] is candidate
        assert kwargs["trace_source_question_ids"] == ["q1", "q2"]
        assert [len(calls) for calls in kwargs["traces"]] == [9, 5]
        assert kwargs["prepared_paths"].paths is kwargs["traces"]
        assert kwargs["pipeline_config"]["system"]["batch_size_request"] == 8
        assert kwargs["fixed_placement"] is placement
        visited.append("pipeline")
        return pd.DataFrame({
            "qps": [0.0, 10.0], "latency_s": [float("inf"), 0.8],
            "simulator_status": ["screened_out", "ok"],
        })

    monkeypatch.setattr(assembly, "_run_sweep", sweep)
    monkeypatch.setattr(assembly, "_run_fixed_dynamic_candidate", construct_fixed)
    monkeypatch.setattr(dynamic_pipeline, "evaluate_candidates", evaluate)
    result = assembly.evaluate_dynamic(config, workflow_schema=schema,
                                       trace=_react_envelope(), fixed_placement=placement)
    assert visited == ["fixed" if fixed else "sweep", "pipeline"]
    assert result.performance_score == -0.8
    assert result.best_hw_params["qps"] == 10.0
    assert result.rago_result_df["quality_trace_priced_queries"].tolist() == [2, 2]
    assert result.sweep_dfs == (None if fixed else {})
    assert config == {"system": {"batch_size_request": 8, "batch_size_decode": 16}}


def test_static_path_preserves_rago_metrics_without_entering_dynamic_pipeline(monkeypatch):
    assembly = RAGCMAssembly(SimpleNamespace(
        performance=PerformanceSelection(selection="max_throughput"),
    ))
    frame = pd.DataFrame({"qps": [7.0], "latency_s": [0.2]})
    monkeypatch.setattr(assembly, "_run_sweep", lambda *a, **kw: AssemblyResult(0.0, {}, frame, {}))

    def unexpected(**kwargs):
        pytest.fail("Static CM must not enter the trace pipeline")

    monkeypatch.setattr(dynamic_pipeline, "evaluate_candidates", unexpected)
    result = assembly.evaluate_static({}, workflow_schema=WorkflowSchema("sequential"))
    assert result.performance_score == 7.0
    assert result.best_hw_params["latency_s"] == 0.2
