"""DES borrows one validated cohort across deployment and process boundaries."""

from concurrent.futures import Future
from copy import deepcopy
from multiprocessing.reduction import ForkingPickler
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.cost_model import closed_loop_sim as sim, dynamic_pipeline, engines, workload
from rag_stack.types import SystemConfig


def _inputs():
    config = {"system": {
        "batching": {"dynamic_timeout_s": 0.0},
        "communication": {"include_non_pd_comm": False},
    }}
    traces = [[{
        "stage": "generator", "input_tokens": 16, "output_tokens": 2,
        "step_idx": 0, "call_id": "answer",
        "input_profile": {"items": 3, "list_sizes": [1, 2]},
    }]]
    groups = [["generator_prefill", "generator_decode"]]
    return config, SystemConfig(num_cores=2), traces, groups


def _latency():
    return engines.CallLatency(
        prefill_s=lambda tokens, batch, tp=1, pp=1: 1e-4,
        decode_s=lambda inp, out, batch, tp=1, pp=1: 1e-4,
    )


def _rows(config, groups):
    rows = [dict(batch_size_request=batch, batch_size_decode=batch,
                 collocation_strategy=groups) for batch in (2, 4)]
    return [sim._CandidateWork(index, row, sim.simulation_signature(row, config))
            for index, row in enumerate(rows)]


def test_standalone_builder_still_rejects_malformed_unprepared_calls():
    config, hardware, traces, groups = _inputs()
    traces[0][0]["input_profile"]["items"] = -1
    with pytest.raises(ValueError, match="non-negative integer"):
        sim.build_sim_inputs(config, hardware, traces, groups,
                             batch_size_request=2, stage_batch_profiles={}, latency=_latency())


@pytest.mark.parametrize("prepared", [False, True])
def test_multiple_candidates_reuse_validation_without_mutating_calls(monkeypatch, prepared):
    config, hardware, raw_traces, groups = _inputs()
    original = deepcopy(raw_traces)
    carrier = workload.PreparedExecutionPaths(raw_traces) if prepared else None
    traces = carrier.paths if carrier is not None else raw_traces
    compile_calls = []
    compile_paths = workload.compile_execution_paths
    def count_compile(*args, **kwargs):
        compile_calls.append(args[0])
        return compile_paths(*args, **kwargs)
    monkeypatch.setattr(workload, "compile_execution_paths", count_compile)
    borrowed = []
    resolve = workload.resolve_execution_paths
    def record_borrow(*args, **kwargs):
        result = resolve(*args, **kwargs)
        borrowed.append(result)
        return result
    monkeypatch.setattr(workload, "resolve_execution_paths", record_borrow)
    monkeypatch.setattr(sim, "_stage_batch_profiles_for_pipeline", lambda *_: {})
    # A prepared logical cohort must not enable fixed-plan profile binding.
    def build_latency(_config, _hardware):
        return _latency()
    monkeypatch.setattr(engines, "build_call_latency", build_latency)
    lowered = []
    def saturation(inputs, **kwargs):
        lowered.append(inputs)
        return SimpleNamespace(result="priced")
    monkeypatch.setattr(sim, "_simulate_to_saturation", saturation)

    results = sim._evaluate_candidate_rows(
        _rows(config, groups), pipeline_config=config, hardware=hardware,
        traces=traces, prepared_paths=carrier, trace_source_question_ids=["q1"],
        prefix_caching=False,
    )

    assert [result[1] for result in results] == ["priced", "priced"], results
    assert len(compile_calls) == (0 if prepared else 1)
    assert all(item is borrowed[0] for item in borrowed)
    assert len(lowered) == 2 and lowered[0] is not lowered[1]
    assert {inputs[1][0].depth for inputs in lowered} == {2, 4}
    assert raw_traces == traces == original
    assert borrowed[0].paths == original
    assert all(inputs[2][0][0].call.source_calls[0] == original[0][0]
               for inputs in lowered)


def test_prepared_paths_and_fixed_workload_keep_separate_pricing_semantics(monkeypatch):
    config, hardware, traces, groups = _inputs()
    carrier = workload.PreparedExecutionPaths(traces)
    flags = []
    def build_latency(_config, _hardware, *, bind_profiles=False):
        flags.append(bind_profiles)
        return _latency()
    monkeypatch.setattr(engines, "build_call_latency", build_latency)
    for compiled in (None, SimpleNamespace(prepared_paths=carrier)):
        inputs, reason = sim.build_sim_inputs(
            config, hardware, carrier.paths, groups, batch_size_request=2,
            stage_batch_profiles={}, prepared_paths=carrier,
            compiled_workload=compiled, prefix_caching=False,
        )
        assert inputs is not None and reason is None
    assert flags == [False, True]

    conflicting = workload.PreparedExecutionPaths(traces)
    with pytest.raises(ValueError, match="conflicts with compiled_workload"):
        sim.build_sim_inputs(
            config, hardware, carrier.paths, groups, batch_size_request=2,
            prepared_paths=conflicting,
            compiled_workload=SimpleNamespace(prepared_paths=carrier),
        )


def test_worker_task_pickle_preserves_prepared_paths_identity(monkeypatch):
    config, hardware, traces, groups = _inputs()
    carrier = workload.PreparedExecutionPaths(traces)
    seen = []
    def price_rows(rows, *, traces, prepared_paths, **kwargs):
        assert workload.resolve_execution_paths(traces, prepared_paths) is prepared_paths
        seen.append(prepared_paths)
        return [(work.ordinal, "priced", None, "") for work in rows]
    monkeypatch.setattr(sim, "_evaluate_candidate_rows", price_rows)
    monkeypatch.setattr(workload, "compile_execution_paths",
                        lambda *args, **kwargs: pytest.fail("Worker must borrow the validated snapshot"))

    class TransportPool:
        def submit(self, function, task):
            # The spawn executor uses this exact object-graph pickle protocol.
            received_task = pickle.loads(ForkingPickler.dumps(task))
            future = Future()
            future.set_result(function(received_task))
            return future
    monkeypatch.setattr(sim, "_des_process_pool", lambda workers: TransportPool())
    results = sim._evaluate_candidates_in_processes(
        _rows(config, groups), workers=2, pipeline_config=config, hardware=hardware,
        traces=carrier.paths, prepared_paths=carrier,
        trace_source_question_ids=["q1"], prefix_caching=False,
    )
    assert [result[:2] for result in results] == [(0, "priced"), (1, "priced")]
    assert seen and all(item.paths == carrier.paths for item in seen)
    assert all(item is not carrier for item in seen)


def test_custom_builder_keeps_its_existing_unprepared_interface(monkeypatch):
    config, hardware, _, groups = _inputs()
    raw_traces = [[{"custom": "downstream builder owns this input"}]]
    seen = []
    def builder(_config, _hardware, traces, _groups, *, batch_size_request,
                batch_size_decode, prefix_caching, candidate_row, latency, latency_cache):
        seen.append(traces)
        return ([], [], [], 0, 1, None), None
    monkeypatch.setattr(sim, "_simulate_to_saturation",
                        lambda *args, **kwargs: SimpleNamespace(result="priced"))
    results = sim._evaluate_candidate_rows(
        _rows(config, groups), pipeline_config=config, hardware=hardware,
        traces=raw_traces, prefix_caching=False, builder_fn=builder,
    )
    assert [result[1] for result in results] == ["priced", "priced"], results
    assert seen == [raw_traces, raw_traces]
    assert all(item is raw_traces for item in seen)


@pytest.mark.parametrize("entry", ["rows", "apply"])
@pytest.mark.parametrize("bad_type", [False, True])
def test_custom_builder_cannot_silently_ignore_an_invalid_prepared_contract(entry, bad_type):
    config, hardware, traces, groups = _inputs()
    carrier = SimpleNamespace(paths=traces) if bad_type else workload.PreparedExecutionPaths(traces)
    kwargs = dict(pipeline_config=config, hardware=hardware, traces=traces,
                  prepared_paths=carrier, prefix_caching=False,
                  builder_fn=lambda *args, **kwargs: pytest.fail("Invalid cohort must be rejected"))
    with pytest.raises(TypeError if bad_type else ValueError, match="prepared_paths"):
        if entry == "rows":
            sim._evaluate_candidate_rows(_rows(config, groups), **kwargs)
        else:
            sim.apply_closed_loop_simulation(pd.DataFrame({"batch_size_request": [2]}), **kwargs)


def test_ordinary_pipeline_prepares_once_across_real_analysis_and_physical_builds(monkeypatch, tmp_path):
    config, hardware, traces, groups = _inputs()
    original = deepcopy(traces)
    config["system"]["rag_cm"] = {"calibration_mode": "uncalibrated_spec"}
    hardware.rag_cm = {"calibration_mode": "uncalibrated_spec"}
    hardware.system_hardware = "unit_des_prepared_pipeline"
    monkeypatch.setenv("RAG_STACK_CALIBRATION_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(sim, "_des_worker_count", lambda count: 1)
    monkeypatch.setattr(engines, "_PREFIX_CACHING", False)
    monkeypatch.setattr(engines, "build_call_latency", lambda config, hardware: _latency())
    compile_calls = []
    compile_paths = workload.compile_execution_paths
    def count_compile(*args, **kwargs):
        compile_calls.append(args[0])
        return compile_paths(*args, **kwargs)
    monkeypatch.setattr(workload, "compile_execution_paths", count_compile)
    borrowed = []
    resolve = workload.resolve_execution_paths
    def record_borrow(*args, **kwargs):
        result = resolve(*args, **kwargs)
        borrowed.append(result)
        return result
    monkeypatch.setattr(workload, "resolve_execution_paths", record_borrow)
    lowered = []
    def saturation(inputs, **kwargs):
        lowered.append(inputs)
        population = inputs[4]
        result = sim.SimResult(
            qps=100.0 + inputs[1][0].depth, latency_mean_s=0.05,
            engine_fill_mean={}, stage_batch_mean={}, window_s=1.0,
            completions=10, population=population,
        )
        return sim._SaturationOutcome(
            result=result, populations_tried=(population,), final_population=population,
            plateau_reached=True, stop_reason="fixture_simulation",
        )
    monkeypatch.setattr(sim, "_simulate_to_saturation", saturation)
    frame = pd.DataFrame([work.row for work in _rows(config, groups)])
    engine = engines.DynamicEngine(SimpleNamespace(is_agentic=False, dataflow_type="sequential"))

    result = dynamic_pipeline.evaluate_candidates(
        engine, frame, {}, pipeline_config=config, hardware=hardware,
        traces=traces, trace_source_question_ids=["q1"],
    )

    assert len(compile_calls) == 1
    assert len(lowered) == 2 and lowered[0] is not lowered[1]
    assert borrowed and all(item is borrowed[0] for item in borrowed)
    assert borrowed[0].paths == traces == original
    assert result["simulator_status"].tolist() == ["ok", "ok"]
    assert result["qps"].tolist() == [102.0, 104.0]
    assert (result["dynamic_analytical_qps"] > 0).all()
