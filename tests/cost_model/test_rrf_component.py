"""Observed RRF input/output profiles drive dispatch and service demand."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from rag_stack.cost_model import rrf, sweep_inputs
from rag_stack.cost_model.dispatch import cost_model_for, performance_stages_for
from rag_stack.cost_model.engines import build_call_latency, _compute_chain_latency
from rag_stack.cost_model.retrieval_worker_profile import RRFFusionProfile, RetrievalWorkerProfileError
from rag_stack.cost_model.types import CostModelResult
from rag_stack.rag_ir.execution_path import compile_execution_paths
from rag_stack.rag_ir.normalize import dispatch_manifest, with_observed_trace_components
from rag_stack.taxonomy import Stage


def _config():
    return {
        "node_lines": [{"nodes": [
            {"stage": "query_expansion", "modules": [{"component": "multi_query_expansion"}]},
            {"stage": "semantic_retrieval", "top_k": 5,
             "modules": [{"component": "vectordb", "vectordb": "dense"}]},
            {"stage": "generator", "modules": [{"component": "vllm"}]},
        ]}],
        "vectordb": [{"name": "dense", "db_type": "faiss_ivf"}],
        "system": {"batch_size_request": 1},
    }


def _fusion(sizes=(5, 4), *, output=3, union=None):
    incoming = {"items": sum(sizes), "list_sizes": list(sizes)}
    if union is not None:
        incoming["unique_items"] = union
    return {
        "stage": "retrieval_fusion", "model_id": "rrf_pure",
        "input_tokens": 0, "output_tokens": 0,
        "input_profile": incoming,
        "output_profile": {"items": output, "unique_items": output},
    }


def _trace(*fusions):
    calls = []
    for fusion in fusions:
        calls.extend([
            {"stage": "semantic_retrieval_encode", "input_tokens": 12, "output_tokens": 0},
            {"stage": "semantic_retrieval_vectorsearch", "input_tokens": 12, "output_tokens": 0},
        ])
        if fusion is not None:
            calls.append(deepcopy(fusion))
    calls.append({"stage": "generator", "input_tokens": 12, "output_tokens": 1})
    return [[{**call, "step_idx": index} for index, call in enumerate(calls)]]


def test_fanout_does_not_bind_an_unobserved_fusion_model():
    config = _config()
    assert Stage.RETRIEVAL_FUSION not in [entry.stage for entry in dispatch_manifest(config)]
    bound = with_observed_trace_components(config, _trace(None)[0:1])
    assert Stage.RETRIEVAL_FUSION not in [entry.stage for entry in dispatch_manifest(bound)]


def test_observed_fusion_binds_one_model_with_cpu_resource():
    config = with_observed_trace_components(_config(), _trace(_fusion(), _fusion((3, 2))))
    entries = [entry for entry in dispatch_manifest(config) if entry.stage is Stage.RETRIEVAL_FUSION]
    assert len(entries) == 1
    assert entries[0].component == "rrf"
    assert cost_model_for(entries[0].stage, entries[0].component) == "rrf"
    assert performance_stages_for(entries[0].stage, entries[0].component)[0].resource == "cpu_retrieval"


def test_work_profile_uses_observed_lists_overlap_and_output():
    call = _fusion((5, 4), union=6, output=3)
    call["input_profile"]["list_unique_items"] = [4, 4]
    profile = rrf.build_fusion_work_profile(call)
    assert profile.metadata["input_items"] == 9
    assert profile.metadata["union_items"] == 6
    assert profile.outputs["candidates"].items == 3
    assert profile.metadata["implementation_path"] == "multi_list_pandas"


def test_missing_profiles_or_wrong_implementation_fail_without_config_inference():
    for call in ({"stage": "retrieval_fusion", "model_id": "rrf_pure"},
                 {**_fusion(), "model_id": "another_rrf"}):
        with pytest.raises(ValueError):
            rrf.build_fusion_work_profile(call)
    with pytest.raises(ValueError, match="observed evaluator"):
        rrf.RRFCostModel().estimate(_config(), SimpleNamespace())


def test_observed_union_bounds_missing_per_list_distinct_counts():
    profile = rrf.build_fusion_work_profile(_fusion((3, 2), union=2, output=2))
    assert profile.metadata["union_items"] == 2
    assert profile.outputs["candidates"].items == 2
    assert any("observed-union upper bound" in assumption
               for assumption in profile.metadata["assumptions"])


def test_empty_observed_output_and_lists_remain_zero_work():
    profile = rrf.build_fusion_work_profile(_fusion((0,), union=0, output=0))
    assert profile.metadata["input_lists"] == 1
    assert profile.outputs["candidates"].items == 0
    assert all(phase.total_bytes == 0 for phase in profile.phases)


def test_observed_output_cannot_exceed_input_union():
    with pytest.raises(ValueError, match="union cardinality"):
        rrf.build_fusion_work_profile(_fusion((4, 4), output=6, union=5))


def test_rrf_table_averages_observed_calls_and_accounts_for_missing_visits(monkeypatch):
    from rag_stack.cost_model import retrieval_worker_profile
    seen = []
    def service(items):
        seen.append(items)
        return 0.001 + items * 0.00001
    monkeypatch.setattr(retrieval_worker_profile, "resolve_rrf_fusion_profile",
                        lambda hw, cfg: SimpleNamespace(service_s=service))
    config = with_observed_trace_components(_config(), _trace(_fusion((5, 4)), None, _fusion((3, 2))))
    config["system"]["batch_size_request"] = 3
    result = rrf.RRFCostModel().estimate(config, SimpleNamespace())
    assert seen == [9, 5]
    assert result.latency_s == pytest.approx(0.00109 + 0.00105)


def test_single_list_never_uses_the_multilist_calibration(monkeypatch):
    from rag_stack.cost_model import retrieval_worker_profile
    def forbidden(*args):
        raise AssertionError("multi-list profile used for the single-list implementation")
    monkeypatch.setattr(retrieval_worker_profile, "resolve_rrf_fusion_profile", forbidden)
    call = _fusion((5,), output=3)
    with pytest.raises(RetrievalWorkerProfileError, match="single-list.*Python fast path"):
        rrf.fusion_service_for_call(call, _config(), SimpleNamespace())
    config = _config()
    config["system"]["rag_cm"] = {"calibration_mode": "uncalibrated_spec"}
    hardware = SimpleNamespace(num_cores=4, mem_bandwidth_1t=1e9, peak_int_ops=4e9)
    assert rrf.fusion_service_for_call(call, config, hardware) > 0
    assert rrf.build_fusion_work_profile(call).metadata["implementation_path"] == "single_list_python"


class _SearchModel:
    def __init__(self, per_query):
        self.per_query = per_query

    def estimate(self, config, hardware):
        node = next(n for n in config["node_lines"][0]["nodes"]
                    if n["stage"] == "semantic_retrieval")
        return CostModelResult(latency_s=self.per_query * node["modules"][0]["batch_size"])


def _latency(monkeypatch, config):
    monkeypatch.setattr(sweep_inputs, "build_performance_stage_cost_models", lambda cfg: {
        "semantic_retrieval_encode": _SearchModel(0.001),
        "semantic_retrieval_vectorsearch": _SearchModel(0.010),
        "retrieval_fusion": rrf.RRFCostModel(),
    })
    from rag_stack.cost_model import retrieval_worker_profile
    monkeypatch.setattr(retrieval_worker_profile, "resolve_rrf_fusion_profile",
                        lambda hardware, cfg: RRFFusionProfile(0.001, 0.00001))
    return build_call_latency(config, SimpleNamespace())


def test_search_is_separate_and_fusion_service_uses_each_call_profile(monkeypatch):
    config = _config()
    original = deepcopy(config)
    latency = _latency(monkeypatch, config)
    assert latency.retrieval_fanout == 4
    assert latency.retrieve_s(3) == pytest.approx(0.030)
    assert latency.retrieve_fused_overhead_s(3) == 0.0
    assert latency.rrf_s is None
    assert latency.rrf_call_s(_fusion((5, 4)), 3) == pytest.approx(3 * 0.00109)
    assert latency.rrf_call_s(_fusion((3, 2)), 3) == pytest.approx(3 * 0.00105)
    with pytest.raises(ValueError, match="input_profile"):
        latency.rrf_call_s({"stage": "retrieval_fusion", "model_id": "rrf_pure"}, 3)
    assert config == original


def test_analytic_path_prices_heterogeneous_observed_fusion_once(monkeypatch):
    latency = _latency(monkeypatch, _config())
    traces = _trace(_fusion((5, 4)), _fusion((3, 2)))
    original = deepcopy(traces)
    result = _compute_chain_latency(traces, latency, batch=2)
    assert result.e2e == pytest.approx(2 * 4 * 2 * 0.011 + 2 * (0.00109 + 0.00105))
    assert result.svc_fusion == pytest.approx(2 * (0.00109 + 0.00105))
    assert result.retrieval_calls_per_query == 2
    assert _compute_chain_latency(compile_execution_paths(traces, retrieval_fanout=4), latency, batch=2) == result
    assert traces == original
    without_fusion = _compute_chain_latency(_trace(None, None), latency, batch=2)
    assert without_fusion.svc_fusion == 0.0


def test_explicit_fusion_cannot_silently_disappear_without_a_service():
    from rag_stack.cost_model.engines import CallLatency
    with pytest.raises(ValueError, match="requires an RRF service"):
        _compute_chain_latency(_trace(_fusion()), CallLatency())


def test_preparation_keeps_observed_profiles_and_same_sampled_cohort():
    from rag_stack.cost_model.workload import prepare_trace_workload
    from rag_stack.rag_ir.trace import make_quality_trace_envelope

    traces = [_trace(_fusion())[0], _trace(None)[0], _trace(_fusion((3, 2)), None)[0]]
    envelope = make_quality_trace_envelope(traces, question_ids=["q0", "q1", "q2"])
    original = deepcopy(envelope)
    workload = prepare_trace_workload(envelope, query_expansion_fanout=4, max_queries=2)
    assert envelope == original
    assert workload.source_question_ids == ["q0", "q2"]
    assert workload.traces == workload.execution_paths == [traces[0], traces[2]]
    assert [call["input_profile"]["items"] for call in workload.summary["retrieval_fusion_calls"]] == [9, 5]
    assert workload.summary["fusion_invocations_per_query"] == 1.0
    assert workload.summary["retrieval_invocations_per_query"] == 1.5
    assert workload.summary["retrieval_backend_fanout"] == 6


def test_rago_restore_keeps_execution_order_distinct_from_resource_groups():
    import pandas as pd
    from rag_stack.cost_model.assembly import _restore_fusion_stage

    order = ["semantic_retrieval_encode", "semantic_retrieval_vectorsearch",
             "generator_prefill", "generator_decode"]
    original = pd.DataFrame([{
        "collocation_strategy": [order.copy()], "batch_size_request": 4,
        "pipeline_stage_order": order.copy(),
    }])
    source = original.copy(deep=True)
    expected_order = order[:2] + ["retrieval_fusion"] + order[2:]
    restored = _restore_fusion_stage(original, expected_order)
    assert restored.iloc[0]["pipeline_stage_order"] == expected_order
    assert restored.iloc[0]["collocation_strategy"] == [order, ["retrieval_fusion"]]
    assert restored.iloc[0]["batch_size_retrieval_fusion"] == 4
    pd.testing.assert_frame_equal(original, source)
    pd.testing.assert_frame_equal(_restore_fusion_stage(restored, expected_order), restored)


def test_fixed_fusion_service_does_not_scale_with_dense_retrieval_servers():
    import pandas as pd
    from rag_stack.cost_model.assembly import (
        _filter_stage_sweep_df_for_fixed, _fixed_stage_resource,
    )

    placement = SimpleNamespace(num_retrieval_servers=4, num_chips={})
    assert _fixed_stage_resource("retrieval_fusion", placement) == 1
    assert _fixed_stage_resource("semantic_retrieval_vectorsearch", placement) == 4
    rows = pd.DataFrame({"hardware.num_servers": [1, 4], "batch_size": [8, 8],
                         "time_ms": [20.0, 5.0]})
    kept = _filter_stage_sweep_df_for_fixed(
        rows, stage="retrieval_fusion", fixed_placement=placement, batch_size=8,
    )
    assert kept["time_ms"].tolist() == [20.0]
