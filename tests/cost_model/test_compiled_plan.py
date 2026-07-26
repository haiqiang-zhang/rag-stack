"""Real physical compiler contracts using deterministic CPU service functions."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.cost_model import closed_loop_sim, compiled, dynamic_pipeline, engines
from rag_stack.rag_ir import make_quality_trace_envelope
from rag_stack.types import SystemConfig


def _inputs(*, fusion=True, visits=1, identities=True):
    config = {
        "node_lines": [{"nodes": [
            {"stage": "query_expansion", "modules": [{"component": "multi_query_expansion"}]},
            {"stage": "semantic_retrieval", "top_k": 4,
             "modules": [{"component": "vectordb", "vectordb": "index"}]},
            {"stage": "generator", "modules": [{"component": "vllm", "model": "fixture"}]},
        ]}],
        "vectordb": [{"name": "index", "embedding_dim": 8}],
        "_token_stats": {"avg_chunk_tokens": 16, "avg_query_tokens": 8,
                         "avg_output_tokens": 2, "prompt_template_tokens": 8,
                         "bytes_per_token": 4.0},
        "system": {"rag_cm": {"calibration_mode": "uncalibrated_spec"},
                   "communication": {"include_non_pd_comm": False},
                   "batching": {"dynamic_timeout_s": 0.0},
                   "retrieval": {"faiss_num_threads": 1}},
    }
    calls = []
    for _ in range(visits):
        calls.extend([{"stage": "semantic_retrieval_encode"},
                      {"stage": "semantic_retrieval_vectorsearch"}])
        if fusion:
            calls.append({"stage": "retrieval_fusion", "model_id": "rrf_pure",
                          "input_profile": {"items": 16, "list_sizes": [4, 4, 4, 4]},
                          "output_profile": {"items": 4, "unique_items": 4}})
    calls.append({"stage": "generator"})
    for i, call in enumerate(calls):
        call.update(step_idx=i, input_tokens=8,
                    output_tokens=2 if call["stage"] == "generator" else 0)
        if identities:
            call.update(call_id=f"c{i}", node_id=f"n{i}")
    trace = make_quality_trace_envelope([calls], question_ids=["question-17"])
    hardware = SystemConfig(num_cores=2, system_hardware="unit_compiled_plan",
                            rag_cm={"calibration_mode": "uncalibrated_spec"})
    row = pd.Series({"batch_size_request": 2, "batch_size_decode": 2,
                     "num_retrieval_servers": 1,
                     "collocation_strategy": [["semantic_retrieval_encode"],
                                              ["generator_prefill", "generator_decode"]]})
    latency = engines.CallLatency(
        encode_s=lambda batch: .001, retrieve_s=lambda batch: .002,
        retrieve_kernel_s=lambda batch: .002,
        rrf_s=lambda batch: .0005 * batch,
        prefill_s=lambda tokens, batch, tp=1, pp=1: .0001,
        decode_s=lambda inp, out, batch, tp=1, pp=1: .0001,
    )
    return config, trace, hardware, row, latency


def _plan(**kwargs):
    config, trace, hardware, row, latency = _inputs(**kwargs)
    workload = compiled.compile_workload(config, trace)
    return compiled.bind_plan(workload, hardware=hardware, candidate_row=row,
                              prefix_caching=False, latency=latency), config, trace


def test_compiler_preserves_observed_visits_fusion_and_source_mapping():
    plan, config, trace = _plan(visits=2)
    description = compiled.describe(plan)
    ops = description["queries"][0]["operations"]
    assert description["queries"][0]["question_id"] == "question-17"
    assert [op["kind"] for op in ops] == ["worker", "worker", "prefill", "decode"]
    assert ops[0]["source_call_ids"] == ["c0", "c1", "c2"]
    assert ops[1]["source_call_indices"] == [3, 4, 5]
    assert ops[-2]["source_call_ids"] == ops[-1]["source_call_ids"] == ["c6"]
    assert ops[0]["details"]["operators"][-1]["name"] == "retrieval_fusion"
    assert ops[0]["details"]["operators"][-1]["resource"] == "cpu"
    assert plan.analysis_paths() == [trace["queries"][0]["calls"]]
    assert description["prediction"] is None
    assert description["resources"]["engines"][0]["role"] == "unified"
    assert description["deployment"] == config["system"]
    json.dumps(description, allow_nan=False)


def test_compile_and_describe_do_not_mutate_inputs_or_invent_legacy_ids():
    config, trace, hardware, row, latency = _inputs(fusion=False, identities=False)
    before = deepcopy((config, trace, row.to_dict()))
    plan = compiled.bind_plan(compiled.compile_workload(config, trace), hardware=hardware,
                              candidate_row=row, prefix_caching=False, latency=latency)
    assert (config, trace, row.to_dict()) == before
    description = plan.describe()
    assert description["queries"][0]["operations"][0]["source_call_ids"] == []
    assert description["queries"][0]["operations"][0]["source_call_indices"] == [0, 1]
    assert all(op["name"] != "retrieval_fusion" for op in description["queries"][0]["operations"])
    description["queries"].clear()
    assert plan.describe()["queries"]


def test_source_indices_use_raw_call_positions_with_gapped_duplicate_steps():
    config, trace, hardware, row, latency = _inputs()
    for call, step in zip(trace["queries"][0]["calls"], [4, 4, 7, 12]):
        call["step_idx"] = step
    plan = compiled.bind_plan(compiled.compile_workload(config, trace), hardware=hardware,
                              candidate_row=row, prefix_caching=False, latency=latency)
    ops = plan.describe()["queries"][0]["operations"]
    assert ops[0]["source_call_indices"] == [0, 1, 2]
    assert ops[-1]["source_call_indices"] == [3]
    for op in ops:
        assert [trace["queries"][0]["calls"][i]["call_id"]
                for i in op["source_call_indices"]] == op["source_call_ids"]


def test_changed_nonllm_batch_curve_blocks_retained_pricing_and_simulation(monkeypatch, tmp_path):
    from rag_stack.cost_model.retrieval_encode import RetrievalEncodeCostModel

    config, trace, hardware, row, fixture_latency = _inputs(fusion=False)
    config["node_lines"][0]["nodes"] = config["node_lines"][0]["nodes"][1:]
    hardware.gpu = "compiled_test_gpu"
    monkeypatch.setenv("RAG_STACK_CALIBRATION_PROFILE_DIR", str(tmp_path))
    path = tmp_path / "llm_sim" / "compiled_test_gpu.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"1": .001, "2": .004}))
    model = RetrievalEncodeCostModel()
    monkeypatch.setattr(model, "estimate", lambda cfg, hw: SimpleNamespace(
        latency_s=json.loads(path.read_text())[str(cfg["vectordb"][0]["batch_size"])],
    ))
    monkeypatch.setattr("rag_stack.cost_model.sweep_inputs.build_performance_stage_cost_models",
                        lambda cfg: {"semantic_retrieval_encode": model})

    def build():
        latency = engines.build_call_latency(config, hardware, bind_profiles=True)
        for field in ("retrieve_s", "retrieve_kernel_s", "prefill_s", "decode_s"):
            setattr(latency, field, getattr(fixture_latency, field))
        return compiled.bind_plan(compiled.compile_workload(config, trace), hardware=hardware,
                                  candidate_row=row, prefix_caching=False, latency=latency)

    plan = build()
    description = plan.describe()
    assert plan.latency.encode_s(1) == .001
    path.write_text(json.dumps({"1": .001, "2": .009}))
    with pytest.raises(compiled.CompilationBlocked) as changed:
        plan.latency.encode_s(2)
    assert changed.value.code == "pricing_binding_changed"
    with pytest.raises(compiled.CompilationBlocked):
        compiled.simulate_plan(plan)
    rebound = build()
    assert rebound.latency.encode_s(1) == .001
    assert rebound.latency.encode_s(2) == .009
    assert rebound.describe()["queries"] == description["queries"]
    assert rebound.describe()["pricing_binding"] != description["pricing_binding"]
    monkeypatch.setenv("RAG_STACK_CALIBRATION_PROFILE_DIR", str(tmp_path / "another_generation"))
    with pytest.raises(compiled.CompilationBlocked) as moved:
        rebound.latency.encode_s(1)
    assert moved.value.code == "pricing_binding_changed"


def test_bound_ivf_pricing_retains_corpus_identity_across_later_runs(monkeypatch, tmp_path):
    from rag_stack.cost_model.faiss_ivf_sim.adapter import FaissIVFCostModel
    from rag_stack.cost_model.pricing_binding import capture_pricing_bindings

    config, _, hardware, _, _ = _inputs(fusion=False)
    monkeypatch.setenv("RAG_STACK_CALIBRATION_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(FaissIVFCostModel, "_corpus_id", "preview_corpus")
    model = FaissIVFCostModel()
    binding = capture_pricing_bindings({"semantic_retrieval_vectorsearch": model}, config, hardware)
    monkeypatch.setattr(FaissIVFCostModel, "_corpus_id", "later_run_corpus")
    assert model._corpus_id == "preview_corpus"
    assert any("preview_corpus" in key for key in binding.documents)


def test_bm25_is_structured_block_not_zero_price():
    config, trace, *_ = _inputs()
    config["node_lines"][0]["nodes"].insert(0, {
        "stage": "lexical_retrieval", "modules": [{"component": "bm25"}],
    })
    with pytest.raises(compiled.CompilationBlocked) as caught:
        compiled.compile_workload(config, trace)
    assert caught.value.to_dict()["status"] == "blocked"
    assert caught.value.code == "unsupported_component"


def test_unlowered_observed_call_is_not_silently_omitted():
    config, trace, hardware, row, latency = _inputs(fusion=False)
    calls = trace["queries"][0]["calls"]
    calls.insert(2, {"stage": "passage_reranker", "input_tokens": 8,
                     "output_tokens": 4, "step_idx": 2, "call_id": "rerank"})
    calls[-1]["step_idx"] = 3
    with pytest.raises(compiled.CompilationBlocked, match="every source invocation"):
        compiled.bind_plan(compiled.compile_workload(config, trace), hardware=hardware,
                           candidate_row=row, prefix_caching=False, latency=latency)


def test_simulation_consumes_retained_graph_without_recompilation(monkeypatch):
    plan, _, _ = _plan()
    monkeypatch.setattr(closed_loop_sim, "build_sim_inputs",
                        lambda *a, **kw: pytest.fail("retained graph was recompiled"))
    result = compiled.simulate_plan(plan)
    assert result.qps > 0 and result.latency_mean_s > 0
    assert result.converged


def test_fixed_pipeline_passes_same_bound_plan_to_analysis_and_des(monkeypatch):
    plan, config, _ = _plan()
    frames = pd.DataFrame([plan.candidate_row])
    sentinel = object()
    calls = []
    monkeypatch.setattr(compiled, "bind_plan", lambda *a, **kw: plan)

    def analyze(frame, sweeps, **kwargs):
        assert kwargs["compiled_plan"] is plan
        calls.append("analyze")
        return frame

    def simulate(frame, **kwargs):
        assert kwargs["builder_fn"](None, None, None, None, candidate_row=frame.iloc[0])[0] is plan.sim_inputs
        assert kwargs["screening"] is sentinel
        calls.append("simulate")
        return frame

    monkeypatch.setattr(dynamic_pipeline.hybrid_screening, "screen_des_candidates", lambda *a, **kw: sentinel)
    monkeypatch.setattr(closed_loop_sim, "apply_closed_loop_simulation", simulate)
    result = dynamic_pipeline.evaluate_candidates(
        SimpleNamespace(analyze=analyze), frames, {}, pipeline_config=config,
        hardware=plan.hardware, traces=plan.workload.execution_paths,
        trace_source_question_ids=plan.workload.source_question_ids,
        fixed_placement=object(),
    )
    assert result is frames and calls == ["analyze", "simulate"]


def test_disaggregated_plan_retains_real_kv_and_host_transfers(monkeypatch):
    config, trace, hardware, row, latency = _inputs(fusion=False)
    config["system"]["communication"]["include_non_pd_comm"] = True
    hardware.interconnect = {"gpu_gpu": {"bandwidth_gbps": 100, "latency_us": 1},
                             "gpu_cpu": {"bandwidth_gbps": 20, "latency_us": 2}}
    config["system"]["interconnect"] = hardware.interconnect
    row["collocation_strategy"] = [["semantic_retrieval_encode"],
                                   ["generator_prefill"], ["generator_decode"]]
    monkeypatch.setattr("rag_stack.cost_model.communication._genz_model_for", lambda *a: 16)
    plan = compiled.bind_plan(compiled.compile_workload(config, trace), hardware=hardware,
                              candidate_row=row, prefix_caching=False, latency=latency)
    description = plan.describe()
    assert [e["role"] for e in description["resources"]["engines"]] == ["prefill", "decode"]
    transfers = [op for op in description["queries"][0]["operations"] if op["kind"] == "transfer"]
    host = next(op for op in transfers if op["resource_ids"] == ["link:gpu_cpu"])
    assert host["source_call_ids"] == ["c1", "c2"]
    kv = next(op for op in transfers if op["resource_ids"] == ["link:gpu_gpu"])
    assert kv["source_call_ids"] == ["c2"]
    assert kv["details"]["payload_bytes"] == 8 * 16
    assert kv["details"]["selection"] == "replica_pair"


def test_bound_plan_rejects_changed_batch_and_preserves_hardware_snapshot():
    config, trace, hardware, row, latency = _inputs()
    plan = compiled.bind_plan(compiled.compile_workload(config, trace), hardware=hardware,
                              candidate_row=row, prefix_caching=False, latency=latency)
    hardware.num_cores = 999
    assert plan.hardware.num_cores == 2
    row["batch_size_decode"] = 16
    with pytest.raises(compiled.CompilationBlocked, match="deployment no longer matches"):
        plan.validate_candidate(row)


def test_assembly_compiles_once_and_reuses_real_plan_for_analysis_and_simulation(monkeypatch):
    from rag_stack.cost_model.assembly import RAGCMAssembly
    from rag_stack.cost_model.types import AssemblyResult

    config, _, hardware, _, latency = _inputs(fusion=False)
    config["node_lines"][0]["nodes"] = config["node_lines"][0]["nodes"][-1:]
    config.pop("vectordb")
    stages = ["generator_prefill", "generator_decode"]
    role = {"devices": ["cuda:0"], "tp": 1, "pp": 1,
            "replicas": 1, "replica_devices": [["cuda:0"]]}
    layout = {"available_devices": ["cuda:0"],
              "resource_groups": [{"id": "gpu:0", "kind": "gpu", "devices": ["cuda:0"],
                                   "performance_stages": stages}],
              "performance_stages": {stage: {**role, "kind": "gpu", "resource": "gpu:0",
                                             "num_chips": 1} for stage in stages},
              "engines": {"generator": {**role, "pd_serving": "collocated_pd"}}}
    system = {**config["system"], "layout": layout,
              "batch_size_request": 2, "batch_size_decode": 2}
    trace = make_quality_trace_envelope([[{
        "stage": "generator", "step_idx": 0, "input_tokens": 8,
        "output_tokens": 2, "call_id": "answer", "node_id": "generator",
    }]], question_ids=["source-42"])
    row = {"batch_size_request": 2, "batch_size_decode": 2,
           "num_chips_generator_prefill": 1, "num_chips_generator_decode": 1,
           "collocation_strategy": [stages], "candidate_system_layout": layout}
    assembly = RAGCMAssembly(hardware)
    monkeypatch.setattr(assembly, "_run_fixed_dynamic_candidate", lambda *a, **kw:
                        AssemblyResult(0.0, {}, pd.DataFrame([row])))
    monkeypatch.setattr(engines, "build_call_latency", lambda *a, **kw: latency)
    before = deepcopy((config, system, trace))
    plan = assembly.compile_fixed(config, system, trace=trace)
    assert (config, system, trace) == before
    assert [op["kind"] for op in plan.describe()["queries"][0]["operations"]] == ["prefill", "decode"]
    monkeypatch.setattr(engines, "build_call_latency", lambda *a, **kw:
                        pytest.fail("analysis must retain the plan's pricing functions"))
    monkeypatch.setattr(closed_loop_sim, "build_sim_inputs", lambda *a, **kw:
                        pytest.fail("analysis and simulation must retain the plan's physical graph"))
    analytical = assembly.analyze_plan(plan)
    assert len(analytical) == 1 and analytical.iloc[0]["qps"] > 0
    simulation = assembly.simulate_plan(plan)
    assert simulation.qps > 0 and simulation.latency_mean_s > 0
