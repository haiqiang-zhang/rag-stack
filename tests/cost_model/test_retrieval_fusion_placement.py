"""Fusion remains host work through candidate layout and communication."""

import pandas as pd
import pytest

from rag_stack.cost_model.candidate_layout import (
    candidate_system_layout_from_row,
    materialize_candidate_system_layouts,
)
from rag_stack.cost_model.candidate_reprice import (
    reprice_candidate_genz_services,
    require_candidate_genz_parallelism,
)
from rag_stack.cost_model.communication import apply_communication_costs
from rag_stack.types import SystemConfig


def _resolved_pipeline(expansion):
    return {"node_lines": [{"nodes": [
        {"stage": "query_expansion", "modules": [{"component": expansion}]},
        {
            "stage": "semantic_retrieval", "top_k": 4,
            "modules": [{"component": "vectordb", "vectordb": "vdb"}],
        },
        {"stage": "generator", "modules": [{"component": "vllm"}]},
    ]}]}


def _host_candidate():
    return materialize_candidate_system_layouts(
        pd.DataFrame([{
            "collocation_strategy": [
                ["semantic_retrieval_vectorsearch"], ["retrieval_fusion"],
            ],
            "pipeline_stage_order": [
                "semantic_retrieval_vectorsearch", "retrieval_fusion",
            ],
            "stage_devices": {"retrieval_fusion": ["cuda:0"]},
            "available_devices": ["cuda:0"],
            "num_chips_e2e": 0,
            "num_retrieval_servers": 4,
            "batch_size_retrieval_fusion": 8,
            "qps_semantic_retrieval_vectorsearch": 100.0,
            "qps_retrieval_fusion": 200.0,
            "qps": 100.0,
            "latency_s": 0.01,
        }]),
        pipeline_config={"system": {}},
    )


def test_fusion_layout_has_no_gpu_or_retrieval_replication():
    out = _host_candidate()
    layout = candidate_system_layout_from_row(out.iloc[0])
    fusion = layout["performance_stages"]["retrieval_fusion"]
    assert fusion["kind"] == "cpu_retrieval"
    assert fusion["devices"] == []
    assert fusion["num_chips"] == 0
    assert not {"tp", "pp", "num_servers", "replicas"} & fusion.keys()
    assert "retrieval_fusion" not in layout["engines"]
    assert layout["gpu_occupants"] == {}
    assert layout["performance_stages"]["semantic_retrieval_vectorsearch"]["num_servers"] == 4
    resource = next(
        group for group in layout["resource_groups"]
        if "retrieval_fusion" in group["performance_stages"]
    )
    assert resource["devices"] == []
    assert "num_servers" not in resource


def test_fusion_never_reaches_gpu_parallelism_or_exact_device_pricing():
    class CPUModel:
        def parallelism_model(self, *args, **kwargs):
            raise AssertionError("CPU fusion has no GPU parallelism model")

        def estimate_stage_for_devices(self, *args, **kwargs):
            raise AssertionError("CPU fusion cannot be priced on GPU devices")

    models = {"retrieval_fusion": CPUModel()}
    frame = _host_candidate()
    require_candidate_genz_parallelism(
        frame, pipeline_config={}, stage_cost_models=models,
    )
    out = reprice_candidate_genz_services(
        frame,
        pipeline_config={},
        hardware=SystemConfig(num_cores=2),
        stage_cost_models=models,
    )
    assert not out.iloc[0]["candidate_genz_repriced"]
    assert out.iloc[0]["candidate_genz_latency_delta_s"] == 0


def test_fusion_transfer_uses_fused_passages_without_cpu_to_cpu_charge():
    frame = pd.DataFrame([{
        "collocation_strategy": [
            ["semantic_retrieval_vectorsearch"], ["retrieval_fusion"],
            ["passage_reranker"],
        ],
        "pipeline_stage_order": [
            "semantic_retrieval_vectorsearch", "retrieval_fusion", "passage_reranker",
        ],
        "latency_s": 0.0,
        "qps": 1e12,
        "qps_per_chip": 1e12,
    }])
    config = {
        "_token_stats": {
            "avg_chunk_tokens": 16,
            "avg_query_tokens": 4,
            "avg_output_tokens": 2,
            "prompt_template_tokens": 8,
            "bytes_per_token": 4.0,
        },
    }
    out = apply_communication_costs(
        frame,
        pipeline_config=config,
        hardware=SystemConfig(gpu={"ICN": 10.0, "ICN_LL": 0.0}),
        seq_len_prefill=64,
        qe_seq_len=None,
        reranker_input_fanout=4,
        query_expansion_fanout=4,
    )
    boundaries = out.iloc[0]["comm_boundaries"]
    assert "semantic_retrieval_vectorsearch" not in boundaries
    assert "retrieval_fusion→passage_reranker" in boundaries
    assert out.iloc[0]["comm_latency_s"] == pytest.approx(4 * 16 * 4 / 1e10)


@pytest.mark.parametrize(
    "expansion, expected_fusion",
    [("hyde", False), ("query_decompose", False), ("multi_query_expansion", False)],
)
def test_fixed_stage_inference_never_guesses_fusion_from_expansion(expansion, expected_fusion):
    from rag_stack.system_layout import cm_stage_names_from_pipeline

    stages = cm_stage_names_from_pipeline(_resolved_pipeline(expansion))
    assert ("retrieval_fusion" in stages) is expected_fusion
    if expected_fusion:
        assert stages.index("retrieval_fusion") == stages.index("semantic_retrieval_vectorsearch") + 1
        assert stages.count("retrieval_fusion") == 1


@pytest.mark.parametrize("observed_fusion", [False, True])
def test_measured_layout_preserves_observed_fusion_as_host_work(observed_fusion):
    from rag_stack.performance_context import PerformanceContext
    from rag_stack.system_layout import cm_fixed_placement_inputs

    context = PerformanceContext.__new__(PerformanceContext)
    context.hardware = SystemConfig(num_cores=8)
    context.config = {"system": {
        "system_design_space": {
            "available_gpus": ["cuda:0"],
            "batch_size_request": 2,
            "batch_size_decode": 2,
        },
        "cm_search_space": {"min_num_gpus": 1, "max_num_gpus": 1},
        "retrieval": {"faiss_num_threads": 1, "num_servers": 4},
    }}
    pipeline = _resolved_pipeline("multi_query_expansion")
    if observed_fusion:
        pipeline["_evaluator_fusion_calls"] = [{
            "stage": "retrieval_fusion", "model_id": "rrf_pure",
        }]
    system = context.resolve_system_config({}, resolved_config=pipeline)
    layout = system["layout"]
    groups, chips, servers, stages = cm_fixed_placement_inputs(system, pipeline)

    assert servers == 4
    assert "retrieval_fusion" not in chips
    assert layout["total_gpu_slots"] == 1
    if not observed_fusion:
        assert "retrieval_fusion" not in stages
        assert "retrieval_fusion" not in layout["performance_stages"]
    else:
        assert ["retrieval_fusion"] in groups
        assert "retrieval_fusion" in stages
        fusion = layout["performance_stages"]["retrieval_fusion"]
        assert fusion["kind"] == "cpu_retrieval"
        assert fusion["num_chips"] == 0
        assert fusion["devices"] == []
        assert "num_servers" not in fusion
        assert "retrieval_fusion" not in layout["engines"]
        assert all(
            "retrieval_fusion" not in occupants
            for occupants in layout["gpu_occupants"].values()
        )
