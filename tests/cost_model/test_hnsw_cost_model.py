"""CPU-only analytical HNSW workload and cost-model behavior."""

from dataclasses import replace
import math

import pytest

from rag_stack.cost_model.faiss_hnsw_sim.cost_model import HNSWCostModel
from rag_stack.cost_model.faiss_hnsw_sim.graph_stats import estimate_graph_stats
from rag_stack.cost_model.faiss_hnsw_sim.hnsw_config import HNSWConfig
from rag_stack.cost_model.faiss_hnsw_sim.system_presets import (
    cpu_amd_epyc_7313,
    cpu_apple_m3_pro,
)
from rag_stack.cost_model.faiss_hnsw_sim.workload_predictor import WorkloadPredictor


@pytest.fixture
def config():
    return HNSWConfig(N=1_000_000, d=128, M=16, ef_search=64, k=10, batch_size=1)


def _estimate(config, hardware):
    workload = WorkloadPredictor().predict(config)
    return HNSWCostModel(config, hardware, workload).estimate()


@pytest.mark.parametrize("preset", [cpu_apple_m3_pro, cpu_amd_epyc_7313])
def test_estimate_reports_finite_positive_costs(config, preset):
    result = _estimate(config, preset())

    assert math.isfinite(result.total_time_s) and result.total_time_s > 0
    assert math.isfinite(result.throughput_qps) and result.throughput_qps > 0
    assert result.operators
    assert all(math.isfinite(op.time_s) and op.time_s >= 0 for op in result.operators)


@pytest.mark.parametrize(
    "field,values",
    [
        ("ef_search", [16, 64, 256]),
        ("M", [8, 16, 64]),
        ("N", [10_000, 100_000, 1_000_000]),
    ],
)
def test_larger_search_workload_does_not_reduce_latency(config, field, values):
    hardware = cpu_apple_m3_pro()
    times = [_estimate(replace(config, **{field: value}), hardware).total_time_s for value in values]

    assert all(later >= earlier for earlier, later in zip(times, times[1:]))


def test_analytical_workload_is_available_without_a_trained_model(config):
    predictor = WorkloadPredictor()
    graph = estimate_graph_stats(config.N, config.d, config.M, config.ef_construction)
    workload = predictor.predict(config, graph)
    larger_search = predictor.predict(replace(config, ef_search=256), graph)

    assert not predictor.is_trained
    assert workload.ndis_per_query >= workload.nhops_per_query > 0
    assert larger_search.ndis_per_query > workload.ndis_per_query
    assert larger_search.nhops_per_query > workload.nhops_per_query
