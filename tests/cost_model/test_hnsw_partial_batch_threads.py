"""HNSW inter-query threading must be bounded by the live batch."""

import pytest

from rag_stack.cost_model.faiss_hnsw_sim.cost_model import (
    HNSWCoefficients,
    HNSWCostModel,
)
from rag_stack.cost_model.faiss_hnsw_sim.hardware import SystemConfig
from rag_stack.cost_model.faiss_hnsw_sim.hnsw_config import HNSWConfig
from rag_stack.cost_model.faiss_hnsw_sim.workload_predictor import WorkloadProfile


def _estimate(batch_size: int, active_threads: int):
    # Make the roofline compute-bound so its parallel width is directly
    # observable.  These are synthetic hardware facts, not fitted constants.
    hardware = SystemConfig(
        name="compute-bound-test-cpu",
        num_cores=16,
        peak_flops=1.0e6,
        mem_bandwidth_bytes_s=1.0e30,
        mem_bandwidth_1t_bytes_s=1.0e30,
        active_threads=active_threads,
        dram_random_access_latency_ns=0.0,
        scattered_bw_ceiling=1.0,
    )
    config = HNSWConfig(
        N=1_000_000,
        d=128,
        M=16,
        ef_search=64,
        k=10,
        batch_size=batch_size,
    )
    # Remove the separate instruction-overhead phase so this test isolates the
    # existing roofline's inter-query width; the work law itself is unchanged.
    coefficients = HNSWCoefficients(
        cycles_per_visited_probe=0.0,
        cycles_per_heap_op=0.0,
    )
    workload = WorkloadProfile(ndis_per_query=1024.0, nhops_per_query=64.0)
    return HNSWCostModel(config, hardware, workload, coefficients).estimate()


def _work(result) -> tuple[float, float]:
    return (
        sum(op.total_ops for op in result.operators),
        sum(op.total_bytes for op in result.operators),
    )


def test_single_query_uses_one_thread_even_when_more_are_configured():
    one_thread = _estimate(batch_size=1, active_threads=1)
    sixteen_threads = _estimate(batch_size=1, active_threads=16)

    assert sixteen_threads.total_time_s == pytest.approx(one_thread.total_time_s)
    assert _work(sixteen_threads) == pytest.approx(_work(one_thread))


def test_partial_batch_uses_no_more_threads_than_queries():
    four_threads = _estimate(batch_size=4, active_threads=4)
    sixteen_threads = _estimate(batch_size=4, active_threads=16)

    assert sixteen_threads.total_time_s == pytest.approx(four_threads.total_time_s)
    assert _work(sixteen_threads) == pytest.approx(_work(four_threads))


def test_full_batch_remains_capped_by_configured_threads():
    single_query = _estimate(batch_size=1, active_threads=1)
    batch = _estimate(batch_size=16, active_threads=4)

    # Sixteen independent queries executed four at a time take four waves.
    assert batch.total_time_s == pytest.approx(4.0 * single_query.total_time_s)
    assert _work(batch) == pytest.approx(
        tuple(16.0 * value for value in _work(single_query))
    )


def test_instruction_overhead_uses_the_same_inter_query_width():
    """The fixed-cycle graph phase is batch work, not serial batch wall time."""
    hardware = SystemConfig(
        name="instruction-test-cpu",
        num_cores=16,
        peak_flops=1.0e30,
        mem_bandwidth_bytes_s=1.0e30,
        mem_bandwidth_1t_bytes_s=1.0e30,
        active_threads=4,
        dram_random_access_latency_ns=0.0,
        scattered_bw_ceiling=1.0,
    )
    workload = WorkloadProfile(ndis_per_query=0.0, nhops_per_query=64.0)

    def graph_wall(batch_size: int) -> float:
        config = HNSWConfig(
            N=1_000_000,
            d=128,
            M=16,
            ef_search=64,
            k=10,
            batch_size=batch_size,
        )
        result = HNSWCostModel(config, hardware, workload).estimate()
        return next(
            op.time_s for op in result.operators
            if op.name == "2b_graph_overhead"
        )

    assert graph_wall(1) > 0.0
    assert graph_wall(16) == pytest.approx(4.0 * graph_wall(1))
