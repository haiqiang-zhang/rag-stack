"""Unit tests for CPU topology metadata in YAML / SystemConfig / IVF-PQ.

Three topology types are mutually exclusive (v1):

  * uniform       — single-socket, homogeneous cores (default)
  * numa          — multi-socket NUMA (e.g. dual EPYC)
  * heterogeneous — single-socket big.LITTLE (e.g. M3 Pro, Ultra 7)

Tests lock in:

  1. YAML → ``rag_stack.types.SystemConfig`` parses ``cpu.topology:`` block
     into ``topology_type`` / ``numa_sockets`` / ``numa_cores_per_socket``,
     and preserves the legacy flat ``num_p_cores`` shape for backward compat.
  2. ``system_config_to_faiss_hardware`` synthesizes the right
     :class:`CoreTopology` (``is_numa`` for NUMA; P/E peaks for heterogeneous)
     so the IVF-PQ ``effective_bandwidth`` branch is wired correctly.
  3. ``Calibrator`` drops ``numa_local_bytes`` / ``numa_remote_bw_ceiling``
     from its search space when the target hardware is not NUMA — those
     parameters would be tuned to no effect (``numa_enabled`` is False).
  4. ``ConfigValidator`` rejects malformed topology blocks with clear errors.
"""

from __future__ import annotations

from rag_stack.types import SystemConfig
from rag_stack.cost_model.faiss_ivf_sim.systems import (
    SystemConfig as IVFSystemConfig,
    system_config_to_faiss_hardware,
)


# ---------------------------------------------------------------------------
# YAML → rag_stack.SystemConfig roundtrip
# ---------------------------------------------------------------------------


def _uniform_cpu_yaml():
    return {
        "cpu": {
            "num_cores": 16,
            "peak_flops": 1.89e12,
            "peak_int_ops": 0.95e12,
            "mem_bandwidth": 200e9,
            "mem_bandwidth_1t": 30e9,
            "memory_tiers": [
                {"name": "ddr", "bandwidth": 200e9, "bandwidth_1t": 30e9, "capacity": 64e9},
            ],
        },
        "search_space": {"batch_size_request": 256, "max_num_gpus": 8, "min_num_gpus": 1},
    }


def _numa_cpu_yaml():
    return {
        "cpu": {
            "num_cores": 32,
            "peak_flops": 3.79e12,
            "peak_int_ops": 1.89e12,
            "mem_bandwidth": 600e9,
            "mem_bandwidth_1t": 38e9,
            "memory_tiers": [
                {"name": "ddr", "bandwidth": 600e9, "bandwidth_1t": 38e9, "capacity": 384e9},
            ],
            "topology": {
                "type": "numa",
                "sockets": 2,
                "cores_per_socket": 16,
            },
        },
        "search_space": {"batch_size_request": 256, "max_num_gpus": 8, "min_num_gpus": 1},
    }


def _heterogeneous_cpu_yaml():
    return {
        "cpu": {
            "num_cores": 12,
            "peak_flops": 1.0e12,
            "peak_int_ops": 1.0e12,
            "mem_bandwidth": 116.5e9,
            "mem_bandwidth_1t": 36e9,
            "memory_tiers": [{"name": "ddr", "bandwidth": 116.5e9, "bandwidth_1t": 36e9, "capacity": 32e9}],
            "topology": {
                "type": "heterogeneous",
                "p_cores": {
                    "count": 6, "clock_ghz": 4.0,
                    "flops_per_cycle": 32, "int_ops_per_cycle": 32,
                },
                "e_cores": {
                    "count": 6, "clock_ghz": 2.75,
                    "flops_per_cycle": 16, "int_ops_per_cycle": 16,
                },
                "thread_migration": True,
            },
        },
        "search_space": {"batch_size_request": 256, "max_num_gpus": 8, "min_num_gpus": 1},
    }


def test_uniform_parses_with_defaults():
    sc = SystemConfig.from_dict(_uniform_cpu_yaml())
    assert sc.topology_type == "uniform"
    assert sc.numa_sockets == 1
    assert sc.numa_cores_per_socket == 0
    assert sc.num_p_cores == 0   # no heterogeneous fields populated


def test_numa_parses_sockets_and_cores_per_socket():
    sc = SystemConfig.from_dict(_numa_cpu_yaml())
    assert sc.topology_type == "numa"
    assert sc.numa_sockets == 2
    assert sc.numa_cores_per_socket == 16
    # Heterogeneous fields stay 0
    assert sc.num_p_cores == 0
    assert sc.num_e_cores == 0


def test_heterogeneous_parses_p_e_cores():
    sc = SystemConfig.from_dict(_heterogeneous_cpu_yaml())
    assert sc.topology_type == "heterogeneous"
    assert sc.num_p_cores == 6 and sc.num_e_cores == 6
    assert sc.p_core_clock_ghz == 4.0
    assert sc.p_core_flops_per_cycle == 32
    assert sc.thread_migration is True


def test_legacy_flat_heterogeneous_yaml_still_works():
    """Pre-topology YAMLs (no ``topology:`` block, flat ``num_p_cores:``)
    must keep parsing — preserves backward compat for existing configs."""
    legacy = {
        "cpu": {
            "num_cores": 12,
            "peak_flops": 1.0e12,
            "peak_int_ops": 1.0e12,
            "mem_bandwidth": 116.5e9,
            "mem_bandwidth_1t": 36e9,
            "memory_tiers": [{"name": "ddr", "bandwidth": 116.5e9, "bandwidth_1t": 36e9, "capacity": 32e9}],
            "num_p_cores": 6, "num_e_cores": 6,
            "p_core_clock_ghz": 4.0, "e_core_clock_ghz": 2.75,
            "p_core_flops_per_cycle": 32, "e_core_flops_per_cycle": 16,
            "p_core_int_ops_per_cycle": 32, "e_core_int_ops_per_cycle": 16,
        },
        "search_space": {"batch_size_request": 256, "max_num_gpus": 8, "min_num_gpus": 1},
    }
    sc = SystemConfig.from_dict(legacy)
    assert sc.topology_type == "uniform"   # no topology block → uniform
    assert sc.num_p_cores == 6 and sc.num_e_cores == 6
    # Legacy converter still synthesizes a heterogeneous CoreTopology
    # downstream (see test_legacy_flat_converts_to_heterogeneous).


def test_to_dict_roundtrip_preserves_topology():
    """Serialize and re-parse — topology survives."""
    for fixture in (_uniform_cpu_yaml, _numa_cpu_yaml, _heterogeneous_cpu_yaml):
        sc = SystemConfig.from_dict(fixture())
        sc2 = SystemConfig.from_dict(sc.to_dict())
        assert sc.topology_type == sc2.topology_type
        assert sc.numa_sockets == sc2.numa_sockets
        assert sc.numa_cores_per_socket == sc2.numa_cores_per_socket
        assert sc.num_p_cores == sc2.num_p_cores
        assert sc.num_e_cores == sc2.num_e_cores


# ---------------------------------------------------------------------------
# rag_stack.SystemConfig → IVF-PQ SystemConfig conversion
# ---------------------------------------------------------------------------


def test_uniform_converts_to_no_core_topology():
    """Uniform → ``core_topology=None``, ``numa_enabled=False``."""
    sc = SystemConfig.from_dict(_uniform_cpu_yaml())
    ivfpq = system_config_to_faiss_hardware(sc, active_threads=16)
    assert ivfpq.core_topology is None
    assert ivfpq.numa_enabled is False


def test_numa_converts_to_numa_core_topology():
    """NUMA → ``CoreTopology(num_sockets=2)`` (is_numa derived), numa_enabled=True."""
    sc = SystemConfig.from_dict(_numa_cpu_yaml())
    ivfpq = system_config_to_faiss_hardware(sc, active_threads=32)
    assert ivfpq.core_topology is not None
    assert ivfpq.core_topology.is_numa is True
    assert ivfpq.core_topology.num_sockets == 2
    assert ivfpq.numa_enabled is True
    # Heterogeneous fields stay 0 (NUMA-only, no big.LITTLE)
    assert ivfpq.core_topology.num_p_cores == 0
    assert ivfpq.core_topology.num_e_cores == 0


def test_heterogeneous_converts_to_p_e_core_topology():
    """Heterogeneous → CoreTopology with P/E peaks; num_sockets=1, is_numa=False."""
    sc = SystemConfig.from_dict(_heterogeneous_cpu_yaml())
    ivfpq = system_config_to_faiss_hardware(sc, active_threads=12)
    assert ivfpq.core_topology is not None
    assert ivfpq.core_topology.is_heterogeneous is True
    assert ivfpq.core_topology.is_numa is False
    assert ivfpq.numa_enabled is False
    assert ivfpq.core_topology.num_p_cores == 6
    assert ivfpq.core_topology.num_e_cores == 6
    # P-core peak = clock * flops_per_cycle = 4.0e9 * 32 = 1.28e11
    assert ivfpq.core_topology.p_core_peak_flops == 4.0e9 * 32


def test_legacy_flat_converts_to_heterogeneous():
    """Backward-compat: legacy flat fields → heterogeneous CoreTopology."""
    legacy = {
        "cpu": {
            "num_cores": 12,
            "peak_flops": 1.0e12, "peak_int_ops": 1.0e12, "mem_bandwidth": 116.5e9,
            "mem_bandwidth_1t": 36e9,
            "memory_tiers": [{"name": "ddr", "bandwidth": 116.5e9, "bandwidth_1t": 36e9, "capacity": 32e9}],
            "num_p_cores": 6, "num_e_cores": 6,
            "p_core_clock_ghz": 4.0, "e_core_clock_ghz": 2.75,
            "p_core_flops_per_cycle": 32, "e_core_flops_per_cycle": 16,
            "p_core_int_ops_per_cycle": 32, "e_core_int_ops_per_cycle": 16,
        },
        "search_space": {"batch_size_request": 256, "max_num_gpus": 8, "min_num_gpus": 1},
    }
    sc = SystemConfig.from_dict(legacy)
    ivfpq = system_config_to_faiss_hardware(sc, active_threads=12)
    assert ivfpq.core_topology is not None
    assert ivfpq.core_topology.is_heterogeneous
    assert not ivfpq.numa_enabled


# ---------------------------------------------------------------------------
# Cost model: numa_enabled gates the cross-socket BW branch
# ---------------------------------------------------------------------------


def test_uniform_skips_numa_branch_in_effective_bandwidth():
    """Uniform: even if numa_local_bytes>0 (stale), branch isn't taken.

    Uses a large access block (1 MiB) so ``bw_1t * threads`` saturates
    above either ceiling and ``min(...)`` picks the ceiling — the regime
    where the NUMA-vs-scattered choice matters. Tiny blocks give a
    latency-bound result regardless of which ceiling would have applied.
    """
    from rag_stack.cost_model.cpu_specs import CacheLevel

    hw = IVFSystemConfig(
        name="test_uniform",
        num_cores=16, peak_flops=1e12, peak_int_ops=1e12,
        mem_bandwidth_bytes_s=200e9, mem_bandwidth_1t_bytes_s=200e9, simd_width_bits=512,
        cache_hierarchy=[CacheLevel("ddr", int(32e9), 200e9, 200e9, 100.0)],
        active_threads=16,
        scattered_bw_ceiling=0.35,
        numa_local_bytes=1_000_000,   # stale value, shouldn't matter
        core_topology=None,           # uniform → numa_enabled=False
    )
    assert hw.numa_enabled is False
    bw = hw.effective_bandwidth(
        working_set_bytes=10_000_000, access_block_bytes=1_000_000, threads=16,
    )
    # Should be capped by 200e9 * 0.35 = 70e9.
    assert bw <= 70e9 * 1.01, f"bw={bw}: should be at scattered ceiling 70 GB/s"


def test_numa_enabled_does_take_numa_branch():
    """NUMA: working_set > numa_local_bytes → socket-saturation branch.

    Same large-block regime as the uniform test so ceiling is the
    binding constraint.
    """
    import pytest
    from rag_stack.cost_model.cpu_specs import CacheLevel, CoreTopology

    hw = IVFSystemConfig(
        name="test_numa",
        num_cores=32, peak_flops=3.79e12, peak_int_ops=1.89e12,
        mem_bandwidth_bytes_s=600e9, mem_bandwidth_1t_bytes_s=10e9, simd_width_bits=512,
        cache_hierarchy=[CacheLevel("ddr", int(384e9), 600e9, 10e9, 100.0)],
        active_threads=32,
        scattered_bw_ceiling=0.35,
        numa_local_bytes=1_000_000,
        core_topology=CoreTopology(num_sockets=2),
    )
    assert hw.numa_enabled is True
    bw = hw.effective_bandwidth(
        working_set_bytes=10_000_000, access_block_bytes=1_000_000, threads=32,
    )
    # NUMA branch reaches the per-socket scattered aggregate at cores/socket
    # and stays there. The old remote-gain coefficient was not identifiable
    # from the standalone grid and is intentionally absent.
    node_agg = 600e9 * 0.35
    assert bw == pytest.approx(node_agg)


def test_partitioned_batch_uses_per_worker_numa_footprint():
    """Mode 0 partitions a query batch, so NUMA first-touch is per worker.

    The full batch footprint exceeds ``numa_local_bytes``; each worker's slice
    does not. A partitioned mode-0 scan should avoid the socket-saturation
    branch while a cooperative same-query scan still sees the full footprint.
    """
    from rag_stack.cost_model.cpu_specs import CacheLevel, CoreTopology

    hw = IVFSystemConfig(
        name="test_partitioned_numa",
        num_cores=32, peak_flops=3.79e12, peak_int_ops=1.89e12,
        mem_bandwidth_bytes_s=600e9, mem_bandwidth_1t_bytes_s=38e9,
        simd_width_bits=512,
        cache_hierarchy=[CacheLevel("ddr", int(384e9), 600e9, 38e9, 100.0)],
        active_threads=4,
        scattered_bw_ceiling=0.35,
        numa_local_bytes=150_000_000,
        core_topology=CoreTopology(num_sockets=2),
    )

    common = dict(
        total_ops=1.0,
        total_bytes=400_000_000,
        unique_bytes=400_000_000,
        working_set_bytes=4_000_000,
        warm_cache=True,
        compute_mode="int",
        access_block_bytes=1_000_000,
        compute_threads=4,
        memory_threads=4,
    )
    partitioned = hw.roofline_time(**common, partitioned=True)
    cooperative = hw.roofline_time(**common, partitioned=False)

    assert partitioned["memory_time_s"] < cooperative["memory_time_s"]


# ---------------------------------------------------------------------------
# ConfigValidator
# ---------------------------------------------------------------------------


def _make_full_config(cpu_block: dict) -> dict:
    """Wrap a cpu block into a minimal valid config so the validator runs."""
    return {
        "global": {"eval_backend": "flashrag", "n_iterations": 1},
        "dataset": {},
        "system": {
            "cpu": cpu_block,
            "cm_search_space": {"batch_size_request": 256, "max_num_gpus": 8, "min_num_gpus": 1},
        },
        "vectordb": [{"name": "x", "db_type": "faiss_hnsw"}],
        "optimizer": {"type": "smac_mobo", "params": {}},
        "eval_backend_setting": {"metrics": [{"metric_name": "em"}]},
        "algo_search_space": {
            "pipeline": {"methods": [{
                "mode": "sequential",
                "node_lines": [{
                    "node_line_name": "x",
                    "nodes": [
                        {"stage": "semantic_retrieval", "top_k": 4,
                         "modules": [{"component": "vectordb", "vectordb": "x"}]},
                        {"stage": "generator",
                         "modules": [{"component": "vllm", "model": "q"}]},
                    ],
                }],
            }]},
        },
    }


def _topology_errors(cpu_block: dict) -> list:
    """Return only the topology-related errors so we don't get false positives
    from unrelated cross-checks elsewhere in the validator."""
    from rag_stack.config_validator import ConfigValidator

    cfg = _make_full_config(cpu_block)
    errs = ConfigValidator(cfg).validate()
    return [e for e in errs if "topology" in e]


def test_validator_accepts_valid_numa():
    errs = _topology_errors({
        "num_cores": 32,
        "peak_flops": 3.79e12,
        "topology": {"type": "numa", "sockets": 2, "cores_per_socket": 16},
    })
    assert not errs, errs


def test_validator_rejects_numa_with_wrong_core_count():
    errs = _topology_errors({
        "num_cores": 32,
        "peak_flops": 3.79e12,
        "topology": {"type": "numa", "sockets": 2, "cores_per_socket": 8},  # 2*8=16 != 32
    })
    assert any("!= num_cores" in e for e in errs), errs


def test_validator_rejects_numa_single_socket():
    errs = _topology_errors({
        "num_cores": 16,
        "peak_flops": 1.89e12,
        "topology": {"type": "numa", "sockets": 1, "cores_per_socket": 16},
    })
    assert any("sockets >= 2" in e for e in errs), errs


def test_validator_rejects_unknown_type():
    errs = _topology_errors({
        "num_cores": 16,
        "peak_flops": 1.89e12,
        "topology": {"type": "bogus"},
    })
    assert any("not one of" in e for e in errs), errs


def test_validator_rejects_heterogeneous_wrong_p_plus_e():
    errs = _topology_errors({
        "num_cores": 12,
        "peak_flops": 1e12,
        "topology": {
            "type": "heterogeneous",
            "p_cores": {"count": 4, "clock_ghz": 4.0, "flops_per_cycle": 32, "int_ops_per_cycle": 32},
            "e_cores": {"count": 6, "clock_ghz": 2.75, "flops_per_cycle": 16, "int_ops_per_cycle": 16},
        },
    })
    assert any("!= num_cores" in e for e in errs), errs


def test_validator_rejects_heterogeneous_with_numa_sockets():
    """v1: heterogeneous and numa are mutually exclusive."""
    errs = _topology_errors({
        "num_cores": 12,
        "peak_flops": 1e12,
        "topology": {
            "type": "heterogeneous",
            "p_cores": {"count": 6, "clock_ghz": 4.0, "flops_per_cycle": 32, "int_ops_per_cycle": 32},
            "e_cores": {"count": 6, "clock_ghz": 2.75, "flops_per_cycle": 16, "int_ops_per_cycle": 16},
            "sockets": 2,
        },
    })
    assert any("not supported in v1" in e or "must not set 'sockets'" in e for e in errs), errs


def test_validator_no_error_for_uniform_or_absent_topology():
    """Uniform OR no topology block → no topology errors."""
    errs = _topology_errors({
        "num_cores": 16, "peak_flops": 1e12,
        "topology": {"type": "uniform"},
    })
    assert not errs, errs
    errs = _topology_errors({
        "num_cores": 16, "peak_flops": 1e12,
        # no topology key
    })
    assert not errs, errs
