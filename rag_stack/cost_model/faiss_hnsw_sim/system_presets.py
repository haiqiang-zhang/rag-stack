"""HNSW-flavored CPU presets.

Each factory returns the lean :class:`SystemConfig` from :mod:`.hardware`
populated with HNSW-relevant efficiency knobs. Hardware fields (peak
FLOPS, cache hierarchy, core topology) come from the shared
:mod:`rag_stack.cost_model.cpu_specs` module so the same physical numbers
back both IVF-PQ and HNSW presets.
"""

from rag_stack.cost_model.cpu_specs import (
    get_spec,
    CPUSpec,
    EPYC_7313,
    EPYC_9124,
    M3_PRO,
    ULTRA7_265K,
)

from .hardware import SystemConfig


def _from_spec(spec: CPUSpec, **algo_kwargs) -> SystemConfig:
    """Build an HNSW ``SystemConfig`` from a shared physical spec.

    The spec contributes hardware-fact fields; ``algo_kwargs`` carries
    HNSW-specific calibration (efficiencies, DRAM random-access latency,
    NUMA ceilings).
    """
    return SystemConfig(
        name=spec.name,
        num_cores=spec.num_cores,
        peak_flops=spec.peak_flops,
        mem_bandwidth_bytes_s=spec.mem_bandwidth_bytes_s,
        mem_bandwidth_1t_bytes_s=spec.mem_bandwidth_1t_bytes_s,
        simd_width_bits=spec.simd_width_bits,
        cache_hierarchy=list(spec.cache_hierarchy),
        core_topology=spec.core_topology,
        **algo_kwargs,
    )


def cpu_apple_m3_pro(
    active_threads: int = 0,
    simd_efficiency: float = 0.45,
    dram_random_access_latency_ns: float = 80.0,
    scattered_bw_ceiling: float = 0.55,         # Apple unified memory: higher than DDR
    numa_local_bytes: int = 0,
    numa_remote_bw_ceiling: float = 0.06,
    **kwargs,
) -> SystemConfig:
    """Apple M3 Pro (MacBook Pro 14", Nov 2023). HNSW flavor."""
    return _from_spec(
        get_spec("m3_pro"),
        active_threads=active_threads,
        simd_efficiency=simd_efficiency,
        dram_random_access_latency_ns=dram_random_access_latency_ns,
        scattered_bw_ceiling=scattered_bw_ceiling,
        numa_local_bytes=numa_local_bytes,
        numa_remote_bw_ceiling=numa_remote_bw_ceiling,
    )


def cpu_amd_epyc_7313(
    active_threads: int = 0,
    simd_efficiency: float = 0.40,
    dram_random_access_latency_ns: float = 100.0,
    scattered_bw_ceiling: float = 0.35,
    numa_local_bytes: int = 1_500_000_000,
    numa_remote_bw_ceiling: float = 0.05,
    **kwargs,
) -> SystemConfig:
    """AMD EPYC 7313 (2-socket, Milan / Zen 3, AVX2). HNSW flavor."""
    return _from_spec(
        get_spec("epyc_7313"),
        active_threads=active_threads,
        simd_efficiency=simd_efficiency,
        dram_random_access_latency_ns=dram_random_access_latency_ns,
        scattered_bw_ceiling=scattered_bw_ceiling,
        numa_local_bytes=numa_local_bytes,
        numa_remote_bw_ceiling=numa_remote_bw_ceiling,
    )


def cpu_amd_epyc_9124(
    active_threads: int = 0,
    simd_efficiency: float = 0.50,
    dram_random_access_latency_ns: float = 120.0,   # MEASURED on this box (pointer-chase,
                                                    # 128M working set, single-core core-0): 126ns;
                                                    # ~120 representative of a large scattered index.
                                                    # (DDR5 nominal 60 was ~2x too low — 2S NUMA + TLB.)
    scattered_bw_ceiling: float = 0.65,             # 12-channel DDR5
    numa_local_bytes: int = 0,                      # 2S; per-CCD L3 modeled explicitly
    numa_remote_bw_ceiling: float = 0.06,
    **kwargs,
) -> SystemConfig:
    """Dual-socket AMD EPYC 9124 (2S x 16 cores, Genoa / Zen 4). HNSW flavor."""
    return _from_spec(
        get_spec("epyc_9124"),
        active_threads=active_threads,
        simd_efficiency=simd_efficiency,
        dram_random_access_latency_ns=dram_random_access_latency_ns,
        scattered_bw_ceiling=scattered_bw_ceiling,
        numa_local_bytes=numa_local_bytes,
        numa_remote_bw_ceiling=numa_remote_bw_ceiling,
    )


def cpu_intel_core_ultra7_265k(
    active_threads: int = 0,
    simd_efficiency: float = 0.40,
    dram_random_access_latency_ns: float = 75.0,
    scattered_bw_ceiling: float = 0.40,
    numa_local_bytes: int = 0,
    numa_remote_bw_ceiling: float = 0.06,
    **kwargs,
) -> SystemConfig:
    """Intel Core Ultra 7 265K (Arrow Lake, desktop, 8P+16E). HNSW flavor."""
    return _from_spec(
        get_spec("ultra7_265k"),
        active_threads=active_threads,
        simd_efficiency=simd_efficiency,
        dram_random_access_latency_ns=dram_random_access_latency_ns,
        scattered_bw_ceiling=scattered_bw_ceiling,
        numa_local_bytes=numa_local_bytes,
        numa_remote_bw_ceiling=numa_remote_bw_ceiling,
    )


# Registry: name → factory function
SYSTEM_PRESETS = {
    "m3_pro":      cpu_apple_m3_pro,
    "epyc_7313":   cpu_amd_epyc_7313,
    "epyc_9124":   cpu_amd_epyc_9124,
    "ultra7_265k": cpu_intel_core_ultra7_265k,
}
