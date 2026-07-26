"""IVF-PQ-flavored CPU presets.

Each factory returns a :class:`SystemConfig` (the IVF-PQ flavor: includes
PQ scan efficiencies, per-probe overheads, and random-access penalties).

Hardware fields (peak FLOPS, cache hierarchy, etc.) are sourced from the
shared :mod:`rag_stack.cost_model.cpu_specs` module so the same physical
numbers are used here and in :mod:`rag_stack.cost_model.faiss_hnsw_sim.system_presets`.
"""

from rag_stack.cost_model.cpu_specs import (
    get_spec,
    CPUSpec,
    EPYC_7313,
    EPYC_9124,
    M3_PRO,
    ULTRA7_265K,
)

from .systems import RandomAccessPenalty, SystemConfig


def _from_spec(spec: CPUSpec, **algo_kwargs) -> SystemConfig:
    """Build an IVF-PQ ``SystemConfig`` from a shared physical spec.

    The spec contributes the hardware-fact fields (including the full
    :class:`CoreTopology` — uniform/NUMA/heterogeneous all live there);
    ``algo_kwargs`` carries the IVF-PQ-specific calibration knobs
    (efficiencies, overheads, random-access penalties, calibrated
    calibrated ``numa_local_bytes`` crossover).
    """
    return SystemConfig(
        name=spec.name,
        num_cores=spec.num_cores,
        peak_flops=spec.peak_flops,
        peak_int_ops=spec.peak_int_ops,
        mem_bandwidth_bytes_s=spec.mem_bandwidth_bytes_s,
        mem_bandwidth_1t_bytes_s=spec.mem_bandwidth_1t_bytes_s,
        simd_width_bits=spec.simd_width_bits,
        cache_hierarchy=list(spec.cache_hierarchy),
        core_topology=spec.core_topology,
        **algo_kwargs,
    )


def cpu_apple_m3_pro(
    active_threads: int = 0,
    blas_efficiency: float = 0.43,
    simd_efficiency: float = 0.45,
    flat_scan_efficiency: float = 0.45,
    pq_fastscan_efficiency: float = 0.36,
    pq_fastscan_batch_efficiency: float = 0.70,
    pq_standard_efficiency: float = 0.20,
    pq_fastscan_per_probe_overhead_us: float = 0.58,
    pq_standard_per_probe_overhead_us: float = 0.58,
    scattered_bw_ceiling: float = 0.55,         # Apple unified memory: higher than DDR
    numa_local_bytes: int = 0,
    **kwargs,
) -> SystemConfig:
    """Apple M3 Pro (MacBook Pro 14", Nov 2023). IVF-PQ flavor."""
    return _from_spec(
        get_spec("m3_pro"),
        active_threads=active_threads,
        blas_efficiency=blas_efficiency,
        simd_efficiency=simd_efficiency,
        flat_scan_efficiency=flat_scan_efficiency,
        pq_fastscan_efficiency=pq_fastscan_efficiency,
        pq_fastscan_batch_efficiency=pq_fastscan_batch_efficiency,
        pq_standard_efficiency=pq_standard_efficiency,
        pq_fastscan_per_probe_overhead_us=pq_fastscan_per_probe_overhead_us,
        pq_standard_per_probe_overhead_us=pq_standard_per_probe_overhead_us,
        scattered_bw_ceiling=scattered_bw_ceiling,
        numa_local_bytes=numa_local_bytes,
    )


def cpu_amd_epyc_7313(
    active_threads: int = 0,
    blas_efficiency: float = 0.60,
    simd_efficiency: float = 0.40,
    flat_scan_efficiency: float = 0.40,
    pq_fastscan_efficiency: float = 0.20,
    pq_fastscan_batch_efficiency: float = 0.55,
    pq_standard_efficiency: float = 0.20,
    pq_fastscan_per_probe_overhead_us: float = 0.0,
    pq_standard_per_probe_overhead_us: float = 0.0,
    scattered_bw_ceiling: float = 0.35,
    numa_local_bytes: int = 1_500_000_000,
    **kwargs,
) -> SystemConfig:
    """AMD EPYC 7313 (2-socket, Milan / Zen 3). IVF-PQ flavor."""
    return _from_spec(
        get_spec("epyc_7313"),
        active_threads=active_threads,
        blas_efficiency=blas_efficiency,
        simd_efficiency=simd_efficiency,
        flat_scan_efficiency=flat_scan_efficiency,
        pq_fastscan_efficiency=pq_fastscan_efficiency,
        pq_fastscan_batch_efficiency=pq_fastscan_batch_efficiency,
        pq_standard_efficiency=pq_standard_efficiency,
        pq_fastscan_per_probe_overhead_us=pq_fastscan_per_probe_overhead_us,
        pq_standard_per_probe_overhead_us=pq_standard_per_probe_overhead_us,
        scattered_bw_ceiling=scattered_bw_ceiling,
        numa_local_bytes=numa_local_bytes,
    )


def cpu_amd_epyc_9124(
    active_threads: int = 0,
    blas_efficiency: float = 0.70,
    simd_efficiency: float = 0.50,
    flat_scan_efficiency: float = 0.50,
    pq_fastscan_efficiency: float = 0.26,
    pq_fastscan_batch_efficiency: float = 0.73,
    pq_standard_efficiency: float = 0.10,
    pq_fastscan_per_probe_overhead_us: float = 0.0,
    pq_standard_per_probe_overhead_us: float = 3.0,
    scattered_bw_ceiling: float = 0.65,             # batch mode reuses inverted-list lines
    numa_local_bytes: int = 0,                      # Calibrator-fit on dual-socket
    **kwargs,
) -> SystemConfig:
    """AMD EPYC 9124 dual-socket (2S × 16 cores, Genoa / Zen 4, AVX-512).

    Aligned to configs/rag_stack/sgs_local_vllm.yaml — same physical
    hardware description (32 cores, 24-ch DDR5-4800; measured STREAM ~308
    GB/s aggregate / ~38 GB/s single-thread, 128 MiB L3 aggregate, 2 NUMA
    sockets).

    ``numa_local_bytes`` is tuned by the calibrator against real FAISS
    measurements; the
    NUMA branch in ``effective_bandwidth`` is gated by
    ``core_topology.is_numa`` (a hardware fact, num_sockets > 1).

    Only calibrated mode 0 is supported, so random-access pressure is a
    per-query function of nprobe/nlist and has no thread-cache exponent.
    """
    return _from_spec(
        get_spec("epyc_9124"),
        active_threads=active_threads,
        blas_efficiency=blas_efficiency,
        simd_efficiency=simd_efficiency,
        flat_scan_efficiency=flat_scan_efficiency,
        pq_fastscan_efficiency=pq_fastscan_efficiency,
        pq_fastscan_batch_efficiency=pq_fastscan_batch_efficiency,
        pq_standard_efficiency=pq_standard_efficiency,
        pq_fastscan_per_probe_overhead_us=pq_fastscan_per_probe_overhead_us,
        pq_standard_per_probe_overhead_us=pq_standard_per_probe_overhead_us,
        scattered_bw_ceiling=scattered_bw_ceiling,
        numa_local_bytes=numa_local_bytes,
        random_access_penalty_standard=RandomAccessPenalty(
            c1=60.0, c2=50.0, c3=0.0,
        ),
    )


def cpu_intel_core_ultra7_265k(
    active_threads: int = 0,
    blas_efficiency: float = 0.55,
    simd_efficiency: float = 0.40,
    flat_scan_efficiency: float = 0.40,
    # IVF-PQ params kept at conservative defaults — NOT calibrated for this CPU.
    # Run the IVF-PQ calibrator first if you actually want IVF-PQ on this box.
    pq_fastscan_efficiency: float = 0.25,
    pq_fastscan_batch_efficiency: float = 0.60,
    pq_standard_efficiency: float = 0.15,
    pq_fastscan_per_probe_overhead_us: float = 0.0,
    pq_standard_per_probe_overhead_us: float = 2.0,
    scattered_bw_ceiling: float = 0.40,
    numa_local_bytes: int = 0,
    **kwargs,
) -> SystemConfig:
    """Intel Core Ultra 7 265K (Arrow Lake, desktop, 8P+16E). IVF-PQ flavor."""
    return _from_spec(
        get_spec("ultra7_265k"),
        active_threads=active_threads,
        blas_efficiency=blas_efficiency,
        simd_efficiency=simd_efficiency,
        flat_scan_efficiency=flat_scan_efficiency,
        pq_fastscan_efficiency=pq_fastscan_efficiency,
        pq_fastscan_batch_efficiency=pq_fastscan_batch_efficiency,
        pq_standard_efficiency=pq_standard_efficiency,
        pq_fastscan_per_probe_overhead_us=pq_fastscan_per_probe_overhead_us,
        pq_standard_per_probe_overhead_us=pq_standard_per_probe_overhead_us,
        scattered_bw_ceiling=scattered_bw_ceiling,
        numa_local_bytes=numa_local_bytes,
    )


# Registry: name → factory function
SYSTEM_PRESETS = {
    "m3_pro":      cpu_apple_m3_pro,
    "epyc_7313":   cpu_amd_epyc_7313,
    "epyc_9124":   cpu_amd_epyc_9124,
    "ultra7_265k": cpu_intel_core_ultra7_265k,
}
