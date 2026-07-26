"""Shared physical CPU specifications.

Pure hardware facts that come from manufacturer datasheets and STREAM
benchmarks: peak FLOPS, peak INT ops, memory bandwidth, cache hierarchy,
core topology, SIMD width.

Algorithm-specific calibrations (efficiencies, per-probe overheads, random-
access penalties, etc.) live in each algorithm's own preset module — they
are not hardware facts and don't belong here.

This file is consumed by:
- :mod:`rag_stack.cost_model.faiss_ivf_sim.system_presets`
- :mod:`rag_stack.cost_model.faiss_hnsw_sim.system_presets`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Hardware primitives
# ---------------------------------------------------------------------------

@dataclass
class CacheLevel:
    """A single level of cache hierarchy.

    Bandwidth is stored at TWO explicit, unambiguous granularities — there is
    NO ``memory_efficiency`` fudge factor that conflates the two:

    - ``bandwidth_bytes_s``    : AGGREGATE (all-core sustained STREAM) at this tier.
    - ``bandwidth_1t_bytes_s`` : SINGLE-THREAD (one-core sustained STREAM) at this tier.

    Effective bandwidth at ``t`` threads is ``min(t * bandwidth_1t, bandwidth)``
    (linear up to the aggregate cap). At ``t = 1`` (the only regime the cost
    model is calibrated/trusted in) it is exactly the measured single-thread
    value. Both numbers are hardware facts from STREAM benchmarks.
    """
    name: str
    size_bytes: int                # PER-INSTANCE capacity (one core's private L1/L2,
                                   # or one shared pool's L3) — NOT the aggregate.
    bandwidth_bytes_s: float       # aggregate (all-core sustained)
    bandwidth_1t_bytes_s: float    # single-thread (one-core sustained)
    latency_ns: float
    # How many cores share ONE instance of this cache level. 1 = private per core
    # (L1/L2 on most CPUs); >1 = shared (e.g. L3 per CCX/CCD/socket). A single
    # thread sees ``size_bytes`` (its private L1/L2, or its shared pool uncontended);
    # ``t`` threads contending one shared instance each see ``size_bytes /
    # min(t, cores_per_instance)``. Defaults to 1 → callers that stored AGGREGATE
    # sizes (other presets) keep their old single-thread behavior unchanged.
    cores_per_instance: int = 1


@dataclass
class CoreTopology:
    """Core / socket topology metadata.

    Holds two ORTHOGONAL kinds of non-uniformity:

    1. **Heterogeneous cores (big.LITTLE)**: when ``num_e_cores > 0``,
       cost model uses ``aggregate_peak`` / ``bottleneck_peak`` to mix
       P/E core throughput. Apple Silicon / Intel Arrow Lake style.
    2. **NUMA (multi-socket)**: when ``num_sockets > 1``, cost model
       enables the cross-socket bandwidth-penalty branch in
       ``effective_bandwidth`` via the derived :attr:`is_numa` property.

    Both flags are **hardware facts** sourced from YAML / preset — they
    are NEVER fit by the calibrator. The numeric calibration knobs
    (``numa_local_bytes`` / ``numa_remote_bw_ceiling``) remain on
    :class:`rag_stack.cost_model.faiss_ivf_sim.systems.SystemConfig`
    and ARE calibrator-tuned, but only when NUMA mode is active.

    Default values describe a "uniform single-socket homogeneous" CPU:
    callers fill in only the fields relevant to their topology.

    Thread assignment for heterogeneous cores (FAISS / OMP greedy):
    threads land on P-cores first, then E-cores (Apple Silicon
    behavior).
    """
    num_p_cores: int = 0
    num_e_cores: int = 0
    p_core_peak_flops: float = 0.0       # FLOP/s per single P-core
    e_core_peak_flops: float = 0.0       # FLOP/s per single E-core
    p_core_peak_int_ops: float = 0.0     # INT-OP/s per single P-core
    e_core_peak_int_ops: float = 0.0     # INT-OP/s per single E-core
    p_core_clock_hz: float = 0.0
    e_core_clock_hz: float = 0.0

    # Physical socket count (hardware fact). >1 ⇒ NUMA mode (see :attr:`is_numa`).
    num_sockets: int = 1

    @property
    def total_cores(self) -> int:
        return self.num_p_cores + self.num_e_cores

    @property
    def is_heterogeneous(self) -> bool:
        return self.num_e_cores > 0

    @property
    def is_numa(self) -> bool:
        """True iff this is a multi-socket system (NUMA mode on)."""
        return self.num_sockets > 1

    def aggregate_peak(self, num_threads: int, mode: str = "flops") -> float:
        """Aggregate peak throughput for cooperative work (mode 1).

        All threads contribute; total = sum of per-core peaks.
        """
        p_peak, e_peak = self._peaks(mode)
        p_t = min(num_threads, self.num_p_cores)
        e_t = min(max(0, num_threads - self.num_p_cores), self.num_e_cores)
        return p_t * p_peak + e_t * e_peak

    def bottleneck_peak(self, num_threads: int, mode: str = "flops") -> float:
        """Effective peak for partitioned work (mode 0).

        Wall time = slowest thread. If E-cores are active, they're the
        bottleneck; effective throughput = total_work / wall_time.
        """
        p_peak, e_peak = self._peaks(mode)
        p_t = min(num_threads, self.num_p_cores)
        e_t = min(max(0, num_threads - self.num_p_cores), self.num_e_cores)
        if e_t > 0:
            return num_threads * e_peak
        return p_t * p_peak

    def _peaks(self, mode: str):
        if mode == "flops":
            return self.p_core_peak_flops, self.e_core_peak_flops
        return self.p_core_peak_int_ops, self.e_core_peak_int_ops


@dataclass
class CPUSpec:
    """Pure physical CPU specification (no algorithm-specific knobs).

    All topology metadata (heterogeneous cores, multi-socket count) lives
    on :class:`CoreTopology` — set ``core_topology`` explicitly when the
    CPU is non-uniform. ``core_topology=None`` is the uniform default
    (single socket, homogeneous cores).
    """
    name: str
    num_cores: int
    peak_flops: float                          # all-cores FP32 FLOP/s
    peak_int_ops: float                        # all-cores INT-OP/s
    mem_bandwidth_bytes_s: float               # AGGREGATE DRAM B/s (all-core sustained STREAM)
    mem_bandwidth_1t_bytes_s: float            # SINGLE-THREAD DRAM B/s (one-core sustained STREAM)
    simd_width_bits: int                       # 256 = AVX2, 512 = AVX-512, 128 = NEON
    cache_hierarchy: list[CacheLevel] = field(default_factory=list)
    core_topology: Optional[CoreTopology] = None
    # Short stable registry key (e.g. "epyc_9124") — the CPU sibling of gpu.hardware_key.
    # Names the FAISS calibration data/profile. Backfilled from the CPU_SPECS key by get_spec.
    hardware_key: str = ""

    @property
    def is_numa(self) -> bool:
        """Derived: True iff multi-socket (forwarded from core_topology)."""
        return self.core_topology is not None and self.core_topology.is_numa

    @property
    def num_sockets(self) -> int:
        """Derived: physical socket count (forwarded from core_topology)."""
        return self.core_topology.num_sockets if self.core_topology is not None else 1


# ---------------------------------------------------------------------------
# Preset constants
# ---------------------------------------------------------------------------

_M3_PRO_TOPOLOGY = CoreTopology(
    num_p_cores=6,
    num_e_cores=6,
    # P-core: 4 NEON units × 4 FP32 × 2 (FMA) = 32 FLOP/cycle
    p_core_peak_flops=4.056e9 * 32,
    # E-core: 2 NEON units × 4 FP32 × 2 = 16 FLOP/cycle
    e_core_peak_flops=2.75e9 * 16,
    p_core_peak_int_ops=4.056e9 * 32,
    e_core_peak_int_ops=2.75e9 * 16,
    p_core_clock_hz=4.056e9,
    e_core_clock_hz=2.75e9,
)

M3_PRO = CPUSpec(
    name="Apple M3 Pro (MacBook Pro 14\", Nov 2023)",
    num_cores=12,                              # 6 P + 6 E
    peak_flops=(6 * _M3_PRO_TOPOLOGY.p_core_peak_flops
                + 6 * _M3_PRO_TOPOLOGY.e_core_peak_flops),
    peak_int_ops=(6 * _M3_PRO_TOPOLOGY.p_core_peak_int_ops
                  + 6 * _M3_PRO_TOPOLOGY.e_core_peak_int_ops),
    mem_bandwidth_bytes_s=116.5e9,             # aggregate STREAM Triad
    # Single-thread BW INHERITED from prior calibration (= agg × old memory_efficiency
    # 0.3104); not freshly measured on this Mac. Re-measure with STREAM to refine.
    mem_bandwidth_1t_bytes_s=36.16e9,
    simd_width_bits=128,                       # ARM NEON
    cache_hierarchy=[
        CacheLevel("L1d", 128 * 1024 * 6 + 64 * 1024 * 6, 800e9, 248.3e9, 1.0),
        CacheLevel("L2",  32 * 1024 * 1024 + 4 * 1024 * 1024, 400e9, 124.1e9, 4.0),
        CacheLevel("SLC", 12 * 1024 * 1024, 200e9, 62.07e9, 10.0),
    ],
    core_topology=_M3_PRO_TOPOLOGY,
)


# AMD EPYC 7313 (Milan, 2-socket, 32 cores, AVX2)
_EPYC_7313_BOOST_GHZ = 3.7
_EPYC_7313_FLOP_PER_CYCLE = 32                 # 2 FMA × AVX2 (8 FP32) × 2
_EPYC_7313_INT_PER_CYCLE = 16
EPYC_7313 = CPUSpec(
    name="AMD EPYC 7313 (2S, 32 cores, Zen 3 Milan)",
    num_cores=32,
    peak_flops=32 * _EPYC_7313_BOOST_GHZ * 1e9 * _EPYC_7313_FLOP_PER_CYCLE,
    peak_int_ops=32 * _EPYC_7313_BOOST_GHZ * 1e9 * _EPYC_7313_INT_PER_CYCLE,
    mem_bandwidth_bytes_s=120e9,               # aggregate NUMA-aware multi-thread peak
    # Single-thread BW INHERITED from prior calibration (= agg × old memory_efficiency
    # 0.8); not freshly measured on this box. Re-measure with STREAM to refine.
    mem_bandwidth_1t_bytes_s=96e9,
    simd_width_bits=256,
    cache_hierarchy=[
        CacheLevel("L1d", 32 * 1024 * 32, 46e9, 36.8e9, 1.0),
        CacheLevel("L2",  512 * 1024 * 32, 86e9, 68.8e9, 4.0),
        CacheLevel("L3",  32 * 1024 * 1024 * 8, 56e9, 44.8e9, 10.0),  # 8 CCDs
    ],
    # NUMA but homogeneous cores → only num_sockets is non-default.
    core_topology=CoreTopology(num_sockets=2),
)


# AMD EPYC 9124 dual-socket (Genoa, 2S × 16 cores, Zen 4, AVX-512)
#
# Per-core FLOP/cycle on Zen 4: although AVX-512 is supported natively,
# the 2× 256-bit FMA datapath is "double-pumped" for 512-bit instructions
# — effective throughput is 1× 512-bit FMA per cycle = 16 FP32 × 2 (FMA) =
# 32 FLOP/cycle/core, the same as AVX2's 2× 256-bit FMA. This number
# describes what FAISS-style workloads actually see, not the optimistic
# "2× full 512-bit FMA" figure that ignores double-pumping.
_EPYC_9124_ALL_CORE_BOOST_GHZ = 3.6
_EPYC_9124_FLOP_PER_CYCLE = 32                 # Zen 4 effective AVX-512 throughput
_EPYC_9124_INT_PER_CYCLE = 16                  # ~half FP, typical INT scaling

# Aligned to configs/rag_stack/sgs_local_vllm.yaml — both describe the same
# physical 2-socket EPYC 9124 deployment (lscpu: Socket(s)=2,
# Core(s) per socket=16, Thread(s) per core=1, AMD F19h M10h).
EPYC_9124 = CPUSpec(
    name="AMD EPYC 9124 (2S, 32 cores, Zen 4 Genoa, AVX-512)",
    num_cores=32,
    # These fields are all-core peaks, so use AMD's 3.6 GHz all-core boost,
    # not the 3.7 GHz maximum boost available to an individual core.
    peak_flops=32 * _EPYC_9124_ALL_CORE_BOOST_GHZ * 1e9 * _EPYC_9124_FLOP_PER_CYCLE,
    peak_int_ops=32 * _EPYC_9124_ALL_CORE_BOOST_GHZ * 1e9 * _EPYC_9124_INT_PER_CYCLE,
    # MEASURED on this box (STREAM Triad, node-local): all-32-core dual-socket
    # interleaved sustained = 308 GB/s; single-core local = 38 GB/s.
    mem_bandwidth_bytes_s=308e9,               # aggregate (measured dual-socket STREAM)
    mem_bandwidth_1t_bytes_s=38e9,             # single-thread (measured single-core STREAM)
    simd_width_bits=512,
    cache_hierarchy=[
        # PER-INSTANCE capacity + sharing (measured: lscpu -C / sysfs on this box):
        #   L1d 32KB private/core, L2 1MB private/core, L3 16MB shared by 4 cores
        #   (8 CCX × 16MB = 128MB total). So a SINGLE thread sees 16MB L3, NOT 128MB.
        # Single-thread BW per tier MEASURED via STREAM size-sweep (L1→207, L2→116,
        # L3→67). Both BW columns MUST decrease toward DDR or tier selection misroutes.
        CacheLevel("L1d", 32 * 1024,        2000e9, 207e9, 1.0,  cores_per_instance=1),
        CacheLevel("L2",  1024 * 1024,      1500e9, 116e9, 4.0,  cores_per_instance=1),
        CacheLevel("L3",  16 * 1024 * 1024, 1000e9,  67e9, 15.0, cores_per_instance=4),  # latency MEASURED (pointer-chase 4M): 15.1ns (was nominal 10)
    ],
    # 2 sockets — NUMA mode enabled via derived CoreTopology.is_numa
    # (= num_sockets > 1). numa_local_bytes / numa_remote_bw_ceiling
    # remain calibrator-tuned per workload.
    core_topology=CoreTopology(num_sockets=2),
)


# Intel Core Ultra 7 265K (Arrow Lake, desktop, 8P+16E, AVX2)
_ULTRA7_265K_TOPOLOGY = CoreTopology(
    num_p_cores=8,
    num_e_cores=16,
    p_core_peak_flops=5.5e9 * 32,              # AVX2: 2 FMA × 8 FP32 × 2 = 32 FLOP/cycle
    e_core_peak_flops=4.6e9 * 32,              # Skymont: 2× 256-bit SIMD = 32 FLOP/cycle
    p_core_peak_int_ops=5.5e9 * 16,
    e_core_peak_int_ops=4.6e9 * 16,
    p_core_clock_hz=5.5e9,
    e_core_clock_hz=4.6e9,
)

ULTRA7_265K = CPUSpec(
    name="Intel Core Ultra 7 265K (Arrow Lake, 8P+16E)",
    num_cores=24,                              # 8 P + 16 E (no SMT)
    peak_flops=(8 * _ULTRA7_265K_TOPOLOGY.p_core_peak_flops
                + 16 * _ULTRA7_265K_TOPOLOGY.e_core_peak_flops),
    peak_int_ops=(8 * _ULTRA7_265K_TOPOLOGY.p_core_peak_int_ops
                  + 16 * _ULTRA7_265K_TOPOLOGY.e_core_peak_int_ops),
    mem_bandwidth_bytes_s=70e9,                # aggregate DDR5-6400 × 2ch, multi-thread effective
    # Single-thread BW INHERITED from preset default (= agg × old memory_efficiency
    # 0.35); UNCALIBRATED — run STREAM + the IVF calibrator before trusting CM here.
    mem_bandwidth_1t_bytes_s=24.5e9,
    simd_width_bits=256,
    cache_hierarchy=[
        CacheLevel("L1d", 48 * 1024 * 8 + 32 * 1024 * 16, 600e9, 210e9, 1.0),
        CacheLevel("L2",  3 * 1024 * 1024 * 8 + 4 * 1024 * 1024 * 4, 300e9, 105e9, 4.0),
        CacheLevel("L3",  30 * 1024 * 1024, 150e9, 52.5e9, 12.0),
    ],
    core_topology=_ULTRA7_265K_TOPOLOGY,
)


# Registry: name → CPUSpec (hardcoded, known-good fallback)
CPU_SPECS: dict[str, CPUSpec] = {
    "m3_pro":      M3_PRO,
    "epyc_7313":   EPYC_7313,
    "epyc_9124":   EPYC_9124,
    "ultra7_265k": ULTRA7_265K,
}


# ── CPU spec resolution by key ───────────────────────────────────────────────
# Hardware facts are filled MANUALLY: add a CPUSpec to CPU_SPECS above (measure via
# STREAM / pointer-chase / lscpu + datasheet FLOPS), then reference it by key from a
# config (``cpu: epyc_9124``) or the faiss ``--system`` flag. A full spec can also be
# written inline in the YAML ``system.cpu`` block. Cost-model presets call
# ``get_spec(name)`` (not the constant) so a new machine is one dict entry here.


def get_spec(name: str) -> "CPUSpec":
    """Resolve a CPU preset key to its CPUSpec (the manual ``CPU_SPECS`` table).
    Backfills ``hardware_key`` from the registry key so every CPU carries its key."""
    spec = CPU_SPECS[name]
    if not spec.hardware_key:
        spec.hardware_key = name
    return spec
