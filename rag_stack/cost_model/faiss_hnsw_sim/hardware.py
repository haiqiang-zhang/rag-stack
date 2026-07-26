"""Hardware configuration for the FAISS HNSW cost simulator.

Lean HNSW-flavored ``SystemConfig`` and the roofline machinery the HNSW
cost model actually consumes — no PQ-specific fields, no per-probe
overheads, no random-access-penalty polynomials. HNSW models random graph
access with its own ``cache_random_access_latency_multiplier`` coefficient
(see :mod:`.cost_model`); it does not need IVF-PQ's RAP curves.

Pure physical CPU constants (peak FLOPS, cache hierarchy, core topology)
live in :mod:`rag_stack.cost_model.cpu_specs` and are pulled by
:mod:`.system_presets`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from rag_stack.cost_model.cpu_specs import CacheLevel, CoreTopology  # re-exported
from rag_stack.hnsw_profile_schema import (
    HNSW_HARDWARE_MODEL_PARAMETER_FIELDS,
)


# ---------------------------------------------------------------------------
# SystemConfig (HNSW flavor)
# ---------------------------------------------------------------------------

@dataclass
class SystemConfig:
    """Hardware spec used by the HNSW roofline model.

    Only the fields HNSW actually reads. See the module docstring for what's
    deliberately absent.
    """
    name: str
    num_cores: int
    peak_flops: float                              # all-cores FP32 FLOP/s
    mem_bandwidth_bytes_s: float                   # AGGREGATE DRAM B/s (all-core sustained STREAM)
    mem_bandwidth_1t_bytes_s: float                # SINGLE-THREAD DRAM B/s (one-core sustained STREAM)
    simd_width_bits: int = 256
    cache_hierarchy: list[CacheLevel] = field(default_factory=list)
    active_threads: int = 0                        # 0 → all cores

    # Low-level analytical defaults only. Production construction must pass all
    # five fields from an exact CPU profile (enforced by the global-config
    # converter); they are not portable algorithm defaults.
    # SIMD efficiency (FAISS's hand-tuned exhaustive_L2sqr_seq); the only
    # compute-mode HNSW ever asks for.
    simd_efficiency: float = 0.55
    # NOTE: no ``memory_efficiency`` — bandwidth is a hardware fact stored at two
    # explicit granularities (aggregate + single-thread); BW(t) = min(t*1t, agg).

    # Random-access (graph traversal) parameters.
    dram_random_access_latency_ns: float = 100.0
    scattered_bw_ceiling: float = 0.35
    numa_local_bytes: int = 0
    numa_remote_bw_ceiling: float = 0.06

    # Heterogeneous-core support.
    core_topology: Optional[CoreTopology] = None
    thread_migration: bool = True

    # ------------------------------------------------------------------
    # Thread-count properties
    # ------------------------------------------------------------------

    @property
    def omp_threads(self) -> int:
        """Actual OMP thread count (may include SMT)."""
        return self.active_threads if self.active_threads > 0 else self.num_cores

    @property
    def effective_num_threads(self) -> int:
        """Threads used for compute / bandwidth scaling (capped at physical cores)."""
        return min(self.omp_threads, self.num_cores)

    @property
    def effective_clock_hz(self) -> float:
        """Effective single-thread clock in Hz.

        Used to convert cycle-count estimates (e.g. per-hop graph overhead)
        into wall-clock time. Resolution order:

          1. ``core_topology.p_core_clock_hz`` if explicitly set.
          2. Derive from ``peak_flops`` assuming Intel/AMD-style SIMD FMA
             density (``simd_width_bits / 8`` FLOP/cycle).
          3. Fallback 3.0 GHz.
        """
        if self.core_topology is not None and self.core_topology.p_core_clock_hz > 0:
            return self.core_topology.p_core_clock_hz
        flops_per_cycle = max(1.0, self.simd_width_bits / 8.0)
        if self.core_topology is not None and self.core_topology.num_p_cores > 0:
            return self.core_topology.p_core_peak_flops / flops_per_cycle
        if self.num_cores > 0 and self.peak_flops > 0:
            return self.peak_flops / self.num_cores / flops_per_cycle
        return 3.0e9

    # ------------------------------------------------------------------
    # Roofline pieces (only what the HNSW cost model uses)
    # ------------------------------------------------------------------

    @staticmethod
    def _bw_at_threads(bw_1t: float, bw_aggregate: float, threads: int) -> float:
        """Effective bandwidth at ``threads`` (Universal Scalability Law).

        ``BW(t) = t*bw_1t / (1 + (t-1)*bw_1t/agg)``, clamped to ``agg``. Anchored
        at ``BW(1)=bw_1t`` exactly (preserves single-thread calibration) and
        saturates at the aggregate. Concave contention model — validated to beat
        plain ``min(t*bw_1t, agg)`` for mid-range threads on the IVF cost model
        (real FAISS BW scales sub-linearly). Both endpoints are explicit hardware
        facts → portable. Multi-thread itself is uncalibrated for HNSW.
        """
        t = max(1, threads)
        if bw_aggregate <= 0:
            return t * bw_1t
        usl = t * bw_1t / (1.0 + (t - 1) * bw_1t / bw_aggregate)
        return min(usl, bw_aggregate)

    def _compute_peak_ops(self, num_threads: int, partitioned: bool = False) -> float:
        """Effective peak FLOP/s for SIMD-bound work.

        Heterogeneous cores: aggregate (mode 1 / migration) vs slowest-thread
        bottleneck (mode 0 / pinned). Homogeneous: linear scaling.
        Always uses ``simd_efficiency`` — HNSW's only compute mode.
        """
        topo = self.core_topology
        if topo is not None and topo.is_heterogeneous:
            if self.thread_migration or not partitioned:
                raw_peak = topo.aggregate_peak(num_threads, mode="flops")
            else:
                raw_peak = topo.bottleneck_peak(num_threads, mode="flops")
        else:
            ct_scale = num_threads / self.num_cores if self.num_cores > 0 else 1.0
            raw_peak = self.peak_flops * ct_scale
        return raw_peak * self.simd_efficiency

    def effective_bandwidth(
        self,
        working_set_bytes: float,
        threads: Optional[int] = None,
    ) -> float:
        """Sequential effective bandwidth from the cache tier holding the working set.

        HNSW only uses sequential access at this entry point — the random-
        access (per-hop) cost is computed elsewhere with HNSW's own
        latency-multiplier coefficient, so there's no scattered-access
        branch here.
        """
        if threads is None:
            threads = self.effective_num_threads
        tier_bw = self.mem_bandwidth_bytes_s          # aggregate
        tier_bw_1t = self.mem_bandwidth_1t_bytes_s    # single-thread
        for cache in self.cache_hierarchy:
            if working_set_bytes <= cache.size_bytes:
                tier_bw = cache.bandwidth_bytes_s
                tier_bw_1t = cache.bandwidth_1t_bytes_s
                break
        return self._bw_at_threads(tier_bw_1t, tier_bw, threads)


# ---------------------------------------------------------------------------
# Global YAML SystemConfig → HNSW SystemConfig converter
# ---------------------------------------------------------------------------

def system_config_to_faiss_hardware(hw, **kwargs) -> SystemConfig:
    """Convert a global :class:`rag_stack.types.SystemConfig` (from YAML)
    into the HNSW-flavored :class:`SystemConfig`.

    Reads raw hardware facts from ``hw``. Every non-raw HNSW hardware-model
    parameter must be supplied explicitly (normally by the exact CPU profile);
    custom CPUs may not silently inherit the dataclass's generic defaults.
    """
    missing_parameters = sorted(
        set(HNSW_HARDWARE_MODEL_PARAMETER_FIELDS) - set(kwargs)
    )
    if missing_parameters:
        raise ValueError(
            "HNSW custom-CPU construction requires all exact-profile hardware "
            f"model parameters; missing {missing_parameters}"
        )
    if hw.mem_bandwidth_1t <= 0:
        raise ValueError(
            "single-thread DRAM bandwidth (cpu.mem_bandwidth_1t) is required for "
            "the cost model but was not provided. It is an INDEPENDENT hardware "
            "fact (per-core MLP/latency limited, ~constant per core) and is NOT "
            "derivable from aggregate mem_bandwidth, which scales with memory "
            "channels / core count / NUMA sockets. Measure it with STREAM at 1 "
            "thread and set cpu.mem_bandwidth_1t (+ each memory_tiers[].bandwidth_1t), "
            "or use a profiled cpu preset."
        )

    cache_hierarchy = [
        CacheLevel(
            name=tier.name,
            size_bytes=int(tier.capacity),
            bandwidth_bytes_s=tier.bandwidth,
            # Fall back to DRAM single-thread when a tier omits its own (cache 1t
            # ≥ DRAM 1t, a safe floor — never a ratio of the tier's aggregate).
            bandwidth_1t_bytes_s=(tier.bandwidth_1t if tier.bandwidth_1t > 0 else hw.mem_bandwidth_1t),
            latency_ns=10.0,
        )
        for tier in hw.memory_tiers
    ]

    core_topology = None
    if hw.topology_type == "numa":
        core_topology = CoreTopology(num_sockets=hw.numa_sockets)
    elif hw.num_p_cores > 0 and hw.p_core_clock_ghz > 0:
        p_clock = hw.p_core_clock_ghz * 1e9
        e_clock = hw.e_core_clock_ghz * 1e9
        core_topology = CoreTopology(
            num_p_cores=hw.num_p_cores,
            num_e_cores=hw.num_e_cores,
            p_core_peak_flops=p_clock * hw.p_core_flops_per_cycle,
            e_core_peak_flops=e_clock * hw.e_core_flops_per_cycle,
            p_core_peak_int_ops=p_clock * hw.p_core_int_ops_per_cycle,
            e_core_peak_int_ops=e_clock * hw.e_core_int_ops_per_cycle,
            p_core_clock_hz=p_clock,
            e_core_clock_hz=e_clock,
        )

    return SystemConfig(
        name="from_system_config",
        num_cores=hw.num_cores,
        peak_flops=hw.peak_flops,
        mem_bandwidth_bytes_s=hw.mem_bandwidth,
        mem_bandwidth_1t_bytes_s=hw.mem_bandwidth_1t,
        cache_hierarchy=cache_hierarchy,
        core_topology=core_topology,
        thread_migration=hw.thread_migration,
        **kwargs,
    )
