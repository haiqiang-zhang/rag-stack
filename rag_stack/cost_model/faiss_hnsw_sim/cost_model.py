"""HNSW cost model: maps workload profile to latency via roofline model.

Architecture:
    WorkloadPredictor (ML, hardware-agnostic)
        -> WorkloadProfile (ndis, nhops)
            -> HNSWCostModel (analytical roofline, hardware-specific)
                -> HNSWCostBreakdown (latency, throughput, per-phase)

Uses the lean SystemConfig from .hardware for hardware modeling.
No code duplication -- the same roofline_time() and effective_bandwidth()
methods handle both IVF-PQ and HNSW workloads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from .hardware import SystemConfig
from .hnsw_config import HNSWConfig
from .workload_predictor import WorkloadProfile


@dataclass
class HNSWCoefficients:
    """Tunable empirical graph-search coefficients of the HNSW cost model.

    These are fitted from a standalone FAISS HNSW measurement grid. CPU
    identity, effective hardware-model parameters, and fit diagnostics
    belong to the strict system profile rather than this numerical object.
    Defaults remain available only to low-level analytical tests; production
    always supplies the four values from an exact CPU-keyed profile.
    """

    # Greedy hops per upper layer (ef=1). No closed-form exists;
    # empirically ~2 for typical graphs.
    hops_per_upper_layer: float = 2.0

    # Cache-line access latency multiplier for random pointer chasing.
    # Preset latency_ns values reflect best-case (sequential / prefetchable)
    # access. HNSW defeats the prefetcher and pays full TLB + snoop-filter +
    # interconnect cost per access. Empirically ~3.5x on both Zen 3 and
    # Arrow Lake.
    cache_random_access_latency_multiplier: float = 3.5

    # Per-neighbor visited-set probe cost (CPU cycles). Each graph hop
    # iterates M0 neighbors and probes the hash-set of already-visited
    # nodes. Expressed in cycles so the coefficient is largely clock-agnostic.
    cycles_per_visited_probe: float = 40.0

    # Per-heap-op cost for the candidate / result priority queues (cycles).
    # Each hop performs O(log ef_search) heap ops.
    cycles_per_heap_op: float = 75.0


_DEFAULT_COEFFICIENTS = HNSWCoefficients()



@dataclass
class HNSWOperatorCost:
    """Cost breakdown for a single phase of HNSW search."""

    name: str
    total_ops: float = 0.0          # FLOPs
    total_bytes: float = 0.0        # Total bytes accessed
    access_block_bytes: float = 0.0  # Size of each random access block
    compute_time_s: float = 0.0
    memory_time_s: float = 0.0
    time_s: float = 0.0
    bottleneck: str = ""


@dataclass
class HNSWCostBreakdown:
    """Full cost breakdown for a batch of HNSW queries."""

    config: HNSWConfig
    workload: WorkloadProfile
    operators: list[HNSWOperatorCost] = field(default_factory=list)

    @property
    def total_time_s(self) -> float:
        return sum(op.time_s for op in self.operators)

    @property
    def latency_ms(self) -> float:
        return self.total_time_s * 1000

    @property
    def throughput_qps(self) -> float:
        if self.total_time_s > 0:
            return self.config.batch_size / self.total_time_s
        return float("inf")

    @property
    def overall_bottleneck(self) -> str:
        if not self.operators:
            return "N/A"
        slowest = max(self.operators, key=lambda op: op.time_s)
        return f"{slowest.name} ({slowest.bottleneck})"


class HNSWCostModel:
    """Analytical roofline cost model for HNSW search.

    Takes a WorkloadProfile (from ML predictor or analytical fallback)
    and maps it to latency using the hardware's roofline model.
    """

    def __init__(
        self,
        config: HNSWConfig,
        hardware: SystemConfig,
        workload: WorkloadProfile,
        coefficients: Optional[HNSWCoefficients] = None,
    ):
        self.cfg = config
        self.hw = hardware
        self.workload = workload
        self.coef = coefficients if coefficients is not None else _DEFAULT_COEFFICIENTS

    def estimate(self) -> HNSWCostBreakdown:
        """Compute latency for a batch of HNSW queries."""
        breakdown = HNSWCostBreakdown(
            config=self.cfg, workload=self.workload,
        )
        B = self.cfg.batch_size

        # Phase 1: Upper layer greedy descent
        breakdown.operators.append(self._upper_layer_traversal(B))

        # Phase 2: Base layer beam search (dominant cost)
        breakdown.operators.append(self._base_layer_search(B))

        # Phase 2b: Graph traversal overhead (visited-set + heap ops per hop)
        breakdown.operators.append(self._graph_overhead(B))

        # Phase 3: Heap selection (negligible)
        breakdown.operators.append(self._heap_selection(B))

        # Apply roofline to each phase.  FAISS HNSW parallelises a batch across
        # independent queries; it cannot keep more search threads busy than the
        # number of queries in this call.  This matches IndexIVF's inter-query
        # rule (and the runtime's OpenMP search call): nt = min(B, omp_threads).
        # Passing every configured thread for a partial batch invented compute
        # and memory parallelism that the runtime cannot use.
        threads = max(1, min(B, self.hw.effective_num_threads))
        for op in breakdown.operators:
            self._apply_roofline(op, B, threads)

        return breakdown

    def _upper_layer_traversal(self, B: int) -> HNSWOperatorCost:
        """Phase 1: Greedy descent through upper layers (layers L -> 1).

        Each layer: hops_per_layer greedy hops, each examining M neighbors.
        Each hop requires 2 random accesses (vector from data_level0_memory_
        + neighbor list from linkLists_, stored separately for upper layers).
        """
        num_layers = self.cfg.num_layers
        M = self.cfg.M
        d = self.cfg.d
        hops_per_layer = self.coef.hops_per_upper_layer

        total_hops = num_layers * hops_per_layer
        dist_comps = total_hops * M  # M neighbors checked per hop

        # FLOPs: each distance computation = 2*d (subtract + square, or FMA)
        total_ops = dist_comps * 2 * d * B

        # Memory: 2 random accesses per hop
        # Access 1: vector from data_level0_memory_ (d * 4 bytes)
        # Access 2: neighbor list from linkLists_ (M * 4 bytes)
        # Use the larger block size for the roofline model
        vector_bytes = d * 4
        neighbor_bytes = M * 4
        # Total bytes per hop = both accesses
        bytes_per_hop = vector_bytes + neighbor_bytes
        total_bytes = total_hops * bytes_per_hop * B

        return HNSWOperatorCost(
            name="1_upper_layer_traversal",
            total_ops=total_ops,
            total_bytes=total_bytes,
            access_block_bytes=max(vector_bytes, neighbor_bytes),
        )

    def _base_layer_search(self, B: int) -> HNSWOperatorCost:
        """Phase 2: Beam search at layer 0 (dominant cost).

        Uses ML-predicted workload profile for ndis and nhops.
        Each distance computation requires a random read of the candidate's
        vector from data_level0_memory_ (d * 4 bytes, within a node block
        of (M0 + 1 + d) * 4 bytes).  The total random accesses scale with
        ndis (distance computations), not nhops (node expansions).
        """
        ndis = self.workload.ndis_per_query
        d = self.cfg.d

        total_ops = ndis * 2 * d * B
        # Each distance computation reads a vector (d*4 bytes) from a random
        # node in data_level0_memory_.  The cache line / block accessed is
        # the full node block, but the useful payload is the vector.
        vector_bytes = d * 4
        block_bytes = self.cfg.node_block_bytes_layer0
        total_bytes = ndis * vector_bytes * B

        return HNSWOperatorCost(
            name="2_base_layer_search",
            total_ops=total_ops,
            total_bytes=total_bytes,
            access_block_bytes=float(block_bytes),
        )

    def _graph_overhead(self, B: int) -> HNSWOperatorCost:
        """Phase 2b: Per-hop visited-set and heap overhead.

        Each hop iterates M0 neighbors and probes the visited table, then
        performs heap ops on the candidate queue. This cost scales with
        nhops (not ndis), so it dominates on data distributions where
        ndis/nhops is low (e.g. clustered real data like SIFT).

        Modeled as a fixed per-hop time; bypasses the roofline model
        because it's instruction-dominated, not memory-bound.
        """
        nhops = self.workload.nhops_per_query
        M0 = self.cfg.M0
        ef = self.cfg.ef_search

        period_s = 1.0 / self.hw.effective_clock_hz
        cycles_per_hop = (
            self.coef.cycles_per_visited_probe * M0
            + self.coef.cycles_per_heap_op * math.log2(max(ef, 2))
        )
        # FAISS parallelises independent HNSW queries across OpenMP workers.
        # This instruction-only operator bypasses ``_apply_roofline`` (it has
        # no FLOPs/bytes), so apply the same inter-query width explicitly
        # instead of serialising all B queries in batch wall time.
        parallel_queries = max(
            1, min(B, self.hw.effective_num_threads),
        )
        time_s = (
            nhops * cycles_per_hop * period_s
            * B / parallel_queries
        )

        return HNSWOperatorCost(
            name="2b_graph_overhead",
            total_ops=0.0,
            total_bytes=0.0,
            compute_time_s=time_s,
            memory_time_s=0.0,
            time_s=time_s,
            bottleneck="instruction",
        )

    def _heap_selection(self, B: int) -> HNSWOperatorCost:
        """Phase 3: Final top-k selection from ef_search candidates.

        ef_search * log2(ef_search) comparisons; working set fits in L1.
        """
        ef = self.cfg.ef_search
        comparisons = ef * math.log2(max(ef, 2))
        total_ops = comparisons * B
        # Working set = ef_search * sizeof(pair<float,int>) ~ ef * 8 bytes
        total_bytes = ef * 8.0 * B

        return HNSWOperatorCost(
            name="3_heap_selection",
            total_ops=total_ops,
            total_bytes=total_bytes,
            access_block_bytes=0.0,  # Sequential / L1-resident
        )

    def _apply_roofline(
        self, op: HNSWOperatorCost, B: int, threads: int,
    ) -> None:
        """Apply roofline model to a single phase.

        HNSW has a unique access pattern compared to IVF-PQ:
        - Scattered random reads across the full index (working set)
        - Single-query search is serial pointer chasing (1 memory thread)
        - Within each hop, ~M0 neighbor reads are independent (pipelinable)
        - The index may fit in LLC, making effective latency << DRAM latency

        We compute memory time directly using the cache tier that matches the
        index working set, rather than using roofline_time's warm_cache path
        (which selects tier based on unique_bytes, not the actual working set).
        """
        if op.total_ops == 0 and op.total_bytes == 0:
            return

        abb = op.access_block_bytes if op.access_block_bytes > 0 else None
        index_ws = float(self.cfg.index_size_bytes)

        # ── Compute time ─────────────────────────────────────────────
        ct = min(threads, self.hw.num_cores)
        partitioned = (B > 1)
        peak_ops = self.hw._compute_peak_ops(ct, partitioned)
        t_compute = op.total_ops / peak_ops if peak_ops > 0 else float("inf")

        # ── Memory time ──────────────────────────────────────────────
        if abb is not None and abb > 0:
            # Scattered random access: determine effective latency and
            # bandwidth based on where the index sits in the cache hierarchy.
            #
            # When the index fits entirely in a cache tier, use that tier.
            # When it exceeds a tier, blend between the largest cache that
            # partially holds data and DRAM, weighted by hit ratio.
            # hit_ratio = cache_size / index_size (uniform random access).
            dram_latency_ns = self.hw.dram_random_access_latency_ns
            dram_bw = self.hw.mem_bandwidth_bytes_s             # aggregate
            dram_bw_1t = self.hw.mem_bandwidth_1t_bytes_s       # single-thread
            access_latency_ns = dram_latency_ns
            tier_bw = dram_bw
            tier_bw_1t = dram_bw_1t
            largest_partial_cache = None

            cmul = self.coef.cache_random_access_latency_multiplier

            for cache in self.hw.cache_hierarchy:
                if index_ws <= cache.size_bytes:
                    # Fits entirely in this cache tier
                    access_latency_ns = cache.latency_ns * cmul
                    tier_bw = cache.bandwidth_bytes_s
                    tier_bw_1t = cache.bandwidth_1t_bytes_s
                    largest_partial_cache = None
                    break
                elif cache.size_bytes > 0:
                    largest_partial_cache = cache

            if largest_partial_cache is not None:
                # Index exceeds all caches. Blend the largest cache with DRAM.
                hit_ratio = largest_partial_cache.size_bytes / index_ws
                cache_lat = largest_partial_cache.latency_ns * cmul
                access_latency_ns = (
                    hit_ratio * cache_lat
                    + (1 - hit_ratio) * dram_latency_ns
                )
                tier_bw = (
                    hit_ratio * largest_partial_cache.bandwidth_bytes_s
                    + (1 - hit_ratio) * dram_bw
                )
                tier_bw_1t = (
                    hit_ratio * largest_partial_cache.bandwidth_1t_bytes_s
                    + (1 - hit_ratio) * dram_bw_1t
                )

            latency_s = access_latency_ns * 1e-9
            # Single-thread tier bandwidth (no memory_efficiency fudge).
            peak_bw_1t = tier_bw_1t

            # Per-block bandwidth: bw = block / (latency + block / peak_bw)
            bw_1t = abb / (latency_s + abb / peak_bw_1t)

            # Thread scaling: for batch_size=1, HNSW is single-threaded
            # pointer chasing.  For batch_size>1, queries are independent.
            mem_threads = threads if B > 1 else 1

            # Apply ceiling (DRAM controller / NUMA limits)
            if self.hw.numa_local_bytes > 0 and index_ws > self.hw.numa_local_bytes:
                ceiling = self.hw.mem_bandwidth_bytes_s * self.hw.numa_remote_bw_ceiling
            else:
                ceiling = self.hw.mem_bandwidth_bytes_s * self.hw.scattered_bw_ceiling
            eff_bw = min(bw_1t * mem_threads, ceiling)

            t_memory = op.total_bytes / eff_bw if eff_bw > 0 else float("inf")
        else:
            # Sequential / cache-resident (e.g. heap selection)
            eff_bw = self.hw.effective_bandwidth(index_ws, threads=threads)
            t_memory = op.total_bytes / eff_bw if eff_bw > 0 else float("inf")

        op.compute_time_s = t_compute
        op.memory_time_s = t_memory
        op.time_s = max(t_compute, t_memory)
        op.bottleneck = "compute" if t_compute >= t_memory else "memory"
