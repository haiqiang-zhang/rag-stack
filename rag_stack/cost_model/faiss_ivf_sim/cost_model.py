"""
Faiss IVF-PQ Cost Model

Breaks down a single IVF-PQ query into discrete operators, computes
FLOPs, bytes transferred, and applies the roofline model to obtain
latency for each operator.

IVF-PQ Search Pipeline (per query):
═══════════════════════════════════════════════════════════════════
   ┌───────────────┐
   │ 1. Coarse     │  Compute query-to-centroid distances
   │    Quantizer  │  to find top-nprobe partitions.
   └───────┬───────┘
           │
   ┌───────▼───────┐
   │  2. LUT       │  Build per-query distance lookup table
   │     Build     │  (M × ksub entries) for PQ sub-spaces.
   └───────┬───────┘
           │
   ┌───────▼───────┐
   │  3. PQ Scan   │  Scan PQ codes in selected partitions,
   │     & ADC     │  accumulate sub-distances via table lookups.
   └───────┬───────┘
           │
   ┌───────▼───────┐
   │  4. Heap      │  Maintain top-k candidates via max-heap.
   │     Selectio  │
   └───────┬───────┘
           │
   ┌───────▼───────┐  (optional)
   │  5. Refine    │  Re-rank top candidates with exact distances
   │     / Rerank  │  using original or SQ-compressed vectors.
   └───────────────┘
═══════════════════════════════════════════════════════════════════

Each operator is modeled via the roofline:
    T_op = max(T_compute, T_memory)
    T_compute = total_ops / peak_throughput
    T_memory  = total_bytes / effective_bandwidth
"""

from dataclasses import dataclass, field
from typing import Optional
import math

from .systems import SystemConfig
from .ivfpq_config import IVFPQConfig
# SIMD coefficients available for future calibration, not applied by default
# from .simd_model import get_simd_coefficients, SIMDCoefficients

# ══════════════════════════════════════════════════════════════════════════════
# FAISS BLAS Configuration Constants
# From faiss/utils/distances.cpp - these are FAISS global variables that control
# the BLAS-based distance computation tiling strategy.
# ══════════════════════════════════════════════════════════════════════════════
# FastScan (nbits=4) QBS query-block amortization (see _pq_scan): effective
# per-query scan cost ramps from full (B=1) down to *_FLOOR by batch *_SAT.
_FASTSCAN_QBS_FLOOR = 0.65
_FASTSCAN_QBS_SAT = 128

FAISS_BLAS_THRESHOLD = 20        # nq >= 20 triggers BLAS path (distance_compute_blas_threshold)
FAISS_BLAS_QUERY_BS = 4096       # Query block size (distance_compute_blas_query_bs)
FAISS_BLAS_DATABASE_BS = 1024    # Database/centroid block size (distance_compute_blas_database_bs)


@dataclass
class OperatorCost:
    """Cost breakdown for a single operator in the IVF-PQ pipeline.

    Memory model:
    - unique_bytes: Data loaded from DRAM exactly once (compulsory misses)
    - total_bytes: Total memory accesses including cache reuse
    - reuse_bytes = total_bytes - unique_bytes (served from cache)

    Time model:
    - dram_time_s: Time to load unique_bytes from DRAM
    - cache_time_s: Time for cache-resident reuse accesses
    - memory_time_s = dram_time_s + cache_time_s
    """
    name: str
    total_ops: float          # Total operations (FLOPs or INT-OPs)
    total_bytes: float        # Total bytes accessed (including cache reuse)
    unique_bytes: float       # Unique bytes from DRAM (compulsory misses)
    working_set_bytes: float  # Working set size (for cache tier selection)
    # Compute mode determines which peak throughput + efficiency to use:
    #   "blas"    - BLAS GEMM: peak_flops × blas_efficiency
    #   "simd"    - hand-optimized SIMD (FP): peak_flops × simd_efficiency
    #   "int"     - generic integer: raw peak_int_ops (no fitted fudge)
    #   "pq_scan_fastscan" - FastScan PQ (vpshufb): peak_int_ops × pq_fastscan_efficiency
    #   "pq_scan_standard" - Standard PQ (table lookup): peak_int_ops × pq_standard_efficiency
    compute_mode: str = "int"
    # Throughput GAIN on the compute ceiling (denominator), ≥1. FastScan's QBS
    # query-block amortization is a RATE effect (more vectors/cycle when the batch
    # fills), so it lives here on peak_ops — NOT on total_ops, which must stay the
    # pure algorithmic count so operational_intensity is the true (B-invariant)
    # arithmetic intensity. 1.0 = no gain.
    compute_peak_scale: float = 1.0
    # FastScan-specific: >0 routes this op's unique-byte (DRAM) time through a
    # calibrated fraction of STREAM (cold scattered scan across inverted lists),
    # bypassing the optimistic warm-cache tier. 0 = generic path. See SystemConfig.
    scan_bw_ceiling: float = 0.0
    access_block_bytes: float = 0.0  # >0 for scattered-sequential access (e.g. PQ scan across inverted lists)
    compute_time_s: float = 0.0
    memory_time_s: float = 0.0
    dram_time_s: float = 0.0  # Time for DRAM load (compulsory misses)
    cache_time_s: float = 0.0 # Time for cache reuse
    time_s: float = 0.0
    bottleneck: str = ""
    operational_intensity: float = 0.0


@dataclass
class QueryCostBreakdown:
    """Full cost breakdown for a batch of IVF-PQ queries."""
    config: IVFPQConfig
    hardware: SystemConfig
    operators: list[OperatorCost] = field(default_factory=list)

    @property
    def total_time_s(self) -> float:
        """Total time including overhead.

        Overhead components:
        - fastscan/pq_standard_per_probe_overhead_us: per query×nprobe visit (set_list, cache cold
          miss per inverted list). At bs=1 this dominates; at bs=1000 it's
          amortized by prefetch_lists() and batch processing.
        """
        ops_time = sum(op.time_s for op in self.operators)
        B = self.config.batch_size
        # Per-probe overhead: B × nprobe visits, each with set_list + cold miss.
        # At large B, prefetch_lists() and batch processing amortize this,
        # so scale down by sqrt(B) to model diminishing per-visit cost.
        probe_visits = B * self.config.nprobe
        if self.config.index_type == "flat":
            # IVF-Flat scans each query's probed lists in a single-thread
            # per-query loop — there is no cross-query list batching (the
            # prefetch_lists / FastScan-QBS effect that motivates sqrt(B)
            # amortization for PQ). So per-list-visit overhead scales
            # linearly with B (validated against real bs=1..1000 on epyc_9124).
            amortization = 1.0
        elif self.config.nbits == 4:
            # FastScan: per-(query,probe) overhead is mostly per-query — measured
            # per-query latency only drops ~1.5× from B=1→64, far less than sqrt(B)'s
            # 8×. Use a fittable exponent (default 0.5 = old sqrt-B; calibrates ~0.1-0.2).
            amortization = B ** getattr(self.hardware, "pq_fastscan_overhead_alpha", 0.5)
        else:
            amortization = B ** 0.5  # sqrt(B): 1 at bs=1, 31.6 at bs=1000
        if self.config.index_type == "flat":
            per_probe_us = self.hardware.flat_per_probe_overhead_us
        elif self.config.nbits == 4:
            per_probe_us = self.hardware.pq_fastscan_per_probe_overhead_us
        else:
            per_probe_us = self.hardware.pq_standard_per_probe_overhead_us
        # Per-(query,probe) visits are real WORK that parallelizes across the
        # mode-0 query workers. Divide by that
        # parallel width ``p``. This is a no-op at the bs=1 / 1-thread calibration
        # point (p=1) so it does NOT change the trusted regime, but it removes the
        # spurious "overhead floor" that made batched / multi-thread predictions
        # far too slow (e.g. flat per-probe was 5 ms at bs=32 regardless of cores).
        omp = self.hardware.omp_threads
        p = max(1, min(B, omp))
        probe_overhead_s = (probe_visits / amortization * per_probe_us * 1e-6) / p
        # Per-BATCH search overhead: faiss search() call setup + per-query heap
        # init / result reorder. The measured latency intercept at bs=1 that
        # dominates the nprobe=1 baseline. This is a FAISS-call cost shared by
        # ALL index types (flat / PQ / FastScan all go through IndexIVF::search),
        # so it is applied uniformly — modeling it only for flat left PQ &
        # FastScan systematically under-predicting at low nprobe by ~the call
        # cost (validated: real np=1 latency ≈ this intercept + tiny scan).
        # Modeled per-batch (added once, not ×B): the per-query component is
        # negligible vs the scan at large batch, so ×B would overcount there.
        batch_search_overhead_s = self.hardware.flat_fixed_overhead_us * 1e-6
        # PQ builds a per-query distance LUT whose query-transform + init cost
        # grows with the vector dim d: the measured per-call floor is ~5µs at
        # d=128 but ~24µs at d~1000 (gist d=960 AND dragonball d=1024 alike — it
        # is d-driven, not corpus-size). A constant under-predicts low-nprobe PQ
        # on high-d embeddings / tiny corpora; this d-aware term reduces to ~the
        # constant at small d so the in-regime (d≤384) fit is preserved. PQ-only
        # (IVF-Flat has no LUT — its floor stays flat_fixed_overhead_us).
        if self.config.index_type != "flat":
            batch_search_overhead_s += (
                self.hardware.pq_setup_per_dim_us * self.config.d * 1e-6)
        return ops_time + probe_overhead_s + batch_search_overhead_s

    @property
    def total_ops(self) -> float:
        return sum(op.total_ops for op in self.operators)

    @property
    def total_bytes(self) -> float:
        return sum(op.total_bytes for op in self.operators)

    @property
    def total_unique_bytes(self) -> float:
        """Total unique bytes loaded from DRAM (compulsory misses)."""
        return sum(op.unique_bytes for op in self.operators)

    @property
    def total_dram_time_s(self) -> float:
        """Total time spent loading from DRAM."""
        return sum(op.dram_time_s for op in self.operators)

    @property
    def total_cache_time_s(self) -> float:
        """Total time spent on cache-resident accesses."""
        return sum(op.cache_time_s for op in self.operators)

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


class IVFPQCostModel:
    """
    Analytical cost model for Faiss IVF-PQ search.

    Models each pipeline stage as an operator with compute + memory costs,
    then applies the roofline model per operator.
    """

    def __init__(self, config: IVFPQConfig, hardware: SystemConfig, warm_cache: bool = False):
        """
        Initialize the cost model.

        Args:
            config: IVF-PQ index configuration.
            hardware: Hardware configuration for roofline model.
            warm_cache: If True, model steady-state behavior where frequently accessed
                data (like centroids) stays in cache between queries. Use for high-QPS
                throughput benchmarks with warmup. If False (default), model cold-start
                behavior where unique_bytes always come from DRAM. Use for typical RAG
                latency modeling where queries are sparse.
        """
        self.cfg = config
        self.hw = hardware
        self.warm_cache = warm_cache

    def estimate(self) -> QueryCostBreakdown:
        """
        Estimate the total cost of a batch of queries.

        Returns a QueryCostBreakdown with per-operator and aggregate metrics.
        """
        breakdown = QueryCostBreakdown(config=self.cfg, hardware=self.hw)
        B = self.cfg.batch_size

        # Stage 1: Coarse quantizer
        # Determine BLAS vs SIMD path for coarse quantizer distance computation.
        # In mode 0 (inter-query), each thread processes ceil(B/threads) queries
        # independently. If that per-thread batch < BLAS threshold, FAISS uses
        # the SIMD path instead of BLAS.
        use_blas = B >= FAISS_BLAS_THRESHOLD
        # FAISS IndexIVF::search() mode 0 (IndexIVF.cpp:356-361).
        omp_threads = self.hw.omp_threads
        eff_par = min(B, omp_threads) if min(B, omp_threads) > 1 else 1
        per_thread_batch = math.ceil(B / eff_par) if eff_par > 0 else B
        use_blas = per_thread_batch >= FAISS_BLAS_THRESHOLD
        breakdown.operators.append(self._coarse_quantizer(B, use_blas=use_blas))

        if self.cfg.index_type == "flat":
            # IVF-Flat: a single exact full-vector scan replaces the PQ
            # LUT build + PQ scan, and there is no refinement (the scan is
            # already exact). Coarse quantizer (above) and heap (below) are
            # shared with IVF-PQ verbatim.
            breakdown.operators.append(self._flat_scan(B))
            breakdown.operators.append(self._heap_selection(B))
        else:
            # IVF-PQ pipeline (unchanged).
            # Stage 2: LUT construction
            breakdown.operators.append(self._lut_build(B))

            # Stage 3: PQ scan (ADC)
            breakdown.operators.append(self._pq_scan(B))

            # Stage 4: Heap / top-k selection
            breakdown.operators.append(self._heap_selection(B))

            # Stage 5: Refinement (optional)
            if self.cfg.refine_ratio > 0:
                breakdown.operators.append(self._refinement(B))

        # Apply the calibrated inter-query (mode 0) roofline.
        for op in breakdown.operators:
            self._apply_roofline(op, B)

        return breakdown


    def _apply_roofline(self, op: OperatorCost, B: int) -> None:
        """Apply the calibrated inter-query (mode 0) roofline.

        Compute threads: capped at physical cores (SMT shares ALU).
        Memory threads: all OMP threads (SMT provides extra MLP).

        Work is partitioned over ``min(B, omp_threads)`` query workers.
        """
        omp_threads = self.hw.omp_threads
        num_cores = self.hw.num_cores
        omp_par = min(B, omp_threads) if min(B, omp_threads) > 1 else 1
        ct = min(omp_par, num_cores)
        mt = omp_par

        result = self.hw.roofline_time(
            total_ops=op.total_ops,
            total_bytes=op.total_bytes,
            working_set_bytes=op.working_set_bytes,
            unique_bytes=op.unique_bytes,
            batch_size=B,
            warm_cache=self.warm_cache,
            compute_mode=op.compute_mode,
            compute_peak_scale=op.compute_peak_scale,
            scan_bw_ceiling=op.scan_bw_ceiling,
            access_block_bytes=op.access_block_bytes or None,
            compute_threads=ct,
            memory_threads=mt,
            partitioned=True,
        )

        op.compute_time_s = result["compute_time_s"]
        op.memory_time_s = result["memory_time_s"]
        op.dram_time_s = result["dram_time_s"]
        op.cache_time_s = result["cache_time_s"]
        op.time_s = result["time_s"]
        op.bottleneck = result["bottleneck"]
        op.operational_intensity = result["operational_intensity"]

    # ── Stage 1: Coarse Quantizer ─────────────────────────────────────────

    def _coarse_quantizer(self, B: int, use_blas: bool = True) -> OperatorCost:
        """
        Compute distances from each query to all nlist centroids and select top-nprobe.

        FAISS uses optimized L2 distance computation:
            ||q - c||² = ||q||² + ||c||² - 2<q,c>

        Where:
        - ||c||² is precomputed and stored (nlist × 4 bytes)
        - ||q||² is computed once per query (B × d FLOPs)
        - <q,c> is computed via BLAS GEMM (B × nlist × 2d FLOPs) - this dominates

        FAISS has two code paths (see faiss/utils/distances.cpp):

        1. BLAS path (B >= 20, controlled by distance_compute_blas_threshold):
           - Uses GEMM with tiled computation
           - Query block size: 4096 (distance_compute_blas_query_bs)
           - Database block size: 1024 (distance_compute_blas_database_bs)
           - High efficiency via MKL/Accelerate/OpenBLAS
           - Note: BLAS is a hard requirement for FAISS compilation

        2. Sequential SIMD path (B < 20):
           - Direct nested loop with SIMD (exhaustive_L2sqr_seq)
           - No tiling, processes one query at a time
           - Lower overhead for small batches

        Additionally, selecting top-nprobe centroids requires partial sorting:
        - Using heap-based selection: O(nlist × log(nprobe)) comparisons per query
        """
        cfg = self.cfg

        # FLOPs breakdown:
        # 1. Query norms ||q||²: B × d FLOPs (negligible)
        query_norm_ops = B * cfg.d

        # 2. Inner products <q,c> via GEMM: B × nlist × 2d FLOPs
        #    This is the dominant computation and uses highly optimized BLAS
        gemm_ops = B * cfg.nlist * 2 * cfg.d

        # 3. Combine: ||q||² + ||c||² - 2<q,c> → B × nlist × 2 FLOPs (add + sub)
        combine_ops = B * cfg.nlist * 2

        # 4. Top-nprobe selection via heap: ~B × nlist × log2(nprobe) comparisons
        #    This is integer work but relatively fast
        topk_ops = B * cfg.nlist * max(1, math.log2(cfg.nprobe)) if cfg.nprobe > 0 else 0

        # Total FLOPs (GEMM dominates, uses BLAS efficiency)
        total_ops = query_norm_ops + gemm_ops + combine_ops + topk_ops

        # Bytes:
        # - Read queries: B × d × 4
        # - Read centroids: nlist × d × 4 (reused across batch)
        # - Read precomputed centroid norms: nlist × 4
        # - Write distances (temporary): B × nlist × 4
        query_bytes = B * cfg.d * 4
        centroid_bytes = cfg.nlist * cfg.d * 4
        centroid_norms_bytes = cfg.nlist * 4
        distance_output_bytes = B * cfg.nlist * 4

        # Unique bytes: data loaded from DRAM exactly once (no reuse counted)
        # - Queries: each loaded once
        # - Centroids: loaded once, reused B times
        # - Centroid norms: loaded once, reused B times
        # - Output distances: written once
        unique_bytes = query_bytes + centroid_bytes + centroid_norms_bytes + distance_output_bytes

        # Working set calculation (needed for both cache tier selection and total_bytes)
        if use_blas:
            # BLAS path: tiled GEMM computation
            # Working set = centroid tile + query tile + output tile + norms
            #
            # FAISS processes in blocks:
            # - Query block: min(B, 4096) queries at a time
            # - Database block: min(nlist, 1024) centroids at a time
            query_tile_size = min(B, FAISS_BLAS_QUERY_BS)
            db_tile_size = min(cfg.nlist, FAISS_BLAS_DATABASE_BS)

            ws = (
                db_tile_size * cfg.d * 4 +           # Centroid tile
                centroid_norms_bytes +                # All centroid norms (small, kept in cache)
                query_tile_size * cfg.d * 4 +        # Query tile
                query_tile_size * db_tile_size * 4   # Output tile (distances)
            )
        else:
            # Non-BLAS path: direct computation without tiling
            # Processes queries sequentially, needs all centroids in memory
            # Working set = all centroids + single query + output for one query
            ws = (
                centroid_bytes +          # All centroids (no tiling)
                centroid_norms_bytes +    # All centroid norms
                cfg.d * 4 +               # Single query vector
                cfg.nlist * 4             # Distances for one query
            )

        # Total bytes: models effective memory traffic considering cache reuse
        #
        # Both BLAS and SIMD aim to reuse centroids across queries.
        # The key question: does the working set fit in cache?
        # - If yes: data loaded once from DRAM, reused from cache
        # - If no: data may need to be reloaded from DRAM
        #
        # We use the hardware's cache hierarchy to determine reuse behavior.

        # Get largest cache size from hardware config
        largest_cache_bytes = 0
        if self.hw.cache_hierarchy:
            largest_cache_bytes = max(c.size_bytes for c in self.hw.cache_hierarchy)

        if use_blas:
            # BLAS path: tiled GEMM computation
            #
            # Tiling ensures each tile's working set fits in cache.
            # At tile boundaries, data needs to be reloaded.
            # Model: centroids reloaded per query tile, queries per db tile.
            num_query_tiles = math.ceil(B / FAISS_BLAS_QUERY_BS)
            num_db_tiles = math.ceil(cfg.nlist / FAISS_BLAS_DATABASE_BS)

            # Tile boundary overhead as reuse traffic (served from cache)
            total_bytes = (query_bytes * num_db_tiles +
                          centroid_bytes * num_query_tiles +
                          centroid_norms_bytes +
                          distance_output_bytes)
        else:
            # SIMD path: sequential processing without explicit tiling
            #
            # Cache behavior depends on whether working set fits in cache:
            # - If ws <= cache: centroids loaded once, reused B times
            # - If ws > cache: centroids may be evicted between queries

            if largest_cache_bytes > 0 and ws > largest_cache_bytes:
                # Working set exceeds largest cache level
                #
                # When ws = k × cache (k > 1), approximately (k-1)/k of data
                # is evicted after processing each query.
                #
                # For B queries:
                # - Query 1: Load all centroids (cold miss)
                # - Query 2 to B: Each reloads (k-1)/k fraction
                #
                # Total: reload_factor = 1 + (B-1) × (k-1)/k
                cache_overflow_ratio = ws / largest_cache_bytes
                eviction_fraction = (cache_overflow_ratio - 1) / cache_overflow_ratio
                reload_factor = min(B, 1 + (B - 1) * eviction_fraction)
                total_bytes = (query_bytes +
                              centroid_bytes * reload_factor +
                              centroid_norms_bytes * reload_factor +
                              distance_output_bytes)
            else:
                # Working set fits in cache, good reuse across queries
                total_bytes = unique_bytes

        return OperatorCost(
            name="1_coarse_quantizer",
            total_ops=total_ops,
            total_bytes=total_bytes,
            unique_bytes=unique_bytes,
            working_set_bytes=ws,
            compute_mode="blas" if use_blas else "simd",
        )

    # ── Stage 2: LUT Construction ─────────────────────────────────────────

    def _lut_build(self, B: int) -> OperatorCost:
        """
        Build the distance lookup table (LUT) for PQ sub-spaces.

        For each query and each probed partition (when using residuals):
          - Compute residual: query - centroid[list_no] (d FLOPs)
          - For each sub-quantizer m (0..M-1), compute distance from
            query sub-vector to all ksub PQ centroids:
            FLOPs = M × ksub × 2 × dsub

        With precomputed tables (Faiss optimization):
          - The nlist × M × ksub table is precomputed at index time.
          - At query time, only need to add query-dependent terms.
          - FLOPs ≈ B × nprobe × M × ksub × (2 × dsub) if no precomputed table
          - FLOPs ≈ B × M × ksub × (2 × dsub)          if precomputed table (once per query)

        Memory: Load PQ codebook (M × ksub × dsub × 4) +
                Write LUT (B × nprobe × M × ksub × 4) [or B × M × ksub × 4 w/ precomp].
        """
        cfg = self.cfg

        if cfg.nbits == 4:
            # ── FastScan (nbits=4): compact uint8 LUT built ONCE per query ──
            # IVFPQFastScan computes the per-query distance LUT (compute_LUT_uint8)
            # from the query subvectors and REUSES it across ALL probed lists — it
            # is NOT recomputed per probe. ``by_residual`` only changes the query
            # transform (residual vs raw subvector), not the per-list LUT cost.
            # ksub = 16 (4-bit). (The old "no precomputed table" branch wrongly
            # recomputed the full M×ksub×dsub table per probe — ∝ nprobe·dsub —
            # which over-predicted high-nprobe / high-d FastScan LUT build ~50×,
            # e.g. gist960 PQ16x4fs np128: 110 µs vs a true ~2 µs.)
            lut_bytes_per_partition = cfg.M * 16
            lut_compute_ops = B * cfg.M * cfg.ksub * 2 * cfg.dsub   # once per query
            combine_ops = B * cfg.nprobe * cfg.M * cfg.ksub          # per-list uint8 apply
            total_ops = lut_compute_ops + combine_ops

            codebook_bytes = cfg.M * cfg.ksub * cfg.dsub * 4
            lut_write = B * cfg.nprobe * lut_bytes_per_partition
            query_bytes = B * cfg.d * 4
            nprobe_ratio = cfg.nprobe / cfg.nlist

            # Only FAISS-default non-residual FastScan is calibrated. The
            # configuration object rejects residual FastScan before pricing.
            precomp_read = 0.0
            precomp_read_effective = 0.0
            ws = codebook_bytes + cfg.M * 16

            unique_bytes = codebook_bytes + precomp_read + lut_write + query_bytes
            total_bytes = codebook_bytes + precomp_read_effective + lut_write + query_bytes
        elif cfg.use_precomputed_table and cfg.by_residual:
            # ── Standard PQ (nbits=8) with precomputed residual tables ──
            # The nlist × M × ksub table is precomputed at index time; at query
            # time only the query-dependent term is computed, then combined with
            # the precomputed table per probed list (random access → cache thrash
            # at large nprobe).
            lut_bytes_per_partition = cfg.M * cfg.ksub * 4
            precomp_per_partition = cfg.M * cfg.ksub * 4

            lut_compute_ops = B * cfg.M * cfg.ksub * 2 * cfg.dsub
            combine_ops = B * cfg.nprobe * cfg.M * cfg.ksub
            total_ops = lut_compute_ops + combine_ops

            precomp_table_total = cfg.nlist * cfg.M * cfg.ksub * 4

            codebook_bytes = cfg.M * cfg.ksub * cfg.dsub * 4
            precomp_read = B * cfg.nprobe * precomp_per_partition

            # Random access penalty: SUPERLINEAR growth with nprobe (scattered
            # access to the precomputed table causes cache thrashing / TLB misses).
            nprobe_ratio = cfg.nprobe / cfg.nlist
            random_access_factor = self.hw.random_access_penalty_standard.compute(
                nprobe_ratio
            )
            precomp_read_effective = precomp_read * random_access_factor

            lut_write = B * cfg.nprobe * lut_bytes_per_partition
            query_bytes = B * cfg.d * 4

            unique_bytes = codebook_bytes + precomp_read + lut_write + query_bytes
            total_bytes = codebook_bytes + precomp_read_effective + lut_write + query_bytes

            ws = min(precomp_table_total, precomp_per_partition * cfg.nprobe * (1 + nprobe_ratio))
        else:
            # No precomputed table: compute from scratch per (query, partition)
            # Residual computation: B × nprobe × d FLOPs
            residual_ops = B * cfg.nprobe * cfg.d if cfg.by_residual else 0

            # Distance computation: B × nprobe × M × ksub × 2 × dsub
            dist_ops = B * cfg.nprobe * cfg.M * cfg.ksub * 2 * cfg.dsub
            total_ops = residual_ops + dist_ops

            # Bytes
            codebook_bytes = cfg.M * cfg.ksub * cfg.dsub * 4
            centroid_bytes = cfg.nprobe * cfg.d * 4  # centroids reused across batch
            query_bytes = B * cfg.d * 4
            lut_write = B * cfg.nprobe * cfg.M * cfg.ksub * 4

            # Unique bytes: codebook + centroids (once) + queries + LUT output
            unique_bytes = codebook_bytes + centroid_bytes + query_bytes + lut_write

            # Total bytes: centroids accessed B times, queries accessed nprobe times
            # With good caching, reuse happens in cache
            total_bytes = codebook_bytes + centroid_bytes * B + query_bytes + lut_write

            ws = codebook_bytes + cfg.nprobe * cfg.d * 4

        return OperatorCost(
            name="2_lut_build",
            total_ops=total_ops,
            total_bytes=total_bytes,
            unique_bytes=unique_bytes,
            working_set_bytes=ws,
            compute_mode="simd",
        )

    # ── Stage 3: PQ Scan (Asymmetric Distance Computation) ────────────────

    def _pq_scan(self, B: int) -> OperatorCost:
        """
        Scan PQ codes in probed partitions and compute approximate distances.

        This is the core IVF-PQ operation: for each vector in probed lists,
        decode M PQ codes and sum up M looked-up distances.

        Faiss standard (nbits=8):
          - Per vector: M table lookups + (M-1) additions = ~2M INT-OPs
          - Total: B × n_scanned × 2M INT-OPs

        Faiss FastScan (nbits=4):
          - Uses SIMD shuffle (pshufb) to perform 32 lookups simultaneously.
          - Per 32 vectors: M/2 SIMD ops → much higher throughput.
          - We model this as reduced effective ops per vector.

        Memory:
          - Read PQ codes: B × n_scanned × pq_code_bytes
          - Read LUT: B × nprobe × M × ksub × 4 (may be cached if small)
          - Working set: LUT per query (M × ksub × 4) + codes streaming

        For by_residual=True with precomputed tables:
          - Additional memory traffic for precomputed table access
          - Random access pattern causes cache thrashing at large nprobe
          - This is the PRIMARY source of superlinear slowdown

        This stage is typically the dominant cost and is often MEMORY-BOUND
        for Faiss (unlike ScaNN which uses lower-precision PQ and is more
        compute-oriented).
        """
        cfg = self.cfg
        n_scanned = cfg.vectors_scanned

        if cfg.nbits == 4:
            # FastScan mode: SIMD-optimized, processes 32 vectors at a time
            #
            # From FAISS source (pq4_kernel_qbs_256, kernels_simd256.h):
            # Per sq iteration (2 sub-quantizers) per query:
            #   1 vmovdqu (LUT load) + 2 vpshufb (lookup) +
            #   4 vpaddw (accumulate) + 2 vpsrlw (shift) = 9 SIMD ops
            # Plus shared per iteration: 1 load + 1 shift + 2 AND = 4 ops
            #
            # Effective ops per vector: M (lookup + add for each sub-quantizer)
            # Overhead (loads, shifts, masks) is captured by pq_fastscan_efficiency
            # in HardwareConfig, not by inflating ops_per_vector.
            ops_per_vector = cfg.M
            # LUT is smaller (16 entries per sub-quantizer, fits in SIMD reg)
            lut_bytes_per_query = cfg.M * 16  # 16 entries × 1 byte (uint8)
        elif cfg.nbits == 8:
            # Standard ADC: M lookups + (M-1) additions per vector
            ops_per_vector = 2 * cfg.M
            lut_bytes_per_query = cfg.lut_size_bytes  # M × 256 × 4
        else:
            # General case
            ops_per_vector = 2 * cfg.M
            lut_bytes_per_query = cfg.lut_size_bytes

        # PURE algorithmic op count (B-invariant arithmetic intensity). The QBS
        # batch amortization below is NOT folded in here — it is a throughput/rate
        # effect and lives on the compute ceiling (compute_peak_scale), so
        # operational_intensity = total_ops/total_bytes stays honest.
        total_ops = B * n_scanned * ops_per_vector

        # FastScan QBS query-block amortization: the per-query fixed work (load a
        # list's packed codes once, set up the uint8 LUT in SIMD registers) is
        # SHARED across the queries that hit the same list within a QBS block, so
        # the effective per-query scan THROUGHPUT RISES as the query batch fills and
        # saturates at the block size. Measured on EPYC 9124 (sift PQ16x4fs np128):
        # per-query 0.104ms @bs16 → 0.073ms @bs256+ (effective eff 0.54→0.77). The
        # base model is linear in B (constant per-query) and over-predicts large
        # batches ~40%. Expressed as a peak GAIN = 1/ramp ≥ 1 (1.0 at B=1, preserving
        # the bs=1 calibration; rises to 1/floor by B=_FASTSCAN_QBS_SAT). nbits=4 only.
        compute_peak_scale = 1.0
        if cfg.nbits == 4 and B > 1:
            floor = getattr(self.hw, "pq_fastscan_qbs_floor", _FASTSCAN_QBS_FLOOR)
            ramp = floor + (1.0 - floor) * max(
                0.0, (_FASTSCAN_QBS_SAT - B) / (_FASTSCAN_QBS_SAT - 1))
            compute_peak_scale = 1.0 / ramp

        # Memory: streaming PQ codes + LUT access
        #
        # LUT read: loaded once per (query, partition) pair from dis_tables
        lut_read = B * cfg.nprobe * lut_bytes_per_query

        if cfg.nbits == 4:
            # FastScan (search_implem_12): codes loaded once per UNIQUE LIST per
            # thread. Sorted by list_no, the kernel does ONE pass over codes for
            # all nc queries hitting that list (QBS SIMD batching). No cache reuse
            # across queries — it's compute reuse in SIMD registers.
            #
            # Per thread: unique lists ≈ nlist * (1 - (1-1/nlist)^(pairs_per_thread))
            omp_threads = self.hw.omp_threads
            num_cores = self.hw.num_cores
            eff_threads = min(min(B, omp_threads), num_cores) if B > 1 else 1
            queries_per_thread = math.ceil(B / eff_threads)
            pairs_per_thread = queries_per_thread * cfg.nprobe
            unique_per_thread = cfg.nlist * (1 - (1 - 1 / cfg.nlist) ** pairs_per_thread)
            # Cell imbalance: the lists queries actually touch are LARGER than the
            # balanced average (queries cluster in dense cells), so the codes read
            # per touched list is avg_vectors_per_partition × cell_imbalance — the
            # same imbalance the standard-PQ codes_bytes uses via vectors_scanned.
            # Omitting it under-counts FastScan memory traffic on clustered corpora
            # (glove f≈1.9, gist f≈1.4) → too fast once batch/MT makes codes the
            # bottleneck (single-query FastScan is compute-bound, so it hid there).
            codes_bytes = (eff_threads * unique_per_thread
                           * cfg.avg_vectors_per_partition * cfg.cell_imbalance
                           * cfg.pq_code_bytes)

            # Per-list-visit overhead: memset triggers write-allocate (1x DRAM read),
            # pq4_pack_LUT_qbs_q_map writes to same cache lines (no extra DRAM).
            lut_write = B * cfg.nprobe * lut_bytes_per_query  # write-allocate only

            # FastScan precomp table is accessed in compute_LUT_uint8 (LUT build
            # stage), NOT during scan. Already modeled in _lut_build().
            total_bytes = codes_bytes + lut_read + lut_write
            unique_bytes = total_bytes
        elif cfg.by_residual and cfg.use_precomputed_table:
            # Standard PQ (nbits>=8): the residual precomputed table is consumed
            # by _lut_build() when FAISS prepares the list-specific LUT for each
            # (query, list). The scan itself reads PQ codes and the already-built
            # LUT; charging the precomputed table again here double-counts that
            # traffic and makes high-nprobe / multi-thread standard PQ far too
            # slow.
            codes_bytes = B * n_scanned * cfg.pq_code_bytes
            total_bytes = codes_bytes + lut_read
            unique_bytes = codes_bytes
        else:
            # Standard PQ without precomputed tables
            codes_bytes = B * n_scanned * cfg.pq_code_bytes
            total_bytes = codes_bytes + lut_read
            unique_bytes = codes_bytes + lut_read

        # Working set: LUT for one query + one partition's codes
        ws_lut = lut_bytes_per_query
        ws_codes = cfg.avg_vectors_per_partition * cfg.pq_code_bytes
        ws = ws_lut + min(ws_codes, 4 * 1024 * 1024)  # cap at 4MB per partition

        # Scattered-sequential access: PQ codes are sequential within each
        # inverted list but lists are separate heap allocations (BlockInvertedLists).
        # Jumping between lists incurs DRAM random access latency.
        bytes_per_list = cfg.avg_vectors_per_partition * cfg.pq_code_bytes

        return OperatorCost(
            name="3_pq_scan_adc",
            total_ops=total_ops,
            total_bytes=total_bytes,
            unique_bytes=unique_bytes,
            working_set_bytes=ws,
            # FastScan: single-query vs QBS batch mode (like coarse_quantizer SIMD vs BLAS)
            # QBS batching gives higher SIMD utilization when B > 1.
            compute_mode=(
                ("pq_scan_fastscan_batch" if B > 1 else "pq_scan_fastscan")
                if cfg.nbits == 4 else "pq_scan_standard"
            ),
            compute_peak_scale=compute_peak_scale,
            # FastScan only: cold scattered scan → calibrated effective BW (memory-bound).
            scan_bw_ceiling=(getattr(self.hw, "pq_fastscan_scan_bw_ceiling", 0.0)
                             if cfg.nbits == 4 else 0.0),
            access_block_bytes=bytes_per_list,
        )

    # ── Stage 3 (Flat): Exact Full-Vector Scan ────────────────────────────

    def _flat_scan(self, B: int) -> OperatorCost:
        """Scan full vectors in probed partitions and compute EXACT distances.

        IVF-Flat's replacement for IVF-PQ's LUT build + PQ scan. FAISS
        ``run_scan_codes`` loops over every vector in each probed inverted
        list and calls ``scanner.distance_to_code`` (→ ``fvec_L2sqr`` /
        ``fvec_inner_product``) ONE vector at a time — pure per-vector SIMD,
        NO BLAS/GEMM batching, even for large query batches (unlike the
        coarse quantizer, which is a brute-force IndexFlat that DOES use BLAS
        for nq ≥ 20). There is no LUT, no precomputed table, and no residual
        encoding (IndexIVFFlat forces ``by_residual = false``).

        FLOPs:  B × vectors_scanned × 2d   (exact L2/IP per candidate)
        Bytes:  B × vectors_scanned × code_bytes   (full FP32 vectors, no
                cross-query reuse — in mode 0 each query independently scans
                its own probed lists). This is the dominant cost and is
                typically MEMORY-BOUND: a flat vector is d×4 bytes vs a PQ
                code's M bytes, so flat reads far more per candidate.

        Access pattern: vectors are contiguous within a list
        (``code_size = d × flat_dtype_bytes``) but lists are separate
        allocations, so jumping between probed lists incurs scattered-
        sequential DRAM access (same model the PQ scan uses).
        """
        cfg = self.cfg
        n_scanned = cfg.vectors_scanned

        # Exact distance: ~2d FLOPs per candidate (sub/mul/add accumulate).
        total_ops = B * n_scanned * 2 * cfg.d

        # Memory: stream full vectors from probed lists + read queries.
        # No precomputed table / LUT; no cross-query reuse (treated as cold).
        codes_bytes = B * n_scanned * cfg.code_bytes
        query_bytes = B * cfg.d * 4
        total_bytes = codes_bytes + query_bytes
        unique_bytes = codes_bytes + query_bytes

        # Working set: one query vector + one inverted list's vectors,
        # capped at 4MB per partition (mirrors _pq_scan).
        ws = cfg.d * 4 + min(
            cfg.avg_vectors_per_partition * cfg.code_bytes, 4 * 1024 * 1024
        )

        # Scattered-sequential: contiguous within a list, random jump between
        # lists (InvertedLists are separate heap allocations).
        bytes_per_list = cfg.avg_vectors_per_partition * cfg.code_bytes

        return OperatorCost(
            name="3_flat_scan",
            total_ops=total_ops,
            total_bytes=total_bytes,
            unique_bytes=unique_bytes,
            working_set_bytes=ws,
            # Per-vector SIMD distance kernel (fvec_L2sqr), never BLAS.
            # Own efficiency knob (flat_scan_efficiency): memory-bound full-vector
            # scan, distinct from the compute-bound "simd" coarse/LUT/refine path.
            compute_mode="flat_scan",
            access_block_bytes=bytes_per_list,
        )

    # ── Stage 4: Heap / Top-k Selection ───────────────────────────────────

    def _heap_selection(self, B: int) -> OperatorCost:
        """
        Maintain a max-heap of size k during scanning.

        Per vector: 1 comparison + amortized log(k) heap operations.
        Since most vectors are rejected, effective ops ≈ 1 compare per vector
        + k*log(k)*nprobe heap adjustments (much smaller than scan).

        This is typically negligible compared to PQ scan.
        """
        cfg = self.cfg
        n_scanned = cfg.vectors_scanned

        # Comparison: 1 per vector; heap insert: amortized
        # Estimate: ~3 ops per vector (compare + branch + occasional heap op)
        total_ops = B * n_scanned * 3

        # Bytes: negligible (heap is tiny: k × 12 bytes for dist+id)
        # Heap is register/L1 resident, minimal DRAM traffic
        heap_bytes = B * cfg.k * 12  # (float dist + int64 id)
        total_bytes = heap_bytes
        unique_bytes = heap_bytes  # No reuse, heap is too small to matter

        return OperatorCost(
            name="4_heap_selection",
            total_ops=total_ops,
            total_bytes=total_bytes,
            unique_bytes=unique_bytes,
            working_set_bytes=heap_bytes,
            compute_mode="int",
        )

    # ── Stage 5: Refinement ───────────────────────────────────────────────

    def _refinement(self, B: int) -> OperatorCost:
        """
        Re-rank top candidates using exact (or SQ-compressed) vectors.

        Load original vectors for top (k × refine_ratio) candidates,
        compute exact L2 distances, select final top-k.

        FLOPs: B × n_refine × 2d  (exact distance computation)
        Bytes: B × n_refine × d × dtype_bytes (random access to stored vectors)

        This is typically MEMORY-BOUND due to random access patterns.
        """
        cfg = self.cfg
        n_refine = cfg.refine_candidates

        if n_refine == 0:
            return OperatorCost(
                name="5_refinement",
                total_ops=0, total_bytes=0, unique_bytes=0,
                working_set_bytes=0, compute_mode="simd",
            )

        # FLOPs: exact L2 distance
        total_ops = B * n_refine * 2 * cfg.d

        # Bytes: load refinement vectors (random access pattern)
        # Each candidate vector is loaded once (no reuse between candidates)
        vector_bytes = B * n_refine * cfg.refine_vector_bytes
        query_bytes = B * cfg.d * 4

        # Unique bytes: vectors + queries (no significant reuse in refinement)
        # Queries are reused across candidates but small
        unique_bytes = vector_bytes + query_bytes
        total_bytes = vector_bytes + query_bytes

        # Working set: small (one batch of candidates at a time)
        ws = n_refine * cfg.refine_vector_bytes + cfg.d * 4

        return OperatorCost(
            name="5_refinement",
            total_ops=total_ops,
            total_bytes=total_bytes,
            unique_bytes=unique_bytes,
            working_set_bytes=ws,
            compute_mode="simd",
        )


def estimate_search_cost(
    config: IVFPQConfig,
    hardware: SystemConfig,
    warm_cache: bool = False,
) -> QueryCostBreakdown:
    """
    Convenience function to estimate IVF-PQ search cost.

    Args:
        config: IVF-PQ index configuration.
        hardware: Hardware configuration for roofline model.
        warm_cache: If True, model steady-state behavior (high-QPS benchmarks with warmup).
            If False (default), model cold-start behavior (typical RAG latency).
    """
    model = IVFPQCostModel(config, hardware, warm_cache=warm_cache)
    return model.estimate()
