"""Hardware configuration for the FAISS IVF-PQ cost simulator.

Defines the IVF-PQ-flavored ``SystemConfig`` (hardware fields + IVF-PQ
efficiency knobs + per-probe overheads + random-access penalties) and the
roofline machinery that consumes it.

Pure physical CPU constants (peak FLOPS, cache hierarchy, core topology)
live in :mod:`rag_stack.cost_model.cpu_specs` and are pulled by
:mod:`.system_presets`. Algorithm-specific persistence (calibrated
coefficient JSONs) lives in :mod:`.coefficients`.
"""

from dataclasses import dataclass, field
from typing import Optional

from rag_stack.cost_model.cpu_specs import CacheLevel, CoreTopology  # re-exported for callers


# Multi-thread bandwidth scaling model (default "usl"). "linear" = naive
# min(t*bw1t, agg); used only for ablation/benchmarking — do not change in prod.
BW_THREAD_MODEL = "usl"

# FastScan distance kernel is byte-wide vpshufb: a 256-bit shuffle does 32 byte
# table-lookups vs the 8 int32 lanes that peak_int_ops is sized for → ~4× the
# int32 compute ceiling. Used to lift the FastScan compute roof to its physical
# value so the kernel is correctly memory-bound (see _compute_peak_ops).
FASTSCAN_SIMD_BYTE_RATIO = 4.0


@dataclass
class RandomAccessPenalty:
    """
    Polynomial coefficients for random access penalty modeling.

    Models cache thrashing when accessing precomputed tables with large nprobe.
    The amplification factor is computed as:
        α(ρ) = 1 + (1−ρ)·(c1·ρ + c2·ρ² + c3·ρ³)
    where ρ = nprobe / nlist (probe ratio).

    The (1−ρ) saturation makes the penalty VANISH at ρ→1 (full scan): probing
    every list once is a single sequential pass over the precomputed table — no
    random jumps, no thrashing. Thrashing is maximal at intermediate ρ (many
    scattered slices, table ≫ cache) and zero at both ρ→0 (one slice) and ρ→1
    (sequential). Without it, α grew monotonically to 1+c1+c2+c3 at full scan,
    which over-predicted the LUT build of high-nprobe PQ (e.g. gist960 PQ32
    np1024: lut 7.1ms→~1ms, total +33%→~0%). At the calibrated ρ≤0.125 regime
    the factor is ≈(1−ρ)≈0.9, absorbed into c1.

    Physical interpretation:
    - c1 (linear): Basic random access overhead
    - c2 (quadratic): Cache miss rate increase
    - c3 (cubic): Severe thrashing when working set >> cache size

    Attributes:
        c1: Linear coefficient (default 3.0 for FastScan, 2.0 for standard)
        c2: Quadratic coefficient (default 30.0 for FastScan, 15.0 for standard)
        c3: Cubic coefficient (default 150.0 for FastScan, 0.0 for standard)
    """
    c1: float = 0.0  # linear term
    c2: float = 0.0  # quadratic term
    c3: float = 0.0  # cubic term

    def compute(self, probe_ratio: float) -> float:
        """Compute the random access amplification factor.

        Args:
            probe_ratio: nprobe / nlist.
            probe_ratio: nprobe / nlist for calibrated inter-query mode 0.
        """
        effective_ratio = probe_ratio
        # Penalty vanishes at full scan (ρ→1): sequential single pass, no thrash.
        saturation = max(0.0, 1.0 - probe_ratio)
        return (1.0 + saturation * (
                self.c1 * effective_ratio +
                self.c2 * (effective_ratio ** 2) +
                self.c3 * (effective_ratio ** 3)))


@dataclass
class SystemConfig:
    """
    Hardware configuration for roofline-based performance modeling.

    The roofline model computes latency for each operation as:
        T_op = max(T_compute, T_memory)
    where:
        T_compute = total_ops / (effective_peak_ops * efficiency)
        T_memory  = total_bytes / effective_bandwidth

    Bandwidth has ONE consistent semantic across the whole model: every memory
    level (DRAM + each cache tier) stores an AGGREGATE (all-core) bandwidth and
    a SINGLE-THREAD (one-core) bandwidth. The effective bandwidth at ``t``
    threads is::

        BW(t) = min(t * bw_single_thread, bw_aggregate)

    i.e. linear thread scaling up to the aggregate cap. At ``t = 1`` (the only
    regime the model is calibrated/trusted in — warm cache, active_threads=1)
    this is exactly the measured single-thread bandwidth. There is NO
    ``memory_efficiency`` fudge factor: the old knob conflated "STREAM
    efficiency" with "single-thread fraction of aggregate" and fit to values
    that implied one core pulling many times its fair share of DRAM.

    The tier (and thus both bandwidth columns) is selected by where the working
    set fits in cache (L1 > L2 > L3 > DRAM).

    Attributes:
        name: Human-readable name for this hardware configuration.
        num_cores: Number of physical cores.
        peak_flops: Peak floating-point operations per second (all cores).
        peak_int_ops: Peak integer operations per second (for PQ table lookups).
        mem_bandwidth_bytes_s: AGGREGATE DRAM bandwidth in bytes/s (all-core sustained STREAM).
        mem_bandwidth_1t_bytes_s: SINGLE-THREAD DRAM bandwidth in bytes/s (one-core sustained STREAM).
        simd_width_bits: SIMD register width (e.g., 256 for AVX2, 512 for AVX-512).
        cache_hierarchy: List of CacheLevel from fastest to slowest (each carries
            both an aggregate and a single-thread bandwidth).
        active_threads: Number of threads actually used (for scaling peak performance).
        simd_efficiency: Efficiency factor for hand-optimized SIMD code (0.0-1.0).
            Used for FAISS sequential distance computation (exhaustive_L2sqr_seq).
            Lower than blas_efficiency due to less favorable cache patterns.
            Default 0.55 for CPU.
    """
    name: str
    num_cores: int
    peak_flops: float                        # FLOP/s
    peak_int_ops: float                      # INT-OP/s (for PQ lookups)
    mem_bandwidth_bytes_s: float             # AGGREGATE DRAM B/s (all-core sustained STREAM)
    mem_bandwidth_1t_bytes_s: float          # SINGLE-THREAD DRAM B/s (one-core sustained STREAM)
    simd_width_bits: int = 256               # AVX2 default
    cache_hierarchy: list[CacheLevel] = field(default_factory=list)
    active_threads: int = 0                  # 0 means use all cores
    blas_efficiency: float = 0.8             # Efficiency for BLAS GEMM operations (higher than general compute)
    simd_efficiency: float = 0.55            # Efficiency for hand-optimized SIMD FP distance (coarse small-batch / LUT build / refine)
    flat_scan_efficiency: float = 0.55       # IVF-Flat exact full-vector scan (hand-SIMD fvec_L2sqr, memory-bound) — separate from simd_efficiency
    pq_fastscan_efficiency: float = 0.20   # FastScan PQ scan efficiency (vpshufb, nbits=4, single-query)
    pq_fastscan_batch_efficiency: float = 0.55  # FastScan SIMD batch efficiency (vpshufb, nbits=4, multi-query)
    # FastScan QBS compute-amortization FLOOR: per-query scan COMPUTE (vpshufb, per
    # query×vector) is mostly linear in B — only the SIMD-pipeline gain amortizes it,
    # which saturates at this floor by large batch. 1.0 = no compute amortization
    # (the db-code LOADS are already amortized on the memory side). The old 0.65
    # double-counted the load amortization → over-predicted batch speed. Fittable.
    pq_fastscan_qbs_floor: float = 0.65
    # FastScan per-probe OVERHEAD batch-amortization exponent: per-query per-probe
    # overhead ∝ 1/B**alpha. The default 0.5 (sqrt-B) amortizes ~8× by B=64, but
    # FastScan's per-(query,probe) overhead is mostly per-query (measured per-query
    # latency only drops ~1.5× from B=1→64), so alpha should be ~0.1–0.2. Fittable.
    # nbits=4 only; standard PQ (nbits=8) keeps the sqrt-B form (it fits there).
    pq_fastscan_overhead_alpha: float = 0.5
    pq_standard_efficiency: float = 0.20  # Standard PQ scan efficiency (table lookup, nbits=8)
    pq_fastscan_per_probe_overhead_us: float = 0.0   # Per query×nprobe visit overhead (FastScan)
    pq_standard_per_probe_overhead_us: float = 0.0   # Per query×nprobe visit overhead (Standard PQ)
    flat_per_probe_overhead_us: float = 0.0       # Per query×nprobe visit overhead (IVF-Flat exact scan)
    flat_fixed_overhead_us: float = 0.0           # Per-QUERY fixed overhead (IVF-Flat): faiss call + heap init + reorder
    pq_setup_per_dim_us: float = 0.0              # Per-call PQ LUT-setup floor PER vector dim (d-aware; PQ only, IVF-Flat has no LUT)
    # Random access penalty for precomputed table access (calibratable)
    random_access_penalty_standard: RandomAccessPenalty = field(
        default_factory=lambda: RandomAccessPenalty(c1=2.0, c2=15.0, c3=0.0)
    )
    # Scattered-sequential access model parameters (for PQ scan across inverted lists)
    # Max fraction of tier bandwidth achievable under scattered access with full threads.
    # DRAM controllers have limited concurrency for random access patterns.
    scattered_bw_ceiling: float = 0.35
    # FastScan-SPECIFIC effective scan bandwidth, as a fraction of single-thread
    # STREAM. The PQ codes are streamed once per query across nprobe SEPARATE
    # BlockInvertedLists heap allocations (cold per-list, no reuse), so the scan
    # achieves only ~0.3× STREAM — NOT the cache-tier BW the generic warm-cache
    # path would pick from the tiny per-partition working set. When >0 this routes
    # the FastScan scan's unique-byte time through this ceiling (bypassing the
    # cache tier), making the kernel correctly memory-bound. 0 = off (legacy:
    # generic warm-cache path; do NOT use for FastScan once calibrated). Does NOT
    # affect flat / standard-PQ (they keep the generic path).
    pq_fastscan_scan_bw_ceiling: float = 0.0
    # NUMA: multi-socket systems have local and remote memory.
    # Whether the cost model goes through the NUMA branch at all is
    # controlled by :attr:`numa_enabled` (sourced from
    # ``core_topology.is_numa``, a hardware fact). When NUMA is enabled
    # AND scattered working_set exceeds ``numa_local_bytes``, the
    # bandwidth saturates at the calibrated one-node ceiling. Dedicated
    # remote-socket gain is not modeled without an identifiable standalone grid.
    #
    # The crossover knob below is calibrator-tuned so the empirical FAISS workload
    # tells us the EFFECTIVE crossover threshold (not the raw physical
    # per-socket DDR capacity, which is an upper bound).
    numa_local_bytes: int = 0
    # Heterogeneous core topology (big.LITTLE). None = homogeneous (backward compatible).
    core_topology: Optional[CoreTopology] = None
    # Thread migration model for big.LITTLE architectures.
    # True (default): OS freely migrates threads between P/E cores. Use aggregate_peak
    #   (sum of all active cores). Appropriate when OMP_PROC_BIND is not set (default).
    #   FAISS does not set thread affinity — it relies on OS default scheduling.
    # False: threads are pinned to cores (OMP_PROC_BIND=close/spread). Use bottleneck_peak
    #   (wall time = slowest E-core thread). Appropriate for pinned deployments.
    thread_migration: bool = True

    @property
    def numa_enabled(self) -> bool:
        """True iff the system is multi-socket NUMA (hardware fact from YAML/preset).

        Routes ``effective_bandwidth`` through the cross-socket bandwidth
        penalty branch. Single-socket / UMA systems return False and the
        scattered bandwidth ceiling is always ``scattered_bw_ceiling``.
        """
        return self.core_topology is not None and self.core_topology.is_numa

    @property
    def omp_threads(self) -> int:
        """Actual OMP thread count (including SMT). Used for scheduling decisions."""
        return self.active_threads if self.active_threads > 0 else self.num_cores

    @property
    def effective_num_threads(self) -> int:
        """Threads for compute/bandwidth scaling, capped at physical cores.

        SMT threads share execution units and memory controllers, so
        64 threads on 32 cores ≈ 32 cores for throughput scaling.
        """
        return min(self.omp_threads, self.num_cores)

    @property
    def effective_clock_hz(self) -> float:
        """Effective single-thread clock in Hz.

        Used to convert cycle-count estimates (e.g. per-hop graph overhead
        in HNSW) into wall-clock time. For heterogeneous cores, returns the
        P-core clock since latency-sensitive single-thread workloads are
        scheduled on P-cores.

        Resolution order:
          1. ``core_topology.p_core_clock_hz`` if explicitly set (most reliable).
          2. Derive from ``peak_flops`` assuming Intel-style AVX2/AVX-512
             FMA density (2 FMA × 2 ops × SIMD lanes = simd_width_bits / 8
             FLOP/cycle). This matches Intel / AMD; it over-estimates on
             ARM NEON (where FLOP/cycle depends on unit count, not SIMD
             width) unless p_core_clock_hz is provided.
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


    def _bw_at_threads(self, bw_1t: float, bw_aggregate: float, threads: int,
                       working_set_bytes: Optional[float] = None,
                       scattered: bool = False) -> float:
        """Effective memory bandwidth at ``threads`` threads.

        On a **NUMA (multi-socket)** box, a working set larger than
        ``numa_local_bytes`` is allocated NUMA-local to ~one socket (first-touch),
        so bandwidth scales ~LINEARLY within that socket up to its per-node
        aggregate, SATURATING at ``cores/socket`` threads; threads on the OTHER
        socket reach the data over the cross-socket link and add only at
        zero additional gain. This reproduces the real "saturate at one socket"
        knee (e.g. 2S×16 EPYC flattens at 16 threads:
        38→84→163→250→250 GB/s) — which the concave USL CANNOT reach (it
        saturates asymptotically, far too late, and under-predicts at the knee).
        The per-node aggregate is ``bw_aggregate * scattered_bw_ceiling`` (the
        already-calibrated random-access fraction of full STREAM).

        For a uniform single-socket box (or a cache-resident working set, or
        ``numa_local_bytes=0``) this falls back to the USL contention form,
        anchored at ``BW(1)=bw_1t`` so the single-thread calibration is unchanged.
        """
        t = max(1, threads)
        if bw_aggregate <= 0:
            return t * bw_1t
        # ── NUMA: linear-to-per-node-aggregate + socket saturation + remote ──
        if (self.numa_enabled and self.numa_local_bytes > 0
                and working_set_bytes is not None
                and working_set_bytes > self.numa_local_bytes
                and self.core_topology is not None
                and self.core_topology.num_sockets > 1):
            cps = max(1, self.num_cores // self.core_topology.num_sockets)
            node_agg = max(bw_1t, bw_aggregate * self.scattered_bw_ceiling)
            slope = (node_agg - bw_1t) / max(1, cps - 1)   # per-thread within one node
            if t <= cps:
                return bw_1t + (t - 1) * slope             # linear to node_agg at cps
            # The standalone grid does not identify a remote-socket gain once
            # per-worker first-touch footprints are accounted for. Do not add a
            # guessed interconnect coefficient: throughput stays capped at the
            # measured one-node aggregate until dedicated evidence exists.
            return node_agg
        # ── non-NUMA / cache-resident: USL contention (asymptotes to aggregate) ──
        # Scattered (random) access saturates below full sequential STREAM.
        agg = bw_aggregate * self.scattered_bw_ceiling if scattered else bw_aggregate
        if BW_THREAD_MODEL == "linear":
            return min(t * bw_1t, agg)
        usl = t * bw_1t / (1.0 + (t - 1) * bw_1t / agg)
        return min(usl, agg)


    def _compute_peak_ops(self, num_threads: int, compute_mode: str,
                          partitioned: bool = False) -> float:
        """Compute effective peak ops/s for given thread count and mode.

        Handles heterogeneous (big.LITTLE) cores when core_topology is set.
        Falls back to linear scaling for homogeneous architectures.

        Args:
            num_threads: Active compute threads.
            compute_mode: "blas", "simd", "pq_scan", or "int".
            partitioned: If True (mode 0), wall time = slowest thread (E-core
                bottleneck on big.LITTLE). If False, aggregate throughput.
        """
        if compute_mode in ("blas", "simd"):
            ops_type = "flops"
        else:
            ops_type = "int_ops"

        efficiency = {
            "blas": self.blas_efficiency,
            "simd": self.simd_efficiency,
            "flat_scan": self.flat_scan_efficiency,
            "pq_scan_fastscan": self.pq_fastscan_efficiency,
            "pq_scan_fastscan_batch": self.pq_fastscan_batch_efficiency,
            "pq_scan_standard": self.pq_standard_efficiency,
        }.get(compute_mode, 1.0)

        topo = self.core_topology
        if topo is not None and topo.is_heterogeneous:
            if self.thread_migration or not partitioned:
                # thread_migration=True (default): OS freely migrates threads
                # between P/E cores. FAISS does not set OMP_PROC_BIND, so the
                # OS scheduler reassigns work from finished P-core threads to
                # idle cores. Use aggregate (sum of all active cores' peaks).
                # Also used for cooperative work (mode 1, partitioned=False)
                # where aggregate throughput is always correct.
                raw_peak = topo.aggregate_peak(num_threads, mode=ops_type)
            else:
                # thread_migration=False: threads pinned to cores
                # (OMP_PROC_BIND=close/spread). In partitioned mode 0, each
                # thread gets equal work — wall time = slowest (E-core) thread.
                raw_peak = topo.bottleneck_peak(num_threads, mode=ops_type)
        else:
            # Homogeneous: per-core peak × active physical cores. Any
            # non-linear scaling must emerge from the operator/resource model;
            # there is no fitted generic compute-contention coefficient.
            base_peak = self.peak_flops if ops_type == "flops" else self.peak_int_ops
            if self.num_cores > 0:
                raw_peak = (base_peak / self.num_cores) * num_threads
            else:
                raw_peak = base_peak

        return raw_peak * efficiency

    def effective_bandwidth(
        self,
        working_set_bytes: float,
        access_block_bytes: Optional[float] = None,
        threads: Optional[int] = None,
        numa_working_set_bytes: Optional[float] = None,
    ) -> float:
        """
        Return the effective bandwidth based on where the working set fits
        in the cache hierarchy and the access pattern.

        Args:
            working_set_bytes: Total working set size for cache tier selection.
            access_block_bytes: If specified, models scattered-sequential access
                where data is read in contiguous blocks of this size, with random
                jumps between blocks (e.g., PQ scan across inverted lists).
                If None, models pure sequential access.
            threads: Number of threads for bandwidth scaling. If None, uses
                effective_num_threads.
            numa_working_set_bytes: Optional footprint used only for NUMA
                first-touch/socket-saturation decisions. Cache tier selection
                still uses ``working_set_bytes``.
        """
        if threads is None:
            threads = self.effective_num_threads
        if numa_working_set_bytes is None:
            numa_working_set_bytes = working_set_bytes

        # Step 1: select cache tier by PER-THREAD-AVAILABLE capacity (not aggregate).
        # A thread sees its private L1/L2 (cores_per_instance=1 → full size_bytes),
        # and its share of a shared pool (L3): size_bytes / min(threads, cores_per
        # _instance). So a single thread on this box sees 16MB L3, not the 128MB
        # aggregate — large working sets correctly spill to L3/DRAM. Presets that
        # still store AGGREGATE sizes default cores_per_instance=1 → size_bytes is
        # used as-is (their prior behavior is unchanged).
        tier_bw = self.mem_bandwidth_bytes_s          # aggregate
        tier_bw_1t = self.mem_bandwidth_1t_bytes_s    # single-thread
        # Random-access latency is ALSO a per-tier fact: a scattered jump to a
        # block that lives in L2/L3 costs that tier's latency (~4/15 ns), NOT the
        # DRAM random-access latency (~120 ns). For a small index whose probed
        # working set fits in cache, the per-block latency term otherwise dominates
        # and over-charges by ~10-40× (the tiny-corpus / cache-resident regime).
        # Selected from working_set_bytes — i.e. derived from N/M/nprobe — so it is
        # correct per-config, not a hardcoded DRAM assumption.
        # The frozen component grid cannot identify a separate effective DRAM
        # latency once its list blocks are bandwidth dominated. Keep cache-tier
        # physical latencies, but do not inject a guessed DRAM latency term.
        tier_latency_ns = 0.0
        for cache in self.cache_hierarchy:
            cpi = getattr(cache, "cores_per_instance", 1) or 1
            avail = cache.size_bytes / max(1, min(int(threads), cpi))
            if working_set_bytes <= avail:
                tier_bw = cache.bandwidth_bytes_s
                tier_bw_1t = cache.bandwidth_1t_bytes_s
                tier_latency_ns = cache.latency_ns
                break

        if access_block_bytes is None:
            # Sequential access (NUMA-aware via working_set_bytes).
            return self._bw_at_threads(
                tier_bw_1t, tier_bw, threads, numa_working_set_bytes)

        # Scattered-sequential: per-block latency model
        #
        # bw_1t = block_bytes / (latency + block_bytes / peak_bw_1t)
        #   - Large blocks → bw_1t ≈ peak_bw_1t (bandwidth-dominated)
        #   - Small blocks → bw_1t ≈ block_bytes / latency (latency-dominated)
        #
        # peak_bw_1t is the SINGLE-THREAD tier bandwidth (no efficiency fudge).
        # Multi-thread: each thread issues independent streams, total BW
        # capped by DRAM controller capacity for random access (the ceiling).
        latency_s = tier_latency_ns * 1e-9
        peak_bw_1t = tier_bw_1t
        bw_1t = access_block_bytes / (latency_s + access_block_bytes / peak_bw_1t)

        # Thread scaling: NUMA-aware (socket saturation) for a node-local working
        # set, else USL with the scattered ceiling. ``scattered=True`` makes
        # _bw_at_threads apply scattered_bw_ceiling consistently (the old in-line
        # ceiling-replacement NUMA branch — which divided the ceiling for ALL
        # threads incl. single-thread — was wrong and is removed).
        return self._bw_at_threads(bw_1t, tier_bw, threads,
                                   working_set_bytes=numa_working_set_bytes,
                                   scattered=True)

    def roofline_time(
        self,
        total_ops: float,
        total_bytes: float,
        working_set_bytes: Optional[float] = None,
        unique_bytes: Optional[float] = None,
        batch_size: int = 1,
        warm_cache: bool = False,
        compute_mode: str = "int",
        compute_peak_scale: float = 1.0,
        scan_bw_ceiling: float = 0.0,
        access_block_bytes: Optional[float] = None,
        compute_threads: Optional[int] = None,
        memory_threads: Optional[int] = None,
        partitioned: bool = False,
    ) -> dict:
        """
        Compute execution time using the roofline model with DRAM + cache separation.

        The roofline model: T = max(T_compute, T_memory)
        - T_compute = total_ops / peak_ops
        - T_memory = T_dram + T_cache

        Args:
            compute_mode: Which peak throughput + efficiency to use:
                "blas"    - BLAS GEMM: peak_flops × blas_efficiency
                "simd"    - hand-optimized SIMD (FP): peak_flops × simd_efficiency
                "int"     - generic integer: raw peak_int_ops (no unidentifiable fudge)
                "pq_scan_fastscan" - FastScan PQ (vpshufb): peak_int_ops × pq_fastscan_efficiency
                "pq_scan_standard" - Standard PQ (table lookup): peak_int_ops × pq_standard_efficiency
            access_block_bytes: If specified, models scattered-sequential access.
            partitioned: If True (mode 0), work is evenly split across threads
                and wall time = slowest thread. On big.LITTLE, E-cores bottleneck.
                If False (mode 1), all threads cooperate and aggregate throughput applies.

        Returns:
            Dict with compute_time_s, memory_time_s, dram_time_s, cache_time_s,
            time_s, and bottleneck.
        """
        # Compute: scale with compute_threads (physical cores, no SMT benefit)
        ct = compute_threads if compute_threads is not None else self.effective_num_threads
        # compute_peak_scale (≥1): a kernel-supplied throughput GAIN on the ceiling
        # (e.g. FastScan QBS batch amortization). Keeps total_ops as the pure
        # algorithmic count so operational_intensity stays B-invariant/honest.
        peak_ops = self._compute_peak_ops(ct, compute_mode, partitioned) * compute_peak_scale
        # FastScan memory-bound model (paired, activated by scan_bw_ceiling>0): lift the
        # compute roof to its physical vpshufb value (byte-wide shuffle ≈ 4× the int32
        # peak_int_ops). Without it the fit drives pq_fastscan_efficiency>1 to drag the
        # roof down to a latency really set by the scattered code loads → mis-labels the
        # kernel compute-bound. With it, compute sits above the scan rate (inactive) and
        # the scan_bw_ceiling memory path correctly wins. Legacy (ceiling=0): unchanged.
        if scan_bw_ceiling > 0 and compute_mode in (
                "pq_scan_fastscan", "pq_scan_fastscan_batch"):
            peak_ops *= FASTSCAN_SIMD_BYTE_RATIO

        ws = working_set_bytes if working_set_bytes is not None else total_bytes

        if unique_bytes is None:
            unique_bytes = total_bytes

        unique_bytes = min(unique_bytes, total_bytes)
        reuse_bytes = total_bytes - unique_bytes

        # Memory: scale with memory_threads (all HW threads, including SMT).
        #
        # In FAISS parallel_mode=0 the batch is partitioned across query workers:
        # each worker scans an independent slice of the batch. NUMA first-touch /
        # socket-saturation should therefore see a per-worker footprint, while
        # cache-tier selection still sees the full unique stream and the numerator
        # below remains the whole batch's bytes.
        mt = memory_threads if memory_threads is not None else self.effective_num_threads
        abb = access_block_bytes  # shorthand, None for sequential access
        numa_working_set = unique_bytes
        if partitioned and mt > 1:
            numa_working_set = max(unique_bytes / mt, ws)
        dram_bw = self._bw_at_threads(
            self.mem_bandwidth_1t_bytes_s, self.mem_bandwidth_bytes_s, mt,
            working_set_bytes=numa_working_set   # NUMA-aware (socket saturation)
        )
        # Cache reuse BW: sequential (data already in cache, no DRAM latency penalty)
        cache_bw = self.effective_bandwidth(ws, threads=mt)

        # Bandwidth for unique_bytes (DRAM loads):
        # - Scattered access penalty applies here (random jumps between blocks)
        # - warm_cache=True: use cache BW if data fits (steady-state, high-QPS)
        # - warm_cache=False: always use DRAM BW (cold-start, typical RAG)
        if scan_bw_ceiling > 0:
            # FastScan: cold scattered scan across separate inverted lists. The
            # effective BW is a calibrated fraction of STREAM, thread-scaled by the
            # same NUMA/USL model — NOT the optimistic cache-tier BW the generic
            # warm-cache path would pick from the tiny per-partition working set.
            unique_bw = self._bw_at_threads(
                self.mem_bandwidth_1t_bytes_s * scan_bw_ceiling,
                self.mem_bandwidth_bytes_s * scan_bw_ceiling, mt,
                working_set_bytes=numa_working_set)
        elif warm_cache:
            unique_bw = self.effective_bandwidth(
                unique_bytes, abb, threads=mt,
                numa_working_set_bytes=numa_working_set)
        else:
            unique_bw = dram_bw

        t_compute = total_ops / peak_ops if peak_ops > 0 else float("inf")

        t_unique = unique_bytes / unique_bw if unique_bw > 0 else float("inf")
        t_reuse = reuse_bytes / cache_bw if cache_bw > 0 and reuse_bytes > 0 else 0.0
        t_memory = t_unique + t_reuse

        if warm_cache and unique_bw > dram_bw:
            t_dram = 0.0
            t_cache = t_memory
        else:
            t_dram = t_unique
            t_cache = t_reuse

        t_total = max(t_compute, t_memory)
        bottleneck = "compute" if t_compute >= t_memory else "memory"

        return {
            "compute_time_s": t_compute,
            "memory_time_s": t_memory,
            "dram_time_s": t_dram,
            "cache_time_s": t_cache,
            "time_s": t_total,
            "bottleneck": bottleneck,
            "operational_intensity": total_ops / total_bytes if total_bytes > 0 else float("inf"),
            "effective_bandwidth_bytes_s": cache_bw,
            "dram_bandwidth_bytes_s": dram_bw,
        }




def system_config_to_faiss_hardware(hw, **kwargs) -> SystemConfig:
    """Convert a rag_stack SystemConfig to a FaissHardwareConfig.

    Uses num_cores, peak_flops, peak_int_ops, mem_bandwidth, and all
    memory_tiers from SystemConfig. Synthesizes ``CoreTopology`` from
    ``topology_type`` so the IVF-PQ cost model knows about big.LITTLE
    P/E split and / or NUMA multi-socket layout. Extra kwargs
    (e.g. active_threads, simd_efficiency, numa_local_bytes from
    calibration) are forwarded to ``SystemConfig``.

    ``topology_type`` mapping:
      * ``uniform``        — ``core_topology=None`` (UMA, single socket)
      * ``numa``           — ``CoreTopology(num_sockets=N)`` (N >= 2)
      * ``heterogeneous``  — ``CoreTopology(num_p_cores=..., num_e_cores=...,
                              p_core_peak_flops=..., ...)`` (big.LITTLE)

    Whether the IVF-PQ ``effective_bandwidth`` takes the cross-socket
    penalty branch is governed by ``SystemConfig.numa_enabled``
    (= ``core_topology.is_numa`` = ``core_topology.num_sockets > 1``).

    Args:
        hw: A rag_stack.types.SystemConfig instance.
        **kwargs: Passed to HardwareConfig (active_threads, efficiencies,
            numa_local_bytes, etc.)
    """
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

    cache_hierarchy = []
    for tier in hw.memory_tiers:
        # Fall back to the DRAM single-thread value only when a tier omits its own
        # (cache single-thread BW ≥ DRAM, so this is a safe conservative floor —
        # still never derived from the tier's aggregate via a ratio).
        tier_bw_1t = tier.bandwidth_1t if tier.bandwidth_1t > 0 else hw.mem_bandwidth_1t
        cache_hierarchy.append(CacheLevel(
            name=tier.name,
            size_bytes=int(tier.capacity),
            bandwidth_bytes_s=tier.bandwidth,
            bandwidth_1t_bytes_s=tier_bw_1t,
            latency_ns=10.0,
        ))

    core_topology: Optional[CoreTopology] = None
    if hw.topology_type == "heterogeneous":
        # big.LITTLE — populate P/E fields from per-cycle ratios + clock.
        # is_numa stays False (mutually-exclusive constraint).
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
    elif hw.topology_type == "numa":
        # NUMA — only num_sockets is meaningful here (is_numa is a
        # derived property = num_sockets > 1).
        # numa_local_bytes comes via **kwargs
        # from the calibration profile and live on the IVF-PQ
        # SystemConfig (NOT on CoreTopology).
        core_topology = CoreTopology(num_sockets=hw.numa_sockets)
    elif hw.num_p_cores > 0 and hw.p_core_clock_ghz > 0:
        # Backward-compat: pre-topology YAMLs used flat heterogeneous
        # fields with no explicit topology_type. Detect that shape and
        # build a heterogeneous CoreTopology.
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
        peak_int_ops=hw.peak_int_ops,
        mem_bandwidth_bytes_s=hw.mem_bandwidth,
        mem_bandwidth_1t_bytes_s=hw.mem_bandwidth_1t,
        cache_hierarchy=cache_hierarchy,
        core_topology=core_topology,
        thread_migration=hw.thread_migration,
        **kwargs,
    )
