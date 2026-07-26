"""
Faiss IVF-PQ index configuration.

Encapsulates all algorithm-level parameters that determine the
computational and memory costs of an IVF-PQ search:

Index structure:
  - nlist:    Number of IVF partitions (Voronoi cells).
  - nprobe:   Number of partitions probed at query time.

Product Quantization:
  - d:        Original vector dimensionality.
  - M:        Number of PQ sub-quantizers (sub-vectors).
  - nbits:    Bits per sub-quantizer code (default 8 → 256 centroids).
  - dsub:     Dimension per sub-vector = d / M.

Database:
  - N:        Total number of database vectors.

Search:
  - k:        Number of nearest neighbors to return.
  - batch_size: Number of queries per batch.

Optional refinement (re-ranking):
  - refine_ratio:  If > 0, the top (k × refine_ratio) candidates from PQ
                    are re-ranked using original (or SQ-compressed) vectors.
  - refine_dtype_bytes: Bytes per dimension for refinement vectors
                        (4 = FP32, 2 = FP16, 1 = INT8/SQ8).
"""

from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class IVFPQConfig:
    """Configuration for a Faiss IVF-PQ index."""

    # ── Vector & Database ─────────────────────────────────────────────────
    d: int = 768               # Vector dimensionality
    N: int = 1_000_000_000     # Number of database vectors (1B default)

    # ── Index type ────────────────────────────────────────────────────────
    # "pq"   = IVF-PQ: probed lists hold compressed PQ codes, scanned via a
    #          per-query distance LUT (ADC). All PQ-specific machinery below
    #          (M / nbits / by_residual / precomputed tables / refinement)
    #          only applies to this mode.
    # "flat" = IVF-Flat: probed lists hold full vectors; the scan computes
    #          EXACT distances per vector (FAISS run_scan_codes → fvec_L2sqr,
    #          SIMD, no BLAS). No LUT, no precomputed table, no refinement.
    index_type: str = "pq"
    flat_dtype_bytes: int = 4  # IVF-Flat: bytes/dim of stored vectors (4=FP32; SQ16=2, SQ8=1)

    # ── IVF Partitioning ──────────────────────────────────────────────────
    nlist: int = 65536          # Number of IVF partitions
    nprobe: int = 64            # Partitions probed per query

    # ── Product Quantization ──────────────────────────────────────────────
    M: int = 96                 # Number of sub-quantizers
    nbits: int = 8              # Bits per sub-quantizer code

    # ── Search Parameters ─────────────────────────────────────────────────
    k: int = 10                 # Top-k neighbors to return
    batch_size: int = 1         # Query batch size
    parallel_mode: int = 0      # only calibrated/supported mode: inter-query

    # ── Refinement (Optional) ─────────────────────────────────────────────
    refine_ratio: float = 0.0   # 0 = no refinement; e.g., 10 → refine top 10*k
    refine_dtype_bytes: int = 4 # Bytes per dim for refinement vectors (FP32)

    # ── Precomputed Tables ────────────────────────────────────────────────
    use_precomputed_table: bool = True  # Faiss precomputes (nlist × M × ksub) table
    # None = use the FAISS per-kernel default (resolved in __post_init__):
    # IndexIVFPQ → True, IndexIVFPQFastScan → False ("faster"). Explicit True/False
    # is honored — the residual capability is preserved, only the UNSET default is
    # made FAISS-consistent. (The cost model fully supports by_residual=True for
    # both kernels; the paper benchmark just doesn't exercise that axis.)
    by_residual: Optional[bool] = None  # Encode residuals (more accurate, extra cost)

    # ── Cell-imbalance factor ─────────────────────────────────────────────
    # Real (clustered) data has uneven IVF cells, and queries (~data distribution)
    # land preferentially in big cells, so the actual vectors scanned per query is
    # cell_imbalance × the balanced estimate (nprobe·N/nlist). 1.0 = balanced (the
    # backward-compatible default). It is a DATA property (= nlist·Σsize_i²/N² from
    # the trained index's cell sizes), independent of hardware / code type — see the
    # per-(corpus,embedding) imbalance profile.
    cell_imbalance: float = 1.0

    def __post_init__(self):
        assert self.index_type in ("pq", "flat"), \
            f"index_type must be 'pq' or 'flat', got {self.index_type!r}"
        # PQ-only invariants: M / nbits are meaningless for IVF-Flat (full
        # vectors, no sub-quantizers), so only enforce them in PQ mode.
        if self.index_type == "pq":
            assert self.d % self.M == 0, f"d ({self.d}) must be divisible by M ({self.M})"
            assert self.nbits in (4, 8, 10, 12, 16), f"nbits must be 4, 8, 10, 12, or 16"
            if self.by_residual is None:
                # FAISS per-kernel default: IndexIVFPQFastScan sets by_residual=false
                # ("set to false by default because it's faster"); IndexIVFPQ defaults
                # true. nbits==4 ⇒ FastScan ⇒ False. Explicit callers are untouched.
                self.by_residual = (self.nbits != 4)
            if self.nbits == 4 and self.by_residual:
                raise ValueError(
                    "residual FastScan is unsupported by the calibrated IVF "
                    "cost model: the frozen standalone grid contains only "
                    "FAISS-default non-residual FastScan"
                )
        else:  # flat
            # IVFFlat stores full vectors: no residual encoding, no precomputed
            # table, no PQ refinement. Force these off so PQ-only cost terms
            # (precomputed-table overhead, refinement stage) never apply, even
            # if a caller left the PQ defaults in place.
            self.by_residual = False
            self.use_precomputed_table = False
            self.refine_ratio = 0.0
        # Shared IVF invariants (apply to both PQ and Flat).
        assert self.nprobe <= self.nlist, f"nprobe ({self.nprobe}) > nlist ({self.nlist})"
        if self.parallel_mode != 0:
            raise ValueError(
                "FAISS IVF parallel_mode=1 is unsupported by the calibrated "
                "cost model: the frozen standalone grid contains mode 0 only"
            )

    # ── Derived Properties ────────────────────────────────────────────────

    @property
    def dsub(self) -> int:
        """Dimension per sub-vector."""
        return self.d // self.M

    @property
    def ksub(self) -> int:
        """Number of centroids per sub-quantizer."""
        return 1 << self.nbits  # 2^nbits

    @property
    def pq_code_bytes(self) -> int:
        """
        Bytes per PQ-encoded vector.
        For nbits=8:  M bytes.
        For nbits=4:  M/2 bytes (packed).
        General: ceil(M * nbits / 8).
        """
        return math.ceil(self.M * self.nbits / 8)

    @property
    def code_bytes(self) -> int:
        """Bytes per stored vector inside an inverted list (what the scan reads).

        IVF-Flat stores full vectors (``d × flat_dtype_bytes``; FP32 by
        default, matching FAISS ``code_size = sizeof(float) * d``).
        IVF-PQ stores compressed PQ codes (``pq_code_bytes``).
        """
        if self.index_type == "flat":
            return self.d * self.flat_dtype_bytes
        return self.pq_code_bytes

    @property
    def scan_fraction(self) -> float:
        """
        Fraction of the database scanned per query.
        With uniform partition sizes: nprobe / nlist.
        """
        return self.nprobe / self.nlist

    @property
    def vectors_scanned(self) -> int:
        """Expected vectors scanned per query = balanced estimate × cell_imbalance.

        Balanced (uniform partitions) is nprobe·N/nlist; real clustered data scans
        cell_imbalance× more (queries hit big cells). cell_imbalance defaults to 1.0.
        """
        return int(self.N * self.scan_fraction * self.cell_imbalance)

    @property
    def avg_vectors_per_partition(self) -> int:
        """Average number of vectors per IVF partition."""
        return self.N // self.nlist

    @property
    def lut_size_bytes(self) -> float:
        """
        Size of the distance lookup table per query.
        Standard (nbits=8): M × ksub × 4 bytes (FP32 distances).
        For Faiss ADC: M × 256 × 4 = M × 1024 bytes.
        """
        return self.M * self.ksub * 4  # FP32 per entry

    @property
    def precomputed_table_bytes(self) -> float:
        """
        Size of the precomputed distance table.
        Faiss precomputes: nlist × M × ksub × 4 bytes.
        This avoids recomputing residual-based LUTs per list.
        """
        if self.use_precomputed_table:
            return self.nlist * self.M * self.ksub * 4
        return 0

    @property
    def index_size_bytes(self) -> float:
        """
        Total index memory footprint.
        = PQ codes + coarse centroids + (optional) precomputed tables + overhead.
        """
        if self.index_type == "flat":
            # IVF-Flat: full vectors + coarse centroids + 64-bit IDs.
            # No PQ codebooks and no precomputed tables.
            vectors = self.N * self.code_bytes
            coarse_centroids = self.nlist * self.d * 4  # FP32
            overhead_ids = self.N * 8
            return vectors + coarse_centroids + overhead_ids
        pq_codes = self.N * self.pq_code_bytes
        coarse_centroids = self.nlist * self.d * 4  # FP32
        pq_codebooks = self.M * self.ksub * self.dsub * 4  # PQ centroid table
        overhead_ids = self.N * 8  # 64-bit IDs
        precomp = self.precomputed_table_bytes
        return pq_codes + coarse_centroids + pq_codebooks + overhead_ids + precomp

    @property
    def refine_candidates(self) -> int:
        """Number of candidates to re-rank if refinement is enabled."""
        if self.refine_ratio > 0:
            return int(self.k * self.refine_ratio)
        return 0

    @property
    def refine_vector_bytes(self) -> float:
        """Bytes to load per refinement candidate."""
        return self.d * self.refine_dtype_bytes

    def summary(self) -> dict:
        """Return a summary dict of key derived metrics."""
        return {
            "index_type": self.index_type,
            "d": self.d,
            "N": self.N,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "M": self.M,
            "nbits": self.nbits,
            "dsub": self.dsub,
            "ksub": self.ksub,
            "pq_code_bytes": self.pq_code_bytes,
            "scan_fraction": self.scan_fraction,
            "vectors_scanned": self.vectors_scanned,
            "lut_size_bytes": self.lut_size_bytes,
            "index_size_GB": self.index_size_bytes / (1024**3),
            "refine_candidates": self.refine_candidates,
        }
