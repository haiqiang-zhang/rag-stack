
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Result from actual FAISS benchmark."""
    index_name: str
    nprobe: int
    recall_at_k: float
    latency_ms: float
    qps: float
    num_queries: int
    batch_size: int = 1


@dataclass
class SimulatorResult:
    """Result from simulator prediction."""
    nprobe: int
    latency_ms: float
    qps: float
    bottleneck: str


@dataclass
class ComparisonResult:
    """Comparison between benchmark and simulator."""
    nprobe: int
    actual_qps: float
    predicted_qps: float
    actual_latency_ms: float
    predicted_latency_ms: float
    qps_error_pct: float
    latency_error_pct: float


@dataclass
class IndexBenchmarkData:
    """Cached benchmark data for a single index configuration."""

    index_name: str
    d: int
    N: int
    nlist: int
    M: int
    nbits: int
    by_residual: bool
    k: int
    batch_size: int
    parallel_mode: int = 0
    index_type: str = "pq"  # "pq" (IVF-PQ) | "flat" (IVF-Flat)
    results: list[BenchmarkResult] = None
