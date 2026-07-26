"""Extract graph statistics from FAISS HNSW indices.

Provides cheap-to-compute, data-dependent features used as ML model inputs.
When no real index is available, analytical fallback estimates are provided.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import faiss
import numpy as np


@dataclass
class HNSWGraphStats:
    """Statistics extracted from an HNSW index graph structure."""

    actual_avg_degree: float   # Mean out-degree at layer 0
    degree_std: float          # Std dev of layer-0 degree distribution
    num_layers: int            # Actual max_level from the index
    N: int                     # Number of indexed vectors
    lid: float = float("nan")  # Local Intrinsic Dimensionality (Amsaleg 2015)


def compute_lid(
    data: np.ndarray,
    n_samples: int = 1000,
    k: int = 100,
    seed: int = 42,
) -> float:
    """Estimate dataset-level Local Intrinsic Dimensionality (LID).

    Uses the MLE estimator from Amsaleg et al. (2015):
        LID(q) = - 1 / mean_{i=1..k} log(r_i(q) / r_k(q))
    where r_i(q) is the distance from q to its i-th nearest neighbor.

    Low LID (5-15): tightly clustered data (e.g., SIFT, image features)
    High LID (~d):   uniform random data (no low-dim structure)

    Args:
        data: (N, d) float32 vectors.
        n_samples: Number of query points sampled to average LID over.
        k: Neighborhood size for MLE estimator.

    Returns:
        Mean LID across sampled query points.
    """
    N, d = data.shape
    n_samples = min(n_samples, N)
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(N, size=n_samples, replace=False)

    # Exact k-NN via FAISS flat index (k+1 because first hit is self)
    index = faiss.IndexFlatL2(d)
    index.add(data.astype(np.float32, copy=False))
    D_sq, _ = index.search(data[sample_idx].astype(np.float32, copy=False), k + 1)

    # Convert squared L2 to L2; skip self (first column)
    D = np.sqrt(np.maximum(D_sq, 0.0))[:, 1:]  # (n_samples, k)

    r_k = D[:, -1:]  # (n_samples, 1)
    # Skip rows where r_k == 0 (duplicate points collapse the estimator)
    valid = (r_k > 0).squeeze()
    D = D[valid]
    r_k = r_k[valid]

    ratios = D / r_k                                  # last column = 1.0
    log_ratios = np.log(np.clip(ratios, 1e-12, None))  # last column = 0.0
    mean_log = np.mean(log_ratios, axis=1)             # (n_valid,)

    # mean_log must be < 0; if it's 0 (all points equidistant) skip
    valid2 = mean_log < -1e-10
    lids = -1.0 / mean_log[valid2]

    lids = lids[np.isfinite(lids) & (lids > 0)]
    if len(lids) == 0:
        return float(d)  # degenerate case: fall back to ambient dim
    return float(np.mean(lids))


def extract_graph_stats(
    index,  # faiss.IndexHNSWFlat or similar
    sample_size: int = 5000,
) -> HNSWGraphStats:
    """Extract graph statistics from a built FAISS HNSW index.

    Samples node degrees at layer 0 to compute mean and std.
    Uses faiss.IndexHNSWFlat.hnsw.nb_neighbors() for the max degree
    and hnsw.neighbors array for actual connectivity.

    Args:
        index: A FAISS HNSW index (IndexHNSWFlat, IndexHNSWSQ, etc.).
        sample_size: Number of nodes to sample for degree statistics.
    """
    hnsw = index.hnsw
    N = index.ntotal
    num_layers = hnsw.max_level

    # Sample node degrees at layer 0
    n_sample = min(sample_size, N)
    if n_sample <= 0:
        return HNSWGraphStats(
            actual_avg_degree=0.0, degree_std=0.0,
            num_layers=num_layers, N=N,
        )

    # Convert FAISS SWIG vectors to numpy for fast access
    neighbors_arr = faiss.vector_to_array(hnsw.neighbors)   # int32
    offsets_arr = faiss.vector_to_array(hnsw.offsets)        # uint64
    M0 = hnsw.nb_neighbors(0)

    # Sample indices uniformly
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(N, size=n_sample, replace=False)

    degrees = np.empty(n_sample, dtype=np.float64)
    for i, node_id in enumerate(sample_ids):
        # offsets[node_id] is the start of this node's neighbor block.
        # Layer-0 neighbors occupy the first M0 slots.
        # FAISS uses -1 for empty slots.
        offset = int(offsets_arr[node_id])
        layer0_neighbors = neighbors_arr[offset:offset + M0]
        degrees[i] = np.sum(layer0_neighbors >= 0)

    return HNSWGraphStats(
        actual_avg_degree=float(np.mean(degrees)),
        degree_std=float(np.std(degrees)),
        num_layers=num_layers,
        N=N,
    )


def estimate_graph_stats(
    N: int,
    d: int,
    M: int,
    ef_construction: int = 200,
) -> HNSWGraphStats:
    """Analytical fallback when no real index is available.

    Estimates graph statistics from construction parameters.
    For large N with typical ef_construction, most layer-0 nodes
    are close to the M0 = 2*M cap.
    """
    M0 = 2 * M
    # For large N, avg degree approaches M0; for small N it's closer to M.
    # Use a simple interpolation based on N relative to ef_construction.
    saturation = min(1.0, N / (10 * ef_construction))
    avg_degree = M + (M0 - M) * saturation

    # Std dev is typically small relative to mean for well-constructed graphs
    degree_std = avg_degree * 0.1

    num_layers = int(math.log(N) / math.log(M)) if N > 1 and M > 1 else 0

    return HNSWGraphStats(
        actual_avg_degree=avg_degree,
        degree_std=degree_std,
        num_layers=num_layers,
        N=N,
    )
