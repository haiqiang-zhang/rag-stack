"""Collect HNSW workload training data via instrumented FAISS search.

Uses faiss.cvar.hnsw_stats to measure distance computations (ndis)
and node expansions (nhops) per query across parameter configurations.
"""

from __future__ import annotations

import logging
from typing import Optional

import faiss
import numpy as np
import pandas as pd

from .graph_stats import HNSWGraphStats, compute_lid, extract_graph_stats

logger = logging.getLogger(__name__)

# Column names for the output DataFrame
COLUMNS = [
    "N", "d", "M", "ef_search", "ef_construction",
    "actual_avg_degree", "degree_std", "num_layers", "lid",
    "ndis_per_query", "nhops_per_query",
]


def _build_index(
    data: np.ndarray,
    M: int,
    ef_construction: int,
) -> faiss.IndexHNSWFlat:
    """Build a FAISS HNSW index."""
    d = data.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.add(data)
    return index


def _measure_workload(
    index: faiss.IndexHNSWFlat,
    queries: np.ndarray,
    ef_search: int,
    k: int = 10,
) -> tuple[float, float]:
    """Run search and return (ndis_per_query, nhops_per_query).

    Uses faiss.cvar.hnsw_stats global counters.
    """
    n_queries = queries.shape[0]
    index.hnsw.efSearch = ef_search

    stats = faiss.cvar.hnsw_stats
    stats.reset()
    index.search(queries, k)

    ndis_per_query = stats.ndis / n_queries
    nhops_per_query = stats.nhops / n_queries
    return ndis_per_query, nhops_per_query


def collect_from_index(
    index: faiss.IndexHNSWFlat,
    queries: np.ndarray,
    graph_stats: HNSWGraphStats,
    M: int,
    ef_construction: int,
    ef_search_values: list[int],
    k: int = 10,
) -> list[dict]:
    """Collect workload samples for one index across ef_search values."""
    d = queries.shape[1]
    rows = []
    for ef_search in ef_search_values:
        ndis, nhops = _measure_workload(index, queries, ef_search, k)
        rows.append({
            "N": graph_stats.N,
            "d": d,
            "M": M,
            "ef_search": ef_search,
            "ef_construction": ef_construction,
            "actual_avg_degree": graph_stats.actual_avg_degree,
            "degree_std": graph_stats.degree_std,
            "num_layers": graph_stats.num_layers,
            "lid": graph_stats.lid,
            "ndis_per_query": ndis,
            "nhops_per_query": nhops,
        })
        logger.debug(
            f"M={M} ef_c={ef_construction} ef_s={ef_search}: "
            f"ndis={ndis:.1f} nhops={nhops:.1f}"
        )
    return rows


def collect_training_data(
    datasets: dict[str, np.ndarray],
    M_values: Optional[list[int]] = None,
    ef_search_values: Optional[list[int]] = None,
    ef_construction_values: Optional[list[int]] = None,
    n_queries: int = 1000,
    k: int = 10,
) -> pd.DataFrame:
    """Sweep configs x datasets and collect workload profiles.

    Args:
        datasets: {name: vectors} mapping. Each vectors array is (N, d).
        M_values: HNSW M parameter values to sweep.
        ef_search_values: ef_search values to sweep.
        ef_construction_values: ef_construction values to sweep.
        n_queries: Number of queries to average over.
        k: Top-k for search.

    Returns:
        DataFrame with columns matching COLUMNS.
    """
    if M_values is None:
        M_values = [8, 12, 16, 24, 32, 48, 64]
    if ef_search_values is None:
        ef_search_values = [16, 32, 64, 128, 256, 512]
    if ef_construction_values is None:
        ef_construction_values = [100, 200, 400]

    all_rows: list[dict] = []

    for ds_name, data in datasets.items():
        N, d = data.shape
        logger.info(f"Dataset '{ds_name}': N={N}, d={d}")

        # Compute LID once per dataset (independent of HNSW params)
        lid_sample = min(1000, max(100, N // 50))
        lid_value = compute_lid(data, n_samples=lid_sample, k=100)
        logger.info(f"  LID = {lid_value:.2f}")

        # Generate queries from the same distribution (sample from data)
        rng = np.random.default_rng(42)
        q_idx = rng.choice(N, size=min(n_queries, N), replace=False)
        queries = data[q_idx].copy()
        # Add small noise so queries aren't exact index vectors
        queries += rng.normal(0, 0.01, queries.shape).astype(np.float32)

        for M in M_values:
            for ef_c in ef_construction_values:
                logger.info(f"  Building index: M={M}, ef_construction={ef_c}")
                index = _build_index(data, M, ef_c)
                graph_stats = extract_graph_stats(index)
                graph_stats.lid = lid_value

                rows = collect_from_index(
                    index, queries, graph_stats,
                    M, ef_c, ef_search_values, k,
                )
                all_rows.extend(rows)

                # Free memory
                del index

    return pd.DataFrame(all_rows, columns=COLUMNS)


def generate_synthetic_datasets(
    N_values: Optional[list[int]] = None,
    d_values: Optional[list[int]] = None,
) -> dict[str, np.ndarray]:
    """Generate synthetic datasets spanning a range of LID values.

    Creates three dataset families per (N, d):
    - Uniform random (high LID ~ d/2)
    - Loose clusters (mid LID): Gaussian mixture, noise_std = 1.0
    - Tight clusters (low LID ~ 10-20): small noise on top of low-dim manifold

    Returns:
        {name: vectors} dict.
    """
    if N_values is None:
        N_values = [10_000, 50_000, 100_000]
    if d_values is None:
        d_values = [64, 128, 256, 512]

    datasets: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(0)

    for N in N_values:
        for d in d_values:
            # Uniform random (high LID)
            datasets[f"random_N{N}_d{d}"] = (
                rng.standard_normal((N, d)).astype(np.float32)
            )

            # Loose clusters: k centers, unit-variance noise
            for n_clusters in (10, 15):
                centers = rng.standard_normal((n_clusters, d)).astype(np.float32) * 10
                labels = rng.integers(0, n_clusters, size=N)
                data = centers[labels] + rng.standard_normal((N, d)).astype(np.float32)
                datasets[f"loose_N{N}_d{d}_k{n_clusters}"] = data.astype(np.float32)

            # Tight clusters: low-dim manifold + small noise (emulates real data
            # like SIFT where intrinsic dim << ambient dim)
            for intrinsic_d in (8, 16):
                # Random projection from intrinsic_d to d
                proj = rng.standard_normal((intrinsic_d, d)).astype(np.float32)
                proj /= np.linalg.norm(proj, axis=1, keepdims=True)
                latent = rng.standard_normal((N, intrinsic_d)).astype(np.float32)
                data = latent @ proj + rng.standard_normal((N, d)).astype(np.float32) * 0.1
                datasets[f"tight_N{N}_d{d}_id{intrinsic_d}"] = data.astype(np.float32)

    return datasets


def load_sift_learn(
    path: str = "data/sift/sift_learn.fvecs",
    max_N: int = 100_000,
) -> Optional[np.ndarray]:
    """Load SIFT learning set (disjoint from sift_base.fvecs).

    Returns None if the file isn't present so training can still run.
    """
    import os
    if not os.path.exists(path):
        logger.warning(f"SIFT learn set not found at {path}, skipping")
        return None

    raw = np.fromfile(path, dtype=np.float32)
    d = raw[:1].view(np.int32)[0]
    data = raw.reshape(-1, d + 1)[:, 1:].copy()
    if max_N and len(data) > max_N:
        data = data[:max_N]
    return data.astype(np.float32, copy=False)
