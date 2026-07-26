"""HNSW real-FAISS measurement helpers (latency + workload).

Measures ``(latency, ndis, nhops)`` for real FAISS HNSW searches over a config
grid — ``ndis``/``nhops`` via ``faiss.cvar.hnsw_stats``, latency via
``time.perf_counter`` (warm cache, 1 thread). The output (``HNSWBenchmarkPoint``)
feeds the HARDWARE-roofline fit.

NOTE: the in-code Ax Bayesian-optimization calibrator (``Calibrator`` + the
``calibrate`` CLI) was REMOVED — it pinned coefficients to its bracketed bounds
and was deprecated. The 4 roofline coefficients are now fit DIRECTLY (scipy
differential_evolution over wide bounds, against real multi-dataset measurements)
by ``benchmarks/rag_cm_accuracy/hnsw/calibrate_hardware.py``, which imports
``collect_hnsw_benchmarks`` from here. The per-dataset ndis/nhops half is trained
by ``train.py`` (sklearn). So this module is now measurement-only.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .cost_model import HNSWCoefficients, HNSWCostModel
from .hnsw_config import HNSWConfig
from .workload_predictor import WorkloadProfile
from .coefficients import save_coefficients

logger = logging.getLogger(__name__)


# ── Benchmark data types ──────────────────────────────────────────────


@dataclass
class HNSWBenchmarkPoint:
    """One real measurement of (config, workload, latency)."""

    # Index / query config
    N: int
    d: int
    M: int
    ef_search: int
    ef_construction: int
    batch_size: int
    k: int

    # Measured workload (held fixed in the fit)
    ndis_per_query: float
    nhops_per_query: float

    # Measured wall-clock latency (fit target)
    latency_ms: float

    # Dataset label for logging / filtering
    dataset: str = ""


# ── Benchmark collection ──────────────────────────────────────────────


def _build_index(data: np.ndarray, M: int, ef_construction: int):
    """Build a FAISS HNSW index. Import faiss lazily."""
    import faiss

    d = data.shape[1]
    index = faiss.IndexHNSWFlat(d, M)
    index.hnsw.efConstruction = ef_construction
    index.add(data)
    return index


def _measure_latency_and_workload(
    index,
    queries: np.ndarray,
    ef_search: int,
    k: int,
    batch_size: int,
    n_repeat: int,
    warm_cache: bool,
) -> tuple[float, float, float]:
    """Return (latency_ms_per_batch, ndis_per_query, nhops_per_query).

    Uses ``time.perf_counter`` over ``n_repeat`` trials of one batch.
    The workload counters are reset once and accumulated across all
    trials so the average is stable.
    """
    import faiss

    index.hnsw.efSearch = ef_search
    n_queries = queries.shape[0]
    assert batch_size <= n_queries, (
        f"batch_size={batch_size} exceeds available queries={n_queries}"
    )

    # Warm up
    if warm_cache:
        for _ in range(3):
            index.search(queries[:batch_size], k)

    # Latency trials
    times = []
    for i in range(n_repeat):
        offset = (i * batch_size) % (n_queries - batch_size + 1)
        batch = queries[offset : offset + batch_size]
        t0 = time.perf_counter()
        index.search(batch, k)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    latency_ms = float(np.median(times)) * 1000.0

    # Workload counters (separate pass, measured over enough queries
    # for a stable average -- ~min(200, n_queries))
    stats = faiss.cvar.hnsw_stats
    stats.reset()
    n_workload = min(200, n_queries)
    index.search(queries[:n_workload], k)
    ndis = stats.ndis / n_workload
    nhops = stats.nhops / n_workload

    return latency_ms, ndis, nhops


def collect_hnsw_benchmarks(
    datasets: dict[str, np.ndarray],
    queries_per_dataset: dict[str, np.ndarray],
    M_values: list[int],
    ef_search_values: list[int],
    ef_construction: int = 200,
    batch_sizes: Optional[list[int]] = None,
    k: int = 10,
    n_repeat: int = 30,
    warm_cache: bool = True,
) -> list[HNSWBenchmarkPoint]:
    """Run real FAISS HNSW searches and collect latency + workload.

    Args:
        datasets: {name: vectors (N, d)}. Mix real (e.g. SIFT) and
            synthetic (random / clustered) to cover LID span.
        queries_per_dataset: {name: queries (nq, d)}. ``nq`` must be at
            least ``max(batch_sizes)``.
        M_values, ef_search_values, ef_construction: HNSW parameters
            to sweep.
        batch_sizes: List of query batch sizes. Defaults to [1].
        k: Top-k.
        n_repeat: Timing trials per (config, ef) point.
        warm_cache: If True, warm the cache before measuring.

    Returns:
        List of measurements suitable as input to ``Calibrator.calibrate``.
    """
    if batch_sizes is None:
        batch_sizes = [1]

    points: list[HNSWBenchmarkPoint] = []

    for ds_name, data in datasets.items():
        queries = queries_per_dataset[ds_name]
        N, d = data.shape
        logger.info("Dataset '%s': N=%d d=%d", ds_name, N, d)

        for M in M_values:
            logger.info("  Building HNSW M=%d ef_c=%d", M, ef_construction)
            index = _build_index(data, M, ef_construction)

            for ef in ef_search_values:
                for B in batch_sizes:
                    lat_ms, ndis, nhops = _measure_latency_and_workload(
                        index, queries, ef, k, B, n_repeat, warm_cache,
                    )
                    points.append(HNSWBenchmarkPoint(
                        N=N, d=d, M=M,
                        ef_search=ef,
                        ef_construction=ef_construction,
                        batch_size=B,
                        k=k,
                        ndis_per_query=ndis,
                        nhops_per_query=nhops,
                        latency_ms=lat_ms,
                        dataset=ds_name,
                    ))
                    logger.info(
                        "    M=%d ef=%d B=%d: lat=%.3fms ndis=%.1f nhops=%.1f",
                        M, ef, B, lat_ms, ndis, nhops,
                    )
            del index

    return points
