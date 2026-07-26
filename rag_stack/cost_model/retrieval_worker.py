"""Structural service plan for one batched dense-retrieval worker visit.

The runtime receives ``B`` user rows, expands each row into ``K`` backend
queries, flattens those ``B*K`` rows, and chunks the flat list by the deployed
backend cap.  Every chunk executes its own encode, vector-search, and backend
worker glue.  Row-local fusion/fetch runs once after all backend chunks finish.

This module deliberately knows nothing about traces, measured performance, or
the discrete-event simulator.  Both the analytical capacity walk and the DES
use the same plan so they cannot silently disagree about worker units.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class RetrievalBackendChunk:
    """Priced operators for one flattened backend-query chunk."""

    backend_rows: int
    encode_batch: int
    encode_s: float
    vectorsearch_s: float
    backend_glue_s: float


@dataclass(frozen=True)
class RetrievalWorkerPlan:
    """Complete service plan for one retrieval worker batch."""

    user_rows: int
    backend_rows: int
    chunks: Tuple[RetrievalBackendChunk, ...]
    outer_s: float

    @property
    def encode_s(self) -> float:
        return sum(chunk.encode_s for chunk in self.chunks)

    @property
    def retrieval_s(self) -> float:
        """CPU-side search + glue + once-per-user-batch outer service."""
        return (
            sum(
                chunk.vectorsearch_s + chunk.backend_glue_s
                for chunk in self.chunks
            )
            + self.outer_s
        )

    @property
    def total_s(self) -> float:
        return self.encode_s + self.retrieval_s


def retrieval_chunk_sizes(
    multiplicities: Sequence[int],
    *,
    chunk_cap: int,
) -> Tuple[int, ...]:
    """Flatten row multiplicities, then split by the backend request cap."""
    cap = int(chunk_cap)
    if cap <= 0:
        raise ValueError(f"retrieval chunk_cap must be positive, got {chunk_cap!r}")
    backend_rows = sum(max(1, int(value)) for value in multiplicities)
    return tuple(
        min(cap, backend_rows - start)
        for start in range(0, backend_rows, cap)
    )


def build_retrieval_worker_plan(
    multiplicities: Sequence[int],
    *,
    chunk_cap: int,
    encode_s: Optional[Callable[[int], float]] = None,
    vectorsearch_s: Optional[Callable[[int], float]] = None,
    backend_glue_s: Optional[Callable[[int], float]] = None,
    outer_s: Optional[Callable[[int], float]] = None,
    encode_batch_size: Optional[Callable[[int], int]] = None,
) -> RetrievalWorkerPlan:
    """Price one physical retrieval-worker visit from structural row units.

    ``backend_glue_s`` receives each backend chunk size, so both its per-batch
    and per-backend-row terms repeat with ``n_chunks``.  ``outer_s`` receives
    the original user-row count and is called exactly once, after all chunks.
    """
    row_units = tuple(max(1, int(value)) for value in multiplicities)
    user_rows = len(row_units)
    backend_rows = sum(row_units)
    sizes = retrieval_chunk_sizes(row_units, chunk_cap=chunk_cap)
    map_encode_batch = encode_batch_size or (lambda value: value)

    chunk_plans = []
    for size in sizes:
        encode_batch = max(1, int(map_encode_batch(size)))
        chunk_plans.append(RetrievalBackendChunk(
            backend_rows=size,
            encode_batch=encode_batch,
            encode_s=(
                float(encode_s(encode_batch)) if encode_s is not None else 0.0
            ),
            vectorsearch_s=(
                float(vectorsearch_s(size))
                if vectorsearch_s is not None else 0.0
            ),
            backend_glue_s=(
                float(backend_glue_s(size))
                if backend_glue_s is not None else 0.0
            ),
        ))
    chunks = tuple(chunk_plans)
    return RetrievalWorkerPlan(
        user_rows=user_rows,
        backend_rows=backend_rows,
        chunks=chunks,
        outer_s=(
            float(outer_s(user_rows))
            if outer_s is not None and user_rows > 0 else 0.0
        ),
    )
