"""HNSW index configuration for cost modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class HNSWConfig:
    """Configuration for an HNSW index search operation.

    Parameters mirror FAISS IndexHNSWFlat / hnswlib conventions.
    """

    N: int              # Number of indexed vectors
    d: int              # Vector dimension
    M: int              # Max edges per node (upper layers); layer-0 uses M0 = 2*M
    ef_search: int      # Beam width for search (candidate list size)
    k: int = 10         # Top-k results to return
    batch_size: int = 1
    ef_construction: int = 200  # Construction beam width (affects graph quality)

    @property
    def M0(self) -> int:
        """Max edges per node at layer 0 (= 2*M, FAISS/hnswlib convention)."""
        return 2 * self.M

    @property
    def num_layers(self) -> int:
        """Expected maximum layer index = floor(ln(N) / ln(M)).

        Follows skip-list theory (Pugh 1990) and HNSW paper
        (Malkov & Yashunin 2020, levelMult = 1/log(M)).
        """
        if self.N <= 1 or self.M <= 1:
            return 0
        return int(math.log(self.N) / math.log(self.M))

    @property
    def node_block_bytes_layer0(self) -> int:
        """Bytes per node at layer 0 in data_level0_memory_.

        Layout: [link_count(4B)] [neighbor_IDs(M0*4B)] [vector(d*4B)]
        """
        return (self.M0 + 1 + self.d) * 4

    @property
    def index_size_bytes(self) -> int:
        """Total index memory footprint in bytes.

        Layer 0: N * M0 * 4  (neighbor IDs) + N * d * 4  (vectors)
        Upper layers: N/(M-1) * M * 4  (geometric sum of neighbor IDs)
        """
        layer0_links = self.N * self.M0 * 4
        layer0_vectors = self.N * self.d * 4
        if self.M > 1:
            upper_nodes = self.N // (self.M - 1)
            upper_links = upper_nodes * self.M * 4
        else:
            upper_links = 0
        return layer0_links + layer0_vectors + upper_links
