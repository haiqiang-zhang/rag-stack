"""Small numeric/tensor helpers shared across the agent package."""

from typing import Dict

import numpy as np
import torch
from torch import Tensor

def _round(v):
    return round(v, 4) if isinstance(v, float) else v


# ======================================================================
# Optimizer
# ======================================================================



def _pareto_mask(Y: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (maximization)."""
    n = len(Y)
    is_p = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_p[i]:
            continue
        dominated = np.all(Y[i] >= Y, axis=1) & np.any(Y[i] > Y, axis=1)
        dominated[i] = False
        is_p[dominated] = False
    return is_p


def _unique_rows_with_index(x: Tensor) -> "tuple[Tensor, list[int]]":
    """First-occurrence row dedup that ALSO returns kept indices, so parallel
    metadata (candidate provenance) stays aligned with the deduped tensor."""
    rounded = (x * 1e8).round() / 1e8
    _, idx = torch.unique(rounded, dim=0, return_inverse=True)
    seen: Dict[int, int] = {}
    keep: list = []
    for i, k in enumerate(idx.tolist()):
        if k not in seen:
            seen[k] = i
            keep.append(i)
    return x[torch.tensor(keep, dtype=torch.long)], keep


def _unique_rows(x: Tensor) -> Tensor:
    rounded = (x * 1e8).round() / 1e8
    _, idx = torch.unique(rounded, dim=0, return_inverse=True)
    seen: Dict[int, int] = {}
    keep = []
    for i, k in enumerate(idx.tolist()):
        if k not in seen:
            seen[k] = i
            keep.append(i)
    return x[torch.tensor(keep, dtype=torch.long)]
