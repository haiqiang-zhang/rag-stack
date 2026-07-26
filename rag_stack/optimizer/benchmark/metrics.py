"""Benchmark metrics: Hypervolume and Inverted Generational Distance."""

import numpy as np


def hypervolume_2d(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact 2D hypervolume indicator (maximization convention).

    Args:
        points: (n, 2) array of objective values.
        ref_point: (2,) reference point (lower bound).

    Returns:
        Hypervolume value. 0.0 if no points dominate the reference.
    """
    valid = np.all(points > ref_point, axis=1)
    if not valid.any():
        return 0.0
    pts = points[valid]

    # Sort by first objective descending, filter to non-dominated
    idx = np.argsort(-pts[:, 0])
    pts = pts[idx]
    nd = [pts[0]]
    for p in pts[1:]:
        if p[1] > nd[-1][1]:
            nd.append(p)
    nd = np.array(nd)

    # Sweep-line
    hv = 0.0
    for i, p in enumerate(nd):
        x_right = nd[i + 1][0] if i + 1 < len(nd) else ref_point[0]
        hv += (p[0] - x_right) * (p[1] - ref_point[1])
    return float(hv)


def _hypervolume(points: np.ndarray, ref_point: np.ndarray) -> float:
    """Exact d-dim hypervolume (maximization convention).

    d=2 uses the local sweep-line implementation. d≥3 delegates to
    BoTorch's ``DominatedPartitioning`` (the same routine qNEHVI uses
    internally), which handles arbitrary dimension correctly. Previous
    code routed all inputs through ``hypervolume_2d`` — silently
    correct for 2-obj but produced garbage for d≥3 setups (the inner
    loop only checked column 1, ignoring columns 2..d-1).
    """
    if len(points) == 0:
        return 0.0
    if points.shape[1] == 2:
        return hypervolume_2d(points, ref_point)
    import torch
    from botorch.utils.multi_objective.box_decompositions.dominated import (
        DominatedPartitioning,
    )
    Y = torch.as_tensor(points, dtype=torch.double)
    rp = torch.as_tensor(ref_point, dtype=torch.double)
    valid = (Y > rp).all(dim=1)
    if not valid.any():
        return 0.0
    bd = DominatedPartitioning(ref_point=rp, Y=Y[valid])
    return float(bd.compute_hypervolume().item())


def hypervolume_absolute(points: np.ndarray, ref_point: np.ndarray,
                         norm: "tuple | None" = None) -> float:
    """Absolute hypervolume of a Pareto front (higher = better).

    The real-world metric: unlike :func:`hv_difference` it needs NO true Pareto
    front (which is unknowable in practice), so it matches how rag_stack e2e
    reports HV. When ``norm=(s_0, s_1, ...)`` is given, each objective ``i`` is
    normalized to ``clip(value / s_i, 0, 1)`` and HV is taken against the origin
    — the e2e ``hv_norm`` convention (``PERF_MAX``, ``Q_MAX``) so benchmark HV is
    directly comparable to e2e HV. Without ``norm``, raw HV against ``ref_point``.
    """
    if len(points) == 0:
        return 0.0
    pts = np.asarray(points, dtype=float)
    if norm is not None:
        pts = np.column_stack(
            [np.clip(pts[:, i] / float(norm[i]), 0.0, 1.0) for i in range(pts.shape[1])]
        )
        ref = np.zeros(pts.shape[1])
    else:
        ref = np.asarray(ref_point, dtype=float)
    return _hypervolume(pts, ref)


def hv_difference(predicted: np.ndarray, true_pareto: np.ndarray,
                  ref_point: np.ndarray) -> float:
    """Hypervolume difference between predicted and true Pareto fronts.

    Args:
        predicted: (n, m) predicted Pareto objective values.
        true_pareto: (n_true, m) true Pareto objective values.
        ref_point: (m,) reference point.

    Returns:
        HV(true) - HV(predicted). Lower is better (0 = perfect).
    """
    hv_true = _hypervolume(true_pareto, ref_point)
    hv_pred = _hypervolume(predicted, ref_point) if len(predicted) > 0 else 0.0
    return hv_true - hv_pred


def inverted_generational_distance(predicted: np.ndarray,
                                   true_pareto: np.ndarray) -> float:
    """IGD: average distance from each true Pareto point to nearest predicted point.

    Measures both convergence and diversity of the predicted front.
    Lower is better (0 = perfect coverage).

    Args:
        predicted: (n, m) predicted Pareto objective values.
        true_pareto: (n_true, m) true Pareto objective values.

    Returns:
        IGD value. float('inf') if predicted is empty.
    """
    if len(predicted) == 0:
        return float("inf")

    # For each true point, find min distance to any predicted point
    distances = []
    for t in true_pareto:
        d = np.linalg.norm(predicted - t, axis=1).min()
        distances.append(d)
    return float(np.mean(distances))
