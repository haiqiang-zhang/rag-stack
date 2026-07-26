"""Pure helpers around objective values and quality-metric aggregation.

Extracted from :mod:`rag_stack.controller` and :mod:`rag_stack.quality_store`
so both can share a single implementation. No state, no I/O — safe to call
from anywhere.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional

logger = logging.getLogger("RAG-Stack")


def aggregate_quality_metrics(
    metrics: Optional[Mapping[str, Optional[float]]],
    spec: Optional[Mapping] = None,
) -> float:
    """Aggregate multiple GT evaluation sub-metrics into a single quality score.

    Behavior:

    * ``spec is None`` (default, legacy): unweighted mean over **all** non-
      ``None`` values; returns ``0.0`` when the dict is empty or every value
      is ``None``.

    * ``spec={'mode': 'mean', 'metrics': [name, ...]}`` — mean over the
      named subset, ignoring entries whose value is ``None`` or absent.

    * ``spec={'mode': 'weighted', 'weights': {name: float, ...}}`` —
      weighted mean (sum of w·v over present non-``None`` entries, divided
      by sum of w over those same entries). Missing/None metrics are
      dropped from BOTH numerator and denominator (no zero-fill).

    Missing-metric handling for ``mean``/``weighted`` is permissive — a
    YAML referencing ``f1`` before ``f1`` is wired into the evaluator
    will still get a sane number, just emit one ``WARNING`` per call.
    To avoid log spam, a missing metric is also recorded so we only warn
    once per process for the same name.

    YAML schema (under ``gt_evaluation``):

    .. code-block:: yaml

        gt_evaluation:
          combined_quality:
            mode: mean
            metrics: [deepeval_faithfulness, deepeval_answer_relevancy, em, f1]
          # or
          # combined_quality:
          #   mode: weighted
          #   weights:
          #     deepeval_faithfulness: 0.4
          #     em: 0.3
          #     f1: 0.3
    """
    if not metrics:
        return 0.0

    if spec is None:
        # Legacy: unweighted mean over ALL non-None.
        valid = [v for v in metrics.values() if v is not None]
        return sum(valid) / len(valid) if valid else 0.0

    mode = str(spec.get("mode", "mean")).lower()
    if mode == "mean":
        names = list(spec.get("metrics") or [])
        if not names:
            # Empty metrics list → fall back to legacy "all" behavior so a
            # misconfigured YAML doesn't silently return 0.
            valid = [v for v in metrics.values() if v is not None]
            return sum(valid) / len(valid) if valid else 0.0
        _warn_missing(names, metrics)
        vals = [metrics[n] for n in names if metrics.get(n) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    if mode == "weighted":
        weights = dict(spec.get("weights") or {})
        if not weights:
            raise ValueError(
                "aggregate_quality_metrics: spec.mode='weighted' requires "
                "non-empty 'weights' dict"
            )
        _warn_missing(list(weights.keys()), metrics)
        total_score = 0.0
        total_weight = 0.0
        for name, w in weights.items():
            v = metrics.get(name)
            if v is None:
                continue
            total_score += float(w) * float(v)
            total_weight += float(w)
        return total_score / total_weight if total_weight > 0 else 0.0

    raise ValueError(
        f"aggregate_quality_metrics: unknown spec.mode {mode!r}; "
        f"supported: 'mean' | 'weighted'"
    )


# Process-local dedup so missing-metric warnings fire once per name.
_warned_missing: set[str] = set()


def _warn_missing(names, metrics: Mapping) -> None:
    """Emit a one-time WARNING for any metric name in ``names`` that is
    absent from ``metrics`` (None values are NOT considered missing — those
    are valid eval-failure signals, e.g. DeepEval rate-limit)."""
    for n in names:
        if n in metrics:
            continue
        if n in _warned_missing:
            continue
        _warned_missing.add(n)
        logger.warning(
            f"combined_quality references metric {n!r} but it is not "
            f"emitted by the evaluator (available: "
            f"{sorted(metrics.keys())!r}). It will be skipped."
        )


def format_objective_value(name: str, val) -> str:
    """Format an objective value for logging.

    Handles scalars (``float``/``int``), the nested ``{"combined": float, ...}``
    payload emitted for composite quality objectives, and ``None``.
    """
    if val is None:
        return f"{name}=N/A"
    if isinstance(val, dict):
        combined = val.get("combined", "N/A")
        if isinstance(combined, (int, float)):
            return f"{name}={combined:.4f}"
        return f"{name}={combined}"
    if isinstance(val, (int, float)):
        return f"{name}={val:.4f}"
    return f"{name}={val}"
