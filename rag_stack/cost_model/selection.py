"""Selection policies for picking the best hardware configuration.

Each policy is registered via ``@register_selection(name)`` and looked up
by name in ``apply_selection``. A policy function must:

1. Accept ``(df, **kwargs)`` where kwargs come from PerformanceSelection.
2. Pick the best row from the RAGO sweep DataFrame.
3. Return ``(hw_params: dict, score: float)`` — ``score`` is a scalar for
   MOBO maximization (higher is better), ``hw_params`` captures the selected
   hardware configuration as a plain dict.

All metric directions are declared once in ``METRIC_SPEC``. Policies use
``_to_max`` / ``_violation`` helpers so that adding a new metric only
requires updating ``METRIC_SPEC``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Callable

import numpy as np
import pandas as pd

from rag_stack.types import PerformanceSelection

logger = logging.getLogger("RAG-Stack")

# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------

# logical_name → (df column, higher_is_better)
METRIC_SPEC: dict[str, tuple[str, bool]] = {
    "latency_s":               ("latency_s",       False),
    "ttft_s":                  ("latency_s_ttft",  False),
    "tpot_s":                  ("latency_s_tpot",  False),
    "throughput_qps":          ("qps",             True),
    "throughput_qps_per_chip": ("qps_per_chip",    True),
}


def _column(metric: str) -> str:
    if metric not in METRIC_SPEC:
        raise KeyError(
            f"Unknown metric '{metric}'. Add it to METRIC_SPEC. "
            f"Known: {list(METRIC_SPEC)}"
        )
    return METRIC_SPEC[metric][0]


def _higher_is_better(metric: str) -> bool:
    return METRIC_SPEC[metric][1]


def _to_max(value, metric: str):
    """Convert raw metric value(s) so higher = better. Works on scalars,
    numpy arrays, and pandas Series."""
    return value if _higher_is_better(metric) else -value


def _violation(value, threshold: float, metric: str):
    """Constraint violation amount (>= 0). Threshold stated in natural
    sense: ``latency ≤ L*`` for lower-is-better, ``qps ≥ Q*`` for
    higher-is-better. Returns 0 when satisfied."""
    if _higher_is_better(metric):
        return np.maximum(threshold - value, 0.0)
    return np.maximum(value - threshold, 0.0)


# Score returned for configurations with no SLO-feasible point. Far below
# any feasible score so GP / MOBO learns "infeasible = strictly worse".
INFEASIBLE_SCORE = -1e9


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SelectionFn = Callable[..., "tuple[dict, float]"]

SELECTION_REGISTRY: dict[str, SelectionFn] = {}


def register_selection(name: str):
    """Decorator to register a selection policy function."""
    def decorator(fn: SelectionFn) -> SelectionFn:
        SELECTION_REGISTRY[name] = fn
        return fn
    return decorator


def _unwrap_row(row: pd.Series) -> dict:
    """``row.to_dict()`` with numpy scalars converted to Python primitives."""
    return {col: val.item() if hasattr(val, "item") else val
            for col, val in row.items()}


def _select_single(df: pd.DataFrame, metric: str) -> tuple[dict, float]:
    """Pick the row that optimizes a single metric, respecting direction."""
    series = df[_column(metric)].astype(float)
    idx = series.idxmax() if _higher_is_better(metric) else series.idxmin()
    row = df.loc[idx]
    return _unwrap_row(row), float(_to_max(series.loc[idx], metric))


# ---------------------------------------------------------------------------
# Built-in policies
# ---------------------------------------------------------------------------

@register_selection("min_latency")
def select_min_latency(df: pd.DataFrame, **kwargs) -> tuple[dict, float]:
    return _select_single(df, "latency_s")


@register_selection("min_ttft")
def select_min_ttft(df: pd.DataFrame, **kwargs) -> tuple[dict, float]:
    return _select_single(df, "ttft_s")


@register_selection("min_tpot")
def select_min_tpot(df: pd.DataFrame, **kwargs) -> tuple[dict, float]:
    return _select_single(df, "tpot_s")


@register_selection("max_throughput")
def select_max_throughput(df: pd.DataFrame, **kwargs) -> tuple[dict, float]:
    return _select_single(df, "throughput_qps")


@register_selection("weighted")
def select_weighted(
    df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    **kwargs,
) -> tuple[dict, float]:
    """Maximize weighted sum of per-metric normalized scores.

    Each metric is converted to higher-is-better via ``_to_max`` and then
    min-max normalized within the current df to ``[0, 1]``. The returned
    score is the (un-negated) weighted sum, already in higher-is-better
    convention.
    """
    if weights is None:
        return select_min_latency(df)

    score = np.zeros(len(df))
    for metric, w in weights.items():
        if w <= 0:
            continue
        values = df[_column(metric)].astype(float).to_numpy()
        max_values = _to_max(values, metric)
        rng = max_values.max() - max_values.min()
        if rng > 0:
            normalized = (max_values - max_values.min()) / rng
        else:
            normalized = np.zeros_like(max_values)
        score += w * normalized

    best_pos = int(np.argmax(score))
    row = df.iloc[best_pos]
    return _unwrap_row(row), float(score[best_pos])


@register_selection("constrained")
def select_constrained(
    df: pd.DataFrame,
    constraint: dict[str, float] | None = None,
    optimize: str | None = None,
    **kwargs,
) -> tuple[dict, float]:
    """Hard SLO: filter feasible by ``constraint``, then optimize ``optimize``.

    Constraint thresholds are stated in natural sense (``latency_s: 0.5``
    means ``latency ≤ 0.5``; ``throughput_qps: 100`` means ``qps ≥ 100``).
    If no point is feasible, returns ``({"feasible": False, ...},
    INFEASIBLE_SCORE)`` so MOBO learns the SLO boundary as a hard cliff.
    """
    if constraint is None or optimize is None:
        return select_min_latency(df)

    feasible = df
    for metric, threshold in constraint.items():
        col = _column(metric)
        if _higher_is_better(metric):
            feasible = feasible[feasible[col] >= threshold]
        else:
            feasible = feasible[feasible[col] <= threshold]

    if len(feasible) == 0:
        # Diagnostic: report min violation per constraint metric, so logs
        # show *why* the config was infeasible.
        violations = {
            m: float(_violation(df[_column(m)].astype(float), thr, m).min())
            for m, thr in constraint.items()
        }
        logger.warning(
            f"No feasible point satisfies constraint {constraint}; "
            f"min violations: {violations}. Returning INFEASIBLE_SCORE."
        )
        return {"feasible": False, "violations": violations}, INFEASIBLE_SCORE

    series = feasible[_column(optimize)].astype(float)
    idx = series.idxmax() if _higher_is_better(optimize) else series.idxmin()
    row = _unwrap_row(feasible.loc[idx])
    row["feasible"] = True
    return row, float(_to_max(series.loc[idx], optimize))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

DEFAULT_POLICY = "min_latency"


def apply_selection(
    df: pd.DataFrame,
    selection: PerformanceSelection | None,
) -> tuple[dict, float]:
    """Look up the policy in ``SELECTION_REGISTRY`` and return (hw_params, score).

    The returned score is a scalar for MOBO maximization (higher is better).
    """
    # Dynamic sweeps retain infeasible and analytically screened rows for a
    # complete audit CSV.  Those rows carry qps=0/latency=inf and must never
    # participate in weighted normalization, SLO checks, or winner selection.
    # Static/legacy tables have no simulator_status and preserve their exact
    # historical behavior.
    eligible = df
    if "simulator_status" in df.columns:
        eligible = df[df["simulator_status"] == "ok"]
        if eligible.empty:
            raise RuntimeError(
                "selection received no closed-loop simulator-feasible rows"
            )

    sel_name = selection.selection if selection else DEFAULT_POLICY
    fn = SELECTION_REGISTRY.get(sel_name)
    if fn is None:
        logger.warning(f"Unknown selection '{sel_name}', using {DEFAULT_POLICY}")
        fn = SELECTION_REGISTRY[DEFAULT_POLICY]

    if selection is not None:
        kwargs = asdict(selection)
        kwargs.pop("selection", None)
        kwargs.pop("objectives", None)
        return fn(eligible, **kwargs)
    return fn(eligible)
