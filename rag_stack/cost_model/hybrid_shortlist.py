"""Deterministic analytical-to-DES candidate shortlisting.

The shortlist is deliberately a pure :class:`pandas.DataFrame` transform.  It
does not know model names, device names, or RAG stage implementations; callers
declare the columns which identify their dataflow/P-D stratum and concrete
topology.  Timeout replicas are collapsed to one *parent* for ranking and are
expanded atomically after selection, so the deployment timeout axis is never
partially sampled.

This module only defines the policy.  It is not wired into the authoritative
DES path here; integration belongs to the assembly orchestration layer.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from typing import Any, Hashable, Iterable

import pandas as pd


@dataclass(frozen=True)
class HybridShortlistConfig:
    """Column contract and policy limits for :func:`hybrid_shortlist`.

    ``stratum_columns`` should encode the active dataflow plus the generator
    and query-expansion P/D modes.  ``topology_columns`` should identify the
    concrete device/parallelism layout.  Keeping both caller-supplied avoids a
    model/device table in the shortlist policy.

    If ``parent_columns`` is omitted, all input columns except the timeout,
    analytical score, and optional validity columns form the timeout-free
    parent identity.  An explicit list is useful when the input carries
    unrelated diagnostics which should not distinguish candidates.
    """

    score_column: str
    stratum_columns: tuple[str, ...]
    topology_columns: tuple[str, ...]
    timeout_column: str = "dynamic_batch_timeout_s"
    parent_columns: tuple[str, ...] | None = None
    valid_column: str | None = None
    higher_is_better: bool = True
    require_positive_score: bool = True
    topologies_per_stratum: int = 4
    parents_per_topology: int = 5
    near_best_fraction: float = 0.02
    global_rescue_parents: int = 10

    def __post_init__(self) -> None:
        if not self.score_column:
            raise ValueError("score_column must be non-empty")
        if not self.stratum_columns:
            raise ValueError("stratum_columns must be non-empty")
        if not self.topology_columns:
            raise ValueError("topology_columns must be non-empty")
        for name, value in (
            ("topologies_per_stratum", self.topologies_per_stratum),
            ("parents_per_topology", self.parents_per_topology),
            ("global_rescue_parents", self.global_rescue_parents),
        ):
            if int(value) < 0:
                raise ValueError(f"{name} must be >= 0")
        if not math.isfinite(float(self.near_best_fraction)):
            raise ValueError("near_best_fraction must be finite")
        if not 0.0 <= float(self.near_best_fraction) < 1.0:
            raise ValueError("near_best_fraction must be in [0, 1)")


@dataclass(frozen=True)
class HybridShortlistResult:
    """Selected rows plus a row-level audit trail and aggregate counts."""

    selected: pd.DataFrame
    diagnostics: pd.DataFrame
    summary: dict[str, Any]


@dataclass
class _Parent:
    key: Hashable
    stratum: Hashable
    topology: Hashable
    positions: list[int]
    score: float
    first_position: int
    eligible: bool


def _freeze(value: Any) -> Hashable:
    """Return a deterministic hashable representation of nested DF cells."""

    if isinstance(value, dict):
        return tuple(
            (str(key), _freeze(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze(item) for item in value), key=repr))
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return ("__missing__",)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _key(row: pd.Series, columns: Iterable[str]) -> tuple[Hashable, ...]:
    return tuple(_freeze(row[column]) for column in columns)


def _valid_flag(value: Any) -> bool:
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def _finite_float(value: Any) -> tuple[bool, float]:
    """Coerce a score without letting ``pd.NA`` escape as ``TypeError``."""

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return False, math.nan
    return bool(math.isfinite(number)), number


def _rank_key(parent: _Parent, *, higher_is_better: bool) -> tuple[float, int]:
    primary = -parent.score if higher_is_better else parent.score
    return primary, parent.first_position


def _near_best(
    score: float,
    best: float,
    *,
    fraction: float,
    higher_is_better: bool,
) -> bool:
    """Direction-aware relative band with stable behavior at zero."""

    tolerance = float(fraction) * max(abs(float(best)), 1e-12)
    if higher_is_better:
        return score >= best - tolerance
    return score <= best + tolerance


def _required_columns(config: HybridShortlistConfig) -> set[str]:
    required = {
        config.score_column,
        config.timeout_column,
        *config.stratum_columns,
        *config.topology_columns,
    }
    if config.parent_columns is not None:
        required.update(config.parent_columns)
    if config.valid_column is not None:
        required.add(config.valid_column)
    return required


def hybrid_shortlist(
    frame: pd.DataFrame,
    config: HybridShortlistConfig,
) -> HybridShortlistResult:
    """Select a stable, stratified analytical shortlist for authoritative DES.

    Policy order:

    1. collapse timeout siblings into atomic timeout-free parents;
    2. within each dataflow/P-D stratum retain the best topology groups;
    3. within each retained topology keep the top batch parents plus every
       parent within the configured relative band of that topology's best;
    4. union a global best-parent rescue set; and
    5. expand selected parents back to *all* timeout siblings in original order.

    A timeout parent is eligible only when every sibling has a finite (and, by
    default, positive) score and passes ``valid_column``.  This atomic rule
    reconciles sibling preservation with the requirement that an invalid/NaN
    row must never enter DES.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    missing = sorted(_required_columns(config) - set(frame.columns))
    if missing:
        raise KeyError(f"hybrid shortlist missing required columns: {missing}")

    if frame.empty:
        diagnostics = pd.DataFrame(
            columns=(
                "position", "original_index", "eligible", "parent_eligible",
                "selected", "selection_reasons", "exclusion_reason", "score",
                "parent_score", "stratum_key", "topology_key", "parent_key",
                "timeout_value", "topology_rank_within_stratum",
                "parent_rank_within_topology", "global_parent_rank",
            ),
            index=frame.index,
        )
        return HybridShortlistResult(
            selected=frame.copy(),
            diagnostics=diagnostics,
            summary={
                "input_rows": 0,
                "eligible_rows": 0,
                "eligible_parents": 0,
                "selected_rows": 0,
                "selected_parents": 0,
                "selected_topologies": 0,
                "selection_reason_counts": {},
            },
        )

    if config.parent_columns is None:
        ignored = {config.timeout_column, config.score_column}
        if config.valid_column is not None:
            ignored.add(config.valid_column)
        parent_columns = tuple(column for column in frame.columns if column not in ignored)
    else:
        parent_columns = tuple(config.parent_columns)

    # Positional bookkeeping preserves duplicate/non-monotonic input indices.
    work = frame.copy()
    numeric_score = pd.to_numeric(work[config.score_column], errors="coerce")
    score_pairs = numeric_score.map(_finite_float)
    finite = score_pairs.map(lambda pair: pair[0])
    score_values = score_pairs.map(lambda pair: pair[1])
    row_eligible = finite.copy()
    if config.require_positive_score:
        row_eligible &= score_values > 0.0
    if config.valid_column is not None:
        row_eligible &= work[config.valid_column].map(_valid_flag)

    stratum_keys: list[Hashable] = []
    topology_keys: list[Hashable] = []
    parent_keys: list[Hashable] = []
    grouped_positions: dict[Hashable, list[int]] = defaultdict(list)
    for position in range(len(work)):
        row = work.iloc[position]
        stratum = _key(row, config.stratum_columns)
        topology = _key(row, config.topology_columns)
        # Include stratum/topology even if a caller's explicit parent columns
        # accidentally omit them: one parent can never straddle either boundary.
        parent = (stratum, topology, _key(row, parent_columns))
        stratum_keys.append(stratum)
        topology_keys.append(topology)
        parent_keys.append(parent)
        grouped_positions[parent].append(position)

    parents: dict[Hashable, _Parent] = {}
    for parent_key, positions in grouped_positions.items():
        eligible = all(bool(row_eligible.iloc[position]) for position in positions)
        scores = [float(score_values.iloc[position]) for position in positions]
        if eligible:
            parent_score = max(scores) if config.higher_is_better else min(scores)
        else:
            parent_score = math.nan
        first = min(positions)
        parents[parent_key] = _Parent(
            key=parent_key,
            stratum=stratum_keys[first],
            topology=topology_keys[first],
            positions=positions,
            score=parent_score,
            first_position=first,
            eligible=eligible,
        )

    eligible_parents = [parent for parent in parents.values() if parent.eligible]
    by_stratum_topology: dict[Hashable, dict[Hashable, list[_Parent]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for parent in eligible_parents:
        by_stratum_topology[parent.stratum][parent.topology].append(parent)

    topology_rank: dict[tuple[Hashable, Hashable], int] = {}
    parent_rank: dict[Hashable, int] = {}
    global_rank: dict[Hashable, int] = {}
    reasons: dict[Hashable, set[str]] = defaultdict(set)

    for stratum, topology_groups in by_stratum_topology.items():
        ranked_topologies: list[tuple[_Parent, Hashable]] = []
        for topology, topology_parents in topology_groups.items():
            ranked = sorted(
                topology_parents,
                key=lambda parent: _rank_key(
                    parent, higher_is_better=config.higher_is_better,
                ),
            )
            for rank, parent in enumerate(ranked, start=1):
                parent_rank[parent.key] = rank
            ranked_topologies.append((ranked[0], topology))
        ranked_topologies.sort(
            key=lambda pair: _rank_key(
                pair[0], higher_is_better=config.higher_is_better,
            )
        )
        for rank, (_, topology) in enumerate(ranked_topologies, start=1):
            topology_rank[(stratum, topology)] = rank
        retained = ranked_topologies[: int(config.topologies_per_stratum)]
        for _, topology in retained:
            ranked = sorted(
                topology_groups[topology],
                key=lambda parent: _rank_key(
                    parent, higher_is_better=config.higher_is_better,
                ),
            )
            if not ranked:
                continue
            best = ranked[0].score
            for rank, parent in enumerate(ranked, start=1):
                if rank <= int(config.parents_per_topology):
                    reasons[parent.key].add("stratum_topology_top")
                # A zero band disables this optional expansion.  Treating
                # zero as "include every exact tie" is unbounded on the flat
                # capacity plateaus produced by saturated queue networks.
                if float(config.near_best_fraction) > 0.0 and _near_best(
                    parent.score,
                    best,
                    fraction=float(config.near_best_fraction),
                    higher_is_better=config.higher_is_better,
                ):
                    reasons[parent.key].add("stratum_topology_near_best")

    globally_ranked = sorted(
        eligible_parents,
        key=lambda parent: _rank_key(
            parent, higher_is_better=config.higher_is_better,
        ),
    )
    for rank, parent in enumerate(globally_ranked, start=1):
        global_rank[parent.key] = rank
    for parent in globally_ranked[: int(config.global_rescue_parents)]:
        reasons[parent.key].add("global_rescue")

    selected_positions = sorted(
        position
        for parent_key in reasons
        for position in parents[parent_key].positions
    )
    selected = frame.iloc[selected_positions].copy()

    diagnostic_rows: list[dict[str, Any]] = []
    for position in range(len(work)):
        parent = parents[parent_keys[position]]
        selected_reasons = tuple(sorted(reasons.get(parent.key, ())))
        selected_row = bool(selected_reasons)
        if selected_row:
            exclusion = ""
        elif not bool(row_eligible.iloc[position]):
            exclusion = "invalid_or_nonfinite_row"
        elif not parent.eligible:
            exclusion = "invalid_timeout_sibling"
        elif topology_rank.get((parent.stratum, parent.topology), math.inf) > int(
            config.topologies_per_stratum
        ):
            exclusion = "topology_rank_pruned"
        else:
            exclusion = "parent_rank_pruned"
        diagnostic_rows.append({
            "position": position,
            "original_index": frame.index[position],
            "eligible": bool(row_eligible.iloc[position]),
            "parent_eligible": bool(parent.eligible),
            "selected": selected_row,
            "selection_reasons": selected_reasons,
            "exclusion_reason": exclusion,
            "score": float(score_values.iloc[position]) if finite.iloc[position] else math.nan,
            "parent_score": parent.score,
            "stratum_key": parent.stratum,
            "topology_key": parent.topology,
            "parent_key": parent.key,
            "timeout_value": work.iloc[position][config.timeout_column],
            "topology_rank_within_stratum": topology_rank.get(
                (parent.stratum, parent.topology)
            ),
            "parent_rank_within_topology": parent_rank.get(parent.key),
            "global_parent_rank": global_rank.get(parent.key),
        })
    diagnostics = pd.DataFrame(diagnostic_rows, index=frame.index)

    reason_counts: Counter[str] = Counter()
    for parent_reasons in reasons.values():
        reason_counts.update(parent_reasons)
    selected_parent_keys = set(reasons)
    selected_topology_keys = {
        (parents[key].stratum, parents[key].topology)
        for key in selected_parent_keys
    }
    summary = {
        "input_rows": len(frame),
        "eligible_rows": int(row_eligible.sum()),
        "eligible_parents": len(eligible_parents),
        "selected_rows": len(selected),
        "selected_parents": len(selected_parent_keys),
        "selected_topologies": len(selected_topology_keys),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "policy": {
            "topologies_per_stratum": int(config.topologies_per_stratum),
            "parents_per_topology": int(config.parents_per_topology),
            "near_best_fraction": float(config.near_best_fraction),
            "global_rescue_parents": int(config.global_rescue_parents),
            "higher_is_better": bool(config.higher_is_better),
            "require_positive_score": bool(config.require_positive_score),
        },
    }
    return HybridShortlistResult(
        selected=selected,
        diagnostics=diagnostics,
        summary=summary,
    )


__all__ = [
    "HybridShortlistConfig",
    "HybridShortlistResult",
    "hybrid_shortlist",
]
