"""Trace-aware DynamicEngine screening before authoritative DES pricing.

RAGO only enumerates concrete deployment configurations.  Before this module
is called, DynamicEngine must score every feasible configuration with the real
trace/workflow and the same component and hardware x stage x batch curve
evidence used by DES.  This module consumes that explicit score only to
*screen* the deployment sweep.  Every retained row is still priced by the
normal closed-loop DES; analytical values never become the final
optimizer-facing QPS/latency.

Rankings follow the same performance selection as final DES selection.
Latency is minimized; throughput is maximized. Compound policies retain
candidates from every relevant ranking. The structural coverage is unchanged:

* generator P/D mode x query-expansion P/D mode x dataflow are independent
  strata;
* concrete device/TP/PP layouts are topology groups;
* request/decode batches are timeout-free parents; and
* every dynamic-timeout sibling of a selected parent is retained atomically.

This module neither reconstructs a score from RAGO capacities nor introduces a
second queueing model.  Missing or invalid DynamicEngine scores in a production
large sweep are a contract violation and fail closed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import math
import os
from typing import Any, Mapping

import pandas as pd

from rag_stack.cost_model.candidate_layout import (
    candidate_system_layout_from_row,
)
from rag_stack.cost_model.hybrid_shortlist import (
    HybridShortlistConfig,
    hybrid_shortlist,
)
from rag_stack.cost_model.selection import DEFAULT_POLICY, METRIC_SPEC
from rag_stack.types import PerformanceSelection


# This is a search-size guard, not a physical/calibration coefficient.  Small
# and fixed-placement calls retain exhaustive DES because screening would not
# save meaningful work.
_MIN_ROWS_FOR_SCREENING = 256

# Search-compute budgets.  They control how many structurally distinct batch
# parents *within each topology* receive the authoritative DES; they are not
# hardware/performance coefficients.  Every feasible concrete topology keeps
# at least one parent, because an analytical ranking error must never remove a
# complete topology.  Ordinary sweeps keep a broad five-parent cover; very
# large Cartesian products keep one parent per topology plus the independent
# global top-ten rescue.
_VERY_LARGE_SWEEP_ROWS = 4096
_PARENTS_PER_TOPOLOGY = 5
_VERY_LARGE_PARENTS_PER_TOPOLOGY = 1

_SINGLE_METRIC_POLICIES = {
    "min_latency": "latency_s",
    "min_ttft": "ttft_s",
    "min_tpot": "tpot_s",
    "max_throughput": "throughput_qps",
}


def _screening_metrics(selection: PerformanceSelection | None) -> tuple[str, ...]:
    """Mirror final selection defaults; retain constraint/weighted endpoints too.

    Analytical SLO estimates are not hard feasibility proofs. Constrained
    policies therefore take the union of objective and constraint rankings,
    leaving the actual SLO check to selection after DES. Weighted policies use
    their normalized score plus each positive-weight metric's ranking.
    """
    policy = selection.selection if selection is not None else DEFAULT_POLICY
    if policy in _SINGLE_METRIC_POLICIES:
        return (_SINGLE_METRIC_POLICIES[policy],)
    if policy == "constrained":
        if selection.constraint is None or selection.optimize is None:
            return ("latency_s",)
        return tuple(dict.fromkeys((selection.optimize, *selection.constraint)))
    if policy == "weighted":
        if selection.weights is None:
            return ("latency_s",)
        return tuple(metric for metric, weight in selection.weights.items() if weight > 0)
    # Custom selection functions need their own analytical score contract.
    # Until one exists, do not silently prune their candidates by QPS.
    return ()


@dataclass(frozen=True)
class HybridScreeningResult:
    """Positional DES plan plus row-level analytical diagnostics."""

    selected_positions: tuple[int, ...]
    diagnostics: pd.DataFrame
    summary: dict[str, Any]


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def hybrid_screening_enabled(row_count: int, *, production: bool) -> bool:
    """Return whether this call should use the production hybrid path.

    ``CM_DES_EXACT_ALL=1`` is the audit escape hatch.  It restores exhaustive
    DES without changing the RAG-CM input protocol or any candidate identity.
    Injected test simulators stay exhaustive unless a test explicitly opts in
    with ``CM_DES_HYBRID_FORCE=1``.
    """

    if _truthy_env("CM_DES_EXACT_ALL"):
        return False
    if not production and not _truthy_env("CM_DES_HYBRID_FORCE"):
        return False
    return int(row_count) >= _MIN_ROWS_FOR_SCREENING


def _finite_score(value: Any, *, higher_is_better: bool) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    valid = parsed > 0.0 if higher_is_better else parsed >= 0.0
    return parsed if math.isfinite(parsed) and valid else None


def _pd_mode(layout: Mapping[str, Any], engine: str) -> str:
    engines = layout.get("engines") or {}
    if not isinstance(engines, Mapping):
        raise TypeError("candidate engines must be a mapping")
    record = engines.get(engine)
    if not isinstance(record, Mapping):
        return "absent"
    value = str(record.get("pd_serving") or "collocated_pd")
    return value if value in {"collocated_pd", "disagg_pd"} else "collocated_pd"


def _topology_identity(layout: Mapping[str, Any]) -> tuple[Any, ...] | None:
    """Stable concrete placement + TP/PP identity for shortlist grouping.

    RAGO's ``device_layout_id`` identifies physical group/rank allocation, but
    exact-device repricing can choose different TP/PP plans for the same device
    allocation at different batches.  Those are distinct DES topologies and
    must not consume one shared top-k topology seat.  Derive this identity from
    the generic candidate-layout contract rather than any model/device table.
    """

    device_layout_id = str(layout.get("device_layout_id") or "").strip()
    if not device_layout_id:
        return None

    resources = []
    for resource in layout.get("resource_groups") or ():
        if not isinstance(resource, Mapping):
            continue
        resources.append((
            str(resource.get("id") or ""),
            str(resource.get("kind") or ""),
            tuple(str(device) for device in (resource.get("devices") or ())),
            tuple(str(stage) for stage in (
                resource.get("performance_stages") or ()
            )),
            int(resource.get("num_servers", 0) or 0),
        ))
    resources.sort(key=lambda item: item[0])

    stages = []
    for stage, info in sorted(
        (layout.get("performance_stages") or {}).items(),
        key=lambda item: str(item[0]),
    ):
        if not isinstance(info, Mapping):
            continue
        stages.append((
            str(stage),
            str(info.get("resource") or info.get("resource_group") or ""),
            tuple(str(device) for device in (info.get("devices") or ())),
            int(info.get("tp", 0) or 0),
            int(info.get("pp", 0) or 0),
            int(info.get("replicas", 0) or 0),
        ))
    return device_layout_id, tuple(resources), tuple(stages)


def _all_rows_result(frame: pd.DataFrame, reason: str) -> HybridScreeningResult:
    diagnostics = pd.DataFrame({
        "position": list(range(len(frame))),
        "original_index": list(frame.index),
        "selected": [True] * len(frame),
        "selection_reasons": [(reason,)] * len(frame),
        "exclusion_reason": [""] * len(frame),
    }, index=frame.index)
    return HybridScreeningResult(
        selected_positions=tuple(range(len(frame))),
        diagnostics=diagnostics,
        summary={
            "mode": "exhaustive",
            "reason": reason,
            "input_rows": len(frame),
            "selected_rows": len(frame),
            "screened_rows": 0,
        },
    )


def screen_des_candidates(
    frame: pd.DataFrame,
    *,
    selection: PerformanceSelection | None = None,
    production: bool = True,
) -> HybridScreeningResult:
    """Build a conservative analytical shortlist for authoritative DES.

    Candidate positions are positional rather than index labels, so duplicate
    DataFrame indexes remain safe.  The explicit DynamicEngine score is a
    screening diagnostic only; the caller must run DES for every selected row
    before publishing optimizer-facing performance.
    """

    if not hybrid_screening_enabled(len(frame), production=production):
        return _all_rows_result(frame, "below_threshold_or_exact_mode")

    metrics = _screening_metrics(selection)
    if not metrics:
        return _all_rows_result(frame, "no_analytical_selection_contract")
    # Use immutable analytical columns, never a RAGO capacity or DES result.
    rankings = [
        (metric, "dynamic_analytical_" + METRIC_SPEC[metric][0], METRIC_SPEC[metric][1])
        for metric in metrics
    ]
    required = {
        *(column for _, column, _ in rankings),
        "batch_size_request",
        "batch_size_decode",
        "dynamic_batch_timeout_s",
        "candidate_system_layout",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        message = (
            "trace-aware DynamicEngine screening contract is incomplete; "
            "missing columns: " + ",".join(missing)
        )
        if production:
            raise RuntimeError(message)
        return _all_rows_result(
            frame,
            "missing_screening_contract:" + ",".join(missing),
        )

    work = frame.copy()
    for _, column, higher_is_better in rankings:
        scores = [
            _finite_score(value, higher_is_better=higher_is_better)
            for value in work[column]
        ]
        invalid = [position for position, score in enumerate(scores) if score is None]
        if invalid:
            domain = "positive" if higher_is_better else "non-negative"
            message = (
                f"trace-aware DynamicEngine screening requires one finite {domain} "
                f"{column!r} per feasible candidate; invalid positions: {invalid[:16]}"
            )
            if production:
                raise RuntimeError(message)
            return _all_rows_result(frame, "invalid_" + column)
        work[column] = scores

    if selection is not None and selection.selection == "weighted" and selection.weights:
        weighted_column = "dynamic_analytical_weighted_score"
        work[weighted_column] = 0.0
        for metric, column, higher_is_better in rankings:
            values = work[column] if higher_is_better else -work[column]
            span = values.max() - values.min()
            if span > 0:
                work[weighted_column] += (
                    selection.weights[metric] * (values - values.min()) / span
                )
        rankings.insert(0, ("weighted", weighted_column, True))

    primary_metric, score_column, higher_is_better = rankings[0]
    score_name = "dynamic_engine_trace_" + score_column.removeprefix("dynamic_")

    layouts: list[dict[str, Any]] = []
    try:
        for position in range(len(work)):
            row = work.iloc[position]
            layout = candidate_system_layout_from_row(row)
            layouts.append(layout)
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:
        if production:
            raise RuntimeError(
                "trace-aware DynamicEngine screening requires a parseable "
                "concrete candidate_system_layout for every candidate"
            ) from exc
        return _all_rows_result(frame, "unparseable_candidate_layout")

    generator_modes: list[str] = []
    qe_modes: list[str] = []
    topology_ids: list[tuple[Any, ...] | None] = []
    dataflows: list[str] = []

    for position, layout in enumerate(layouts):
        row = work.iloc[position]
        generator_modes.append(_pd_mode(layout, "generator"))
        qe_modes.append(_pd_mode(layout, "query_expansion"))
        topology_ids.append(_topology_identity(layout))
        dataflows.append(str(row.get("pipeline_mode") or "dynamic"))

    if not all(topology_ids):
        if production:
            raise RuntimeError(
                "trace-aware DynamicEngine screening requires one concrete "
                "device/TP/PP topology identity per candidate"
            )
        return _all_rows_result(frame, "missing_concrete_topology_identity")

    work["__hybrid_dataflow"] = dataflows
    work["__hybrid_generator_pd"] = generator_modes
    work["__hybrid_qe_pd"] = qe_modes
    work["__hybrid_topology"] = topology_ids

    # Retain every concrete topology in every structural stratum.  The generic
    # shortlist helper has a fixed topologies-per-stratum default, but applying
    # that default here would let analytical error erase all candidates from a
    # lower-ranked physical layout.  Cardinality is bounded on the batch-parent
    # axis below, not by deleting topology classes.
    topology_counts_by_stratum: dict[tuple[str, str, str], set[Any]] = {}
    for dataflow, generator_mode, qe_mode, topology in zip(
        dataflows,
        generator_modes,
        qe_modes,
        topology_ids,
        strict=True,
    ):
        stratum = (dataflow, generator_mode, qe_mode)
        topology_counts_by_stratum.setdefault(stratum, set()).add(topology)
    topologies_per_stratum = max(
        (len(topologies) for topologies in topology_counts_by_stratum.values()),
        default=0,
    )

    parents_per_topology = (
        _VERY_LARGE_PARENTS_PER_TOPOLOGY
        if len(frame) >= _VERY_LARGE_SWEEP_ROWS
        else _PARENTS_PER_TOPOLOGY
    )
    shortlist_config = HybridShortlistConfig(
        score_column=score_column,
        higher_is_better=higher_is_better,
        require_positive_score=False,
        stratum_columns=(
            "__hybrid_dataflow",
            "__hybrid_generator_pd",
            "__hybrid_qe_pd",
        ),
        topology_columns=("__hybrid_topology",),
        timeout_column="dynamic_batch_timeout_s",
        parent_columns=(
            "__hybrid_dataflow",
            "__hybrid_generator_pd",
            "__hybrid_qe_pd",
            "__hybrid_topology",
            "batch_size_request",
            "batch_size_decode",
        ),
        # The topology/parent budgets and global rescue are already a
        # conservative structural cover.  A relative near-best band has
        # no cardinality bound when an analytical surface is flat (the
        # common saturated case), so it can silently expand hundreds of
        # equivalent parents into DES.  Keep only exact structural seats;
        # this is a search policy, not a fitted performance threshold.
        near_best_fraction=0.0,
        topologies_per_stratum=topologies_per_stratum,
        parents_per_topology=parents_per_topology,
    )
    shortlists = [
        hybrid_shortlist(work, replace(
            shortlist_config, score_column=column, higher_is_better=higher,
        ))
        for _, column, higher in rankings
    ]

    diagnostics = shortlists[0].diagnostics.copy()
    if len(shortlists) > 1:
        # Positional merge preserves duplicate input indexes and all timeout
        # siblings. Rank columns describe the primary objective only.
        reasons = [set() for _ in range(len(frame))]
        for (metric, _, _), shortlist in zip(rankings, shortlists, strict=True):
            for position, selected_reasons in enumerate(shortlist.diagnostics["selection_reasons"]):
                reasons[position].update(f"{metric}:{reason}" for reason in selected_reasons)
        diagnostics["selection_reasons"] = [tuple(sorted(value)) for value in reasons]
        diagnostics["selected"] = [bool(value) for value in reasons]
        diagnostics.loc[diagnostics["selected"], "exclusion_reason"] = ""
    for _, column, _ in rankings:
        diagnostics[column] = work[column].tolist()
    if "dynamic_analytical_qps" in work:
        diagnostics["analytical_screening_qps"] = work["dynamic_analytical_qps"].tolist()
    diagnostics["analytical_screening_score"] = work[score_column].tolist()
    diagnostics["analytical_screening_score_name"] = score_name
    diagnostics["analytical_screening_metric"] = primary_metric
    diagnostics["analytical_screening_higher_is_better"] = higher_is_better
    diagnostics["analytical_screening_is_final"] = False
    diagnostics["analytical_dataflow"] = dataflows
    diagnostics["analytical_generator_pd"] = generator_modes
    diagnostics["analytical_query_expansion_pd"] = qe_modes

    selected_positions = tuple(
        int(position)
        for position in diagnostics.loc[diagnostics["selected"], "position"]
    )
    summary = dict(shortlists[0].summary)
    selected_parents = diagnostics[diagnostics["selected"]].drop_duplicates("parent_key")
    reason_counts = Counter(
        reason for reasons in selected_parents["selection_reasons"] for reason in reasons
    )
    summary.update({
        "mode": "hybrid_dynamic_engine_analytical_des",
        "selected_rows": len(selected_positions),
        "screened_rows": len(frame) - len(selected_positions),
        "selected_parents": len(selected_parents),
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "score": score_name,
        "score_column": score_column,
        "metric": primary_metric,
        "screening_metrics": [metric for metric, _, _ in rankings],
        "higher_is_better": higher_is_better,
        "selection": selection.selection if selection is not None else DEFAULT_POLICY,
        "score_role": "shortlist_only_not_optimizer_performance",
        "score_is_final": False,
        "topologies_per_stratum": topologies_per_stratum,
        "parents_per_topology": parents_per_topology,
    })
    return HybridScreeningResult(
        selected_positions=selected_positions,
        diagnostics=diagnostics,
        summary=summary,
    )


__all__ = [
    "HybridScreeningResult",
    "hybrid_screening_enabled",
    "screen_des_candidates",
]
