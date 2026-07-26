import math

import pandas as pd
import pytest

from rag_stack.cost_model.hybrid_shortlist import (
    HybridShortlistConfig,
    hybrid_shortlist,
)


def _config(**overrides):
    values = {
        "score_column": "analytical_qps",
        "stratum_columns": ("dataflow", "generator_pd", "qe_pd"),
        "topology_columns": ("topology",),
        "timeout_column": "timeout",
        "parent_columns": (
            "dataflow", "generator_pd", "qe_pd", "topology", "batch",
        ),
        "valid_column": "valid",
    }
    values.update(overrides)
    return HybridShortlistConfig(**values)


def _row(
    ident,
    score,
    *,
    topology="t0",
    batch=1,
    timeout=0.002,
    dataflow="sequential",
    generator_pd="unified",
    qe_pd="absent",
    valid=True,
):
    return {
        "id": ident,
        "analytical_qps": score,
        "dataflow": dataflow,
        "generator_pd": generator_pd,
        "qe_pd": qe_pd,
        "topology": topology,
        "batch": batch,
        "timeout": timeout,
        "valid": valid,
    }


def test_timeout_siblings_are_selected_atomically_in_original_order():
    frame = pd.DataFrame([
        _row("other", 5.0, topology="slow", batch=8, timeout=0.05),
        _row("winner-50ms", 10.0, batch=16, timeout=0.05),
        _row("winner-2ms", 10.0, batch=16, timeout=0.002),
        _row("winner-10ms", 10.0, batch=16, timeout=0.01),
    ], index=[8, 3, 3, 1])

    result = hybrid_shortlist(
        frame,
        _config(
            topologies_per_stratum=1,
            parents_per_topology=1,
            near_best_fraction=0.0,
            global_rescue_parents=0,
        ),
    )

    assert result.selected["id"].tolist() == [
        "winner-50ms", "winner-2ms", "winner-10ms",
    ]
    assert result.selected.index.tolist() == [3, 3, 1]
    winner_diag = result.diagnostics[result.diagnostics["selected"]]
    assert winner_diag["selection_reasons"].map(set).tolist() == [
        {"stratum_topology_top"},
    ] * 3
    assert result.summary["selected_parents"] == 1
    assert result.summary["selected_rows"] == 3


def test_strata_topology_cap_and_near_best_parent_band():
    rows = []
    # Five topologies in the same stratum: topology t4 must be pruned.  Seven
    # parents in each retained topology are within 2% and therefore extend the
    # top-5 list to all seven.
    for topology_index in range(5):
        base = 100.0 - topology_index * 10.0
        for parent_index in range(7):
            rows.append(_row(
                f"t{topology_index}-p{parent_index}",
                base - parent_index * 0.1,
                topology=f"t{topology_index}",
                batch=parent_index + 1,
            ))
    # A second active-dataflow stratum receives its own topology budget.
    rows.append(_row(
        "react-only", 1.0, topology="react-topology", batch=1,
        dataflow="react", generator_pd="disagg", qe_pd="unified",
    ))
    frame = pd.DataFrame(rows)

    result = hybrid_shortlist(
        frame,
        _config(global_rescue_parents=0),
    )

    selected = set(result.selected["id"])
    assert all(f"t{topology}-p{parent}" in selected
               for topology in range(4) for parent in range(7))
    assert not any(f"t4-p{parent}" in selected for parent in range(7))
    assert "react-only" in selected
    pruned = result.diagnostics[frame["topology"] == "t4"]
    assert set(pruned["exclusion_reason"]) == {"topology_rank_pruned"}


def test_global_top_parent_rescue_crosses_stratum_topology_cap():
    frame = pd.DataFrame([
        _row("kept-topology", 100.0, topology="t0"),
        _row("rescued-topology", 99.0, topology="t1"),
        _row("ordinary-pruned", 50.0, topology="t2"),
    ])
    result = hybrid_shortlist(
        frame,
        _config(
            topologies_per_stratum=1,
            parents_per_topology=1,
            near_best_fraction=0.0,
            global_rescue_parents=2,
        ),
    )

    assert result.selected["id"].tolist() == [
        "kept-topology", "rescued-topology",
    ]
    rescued = result.diagnostics.loc[frame["id"] == "rescued-topology"].iloc[0]
    assert rescued["selection_reasons"] == ("global_rescue",)
    assert rescued["topology_rank_within_stratum"] == 2


@pytest.mark.parametrize(
    "bad_score", [math.nan, math.inf, -math.inf, pd.NA, None, "bad", 0.0, -1.0],
)
def test_invalid_or_nonpositive_score_never_enters_shortlist(bad_score):
    frame = pd.DataFrame([
        _row("good", 10.0, batch=1, timeout=0.002),
        _row("bad", bad_score, batch=2, timeout=0.002),
    ])
    result = hybrid_shortlist(frame, _config(global_rescue_parents=10))

    assert result.selected["id"].tolist() == ["good"]
    assert result.diagnostics.iloc[1]["exclusion_reason"] == "invalid_or_nonfinite_row"


def test_one_invalid_timeout_sibling_rejects_the_whole_parent():
    frame = pd.DataFrame([
        _row("healthy", 8.0, batch=1, timeout=0.002),
        _row("mixed-valid", 100.0, batch=2, timeout=0.002),
        _row("mixed-invalid", 100.0, batch=2, timeout=0.01, valid=False),
    ])
    result = hybrid_shortlist(frame, _config(global_rescue_parents=10))

    assert result.selected["id"].tolist() == ["healthy"]
    assert result.diagnostics.iloc[1]["exclusion_reason"] == "invalid_timeout_sibling"
    assert result.diagnostics.iloc[2]["exclusion_reason"] == "invalid_or_nonfinite_row"
    assert not result.diagnostics.iloc[1]["parent_eligible"]


def test_lower_is_better_near_best_and_nested_topology_cells():
    frame = pd.DataFrame([
        _row("best", 10.0, batch=1, topology={"devices": ["x", "y"]}),
        _row("near", 10.2, batch=2, topology={"devices": ["x", "y"]}),
        _row("far", 10.3, batch=3, topology={"devices": ["x", "y"]}),
    ]).rename(columns={"analytical_qps": "analytical_latency"})
    result = hybrid_shortlist(
        frame,
        _config(
            score_column="analytical_latency",
            higher_is_better=False,
            parents_per_topology=0,
            near_best_fraction=0.02,
            global_rescue_parents=0,
        ),
    )

    assert result.selected["id"].tolist() == ["best", "near"]
    assert result.diagnostics.iloc[2]["exclusion_reason"] == "parent_rank_pruned"


def test_default_parent_identity_ignores_only_timeout_score_and_validity():
    frame = pd.DataFrame([
        _row("same", 10.0, batch=8, timeout=0.002),
        _row("same", 10.0, batch=8, timeout=0.01),
    ])
    config = _config(parent_columns=None, global_rescue_parents=1)
    result = hybrid_shortlist(frame, config)

    assert result.summary["eligible_parents"] == 1
    assert result.selected["id"].tolist() == ["same", "same"]


def test_missing_semantic_column_fails_loudly():
    frame = pd.DataFrame([_row("one", 1.0)]).drop(columns=["qe_pd"])
    with pytest.raises(KeyError, match="qe_pd"):
        hybrid_shortlist(frame, _config())
