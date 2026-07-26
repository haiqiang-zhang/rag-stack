"""Two-counter eval accounting (07-06).

``eval`` (raw) = every attempt: id + folder, incl. penalty/INVALID — always
preserved on disk. ``valid_eval`` = SUCCESS evals only (eval_type GT/CM) —
the ``n_iterations`` budget and apps progress read only this.

Genuine-infeasible penalty rows (GT CONTEXT_OVERFLOW / CM OOM) feed the GP
(live AND replayed on resume via ``EvalType.reload_as_penalty``) but never
consume a budget seat; every other failure stays fully excluded.
"""
import json

from rag_stack.types import EvalType
from rag_stack.project_summary import ProjectSummary, ProjectStatus


def test_reload_as_penalty_is_the_two_genuine_types():
    """User rule 07-06: ONLY context-overflow and OOM may feed the GP."""
    assert EvalType.reload_as_penalty() == {
        EvalType.OOM.value, EvalType.CONTEXT_OVERFLOW.value,
    }


def test_penalty_types_stay_excluded_for_every_other_consumer():
    """reload_as_penalty is a CHECKPOINT-ONLY exception: the quality store
    (and any other excluded_from_data consumer) must keep filtering these
    rows — a penalty row has no real judge scores."""
    assert EvalType.reload_as_penalty() <= EvalType.excluded_from_data()
    # INVALID / DUPLICATE are never replayed anywhere.
    assert EvalType.INVALID.value not in EvalType.reload_as_penalty()
    assert EvalType.DUPLICATE.value not in EvalType.reload_as_penalty()


def test_project_summary_valid_eval_count_field():
    s = ProjectSummary(
        status=ProjectStatus.RUNNING, gt_count=45, cm_count=0,
        n_evaluations=48, valid_eval_count=45,
    )
    d = s.model_dump()
    assert d["valid_eval_count"] == 45
    assert d["n_evaluations"] == 48  # raw attempts stay visible for audit


def test_project_summary_backward_compat_without_field():
    """Old summary.json files (pre-07-06) have no valid_eval_count."""
    old = {"status": "completed", "gt_count": 50, "n_evaluations": 52}
    s = ProjectSummary(**json.loads(json.dumps(old)))
    assert s.valid_eval_count is None
    assert s.gt_count == 50
