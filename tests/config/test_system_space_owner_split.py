"""system_space_owner / system_design_space split semantics.

``performance_source`` says WHERE performance numbers come from;
``system_space_owner`` says WHO searches the deployment design space. The
owner key derives from performance_source when absent (exact pre-split
behaviour), and the optimizer-facing dims block was renamed
``measured_search_space`` → ``system_design_space`` with a read alias.
"""

from __future__ import annotations

import pytest

from rag_stack.performance_context import (
    SYSTEM_SPACE_OWNER_COST_MODEL,
    SYSTEM_SPACE_OWNER_OPTIMIZER,
    resolve_system_space_owner,
    system_design_space,
)


# ---------------------------------------------------------------------------
# resolve_system_space_owner — default derivation + explicit values
# ---------------------------------------------------------------------------

def test_owner_derives_cost_model_when_absent():
    assert resolve_system_space_owner({}) == SYSTEM_SPACE_OWNER_COST_MODEL
    cfg = {"system": {"performance_source": "cost_model"}}
    assert resolve_system_space_owner(cfg) == SYSTEM_SPACE_OWNER_COST_MODEL


def test_owner_derives_optimizer_for_measured():
    cfg = {"system": {"performance_source": "measured"}}
    assert resolve_system_space_owner(cfg) == SYSTEM_SPACE_OWNER_OPTIMIZER


@pytest.mark.parametrize("perf_source", ["cost_model", "measured"])
def test_explicit_owner_wins_over_derivation(perf_source):
    cfg = {"system": {
        "performance_source": perf_source,
        "system_space_owner": "optimizer",
    }}
    assert resolve_system_space_owner(cfg) == SYSTEM_SPACE_OWNER_OPTIMIZER
    cfg["system"]["system_space_owner"] = "cost_model"
    assert resolve_system_space_owner(cfg) == SYSTEM_SPACE_OWNER_COST_MODEL


def test_owner_value_is_normalized_lowercase():
    cfg = {"system": {"system_space_owner": "Optimizer"}}
    assert resolve_system_space_owner(cfg) == SYSTEM_SPACE_OWNER_OPTIMIZER


# ---------------------------------------------------------------------------
# system_design_space — new key, legacy alias, precedence
# ---------------------------------------------------------------------------

def test_design_space_reads_new_key():
    system = {"system_design_space": {"batch_size_request": [1, 8]}}
    assert system_design_space(system) == {"batch_size_request": [1, 8]}


def test_design_space_falls_back_to_legacy_alias():
    system = {"measured_search_space": {"batch_size_request": [1, 8]}}
    assert system_design_space(system) == {"batch_size_request": [1, 8]}


def test_design_space_prefers_new_key_when_both_present():
    # ConfigValidator rejects the double spelling; the reader still has a
    # deterministic preference so partially-validated callers cannot flip.
    system = {
        "system_design_space": {"batch_size_request": [1]},
        "measured_search_space": {"batch_size_request": [2]},
    }
    assert system_design_space(system) == {"batch_size_request": [1]}


def test_design_space_empty_when_unset():
    assert system_design_space({}) == {}
    assert system_design_space(None) == {}
