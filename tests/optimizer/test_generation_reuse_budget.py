"""Generation reuse preserves controller accounting and optimizer budgets."""
import json
import os

import pytest

from rag_stack_evaluator.static_rag_evaluator import gen_fingerprint as gfp



# ── free-row budget accounting ────────────────────────────────────────────

def test_greedy_free_rows_return_budget_seats():
    from rag_stack.optimizer.greedy import GreedyParetoOptimizer
    g = GreedyParetoOptimizer.__new__(GreedyParetoOptimizer)
    g._n_iterations = 10
    g._archive = {f"k{i}": {} for i in range(6)}
    g._free_evals = 0
    assert g._remaining() == 4
    g._free_evals = 2          # two of the six were fingerprint-inherited
    assert g._remaining() == 6


def test_wrapper_free_rows_skip_success_counters():
    from rag_stack.optimizer.botorch.greedy_then_hv_anchor_agent_qnehvi import (
        GreedyThenAnchorAgentQNEHVI,
    )
    w = GreedyThenAnchorAgentQNEHVI.__new__(GreedyThenAnchorAgentQNEHVI)
    w._pending_owner = "greedy"
    w.greedy_budget = 30
    w._greedy_successes = 5
    w._agent_successes = 0
    w.objective_names = ["performance", "quality"]
    w._evaluations = []
    calls = {}

    class _G:
        def add_objectives(self, x, obj_dict, eval_ids=None, free_mask=None):
            calls["free_mask"] = free_mask
    w.greedy = _G()
    w._ensure_agent_started = lambda: None

    x = [{"a": 1}, {"a": 2}, {"a": 3}]
    od = {"performance": [10.0, 20.0, 30.0],
          "quality": [{"combined": 0.1}, {"combined": 0.2}, {"combined": 0.3}]}
    w.add_objectives(x, od, eval_ids=[0, 1, 2], free_mask=[False, True, False])
    assert w._greedy_successes == 7            # only the 2 paid rows count
    assert calls["free_mask"] == [False, True, False]
    assert [e["eval_type"] for e in w._evaluations] == ["GT", "GT_FP", "GT"]
    assert [e["free"] for e in w._evaluations] == [False, True, False]


def test_wrapper_warm_start_split_ignores_free_rows():
    from rag_stack.optimizer.botorch.greedy_then_hv_anchor_agent_qnehvi import (
        GreedyThenAnchorAgentQNEHVI,
    )
    w = GreedyThenAnchorAgentQNEHVI.__new__(GreedyThenAnchorAgentQNEHVI)
    w.greedy_budget = 3
    w.total_success_budget = 5
    w.objective_names = ["performance", "quality"]
    w._using_agent = False
    w._agent_success_limit = None
    seen = {}

    class _G:
        def warm_start(self, evals):
            seen["greedy"] = len(evals)
    class _A:
        def warm_start(self, evals):
            seen["agent"] = len(evals)
    w.greedy, w.agent = _G(), _A()
    w._sync_agent_hooks = lambda: None

    def ev(pf, q, free=False):
        return {"config": {"c": pf}, "objectives": {"performance": pf, "quality": q},
                "free": free, "eval_id": 0}
    # paid, FREE, paid, paid → greedy budget (3 paid) is met at index 3 (4 evals)
    evals = [ev(1, .1), ev(2, .2, free=True), ev(3, .3), ev(4, .4), ev(5, .5)]
    w.warm_start(evals)
    assert w._greedy_successes == 3
    assert seen["greedy"] == 4                 # the free row rides along
    assert w._agent_successes == 1             # one paid eval past the boundary
    assert w._using_agent is True


# ── method-only hard gate ─────────────────────────────────────────────────

def test_gen_fp_gate_allows_method_optimizers_only():
    from rag_stack.controller import _gen_fp_dedup_allowed
    base = {"global": {"gen_fingerprint_dedup": True}}
    for ok_type in ("greedy_then_hv_anchor_agent_qnehvi", "agent_qnehvi"):
        assert _gen_fp_dedup_allowed({**base, "optimizer": {"type": ok_type}})
    for baseline in ("ax_qnehvi", "ax_qnehvi_mixedgp", "smac_mobo", "greedy",
                     "sobol", "agent_qnehvi_slo",
                     "greedy_then_anchor_agent_qnehvi"):  # v2: deregistered 07-07
        with pytest.raises(ValueError, match="METHOD feature"):
            _gen_fp_dedup_allowed({**base, "optimizer": {"type": baseline}})
    # flag absent → False for everyone, no raise
    assert not _gen_fp_dedup_allowed({"optimizer": {"type": "smac_mobo"}})
    assert not _gen_fp_dedup_allowed(
        {"global": {}, "optimizer": {"type": "agent_qnehvi"}})


def test_gen_fp_env_hygiene_on_baseline(monkeypatch):
    """A stale shell export must not leak the feature into a run whose config
    does not request it — the controller pops the env on the OFF path."""
    from rag_stack.controller import _gen_fp_dedup_allowed
    monkeypatch.setenv("RAG_STACK_GEN_FP_DEDUP", "1")
    cfg = {"global": {}, "optimizer": {"type": "smac_mobo"}}
    # mirror the controller __init__ sequence
    if _gen_fp_dedup_allowed(cfg):
        os.environ["RAG_STACK_GEN_FP_DEDUP"] = "1"
    else:
        os.environ.pop("RAG_STACK_GEN_FP_DEDUP", None)
    assert not gfp.enabled()


def test_agent_limit_counts_paid_evals_only():
    """FP-hit rows in the greedy phase must not shrink the agent phase's
    allowance: 30 archive rows with 3 free = 27 paid → agent gets 50-27=23,
    not 50-30=20 (which would end the run at 47/50)."""
    from rag_stack.optimizer.botorch.greedy_then_hv_anchor_agent_qnehvi import (
        GreedyThenAnchorAgentQNEHVI,
    )
    w = GreedyThenAnchorAgentQNEHVI.__new__(GreedyThenAnchorAgentQNEHVI)
    w.greedy_budget = 30
    w.total_success_budget = 50
    w._greedy_successes = 27
    w._greedy_exhausted = True
    w._using_agent = False
    w.greedy_plateau_window = 0
    w._evaluations = (
        [{"config": {}, "objectives": {}, "free": False}] * 27
        + [{"config": {}, "objectives": {}, "free": True}] * 3
    )
    w._sync_agent_hooks = lambda: None

    class _A:
        def warm_start(self, evals):
            pass
    w.agent = _A()
    w._ensure_agent_started()
    assert w._using_agent is True
    assert w._agent_success_limit == 23   # 50 - 27 paid, NOT 50 - 30 rows


# ── MOVE LEDGER (explore/exploit accounting card, 07-07) ─────────────────

def test_move_ledger_labels_and_dhv_telescope():
    from rag_stack.optimizer.botorch.greedy_then_hv_anchor_agent_qnehvi import (
        GreedyAnchorLLMAgent,
    )

    class _SS:
        def active_params(self, f):
            return set(f)

    a = GreedyAnchorLLMAgent.__new__(GreedyAnchorLLMAgent)
    # e0 seed; e1 = 1-knob refinement that advances the front (EXPLOIT hit);
    # e2 = 4-knob jump that advances the front hugely (EXPLORE hit);
    # e3 = 3-knob jump, dominated (EXPLORE miss)
    H = [
        {"performance": 100.0, "correctness": 0.10,
         "flat": {"a": 1, "b": 1, "c": 1, "d": 1}},
        {"performance": 100.0, "correctness": 0.20,
         "flat": {"a": 2, "b": 1, "c": 1, "d": 1}},
        {"performance": 90.0, "correctness": 0.50,
         "flat": {"a": 3, "b": 2, "c": 2, "d": 2}},
        {"performance": 50.0, "correctness": 0.15,
         "flat": {"a": 9, "b": 9, "c": 9, "d": 1}},
    ]
    card = a._move_ledger_card(H, _SS())
    assert "EXPLOIT: 1 evals, 1 advanced the front" in card
    assert "EXPLORE: 2 evals, 1 advanced the front" in card
    assert "best e02" in card                       # the big explore hit
    assert card.strip().endswith("X E E")           # chronological labels
    # ΔHV telescoping: class sums == final HV − seed HV (final-normalized)
    import re
    dhvs = [float(m) for m in re.findall(r"ΔHV \+([0-9.]+)", card)]
    p_rng, q_rng = 100.0, 0.50
    def hv(rows):
        pts = sorted(((h["performance"]/p_rng, h["correctness"]/q_rng) for h in rows), key=lambda t: -t[0])
        s, pq = 0.0, 0.0
        for x, y in pts:
            if y > pq: s += x*(y-pq); pq = y
        return s
    assert abs(sum(dhvs) - (hv(H) - hv(H[:1]))) < 1e-6


def test_move_ledger_short_history_empty():
    from rag_stack.optimizer.botorch.greedy_then_hv_anchor_agent_qnehvi import (
        GreedyAnchorLLMAgent,
    )
    a = GreedyAnchorLLMAgent.__new__(GreedyAnchorLLMAgent)
    assert a._move_ledger_card([{"performance": 1, "correctness": 1, "flat": {}}], None) == ""


# ── failure-ledger loudness (07-08 user rule) ─────────────────────────────

def test_anomalous_failure_rows_warn_and_tally(tmp_path, caplog):
    """INVALID/DUPLICATE never consume budget (two-counter rule) but must be
    LOUD: per-occurrence warning + completion-summary tally."""
    import logging
    from rag_stack.controller import Controller
    from rag_stack.types import EvalType

    c = Controller.__new__(Controller)
    c.project_dir = str(tmp_path)
    c._failure_counts = {}
    caplog.set_level(logging.WARNING, logger="RAG-Stack")
    c._save_eval_record(
        eval_id=7, config={}, pipeline_config={},
        eval_type=EvalType.DUPLICATE,
        eval_error="config matches a previously-recorded invalid config",
        objectives={}, quality_metrics=None, assembly_result=None,
        obj_columns=[],
    )
    c._save_eval_record(
        eval_id=8, config={}, pipeline_config={},
        eval_type=EvalType.CONTEXT_OVERFLOW, eval_error="ctx",
        objectives={}, quality_metrics=None, assembly_result=None,
        obj_columns=[],
    )
    assert c._failure_counts == {"DUPLICATE": 1, "CONTEXT_OVERFLOW": 1}
    warns = [r for r in caplog.records if "DUPLICATE" in r.message]
    assert warns and "budget-free" in warns[0].message
    # infeasibility rows are info-level (normal), no warning
    assert not any("CONTEXT_OVERFLOW" in r.message for r in caplog.records
                   if r.levelno >= logging.WARNING)
