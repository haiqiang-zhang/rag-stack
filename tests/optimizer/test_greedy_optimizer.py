"""GreedyParetoOptimizer — the budget-aware two-mode baseline.

Pins the plan's hierarchy iron rules and budget semantics on the REAL
paper_bench space (rag_dataflow fork, dict-form vdb family selector) with a
deterministic stub evaluator — no controller, no GP, no GPU.

Run: python -m pytest tests/optimizer/test_greedy_optimizer.py -x -q
"""
from __future__ import annotations

from tests import REPO_ROOT

import hashlib
import json
from pathlib import Path

import pytest
import yaml


from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.optimizer import OPTIMIZER_REGISTRY
from rag_stack.optimizer.greedy import (
    GreedyParetoOptimizer,
    _largest_remainder,
)
from rag_stack.search_space import SEP, RagSearchSpace

_CONFIG_PATH = (
    REPO_ROOT
    / "configs" / "rag_stack" / "paper_bench_agent_s44.yaml"
)

_DF = f"pipeline{SEP}sequential{SEP}rag_dataflow"
_MAXITER = f"pipeline{SEP}sequential{SEP}react{SEP}max_iter"
_SEL = f"pipeline{SEP}sequential{SEP}semantic_retrieval{SEP}vectordb{SEP}vectordb"
_HNSW_M = f"vectordb{SEP}faiss_hnsw_index{SEP}M"
_QE_STAGE = f"pipeline{SEP}sequential{SEP}query_expansion"
_RETR_STAGE = f"pipeline{SEP}sequential{SEP}semantic_retrieval"

_OBJS = [
    {"objective_name": "performance", "cost": 1.0, "ref_point": 0.0},
    {"objective_name": "quality", "cost": 10.0, "ref_point": 0.0},
]


def _ssops() -> RagSearchSpace:
    with open(_CONFIG_PATH) as f:
        return RagSearchSpace.from_yaml(yaml.safe_load(f))


def _hash_stub(nested: dict) -> dict:
    h = int(hashlib.sha256(
        json.dumps(nested, sort_keys=True, default=str).encode()
    ).hexdigest()[:10], 16)
    return {
        "performance": [(h % 1000) / 10.0],
        "quality": [{"combined": (h // 1000 % 1000) / 1000.0}],
    }


def _drive(g, stub, cap=10_000) -> int:
    n = 0
    while n < cap:
        cfgs, names = g.suggest()
        if cfgs is None:
            break
        assert names == ["performance", "quality"]
        g.add_objectives(cfgs, stub(cfgs[0]), eval_ids=[n])
        n += 1
    return n


def _tiny_space() -> SearchSpace:
    return SearchSpace(parameters=[
        ChoiceParameter(
            "corpus__chunker__chunk_size", ParameterType.INT,
            values=[128, 256], is_ordered=True, sort_values=True,
        ),
        ChoiceParameter(
            "pipeline__sequential__generator__vllm__model",
            ParameterType.STRING, values=["a", "b"],
            is_ordered=False, sort_values=False,
        ),
    ])


# ---------------------------------------------------------------------------
# construction / registry / helpers
# ---------------------------------------------------------------------------

def test_registry_and_param_validation():
    assert OPTIMIZER_REGISTRY["greedy"] is GreedyParetoOptimizer
    ss = _tiny_space()
    with pytest.raises(ValueError, match="search_mode"):
        GreedyParetoOptimizer(ss, _OBJS, search_mode="sideways")
    with pytest.raises(ValueError, match="stage_weight"):
        GreedyParetoOptimizer(ss, _OBJS, stage_weight="sqrt")
    with pytest.raises(ValueError, match="pass_budget_fraction"):
        GreedyParetoOptimizer(ss, _OBJS, pass_budget_fraction=0.0)



def test_largest_remainder_fair_and_capped():
    out = _largest_remainder(
        27,
        weights={"a": 3.46, "b": 5.83, "c": 6.77, "d": 5.64,
                 "e": 5.21, "f": 2.32, "g": 2.0, "h": 4.09},
        floors={k: 2 for k in "abcdefgh"},
        caps={"a": 10, "b": 56, "c": 108, "d": 49,
              "e": 36, "f": 4, "g": 3, "h": 16},
    )
    assert sum(out.values()) == 27
    assert all(v >= 2 for v in out.values())          # floors honored
    assert out["c"] >= out["g"]                        # weight order respected
    # caps: a 2-combo stage can never be allocated 3
    out2 = _largest_remainder(10, {"x": 1.0, "y": 1.0},
                              floors={"x": 2, "y": 2}, caps={"x": 2, "y": 8})
    assert out2["x"] == 2 and out2["y"] == 8


# ---------------------------------------------------------------------------
# fork + stage decomposition (iron rules 3-5)
# ---------------------------------------------------------------------------

def test_fork_detected_on_paper_space_and_absent_on_flat_space():
    ss = _ssops()
    g = GreedyParetoOptimizer(ss.ss, _OBJS, ssops=ss, n_iterations=10)
    assert g._fork == _DF
    assert len(g._branch_ctxs) == 2
    assert all(_DF not in params for params in g._stage_params.values())

    g2 = GreedyParetoOptimizer(_tiny_space(), _OBJS, n_iterations=4)
    assert g2._fork is None
    assert len(g2._branch_ctxs) == 1


def test_stage_topo_order_gaters_before_gated():
    ss = _ssops()
    g = GreedyParetoOptimizer(ss.ss, _OBJS, ssops=ss, n_iterations=10)
    order = {s: i for i, s in enumerate(g._stages)}
    # the vdb family selector's stage precedes both gated family stages
    assert order[_RETR_STAGE] < order[f"vectordb{SEP}faiss_ivf_index"]
    assert order[_RETR_STAGE] < order[f"vectordb{SEP}faiss_hnsw_index"]
    assert order["corpus__chunker"] == 0


def test_local_assignments_context_live_and_no_cartesian_blowup():
    ss = _ssops()
    g = GreedyParetoOptimizer(ss.ss, _OBJS, ssops=ss, n_iterations=10)
    seq = {_DF: "sequential"}
    react = {_DF: "react"}
    # QE: exactly the 49 hierarchy-valid assignments (off + 3 methods × 4×4),
    # NOT the 24,576 cartesian product (which would blow the enum cap).
    qe = g._local_assignments(_QE_STAGE, seq)
    assert len(qe) == 49
    # react-only stage: empty under sequential, 4 max_iter options under react
    assert g._local_assignments(f"pipeline{SEP}sequential{SEP}react", seq) == []
    assert len(g._local_assignments(f"pipeline{SEP}sequential{SEP}react", react)) == 4
    # vdb_hnsw stage inactive under the (default) ivf context
    assert g._local_assignments(f"vectordb{SEP}faiss_hnsw_index", seq) == []


# ---------------------------------------------------------------------------
# forward mode on the real space
# ---------------------------------------------------------------------------

def test_forward_full_budget_branches_and_react_tuned():
    ss = _ssops()
    g = GreedyParetoOptimizer(
        ss.ss, _OBJS, ssops=ss, search_mode="forward", n_iterations=50,
    )
    n = _drive(g, _hash_stub)
    assert n == 50 and len(g._archive) == 50
    # unique canonical keys (phantom regression)
    assert len({it["key"] for it in g._archive.values()}) == 50
    items = list(g._archive.values())
    # the first two archived configs are the two branch bases
    first_flats = [dict(items[0]["flat"]), dict(items[1]["flat"])]
    assert {str(f.get(_DF)) for f in first_flats} == {"sequential", "react"}
    # branch budgets ∝ log-weight sums: sequential gets more, react substantial
    per_branch = {}
    react_iters = set()
    for it in items:
        b = str(dict(it["flat"]).get(_DF))
        per_branch[b] = per_branch.get(b, 0) + 1
        if b == "react":
            v = dict(it["flat"]).get(_MAXITER)
            if v is not None:
                react_iters.add(v)
    assert per_branch["sequential"] > per_branch["react"] >= 15
    # the previously-frozen react knob is genuinely tuned
    assert len(react_iters) >= 2
    # exhausted-after-budget: graceful None
    assert g.suggest() == (None, None)


def test_lookback_full_budget():
    ss = _ssops()
    g = GreedyParetoOptimizer(
        ss.ss, _OBJS, ssops=ss, search_mode="lookback", n_iterations=50,
    )
    n = _drive(g, _hash_stub)
    assert n == 50
    assert len({it["key"] for it in g._archive.values()}) == 50


# ---------------------------------------------------------------------------
# vdb family = beam competition (design decision A), both outcomes
# ---------------------------------------------------------------------------

def _family_stub(hnsw_perf, hnsw_qual, ivf_perf, ivf_qual):
    def stub(nested: dict) -> dict:
        blob = json.dumps(nested, sort_keys=True, default=str)
        h = int(hashlib.sha256(blob.encode()).hexdigest()[:8], 16)
        jitter = (h % 97) / 10_000.0
        if "faiss_hnsw_index" in blob:
            p, q = hnsw_perf, hnsw_qual
        else:
            p, q = ivf_perf, ivf_qual
        return {"performance": [p + jitter],
                "quality": [{"combined": q + jitter}]}
    return stub


def test_family_nondominated_gets_tuned():
    """hnsw@defaults quality-dominant → survives in the beam → its family
    stage is genuinely refined (≥2 distinct hnsw block values archived)."""
    ss = _ssops()
    g = GreedyParetoOptimizer(
        ss.ss, _OBJS, ssops=ss, search_mode="forward", n_iterations=50,
    )
    n = _drive(g, _family_stub(hnsw_perf=10.0, hnsw_qual=0.9,
                               ivf_perf=90.0, ivf_qual=0.1))
    assert n == 50
    hnsw_m = {dict(it["flat"]).get(_HNSW_M)
              for it in g._archive.values()
              if _HNSW_M in dict(it["flat"])}
    assert len(hnsw_m) >= 2, f"hnsw family never refined: M values {hnsw_m}"


def test_family_dominated_dies_at_breadth_and_budget_still_spent():
    """hnsw@defaults dominated on BOTH objectives → evicted from the beam →
    the gated hnsw FAMILY STAGE yields nothing (never refined), yet the full
    budget is still spent — the honest cross-stage greedy myopia the agent
    gets to beat. (The retrieval stage's own grid-fill may still probe a few
    hnsw-selector combos — AutoRAG grids the whole node — but the family's
    build knobs stay at defaults.)"""
    ss = _ssops()
    g = GreedyParetoOptimizer(
        ss.ss, _OBJS, ssops=ss, search_mode="forward", n_iterations=50,
    )
    n = _drive(g, _family_stub(hnsw_perf=1.0, hnsw_qual=0.01,
                               ivf_perf=90.0, ivf_qual=0.9))
    assert n == 50
    hnsw_entries = [it for it in g._archive.values()
                    if _HNSW_M in dict(it["flat"])]
    hnsw_m = {dict(it["flat"]).get(_HNSW_M) for it in hnsw_entries}
    hnsw_efc = {dict(it["flat"]).get(f"vectordb{SEP}faiss_hnsw_index{SEP}ef_construction")
                for it in hnsw_entries}
    # the family's BUILD knobs are never refined (its gated stage stays empty)
    assert len(hnsw_m) <= 1
    assert len(hnsw_efc) <= 1


# ---------------------------------------------------------------------------
# beam width / weights / degenerate spaces / legacy / warm start
# ---------------------------------------------------------------------------

def test_beam_width_one_plain_chain():
    ss = _ssops()
    g = GreedyParetoOptimizer(
        ss.ss, _OBJS, ssops=ss, search_mode="forward",
        n_iterations=30, beam_width=1,
    )
    n = _drive(g, _hash_stub)
    assert n == 30
    assert all(len(b) <= 1 for b in g._beams.values())


def test_uniform_stage_weight_smoke():
    ss = _ssops()
    g = GreedyParetoOptimizer(
        ss.ss, _OBJS, ssops=ss, search_mode="forward",
        n_iterations=30, stage_weight="uniform",
    )
    assert _drive(g, _hash_stub) == 30


def test_tiny_space_exhausts_gracefully():
    g = GreedyParetoOptimizer(
        _tiny_space(), _OBJS, search_mode="forward", n_iterations=50,
    )
    n = _drive(g, _hash_stub)
    # 4 = whole space; 3 when the dominated base is evicted from the beam and
    # its remaining descendant becomes unreachable — legitimate greedy pruning
    assert 3 <= n <= 4
    assert g.suggest() == (None, None)


def test_legacy_no_budget_fallback():
    g = GreedyParetoOptimizer(_tiny_space(), _OBJS, search_mode="lookback")
    assert g._n_iterations is None
    n = _drive(g, _hash_stub, cap=100)
    assert 3 <= n <= 4  # legacy still terminates on pool exhaustion


def test_warm_start_resume_budget():
    ss = _ssops()
    g1 = GreedyParetoOptimizer(
        ss.ss, _OBJS, ssops=ss, search_mode="forward", n_iterations=20,
    )
    assert _drive(g1, _hash_stub) == 20
    evals = [
        {"config": it["nested"],
         "objectives": {n: it[n] for n in g1.objective_names},
         "eval_id": it["eval_id"]}
        for it in g1._archive.values()
    ]
    g2 = GreedyParetoOptimizer(
        ss.ss, _OBJS, ssops=ss, search_mode="forward", n_iterations=30,
    )
    g2.warm_start(evals)
    # Replay semantics: outcomes queue for planner replay; the archive fills as
    # the first suggests drain the queue (hits) or flush it (divergence — here
    # guaranteed, since the budgets differ and quotas depend on the budget).
    assert len(g2._replay_queue) == 20 and not g2._archive
    n2 = _drive(g2, _hash_stub)
    assert n2 <= 10                     # live evals never exceed the leftover budget
    assert not g2._replay_queue         # every archived outcome replayed or flushed
    assert len(g2._archive) >= 20       # ...and none were lost
    assert len(g2._archive) <= 30


def test_get_pareto_roundtrip():
    ss = _ssops()
    g = GreedyParetoOptimizer(
        ss.ss, _OBJS, ssops=ss, search_mode="forward", n_iterations=15,
    )
    _drive(g, _hash_stub)
    configs, objs, ids = g.get_pareto()
    assert configs and len(configs) == len(objs) == len(ids)
    for o in objs:
        assert set(o) == {"performance", "quality"}


def test_beam_cap_roles_ends_plus_knee():
    """Role-based capping: k>=2 always keeps both chain ends; remaining slots
    go to the largest exclusive-HV interior contributor (the knee) — NSGA-II
    crowding would hand every slot to the ends and starve the knee."""
    ss = _tiny_space()
    g = GreedyParetoOptimizer(ss, _OBJS, n_iterations=4)
    chain = [
        {"performance": 90.0, "quality": 0.20, "flat": {"p": 1}},
        {"performance": 70.0, "quality": 0.50, "flat": {"p": 2}},  # knee (max HV rect)
        {"performance": 50.0, "quality": 0.70, "flat": {"p": 3}},
        {"performance": 30.0, "quality": 0.72, "flat": {"p": 4}},
        {"performance": 10.0, "quality": 0.75, "flat": {"p": 5}},
    ]
    def perfs(items):
        return sorted(it["performance"] for it in items)

    two = g._beam_cap(list(chain), 2)
    assert perfs(two) == [10.0, 90.0]                    # ends only
    three = g._beam_cap(list(chain), 3)
    assert perfs(three) == [10.0, 70.0, 90.0]            # ends + knee (P2)
    four = g._beam_cap(list(chain), 4)
    assert perfs(four) == [10.0, 50.0, 70.0, 90.0]       # + next contributor
    one = g._beam_cap(list(chain), 1)
    assert perfs(one) == [70.0]                          # single best contributor
    assert len(g._beam_cap(list(chain), 9)) == 5         # k >= n → all


def test_untried_locals_probed_before_repeats():
    """The anti-starvation guarantee is ORDER-FREE: within a stage, distinct
    stage-local combos get probed before any combo repeats on another base
    (branch-level visited ledger) — the per-member scan-reset bug that
    starved 7B/14B cannot recur regardless of shuffle outcome."""
    ss = _ssops()
    g = GreedyParetoOptimizer(
        ss.ss, _OBJS, ssops=ss, search_mode="forward", n_iterations=50, seed=43,
    )
    n = _drive(g, _hash_stub)
    assert n == 50
    model_key = next(
        k for k in g._params if k.endswith(f"generator{SEP}vllm{SEP}model")
    )
    tk = next(
        k for k in g._params
        if k.endswith(f"generator{SEP}vllm{SEP}temperature")
    )
    # generator stage locals = (model, temperature) pairs; count repeats
    from collections import Counter
    pairs = Counter(
        (dict(it["flat"]).get(model_key), dict(it["flat"]).get(tk))
        for it in g._archive.values()
        if dict(it["flat"]).get(model_key) is not None
    )
    # the DEFAULT pair rides along on every other stage's probe — exclude
    # it; every NON-default pair is a deliberate generator-stage probe and
    # untried-first guarantees distinct pairs before repeats (a pair count
    # >2 would mean repeats crowded out untried values)
    default_pair = max(pairs, key=pairs.get)
    non_default = {k: c for k, c in pairs.items() if k != default_pair}
    assert non_default, "generator stage never probed"
    assert all(c <= 2 for c in non_default.values()), non_default


def test_lookback_structural_invariants_and_seed_determinism():
    """Random neighbor order (MO-ParamILS) keeps every structural invariant:
    full spend, unique keys, both branches, and — same seed → identical
    trajectory / different seed → different trajectory (the ONLY stochastic
    element is the within-stratum shuffle)."""
    ss = _ssops()
    def run(seed):
        g = GreedyParetoOptimizer(
            ss.ss, _OBJS, ssops=ss, search_mode="lookback",
            n_iterations=50, seed=seed,
        )
        n = _drive(g, _hash_stub)
        keys = [it["key"] for it in g._archive.values()]
        return n, keys
    n1, k1 = run(43)
    n2, k2 = run(43)
    n3, k3 = run(44)
    assert n1 == n2 == 50 and k1 == k2          # seed-reproducible
    assert n3 == 50 and k3 != k1                # seeds genuinely differ
    assert len(set(k1)) == 50                   # no dead-knob duplicates


# ── checkpoint-resume replay (v3) ─────────────────────────────────────────

def _canon(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, default=str)


def _run_live(g, n_evals: int, hist: list, seq: list) -> int:
    """Drive n_evals LIVE evals with the hash stub, recording history + proposals."""
    n = 0
    while n < n_evals:
        cfgs, names = g.suggest()
        if cfgs is None:
            break
        assert names == ["performance", "quality"]
        objs = _hash_stub(cfgs[0])
        seq.append(_canon(cfgs[0]))
        g.add_objectives(cfgs, objs, eval_ids=[len(hist)])
        hist.append({
            "config": cfgs[0],
            "objectives": {"performance": objs["performance"][0],
                           "quality": objs["quality"][0]},
            "eval_id": len(hist),
        })
        n += 1
    return n


def test_warm_start_replay_resumes_exact_trajectory():
    """Resume = replay: a fresh optimizer warm-started from the first K
    archived outcomes must (a) re-feed them through the planner without
    proposing them live, and (b) continue with EXACTLY the original run's
    remaining proposal sequence — the property direct archive injection
    (the old warm_start) could not provide mid-run."""
    ss = _ssops()
    g1 = GreedyParetoOptimizer(ss.ss, _OBJS, ssops=ss, search_mode="forward",
                               n_iterations=24, seed=7)
    hist, seq1 = [], []
    n1 = _run_live(g1, 24, hist, seq1)
    assert n1 >= 16

    K = 11
    g2 = GreedyParetoOptimizer(ss.ss, _OBJS, ssops=ss, search_mode="forward",
                               n_iterations=24, seed=7)
    g2.warm_start(hist[:K])
    assert len(g2._replay_queue) == K and not g2._archive
    hist2, seq2 = list(hist[:K]), []
    n2 = _run_live(g2, n1 - K, hist2, seq2)
    assert n2 == n1 - K
    assert seq2 == seq1[K:], "resumed run diverged from the original suffix"
    assert set(g2._archive) == set(g1._archive)
    assert not g2._replay_queue and g2._replay_fed == K


def test_warm_start_replay_rides_over_missing_eval():
    """An eval that never landed (INVALID) is absent from the archive: replay
    must hand exactly that config out LIVE, then continue replaying the rest
    of history behind it."""
    ss = _ssops()
    g1 = GreedyParetoOptimizer(ss.ss, _OBJS, ssops=ss, search_mode="forward",
                               n_iterations=24, seed=7)
    hist, seq1 = [], []
    n1 = _run_live(g1, 14, hist, seq1)
    assert n1 == 14

    GAP = 5
    g2 = GreedyParetoOptimizer(ss.ss, _OBJS, ssops=ss, search_mode="forward",
                               n_iterations=24, seed=7)
    g2.warm_start(hist[:GAP] + hist[GAP + 1:])   # drop one mid-history outcome
    cfgs, _ = g2.suggest()
    assert cfgs is not None
    assert _canon(cfgs[0]) == seq1[GAP], "live frontier should be the gap config"
    # feed it the SAME outcome the original run measured → trajectories realign:
    # the next suggest drains the queued remainder (hist[GAP+1:]) internally and
    # returns a config beyond recorded history.
    g2.add_objectives(cfgs, _hash_stub(cfgs[0]), eval_ids=[GAP])
    cfgs2, _ = g2.suggest()
    assert cfgs2 is not None
    assert not g2._replay_queue, "remainder of history should have replayed"
    assert _canon(cfgs2[0]) not in seq1, "post-history proposal must be new"
    assert set(g2._archive) == set(g1._archive)


def test_warm_start_replay_divergence_flushes():
    """History the planner will NOT re-propose early (late-stage outcomes
    only — the situation after a code change altered the schedule): after a
    streak of unmatched proposals the queue must flush into the archive
    (old direct-injection behavior) — recorded outcomes are never lost, and
    the run continues live."""
    ss = _ssops()
    g1 = GreedyParetoOptimizer(ss.ss, _OBJS, ssops=ss, search_mode="forward",
                               n_iterations=24, seed=7)
    hist, seq1 = [], []
    _run_live(g1, 12, hist, seq1)

    late = hist[9:12]   # deep-stage outcomes; a fresh planner opens with bases
    g2 = GreedyParetoOptimizer(ss.ss, _OBJS, ssops=ss, search_mode="forward",
                               n_iterations=24, seed=7)
    g2.warm_start(late)
    assert len(g2._replay_queue) == 3
    late_keys = {g2._key(g2._flatten_config(ev["config"])) for ev in late}
    lives = 0
    while g2._replay_queue and lives < 8:
        cfgs, _ = g2.suggest()
        assert cfgs is not None
        g2.add_objectives(cfgs, _hash_stub(cfgs[0]))
        lives += 1
    assert not g2._replay_queue, "queue must flush after the miss streak"
    assert late_keys <= set(g2._archive), "flushed outcomes must land in archive"
    assert lives <= 4, "flush should trigger within the 3-miss streak"


# ── M4 sibling seeding (default OFF = baseline-identical; M2 deleted 07-07) ──

def test_m4_default_inert_and_flag_fires():
    """OFF flags must reproduce the exact baseline proposal sequence (the
    mechanism contract: prove a flag actually fires before A/B-ing it)."""
    ss = _ssops()

    def drive(g, n):
        seq = []
        for i in range(n):
            cfgs, _ = g.suggest()
            if cfgs is None:
                break
            seq.append(_canon(cfgs[0]))
            g.add_objectives(cfgs, _hash_stub(cfgs[0]), eval_ids=[i])
        return seq

    def seq(seed, **kw):
        return drive(GreedyParetoOptimizer(
            ss.ss, _OBJS, ssops=ss, search_mode="forward", n_iterations=30,
            seed=seed, **kw), 30)

    base = seq(7)
    assert seq(7, sibling_seeding=False) == base, "explicit OFF flag changed behavior!"

    # whether the endorsement reorder changes an early pick depends on the
    # process hash seed (set-order upstream of the shuffled untried list) —
    # any divergence across three seeds x 30 steps proves the flag fires.
    assert any(
        seq(s, sibling_seeding=True) != seq(s) for s in (7, 11, 13)
    ), "sibling_seeding never fired"
