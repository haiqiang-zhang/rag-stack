"""SLOAgentGuidedQNEHVI (``agent_qnehvi_slo``) — suggest-time SLO gating.

Drives the always-Agent optimizer with self-contained Agent and cm_probe test
doubles over the real hierarchical RAG search space, and asserts:

  * init (Sobol) batches are rejection-sampled to CM-feasible configs;
  * BO-phase suggestions are never known-infeasible per the probe;
  * probe verdicts are cached by config_hash (probe called once per config);
  * a probe that RAISES marks the config infeasible;
  * a missing cm_probe fails fast at construction.

Run: python -m pytest tests/optimizer/test_slo_agent_qnehvi.py -x
"""

from tests import REPO_ROOT

from pathlib import Path

import pytest
import yaml


from rag_stack.search_space import RagSearchSpace
from rag_stack.optimizer import OPTIMIZER_REGISTRY
from rag_stack.optimizer.botorch import SLOAgentGuidedQNEHVI
from rag_stack.optimizer.utils import config_hash

_CONFIG_PATH = (
    REPO_ROOT
    / "configs" / "rag_stack" / "paper_bench_agent_s43.yaml"
)

_OBJECTIVES = [
    {"objective_name": "performance", "ref_point": 0, "cost": 1},
    {"objective_name": "quality", "ref_point": 0, "cost": 10},
]

# Sentinel matching cost_model.selection.INFEASIBLE_SCORE.
_INFEASIBLE = -1e9


def _ssops() -> RagSearchSpace:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return RagSearchSpace.from_yaml(cfg)


def _stub_feasible(cfg: dict) -> bool:
    """Deterministic ~50/50 split keyed on the config's own hash."""
    return int(config_hash(cfg), 16) % 2 == 0


class _Probe:
    """Stub cm_probe recording every call (for cache assertions)."""

    def __init__(self):
        self.calls = []

    def __call__(self, cfg: dict) -> float:
        self.calls.append(config_hash(cfg))
        return 100.0 if _stub_feasible(cfg) else _INFEASIBLE


class _NoopAgent:
    """Self-contained Agent double; SLO behavior is the subject of these tests."""

    def advise(self, history, ssops, t, mo):
        del history, ssops, t, mo
        return None


def _make_opt(probe, ssops, **kw):
    params = dict(
        objectives=_OBJECTIVES,
        seed=7,
        n_both_init=4,
        ssops=ssops,
        n_candidates=64,            # small pool for test speed
        cm_probe=probe,
        slo_max_probes_per_suggest=64,   # ample → nothing slips through unprobed
    )
    params.update(kw)
    optimizer = SLOAgentGuidedQNEHVI(ssops.ss, **params)
    optimizer.agent = _NoopAgent()
    return optimizer


def test_registry_exposes_agent_qnehvi_slo():
    assert OPTIMIZER_REGISTRY["agent_qnehvi_slo"] is SLOAgentGuidedQNEHVI


def test_missing_probe_fails_fast():
    ssops = _ssops()
    with pytest.raises(ValueError, match="cm_probe"):
        _make_opt(None, ssops)


def test_init_and_bo_suggestions_are_feasible():
    ssops = _ssops()
    probe = _Probe()
    opt = _make_opt(probe, ssops)

    # ── init phase: Sobol batch is rejection-sampled through the probe ──
    configs, objs = opt.suggest(4)
    assert objs == ["performance", "quality"]
    assert len(configs) == 4
    for cfg in configs:
        assert _stub_feasible(cfg), "init suggested a CM-infeasible config"

    # cache: one probe per distinct config, verdicts recorded
    assert len(probe.calls) == len(set(probe.calls))
    assert opt._slo_cache, "probe verdicts must be cached"
    # Rejection-existence proof. The hash-parity stub is ~50/50 per config,
    # but one seed's small init batch can be all-feasible by luck (and which
    # configs Sobol draws shifts whenever the search space's shape changes),
    # so scan a few seeds instead of pinning one seed's draw parity — every
    # seed still asserts the gate (feasible-only suggestions) above/below.
    def _saw_rejection(o) -> bool:
        return any(v is False for v in o._slo_cache.values())

    saw = _saw_rejection(opt)
    for extra_seed in (11, 13, 17, 23):
        if saw:
            break
        alt = _make_opt(_Probe(), ssops, seed=extra_seed)
        alt_configs, _ = alt.suggest(4)
        for cfg in alt_configs:
            assert _stub_feasible(cfg), "init suggested a CM-infeasible config"
        saw = _saw_rejection(alt)
    assert saw, (
        "no seed in (7, 11, 13, 17, 23) saw a single stub rejection — the "
        "SLO gate appears to never reject"
    )

    # feed synthetic objectives so the BO phase has data
    opt.add_objectives(
        configs,
        {
            "performance": [100.0, 90.0, 80.0, 70.0],
            "quality": [0.2, 0.5, 0.7, 0.4],
        },
        eval_ids=list(range(4)),
    )

    # ── BO phase: acqf-ranked walk must skip infeasible candidates ──
    for t in range(2):
        n_calls_before = len(probe.calls)
        [cfg], _ = opt.suggest(1)
        assert _stub_feasible(cfg), f"BO iter {t} suggested infeasible config"
        # probes are fresh-only: cached configs never re-probed
        assert len(probe.calls) == len(set(probe.calls))
        assert len(probe.calls) - n_calls_before <= opt.slo_max_probes_per_suggest
        opt.add_objectives(
            [cfg], {"performance": [60.0 - t], "quality": [0.6 + 0.1 * t]},
            eval_ids=[4 + t],
        )

    pareto_cfgs, pareto_objs, _ = opt.get_pareto()
    assert pareto_cfgs and len(pareto_cfgs) == len(pareto_objs)


def test_probe_exception_means_infeasible():
    ssops = _ssops()
    seen = {}

    def exploding_probe(cfg):
        seen[config_hash(cfg)] = True
        raise RuntimeError("model too large for GPU count")

    opt = _make_opt(exploding_probe, ssops, slo_init_probe_factor=2)
    # every probe raises → all verdicts cached as infeasible, but the batch
    # is still filled (unfiltered fallback) so the loop never deadlocks.
    configs, _ = opt.suggest(4)
    assert len(configs) == 4
    assert opt._slo_cache and all(v is False for v in opt._slo_cache.values())
    assert seen, "probe should have been consulted"


def test_budget_exhaustion_allows_unprobed_candidate():
    """With a zero fresh-probe budget the optimizer must still produce a
    batch (unknown verdicts pass through; eval-time selection catches them)."""
    ssops = _ssops()
    probe = _Probe()
    opt = _make_opt(
        probe, ssops, slo_max_probes_per_suggest=0, slo_init_probe_factor=1,
    )
    opt._probes_left = 0
    opt.slo_init_probe_factor = 0  # force zero budget even in init
    configs, _ = opt.suggest(4)
    assert len(configs) == 4
