"""SLO-gated agent-guided qNEHVI — CM-prefiltered candidate generation.

``agent_qnehvi`` plus a hard performance SLO enforced at SUGGESTION time:
before a candidate can be acqf-scored / suggested, the STATIC cost model is
consulted (config-driven RAGO path — ``resolve_config`` +
``RAGCMAssembly.evaluate_static``; at suggest time no quality run has
happened, so no rag_ir trace exists and the trace path is unavailable by
construction) and SLO-infeasible configs are rejected.

The SLO itself is NOT a new config surface: it lives where it always has, in
``system.performance_objectives`` with ``selection: constrained`` +
``constraint`` (natural sense: ``latency_s ≤ L*``, ``throughput_qps ≥ Q*``) +
``optimize``. The probe simply replays the run's own selection policy and
tests for the ``INFEASIBLE_SCORE`` sentinel, so suggest-time gating and
eval-time scoring can never disagree about what "meets the SLO" means.

Why gate at suggest time when ``select_constrained`` already feeds
INFEASIBLE_SCORE to the GP? Because by then the GT (LLM-judge) quality eval —
~10× the CM cost — has already been spent. Gating up front means the expensive
quality evaluation only ever runs on configs that CAN meet the SLO. This is
only meaningful for the static quality-only flow
(``performance_source: cost_model`` + ``gt_evaluation.backend: static_gt``);
the controller fail-fast-validates that before building this optimizer.

Cost control — one CM probe is a full RAGO sweep (seconds) and the acqf pool
is ~768 candidates/iter, so probing the whole pool would dwarf the BO step:

  * every probe verdict is cached by ``config_hash`` for the whole run;
  * the pool is prefiltered with CACHED verdicts only — known-infeasible
    configs never reach acqf scoring, at zero fresh CM calls;
  * FRESH probes happen lazily, walking DOWN the acqf ranking, capped at
    ``slo_max_probes_per_suggest`` per BO step — CM time is spent only on
    candidates the acqf actually rates highly;
  * on budget exhaustion the best unprobed candidate is suggested anyway
    (logged) — the run's ``constrained`` selection still catches it at eval
    time, so a slip-through is penalized, never silently accepted.

Init (Sobol) batches are rejection-sampled through the same probe so the
``n_both_init`` GT evaluations are not wasted on SLO-violating configs.
"""

import logging
from typing import Callable, Dict, List, Optional

from torch import Tensor
import torch

from rag_stack.optimizer.agent import AgentGuidedQNEHVI
from ..utils import config_hash

logger = logging.getLogger("RAG-Stack")

# Cutoff matching rag_stack.cost_model.selection.INFEASIBLE_SCORE (-1e9):
# any probe score at or below this is the infeasible sentinel. Deliberately
# NOT imported — importing the cost_model package auto-imports every
# cost-model submodule (heavy), and the probe callable must stay this
# optimizer's only cost-model coupling. Real scores (qps, -latency) are
# orders of magnitude above this.
_INFEASIBLE_CUTOFF = -1e8


class SLOAgentGuidedQNEHVI(AgentGuidedQNEHVI):
    """AgentGuidedQNEHVI whose suggestions are pre-screened by a CM SLO probe.

    Args (on top of AgentGuidedQNEHVI):
        cm_probe: ``decoded_config -> float`` — static cost-model performance
            score under the run's selection policy. Configs scoring the
            INFEASIBLE sentinel (or raising — the CM is deterministic, a
            config that cannot deploy can never meet the SLO) are rejected.
            The controller injects ``PerformanceContext.cm_performance_probe``.
        slo_max_probes_per_suggest: cap on FRESH CM probes per BO suggest
            (cached verdicts are free and unlimited).
        slo_init_probe_factor: during the Sobol init phase the probe budget
            is ``factor × batch_size`` per batch (rejection sampling needs
            headroom; every verdict lands in the run-wide cache).
    """

    def __init__(
        self,
        config_space,
        objectives: List[dict],
        cm_probe: Optional[Callable[[dict], float]] = None,
        slo_max_probes_per_suggest: int = 8,
        slo_init_probe_factor: int = 5,
        **kwargs,
    ):
        super().__init__(config_space, objectives, **kwargs)
        if cm_probe is None:
            raise ValueError(
                "agent_qnehvi_slo requires cm_probe (decoded config -> CM "
                "performance score). The controller injects it automatically "
                "when optimizer.type=agent_qnehvi_slo; standalone users must "
                "pass cm_probe= explicitly."
            )
        self._cm_probe = cm_probe
        self.slo_max_probes_per_suggest = int(slo_max_probes_per_suggest)
        self.slo_init_probe_factor = max(1, int(slo_init_probe_factor))
        self._slo_cache: Dict[str, bool] = {}   # config_hash -> feasible?
        self._probes_left = 0                    # per-suggest fresh-probe budget
        self._n_probes_total = 0

    # ------------------------------------------------------------------
    # probe + cache
    # ------------------------------------------------------------------

    def _feasible(self, cfg: dict) -> Optional[bool]:
        """SLO verdict for a decoded config.

        True/False from cache or a fresh CM probe; ``None`` = unknown (fresh-
        probe budget exhausted — never blocks a suggestion). A probe that
        RAISES marks the config infeasible: the cost model is deterministic
        (project-wide no-retry policy), and a config that cannot even find a
        feasible deployment can never meet the SLO.
        """
        key = config_hash(cfg)
        if key in self._slo_cache:
            return self._slo_cache[key]
        if self._probes_left <= 0:
            return None
        self._probes_left -= 1
        self._n_probes_total += 1
        try:
            feas = float(self._cm_probe(cfg)) > _INFEASIBLE_CUTOFF
        except Exception as exc:
            logger.warning(
                f"SLO probe raised (config treated as infeasible): {exc}"
            )
            feas = False
        self._slo_cache[key] = feas
        return feas

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    def suggest(self, n_suggestions: int = 1):
        # reset the fresh-probe budget per suggest; the init path sizes its
        # own (larger) budget inside _sobol_batch.
        if len(self._configs) >= self.n_both_init:
            self._probes_left = self.slo_max_probes_per_suggest
        before = self._n_probes_total
        out = super().suggest(n_suggestions)
        used = self._n_probes_total - before
        if used:
            n_bad = sum(1 for v in self._slo_cache.values() if not v)
            logger.info(
                f"SLO: {used} fresh CM probe(s) this suggest; cache holds "
                f"{len(self._slo_cache)} verdicts ({n_bad} infeasible)"
            )
        return out

    # ------------------------------------------------------------------
    # feasibility hooks (called by the base class)
    # ------------------------------------------------------------------

    def _prefilter_pool(self, cand: Tensor) -> Tensor:
        """Drop known-infeasible candidates BEFORE acqf scoring — cached
        verdicts only, zero fresh CM calls (pool is ~768 points/iter)."""
        if not self._slo_cache:
            return cand
        keep, n_drop = [], 0
        for i in range(cand.shape[0]):
            cfg = self.ssops.decode_to_config(cand[i])
            if self._slo_cache.get(config_hash(cfg)) is False:
                n_drop += 1
            else:
                keep.append(i)
        if not keep:
            # entire pool known-infeasible — keep it so the BO step stays
            # alive; the final-gate fallback will log the slip-through.
            logger.warning(
                "SLO: entire candidate pool is known-infeasible; skipping "
                "pool prefilter this iter"
            )
            return cand
        if n_drop:
            logger.info(
                f"SLO: dropped {n_drop}/{cand.shape[0]} known-infeasible "
                "pool candidates before acqf scoring"
            )
            cand = cand[torch.tensor(keep, dtype=torch.long)]
        return cand

    def _accept_candidate(self, x: Tensor, cfg: dict) -> bool:
        if not super()._accept_candidate(x, cfg):
            return False
        verdict = self._feasible(cfg)
        if verdict is None:
            logger.info(
                "SLO: fresh-probe budget exhausted this suggest; accepting the "
                "acqf-best unprobed candidate (eval-time constrained selection "
                "still penalizes it if infeasible)"
            )
            return True
        return verdict

    # ------------------------------------------------------------------
    # init phase
    # ------------------------------------------------------------------

    _MAX_SOBOL_ROUNDS = 10

    def _sobol_batch(self, n: int) -> List[dict]:
        """Rejection-sample CM-feasible Sobol configs.

        Budgeted at ``slo_init_probe_factor × n`` fresh probes per call; on
        exhaustion (or a mostly-infeasible space) the batch is topped up with
        unprobed/unfiltered samples so the loop can never deadlock.
        """
        self._probes_left = max(
            self._probes_left, self.slo_init_probe_factor * n
        )
        feasible: List[dict] = []
        unknown: List[dict] = []
        n_rejected = 0
        for _ in range(self._MAX_SOBOL_ROUNDS):
            for cfg in super()._sobol_batch(n):
                verdict = self._feasible(cfg)
                if verdict is False:
                    n_rejected += 1
                elif verdict is None:
                    unknown.append(cfg)   # budget exhausted — last resort
                else:
                    feasible.append(cfg)
                if len(feasible) >= n:
                    break
            if len(feasible) >= n or self._probes_left <= 0:
                break
        if n_rejected:
            logger.info(
                f"SLO init: rejected {n_rejected} CM-infeasible Sobol "
                f"config(s) while drawing {n}"
            )
        if len(feasible) < n:
            short = n - len(feasible)
            logger.warning(
                f"SLO init: only {len(feasible)}/{n} CM-feasible configs "
                f"found; filling {short} slot(s) with unfiltered samples"
            )
            feasible += unknown[:short]
            while len(feasible) < n:
                feasible += super()._sobol_batch(n - len(feasible))
        return feasible[:n]
