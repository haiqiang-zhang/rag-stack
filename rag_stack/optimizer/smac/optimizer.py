"""Official SMAC MOBO baseline optimizer (non-decoupled).

Uses SMAC3's recommended HyperparameterOptimizationFacade with ParEGO
multi-objective scalarization via the ask/tell interface.

Non-decoupled: every iteration evaluates ALL objectives (GT only, cm=0).
Used as a baseline to compare against the decoupled approach.

SMAC minimizes by default, so maximization objectives are negated before
being passed to SMAC and negated back for get_pareto().
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from ax.core.search_space import SearchSpace

from smac import HyperparameterOptimizationFacade, Scenario
from smac.multi_objective.parego import ParEGO
from smac.runhistory.dataclasses import TrialInfo, TrialValue

from ..base import BaseOptimizer
from ..utils import config_signature as _config_signature
from .configspace import (
    ax_to_configspace,
    cs_config_to_dict,
    dict_to_cs_config,
)

logger = logging.getLogger("RAG-Stack")


def _pareto_mask(Y: np.ndarray) -> np.ndarray:
    """Boolean mask of non-dominated rows (maximization)."""
    n = len(Y)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        dominated = np.all(Y[i] >= Y, axis=1) & np.any(Y[i] > Y, axis=1)
        dominated[i] = False
        is_pareto[dominated] = False
    return is_pareto


class SMACMOBOOptimizer(BaseOptimizer):
    """Non-decoupled MOBO baseline using official SMAC3 facade (ParEGO).

    Every suggest() call returns all objective indices, so the runner
    evaluates all objectives and counts it as a GT call. cm=0 always.

    Args:
        config_space: Ax SearchSpace.
        objectives:   List of objective dicts with keys 'objective_name',
                      'cost', 'ref_point'.
        seed:         Random seed.
        n_gt_init:    Number of random initialization configs before
                      surrogate-guided search begins.
        ssops:        SearchSpace naming-convention ops the DOWNSTREAM consumer
                      decodes with (controller passes its RagSearchSpace ops).
                      MUST be forwarded to ``cs_config_to_dict`` — the generic
                      fallback nests RAG params wrong (no vectordb namespace
                      lift), and the controller then fails every config with
                      "Unresolved sweep parameter".
    """

    def __init__(
        self,
        config_space: SearchSpace,
        objectives: List[dict],
        seed: int = 42,
        n_gt_init: int = 10,
        ssops=None,
    ):
        self.search_space = config_space
        # Decoder for suggest() output — must match what the controller expects
        # (see class docstring). None falls back to the generic 2-segment decode
        # inside cs_config_to_dict (benchmark/synthetic callers only).
        self._ssops = ssops
        self.objectives = objectives
        self.objective_costs = {obj["objective_name"]: obj["cost"] for obj in objectives}
        self.seed = seed
        self.objective_names = [obj["objective_name"] for obj in objectives]
        # Required BaseOptimizer contract attr (controller reads it directly for
        # the Phase 1 init batch + the exhaustive-shortcut threshold). Equals
        # SMAC's initial-design size so the controller requests exactly the
        # random configs SMAC's initial design emits via ask() before the
        # model-based phase begins.
        self.n_both_init: int = int(n_gt_init)

        self.cs = ax_to_configspace(config_space, seed=seed)

        scenario = Scenario(
            configspace=self.cs,
            objectives=self.objective_names,
            n_trials=100_000,   # runner controls actual iteration count
            seed=seed,
        )

        # Wrap target_function with correct signature; unused since we use ask/tell
        def _dummy_target(config, seed):
            return {name: 0.0 for name in self.objective_names}

        # Default intensifier has ``max_config_calls=3`` which re-evaluates
        # each config up to 3 times — meant for noisy benchmarks but wastes
        # 2/3 of our deterministic GT budget on duplicates.  Force one-shot.
        intensifier = HyperparameterOptimizationFacade.get_intensifier(
            scenario, max_config_calls=1
        )

        self._smac = HyperparameterOptimizationFacade(
            scenario,
            _dummy_target,
            multi_objective_algorithm=ParEGO(scenario=scenario, seed=seed),
            initial_design=HyperparameterOptimizationFacade.get_initial_design(
                scenario, n_configs=n_gt_init
            ),
            intensifier=intensifier,
            overwrite=True,
        )

        # Bookkeeping for get_pareto()
        self._configs: List[dict] = []
        self._obj_values: List[List[float]] = []
        self._eval_ids: List[int] = []

        # Pending trial from the last ask()
        self._pending_trial: Optional[TrialInfo] = None

        # Checkpoint-resume state (see warm_start): signature → negated cost
        # list for every injected historical eval, and the set of signatures
        # SMAC re-proposed after injection (answered from history, not
        # re-evaluated).
        self._warm_costs: Dict[str, List[float]] = {}
        self._warm_resuggested: set = set()

        logger.info(
            f"SMACMOBOOptimizer: dim={len(config_space.parameters)}, "
            f"n_gt_init={n_gt_init}, seed={seed}, objectives={self.objective_names}"
        )

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    def suggest(self, n_suggestions: int = 1) -> Tuple[Optional[List[dict]], Optional[List[str]]]:
        """Ask SMAC for the next configuration (all objectives = GT).

        ``n_suggestions`` is intentionally ignored: this wrapper runs a strict
        one-at-a-time ask/tell protocol (a single ``_pending_trial``;
        ``add_objectives`` tells exactly one result back). When the controller's
        Phase 1 asks for ``n_both_init`` configs it gets 1 — the remaining
        initial-design configs are simply served one per Phase 2 iteration by
        subsequent ``ask()`` calls, so the total GT budget and the random-init →
        model-based sequence are unchanged.

        Resume guard: after ``warm_start`` the same-seed initial design can
        re-propose configs whose results are already in the checkpoint. Those
        are answered from history (told back to SMAC) instead of being
        returned for a fresh GT evaluation — resuming must not re-burn
        completed trials. Bounded loop: each replay consumes one proposal.
        """
        for _ in range(len(self._warm_costs) + self.n_both_init + 5):
            trial_info = self._smac.ask()
            config_dict = cs_config_to_dict(
                trial_info.config, self.search_space, ssops=self._ssops
            )
            sig = _config_signature(config_dict)
            known = self._warm_costs.get(sig)
            if known is not None and sig not in self._warm_resuggested:
                try:
                    self._smac.tell(trial_info, TrialValue(cost=list(known), time=0.0))
                    self._warm_resuggested.add(sig)
                    logger.info(
                        "SMAC resume: re-proposed an already-evaluated config; "
                        "answered from checkpoint history instead of re-running"
                    )
                    continue
                except Exception as exc:  # noqa: BLE001 — dedup is best-effort
                    logger.warning(
                        f"SMAC resume: replay-tell failed ({exc}); "
                        "returning the config for a fresh evaluation"
                    )
            self._pending_trial = trial_info
            return [config_dict], list(self.objective_names)
        # Pathological fallback: stop deduping rather than loop forever.
        self._pending_trial = trial_info
        return [config_dict], list(self.objective_names)

    def warm_start(self, evaluations: List[Dict]) -> None:
        """Inject checkpoint evaluations into SMAC's runhistory (resume).

        DATA injection, mirroring the GP optimizers' warm_start semantics —
        not an ask/tell sequence replay (the checkpoint legitimately has holes
        where trials went INVALID, so the original chain cannot be replayed).
        Each historical (config, objectives) pair is told to SMAC as a
        finished trial via ``tell`` without a prior ``ask`` (supported SMAC3
        warmstarting path); the RF/ParEGO machinery trains on them at the next
        ``ask()``. If the same-seed initial design later re-proposes an
        injected config, ``suggest`` answers it from history instead of
        re-burning a GT evaluation.

        SMAC is non-decoupled: rows missing any objective (e.g. CM-only rows
        from decoupled runs) are skipped. Raises when NOTHING could be
        injected so the controller falls back to its "starting fresh" path
        loudly rather than silently resuming with an empty model.
        """
        injected = 0
        skipped = 0
        # SMAC's intensifier draws its per-instance seed from the first
        # RandomState(scenario.seed) draw; tell under that SAME seed so the
        # intensifier natively recognizes restored configs as evaluated
        # (insight grafted from the parallel implementation, 6434e1c4 — the
        # suggest() history-answer guard stays as belt-and-suspenders).
        trial_seed = int(np.random.RandomState(self.seed).randint(0, 2**31 - 1))
        for ev in sorted(evaluations, key=lambda e: int(e.get("eval_id", -1))):
            objectives = ev.get("objectives") or {}
            values: Dict[str, float] = {}
            for name in self.objective_names:
                v = objectives.get(name)
                if isinstance(v, dict):
                    v = v.get("combined")
                if v is None:
                    break
                values[name] = float(v)
            if len(values) != len(self.objective_names):
                skipped += 1
                continue
            try:
                cs_config = dict_to_cs_config(
                    ev["config"], self.cs, self.search_space, ssops=self._ssops
                )
            except Exception as exc:  # noqa: BLE001 — one bad row must not kill resume
                skipped += 1
                logger.warning(
                    f"SMAC warm_start: eval {ev.get('eval_id')} config could not "
                    f"be re-encoded to ConfigSpace ({exc}); skipping"
                )
                continue
            cost = [-values[name] for name in self.objective_names]
            try:
                self._smac.tell(
                    TrialInfo(config=cs_config, seed=trial_seed),
                    TrialValue(cost=cost, time=0.0),
                )
            except Exception as exc:  # noqa: BLE001 — e.g. duplicate trial
                skipped += 1
                logger.warning(
                    f"SMAC warm_start: tell failed for eval {ev.get('eval_id')} "
                    f"({exc}); skipping"
                )
                continue
            sig = _config_signature(ev["config"])
            self._warm_costs[sig] = cost
            self._configs.append(ev["config"])
            self._obj_values.append(
                [values[name] for name in self.objective_names]
            )
            self._eval_ids.append(int(ev.get("eval_id", -1)))
            injected += 1
        if injected == 0:
            raise RuntimeError(
                f"SMAC warm_start: none of the {len(evaluations)} checkpoint "
                f"evaluations could be injected (skipped={skipped})"
            )
        logger.info(
            f"SMAC warm_start: injected {injected}/{len(evaluations)} "
            f"evaluations into the runhistory (skipped={skipped}); "
            f"runhistory now has {len(self._smac.runhistory)} trial(s)"
        )

    def add_objectives(
        self,
        x: List[dict],
        obj_dict: Dict[str, List[float]],
        eval_ids: Optional[List[int]] = None,
    ) -> None:
        """Tell SMAC the objective values for the last suggested config.

        SMAC minimizes, so we negate our maximization objectives before telling.
        """
        if self._pending_trial is None:
            logger.warning("add_objectives called without a pending trial; skipping")
            return

        # Build negated cost list in objective order (SMAC minimizes → negate our maximization values)
        cost = [-(obj_dict[name][0]["combined"] if isinstance(obj_dict[name][0], dict) else obj_dict[name][0]) for name in self.objective_names]

        trial_value = TrialValue(cost=cost, time=0.0)
        self._smac.tell(self._pending_trial, trial_value)
        self._pending_trial = None

        # Store raw (non-negated) values for get_pareto()
        self._configs.append(x[0])
        vals = [
            (v[0]["combined"] if isinstance(v[0], dict) else v[0])
            for name in self.objective_names
            for v in [obj_dict.get(name, [float("nan")])]
        ]
        self._obj_values.append(vals)
        self._eval_ids.append(eval_ids[0] if eval_ids else -1)

    def get_pareto(self) -> Tuple[List[dict], List[Dict[str, float]], List[int]]:
        if not self._configs:
            return [], [], []

        obj_arr = np.array(self._obj_values)
        mask = _pareto_mask(obj_arr)
        pareto_configs = [self._configs[i] for i in range(len(self._configs)) if mask[i]]
        pareto_objs = [
            {name: self._obj_values[i][j] for j, name in enumerate(self.objective_names)}
            for i in range(len(self._obj_values)) if mask[i]
        ]
        pareto_eval_ids = [self._eval_ids[i] for i in range(len(self._eval_ids)) if mask[i]]
        return pareto_configs, pareto_objs, pareto_eval_ids
