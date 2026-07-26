"""Pure Sobol (quasi-random) baseline via Ax's ``AxClient``.

The trivial reference point for every model-based optimizer in the registry:
no surrogate, no acquisition function — just Sobol quasi-random sampling for
the entire budget. If a fancier optimizer (``ax_qnehvi``,
``ax_qnehvi_mixedgp``, ``agent_qnehvi``) cannot beat this on hypervolume, the
modelling adds nothing on that problem.

Why route Sobol through Ax instead of a raw ``torch.quasirandom`` sequence:
**hierarchical search spaces**. Ax's ``SearchSpace`` carries
``ChoiceParameter.dependents`` (a branch choice activates a different subset of
parameters), and Ax's Sobol generator is hierarchy-aware end-to-end:

  - ``_search_space_to_ax_parameters`` forwards each choice's ``dependents`` to
    AxClient (shared with the BoTorch ``ax_qnehvi`` optimizer).
  - ``get_next_trial()`` draws the branch choice, then only samples + returns
    the parameters active for that branch. Inactive params are omitted, so the
    nested dict ``decode_to_nested`` produces has exactly the right shape —
    same as the model-based optimizers. A flat Sobol cube would instead waste
    samples on inactive dimensions and need ad-hoc masking.

Implementation is the model-based optimizers minus the BO step: the
``GenerationStrategy`` has a single ``Generators.SOBOL`` node with
``num_trials=-1`` so it sources every trial. ``suggest`` / ``add_objectives`` /
``warm_start`` / ``get_pareto`` semantics are identical to ``AxQNEHVI``
(reused helpers), so it slots into the controller with no special-casing.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from ax.adapter.registry import Generators
from ax.core.search_space import SearchSpace
from ax.generation_strategy.generation_node import GenerationStep
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.ax_client import AxClient, ObjectiveProperties

from rag_stack.search_space.ax_search_space_ops import AxSearchSpaceOps
from .base import BaseOptimizer
from .botorch.ax_qnehvi import (
    _pareto_mask,
    _search_space_to_ax_parameters,
    _warm_start_ax_client_optimizer,
)

logger = logging.getLogger("RAG-Stack")


class Sobol(BaseOptimizer):
    """Pure Sobol quasi-random baseline (no surrogate, no acquisition).

    Hierarchy-aware: built on Ax's ``SearchSpace`` so ``ChoiceParameter``
    branch dependents are respected — each sampled trial only carries the
    parameters active for its chosen branch, identical to the model-based
    optimizers in the registry.

    Args:
        config_space: Ax ``SearchSpace`` (hierarchy via
            ``ChoiceParameter.dependents``).
        objectives: List of dicts with keys ``objective_name``, ``cost``,
            ``ref_point``. ``ref_point`` becomes the ``ObjectiveProperties``
            threshold — unused by Sobol generation, but kept so the experiment
            definition (and thus Pareto/hypervolume bookkeeping) matches the
            other optimizers exactly.
        seed: Random seed for the Sobol sequence (``AxClient.random_seed``).

    """

    def __init__(
        self,
        config_space: SearchSpace,
        objectives: List[dict],
        seed: int = 42,
        ssops: Optional[AxSearchSpaceOps] = None,
    ):
        self.search_space = config_space
        self.objectives = objectives
        self.seed = seed
        self.objective_names: List[str] = [obj["objective_name"] for obj in objectives]
        self.objective_costs: Dict[str, float] = {
            obj["objective_name"]: float(obj.get("cost", 1.0)) for obj in objectives
        }
        self._ssops: AxSearchSpaceOps = (
            ssops if ssops is not None else AxSearchSpaceOps(config_space)
        )

        # Single Sobol node sourcing the entire budget (num_trials=-1 → never
        # exhausts, so the strategy never transitions to a model-based step).
        generation_strategy = GenerationStrategy(
            name="sobol",
            steps=[
                GenerationStep(
                    generator=Generators.SOBOL,
                    num_trials=-1,
                ),
            ],
        )

        self._client = AxClient(
            random_seed=seed,
            generation_strategy=generation_strategy,
            verbose_logging=False,
        )
        self._client.create_experiment(
            name="sobol_baseline",
            parameters=_search_space_to_ax_parameters(config_space),
            objectives={
                obj["objective_name"]: ObjectiveProperties(
                    minimize=False,
                    threshold=float(obj["ref_point"]),
                )
                for obj in objectives
            },
        )

        # No init/BO split for Sobol, but the controller reads n_both_init to
        # size its Phase 1 batch. Expose 0 so it treats every trial uniformly.
        self.n_both_init: int = 0

        self._configs: List[dict] = []
        self._obj_values: List[List[float]] = []
        self._eval_ids: List[int] = []

        self._pending_trial_indices: List[int] = []
        self._pending_configs: List[dict] = []

        logger.info(
            f"Sobol: dim={len(config_space.parameters)}, seed={seed}, "
            f"objectives={self.objective_names}, "
            f"generator=Sobol (quasi-random, hierarchy-aware), "
            f"thresholds={[o['ref_point'] for o in objectives]}"
        )

    def suggest(self, n_suggestions: int = 1) -> Tuple[Optional[List[dict]], Optional[List[str]]]:
        n = max(1, int(n_suggestions))
        configs: List[dict] = []
        indices: List[int] = []
        for _ in range(n):
            params, trial_idx = self._client.get_next_trial()
            configs.append(self._ssops.decode_to_nested(params))
            indices.append(trial_idx)
        self._pending_trial_indices = indices
        self._pending_configs = configs
        return configs, list(self.objective_names)

    def add_objectives(
        self,
        x: List[dict],
        obj_dict: Dict[str, List],
        eval_ids: Optional[List[int]] = None,
    ) -> None:
        """Same identity-aligned + abandon-leftover semantics as ``AxQNEHVI``."""
        if not self._pending_trial_indices:
            logger.warning(
                "Sobol.add_objectives called without pending trials; skipping"
            )
            return

        n_pending = len(self._pending_trial_indices)
        n_results = len(x)
        if n_results > n_pending:
            raise ValueError(
                f"Sobol: got {n_results} results but only {n_pending} pending trials"
            )

        completed_positions: set = set()
        for i in range(n_results):
            pos = next(
                (j for j, pc in enumerate(self._pending_configs)
                 if pc is x[i] and j not in completed_positions),
                None,
            )
            if pos is None:
                raise ValueError(
                    f"Sobol: result config #{i} (id={id(x[i])}) not found in "
                    f"pending list — controller must pass back the same dict objects "
                    f"returned by suggest()"
                )
            completed_positions.add(pos)
            trial_idx = self._pending_trial_indices[pos]

            raw_data: Dict[str, float] = {}
            for name in self.objective_names:
                val = obj_dict[name][i]
                scalar = val["combined"] if isinstance(val, dict) else val
                raw_data[name] = float(scalar)

            self._client.complete_trial(trial_idx, raw_data=raw_data)

            self._configs.append(x[i])
            self._obj_values.append([raw_data[n] for n in self.objective_names])
            self._eval_ids.append(eval_ids[i] if eval_ids else -1)

        for j, trial_idx in enumerate(self._pending_trial_indices):
            if j not in completed_positions:
                try:
                    self._client.abandon_trial(trial_idx)
                    logger.info(
                        f"Sobol: abandoned trial {trial_idx} (config marked INVALID by controller)"
                    )
                except Exception as e:
                    logger.warning(
                        f"Sobol: failed to abandon trial {trial_idx}: {e}"
                    )

        self._pending_trial_indices = []
        self._pending_configs = []

    def warm_start(self, evaluations: List[Dict]) -> None:
        _warm_start_ax_client_optimizer(self, evaluations, "Sobol")

    def get_pareto(self) -> Tuple[List[dict], List[Dict[str, float]], List[int]]:
        if not self._configs:
            return [], [], []

        obj_arr = np.array(self._obj_values)
        mask = _pareto_mask(obj_arr)
        n = len(self._configs)
        pareto_configs = [self._configs[i] for i in range(n) if mask[i]]
        pareto_objs = [
            {name: self._obj_values[i][j] for j, name in enumerate(self.objective_names)}
            for i in range(n) if mask[i]
        ]
        pareto_eval_ids = [self._eval_ids[i] for i in range(n) if mask[i]]
        return pareto_configs, pareto_objs, pareto_eval_ids
