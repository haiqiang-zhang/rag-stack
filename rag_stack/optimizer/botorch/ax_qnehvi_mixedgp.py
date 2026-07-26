"""qNEHVI baseline with ``MixedSingleTaskGP`` surrogate via Ax's ``BO_MIXED``.

Sibling of :class:`AxQNEHVI` (which uses ``Generators.BOTORCH_MODULAR`` →
``SingleTaskGP`` with Ax's full transform chain including ``OneHot``). The
problem with the default chain on categorical-heavy search spaces (RAG
configs, FAISS IVF-PQ params, ``component`` choices, etc.) is that
unordered Choice parameters get expanded to K binary indicator columns and
the GP fits a Matern 5/2 kernel over them — which is NOT Hamming distance
and discards categorical semantics.

``Generators.BO_MIXED`` uses a different transform chain
(``Mixed_transforms``):

    RemoveFixed → ChoiceToNumericChoice → Log → Logit

No ``OneHot``. Unordered Choice parameters stay as integer codes, the
surrogate auto-selector sees ``categorical_features`` populated, and
``MixedSingleTaskGP`` is picked automatically. ``MixedSingleTaskGP``
internally combines:

    ScaleKernel(MaternKernel ⊗ CategoricalKernel)   # interaction
    + ScaleKernel(MaternKernel + CategoricalKernel) # main effects

where ``CategoricalKernel`` is BoTorch's Hamming-distance kernel — the same
class our :mod:`topo_gp_mobo` optimizer uses internally for pure-cat /
mixed-input branches. This makes ``ax_qnehvi_mixedgp`` an apples-to-apples
kernel-strength baseline, isolating the DAG-decomposition / subspace-ARD
contribution of ``topo_decoupled`` from the kernel choice itself.

Everything else (hierarchical handling via ``ChoiceParameter.dependents``,
qLogNEHVI acquisition, Sobol → BO transition, Pareto extraction) is
identical to :class:`AxQNEHVI`.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ax.adapter.registry import Generators
from ax.core.search_space import SearchSpace
from ax.generation_strategy.dispatch_utils import calculate_num_initialization_trials

from rag_stack.search_space.ax_search_space_ops import AxSearchSpaceOps
from ..base import BaseOptimizer
from ..utils import default_tkwargs
from .ax_qnehvi import (
    _build_ax_client,
    _pareto_mask,
    _warm_start_ax_client_optimizer,
)

logger = logging.getLogger("RAG-Stack")


class AxQNEHVIMixedGP(BaseOptimizer):
    """Non-decoupled qNEHVI baseline with ``MixedSingleTaskGP`` surrogate.

    Differences from :class:`AxQNEHVI`:
      * BO step uses ``Generators.BO_MIXED`` instead of
        ``Generators.BOTORCH_MODULAR``. The transform chain omits
        ``OneHot``, so categorical params reach the surrogate as integer
        codes and ``MixedSingleTaskGP`` is auto-selected (Matern over
        numeric + CategoricalKernel/Hamming over cat dims).
      * No ``use_mtgp`` option — BO_MIXED's transform chain is
        single-task-specific; mixing in an MTGP would require a separate
        ``Mixed_MTGP_trans`` setup that Ax doesn't ship.

    All other behavior (hierarchical search spaces via
    ``ChoiceParameter.dependents``, qLogNEHVI acquisition, Sobol → BO
    transition, batch ``suggest``/``add_objectives`` semantics, Pareto
    extraction) matches :class:`AxQNEHVI` exactly.
    """

    def __init__(
        self,
        config_space: SearchSpace,
        objectives: List[dict],
        seed: int = 42,
        n_both_init: Optional[int] = None,
        num_restarts: Optional[int] = None,
        raw_samples: Optional[int] = None,
        use_gpu: bool = True,
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

        self._torch_device = default_tkwargs(prefer_gpu=use_gpu)["device"]

        if n_both_init is None:
            n_both_init = calculate_num_initialization_trials(
                num_tunable_parameters=len(config_space.parameters),
                num_trials=None,
                use_batch_trials=False,
            )
            init_source = "Ax default"
        else:
            init_source = "user-set"
        n_both_init = int(n_both_init)

        optimizer_kwargs: Dict[str, Any] = {}
        if num_restarts is not None:
            optimizer_kwargs["num_restarts"] = int(num_restarts)
        if raw_samples is not None:
            optimizer_kwargs["raw_samples"] = int(raw_samples)
        generator_gen_kwargs = (
            {"model_gen_options": {"optimizer_kwargs": optimizer_kwargs}}
            if optimizer_kwargs
            else {}
        )

        self.n_both_init: int = n_both_init

        # Client-construction attrs for _build_ax_client — warm_start rebuilds
        # the client with a reduced Sobol budget on resume so the YAML
        # n_both_init is the project's TOTAL init budget (no Sobol replay).
        self._bo_generator = Generators.BO_MIXED
        self._gs_name = "sobol+bo_mixed+qlognehvi"
        self._generator_gen_kwargs = generator_gen_kwargs
        self._experiment_name = "ax_qnehvi_mixedgp_baseline"
        _build_ax_client(self, sobol_num_trials=n_both_init)

        self._configs: List[dict] = []
        self._obj_values: List[List[float]] = []
        self._eval_ids: List[int] = []

        self._pending_trial_indices: List[int] = []
        self._pending_configs: List[dict] = []

        logger.info(
            f"AxQNEHVIMixedGP: dim={len(config_space.parameters)}, seed={seed}, "
            f"objectives={self.objective_names}, "
            f"acquisition=qLogNoisyExpectedHypervolumeImprovement, "
            f"surrogate=MixedSingleTaskGP (auto-selected via Generators.BO_MIXED), "
            f"n_both_init={n_both_init} ({init_source}), "
            f"thresholds={[o['ref_point'] for o in objectives]}, "
            f"device={self._torch_device}"
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
        """Same identity-aligned + abandon-leftover semantics as :class:`AxQNEHVI`."""
        if not self._pending_trial_indices:
            logger.warning(
                "AxQNEHVIMixedGP.add_objectives called without pending trials; skipping"
            )
            return

        n_pending = len(self._pending_trial_indices)
        n_results = len(x)
        if n_results > n_pending:
            raise ValueError(
                f"AxQNEHVIMixedGP: got {n_results} results but only {n_pending} pending trials"
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
                    f"AxQNEHVIMixedGP: result config #{i} (id={id(x[i])}) not found in "
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
                        f"AxQNEHVIMixedGP: abandoned trial {trial_idx} (config marked INVALID by controller)"
                    )
                except Exception as e:
                    logger.warning(
                        f"AxQNEHVIMixedGP: failed to abandon trial {trial_idx}: {e}"
                    )

        self._pending_trial_indices = []
        self._pending_configs = []

    def warm_start(self, evaluations: List[Dict]) -> None:
        _warm_start_ax_client_optimizer(self, evaluations, "AxQNEHVIMixedGP")

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
