"""qNEHVI baseline via Ax's ``AxClient``.

Uses Ax's Service API (``AxClient``), which defaults to ``BOTORCH_MODULAR`` with
``qNoisyExpectedHypervolumeImprovement`` (qNEHVI) — the current Ax/BoTorch
recommended acquisition function for multi-objective BO. This class exists to
benchmark **qNEHVI specifically** against the other algorithms in the registry
(ParEGO, EHVI, ParEGO+metrics, …); the "Ax" prefix in the class name just
records which implementation we call — the comparison is algorithm-vs-algorithm.

Non-decoupled: every ``suggest()`` returns all objective names → 1 GT call per
iter, ``cm=0``.

Hierarchical search spaces are supported end-to-end:
- ``ChoiceParameter.dependents`` on the input ``SearchSpace`` is forwarded to
  AxClient via the ``dependents`` field on each choice parameter dict.
- AxClient's ``get_next_trial()`` only returns active parameters for the chosen
  branch; inactive params are omitted, so the ``__`` segment parsing below
  naturally yields the same hierarchical-dict shape produced by
  ``cs_config_to_dict`` in the SMAC optimizers.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ax.adapter.registry import Generators
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from ax.generation_strategy.dispatch_utils import calculate_num_initialization_trials
from ax.generation_strategy.generation_node import GenerationStep
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.service.ax_client import AxClient, ObjectiveProperties
from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)

from rag_stack.search_space.ax_search_space_ops import AxSearchSpaceOps
from ..base import BaseOptimizer
from ..utils import default_tkwargs

logger = logging.getLogger("RAG-Stack")


def _search_space_to_ax_parameters(search_space: SearchSpace) -> List[Dict[str, Any]]:
    """Convert an Ax ``SearchSpace`` into AxClient's parameter-dict format."""
    params: List[Dict[str, Any]] = []
    for name, p in search_space.parameters.items():
        if isinstance(p, RangeParameter):
            entry: Dict[str, Any] = {
                "name": name,
                "type": "range",
                "bounds": [p.lower, p.upper],
                "value_type": "float" if p.parameter_type == ParameterType.FLOAT else "int",
            }
            if getattr(p, "log_scale", False):
                entry["log_scale"] = True
            params.append(entry)
        elif isinstance(p, ChoiceParameter):
            entry = {
                "name": name,
                "type": "choice",
                "values": list(p.values),
                "is_ordered": bool(getattr(p, "is_ordered", False)),
                "sort_values": bool(getattr(p, "sort_values", False)),
            }
            # ChoiceParameter.dependents raises NotImplementedError for
            # non-hierarchical params in Ax ≥ 1.x. is_hierarchical is the
            # correct gate before accessing dependents.
            try:
                if getattr(p, "is_hierarchical", False):
                    deps = p.dependents
                    if deps:
                        entry["dependents"] = dict(deps)
            except (NotImplementedError, AttributeError):
                pass
            params.append(entry)
        else:
            logger.warning(
                f"AxQNEHVI: unsupported parameter type {type(p).__name__} "
                f"for '{name}' — skipping"
            )
    return params


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


def _scalar_objective_value(value: Any) -> float:
    """Return the scalar value Ax optimizes for a controller objective payload."""
    if isinstance(value, dict):
        if "combined" not in value:
            raise ValueError(f"composite objective is missing 'combined': {value}")
        value = value["combined"]
    return float(value)


def _active_ax_parameters(ssops: AxSearchSpaceOps, config: dict) -> Dict[str, Any]:
    """Convert a nested controller config into Ax's flat active parameters."""
    params = ssops.active_flat_config(config, require_all_active=True)
    # Ax parameterization accepts plain JSON-ish Python scalar types. Values
    # produced by decode_flat are normally already plain, but this makes the
    # warm-start path robust to numpy scalar values read back from CSV.
    return {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in params.items()
    }


def _build_ax_client(
    optimizer,
    sobol_num_trials: int,
    sobol_init_position: int = 0,
) -> None:
    """(Re)create ``optimizer._client`` with a given Sobol init budget.

    Called from ``__init__`` with the full YAML ``n_both_init`` budget, and
    from :func:`_warm_start_ax_client_optimizer` with the budget reduced by
    the historical suggestions already consumed. Ax's Sobol GenerationStep
    only counts trials generated by THIS strategy instance — attached
    (warm-started) trials don't count — so without the reduction every
    resume re-emits the same seed-derived Sobol sequence and burns GT evals
    re-evaluating configs that are already in the data.

    ``sobol_init_position`` advances the Sobol sequence so a mid-init resume
    continues at the first never-drawn point of the SAME seed-44 design
    instead of restarting it (SobolGenerator(init_position=...)).
    """
    steps = []
    if sobol_num_trials > 0:
        sobol_kwargs: Dict[str, Any] = {}
        if sobol_init_position > 0:
            sobol_kwargs["init_position"] = int(sobol_init_position)
        steps.append(
            GenerationStep(
                generator=Generators.SOBOL,
                num_trials=int(sobol_num_trials),
                generator_kwargs=sobol_kwargs or None,
                should_deduplicate=True,
            )
        )
    steps.append(
        GenerationStep(
            generator=optimizer._bo_generator,
            num_trials=-1,
            should_deduplicate=True,
            generator_kwargs={
                "botorch_acqf_class": qLogNoisyExpectedHypervolumeImprovement,
                "torch_device": optimizer._torch_device,
            },
            generator_gen_kwargs=optimizer._generator_gen_kwargs,
        )
    )
    optimizer._client = AxClient(
        random_seed=optimizer.seed,
        generation_strategy=GenerationStrategy(
            name=optimizer._gs_name, steps=steps
        ),
        verbose_logging=False,
    )
    optimizer._client.create_experiment(
        name=optimizer._experiment_name,
        parameters=_search_space_to_ax_parameters(optimizer.search_space),
        objectives={
            obj["objective_name"]: ObjectiveProperties(
                minimize=False,
                threshold=float(obj["ref_point"]),
            )
            for obj in optimizer.objectives
        },
    )


def _warm_start_ax_client_optimizer(
    optimizer: "AxQNEHVI",
    evaluations: List[Dict],
    log_label: str,
) -> None:
    """Inject historical complete trials into an AxClient-backed optimizer.

    Before attaching, the client is REBUILT with its Sobol init budget
    shrunk by the number of suggestions the historical run already
    consumed: the YAML ``n_both_init`` is the project's TOTAL init budget,
    so a resume must only draw the still-missing init points (continuing
    the same Sobol sequence via ``init_position``) and then hand over to
    real BO — never replay init configs that are already evaluated.
    """
    loaded = 0
    skipped = 0
    optimizer._pending_trial_indices = []
    optimizer._pending_configs = []

    candidates = [
        ev for ev in evaluations
        if ev.get("config")
        and all(
            name in (ev.get("objectives") or {})
            for name in optimizer.objective_names
        )
    ]
    if candidates:
        # Suggestions consumed by the prior run = contiguous eval ids from 0,
        # INCLUDING invalid ones (they consumed a draw too, they just aren't
        # warm-loadable). max(eval_id)+1 capped at n_both_init gives how far
        # the original Sobol design got.
        consumed = min(
            int(optimizer.n_both_init),
            max(int(ev.get("eval_id", -1)) for ev in candidates) + 1,
        )
        remaining = int(optimizer.n_both_init) - consumed
        _build_ax_client(
            optimizer,
            sobol_num_trials=remaining,
            sobol_init_position=consumed if remaining > 0 else 0,
        )
        logger.info(
            f"{log_label}.warm_start: rebuilt client for resume — "
            f"Sobol init {optimizer.n_both_init} total, {consumed} consumed, "
            f"{remaining} remaining; everything after is BO"
        )

    for ev in sorted(evaluations, key=lambda e: int(e.get("eval_id", -1))):
        cfg = ev.get("config")
        objectives = ev.get("objectives") or {}
        eval_id = int(ev.get("eval_id", -1))
        if not cfg or not all(name in objectives for name in optimizer.objective_names):
            skipped += 1
            continue

        try:
            params = _active_ax_parameters(optimizer._ssops, cfg)
            raw_data = {
                name: _scalar_objective_value(objectives[name])
                for name in optimizer.objective_names
            }
            _, trial_idx = optimizer._client.attach_trial(
                parameters=params,
                arm_name=f"warm_eval_{eval_id:04d}",
            )
            optimizer._client.complete_trial(trial_idx, raw_data=raw_data)
        except Exception as e:  # noqa: BLE001
            skipped += 1
            logger.warning(
                f"{log_label}.warm_start: skipped eval_{eval_id:04d}: {e}"
            )
            continue

        optimizer._configs.append(cfg)
        optimizer._obj_values.append(
            [raw_data[name] for name in optimizer.objective_names]
        )
        optimizer._eval_ids.append(eval_id)
        loaded += 1

    if loaded == 0 and evaluations:
        raise RuntimeError(
            f"{log_label}.warm_start could not load any complete historical trials"
        )
    logger.info(
        f"{log_label}.warm_start: loaded {loaded} historical trials"
        + (f", skipped {skipped}" if skipped else "")
    )


class AxQNEHVI(BaseOptimizer):
    """Non-decoupled qNEHVI baseline via Ax's ``AxClient``.

    The acquisition function is explicitly pinned to
    ``qLogNoisyExpectedHypervolumeImprovement`` (qLogNEHVI) — the log-space
    variant BoTorch now recommends over the original
    ``qNoisyExpectedHypervolumeImprovement`` (qNEHVI) for numerical stability.
    Same family ("NEHVI"), so the class name still fits; see arxiv 2310.20708.

    We build the ``GenerationStrategy`` ourselves instead of relying on
    ``AxClient``'s auto-pick — otherwise the acquisition function would only
    be implicitly selected and could change when Ax's defaults move.

    Every ``suggest()`` returns all objective names, so the runner evaluates all
    objectives and counts it as a GT call. ``cm`` stays 0.

    Args:
        config_space: Ax ``SearchSpace`` (supports hierarchical via
            ``ChoiceParameter.dependents``).
        objectives: List of dicts with keys ``objective_name``, ``cost``,
            ``ref_point``. ``ref_point`` is forwarded as the ``ObjectiveProperties``
            threshold (qLogNEHVI needs one per objective to bound the reference
            hypervolume).
        seed: Random seed passed to ``AxClient``.
        n_both_init: Optional override for the both-objective init count
            (Ax's Sobol step budget). ``None`` (default) → fall back to Ax's
            own rule (``calculate_num_initialization_trials`` = ``max(5, 2·dim)``
            when the total budget is unknown). Pass an explicit int in YAML
            (``params: {n_both_init: 10}``) to match other optimizers' init
            budgets for fair benchmark comparison. Named ``n_both_init`` for
            cross-optimizer parameter consistency — every optimizer in
            rag-stack exposes its init-batch size under this kwarg.
    """

    def __init__(
        self,
        config_space: SearchSpace,
        objectives: List[dict],
        seed: int = 42,
        n_both_init: Optional[int] = None,
        num_restarts: Optional[int] = None,
        raw_samples: Optional[int] = None,
        use_mtgp: bool = False,
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
        # SearchSpace adapter — owns flat→nested decoding. Defaults to the
        # generic ``AxSearchSpaceOps`` (handles 1-seg / 2-seg / branch-label
        # collapse). Callers running RAG schemas with ``vectordb__``/
        # ``corpus__`` namespace deviations should pass a ``RagSearchSpace``.
        self._ssops: AxSearchSpaceOps = (
            ssops if ssops is not None else AxSearchSpaceOps(config_space)
        )

        # GPU device routing. Passed through to BoTorchGenerator below; Ax
        # internally creates some CPU tensors in its transformation layer
        # that ``torch.set_default_device`` doesn't cover, so we set the
        # device explicitly on the generator's surrogate/acquisition.
        # ``default_tkwargs`` is the project-global helper: auto-detects
        # CUDA, falls back to CPU, and downgrades user-asked-cuda to CPU
        # cleanly when CUDA is absent.
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

        # Forward acqf-optimizer budget knobs to BoTorch's optimize_acqf.
        # Path: GenerationStep.generator_gen_kwargs["model_gen_options"]
        # ["optimizer_kwargs"] -> Acquisition.optimize(optimizer_options=...)
        # -> optimize_acqf(num_restarts=..., raw_samples=...). When both are
        # None (default), Ax/BoTorch's own defaults apply unchanged.
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

        # Surrogate selector. Default = Ax's BOTORCH_MODULAR which auto-picks
        # per-outcome SingleTaskGP (ModelListGP). ``use_mtgp=True`` swaps to
        # ST_MTGP — same ModularBoTorchGenerator but with MBM_MTGP transforms
        # that augment X with a task-feature dim and route to a single
        # MultiTaskGP across outcomes. Useful for testing whether tying
        # objectives together via an MTGP helps on small-N MOO.
        bo_generator = Generators.ST_MTGP if use_mtgp else Generators.BOTORCH_MODULAR
        gs_name = "sobol+stmtgp+qlognehvi" if use_mtgp else "sobol+qlognehvi"
        self._use_mtgp = bool(use_mtgp)

        # n_both_init is the controller's Phase 1 init-batch size. For Ax
        # that's the Sobol step's budget — once the Phase 1 batch is
        # consumed, the GenerationStrategy automatically transitions to BO.
        self.n_both_init: int = n_both_init

        # Client-construction attrs, kept so warm_start can REBUILD the
        # client with a reduced Sobol budget on resume (see _build_ax_client).
        self._bo_generator = bo_generator
        self._gs_name = gs_name
        self._generator_gen_kwargs = generator_gen_kwargs
        self._experiment_name = "ax_qnehvi_baseline"
        _build_ax_client(self, sobol_num_trials=n_both_init)

        # Tracking for Pareto extraction (kept independently of Ax's internal
        # data so we can return (configs, obj_dicts, eval_ids) in our format).
        self._configs: List[dict] = []
        self._obj_values: List[List[float]] = []
        self._eval_ids: List[int] = []

        # Pending trial state — list to support batch suggest() in Phase 1
        # (single-element list in Phase 2's iterative loop).
        self._pending_trial_indices: List[int] = []
        self._pending_configs: List[dict] = []

        logger.info(
            f"AxQNEHVI: dim={len(config_space.parameters)}, seed={seed}, "
            f"objectives={self.objective_names}, "
            f"acquisition=qLogNoisyExpectedHypervolumeImprovement, "
            f"surrogate={'MTGP (Generators.ST_MTGP)' if self._use_mtgp else 'SingleTaskGP (Generators.BOTORCH_MODULAR)'}, "
            f"n_both_init={n_both_init} ({init_source}), "
            f"thresholds={[o['ref_point'] for o in objectives]}, "
            f"device={self._torch_device}"
        )

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    def suggest(self, n_suggestions: int = 1) -> Tuple[Optional[List[dict]], Optional[List[str]]]:
        """Ask Ax for the next ``n_suggestions`` configurations (all objectives = GT).

        Ax's ``get_next_trial()`` is single-trial; we loop to deliver a batch,
        and stash the pending trial indices in order so ``add_objectives``
        can complete them positionally.
        """
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
        """Tell Ax the objective values for every pending trial in this batch.

        Aligns ``x`` to ``_pending_trial_indices`` by Python object identity
        against ``_pending_configs`` (controller passes back the same dict
        objects it received from ``suggest()``). Any pending trial whose
        config is not in ``x`` is treated as invalid and abandoned — without
        this, Ax leaves it RUNNING forever and the next batch's positional
        alignment is wrong.
        """
        if not self._pending_trial_indices:
            logger.warning("AxQNEHVI.add_objectives called without pending trials; skipping")
            return

        n_pending = len(self._pending_trial_indices)
        n_results = len(x)
        if n_results > n_pending:
            raise ValueError(
                f"AxQNEHVI: got {n_results} results but only {n_pending} pending trials"
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
                    f"AxQNEHVI: result config #{i} (id={id(x[i])}) not found in "
                    f"pending list — controller must pass back the same dict objects "
                    f"returned by suggest()"
                )
            completed_positions.add(pos)
            trial_idx = self._pending_trial_indices[pos]

            # Coalesce composite dicts ({"combined": 0.7, "bleu": 0.5, ...}) to scalars.
            raw_data: Dict[str, float] = {}
            for name in self.objective_names:
                val = obj_dict[name][i]
                scalar = val["combined"] if isinstance(val, dict) else val
                raw_data[name] = float(scalar)

            self._client.complete_trial(trial_idx, raw_data=raw_data)

            self._configs.append(x[i])
            self._obj_values.append([raw_data[n] for n in self.objective_names])
            self._eval_ids.append(eval_ids[i] if eval_ids else -1)

        # Abandon any pending trials whose configs were invalid (didn't come
        # back in x). Otherwise Ax keeps them as RUNNING — the GP fit
        # treats RUNNING trials as pending observations and qLogNEHVI's
        # noisy-resample branch may waste budget on them.
        for j, trial_idx in enumerate(self._pending_trial_indices):
            if j not in completed_positions:
                try:
                    self._client.abandon_trial(trial_idx)
                    logger.info(
                        f"AxQNEHVI: abandoned trial {trial_idx} (config marked INVALID by controller)"
                    )
                except Exception as e:
                    logger.warning(
                        f"AxQNEHVI: failed to abandon trial {trial_idx}: {e}"
                    )

        self._pending_trial_indices = []
        self._pending_configs = []

    def warm_start(self, evaluations: List[Dict]) -> None:
        _warm_start_ax_client_optimizer(self, evaluations, "AxQNEHVI")

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
