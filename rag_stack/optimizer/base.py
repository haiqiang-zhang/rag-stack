from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union


class BaseOptimizer(ABC):
    """Base interface for multi-objective optimizers.

    All optimizers communicate using real parameter values (plain Python dicts),
    never normalized tensors.  Internal representations (torch Tensors,
    ConfigSpace objects, etc.) are the optimizer's private concern.

    Config dict format (returned by suggest / get_pareto):
        {
            "query_expansion": {"component": "hyde", "max_token": 64},
            "generator": {"temperature": 1.0},
        }

    Required public attributes (set in subclass ``__init__``; the controller
    reads these directly, not via getattr):
        ``objective_names: List[str]``  — ordered list of objective names
                                          (e.g. ``["performance", "quality"]``).
        ``objective_costs: Dict[str, float]`` — per-objective evaluation cost
                                          used to track
                                          ``self._total_cost`` in the controller.
        ``n_both_init: int``           — number of "both-objective" init evals
                                          requested in Phase 1.

    Init kwarg shape: ``objectives: List[dict]`` — each entry has
        ``objective_name: str``, ``cost: float``, ``ref_point: float``, and
        optionally ``sub_metrics: List[{"name", "description"}]`` describing
        the composite decomposition.

    Composite objectives & sub-metrics:
        When an objective has sub-metrics, the value passed to
        ``add_objectives`` is a dict ``{"combined": <scalar>, "<sub_name>":
        <scalar>, …}``; otherwise it is a plain float. Optimizers that use only
        the scalar objective read ``v["combined"]``. Descriptions and declared
        sub-metric names remain project/evaluator metadata rather than a second
        optimizer-constructor contract.
    """

    @abstractmethod
    def suggest(
        self, n_suggestions: int = 1
    ) -> Tuple[Optional[List[dict]], Optional[List[str]]]:
        """Suggest next candidate configs and which objectives to evaluate.

        Returns:
            configs: list of decoded config dicts, or None if search space
                     is exhausted.
            eval_objectives: list of objective names to evaluate, or None.
        """

    @abstractmethod
    def add_objectives(
        self,
        x: List[dict],
        obj_dict: Dict[str, List[Union[float, dict]]],
        eval_ids: Optional[List[int]] = None,
    ) -> None:
        """Record objective values for the most recently suggested configs.

        The optimizer matches these values to the candidates it stored
        internally during the last ``suggest()`` call.

        Args:
            x: list of config dicts for which to record objectives, in the same order as returned by the last suggest().
            obj_dict: {objective_name: [value_per_config]}
                      Values may be floats or dicts. When a dict, the
                      ``"combined"`` key holds the scalar used for optimization.
                      e.g. {"performance": [-1.2], "quality": [{"combined": 0.85, "bleu": 0.7}]}
            eval_ids: optional list of evaluation IDs assigned by the controller.
        """

    @abstractmethod
    def get_pareto(self) -> Tuple[List[dict], List[Dict[str, float]], List[int]]:
        """Extract the current Pareto frontier.

        Returns:
            pareto_configs: list of config dicts on the Pareto front.
            pareto_objectives: list of dicts mapping objective name to value.
                               e.g. [{"performance": -1.2, "quality": 0.85}, ...]
            pareto_eval_ids: list of eval_ids for each Pareto point.
        """

    def warm_start(
        self,
        evaluations: List[Dict],
    ) -> None:
        """Inject historical evaluations to resume from a checkpoint.

        This bypasses the normal suggest→add_objectives flow and directly
        populates the optimizer's internal data structures.

        Args:
            evaluations: list of dicts, each with:
                - ``config``: decoded config dict
                - ``objectives``: {obj_name: scalar_value} for evaluated objectives
                - ``eval_id``: integer evaluation ID
                - ``eval_type``: "GT" or "CM"
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support warm_start"
        )
