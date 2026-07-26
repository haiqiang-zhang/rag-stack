"""Abstract base class for synthetic benchmark functions."""

from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
from ax.core.search_space import SearchSpace


def non_dominated_max(Y: np.ndarray) -> np.ndarray:
    """Maximization-convention non-dominated set, dimension-agnostic.

    O(n log n) sweep for d=2; O(n²) brute force for d≥3. Sizes we deal
    with (pooled eval CSVs, ≤100k rows) make either path acceptable.
    """
    if len(Y) == 0:
        return Y
    if Y.shape[1] == 2:
        order = np.argsort(-Y[:, 0], kind="stable")
        Y_s = Y[order]
        mask = np.zeros(len(Y), dtype=bool)
        best_y1 = -np.inf
        for i, idx in enumerate(order):
            if Y_s[i, 1] > best_y1:
                mask[idx] = True
                best_y1 = Y_s[i, 1]
        return Y[mask]
    n = len(Y)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        diff = Y - Y[i]
        if ((diff >= 0).all(axis=1) & (diff > 0).any(axis=1)).any():
            keep[i] = False
    return Y[keep]


def _flat_to_nested_config(
    flat: Dict[str, Any],
    ssops,
) -> Dict[str, Any]:
    """Convert pymoo's flat parameter dict into the nested config layout
    that ``function.evaluate`` consumes — applies the same naming rules as
    the optimizer's ``decode_to_config`` (active-set pruning + the SSops's
    ``decode_to_nested`` convention).

    The ``ssops`` argument owns the naming-convention rules: pass
    ``RagSearchSpace`` for RAG schemas with ``vectordb__``/``corpus__``
    deviations, or the generic ``AxSearchSpaceOps`` otherwise. Use
    ``function.ssops`` from a ``SyntheticFunction`` to pick automatically.
    """
    active = ssops.active_params(flat)
    return ssops.decode_to_nested(
        {k: v for k, v in flat.items() if k in active}
    )


def run_nsga2_pareto(
    function: "SyntheticFunction",
    n_evals: int,
    pop_size: int = 200,
) -> np.ndarray:
    """Run pymoo's NSGA-II for ``n_evals`` total evaluations on ``function``.
    Return final non-dominated front as ``(n_pareto, n_obj)`` ndarray in
    ``function.objectives`` column order, maximization convention.

    Why NSGA-II: random sampling on complex mixed search spaces (24-D for
    synthetic_rag, 37-D for yahpo) leaves the high-quality region under-
    covered. NSGA-II concentrates queries near the actual Pareto via
    evolutionary selection — the result is a *much* tighter reference
    than any random reference of comparable budget, and is the de-facto
    standard baseline in MOBO benchmarking literature.

    Search-space conversion: each Ax parameter maps to a pymoo variable
    of matching kind (Real / Integer / Choice). Hierarchical activation
    (params only active under specific categorical choices) is left
    implicit — NSGA-II samples ALL params unconditionally, and
    ``function.evaluate`` reads only the active subtree.
    """
    from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.mixed import (
        MixedVariableDuplicateElimination,
        MixedVariableMating,
        MixedVariableSampling,
    )
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.variable import Choice, Integer, Real
    from pymoo.optimize import minimize as pymoo_minimize

    obj_names = [obj["objective_name"] for obj in function.objectives]

    vars_spec: Dict[str, Any] = {}
    for p in function.search_space.parameters.values():
        if isinstance(p, RangeParameter):
            if p.parameter_type == ParameterType.INT:
                vars_spec[p.name] = Integer(bounds=(int(p.lower), int(p.upper)))
            else:
                vars_spec[p.name] = Real(bounds=(float(p.lower), float(p.upper)))
        elif isinstance(p, ChoiceParameter):
            vars_spec[p.name] = Choice(options=list(p.values))
        else:
            raise TypeError(
                f"run_nsga2_pareto: unsupported Ax parameter type "
                f"{type(p).__name__!r} for param {p.name!r}. Add a mapping "
                f"to pymoo variable type."
            )

    class _Problem(ElementwiseProblem):
        def __init__(_self):
            super().__init__(vars=vars_spec, n_obj=len(obj_names))

        def _evaluate(_self, X, out, *args, **kwargs):
            cfg = _flat_to_nested_config(X, function.ssops)
            v = function.evaluate(cfg)
            # NSGA-II minimizes — negate so the "minimum" of -obj == max of obj
            out["F"] = [
                -(v[n]["combined"] if isinstance(v[n], dict) else v[n])
                for n in obj_names
            ]

    n_gen = max(10, n_evals // pop_size)
    res = pymoo_minimize(
        _Problem(),
        NSGA2(
            pop_size=pop_size,
            sampling=MixedVariableSampling(),
            mating=MixedVariableMating(
                eliminate_duplicates=MixedVariableDuplicateElimination()
            ),
            eliminate_duplicates=MixedVariableDuplicateElimination(),
        ),
        termination=("n_gen", n_gen),
        verbose=False,
    )
    # res.F shape (n_pareto, n_obj) under min convention → negate for max
    return -res.F


def pool_evals_pareto(
    evals_dir: str,
    obj_names: List[str],
) -> np.ndarray:
    """Pool every ``*__evals.csv`` in ``evals_dir`` and return their
    non-dominated set as ``(n_pareto, len(obj_names))``, columns in
    ``obj_names`` order.

    ``evals_dir`` must be a *per-benchmark subfolder* under the shared
    output dir — each benchmark gets its own subfolder so a glob inside
    one folder yields exactly one benchmark's CSVs (no cross-mixing).
    Cumulative quality+speed columns named in ``obj_names`` must exist
    in every CSV.

    Used by ``compute_true_pareto`` overrides on functions whose intrinsic
    Pareto computation is a coarse approximation (random sampling on a
    high-dim space) — pooling real optimizer queries is far denser near
    the actual Pareto since BO runs concentrate effort in the high-quality
    region.

    Drops rows with NaN/Inf objectives — surrogate / cost-model failures
    occasionally leave them.
    """
    # Lazy import — keeps the module light for callers that don't pool.
    import pandas as pd

    pattern = os.path.join(evals_dir, "*__evals.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No eval CSVs matching {pattern!r}. Run optimizer benchmarks "
            f"first to generate eval data, then rebuild the Pareto cache."
        )
    chunks = [
        pd.read_csv(f, usecols=obj_names)[obj_names].to_numpy(dtype=float)
        for f in files
    ]
    Y = np.vstack(chunks)
    Y = Y[np.isfinite(Y).all(axis=1)]
    return non_dominated_max(Y)


class SyntheticFunction(ABC):
    """A synthetic multi-objective function with known ground truth.

    Used to benchmark optimizer performance without running real evaluations.
    Subclasses provide a cheap-to-evaluate function and a way to compute the
    true Pareto front, so we can measure HV difference and IGD.

    All objectives use maximization convention (higher is better).
    """

    name: str

    @property
    @abstractmethod
    def search_space(self) -> SearchSpace:
        """Ax SearchSpace defining the hyperparameter space."""

    @property
    @abstractmethod
    def objectives(self) -> List[Dict]:
        """List of objective specs. Each dict must contain the keys
        ``objective_name``, ``cost``, and ``ref_point``.
        """

    @property
    def ssops(self):
        """SearchSpace adapter — owns flat→nested decoding rules.

        Default: generic :class:`AxSearchSpaceOps` (handles 1-seg / 2-seg /
        ``<head>__<branch>__<leaf>`` collapse — works for any Ax
        SearchSpace using the ``__`` convention, including yahpo).
        Override in subclasses with naming-convention deviations
        (e.g. :class:`SyntheticRAGFunction` returns a ``RagSearchSpace``
        for the ``vectordb__``/``corpus__`` namespace rules).
        """
        cached = getattr(self, "_ssops_cache", None)
        if cached is None:
            from rag_stack.search_space.ax_search_space_ops import AxSearchSpaceOps
            cached = AxSearchSpaceOps(self.search_space)
            self._ssops_cache = cached
        return cached

    @abstractmethod
    def evaluate(self, config: dict) -> Dict[str, Any]:
        """Evaluate all objectives for a config dict.

        Args:
            config: Same format as optimizer.suggest() output.

        Returns:
            ``{objective_name: value}`` for ALL objectives. ``value`` is
            either a float (plain scalar objective) or a dict with at
            least ``"combined": float`` plus optional sub_metric and
            diagnostic keys (e.g. topo's per-learner Pattern 1 setup
            attaches per-sub_metric scalars under the same name).
            Consumers that only need the scalar should read
            ``v["combined"] if isinstance(v, dict) else v``.
        """

    def evaluate_noise_free(self, config: dict) -> Dict[str, Any]:
        """Evaluate without observation noise — for inference regret.

        For deterministic functions this is identical to ``evaluate``.
        Stochastic functions (currently only ``SyntheticRAGFunction``)
        override this to bypass noise injection so the runner's snapshot
        ``hv_diff`` measures **what the optimizer believes is best**
        against the **true** Pareto, not noisy observations of it.
        Without this, hv_diff under noise is biased low (winner's curse:
        non-dominated set of noisy obs systematically picks lucky-high
        samples), which corrupts noise-vs-noiseless cross-comparison.
        """
        return self.evaluate(config)

    @abstractmethod
    def compute_true_pareto(
        self,
        n_samples: int = 200_000,
        evals_dir: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute the true Pareto front.

        Args:
            n_samples: Sample budget for sampling-based strategies. Ignored
                by analytical / fully-enumerable functions.
            evals_dir: When set, functions whose intrinsic Pareto is an
                approximation (random sampling on complex search spaces —
                e.g. ``SyntheticRAGFunction``, ``YAHPOGymFunction``) should
                pool optimizer eval CSVs from this directory instead of
                re-sampling, yielding a much tighter reference. Functions
                with exact Paretos ignore it. The standard pool helper is
                ``pool_evals_pareto`` (module-level in this file).

        Returns:
            ``{objective_name: 1D-array}`` keyed by every name in
            ``objectives``. All arrays must have the same length n_pareto.
            Consumers (runner, metrics, apps) build the (n_pareto, n_obj)
            matrix by looking up each obj name explicitly — no positional
            assumptions, so adding/reordering objectives can't silently
            scramble Pareto columns.
        """

    @property
    def ref_point(self) -> np.ndarray:
        return np.array([obj["ref_point"] for obj in self.objectives])

    @property
    def n_objectives(self) -> int:
        return len(self.objectives)
