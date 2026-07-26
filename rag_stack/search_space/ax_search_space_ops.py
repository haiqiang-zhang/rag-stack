"""Bridge layer between Ax ``SearchSpace`` and BoTorch's normalized ``[0,1]^d``
tensor world.

`AxSearchSpaceOps` owns operations Ax does not provide:

  * encode a config dict / Ax ``arm.parameters`` → normalized tensor row
  * decode a normalized tensor row → typed flat dict / nested config dict
  * snap / clamp on normalized tensors (grid snap; inactive-dim canonicalize)
  * dim-index set for BoTorch's ``cat_dims`` interface
  * active-set queries for Ax hierarchical (gated) search spaces

It does **not** wrap things Ax already provides. Callers use the Ax API
directly for those (``ss.is_hierarchical``, ``ChoiceParameter.dependents``,
``ss.cast_observation_features``, …). Hierarchy is expressed with Ax's native
``ChoiceParameter.dependents``; this layer only reads it.

No naming convention assumed: this layer works with any Ax SearchSpace.
RAG-specific ``__`` three-segment naming convention helpers + the single
search-space builder live in :mod:`rag_stack.search_space.rag_search_space`
(``RagSearchSpace``, a subclass of this class).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import torch
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from torch import Tensor

# Value to assign inactive parameters in [0,1] space. Lives here (not in
# rag_search_space.py) because the [0,1] convention is owned by this bridge layer.
INACTIVE_DEFAULT = 0.5

# Universal segment separator for Ax hierarchical parameter names. Owned here
# so any benchmark following the ``a__b__c`` convention can plug in without
# importing rag-stack-specific code.
SEP = "__"


def _choice_level_coords(param: ChoiceParameter) -> List[float]:
    """Normalized [0,1] coordinate of each level of a ``ChoiceParameter``.

    Ordered numeric choices use VALUE spacing — ``(v - lo) / (hi - lo)`` —
    the same geometry Ax's ``BO_MIXED`` chain gives the GP (raw values →
    ``SearchSpaceDigest`` bounds → ``Normalize``): e.g. ``[1, 8, 32, 64]``
    → ``[0, 7/63, 31/63, 1]``, so kernel distances reflect the metric, not
    the level index. Everything else (unordered, or ordered non-numeric)
    uses INDEX spacing ``j/(n-1)``, matching Ax's ``ChoiceToNumericChoice``
    integer codes. Unordered spacing is inert for the GP (Hamming kernel
    only tests equality) but must stay a fixed convention for snap/decode.
    """
    n_vals = len(param.values)
    if n_vals <= 1:
        return [0.0] * n_vals
    if param.is_ordered and all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        for v in param.values
    ):
        vals = [float(v) for v in param.values]
        lo, hi = min(vals), max(vals)
        if hi > lo:
            return [(v - lo) / (hi - lo) for v in vals]
    return [j / (n_vals - 1) for j in range(n_vals)]


class AxSearchSpaceOps:
    """Stateful operational adapter for any Ax ``SearchSpace``.

    Construct once per consumer (long-lived) and call methods directly:

        self.ssops = AxSearchSpaceOps(search_space, tkwargs)
        self.ssops.snap_to_grid(candidates)
        self.ssops.decode_flat(x_row)
        self.ssops.decode_to_config(x_row)
    """

    def __init__(
        self,
        ss: SearchSpace,
        tkwargs: Optional[Dict] = None,
    ):
        self.ss = ss
        self.tkwargs: Dict = (
            dict(tkwargs)
            if tkwargs
            else {"dtype": torch.double, "device": torch.device("cpu")}
        )
        self.param_names: List[str] = list(ss.parameters.keys())
        self.dim: int = len(self.param_names)

        # Cached at construction (one-time cost):
        #
        # cat_dims_set: ChoiceParameter dim indices that are UNORDERED. Ordered
        # Choice (e.g. quant_bits=[4,8,16,32], model_size=["3b","7b","13b","70b"])
        # carries true ordinal structure → treat as numeric so RBF / Matern can
        # use the level ordering. is_ordered is an Ax SearchSpace property —
        # this rule generalizes across benchmarks.
        self._cat_dims_set: Set[int] = {
            i
            for i, n in enumerate(self.param_names)
            if isinstance(ss.parameters[n], ChoiceParameter)
            and not ss.parameters[n].is_ordered
        }
        # choice_grids: per-ChoiceParameter normalized level coordinates
        # (see _choice_level_coords). Single source for encode / snap /
        # decode so the three stay mutually consistent.
        self._choice_grids: Dict[str, List[float]] = {
            n: _choice_level_coords(p)
            for n, p in ss.parameters.items()
            if isinstance(p, ChoiceParameter)
        }
        # Hierarchy (which params are active for a given parameterization) is
        # NOT cached or hand-rolled here: ``active_params`` delegates to Ax's
        # native ``_cast_parameterization`` (single source of truth).

    # ------------------------------------------------------------------
    # Properties — BoTorch cat_dims interface (dim-index based; param_names
    # ordering is our convention, derived from ss.parameters.keys()).
    # ------------------------------------------------------------------

    @property
    def cat_dims_set(self) -> Set[int]:
        return self._cat_dims_set

    def choice_grid(self, name: str) -> List[float]:
        """Normalized [0,1] level coordinates of ChoiceParameter ``name``
        (positionally aligned with ``param.values``)."""
        return self._choice_grids[name]

    # ------------------------------------------------------------------
    # Encoding — config dict / Ax arm.parameters → normalized tensor row.
    # Used by training-data warm-start (encode_flat_config) and Sobol init
    # (encode_arm_params).
    # ------------------------------------------------------------------

    def encode_arm_params(self, flat_params: dict) -> Tensor:
        """Encode a flat ``{original_name: typed_value}`` dict (e.g. from
        an Ax ``Sobol`` ``arm.parameters``) to a normalized tensor row.

        Missing (inactive) params are filled with the dummy midpoint.
        """
        x = torch.full((self.dim,), INACTIVE_DEFAULT, **self.tkwargs)
        for i, name in enumerate(self.param_names):
            if name not in flat_params:
                continue
            param = self.ss.parameters[name]
            v = flat_params[name]
            if isinstance(param, RangeParameter):
                x[i] = _normalize_range(float(v), param)
            elif isinstance(param, ChoiceParameter):
                idx = param.values.index(v)
                x[i] = self._choice_grids[name][idx]
        return x

    def encode_flat_config(self, config_dict: dict) -> Tensor:
        """Encode a nested config dict to a normalized tensor row.

        Walks the nested config_dict to a flat ``{original_name: value}``
        dict via :meth:`build_name_index`, then delegates to
        :meth:`encode_arm_params`.
        """
        if not hasattr(self, "_name_index"):
            self._name_index = self.build_name_index()

        flat: dict = {}

        def _walk(prefix: tuple, val) -> None:
            if isinstance(val, dict):
                for k, v in val.items():
                    _walk(prefix + (k,), v)
            else:
                name = self._name_index.get(prefix)
                if name is not None:
                    flat[name] = val

        for k, v in config_dict.items():
            _walk((k,), v)

        return self.encode_arm_params(flat)

    def hierarchical_selector_paths(self) -> Set[Tuple[str, ...]]:
        """Nested config paths owned by declared hierarchical choices.

        Checkpoint repair uses this schema-derived set to distinguish a real
        selector declaration from an unrelated dict-valued runtime setting.
        """
        return {
            self._index_key(name.split(SEP))
            for name, param in self.ss.parameters.items()
            if isinstance(param, ChoiceParameter) and param.is_hierarchical
        }

    def active_flat_config(
        self,
        config_dict: dict,
        *,
        require_all_active: bool = True,
    ) -> Dict[str, object]:
        """Invert a nested config into its exact active Ax parameterization.

        ``encode_flat_config`` uses one schema name per nested leaf path.  Two
        component branches can intentionally decode to the same lifted path
        (for example ``hyde.temperature`` and
        ``multi_query_expansion.temperature``), so that generic tensor inverse
        alone can put the value in the wrong branch.  The nested config still
        carries its selected ``component``; use it to qualify sibling leaves
        and overlay the correct parameter before returning the active set.

        When ``require_all_active`` is true, a checkpoint missing any active
        coordinate fails closed instead of fabricating that coordinate from an
        encoding default.
        """
        if not hasattr(self, "_name_index"):
            self._name_index = self.build_name_index()

        provided: Set[str] = set()
        flat_input: Dict[str, object] = {}

        def _collect(prefix: Tuple[str, ...], value) -> None:
            if not isinstance(value, dict):
                name = self._name_index.get(prefix)
                if name is not None:
                    provided.add(name)
                    flat_input[name] = value
                return
            component = value.get("component")
            for key, child in value.items():
                _collect(prefix + (key,), child)
                if (
                    component is not None
                    and key != "component"
                    and not isinstance(child, dict)
                ):
                    qualified = SEP.join(
                        [*prefix, str(component), str(key)]
                    )
                    if qualified in self.ss.parameters:
                        provided.add(qualified)
                        generic = self._name_index.get(prefix + (key,))
                        if generic is not None and generic != qualified:
                            flat_input.pop(generic, None)
                        flat_input[qualified] = child

        for key, value in config_dict.items():
            _collect((str(key),), value)

        x = self.encode_arm_params(flat_input).reshape(-1)
        decoded = self.decode_flat(x)
        active = self.active_params(decoded)
        result = {name: decoded[name] for name in active}

        def _overlay(path: List[str], value: dict) -> None:
            component = value.get("component")
            for key, child in value.items():
                if isinstance(child, dict):
                    _overlay([*path, str(key)], child)
                elif component is not None and key != "component":
                    qualified = SEP.join(
                        [*path, str(component), str(key)]
                    )
                    if qualified in self.ss.parameters:
                        result[qualified] = child

        _overlay([], config_dict)

        if require_all_active:
            missing = sorted(name for name in active if name not in provided)
            if missing:
                raise ValueError(
                    "checkpoint config is missing active parameter(s) "
                    f"{missing}; refusing to fabricate encoding defaults"
                )
        return result

    # ------------------------------------------------------------------
    # SEP-segment naming-convention helpers (generic — any SearchSpace
    # using the ``a__b__c`` SEP convention).  Live here, not in
    # ``RagSearchSpace``, so the optimizer layer doesn't have to import
    # RAG-specific code just to read a parameter's node label.
    # ------------------------------------------------------------------

    @staticmethod
    def param_leaf_name(name: str) -> str:
        """Last SEP segment.  ``"retriever__bm25__top_k"`` → ``"top_k"``."""
        return name.rsplit(SEP, 1)[-1]

    @staticmethod
    def param_node_label(name: str) -> str:
        """First SEP segment.  ``"retriever__bm25__top_k"`` → ``"retriever"``."""
        return name.split(SEP, 1)[0]

    def params_under(self, root_name: str) -> Set[str]:
        """Param names whose name equals ``root_name`` or starts with
        ``root_name + SEP`` — expands a node label like ``"retriever"`` to the
        concrete column set under it.
        """
        prefix = root_name + SEP
        return {
            n for n in self.param_names
            if n == root_name or n.startswith(prefix)
        }

    # ------------------------------------------------------------------
    # Tensor ops on encoded tensor space (out-of-place; Ax has no
    # normalized-tensor API).
    # ------------------------------------------------------------------

    def snap_to_grid(self, candidates: Tensor) -> Tensor:
        """Snap every ``ChoiceParameter`` column to its normalized level grid
        (``_choice_level_coords``: value-spaced for ordered numeric, index-
        spaced otherwise).

        RangeParameter columns are left untouched.
        """
        candidates = candidates.clone()
        for i, name in enumerate(self.param_names):
            param = self.ss.parameters[name]
            if isinstance(param, ChoiceParameter):
                values_t = torch.tensor(
                    self._choice_grids[name],
                    dtype=candidates.dtype,
                    device=candidates.device,
                )
                col = candidates[..., i].unsqueeze(-1)
                dists = (col - values_t).abs()
                nearest_idx = dists.argmin(dim=-1)
                candidates[..., i] = values_t[nearest_idx]
        return candidates

    def clamp(self, candidates: Tensor) -> Tensor:
        """Canonicalize INACTIVE-param dims to a fixed midpoint.

        For a hierarchical (gated) search space, two configs that differ only
        in a dead/inactive knob must map to the SAME tensor — otherwise the GP
        learns noise on columns whose value can't affect the objective. Each
        row is decoded, its active set computed via Ax's ``dependents``, and
        every inactive column set to ``INACTIVE_DEFAULT``. No-op for
        non-hierarchical SearchSpaces.
        """
        if not self.ss.is_hierarchical:
            return candidates.clone()
        result = candidates.clone()
        for row_idx in range(result.shape[0]):
            decoded = self.decode_flat(result[row_idx])
            active = self.active_params(decoded)
            for dim_idx, name in enumerate(self.param_names):
                if name not in active:
                    result[row_idx, dim_idx] = INACTIVE_DEFAULT
        return result

    # ------------------------------------------------------------------
    # Decode — normalized tensor → typed flat dict.
    # ------------------------------------------------------------------

    def decode_flat(self, x_row: Tensor) -> dict:
        """Decode one normalized tensor row to a flat
        ``{param_name: typed_value}`` dict.

        Includes ALL parameters (active + inactive); callers that need
        active-set filtering use :meth:`active_params` separately.
        """
        result: Dict[str, object] = {}
        for i, name in enumerate(self.param_names):
            param = self.ss.parameters[name]
            v = float(x_row[i].item())
            if isinstance(param, RangeParameter):
                result[name] = _denormalize_range(v, param)
            elif isinstance(param, ChoiceParameter):
                grid = self._choice_grids[name]
                idx = min(range(len(grid)), key=lambda j: abs(v - grid[j]))
                result[name] = param.values[idx]
        return result

    def is_duplicate_decoded(
        self, new_x: Tensor, existing_x: Tensor
    ) -> bool:
        """Check if ``new_x[0]`` decodes to the same active-parameter tuple
        as any row of ``existing_x``.

        For hierarchical spaces, only active parameters are compared.
        """
        if len(existing_x) == 0:
            return False
        hierarchical = self.ss.is_hierarchical
        new_decoded = self.decode_flat(new_x[0])
        new_active = self.active_params(new_decoded) if hierarchical else None
        new_tuple = self._params_to_tuple(new_decoded, new_active)
        for i in range(len(existing_x)):
            ex_decoded = self.decode_flat(existing_x[i])
            ex_active = (
                self.active_params(ex_decoded) if hierarchical else None
            )
            ex_tuple = self._params_to_tuple(ex_decoded, ex_active)
            if new_tuple == ex_tuple:
                return True
        return False

    @staticmethod
    def _params_to_tuple(
        d: dict, active_keys: Optional[Set[str]] = None
    ) -> tuple:
        keys = sorted(active_keys) if active_keys else sorted(d.keys())
        return tuple(d.get(k) for k in keys)

    # ------------------------------------------------------------------
    # Active-set queries on a decoded flat dict.
    # ------------------------------------------------------------------

    def active_params(self, decoded: dict) -> Set[str]:
        """Names of the active parameters for a (full or partial) parameterization.

        Delegates to Ax's native hierarchical cast — the single source of truth
        for which params survive the ``ChoiceParameter.dependents`` tree.
        Hand-rolling this is exactly what this module's docstring forbids ("does
        not wrap things Ax already provides"); the previous single-pass version
        also over-included grandchildren of a gated-off subtree (a child whose
        grandparent disables the branch but whose parent choice still "votes").
        Ax casts the whole tree correctly in one call.

        ``check_all_parameters_present=False`` tolerates a partial ``decoded``.
        Non-hierarchical spaces have no gating → every parameter is active.
        (Uses Ax's ``_cast_parameterization``; the public ``cast_observation_
        features`` wraps the same logic if the private entry point ever moves.)
        """
        if not self.ss.is_hierarchical:
            return set(self.param_names)
        casted = self.ss._cast_parameterization(
            decoded, check_all_parameters_present=False,
        )
        return set(casted.keys())

    # ------------------------------------------------------------------
    # Flat → nested decode (Ax hierarchical-name convention).
    #
    # Ax names dependents like ``<parent>__<choice_value>__<param>`` (e.g.
    # ``retriever__bm25__top_k`` for the ``bm25`` branch under
    # ``retriever__module_type``). The middle segment is a *branch label*: at
    # eval time only one branch is active, so the evaluator only ever sees one
    # value per ``(parent, leaf)`` slot — middle segments collapse out.
    #
    # Conventions for ``decode_to_nested``:
    #   * 1 segment  ``"name"``                    → ``config[name]``
    #   * 2 segments ``"a__b"``                    → ``config[a][b]``
    #   * 3+ segments ``"a__b__...__z"``           → ``config[a][z]``
    #     (drop ALL middle segments — branch labels)
    #
    # Subclasses override ``_insert_param`` / ``_index_key`` to add
    # benchmark-specific deviations (e.g. namespace lifts, depth-3 retention).
    # ------------------------------------------------------------------

    def decode_to_nested(self, flat: Dict[str, object]) -> dict:
        """Convert a flat ``{name: value}`` dict into the nested-dict layout
        evaluators consume. Performs no active-set filtering — pass only
        active params if you need it. See :meth:`decode_to_config` for the
        tensor-row pipeline (decode + active-filter + nest)."""
        config: dict = {}
        for name, value in flat.items():
            self._insert_param(config, name, value)
        return config

    def _insert_param(self, config: dict, name: str, value: object) -> None:
        """Insert one ``(name, value)`` pair into ``config`` per the universal
        ``__``-segment rule. Override hook for naming-convention deviations."""
        parts = name.split(SEP)
        if len(parts) == 1:
            config[name] = value
            return
        if len(parts) == 2:
            config.setdefault(parts[0], {})[parts[1]] = value
            return
        # 3+ segments: collapse middle (Ax branch labels) — keep first and last.
        config.setdefault(parts[0], {})[parts[-1]] = value

    def decode_to_config(self, x_row: Tensor) -> dict:
        """Decode a normalized tensor row into a hierarchical nested
        config dict, restricted to active parameters. Composes
        :meth:`decode_flat`, :meth:`active_params`, and
        :meth:`decode_to_nested`."""
        flat = self.decode_flat(x_row)
        active = self.active_params(flat)
        return self.decode_to_nested(
            {k: v for k, v in flat.items() if k in active}
        )

    def build_name_index(self) -> Dict[Tuple[str, ...], str]:
        """Reverse of :meth:`decode_to_nested` at the schema level — maps a
        nested-key path tuple back to the flat parameter name. Useful for
        encoders that consume nested config dicts and need to write them
        into a flat tensor.

        Collisions: when two parameters share the same key path (e.g.
        ``retriever__bm25__top_k`` and ``retriever__dense__top_k`` both map
        to ``("retriever", "top_k")``), the first one in :attr:`param_names`
        wins. Downstream :meth:`clamp` zeroes out the inactive column."""
        idx: Dict[Tuple[str, ...], str] = {}
        for name in self.param_names:
            parts = name.split(SEP)
            idx.setdefault(self._index_key(parts), name)
        return idx

    def _index_key(self, parts: List[str]) -> Tuple[str, ...]:
        """Override hook mirroring :meth:`_insert_param` for the inverse map."""
        if len(parts) == 1:
            return (parts[0],)
        if len(parts) == 2:
            return (parts[0], parts[1])
        return (parts[0], parts[-1])


# ---------------------------------------------------------------------------
# Module-level helpers used by the encode/decode paths. Kept module-level
# (not class methods) so callers can re-use them without instantiating the
# ssops class.
# ---------------------------------------------------------------------------


def _normalize_range(v: float, param: RangeParameter) -> float:
    """Map a raw RangeParameter value to [0,1]."""
    if param.log_scale:
        return math.log(v / param.lower) / math.log(param.upper / param.lower)
    return (v - param.lower) / (param.upper - param.lower)


def _denormalize_range(v_norm: float, param: RangeParameter):
    """Inverse of :func:`_normalize_range`. Handles INT casting."""
    if param.log_scale:
        actual = param.lower * (param.upper / param.lower) ** v_norm
    else:
        actual = param.lower + v_norm * (param.upper - param.lower)
    if param.parameter_type == ParameterType.INT:
        actual = max(int(param.lower), round(actual))
    return actual
