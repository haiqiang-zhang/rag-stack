"""RAG-specific SearchSpace adapter — owns ALL RAG-domain knowledge for
hyperparameter search.

The generic Ax adapter ``AxSearchSpaceOps`` (in
:mod:`rag_stack.search_space.ax_search_space_ops`) handles encode/decode for
any Ax ``SearchSpace``. This module specializes that for the rag-stack
domain:

  * **YAML factory**: :meth:`RagSearchSpace.from_yaml` parses a RAG pipeline
    YAML config (``node_lines``, ``modules``, ``vectordb``, ``corpus``,
    ``pipeline.mode``, ``iterative.<method>.<knob>``) into a hierarchical
    Ax ``SearchSpace``.
  * **Naming-convention overrides**: ``vectordb__<vdb>__<param>`` lifts the
    ``vectordb`` namespace, ``corpus__<sub>__<param>`` keeps depth-3 nesting.
  * **FAISS invariant**: ``decode_flat`` clamps ``nprobe ≤ nlist``.
  * **CSV column convention**: :meth:`flatten_2deep` for controller writers.
  * **Iterative-pipeline mode constants**.

The optimizer layer (``ax_qnehvi``, ``topo_decoupled_v2``, etc.) accepts a
generic ``AxSearchSpaceOps`` interface and does NOT depend on this module
— callers that need RAG behavior instantiate ``RagSearchSpace`` and pass
it as ``ssops=...``.
"""

from __future__ import annotations

import itertools
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, cast

from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace
from torch import Tensor

from rag_stack.search_space import ITERATIVE_PIPELINE_MODES
from rag_stack.search_space.ax_search_space_ops import (
    INACTIVE_DEFAULT,
    SEP,
    AxSearchSpaceOps,
)
from rag_stack.search_space.system import extract_system_params
from rag_stack.pipeline_contract import (
    hybrid_capability_errors,
    hybrid_structure_errors,
    normalize_retrieval_aliases,
)
from rag_stack_evaluator.static_rag_evaluator.vectordb import VECTORDB_SKIP_KEYS
from rag_stack_evaluator.static_rag_evaluator.schema.node import NODE_SKIP_KEYS

# Re-export SEP / INACTIVE_DEFAULT for callers that previously imported them
# from ``rag_stack.optimizer.search_space``.
__all__ = [
    "INACTIVE_DEFAULT",
    "SEP",
    "ITERATIVE_PIPELINE_MODES",
    "ITERATIVE_INCOMPATIBLE_NODES",
    "REACT_ACTIVE_STAGES",
    "parse_dataflow_spec",
    "RagSearchSpace",
    "algo_vectordb_blocks",
    "is_sweepable",
    "scalarify_value",
]

# The stages the static_gt react engine actually drives each round: retrieval,
# an OPTIONAL passage_reranker over each round's candidates, and the generator.
# Everything else in a co-mode block (rag_dataflow swept over
# sequential+react) is sequential-only: its knobs are gated OFF via Ax
# ``dependents`` when a trial picks react, and its node is stripped from the
# resolved node_lines (so the cost model never provisions a stage the loop
# never calls). MUST match what ``StaticRAGEvaluatorQualityOnly._run_react``
# extracts from node_lines.
REACT_ACTIVE_STAGES = frozenset({
    "semantic_retrieval", "passage_reranker", "generator",
})

def parse_dataflow_spec(spec) -> Tuple[List[str], Dict[str, dict]]:
    """Parse a ``rag_dataflow`` YAML spec into ``(branch_names, branch_subknobs)``.

    Canonical shape — a list whose entries are either a bare branch name or a
    single-key mapping carrying that branch's SUB-CONFIG (knobs that only
    exist when the branch is chosen)::

        rag_dataflow:
        - sequential
        - react:
            max_iter: [3, 5]

    The hierarchy declared here is wired into Ax ``ChoiceParameter.dependents``
    by the search-space builder — Ax owns activation globally, exactly like
    module ``component`` branches. No knob name is hardcoded anywhere: adding
    a branch sub-knob in the YAML is all it takes.
    """
    if not isinstance(spec, list) or not spec:
        raise ValueError(
            "pipeline.rag_dataflow must be a non-empty list of branches "
            "(a branch name, or {branch: {knob: spec}})."
        )
    names: List[str] = []
    subknobs: Dict[str, dict] = {}
    for entry in spec:
        if isinstance(entry, str):
            name, knobs = entry, {}
        elif isinstance(entry, dict) and len(entry) == 1:
            name, knobs = next(iter(entry.items()))
            knobs = {} if knobs is None else knobs
            if not isinstance(name, str) or not isinstance(knobs, dict):
                raise ValueError(
                    f"pipeline.rag_dataflow branch entry malformed: {entry!r}"
                )
        else:
            raise ValueError(
                "pipeline.rag_dataflow entries must be a branch name or a "
                f"single-key mapping {{branch: {{knob: spec}}}}; got {entry!r}"
            )
        if name in names:
            raise ValueError(f"pipeline.rag_dataflow lists branch {name!r} twice.")
        names.append(name)
        subknobs[name] = knobs
    return names, subknobs


# ===========================================================================
# Sweepable-value helpers — shared by RagSearchSpace and consumers like the
# Controller's config-resolution pipeline.
# ===========================================================================

def is_sweepable(v) -> bool:
    """Return True if ``v`` is an unresolved sweep parameter spec.

    A value is sweepable when it carries *multiple* candidate values that the
    optimizer must pick from at evaluate time. Three YAML shapes qualify:

      * ``[v1, v2, ...]`` with more than one element (ChoiceParameter source).
      * ``{"ordered": [v1, v2, ...]}``   (ordered ChoiceParameter source).
      * ``{"range": [lo, hi], ...}``                   (RangeParameter source).

    Scalars and single-element lists are NOT sweepable — they're fixed values
    that get scalarified to themselves.
    """
    if isinstance(v, list) and len(v) > 1:
        return True
    if isinstance(v, dict) and ("range" in v or "ordered" in v):
        return True
    return False


def sweep_values(v) -> list:
    """All candidate values of a raw YAML knob value, SHAPE-AGNOSTIC.

    scalar → ``[v]``; list → itself; ``{ordered: [...]}`` → its values
    (declaration order); ``{range: [lo, hi]}`` → the two bounds (endpoints,
    not a grid). The one helper every consumer that peeks at UNRESOLVED
    config values must use — pattern-matching ``isinstance(v, list)`` by
    hand silently mishandles the dict sweep forms (a ``{ordered:}`` model
    reached ``resolve_tokenizer_name`` whole and crashed on ``.lower()``).
    """
    if isinstance(v, dict):
        if "ordered" in v:
            ov = v["ordered"]
            return list(ov) if isinstance(ov, list) else [ov]
        if "range" in v:
            rv = v["range"]
            return list(rv) if isinstance(rv, list) else [rv]
    if isinstance(v, list):
        return list(v)
    return [v]


def scalarify_value(v):
    """Collapse a resolved sweep-spec to its scalar value, or raise.

    Called after the optimizer provides concrete overrides. A surviving
    unresolved spec (multi-element list or ``range`` dict) means the
    optimizer didn't decide on that parameter, which is a bug — raise loudly
    rather than silently picking a value.
    """
    if isinstance(v, list):
        if len(v) == 1:
            return v[0]
        raise ValueError(
            f"Unresolved sweep parameter: {v}. "
            f"Optimizer must provide a concrete value for all sweepable parameters."
        )
    if isinstance(v, dict) and ("range" in v or "ordered" in v):
        raise ValueError(
            f"Unresolved range/ordered parameter: {v}. "
            f"Optimizer must provide a concrete value for all sweepable parameters."
        )
    return v


def pipeline_method_blocks(config: dict) -> list:
    """Parse ``algo_search_space.pipeline`` (the schema's pipeline section) into
    the internal list of method blocks every pipeline reader consumes.

    Two valid shapes, routed by eval backend:
      * ``{node_lines: [...]}``  — static_gt (sequential-ONLY): wrapped as a
        single implicit ``sequential`` method block (no ``mode`` search dim).
      * ``{methods: [...]}``     — flashrag: the list of method blocks verbatim,
        each carrying its own ``mode`` + node_lines + method knobs.

    Returns ``[]`` when ``algo_search_space.pipeline`` is absent. This is
    new-schema PARSING (not a back-compat shim) — it never reads a top-level
    ``pipeline`` key.
    """
    pipeline = (config.get("algo_search_space") or {}).get("pipeline")
    if pipeline is None:
        return []
    if isinstance(pipeline, dict):
        if "methods" in pipeline:
            methods = pipeline["methods"]
            if not isinstance(methods, list):
                raise ValueError(
                    "algo_search_space.pipeline.methods must be a list of method blocks."
                )
            return [normalize_retrieval_aliases(block) if isinstance(block, dict) else deepcopy(block)
                    for block in methods]
        if "node_lines" in pipeline:
            # Carry pipeline-level sweep knobs (e.g. `rag_dataflow` + `max_iter`
            # for the collaborative sequential+react co-mode) into the single
            # block, so the builder turns them into FLAT search dims alongside the
            # SHARED node_lines — one knob, no per-method subtree duplication.
            block = deepcopy({
                k: v for k, v in pipeline.items() if k != "methods"
            })
            block.setdefault("mode", "sequential")
            return [normalize_retrieval_aliases(block)]
    raise ValueError(
        "algo_search_space.pipeline must be a mapping with 'node_lines' "
        "(static_gt, sequential-only) or 'methods' (flashrag, list of blocks)."
    )


def algo_corpus_block(config: dict) -> dict:
    """The ``algo_search_space.corpus`` block (corpus chunker search space), or {}."""
    return (config.get("algo_search_space") or {}).get("corpus") or {}


def algo_vectordb_blocks(config: dict) -> list:
    """The ``algo_search_space.vectordb`` block list (vector-index search space).

    The vectordb section is ALGO search space — index family and build knobs
    move retrieval quality — so it lives inside ``algo_search_space`` next to
    ``corpus`` and ``pipeline``. ``resolve_vectordbs`` emits the chosen,
    scalarified blocks at the runtime top level (``config["vectordb"]``) for
    the evaluators, mirroring the corpus/pipeline input→runtime lift.

    A TOP-LEVEL ``vectordb:`` section is the pre-move schema → hard error
    (never silently ignored: the builder would emit no vectordb dims and the
    run would fail far from the cause).
    """
    blocks = (config.get("algo_search_space") or {}).get("vectordb")
    if blocks is None and isinstance(config.get("vectordb"), list):
        raise ValueError(
            "Top-level 'vectordb:' is the old schema — move the section under "
            "'algo_search_space.vectordb:' (it is algo search space, like "
            "corpus/pipeline). The resolved runtime config still exposes "
            "config['vectordb'] for evaluators."
        )
    return blocks or []


def _referenced_vdb_names(config: dict) -> set:
    """Vectordb block names referenced by the RESOLVED pipeline's retrieval
    modules (scalar ``vectordb`` values inside ``config["node_lines"]``).
    Empty set when no module carries a resolved name (e.g. flashrag method
    blocks) — callers then keep every declared block."""
    out: set = set()
    for node_line in config.get("node_lines") or []:
        for node in node_line.get("nodes", []) or []:
            for mod in node.get("modules", []) or []:
                v = mod.get("vectordb")
                if isinstance(v, str) and v:
                    out.add(v)
    return out


# Node types that the iterative cost path (RAGO IterativeRAGSweep) cannot
# represent — `stages` must be exactly [retrieval, prefill, decode].
ITERATIVE_INCOMPATIBLE_NODES = frozenset({
    "passage_reranker",
    "passage_compressor",
    "passage_filter",
    "query_expansion",
})


# ===========================================================================
# RagSearchSpace — Ax adapter + ``__`` naming convention helpers.
# ===========================================================================

class RagSearchSpace(AxSearchSpaceOps):
    """SearchSpace adapter with rag-stack ``__`` three-segment naming
    convention.

    Inherits all generic Ax operations (encode / decode_flat / snap_to_grid /
    clamp / active_params / cat_dims_set / decode_to_config) from
    :class:`AxSearchSpaceOps`. Hierarchy is read from Ax's native
    ``ChoiceParameter.dependents`` — this class adds no parallel hierarchy layer.

    Adds rag-stack-specific naming-convention deviations on top of the
    generic ``a__b__c`` Ax convention handled by :class:`AxSearchSpaceOps`:

      * 3-segment ``vectordb__<vdb>__<param>`` → ``config[<vdb>][<param>]``
        (lift the ``vectordb`` namespace — non-derivable from Ax dependents).
      * 3-segment ``corpus__<sub>__<param>`` → ``config["corpus"][<sub>][<param>]``
        (keep depth-3 instead of collapsing the middle segment).

    Other naming patterns (1-seg, 2-seg, ``<node>__<module>__<param>``,
    multi-segment with branch labels) inherit the generic implementation
    in the parent class — see ``AxSearchSpaceOps._insert_param`` /
    ``_index_key``.

    Also adds:

      * :meth:`from_yaml` — RAG YAML pipeline config → SearchSpace factory
      * :meth:`decode_flat` override — FAISS ``nprobe <= nlist`` invariant
      * :meth:`flatten_2deep` — 2-deep CSV column convention
    """

    # ------------------------------------------------------------------
    # YAML factory.
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        config: dict,
        tkwargs: Optional[Dict] = None,
        *,
        check_performance_support: bool = True,
    ) -> "RagSearchSpace":
        """Build a :class:`RagSearchSpace` adapter from a parsed RAG pipeline
        YAML config dict. The adapter wraps the raw Ax SearchSpace produced
        by :meth:`build_ax_searchspace` and is the canonical entry point —
        callers that need the raw Ax ``SearchSpace`` (e.g. to feed an
        optimizer registry that expects raw) read ``adapter.ss``.
        """
        ss = cls.build_ax_searchspace(
            config, check_performance_support=check_performance_support,
        )
        inst = cls(ss, tkwargs=tkwargs)
        # Stash the parsed YAML so config-aware helpers (e.g. composite-
        # objective derivation) don't have to be threaded the config through
        # every call site.
        inst._config = config
        return inst

    @staticmethod
    def build_ax_searchspace(
        config: dict, *, check_performance_support: bool = True,
    ) -> SearchSpace:
        """Translate a RAG pipeline YAML config into a (possibly hierarchical)
        Ax ``SearchSpace``. Most callers want :meth:`from_yaml` instead, which
        returns the operational adapter; use this static method only when you
        need the raw Ax ``SearchSpace`` to pass to an optimizer registry that
        expects it directly.

        The pipeline schema lives under ``config["algo_search_space"]["pipeline"]``
        — ``{node_lines: [...]}`` (static_gt, sequential-only) or ``{methods: [...]}``
        (flashrag), parsed into a list of method blocks, each carrying ``node_lines``
        + optional method-specific scalar/swept knobs. Top-level ``node_lines:``
        is **rejected by the config validator**; the search-space builder reads
        node_lines only from inside each method block (via
        :func:`_extract_pipeline_mode_params`).
        ``check_performance_support=False`` is for execution previews: resolve
        a structurally valid pipeline and record its real calls before the CM
        reports unsupported operations. It does not change the selected
        performance source, deployment dimensions, or structural validation.
        Optimizer callers keep the default eager capability check.
        """
        for block in pipeline_method_blocks(config):
            errors = hybrid_structure_errors(
                block, backend=str((config.get("global") or {}).get("eval_backend", "static_gt")).lower(),
            )
            if errors:
                raise ValueError("Invalid hybrid retrieval workflow: " + " ".join(errors))
            if check_performance_support:
                errors = hybrid_capability_errors(
                    block,
                    performance_source=str((config.get("system") or {}).get("performance_source", "cost_model")).lower(),
                )
                if errors:
                    raise NotImplementedError(" ".join(errors))
            for node_line in block.get("node_lines", []) or []:
                for node in node_line.get("nodes", []) or []:
                    stage = str(node.get("stage", ""))
                    components = {
                        str(module.get("component", ""))
                        for module in node.get("modules", []) or []
                        if isinstance(module, dict)
                    }
                    if stage == "passage_augmenter":
                        raise NotImplementedError(
                            f"{stage or '<missing-stage>'}/{sorted(components)} "
                            "is not implemented in the current paper search space"
                        )

        parameters: list = []

        # 1. Pipeline tree (mode root + per-method subtrees, all wired via
        #    ChoiceParameter.dependents so only the chosen method's params
        #    are active per trial).
        parameters.extend(_extract_pipeline_mode_params(config))

        # 2. Corpus / chunker preprocessing knobs (always active).
        parameters.extend(_extract_corpus_params(config))

        # 3. Vectordb-level optimizable params (always active).
        parameters.extend(_extract_vectordb_params(config))

        # 4. System params are mode-specific. Cost-model mode emits none
        #    because RAGO sweeps deployment from system.search_space internally;
        #    measured mode emits runtime knobs plus a budget-valid GPU layout id.
        parameters.extend(extract_system_params(config, _make_param))

        # 5. Family gating: attach each vectordb selector's full gated set
        #    (the referenced blocks' params + family-owned system knobs) —
        #    needs every section extracted first, so it runs last.
        _wire_vdb_family_gating(parameters, config)

        return SearchSpace(parameters=parameters)

    # NOTE: SEP-segment naming-convention helpers (param_leaf_name,
    # param_node_label, params_under) live in the
    # generic ``AxSearchSpaceOps`` base — they were never RAG-specific.
    # RagSearchSpace inherits them unchanged.

    # ------------------------------------------------------------------
    # decode_flat override (FAISS nprobe<=nlist invariant) + name-convention
    # deviations (vectordb namespace lift, corpus depth-3 retention).
    # ------------------------------------------------------------------

    def decode_flat(self, x_row: Tensor) -> dict:
        """Decode ``x_row`` to ``{name: typed_value}`` (all params), then
        apply RAG-specific FAISS invariant ``nprobe <= nlist`` (cross-namespace
        suffix-based matching).

        The constraint is a no-op for SearchSpaces without ``nprobe`` /
        ``nlist`` parameters (yahpo, dtlz2, nb201, ...).
        """
        result = super().decode_flat(x_row)
        nprobe_keys = [
            k for k in result if k == "nprobe" or k.endswith(SEP + "nprobe")
        ]
        nlist_keys = [
            k for k in result if k == "nlist" or k.endswith(SEP + "nlist")
        ]
        if nprobe_keys and nlist_keys:
            nlist_val = result[nlist_keys[0]]
            for nprobe_key in nprobe_keys:
                result[nprobe_key] = min(result[nprobe_key], nlist_val)
        return result

    # PQ-only vectordb keys: meaningless for IVF-Flat. When a swept
    # ``index_type`` resolves to "flat", drop these so flat arms that carried
    # swept-but-unused M/nbits collapse to ONE canonical config — correct
    # dedup (config_hash) + no wasted quality evals on identical flat configs.
    _PQ_ONLY_VDB_KEYS = (
        "M", "dsub", "nbits", "by_residual", "use_precomputed_table",
        "refine_ratio", "refine_dtype_bytes",
    )

    def decode_to_config(self, x_row: Tensor) -> dict:
        """Decode an arm to a nested config, then canonicalize IVF-Flat
        vectordbs by dropping PQ-only keys (so flat arms dedup when
        ``index_type`` is a swept dimension)."""
        cfg = super().decode_to_config(x_row)
        for sub in cfg.values():
            if isinstance(sub, dict) and sub.get("index_type") == "flat":
                for k in self._PQ_ONLY_VDB_KEYS:
                    sub.pop(k, None)
        return cfg

    def _insert_param(self, config: dict, name: str, value) -> None:
        """Override: handle ``vectordb__`` / ``corpus__`` 3-segment deviations
        and the new ``pipeline__<method>__...`` per-method-tree convention.
        Delegates everything else to the generic parent."""
        parts = name.split(SEP)
        if len(parts) >= 3:
            head, mid, leaf = parts[0], parts[1], parts[-1]
            if head == "vectordb":
                # vectordb__<vdb>__<param> → config[<vdb>][<param>]
                # (lift the vectordb namespace; non-derivable from Ax
                # dependents — Ax has no notion of "lift this prefix").
                config.setdefault(mid, {})[leaf] = value
                return
            if head == "corpus":
                # corpus__<sub>__<param> → config["corpus"][<sub>][<param>]
                # (keep depth-3 instead of collapsing middle).
                config.setdefault(head, {}).setdefault(mid, {})[leaf] = value
                return
            if head == "pipeline":
                # pipeline__<method>__<rest...> → config["pipeline"][<method>][...]
                # using the same SEP rules as the generic parent on <rest>.
                method = mid
                rest = parts[2:]
                method_cfg = config.setdefault(head, {}).setdefault(method, {})
                if len(rest) == 1:
                    method_cfg[rest[0]] = value
                elif len(rest) == 2:
                    method_cfg.setdefault(rest[0], {})[rest[1]] = value
                else:
                    # 3+ rest segments: collapse middle Ax branch labels
                    # (matches the parent _insert_param for component-style
                    # naming inside a method's subtree).
                    method_cfg.setdefault(rest[0], {})[rest[-1]] = value
                return
        super()._insert_param(config, name, value)

    def _index_key(self, parts: List[str]) -> Tuple[str, ...]:
        """Override mirroring :meth:`_insert_param` for the inverse map."""
        if len(parts) >= 3:
            head, mid, leaf = parts[0], parts[1], parts[-1]
            if head == "vectordb":
                return (mid, leaf)
            if head == "corpus":
                return (head, mid, leaf)
            if head == "pipeline":
                method = mid
                rest = parts[2:]
                if len(rest) == 1:
                    return (head, method, rest[0])
                if len(rest) == 2:
                    return (head, method, rest[0], rest[1])
                return (head, method, rest[0], rest[-1])
        return super()._index_key(parts)

    # ------------------------------------------------------------------
    # CSV column convention — 2-deep flatten used by ``all_evaluations.csv``
    # and ``pareto_frontier.csv``.
    # ------------------------------------------------------------------

    def flatten_2deep(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a nested config dict into ``{"<top>__<leaf>": value}``
        form (2-deep walk). Used to derive CSV column headers + cell
        values from a single nested config."""
        out: Dict[str, Any] = {}
        for top, sub in cfg.items():
            if isinstance(sub, dict):
                for leaf, val in sub.items():
                    out[f"{top}{SEP}{leaf}"] = val
            else:
                out[top] = sub
        return out

    # ------------------------------------------------------------------
    # Exhaustive enumeration — only meaningful for fully-discrete spaces.
    # ------------------------------------------------------------------

    # Cap on the raw Cartesian product the enumerator will visit before
    # giving up. ``controller.get_pareto`` only uses this path when the
    # space is small enough to fully evaluate (≤ ``n_both_init``), so once
    # the product clearly exceeds any reasonable batch budget there is no
    # point materialising it just to throw it away. Hierarchical search
    # spaces with ``dependents`` wiring (e.g. pipeline__mode tree) blow up
    # the raw product by orders of magnitude even though most combinations
    # collapse to identical active configs after dedup.
    _ENUMERATE_PRODUCT_LIMIT = 100_000

    def enumerate_all_configs(self) -> Optional[List[dict]]:
        """Enumerate every discrete combination in the search space.

        Returns a list of nested config dicts if ALL parameters are
        ChoiceParameter AND the raw Cartesian product is below
        :attr:`_ENUMERATE_PRODUCT_LIMIT`. Returns ``None`` when any
        parameter is a RangeParameter (continuous) OR the discrete
        product is too large to be useful.

        Hierarchical-aware: respects ``ChoiceParameter.dependents`` via
        :meth:`active_params` so inactive branches are filtered out before
        nesting. Combinations that differ only in inactive-branch values
        collapse to the same active configuration and are deduplicated.

        Empty search space → ``[{}]`` (one configuration with no overrides).
        """
        params = self.ss.parameters
        if not params:
            return [{}]
        for p in params.values():
            if isinstance(p, RangeParameter):
                return None
        param_names = list(params.keys())
        param_values = [
            cast(ChoiceParameter, params[name]).values for name in param_names
        ]

        # Bail out early when the raw product is too big to materialise.
        # The product is taken BEFORE ``active_params`` dedup so this is
        # an upper bound; in practice the unique active-config count is
        # much smaller, but we'd still have to iterate the full product
        # to discover that — which can be hundreds of billions for a
        # 20+ dim hierarchical RAG search space.
        total = 1
        for vals in param_values:
            total *= len(vals)
            if total > self._ENUMERATE_PRODUCT_LIMIT:
                return None

        seen: set = set()
        configs: List[dict] = []
        for combo in itertools.product(*param_values):
            flat = dict(zip(param_names, combo))
            active = self.active_params(flat)
            active_flat = {k: v for k, v in flat.items() if k in active}
            key = tuple(sorted(active_flat.items()))
            if key in seen:
                continue
            seen.add(key)
            configs.append(self.decode_to_nested(active_flat))
        return configs

    # ------------------------------------------------------------------
    # Decoded-config baseline (fills in fixed/single-value YAML defaults).
    # ------------------------------------------------------------------

    def build_base_decoded_config(self, config: dict) -> dict:
        """Extract a complete decoded config from the parsed YAML.

        Includes fixed/single-value params that are NOT in the search space,
        so callers can merge optimizer-suggested overrides on top to produce
        full pipeline configurations.

        Walks the same shape that :meth:`build_ax_searchspace` reads. Under
        the new list-of-method-blocks schema, returns:

        ``{vdb_name: {...}, "pipeline": {<method>: {<stage>: {...}, <knob>: ...}, ...}}``

        Top-level keys outside ``pipeline`` are vectordb-by-name (matches the
        existing ``vectordb__<vdb>__<param>`` naming convention). Inside
        ``pipeline`` is the per-method tree mirroring what
        :meth:`_insert_param` produces from optimizer-suggested overrides.
        """
        base: dict = {}

        # 1. Corpus/chunker scalar defaults.  Swept chunker values arrive in
        # decoded optimizer configs through ``corpus__chunker__*``; fixed
        # values do not.  They are nevertheless algorithm semantics (they
        # change the passages, quality and quality trace), so they must be in
        # every enriched config used by memo/QualityStore identity.  Keep
        # ``raw_path`` out: corpus bytes are guarded separately by the dataset
        # fingerprint and paths are intentionally portable across machines.
        corpus = algo_corpus_block(config)
        chunker = corpus.get("chunker") if isinstance(corpus, dict) else None
        if isinstance(chunker, dict):
            fixed_chunker: dict = {}
            for k, v in chunker.items():
                if is_sweepable(v):
                    continue
                fixed_chunker[k] = (
                    v[0] if isinstance(v, list) and len(v) == 1 else v
                )
            if fixed_chunker:
                base["corpus"] = {"chunker": fixed_chunker}

        # 2. Vectordb-level scalar / single-element params (excluding search-space-only keys).
        for vdb in algo_vectordb_blocks(config):
            vdb_name = vdb.get("name", "")
            if not vdb_name:
                continue
            vdb_cfg: dict = {}
            for k, v in vdb.items():
                if k in VECTORDB_SKIP_KEYS:
                    continue
                # Nested index_type block: choice + per-type knobs are search
                # params; resolution collapses it. Don't dump the raw dict here.
                if k == "index_type" and isinstance(v, dict):
                    continue
                if is_sweepable(v):
                    continue
                vdb_cfg[k] = v[0] if isinstance(v, list) and len(v) == 1 else v
            if vdb_cfg:
                base[vdb_name] = vdb_cfg

        # 3. Per-method-block params under algo_search_space.pipeline (method blocks).
        pipeline_blocks = pipeline_method_blocks(config)
        if not isinstance(pipeline_blocks, list):
            return base

        pipeline_tree: dict = {}
        for block in pipeline_blocks:
            if not isinstance(block, dict):
                continue
            method = block.get("mode")
            if not isinstance(method, str):
                continue
            method_cfg: dict = {}

            # Method-specific scalar knobs (max_iter_num, threshold, etc.).
            for k, v in block.items():
                if k in ("mode", "node_lines"):
                    continue
                if is_sweepable(v):
                    continue
                method_cfg[k] = v[0] if isinstance(v, list) and len(v) == 1 else v

            # Per-node defaults within this method's node_lines.
            for node_line in block.get("node_lines", []) or []:
                for node in node_line.get("nodes", []) or []:
                    stage = node["stage"]
                    node_cfg: dict = {}

                    for k, v in node.items():
                        if k in ("stage", "strategy", "modules"):
                            continue
                        if is_sweepable(v):
                            continue
                        node_cfg[k] = v[0] if isinstance(v, list) and len(v) == 1 else v

                    modules = node.get("modules", [])
                    if modules:
                        mod = modules[0]
                        node_cfg["component"] = mod["component"]
                        for k, v in mod.items():
                            if k in ("component", "generator_backend"):
                                continue
                            if is_sweepable(v):
                                continue
                            node_cfg[k] = v[0] if isinstance(v, list) and len(v) == 1 else v

                    if node_cfg:
                        method_cfg[stage] = node_cfg

            if method_cfg:
                pipeline_tree[method] = method_cfg

        if pipeline_tree:
            base["pipeline"] = pipeline_tree
        return base

    def enrich_configs(
        self, configs: List[dict], base_decoded: dict,
    ) -> List[dict]:
        """Merge each decoded optimizer config with the base defaults.

        Optimizer configs only carry the search-space overrides; this fills
        in every fixed/default value so the returned configs are complete
        end-to-end pipeline specifications. ``base_decoded`` is the output
        of :meth:`build_base_decoded_config` — passed in so callers cache it
        across calls instead of rebuilding per-config.

        The merge is **recursive** so the per-method pipeline subtree
        (``base_decoded["pipeline"][<method>][<node>][...]``) gets correctly
        composed with the optimizer's overrides without losing un-swept
        defaults at deeper levels.  Component-specific fixed fields are then
        rebound from the selected module template.  ``base_decoded`` uses the
        first module as its structural default, so leaving those fields in
        place after another component is selected would, for example, attach
        the first prompt-maker's prompt to ``long_context_reorder``.
        """
        def _deep_merge(into: dict, src: dict) -> None:
            for key, value in src.items():
                if isinstance(value, dict) and isinstance(into.get(key), dict):
                    _deep_merge(into[key], value)
                else:
                    into[key] = value

        def _bind_selected_component_defaults(
            merged: dict,
            decoded: dict,
        ) -> None:
            source_config = getattr(self, "_config", None)
            merged_pipeline = merged.get("pipeline")
            if not isinstance(source_config, dict) or not isinstance(
                merged_pipeline, dict,
            ):
                return
            decoded_pipeline = decoded.get("pipeline")
            if not isinstance(decoded_pipeline, dict):
                decoded_pipeline = {}

            for block in pipeline_method_blocks(source_config):
                method = block.get("mode")
                method_config = merged_pipeline.get(method)
                if not isinstance(method, str) or not isinstance(
                    method_config, dict,
                ):
                    continue
                decoded_method_config = decoded_pipeline.get(method)
                if not isinstance(decoded_method_config, dict):
                    decoded_method_config = {}

                for node_line in block.get("node_lines", []) or []:
                    for node in node_line.get("nodes", []) or []:
                        stage = node.get("stage")
                        stage_config = method_config.get(stage)
                        modules = node.get("modules", []) or []
                        if not isinstance(stage, str) or not isinstance(
                            stage_config, dict,
                        ) or not modules:
                            continue

                        selected_component = stage_config.get(
                            "component", modules[0].get("component"),
                        )
                        selected_module = next(
                            (
                                module for module in modules
                                if isinstance(module, dict)
                                and module.get("component")
                                == selected_component
                            ),
                            None,
                        )
                        if selected_module is None:
                            continue
                        decoded_stage_config = decoded_method_config.get(stage)
                        if not isinstance(decoded_stage_config, dict):
                            decoded_stage_config = {}

                        # The decoded stage is flat, so remove fixed fields
                        # belonging to unselected module templates and rebind
                        # the selected template. Node-level fields with the
                        # same name retain node ownership. A selected module's
                        # sweepable field must keep its decoded optimizer value;
                        # a field absent from the selected module must be
                        # removed so enriched checkpoint rows self-heal.
                        node_owned_keys = {
                            key for key in node if key not in NODE_SKIP_KEYS
                        }
                        module_owned_keys = {
                            key
                            for module in modules
                            if isinstance(module, dict)
                            for key in module
                            if key not in ("component", "generator_backend")
                            and key not in node_owned_keys
                        }
                        for key in module_owned_keys:
                            if key not in selected_module:
                                stage_config.pop(key, None)
                                continue
                            value = selected_module[key]
                            # Dict-form module selectors are declarations, not
                            # fixed runtime defaults.  If the decoded optimizer
                            # arm supplied the selected scalar family, preserve
                            # it.  Overwriting it with the declaration made the
                            # human-readable checkpoint CSV lossy: resume then
                            # defaulted the selector and marked the real
                            # family's knobs inactive.
                            if (
                                isinstance(value, dict)
                                and key in decoded_stage_config
                                and not isinstance(decoded_stage_config[key], dict)
                            ):
                                stage_config[key] = deepcopy(
                                    decoded_stage_config[key]
                                )
                                continue
                            if is_sweepable(value):
                                if key in decoded_stage_config:
                                    stage_config[key] = deepcopy(
                                        decoded_stage_config[key]
                                    )
                                else:
                                    # Do not retain the first module's fixed
                                    # default for a selected sweepable field
                                    # when the decoded arm supplied no value.
                                    stage_config.pop(key, None)
                                continue
                            stage_config[key] = deepcopy(
                                value[0]
                                if isinstance(value, list) and len(value) == 1
                                else value
                            )

        enriched: List[dict] = []
        for cfg in configs:
            merged = deepcopy(base_decoded)
            _deep_merge(merged, cfg)
            _bind_selected_component_defaults(merged, cfg)
            enriched.append(merged)
        return enriched

    # ------------------------------------------------------------------
    # Config resolution helpers — turn (base YAML + optimizer decoded
    # overrides) into a fully-resolved pipeline config for evaluators.
    # ------------------------------------------------------------------

    def resolve_pipeline_block(
        self, base_config: dict, decoded_config: dict,
    ) -> Tuple[Dict[str, Any], List[dict]]:
        """Select the chosen method's pipeline block + resolve its overrides.

        Returns ``(pipeline_runtime, node_lines)`` for the chosen method,
        ready to drop into the flat config that downstream evaluators
        consume. ``pipeline_runtime`` carries ``{mode, <method-scalar-knobs>}``;
        ``node_lines`` is the deep-copied + overridden + scalarified block.

        Inputs:
          * ``base_config["algo_search_space"]["pipeline"]``: the parsed YAML
            (``{node_lines}`` or ``{methods: [...]}``), normalized by
            ``pipeline_method_blocks`` to a list of ``{mode, node_lines, ...}``.
          * ``decoded_config["pipeline"]``: the optimizer's overrides, shape
            ``{mode: <method>, <method>: {<stage>: {...}, <scalar_knob>: ...}}``.

        Raises if the chosen method has no matching block in the YAML — this
        should never happen for an opt-suggested config (Ax can only sample
        from ``pipeline__mode``'s declared values).
        """
        decoded_pipeline = decoded_config.get("pipeline", {}) or {}
        pipeline_blocks = pipeline_method_blocks(base_config)
        if not isinstance(pipeline_blocks, list):
            raise ValueError("pipeline: must be a list of method blocks.")
        if not pipeline_blocks:
            raise ValueError("pipeline: list is empty; nothing to resolve.")

        # Mode: the optimizer's choice, or the single block's mode when the
        # list has length 1 (no mode parameter emitted in that degenerate case).
        if "mode" in decoded_pipeline:
            mode = str(decoded_pipeline["mode"])
        else:
            mode = str(pipeline_blocks[0]["mode"])

        # Locate the matching block in the YAML.
        chosen_block = next(
            (b for b in pipeline_blocks if str(b.get("mode")) == mode), None,
        )
        if chosen_block is None:
            raise ValueError(
                f"pipeline.mode={mode!r} has no matching block in YAML. "
                f"Available modes: {[b.get('mode') for b in pipeline_blocks]!r}"
            )

        # Per-method optimizer overrides live under decoded_pipeline[mode].
        method_overrides = decoded_pipeline.get(mode, {}) or {}

        # Co-mode: resolve rag_dataflow FIRST — its YAML entries declare each
        # branch's SUB-CONFIG ({react: {max_iter: [...]}}); only the CHOSEN
        # branch's knobs survive into pipeline_runtime (flat, for the
        # consumers), the other branches' knobs are dropped entirely. The
        # decoded override shape mirrors the hierarchy:
        # decoded["pipeline"][method][<branch>][<knob>].
        dataflow: Optional[str] = None
        branch_knobs: Dict[str, dict] = {}
        if "rag_dataflow" in chosen_block:
            dataflow_names, branch_knobs = parse_dataflow_spec(
                chosen_block["rag_dataflow"]
            )
            if "rag_dataflow" in method_overrides:
                dataflow = str(method_overrides["rag_dataflow"])
            elif len(dataflow_names) == 1:
                dataflow = dataflow_names[0]
            else:
                raise ValueError(
                    "pipeline.rag_dataflow is swept but the optimizer supplied "
                    "no choice — missing override."
                )
            if dataflow not in branch_knobs:
                raise ValueError(
                    f"rag_dataflow={dataflow!r} has no matching branch in the "
                    f"YAML spec. Available: {dataflow_names!r}"
                )

        # 1. Build pipeline_runtime: mode + the chosen dataflow branch's
        #    sub-knobs + method-specific scalar/swept knobs.
        runtime: Dict[str, Any] = {"mode": mode}
        if dataflow is not None:
            runtime["rag_dataflow"] = dataflow
            branch_over = method_overrides.get(dataflow) or {}
            for knob, kspec in (branch_knobs.get(dataflow) or {}).items():
                if knob in branch_over:
                    runtime[knob] = branch_over[knob]
                elif not is_sweepable(kspec):
                    runtime[knob] = (
                        kspec[0]
                        if isinstance(kspec, list) and len(kspec) == 1
                        else kspec
                    )
                else:
                    # Swept branch knob without a decoded value — a missing
                    # optimizer override; scalarify raises loudly.
                    runtime[knob] = scalarify_value(kspec)
        for key, value in chosen_block.items():
            if key in ("mode", "node_lines", "rag_dataflow"):
                continue
            # Optimizer-overridden value wins; otherwise fall back to the
            # YAML scalar (or single-element list) value.
            if key in method_overrides:
                runtime[key] = method_overrides[key]
            elif not is_sweepable(value):
                runtime[key] = value[0] if isinstance(value, list) and len(value) == 1 else value
            else:
                # Sweepable knob present in YAML but no decoded value — that
                # would mean a missing optimizer override; scalarify will raise.
                runtime[key] = scalarify_value(value)

        # 2. Resolve node_lines (deep-copy + apply node/module overrides + scalarify).
        node_lines = deepcopy(chosen_block.get("node_lines") or [])
        if dataflow == "react":
            # react drives ONLY retrieval (+ optional reranker) + generator.
            # Strip every other stage BEFORE override application so (a) the
            # cost model never provisions a stage the loop never calls and
            # (b) gated-off (undecoded) sweep lists on dead stages can't trip
            # scalarify.
            for node_line in node_lines:
                node_line["nodes"] = [
                    n for n in node_line.get("nodes", [])
                    if n.get("stage") in REACT_ACTIVE_STAGES
                ]
            node_lines = [nl for nl in node_lines if nl.get("nodes")]
        self._apply_node_lines_overrides(node_lines, method_overrides)

        return runtime, node_lines

    def resolve_vectordbs(
        self, config: dict, decoded_config: dict,
    ) -> None:
        """Materialize the RUNTIME vectordb section (``config["vectordb"]``).

        Input blocks live in the ALGO search space
        (``algo_search_space.vectordb``); this resolves overrides + scalarifies
        them and emits the result at the runtime top level — the same
        input→runtime lift corpus/pipeline get. Blocks not referenced by any
        resolved retrieval module are DROPPED: that is the runtime truth (only
        the module-named store is ever built/queried/priced), and it keeps
        scalarify away from the unchosen family's sweep lists. Must run AFTER
        ``resolve_pipeline_block`` (reads the resolved ``config["node_lines"]``
        for the reference set); both callers do.
        """
        blocks = deepcopy(algo_vectordb_blocks(config))
        referenced = _referenced_vdb_names(config)
        if referenced:
            blocks = [b for b in blocks if str(b.get("name", "")) in referenced]
        for vdb in blocks:
            vdb_name = vdb.get("name", "")
            overrides = decoded_config.get(vdb_name, {})
            # Capture the nested ``index_type: {pq:{...}, flat:{...}}`` spec
            # BEFORE overrides clobber it, so fixed (non-swept) sub-keys of the
            # chosen type can be folded onto the vectordb afterwards.
            it_spec = vdb.get("index_type")
            if overrides:
                vdb.update(overrides)
            # Resolve index_type to a scalar. Swept (≥2 types) → the parent
            # ChoiceParameter supplied the chosen scalar via overrides. Pinned
            # nested dict (1 type, no parent param) → collapse to its key.
            it = vdb.get("index_type")
            if isinstance(it, dict):
                it = next(iter(it)) if it else "pq"
            elif isinstance(it, list):
                it = scalarify_value(it)
            if it is not None:
                vdb["index_type"] = it
            # Fold the chosen index type's FIXED sub-keys up to the vectordb
            # level (swept children already arrived via overrides → present in
            # vdb; setdefault won't clobber them).
            if isinstance(it_spec, dict):
                for k, v in (it_spec.get(it) or {}).items():
                    vdb.setdefault(k, v)
            # IVF-Flat: PQ-only keys (M/nbits/...) are inactive. The optimizer
            # drops them in decode_to_config, but the base YAML still carries
            # them as sweep specs — drop them here too, else scalarify chokes on
            # the unresolved list and EVERY flat config becomes INVALID.
            if vdb.get("index_type") == "flat":
                for k in self._PQ_ONLY_VDB_KEYS:
                    vdb.pop(k, None)
            for k in list(vdb.keys()):
                vdb[k] = scalarify_value(vdb[k])
            # Derive embedding_dim from the chosen embedding_model (single
            # source of truth: EMBEDDING_MODEL_DIMS). Fixed-dim local models own
            # their dim → it never needs hand-setting in YAML and can't drift,
            # which is what lets embedding_model be a search dim. Configurable-
            # dim models (openai*/mock) aren't in the map → any explicit
            # embedding_dim is kept. M-divisibility + explicit/derived conflicts
            # are validated up front in config_validator.
            self._derive_embedding_dim(vdb)
            # Derive PQ M from a target sub-vector dim ``dsub`` (= D/M), once D
            # is known. Must run AFTER _derive_embedding_dim.
            self._derive_pq_dsub(vdb)
        config["vectordb"] = blocks

    @staticmethod
    def _derive_embedding_dim(vdb: dict) -> None:
        """Set ``vdb['embedding_dim']`` from ``embedding_model`` for fixed-dim
        local models. Derive-authoritative: the model owns its dim, so this
        overrides any explicit value (a genuine YAML conflict is rejected
        pre-flight by config_validator). No-op for configurable-dim / unknown
        models — they keep their explicit ``embedding_dim``."""
        from rag_stack_evaluator.static_rag_evaluator.embedding.base import (
            EMBEDDING_MODEL_DIMS,
        )
        model = vdb.get("embedding_model")
        if isinstance(model, str):
            derived = EMBEDDING_MODEL_DIMS.get(model)
            if derived is not None:
                vdb["embedding_dim"] = derived

    @staticmethod
    def _derive_pq_dsub(vdb: dict) -> None:
        """Derive PQ ``M`` from a target sub-vector dim ``dsub`` (= D/M).

        ``dsub`` is the D-normalized quality knob (FAISS sweet spot dsub≈4);
        ``M = nearest divisor of D to D/dsub, capped at 64``. This lets ``dsub``
        be a FLAT search dim that stays valid (M | D, M≤64) and sweet-centered
        across embedding models of different D — so meaningless huge-dsub
        configs are unexpressable. Same philosophy as ``resolve_nlist_factor``
        (search a normalized factor, derive the concrete param). No-op when
        ``dsub`` is absent (raw ``M`` configs untouched — backward compatible);
        also no-op if ``embedding_dim`` is unknown (validator flags that)."""
        if "dsub" not in vdb:
            return
        D = vdb.get("embedding_dim")
        if D is None:
            return  # no dim to derive from — leave dsub for the validator
        vdb["M"] = _nearest_pq_m(int(D), float(vdb.pop("dsub")))

    @staticmethod
    def _apply_node_lines_overrides(
        node_lines: List[dict], method_overrides: Dict[str, Any],
    ) -> None:
        """Apply optimizer node/module overrides + scalarify, in place.

        Same logic as the previous top-level ``resolve_vectordb_and_nodes``,
        but parameterized over an explicit node_lines list (the chosen
        method's) and the per-method overrides extracted from
        ``decoded["pipeline"][<method>]``.
        """
        for node_line in node_lines:
            nodes_to_remove = []
            for node in node_line.get("nodes", []):
                stage = node["stage"]
                overrides = method_overrides.get(stage, {}) or {}
                if not isinstance(overrides, dict):
                    overrides = {}
                modules = node.get("modules", [])
                is_optional = node.get("optional", False)

                # 0. Skip disabled optional nodes.
                if is_optional and overrides.get("enabled") != "true":
                    nodes_to_remove.append(node)
                    continue

                overrides = {k: v for k, v in overrides.items() if k != "enabled"}

                # 1. Select module.
                if "component" in overrides:
                    chosen = overrides["component"]
                    selected = next(
                        (deepcopy(m) for m in modules if m["component"] == chosen),
                        {"component": chosen},
                    )
                    node["modules"] = [selected]
                elif len(modules) > 1:
                    node["modules"] = [modules[0]]

                # 1.5 Dict-form vectordb family selector: fold the CHOSEN
                #     family's template knobs onto the module BEFORE overrides
                #     apply. The selector override then replaces the dict with
                #     the family name; without this fold, non-swept nested
                #     knobs (scalars / single-element lists) would vanish with
                #     the replaced dict. Decoded overrides for nested knobs
                #     already arrive flat at stage level (the pipeline
                #     3+-segment decode collapse drops the family label), so
                #     step 2 lands them on the module as usual — the fold only
                #     supplies defaults (overrides win via the `not in
                #     overrides` guard).
                mod0 = node["modules"][0] if node.get("modules") else None
                sel_spec = mod0.get("vectordb") if isinstance(mod0, dict) else None
                if isinstance(sel_spec, dict):
                    if "vectordb" in overrides:
                        family = str(overrides["vectordb"])
                    elif len(sel_spec) == 1:
                        family = str(next(iter(sel_spec)))
                    else:
                        raise ValueError(
                            f"[{stage}] vectordb family selector is swept "
                            f"({list(sel_spec)}) but the optimizer supplied "
                            "no choice — missing override."
                        )
                    if family not in {str(k) for k in sel_spec}:
                        raise ValueError(
                            f"[{stage}] vectordb={family!r} has no declared "
                            f"family in the module selector ({list(sel_spec)})."
                        )
                    for knob, spec in (sel_spec.get(family) or {}).items():
                        if knob not in overrides:
                            mod0[knob] = spec
                    mod0["vectordb"] = family

                # 2. Apply parameter overrides.
                if overrides:
                    mod = node["modules"][0] if node.get("modules") else None
                    for k, v in overrides.items():
                        if k == "component":
                            continue
                        if mod is not None and k in mod:
                            mod[k] = v
                        elif k in node:
                            node[k] = v
                        elif mod is not None:
                            mod[k] = v

                # 3. Scalarify all params uniformly.
                for mod in node.get("modules", []):
                    for k in list(mod.keys()):
                        mod[k] = scalarify_value(mod[k])
                skip_keys = {"stage", "strategy", "modules", "optional"}
                for k in list(node.keys()):
                    if k not in skip_keys:
                        node[k] = scalarify_value(node[k])

            for node in nodes_to_remove:
                node_line["nodes"].remove(node)

        # Strip 'optional' key from remaining nodes (not a pipeline param).
        for node_line in node_lines:
            for node in node_line.get("nodes", []):
                node.pop("optional", None)

    # ------------------------------------------------------------------
    # eval metrics → project-level composite-objective metadata used by
    # expanded result columns.
    # ------------------------------------------------------------------

    def derive_composite_objectives(
        self, gt_evaluation: Optional[dict],
    ) -> dict:
        """Build ``composite_objectives.quality.sub_metrics`` from
        ``gt_evaluation.metrics``.

        Each metric entry may carry a ``description`` meta-field (free-form
        text for logging / decoupled-GP tooltips) alongside the metric name.

        Returns ``{}`` when no metric carries a description.
        """
        if not gt_evaluation:
            return {}
        metrics = gt_evaluation.get("metrics") or []
        sub_metrics: dict = {}
        any_meta = False
        for m in metrics:
            if not isinstance(m, dict):
                continue
            name = m.get("metric_name")
            if not name:
                continue
            if "description" in m:
                any_meta = True
            sub_metrics[name] = {"description": m.get("description", "")}
        if not any_meta:
            return {}
        return {"quality": {"sub_metrics": sub_metrics}}


# ---------------------------------------------------------------------------
# Unified parameter factory
# ---------------------------------------------------------------------------

def _make_param(name: str, value):
    """Create an Ax parameter from a config value.

    Returns None if the value is not optimizable (scalar, single-element list, etc.).

    Supported formats:
      [v1, v2, ...]        → ChoiceParameter (numeric ordered, or string unordered)
      {ordered: [v1, ...]} → ChoiceParameter with ``is_ordered=True`` in
                             DECLARATION order — for knobs whose YAML value
                             order IS the semantic order but string sort
                             would scramble it (model-size ladders:
                             1.5B < 14B < 3B < 7B lexicographically). Every
                             optimizer reads the flag: ssops numeric-kernel
                             treatment + index spacing, SMAC Ordinal (≥3
                             values), Ax ordinal transforms, greedy endpoint-
                             first probing. Numeric lists are auto-ordered
                             already; wrapping one here just pins the
                             declaration order instead of sorting.
      {range: [lo, hi]}    → RangeParameter  (INT if both int, else FLOAT)
      anything else         → None
    """
    # {ordered: [v1, v2, ...]} → ordered ChoiceParameter, declaration order
    if isinstance(value, dict) and "ordered" in value:
        vals = value["ordered"]
        if not isinstance(vals, list) or len(vals) < 2:
            raise ValueError(
                f"{name}: {{ordered: [...]}} needs a list of >= 2 values, "
                f"got {vals!r}"
            )
        if all(
            isinstance(v, (int, float)) and not isinstance(v, bool)
            for v in vals
        ):
            # Ax requires sort_values=True for numeric ordered choices —
            # numeric semantic order IS magnitude order, so wrapping a
            # numeric list here is redundant-but-harmless (≡ plain list).
            is_int = all(isinstance(v, int) for v in vals)
            return ChoiceParameter(
                name=name,
                parameter_type=ParameterType.INT if is_int else ParameterType.FLOAT,  # type: ignore[arg-type]
                values=sorted(vals),
                is_ordered=True,
                sort_values=True,
            )
        # strings: DECLARATION order is the whole point
        return ChoiceParameter(
            name=name,
            parameter_type=ParameterType.STRING,  # type: ignore[arg-type]
            values=[str(v) for v in vals],
            is_ordered=True,
            sort_values=False,
        )

    # {range: [lo, hi]} → RangeParameter
    if isinstance(value, dict) and "range" in value:
        bounds = value["range"]
        if not isinstance(bounds, list) or len(bounds) != 2:
            return None
        lo, hi = bounds
        if lo == hi:
            return None
        is_int = isinstance(lo, int) and isinstance(hi, int)
        return RangeParameter(
            name=name,
            parameter_type=ParameterType.INT if is_int else ParameterType.FLOAT,  # type: ignore[arg-type]
            lower=min(lo, hi),
            upper=max(lo, hi),
        )

    # [v1, v2, ...] → ChoiceParameter
    if isinstance(value, list) and len(value) > 1:
        if all(isinstance(v, (int, float)) for v in value):
            is_int = all(isinstance(v, int) for v in value)
            return ChoiceParameter(
                name=name,
                parameter_type=ParameterType.INT if is_int else ParameterType.FLOAT,  # type: ignore[arg-type]
                values=sorted(value),
                is_ordered=True,
                sort_values=True,
            )
        else:
            return ChoiceParameter(
                name=name,
                parameter_type=ParameterType.STRING,  # type: ignore[arg-type]
                values=[str(v) for v in value],
                is_ordered=False,
                sort_values=False,
            )

    return None


# ---------------------------------------------------------------------------
# YAML extractors — module-private helpers consumed by
# ``RagSearchSpace.build_ax_searchspace``.
# ---------------------------------------------------------------------------

def _extract_corpus_params(config: dict) -> list:
    """Extract optimizable parameters from the top-level `corpus` block.

    Schema::

        corpus:
          raw_path: ${PROJECT_DIR}/data/raw_corpus.parquet   # scalar metadata
          chunker:
            component: [recursivecharacter, sentence]      # ChoiceParameter
            chunk_size: {range: [128, 1024], n_grid: 5}      # RangeParameter (INT)
            chunk_overlap: {range: [0, 200]}                 # RangeParameter (INT)

    Naming convention: ``corpus__chunker__<param>``. The
    :meth:`RagSearchSpace.decode_to_config` routing rebuilds these into
    ``config["corpus"]["chunker"][param]`` so the Controller can build a
    deterministic cache key from the chunker tuple.
    """
    corpus = algo_corpus_block(config)
    if not isinstance(corpus, dict) or not corpus:
        return []

    params: list = []
    chunker = corpus.get("chunker")
    if isinstance(chunker, dict):
        prefix = f"corpus{SEP}chunker{SEP}"
        for key, value in chunker.items():
            param = _make_param(f"{prefix}{key}", value)
            if param is not None:
                params.append(param)
    return params


def _extract_vectordb_params(config: dict) -> list:
    """Extract optimizable parameters from the vectordb config section.

    Naming convention: ``vectordb__{vdb_name}__{param}``. A nested
    ``index_type: {pq: {...}, flat: {...}}`` block becomes a hierarchical
    parent ChoiceParameter (``vectordb__{vdb}__index_type``) whose dependents
    are that type's child params (``vectordb__{vdb}__{itype}__{key}``), so Ax
    only activates the chosen index type's quantization knobs per trial.
    """
    params = []
    for vdb in algo_vectordb_blocks(config):
        vdb_name = vdb.get("name", "")
        if not vdb_name:
            continue
        prefix = f"vectordb{SEP}{vdb_name}{SEP}"
        for key, value in vdb.items():
            if key in VECTORDB_SKIP_KEYS:
                continue
            if key == "index_type" and isinstance(value, dict):
                params.extend(_extract_index_type_params(prefix, value))
                continue
            param = _make_param(f"{prefix}{key}", value)
            if param is not None:
                params.append(param)
    return params


def _extract_index_type_params(prefix: str, index_type_block: dict) -> list:
    """Build hierarchical params for a nested ``index_type`` block.

    ``index_type: {pq: {M:[...], nbits:[...]}, flat: {...}}`` produces:
      * child params ``{prefix}{itype}__{key}`` for each *swept* sub-key, and
      * (when ≥2 index types) a parent ``{prefix}index_type`` ChoiceParameter
        with ``dependents={itype: [child names]}`` — Ax deactivates the
        non-chosen type's knobs each trial (mirrors ``pipeline__mode``).

    A single-type block pins the index type (no parent param). Fixed
    (non-swept) sub-keys are folded onto the vectordb at resolve time, not here.
    Children decode to ``config[vdb][key]`` (the generic ``vectordb`` lift drops
    the middle ``{itype}`` segment), so the resolved config is flat.
    """
    out: list = []
    dependents: Dict[str, List[str]] = {}
    for itype, sub in index_type_block.items():
        child_names: List[str] = []
        for key, value in (sub or {}).items():
            p = _make_param(f"{prefix}{itype}{SEP}{key}", value)
            if p is not None:
                out.append(p)
                child_names.append(p.name)
        dependents[itype] = child_names
    if len(index_type_block) >= 2:
        out.append(ChoiceParameter(
            name=f"{prefix}index_type",
            parameter_type=ParameterType.STRING,  # type: ignore[arg-type]
            values=list(index_type_block.keys()),
            is_ordered=False,
            sort_values=False,
            dependents=dependents,
        ))
    return out


def _wire_vdb_family_gating(parameters: list, config: dict) -> None:
    """Extend each dict-form family selector's dependents to the FULL gated set.

    ``_extract_vdb_selector_params`` builds the selector with its own nested
    query-knob children only. This pass appends, per family value, every
    top-level ``vectordb__<family>__*`` param of the referenced block (params
    already gated by an in-block parent — e.g. the ``index_type`` children —
    keep their existing single parent; Ax forbids two parents, and the nested
    chain selector→index_type→pq already deactivates them transitively).

    SYSTEM-section knobs are deliberately NOT gated here even when family-
    specific in effect (``system.retrieval.faiss_ivf_parallel_mode`` is IVF-only
    at runtime): algo-space selectors must not gate system-space dims
    (layering), and ``_resolve_system_retrieval`` scalarifies template sweeps
    for knobs absent from the decoded config — a gated-off swept system knob
    would raise there and invalidate every trial on the other branch. The
    residual cost is one always-active binary dim that HNSW ignores.

    Selectors whose values don't all match declared block names are left
    untouched (not a family selector). Ax ChoiceParameters are immutable, so
    the selector is rebuilt in place.
    """
    blocks = {
        str(b.get("name", "")): b
        for b in algo_vectordb_blocks(config)
        if isinstance(b, dict) and b.get("name")
    }
    if not blocks:
        return
    names = {p.name for p in parameters}
    claimed_by: Dict[str, str] = {}
    already_gated: set = set()
    for p in parameters:
        if isinstance(p, ChoiceParameter) and p.is_hierarchical:
            for children in p.dependents.values():
                already_gated.update(children)
    for i, p in enumerate(parameters):
        if not (
            isinstance(p, ChoiceParameter)
            and p.name.split(SEP)[-1] == "vectordb"
            and p.name != "vectordb"
            and all(str(v) in blocks for v in p.values)
        ):
            continue
        deps: Dict[Any, List[str]] = {
            str(v): list((p.dependents or {}).get(str(v), []))
            if p.is_hierarchical else []
            for v in p.values
        }
        for v in p.values:
            fam = str(v)
            if fam in claimed_by:
                raise ValueError(
                    f"vectordb family '{fam}' is referenced by two selectors "
                    f"({claimed_by[fam]!r} and {p.name!r}) — Ax allows one "
                    "gating parent per param. Multi-selector (hybrid) spaces "
                    "need per-selector vectordb blocks."
                )
            claimed_by[fam] = p.name
            fam_prefix = f"vectordb{SEP}{fam}{SEP}"
            deps[fam].extend(sorted(
                n for n in names
                if n.startswith(fam_prefix) and n not in already_gated
            ))
        if not any(deps.values()):
            continue
        parameters[i] = ChoiceParameter(
            name=p.name,
            parameter_type=ParameterType.STRING,  # type: ignore[arg-type]
            values=[str(v) for v in p.values],
            is_ordered=False,
            sort_values=False,
            dependents=deps,
        )


def _nearest_pq_m(embedding_dim: int, dsub: float, max_m: int = 64) -> int:
    """PQ sub-quantizer count ``M`` for a target sub-vector dim ``dsub`` (= D/M).

    M must divide D (FAISS requirement) and stay ≤ ``max_m`` (FAISS guideline:
    beyond ~64 a scalar quantizer is as accurate and faster). Returns the legal
    divisor of D closest to ``D/dsub``; ties break toward the LARGER M (finer
    sub-vectors → higher recall). Because M is snapped to a divisor ≤64 for
    whatever D is given, a single ``dsub`` grid yields a valid, sweet-centered M
    for every embedding model — the D/M heuristic as a flat, D-aware search dim.
    """
    D = int(embedding_dim)
    target = max(1, round(D / float(dsub)))
    divisors = [m for m in range(1, min(D, int(max_m)) + 1) if D % m == 0]
    return min(divisors, key=lambda m: (abs(m - target), -m))


def _extract_node_level_params(
    node_config: dict, stage: str, prefix: str = "",
) -> list:
    """Extract optimizable node-level parameters.

    ``prefix`` (when supplied) namespaces every emitted parameter — used
    by :func:`_extract_pipeline_mode_params` to scope per-method-block
    params under ``pipeline__<method>__...``.
    """
    params = []
    for key, value in node_config.items():
        if key in NODE_SKIP_KEYS:
            continue
        name = f"{prefix}{stage}{SEP}{key}"
        param = _make_param(name, value)
        if param is not None:
            params.append(param)
    return params


def _extract_module_params(
    mod: dict, stage: str, prefix: str = "",
) -> list:
    """Extract optimizable parameters from a single module config.

    ``prefix`` (when supplied) namespaces every emitted parameter — used
    by :func:`_extract_pipeline_mode_params` to scope per-method-block
    params under ``pipeline__<method>__...``.
    """
    mod_type = mod["component"]
    full_prefix = f"{prefix}{stage}{SEP}{mod_type}{SEP}"
    skip_keys = {"component", "generator_backend"}
    params = []

    for key, value in mod.items():
        if key in skip_keys:
            continue
        if key == "vectordb" and isinstance(value, dict):
            params.extend(_extract_vdb_selector_params(full_prefix, value))
            continue
        if key == "vectordb" and isinstance(value, list) and len(value) > 1:
            raise ValueError(
                f"{full_prefix}vectordb: a multi-family selector must use the "
                "dict form — vectordb: {<family>: {<its query-time knobs>}} — "
                "so per-family knob ownership (nprobe/ef_search/…) is declared "
                "by nesting and compiled into Ax gating. The flat list form "
                "leaves the non-chosen family's knobs as always-active phantom "
                "dims."
            )
        param = _make_param(f"{full_prefix}{key}", value)
        if param is not None:
            params.append(param)
    return params


def _extract_vdb_selector_params(prefix: str, selector_block: dict) -> list:
    """Compile a dict-form vectordb FAMILY selector on a retrieval module.

    ``vectordb: {faiss_ivf_index: {nprobe: [...]}, faiss_hnsw_index:
    {ef_search: [...]}}`` produces (mirroring ``_extract_index_type_params``):

      * child params ``{prefix}{family}__{knob}`` for each swept per-family
        query-time knob — ownership is declared by NESTING, not by a code
        table. Children decode to ``config[pipeline][m][<stage>][<knob>]``
        (the pipeline 3+-segment collapse drops the family label), landing
        exactly where the resolver's family flatten puts the runtime value.
      * (when ≥2 families) the parent ``{prefix}vectordb`` ChoiceParameter
        with ``dependents={family: [child names]}``. The referenced
        ``vectordb__<family>__*`` block params and family-owned system knobs
        are appended to the same dependents by ``_wire_vdb_family_gating``
        once all sections are extracted (their names don't exist yet here).

    A single-family dict pins the family (no selector param); its nested
    knobs are emitted ungated (always active).
    """
    out: list = []
    dependents: Dict[str, List[str]] = {}
    for family, sub in selector_block.items():
        child_names: List[str] = []
        for key, value in (sub or {}).items():
            p = _make_param(f"{prefix}{family}{SEP}{key}", value)
            if p is not None:
                out.append(p)
                child_names.append(p.name)
        dependents[str(family)] = child_names
    if len(selector_block) >= 2:
        out.append(ChoiceParameter(
            name=f"{prefix}vectordb",
            parameter_type=ParameterType.STRING,  # type: ignore[arg-type]
            values=[str(f) for f in selector_block.keys()],
            is_ordered=False,
            sort_values=False,
            dependents=dependents,
        ))
    return out


def _extract_pipeline_mode_params(config: dict) -> list:
    """Build pipeline-tree search-space params from the new schema.

    Strict shape (no fallbacks)::

        pipeline:
          - mode: sequential
            node_lines: [...]
            # optional method-specific scalar/swept knobs go alongside:
            # max_iter_num: [2, 3, 4]
          - mode: corag
            max_iter_num: [2, 3, 4]
            node_lines: [...]
          - mode: flare
            threshold: {range: [0.1, 0.5]}
            look_ahead_steps: [32, 64]
            node_lines: [...]

    Produces:

    * A top-level ``pipeline__mode`` ``ChoiceParameter`` whose values are the
      method names from the list, with ``dependents = {method: [param names
      generated for that method's block]}``. Ax then auto-deactivates the
      losing branches' params for any trial.
    * Per-method-block params (node-level, module-level, method scalar knobs)
      with names prefixed ``pipeline__<method>__...``. The :class:`RagSearchSpace`
      ``_insert_param`` override routes these under
      ``config["pipeline"][<method>][...]`` on decode.
    """
    pipeline_cfg = pipeline_method_blocks(config)
    if not pipeline_cfg:
        return []

    params: list = []
    method_dependents: Dict[str, List[str]] = {}
    method_names: List[str] = []

    for block in pipeline_cfg:
        if not isinstance(block, dict):
            raise ValueError(
                f"pipeline block must be a dict, got {type(block).__name__}."
            )
        if "mode" not in block:
            raise ValueError(f"pipeline block missing 'mode' key: {block!r}")
        method = block["mode"]
        if not isinstance(method, str):
            raise ValueError(
                f"pipeline block 'mode' must be a single string "
                f"(use multiple list entries for multi-method sweeps). "
                f"Got: {method!r}"
            )
        if "node_lines" not in block:
            raise ValueError(
                f"pipeline block for mode '{method}' is missing 'node_lines'."
            )
        if method in method_names:
            raise ValueError(f"pipeline contains duplicate method '{method}'.")
        method_names.append(method)

        block_param_names: List[str] = []
        method_prefix = f"pipeline{SEP}{method}{SEP}"

        # Co-mode gating (rag_dataflow swept over sequential+react on ONE
        # shared node_lines): `rag_dataflow` stays a FLAT knob — the shared
        # stages (retrieval / reranker / generator / corpus / vectordb) keep a
        # single set of dims, no per-mode subtree duplication — and it is the
        # gate PARENT of the mode-specific params. WHICH params hang off each
        # branch is read from the CONFIG: a rag_dataflow entry may carry that
        # branch's SUB-CONFIG ({react: {max_iter: [...]}}), parsed by
        # parse_dataflow_spec and wired into Ax dependents exactly like module
        # `component` branches (child names
        # ``pipeline__<method>__<branch>__<knob>``) — nothing knob-specific is
        # hardcoded here. The `sequential` branch additionally gates the
        # sequential-only stages' head params (engine capability: react
        # drives only REACT_ACTIVE_STAGES). Sobol / the GP / the acqf then
        # never spend dimensions on dead knobs.
        dataflow_names: List[str] = []
        branch_children: Dict[str, List[str]] = {}
        seq_only_names: List[str] = []

        # 1. Method-specific scalar/swept knobs (top-level keys other than mode/node_lines).
        for key, value in block.items():
            if key in ("mode", "node_lines"):
                continue
            if key == "rag_dataflow":
                dataflow_names, branch_knobs = parse_dataflow_spec(value)
                for branch, knobs in branch_knobs.items():
                    children: List[str] = []
                    for knob, kspec in (knobs or {}).items():
                        kparam = _make_param(
                            f"{method_prefix}{branch}{SEP}{knob}", kspec,
                        )
                        if kparam is None:
                            continue  # fixed value — the resolver folds it in
                        params.append(kparam)
                        block_param_names.append(kparam.name)
                        children.append(kparam.name)
                    branch_children[branch] = children
                continue
            param = _make_param(f"{method_prefix}{key}", value)
            if param is None:
                continue
            params.append(param)
            block_param_names.append(param.name)

        # Both modes present → rag_dataflow is a search dim and the gate parent.
        comode_gating = {"sequential", "react"} <= set(dataflow_names)

        # 2. Per-node + per-module params under this block's node_lines.
        for node_line in block.get("node_lines", []) or []:
            for node_config in node_line.get("nodes", []) or []:
                stage = node_config["stage"]
                modules = node_config.get("modules", []) or []
                is_optional = node_config.get("optional", False)

                # Node-level params (e.g. top_k).
                node_params = _extract_node_level_params(
                    node_config, stage, prefix=method_prefix,
                )

                # Module choice + per-module params.
                module_type_param = None
                all_mod_params: list = []
                if len(modules) <= 1:
                    if modules:
                        all_mod_params = _extract_module_params(
                            modules[0], stage, prefix=method_prefix,
                        )
                else:
                    components = [m["component"] for m in modules]
                    dependents: Dict[Any, List[str]] = {}
                    for mod in modules:
                        mod_type = mod["component"]
                        mod_params = _extract_module_params(
                            mod, stage, prefix=method_prefix,
                        )
                        dependents[mod_type] = [p.name for p in mod_params]
                        all_mod_params.extend(mod_params)

                    module_type_param = ChoiceParameter(
                        name=f"{method_prefix}{stage}{SEP}component",
                        parameter_type=ParameterType.STRING,  # type: ignore[arg-type]
                        values=components,
                        is_ordered=False,
                        sort_values=False,
                        dependents=dependents,
                    )

                stage_root_names: List[str] = []
                if is_optional:
                    direct_children: List[str] = [p.name for p in node_params]
                    if module_type_param is not None:
                        direct_children.append(module_type_param.name)
                    else:
                        direct_children.extend(p.name for p in all_mod_params)

                    enabled_param = ChoiceParameter(
                        name=f"{method_prefix}{stage}{SEP}enabled",
                        parameter_type=ParameterType.STRING,  # type: ignore[arg-type]
                        values=["true", "false"],
                        is_ordered=False,
                        sort_values=False,
                        dependents={"true": direct_children, "false": []},
                    )
                    params.append(enabled_param)
                    block_param_names.append(enabled_param.name)
                    stage_root_names.append(enabled_param.name)

                params.extend(node_params)
                block_param_names.extend(p.name for p in node_params)
                if module_type_param is not None:
                    params.append(module_type_param)
                    block_param_names.append(module_type_param.name)
                params.extend(all_mod_params)
                block_param_names.extend(p.name for p in all_mod_params)
                if not is_optional:
                    stage_root_names.extend(p.name for p in node_params)
                    if module_type_param is not None:
                        stage_root_names.append(module_type_param.name)
                    else:
                        stage_root_names.extend(p.name for p in all_mod_params)

                # Sequential-only stage in a co-mode block → its tree roots
                # hang off rag_dataflow's `sequential` branch.
                if comode_gating and stage not in REACT_ACTIVE_STAGES:
                    seq_only_names.extend(stage_root_names)

        # Emit the rag_dataflow choice (only a search dim with 2+ branches),
        # with dependents = each branch's YAML-declared sub-knobs, plus the
        # engine-side sequential-only stage heads on the `sequential` branch.
        if len(dataflow_names) >= 2:
            dependents: Dict[Any, List[str]] = {
                b: list(branch_children.get(b, [])) for b in dataflow_names
            }
            if comode_gating:
                dependents["sequential"] = (
                    dependents.get("sequential", []) + seq_only_names
                )
            dataflow_param = ChoiceParameter(
                name=f"{method_prefix}rag_dataflow",
                parameter_type=ParameterType.STRING,  # type: ignore[arg-type]
                values=dataflow_names,
                is_ordered=False,
                sort_values=False,
                dependents=dependents if any(dependents.values()) else None,
            )
            params.append(dataflow_param)
            block_param_names.append(dataflow_param.name)

        method_dependents[method] = block_param_names

    # 3. Top-level pipeline__mode ChoiceParameter — only when 2+ methods.
    # Ax rejects single-value ChoiceParameters; when the YAML lists exactly
    # one method block, the method choice is not a search dimension, so we
    # emit the children flat (they remain always-active without a mode
    # parent). The resolver falls back to ``pipeline[0].mode`` in that case.
    if len(method_names) >= 2:
        mode_param = ChoiceParameter(
            name=f"pipeline{SEP}mode",
            parameter_type=ParameterType.STRING,  # type: ignore[arg-type]
            values=method_names,
            is_ordered=False,
            sort_values=False,
            dependents=method_dependents,
        )
        params.insert(0, mode_param)

    return params
