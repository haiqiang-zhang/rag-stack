"""Unified context for the performance objective (cost_model + measured).

Performance comes from one of two sources, selected by
``system.performance_source``:

* ``cost_model`` (default): analytical RAGO/GenZ simulation.
* ``measured``: real on-GPU vLLM deployment, timed.

This module owns everything performance-related that is not an evaluator
itself:

* the performance-source switch (:func:`resolve_performance_source`),
* the rag_ir mode policy — ``static`` vs ``dynamic`` CM engine selection
  (:func:`resolve_rag_ir_mode` / :func:`stamp_rag_ir_mode`),
* :class:`PerformanceContext` — the model-free config-load → search-space →
  corpus-binding → cost-model setup shared by every performance caller, plus
  the measured-path GPU-layout derivation (:meth:`resolve_system_config`)
  and the :meth:`make_measured_provider` factory.

Extracted from :class:`rag_stack.controller.Controller` so that lightweight
callers — chiefly the historical-surrogate benchmark
(``rag_stack.optimizer.benchmark.synthetic_functions.rag_stack_historical_surrogate``)
— can turn a decoded config into a *performance* score WITHOUT paying for the
full ``Controller.__init__``. The full constructor also builds the optimizer
and the quality evaluator (``_make_quality_evaluator`` → loads embedding /
reranker / generation models), none of which the performance path needs.

``Controller`` *composes* this class and aliases its attributes back onto
itself, so the live optimizer path and any lightweight caller share the EXACT
same setup and the same ``resolve_config`` logic. That makes the surrogate's
"cost-model behaviour matches the live run exactly" guarantee true *by
construction* — the two paths cannot drift because they run the same code.

The only ML-adjacent work here is the ``TokenStats`` tokenizer load inside
:class:`~rag_stack.dataset_manager.DatasetManager` (a tokenizer,
not weighted models), which the cost model genuinely needs for token statistics.

Module-level imports are deliberately LIGHT: the policy helpers above the
class are consumed by ``ConfigValidator`` and the static evaluator's
``trace_builder`` at import time, and must not drag in the heavy
search-space / cost-model stack. All heavy imports happen inside
``PerformanceContext.__init__``.
"""
from __future__ import annotations

import importlib
import logging
import os
from copy import deepcopy
from typing import Any, Optional

logger = logging.getLogger("RAG-Stack")

# ----------------------------------------------------------------------
# Performance source (system.performance_source)
# ----------------------------------------------------------------------

PERFORMANCE_SOURCE_KEY = "performance_source"
PERFORMANCE_SOURCE_COST_MODEL = "cost_model"
PERFORMANCE_SOURCE_MEASURED = "measured"


def resolve_performance_source(config: dict) -> str:
    """Read ``system.performance_source`` (default ``cost_model``).

    Value checking lives in ``ConfigValidator._check_performance_source``;
    this helper only normalizes, so every reader parses the key identically.
    """
    return str(
        (config.get("system") or {}).get(
            PERFORMANCE_SOURCE_KEY, PERFORMANCE_SOURCE_COST_MODEL
        )
    ).lower()


# ----------------------------------------------------------------------
# System-space owner (system.system_space_owner)
#
# ``performance_source`` says WHERE performance numbers come from (analytical
# cost model vs real timed deployment). ``system_space_owner`` says WHO
# searches the system/deployment design space:
#
# * ``cost_model``: no optimizer-facing deployment dims — the CM sweeps every
#   deployment internally per trial and selects the best one (the decoupled
#   search).
# * ``optimizer``: the optimizer samples deployment dims declared in
#   ``system.system_design_space`` (``deployment_preset`` / ``gpu_layout`` /
#   batch knobs); exactly ONE deployment is priced (CM fixed-placement) or
#   launched (measured) per trial.
#
# Legal (owner, performance_source) combinations:
#   (cost_model, cost_model)  — decoupled CM search (default cost_model mode)
#   (optimizer,  measured)    — classic measured mode
#   (optimizer,  cost_model)  — joint-space search priced by the CM
#   (cost_model, measured)    — ILLEGAL (cannot measure a whole sweep);
#                               rejected by ConfigValidator.
# ----------------------------------------------------------------------

SYSTEM_SPACE_OWNER_KEY = "system_space_owner"
SYSTEM_SPACE_OWNER_COST_MODEL = "cost_model"
SYSTEM_SPACE_OWNER_OPTIMIZER = "optimizer"
SYSTEM_SPACE_OWNER_VALID_VALUES = (
    SYSTEM_SPACE_OWNER_COST_MODEL, SYSTEM_SPACE_OWNER_OPTIMIZER,
)


def resolve_system_space_owner(config: dict) -> str:
    """Read ``system.system_space_owner``; derive from performance_source when absent.

    The derivation (``cost_model`` → owner ``cost_model``, ``measured`` →
    owner ``optimizer``) reproduces the pre-split behaviour exactly, so
    configs written before the key existed keep working unchanged. Value
    checking lives in ``ConfigValidator._check_system_space_owner``.
    """
    raw = (config.get("system") or {}).get(SYSTEM_SPACE_OWNER_KEY)
    if raw is not None:
        return str(raw).lower()
    if resolve_performance_source(config) == PERFORMANCE_SOURCE_MEASURED:
        return SYSTEM_SPACE_OWNER_OPTIMIZER
    return SYSTEM_SPACE_OWNER_COST_MODEL


# ----------------------------------------------------------------------
# System design space (system.system_design_space)
# ----------------------------------------------------------------------

SYSTEM_DESIGN_SPACE_KEY = "system_design_space"
# Deprecated spelling, kept as a read alias so historical configs/projects
# stay loadable. New configs must use ``system_design_space``.
SYSTEM_DESIGN_SPACE_LEGACY_KEY = "measured_search_space"

_warned_legacy_design_space = False


def system_design_space(system: dict) -> dict:
    """The optimizer-facing system dims block from a ``config["system"]`` dict.

    Reads ``system_design_space``, falling back to the deprecated
    ``measured_search_space`` spelling (one warning per process). Both keys
    present is a config error (ConfigValidator rejects it); this helper then
    prefers the new key. Returns ``{}`` when neither is set.
    """
    global _warned_legacy_design_space
    system = system or {}
    block = system.get(SYSTEM_DESIGN_SPACE_KEY)
    if isinstance(block, dict):
        return block
    legacy = system.get(SYSTEM_DESIGN_SPACE_LEGACY_KEY)
    if isinstance(legacy, dict):
        if not _warned_legacy_design_space:
            _warned_legacy_design_space = True
            logger.warning(
                "system.measured_search_space is deprecated — rename the key "
                "to system.system_design_space (same contents)."
            )
        return legacy
    return {}


# ----------------------------------------------------------------------
# rag_ir mode policy (global.rag_ir_mode) — which CM ENGINE prices a trial:
#   * ``dynamic`` (default): the quality run's recorded trace + workflow
#     schema replay through the dynamic engine (``evaluate_dynamic``) — real
#     per-call token sizes, sequential and react alike.
#   * ``static``: offline/static TokenStats and config-driven aggregate RAGO
#     inputs through the static engine (``evaluate_static``) — no trace needed.
# ----------------------------------------------------------------------

RAG_IR_MODE_KEY = "rag_ir_mode"
RAG_IR_MODE_STATIC = "static"
RAG_IR_MODE_DYNAMIC = "dynamic"
RAG_IR_MODE_VALID_VALUES = (RAG_IR_MODE_STATIC, RAG_IR_MODE_DYNAMIC)


def resolve_rag_ir_mode(config: dict) -> str:
    """Resolve ``global.rag_ir_mode`` to ``static`` or ``dynamic``.

    Default (key omitted) is ``dynamic``.
    """
    glob = config.get("global", {}) or {}
    if "cm_data_source" in glob:
        raise ValueError(
            "[global.cm_data_source] renamed to `rag_ir_mode` "
            "(values: 'dynamic' — was 'rag_ir' — / 'static'). "
            "Update the config."
        )
    raw = glob.get(RAG_IR_MODE_KEY, RAG_IR_MODE_DYNAMIC)
    mode = str(raw).lower()
    if mode not in RAG_IR_MODE_VALID_VALUES:
        raise ValueError(
            f"[global.rag_ir_mode] {raw!r} is not one of "
            "'static' / 'dynamic' (default: dynamic)."
        )
    return mode


def stamp_rag_ir_mode(root_config: dict, pipeline_config: dict) -> None:
    """Write the resolved rag_ir mode into a per-eval pipeline config."""
    runtime = pipeline_config.setdefault("_runtime", {})
    runtime[RAG_IR_MODE_KEY] = resolve_rag_ir_mode(root_config)


# ----------------------------------------------------------------------
# Deployment constants (measured path)
# ----------------------------------------------------------------------

# measured_search_space knobs that are now DERIVED from the partition or fixed
# deployment constants — NOT copied verbatim from the sampled config. Mirrors
# ``rag_search_space._DERIVED_KNOBS``.
_DERIVED_SYSTEM_KNOBS = frozenset({
    "vllm_gpu_memory_utilization", "vllm_max_num_batched_tokens",
    "tensor_parallel_vllm",
    "pipeline_parallel_vllm", "serving_mode",
    "pipeline_parallel_generator", "gpu_layout",
})


class PerformanceContext:
    """Everything required to evaluate the performance objective.

    Builds — model-free — the validated ``config`` (with corpus size ``N`` and
    token stats injected), the :class:`RagSearchSpace` adapter, the resume-safe
    :class:`DatasetManager`, the base decoded config, the hardware spec, and
    the :class:`StaticAssembly` cost model. Exposes :meth:`resolve_config` so a
    caller can do ``ctx.assembly.evaluate_static(ctx.resolve_config(decoded))``.

    Covers BOTH performance sources (``self.performance_source``):

    * ``cost_model`` — :attr:`static_assembly` + :meth:`cm_performance_probe`,
      with token inputs per the CM data-source policy (module helpers above).
    * ``measured`` — :meth:`resolve_system_config` derives the real GPU
      deployment layout; :meth:`make_measured_provider` builds the
      run-spanning measured-timing provider.

    Attribute names deliberately match the historical ``Controller`` attributes
    (``config``, ``_ssops``, ``search_space``, ``_data``, ``_base_decoded_config``,
    ``hardware``, ``static_assembly``) so ``Controller`` can compose this and alias
    them back with zero public-API change.

    Args:
        config_path: path to a rag-stack YAML config.
        project_dir: project root. Set as ``PROJECT_DIR`` before config load so
            ``${PROJECT_DIR}`` interpolation works; also the resume-safe data
            root for :class:`DatasetManager` (if ``<project_dir>/data/`` exists,
            the supplied ``qa_data_path`` / ``corpus_data_path`` are ignored).
        qa_data_path / corpus_data_path: source parquets, used only when the
            project has no ``data/`` directory yet.
        dataset: optional in-memory :class:`GeneratedDataset` (takes precedence
            over file paths).
        validate: run :class:`ConfigValidator` (default ``True`` — matches the
            live ``Controller`` behaviour).
        check_model_cache: when ``validate`` is ``True``, also run the HF
            model-cache check, which ``snapshot_download``\\ s any configured
            model not present locally (default ``True`` — matches live
            ``Controller``). Cost-model-only callers that never execute the
            models (e.g. the historical-surrogate benchmark) pass ``False`` to
            keep the cheap structural validation but skip the network round-trip.
        check_performance_support: enforce performance capability checks while
            constructing the search space (default ``True``). Execution preview
            workers may use ``False`` with ``validate=False`` to resolve and run
            a structurally valid pipeline before reporting CM compilation gaps.
            Structural search-space validation always remains enabled.
    """

    def __init__(
        self,
        config_path: str,
        project_dir: str,
        qa_data_path: Optional[str] = None,
        corpus_data_path: Optional[str] = None,
        dataset: Optional[Any] = None,
        *,
        validate: bool = True,
        check_model_cache: bool = True,
        check_performance_support: bool = True,
    ):
        # Heavy imports live here (not at module level) so policy-only
        # importers of this module stay light. Import order mirrors
        # controller.py: pulling in the rag_stack.utils package (and the
        # search space) before utils.config avoids the rag_stack.utils <->
        # static-evaluator circular import that triggers when utils.config is
        # the first rag_stack import.
        importlib.import_module("rag_stack.utils.preprocess")
        from rag_stack.search_space import RagSearchSpace
        from rag_stack.utils.config import load_yaml_config
        from rag_stack.types import SystemConfig
        from rag_stack.cost_model.assembly import RAGCMAssembly

        # PROJECT_DIR must be set before config load so ${PROJECT_DIR} expands.
        self.project_dir = os.path.abspath(project_dir)
        os.environ["PROJECT_DIR"] = self.project_dir

        self.config = load_yaml_config(config_path)
        self.performance_source = resolve_performance_source(self.config)
        self.system_space_owner = resolve_system_space_owner(self.config)

        # Pre-flight config validation (same as the live Controller path).
        # ``check_model_cache=False`` skips the network model-cache check (which
        # may snapshot_download missing models) — cost-model-only callers never
        # run the models, so they keep the cheap structural checks but avoid the
        # network round-trip.
        if validate:
            from rag_stack.config_validator import ConfigValidator
            errors = ConfigValidator(
                self.config, check_model_cache=check_model_cache
            ).validate()
            if errors:
                raise ValueError(
                    f"Config validation failed ({len(errors)} error(s)):\n"
                    + "\n".join(f"  - {e}" for e in errors)
                )

        # Offline HF mode — matches Controller. The only model-ish load below is
        # the TokenStats tokenizer in DatasetManager (no weighted models).
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        # Search-space adapter (CSV decode/encode, enumerate, resolve helpers).
        self._ssops = RagSearchSpace.from_yaml(
            self.config, check_performance_support=check_performance_support,
        )
        self.search_space = self._ssops.ss

        # Resume-safe data binding. DatasetManager stores ``self._config =
        # config`` (the SAME dict object), so the inject_* calls below mutate
        # ``self.config`` in place. Imported here (not at module level) because
        # it pulls the heavy static-evaluator package.
        from rag_stack.dataset_manager import DatasetManager
        self._data = DatasetManager(
            project_dir=self.project_dir,
            config=self.config,
            qa_data_path=qa_data_path,
            corpus_data_path=corpus_data_path,
            dataset=dataset,
        )
        # Auto-inject N into FAISS vectordb configs + stash token stats so the
        # cost model + resolve_config see them on the shared config.
        self._data.inject_corpus_size()
        self._data.inject_token_stats()

        # Baseline decoded config (callers use it as a starting point).
        self._base_decoded_config = self._ssops.build_base_decoded_config(self.config)

        # Hardware spec + cost-model assembly. The ``system`` block already
        # carries the search_space hints injected by inject_token_stats.
        hw_dict = self.config.get("system")
        if hw_dict is None:
            raise ValueError("Config missing 'system' section for hardware specification.")
        self.hardware = SystemConfig.from_dict(hw_dict)
        self.assembly = RAGCMAssembly(self.hardware)

    # ------------------------------------------------------------------

    def make_measured_provider(self, quality_evaluator) -> Optional[Any]:
        """Build the run-spanning measured-timing provider, or ``None``.

        Returns ``None`` under ``performance_source: cost_model`` (the
        default). Under ``measured``, constructs a
        :class:`~rag_stack_evaluator.static_rag_evaluator.measured.provider.MeasuredProvider`
        (owns the ModelCache + vLLM deployment) around the given quality
        evaluator — the evaluator is the controller's, so the import stays
        lazy and cost-model runs never touch the measured package.
        """
        if self.performance_source != PERFORMANCE_SOURCE_MEASURED:
            return None
        from rag_stack_evaluator.static_rag_evaluator.measured.provider import MeasuredProvider
        system = self.config.get("system") or {}
        mss = system_design_space(system)
        available_gpus = mss.get("available_gpus") or ["cuda:0"]
        eval_setting = self.config.get("eval_backend_setting") or {}
        selection = (
            (system.get("performance_objectives") or {}).get("selection")
            or "max_throughput"
        )
        provider = MeasuredProvider(
            quality_evaluator,
            available_gpus,
            n_queries=eval_setting.get("queries_per_trial"),
            selection=selection,
        )
        logger.info(
            f"performance_source=measured: real on-hardware timing on "
            f"{len(available_gpus)} GPU(s), selection={selection}, "
            f"queries_per_trial={eval_setting.get('queries_per_trial')}"
        )
        return provider

    def make_cm_fixed_provider(self) -> Optional[Any]:
        """CM fixed-placement provider for the joint-space cell, or ``None``.

        Non-None exactly when ``(system_space_owner, performance_source) ==
        (optimizer, cost_model)``: the optimizer samples ONE deployment per
        trial and the cost model prices exactly it (no sweep) — the CM
        stand-in for a measured run on hardware we don't hold. Every branch
        the controller gates on this provider short-circuits for existing
        cost_model (cm-swept) and measured configs.
        """
        if self.performance_source != PERFORMANCE_SOURCE_COST_MODEL:
            return None
        if self.system_space_owner != SYSTEM_SPACE_OWNER_OPTIMIZER:
            return None
        from rag_stack.cost_model.fixed_placement_provider import (
            CmFixedPlacementProvider,
        )
        logger.info(
            "system_space_owner=optimizer + performance_source=cost_model: "
            "CM prices ONE sampled deployment per trial (fixed placement, "
            "no internal sweep)."
        )
        return CmFixedPlacementProvider(self.assembly)

    def cm_performance_probe(self, decoded_config: dict) -> float:
        """Static cost-model performance score for a candidate decoded config.

        Config-driven RAGO path only (``dag=None``): this is called at
        SUGGEST time, before any quality run, so no rag_ir trace exists and
        the trace-replay path is unavailable by construction. Used by the
        SLO-gated optimizer (``agent_qnehvi_slo``) to test candidates against
        the run's ``constrained`` selection policy — an SLO-violating config
        scores the ``INFEASIBLE_SCORE`` sentinel — before any GT evaluation
        is spent on it.
        """
        pipeline_config = self.resolve_config(decoded_config)
        return self.assembly.evaluate_static(pipeline_config).performance_score

    def resolve_config(self, decoded_config: dict) -> dict:
        """Apply decoded optimizer params to the base YAML config.

        Coordinator that delegates to the search-space adapter and the
        dataset manager. Under the list-of-method-blocks schema, the chosen
        method's node_lines are pulled out of its block and placed at the top
        level for downstream evaluator code (which still consumes
        ``config["node_lines"]``).

        Verbatim port of the historical ``Controller._resolve_config`` body;
        ``Controller._resolve_config`` now delegates here.
        """
        config = deepcopy(self.config)
        config.pop("optimizer", None)

        # 1. Pick the chosen method's block + resolve its overrides.
        #    Produces pipeline_runtime (mode + method scalar knobs) AND
        #    the method-specific node_lines (already deep-copied + scalarified).
        pipeline_runtime, node_lines = self._ssops.resolve_pipeline_block(
            self.config, decoded_config,
        )
        config["pipeline_runtime"] = pipeline_runtime
        config["node_lines"] = node_lines
        config.pop("pipeline", None)

        # 2. Corpus chunker → config["corpus_runtime"]
        chunker_runtime = self._data.resolve_corpus_chunker(decoded_config)
        if chunker_runtime:
            config["corpus_runtime"] = {"chunker": chunker_runtime}
        config.pop("corpus", None)

        # 3. Vectordb overrides + scalarify (mutates config in place).
        self._ssops.resolve_vectordbs(config, decoded_config)

        # 4. System-level retrieval runtime knobs (system.retrieval:
        #    faiss_num_threads / faiss_ivf_parallel_mode).
        #    Searchable/configurable in BOTH
        #    perf modes; the resolved scalars stay in
        #    config["system"]["retrieval"] where the CM retrieval adapters
        #    and resolve_system_config read them.
        self._resolve_system_retrieval(config, decoded_config)

        return config

    def _faiss_num_threads(self, explicit, batch=None) -> int:
        """THE single faiss_num_threads policy for ALL modes (cost_model + measured).

        Explicit config value (``system.retrieval.faiss_num_threads`` in the YAML/JSON) wins;
        otherwise it is DERIVED = ``min(batch, physical cores)``. This is the ONE place the
        value is decided — the CM (``faiss_ivf`` adapter) and the measured retrieval node both
        only READ the resolved field, never compute it. ``batch=None`` (or a sweep-max batch)
        → the full core pool; FAISS (``parallel_mode=0``) and the IVF CM each additionally cap
        at ``min(batch_row, threads)`` internally, so the effective per-query count stays
        ``min(batch, cores)`` in every mode."""
        if isinstance(explicit, int) and explicit >= 1:
            return int(explicit)
        from rag_stack.cost_model.cpu_specs import CPU_SPECS
        spec = CPU_SPECS.get(str(getattr(self.hardware, "cpu_preset", "") or ""))
        if spec is not None:
            cores = spec.num_cores
        else:
            # Inline/custom CPUs are target hardware, not the machine running
            # the optimizer.  Falling back directly to os.cpu_count() made a
            # remote A100x8 transfer use this controller host's 64 cores instead
            # of the target dual-EPYC specification's 128 physical cores.
            # SystemConfig always carries num_cores; the host fallback remains
            # only for deliberately minimal test/legacy stand-ins.
            target_cores = getattr(self.hardware, "num_cores", None)
            try:
                cores = int(target_cores)
            except (TypeError, ValueError):
                cores = os.cpu_count() or 1
            if cores < 1:
                raise ValueError(
                    "target system CPU num_cores must be a positive integer"
                )
        if batch is None:
            return max(1, int(cores))
        return max(1, min(int(batch), int(cores)))

    def _resolve_system_retrieval(self, config: dict, decoded_config: dict) -> None:
        """Scalarify ``system.retrieval`` in place (decoded values winning) and DERIVE
        ``faiss_num_threads`` via :meth:`_faiss_num_threads` (the single source). The CM has
        no single batch (it sweeps up to ``cm_search_space.max_batch_size_request``), so the
        derived pool = ``min(max_batch, cores)``; the CM caps per sweep row internally."""
        from rag_stack.search_space.rag_search_space import scalarify_value
        system = config.get("system")
        if not isinstance(system, dict):
            return  # no system section (synthetic/benchmark spaces)
        retrieval = system.get("retrieval")
        if not isinstance(retrieval, dict):
            # An absent block must still get the DERIVED faiss_num_threads —
            # this is the ONE place the value is decided (the CM adapters and
            # the measured store only read it, defaulting to 1 when unset,
            # which would silently move the operating point to single-thread).
            retrieval = {}
            system["retrieval"] = retrieval
        decoded_sys = (decoded_config or {}).get("system") or {}
        allowed = {"faiss_num_threads", "faiss_ivf_parallel_mode",
                   "faiss_indexing_thread"}
        for key in list(retrieval):
            if key not in allowed:
                retrieval.pop(key, None)
        for key, tmpl_val in retrieval.items():
            if key in decoded_sys:
                retrieval[key] = decoded_sys[key]
            else:
                retrieval[key] = scalarify_value(tmpl_val)
        max_batch = ((system or {}).get("cm_search_space") or {}).get("max_batch_size_request")
        retrieval["faiss_num_threads"] = self._faiss_num_threads(
            retrieval.get("faiss_num_threads"), max_batch)

    def resolve_system_config(
        self, decoded_config: dict, resolved_config: Optional[dict] = None,
        system_dims_override: Optional[dict] = None,
    ) -> dict:
        """Build the real deployment ``system_config`` for the measured path.

        Companion to :meth:`resolve_config` (which yields the evaluator config):
        turns the sampled system dims into a concrete GPU layout that
        ``VllmDeploymentManager`` / the evaluator realize. Used ONLY when
        ``system.performance_source == "measured"``.

        The layout is **derived, not searched** — exactly mirroring how the cost
        model (RAGO) maps a partition + per-stage chip counts onto physical GPUs:

          1. Present GPU stages (``measured_gpu_stages`` over the *resolved*
             pipeline — only enabled nodes; LLMs split into prefill/decode;
             ``semantic_retrieval`` represents the query-embedding model only).
          2. ``gpu_layout`` (measured search dim) or explicit per-engine
             ``num_gpus_<engine>`` + per-stage ``cut_after_<stage>`` values read
             from ``decoded_config["system"]``.
          3. :func:`derive_layout` partitions the stages, sizes each group to
             ``max(member num_gpus)`` (= RAGO ``np.max``), lays groups on
             consecutive ``available_gpus`` slices, and detects per-LLM
             unified-vs-1P1D (prefill+decode same group → unified; different →
             disaggregated PD pair). ``total_slots_needed`` plus
             ``min_total_slots`` / ``max_total_slots`` let ``prepare_trial``
             reject out-of-range arms as ``TrialInvalid``.
          4. vLLM memory util is intentionally NOT materialized here. Measured
             launch code sizes it from live GPU state at subprocess startup.

        Emits the current resolved deployment schema: runtime knobs at the top
        level, ``retrieval`` for CPU retrieval runtime policy, ``vllm`` for vLLM
        launch limits, and ``layout`` as the only placement source of truth.
        The pipeline node ``semantic_retrieval`` is split into GPU ``encode``
        and CPU ``retrieval`` stages here.
        See :mod:`rag_stack.search_space.placement` for the partition machinery.
        """
        from rag_stack.search_space.placement import (
            measured_gpu_stages, stage_engine, project_layout_to_slot_bounds,
            cut_after_key, num_gpus_key, passage_reranker_components,
        )

        system = self.config.get("system") or {}
        mss = system_design_space(system)
        selected_candidate_layout = None
        if system_dims_override is not None:
            # CM-selected deployment → measured schema. The override acts as
            # BOTH the measured template and the sampled values, so the SAME
            # layout builder below emits the exact schema the measured path
            # consumes — no parallel serializer to drift.
            mss = dict(system_dims_override)
            decoded_config = {"system": dict(system_dims_override)}
            if mss.get("candidate_system_layout") is not None:
                from rag_stack.cost_model.candidate_layout import (
                    candidate_system_layout_from_row,
                )
                selected_candidate_layout = candidate_system_layout_from_row(mss)
        gpus = mss.get("available_gpus") or ["cuda:0"]
        if not (isinstance(gpus, list) and gpus):
            gpus = ["cuda:0"]
        default_gpu = gpus[0]
        decoded_sys = {}
        if isinstance(decoded_config, dict):
            decoded_sys = decoded_config.get("system") or {}

        runtime_knobs: dict = {}
        # (a) kept system_design_space knobs — sampled value, else fixed
        #     scalar/1-elem list. The derived/fixed deployment knobs are NOT
        #     copied (they're computed below); ``deployment_preset`` is a
        #     layout SELECTOR (expanded upstream by resolve_trial_system_config),
        #     not a runtime knob.
        for key, tmpl_val in mss.items():
            if (
                key in _DERIVED_SYSTEM_KNOBS
                or key == "available_gpus"
                or key == "candidate_system_layout"
                or key == "deployment_preset"
                or key.startswith("cut_after_")
                or key.startswith("num_gpus_")
                or key.startswith("pipeline_parallel_")
            ):
                continue
            if key in decoded_sys:
                runtime_knobs[key] = decoded_sys[key]
            elif isinstance(tmpl_val, list) and len(tmpl_val) == 1:
                runtime_knobs[key] = tmpl_val[0]
            elif not isinstance(tmpl_val, (list, dict)):
                runtime_knobs[key] = tmpl_val
        # (a2) system.retrieval runtime knobs (faiss_num_threads /
        #      faiss_ivf_parallel_mode) — searchable/configurable in both modes; here they ride
        #      the measured system_config so MeasuredEvaluator can inject them
        #      into the semantic_retrieval module per trial.
        retrieval_tmpl = system.get("retrieval") or {}
        retrieval_values: dict = {}
        if isinstance(retrieval_tmpl, dict):
            allowed_retrieval = {"faiss_num_threads", "faiss_ivf_parallel_mode",
                                 "num_servers", "faiss_indexing_thread"}
            for key, tmpl_val in retrieval_tmpl.items():
                if key not in allowed_retrieval:
                    continue
                if key in decoded_sys:
                    retrieval_values[key] = decoded_sys[key]
                elif isinstance(tmpl_val, list) and len(tmpl_val) == 1:
                    retrieval_values[key] = tmpl_val[0]
                elif not isinstance(tmpl_val, (list, dict)):
                    retrieval_values[key] = tmpl_val
        # (b) Request batch + decode concurrency. `batch_size_request` is the
        # cross-stage cohort cap; `batch_size_decode` is the generator decode
        # engine's continuous-batching capacity. It is not a per-stage
        # microbatch: prefill/retrieval/rerank still use the request cohort,
        # while decode may hold a larger active sequence set. Values below
        # request are floored to request to avoid throttling the cohort.
        #
        # ``batch_preset`` (curated system space): a single ``"<req>x<dec>"``
        # categorical that pins BOTH batches to a hand-picked (request, decode)
        # operating point — the fair analog of deployment_preset for batching.
        # It avoids the redundant/no-op combos an independent request×decode
        # cartesian dim produces (dec<req floors to a no-op; dec≫req rarely
        # fills). When present it wins over the separate batch knobs.
        bp = runtime_knobs.pop("batch_preset", None)
        if bp is not None:
            try:
                req_s, dec_s = str(bp).lower().split("x", 1)
                runtime_knobs["batch_size_request"] = int(req_s)
                runtime_knobs["batch_size_decode"] = int(dec_s)
            except (ValueError, AttributeError) as exc:
                raise ValueError(
                    f"system_design_space.batch_preset={bp!r} must be "
                    f"'<request>x<decode>' (e.g. '64x256'): {exc}"
                ) from exc
        if "batch_size_request" in runtime_knobs:
            batch_size_request = int(runtime_knobs.pop("batch_size_request"))
        else:
            batch_size_request = int(runtime_knobs.pop("vllm_max_num_seqs", 16))
        if "batch_size_decode" in runtime_knobs:
            batch_size_decode = int(runtime_knobs.pop("batch_size_decode"))
        else:
            batch_size_decode = int(runtime_knobs.pop("vllm_max_num_seqs_decode", batch_size_request))
        batch_size_decode = max(batch_size_request, batch_size_decode)
        # faiss_num_threads — SAME single policy as the CM path (_resolve_system_retrieval):
        # explicit config wins, else min(request batch, cores). Measured has a FIXED
        # request batch, so this is the exact effective FAISS thread count.
        request_batch = batch_size_request
        retrieval_values["faiss_num_threads"] = self._faiss_num_threads(
            retrieval_values.get("faiss_num_threads"), request_batch)
        retrieval_values["faiss_ivf_parallel_mode"] = int(
            retrieval_values.get("faiss_ivf_parallel_mode", 0) or 0
        )
        retrieval_values["num_servers"] = max(
            1, int(retrieval_values.get("num_servers", 1) or 1)
        )
        vllm_max_num_seqs = int(runtime_knobs.pop("vllm_max_num_seqs", batch_size_decode))
        vllm_max_num_seqs_qe = int(
            runtime_knobs.pop(
                "vllm_max_num_seqs_query_expansion",
                batch_size_request,
            )
        )
        vllm_kv_cache_dtype = str(runtime_knobs.pop("vllm_kv_cache_dtype", "auto"))
        vllm_max_num_batched_tokens = runtime_knobs.pop(
            "vllm_max_num_batched_tokens", None
        )
        dynamic_timeout = runtime_knobs.pop(
            "dynamic_batch_timeout_s",
            runtime_knobs.pop("measured_dynamic_batch_timeout_s", 0.002),
        )
        # Do NOT pin the per-iteration token budget: let vLLM choose its own
        # (much larger) default. A small fixed cap (the old hardcoded 2048)
        # throttled prefill for long-context agentic runs (react contexts grow to
        # ~2k+ tokens) → chunked/serialized prefill → inflated latency. Only set
        # vllm_max_num_batched_tokens explicitly to pin a value for a study.

        # (c) present GPU stages from the resolved pipeline (only enabled nodes)
        resolved = resolved_config if resolved_config is not None else \
            self.resolve_config(decoded_config)
        stages = measured_gpu_stages(resolved)
        from rag_stack.rag_ir.workflow_schema import build_workflow_schema
        # Deployment describes executable work, independently of whether an
        # analytical model can price its observed candidate shapes. In
        # particular, hybrid fusion may reduce dense candidates before the
        # reranker; that is valid measured execution even when CM dispatch
        # cannot represent the reduction yet.
        workflow_stages = build_workflow_schema(resolved).stages
        has_fusion = "retrieval_fusion" in workflow_stages
        has_lexical = "lexical_retrieval" in workflow_stages

        # (d) per-engine num_gpus + per-stage cut bits from the decoded config
        #     (explicit raw knobs) or expanded from system.gpu_layout below.
        #     Role-qualified counts (``num_gpus_generator_prefill``) ride along
        #     keyed by the stage's full name and take precedence inside
        #     ``derive_layout`` — this is how asymmetric PD deployments
        #     (presets, CM-selected rows) reach the layout.
        num_gpus: dict = {}
        for s in stages:
            e = stage_engine(s)
            if e == "semantic_retrieval":
                num_gpus[e] = 1
                continue
            num_gpus.setdefault(e, int(decoded_sys.get(num_gpus_key(e), 1) or 1))
            role_val = decoded_sys.get(num_gpus_key(s)) if s != e else None
            if role_val is not None:
                num_gpus[s] = int(role_val or 1)
        cut_bits = {
            cut_after_key(s): int(decoded_sys.get(cut_after_key(s), 0) or 0)
            for s in stages[:-1]
        }

        # TEST HOOK (off by default): deterministically exercise a 1P1D generator
        # without relying on the stochastic optimizer. Collocate every other stage
        # (cut=0) and split only the generator's prefill→decode (cut_after_prefill=1)
        # so the arm fits a 4-GPU box. Set RAG_STACK_FORCE_DISAGG_GENERATOR=1.
        import os as _os
        if _os.environ.get("RAG_STACK_FORCE_DISAGG_GENERATOR", "").strip().lower() in (
            "1", "true", "yes"
        ) and "generator_prefill" in stages and "generator_decode" in stages:
            cut_bits = {k: 0 for k in cut_bits}
            cut_bits[cut_after_key("generator_prefill")] = 1
            # The hook's contract is a DETERMINISTIC 1P1D deployment — it must
            # also override a sampled gpu_layout categorical, which would
            # otherwise replace these cut bits inside the projection.
            decoded_sys = dict(decoded_sys)
            decoded_sys.pop("gpu_layout", None)
            logger.warning(
                "RAG_STACK_FORCE_DISAGG_GENERATOR set — forcing generator into 1P1D "
                "(cut_after_generator_prefill=1, all other stages collocated; "
                "any sampled gpu_layout choice is overridden)"
            )

        ss = system.get("cm_search_space") or {}
        min_total_slots = int(ss.get("min_num_gpus", 1) or 1)
        max_total_slots = int(ss.get("max_num_gpus", len(gpus)) or len(gpus))
        max_stage_slots = ss.get("max_num_gpus_per_stage", max_total_slots)
        if isinstance(max_stage_slots, list):
            max_stage_slots = max(max_stage_slots)

        raw_cut_bits = dict(cut_bits)
        raw_num_gpus = dict(num_gpus)
        layout_choice = decoded_sys.get("gpu_layout")
        cut_bits, num_gpus, layout, projected = project_layout_to_slot_bounds(
            stages,
            cut_bits,
            num_gpus,
            gpus,
            min_slots=min_total_slots,
            max_slots=max_total_slots,
            max_num_gpus_per_stage=int(max_stage_slots),
            layout_choice=layout_choice,
            reranker_components=passage_reranker_components(resolved),
        )
        if projected and layout_choice is None:
            logger.info(
                "Projected measured GPU layout into configured slot budget: "
                f"num_gpus {raw_num_gpus} -> {num_gpus}, "
                f"cut_bits {raw_cut_bits} -> {cut_bits}, "
                f"slots={layout.get('total_slots')} "
                f"(bounds={min_total_slots}-{max_total_slots}, available={len(gpus)})"
            )

        def _stage_devices(stage: str) -> list:
            from rag_stack.taxonomy import TAXONOMY
            engine = stage_engine(stage)
            info = layout["engines"].get(engine, {})
            if info.get("pd_serving") == "disagg_pd":
                kind = TAXONOMY.performance_stage(stage).kind
                if kind == "prefill":
                    return list(info.get("prefill_devices") or [])
                if kind == "decode":
                    return list(info.get("decode_devices") or [])
            return list(info.get("devices") or [])

        def _engine_pp(engine: str, devices: list, *, role: str | None = None) -> int:
            chips = max(1, len(devices))
            role_key = f"pipeline_parallel_{engine}_{role}" if role else None
            raw_pp = (
                decoded_sys.get(role_key)
                if role_key and role_key in decoded_sys
                else decoded_sys.get(f"pipeline_parallel_{engine}", 1)
            )
            pp = int(raw_pp or 1)
            if pp < 1 or chips % pp != 0 or pp > chips:
                if pp != 1:
                    logger.warning(
                        "pipeline parallelism for %s%s=%s does not divide engine chips=%s "
                        "(or exceeds it); falling back to PP=1.",
                        engine,
                        f".{role}" if role else "",
                        pp,
                        chips,
                    )
                pp = 1
            return pp

        def _engine_parallel_record(engine: str, devices: list, *, role: str | None = None) -> dict:
            pp = _engine_pp(engine, devices, role=role)
            chips = max(1, len(devices))
            return {
                "devices": list(devices),
                "num_chips": chips,
                "tp": max(1, chips // pp),
                "pp": pp,
            }

        from rag_stack.taxonomy import TAXONOMY

        stage_defs: dict = {}
        resource_groups: list = []

        def _add_lexical_host_work() -> None:
            # Describe the measured CPU service without claiming a dedicated
            # core, replicated retrieval server, GPU allocation, or CM cost.
            resource_groups.append({
                "id": "lexical:0",
                "kind": "cpu_retrieval",
                "devices": [],
                "performance_stages": ["lexical_retrieval"],
            })
            stage_defs["lexical_retrieval"] = {
                "kind": "cpu_retrieval",
                "resource": "lexical:0",
                "devices": [],
                "num_chips": 0,
                "engine": "lexical_retrieval",
            }

        retrieval_group_added = False
        for gi, (group, group_devices) in enumerate(
            zip(layout.get("groups", []), layout.get("group_devices", []))
        ):
            resource_id = f"gpu:{gi}"
            resource_groups.append({
                "id": resource_id,
                "kind": "gpu",
                "devices": list(group_devices),
                "performance_stages": list(group),
            })
            for stage in group:
                devs = _stage_devices(stage)
                stage_defs[stage] = {
                    "kind": "gpu",
                    "resource": resource_id,
                    "devices": devs,
                    "num_chips": max(1, len(devs)),
                    "engine": stage_engine(stage),
                }
            if "semantic_retrieval_encode" in group and not retrieval_group_added:
                resource_groups.append({
                    "id": "vectorsearch:0",
                    "kind": "cpu_retrieval",
                    "num_servers": retrieval_values["num_servers"],
                    "performance_stages": ["semantic_retrieval_vectorsearch"],
                })
                stage_defs["semantic_retrieval_vectorsearch"] = {
                    "kind": "cpu_retrieval",
                    "resource": "vectorsearch:0",
                    "num_servers": retrieval_values["num_servers"],
                    "engine": "semantic_retrieval",
                }
                if has_lexical:
                    _add_lexical_host_work()
                if has_fusion:
                    resource_groups.append({
                        "id": "fusion:0",
                        "kind": "cpu_retrieval",
                        "devices": [],
                        "performance_stages": ["retrieval_fusion"],
                    })
                    stage_defs["retrieval_fusion"] = {
                        "kind": "cpu_retrieval",
                        "resource": "fusion:0",
                        "devices": [],
                        "num_chips": 0,
                        "engine": "retrieval_fusion",
                    }
                retrieval_group_added = True

        if has_lexical and "lexical_retrieval" not in stage_defs:
            _add_lexical_host_work()

        # Preserve the IR's serial ordering, including lexical retrieval
        # before hybrid fusion. This list names deployed operations; it does
        # not assert that every operation has an analytical performance model.
        stage_order = [stage for stage in workflow_stages if stage in stage_defs]

        engines: dict = {}
        for raw_engine, info in (layout.get("engines") or {}).items():
            pd_serving = str(info.get("pd_serving", "collocated_pd"))
            devs = list(info.get("devices") or [default_gpu])
            entry = {
                "pd_serving": pd_serving,
                "devices": devs,
                "num_chips": max(1, len(devs)),
                "performance_stages": [
                    st for st in stage_order
                    if stage_defs.get(st, {}).get("engine") == raw_engine
                ],
            }
            if raw_engine in {"generator", "query_expansion"}:
                pf_name, dec_name = (
                    ps.name for ps in TAXONOMY.performance_stages_of(raw_engine)
                )
                if pd_serving == "disagg_pd":
                    pre = list(info.get("prefill_devices") or [])
                    dec = list(info.get("decode_devices") or [])
                    entry.update({
                        pf_name: _engine_parallel_record(
                            raw_engine, pre, role="prefill"
                        ),
                        dec_name: _engine_parallel_record(
                            raw_engine, dec, role="decode"
                        ),
                    })
                else:
                    entry.update(_engine_parallel_record(raw_engine, devs))
            engines[raw_engine] = entry

        # Per-GPU occupancy in PerformanceStage terms: everything placement-
        # related speaks PerformanceStage; the engine grouping is recoverable
        # via each stage's owning engine.
        gpu_occupants: dict = {}
        for stage in stage_order:
            info = stage_defs.get(stage) or {}
            for d in info.get("devices") or []:
                gpu_occupants.setdefault(d, []).append(stage)

        vllm_engines = {
            "generator": {
                "max_num_seqs": vllm_max_num_seqs,
                "prefill_max_num_seqs": batch_size_request,
                "decode_max_num_seqs": batch_size_decode,
            }
        }
        if "query_expansion" in engines:
            vllm_engines["query_expansion"] = {
                "max_num_seqs": vllm_max_num_seqs_qe,
            }

        sc: dict = dict(runtime_knobs)
        sc.update({
            "batch_size_request": batch_size_request,
            "batch_size_decode": batch_size_decode,
            "batching": {
                "request": batch_size_request,
                "decode": batch_size_decode,
                "dynamic_timeout_s": float(dynamic_timeout),
            },
            "hardware": {
                "available_gpus": list(gpus),
                "available_gpu_count": len(gpus),
            },
            "retrieval": retrieval_values,
            "vllm": {
                "kv_cache_dtype": vllm_kv_cache_dtype,
                "engines": vllm_engines,
            },
            "layout": {
                "performance_stage_order": stage_order,
                "total_gpu_slots": int(layout.get("total_slots", 0) or 0),
                "min_total_gpu_slots": min_total_slots,
                "max_total_gpu_slots": max_total_slots,
                "resource_groups": resource_groups,
                "performance_stages": stage_defs,
                "engines": engines,
                "gpu_occupants": gpu_occupants,
            },
        })
        if layout_choice is not None:
            sc["layout"]["choice"] = layout_choice
        if selected_candidate_layout is not None:
            # The CM selected a topology-distinct RAGO candidate.  Chips/cuts/PP
            # above still build all runtime and vLLM fields through the normal
            # measured resolver, but the physical ranks themselves are not
            # reconstructible from those lossy dims.  Overlay the selected
            # candidate layout after strict active-stage/device validation.
            # Capability validation must precede that overlay: otherwise a
            # hand-written/historical CM row can replace the safely projected
            # one-card encode or reranker layout and measured will silently use
            # only its first GPU while the CM claimed replicated capacity.
            from rag_stack.runtime_parallelism import (
                validate_measured_runtime_parallelism,
            )
            validate_measured_runtime_parallelism(
                resolved,
                layout=selected_candidate_layout,
                source="CM-selected/measured replay",
            )
            exact = deepcopy(selected_candidate_layout)
            exact_stages = exact.get("performance_stages") or {}
            missing = [stage for stage in stage_order if stage not in exact_stages]
            if missing:
                raise ValueError(
                    "Selected candidate_system_layout is missing active stages: "
                    f"{missing}"
                )
            exact_available = [
                str(device) for device in exact.get("available_devices", [])
            ]
            if not exact_available:
                raise ValueError(
                    "Selected candidate_system_layout has no available_devices"
                )
            for stage in stage_order:
                info = exact_stages[stage]
                if not isinstance(info, dict):
                    raise TypeError(
                        f"candidate_system_layout.performance_stages[{stage!r}] "
                        "must be a mapping"
                    )
                devices = [str(device) for device in info.get("devices", [])]
                if info.get("kind") != "cpu_retrieval":
                    if not devices:
                        raise ValueError(
                            f"Selected candidate stage {stage!r} has no devices"
                        )
                    if not set(devices).issubset(exact_available):
                        raise ValueError(
                            f"Selected candidate stage {stage!r} uses devices "
                            f"outside available_devices: {devices} vs {exact_available}"
                        )
            exact_retrieval_servers: set[int] = set()
            for resource in exact.get("resource_groups", []):
                if (
                    isinstance(resource, dict)
                    and resource.get("kind") == "cpu_retrieval"
                    and "semantic_retrieval_vectorsearch" in (
                        resource.get("performance_stages") or []
                    )
                    and resource.get("num_servers") is not None
                ):
                    exact_retrieval_servers.add(
                        max(1, int(resource["num_servers"]))
                    )
            for stage_name, info in exact_stages.items():
                if (
                    isinstance(info, dict)
                    and stage_name == "semantic_retrieval_vectorsearch"
                    and info.get("kind") == "cpu_retrieval"
                    and info.get("num_servers") is not None
                ):
                    exact_retrieval_servers.add(max(1, int(info["num_servers"])))
            if len(exact_retrieval_servers) > 1:
                raise ValueError(
                    "Selected candidate_system_layout has inconsistent CPU "
                    f"retrieval server counts: {sorted(exact_retrieval_servers)}"
                )
            if exact_retrieval_servers:
                selected_servers = next(iter(exact_retrieval_servers))
                override_servers = mss.get("num_servers")
                if (
                    override_servers is not None
                    and max(1, int(override_servers)) != selected_servers
                ):
                    raise ValueError(
                        "Measured num_servers override disagrees with selected "
                        f"candidate CPU resource: {override_servers} != "
                        f"{selected_servers}"
                    )
                # Direct resolver callers may provide only the candidate
                # layout.  In that case the selected CPU resource remains
                # authoritative, rather than reverting to the template's 1.
                sc["retrieval"]["num_servers"] = selected_servers
            sc["hardware"] = {
                "available_gpus": exact_available,
                "available_gpu_count": len(exact_available),
            }
            sc["layout"] = exact
        if vllm_max_num_batched_tokens is not None:
            sc["vllm"]["max_num_batched_tokens"] = int(vllm_max_num_batched_tokens)
        return sc

    def resolve_trial_system_config(
        self, decoded_config: dict, resolved_config: dict,
    ) -> dict:
        """Per-trial system_config funnel for the optimizer-owned system space.

        The ONE entry point both deployment drivers share:

        * no ``deployment_preset`` in the design space → delegate verbatim to
          :meth:`resolve_system_config` (legacy ``gpu_layout`` / raw-dims
          path, byte-identical for existing measured configs);
        * ``deployment_preset`` declared → expand the sampled preset for the
          ACTUAL resolved pipeline shape into raw dims
          (:func:`~rag_stack.search_space.deployment_presets.expand_preset`,
          the strict budget gate — raises ``DeploymentPresetError`` on an
          unfittable preset), merge the trial's orthogonal runtime knobs
          (``batch_size_request`` …), and feed the SAME builder via
          ``system_dims_override`` — schema parity by construction.

        Used by BOTH performance sources under ``system_space_owner:
        optimizer``: measured launches the deployment, the CM fixed-placement
        provider prices it.
        """
        from rag_stack.search_space.deployment_presets import (
            expand_preset, parse_deployment_presets,
        )
        from rag_stack.search_space.placement import (
            measured_gpu_stages,
            passage_reranker_components,
        )

        system = self.config.get("system") or {}
        mss = system_design_space(system)
        if "deployment_preset" not in mss:
            return self.resolve_system_config(
                decoded_config, resolved_config=resolved_config,
            )

        decoded_sys = {}
        if isinstance(decoded_config, dict):
            decoded_sys = decoded_config.get("system") or {}
        chosen = decoded_sys.get("deployment_preset")
        if chosen is None:
            tmpl = mss.get("deployment_preset")
            if isinstance(tmpl, list) and len(tmpl) == 1:
                chosen = tmpl[0]        # single-preset config → no optimizer dim
            elif isinstance(tmpl, str):
                chosen = tmpl
        presets = parse_deployment_presets(system)
        preset = presets.get(str(chosen))
        if preset is None:
            from rag_stack.search_space.deployment_presets import (
                DeploymentPresetError,
            )
            raise DeploymentPresetError(
                f"deployment_preset {chosen!r} is not defined under "
                f"system.deployment_presets (have: {sorted(presets)})."
            )

        stages = measured_gpu_stages(resolved_config)
        dims = expand_preset(
            preset,
            stages,
            list(mss.get("available_gpus") or []),
            system.get("cm_search_space") or {},
            name=str(chosen),
            reranker_components=passage_reranker_components(resolved_config),
        )
        # Orthogonal runtime knobs ride the override: sampled value wins,
        # else fixed scalar / 1-elem template (same precedence as the
        # template loop inside resolve_system_config).
        for key, tmpl_val in mss.items():
            if (
                key in _DERIVED_SYSTEM_KNOBS
                or key in dims
                or key == "deployment_preset"
                or key == "available_gpus"
                or key.startswith("cut_after_")
                or key.startswith("num_gpus_")
                or key.startswith("pipeline_parallel_")
            ):
                continue
            if key in decoded_sys:
                dims[key] = decoded_sys[key]
            elif isinstance(tmpl_val, list) and len(tmpl_val) == 1:
                dims[key] = tmpl_val[0]
            elif not isinstance(tmpl_val, (list, dict)):
                dims[key] = tmpl_val
        sc = self.resolve_system_config(
            decoded_config,
            resolved_config=resolved_config,
            system_dims_override=dims,
        )
        sc["deployment_preset"] = str(chosen)
        return sc
