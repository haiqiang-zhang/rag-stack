"""Central naming taxonomy — THE single source of truth for the three concepts
that name everything in rag_stack:

* :class:`Stage` — pipeline stages. Users set seven of them in YAML
  (``stage:`` key). ``semantic_retrieval`` is special: it is *embedding* plus
  *vector search*, two different execution realities, so the algorithm layer
  (rag_ir normalization — NEVER the cost model) expands it into the two split
  stages ``semantic_retrieval_encode`` / ``semantic_retrieval_vectorsearch``
  before anything downstream sees it (:data:`STAGE_EXPANSION`).
* :class:`Component` — the implementation selected for a stage (``component:``
  key in YAML; the pre-refactor "module type"): ``vllm``, ``hyde``,
  ``llmlingua2``, ``vectordb``, …
* :class:`PerformanceStage` — the flat, globally-unique performance/deployment
  execution units that sub cost models PRODUCE and that every placement-facing
  artifact speaks: resource groups, layouts, ``cut_after_*`` knobs,
  sweep-df columns, proj-dir experiment data.

Naming law (declared here, obeyed by every sub cost model):

* an **llm-shaped** execution (autoregressive prefill + decode) produces the
  PerformanceStage pair ``{stage}_prefill`` + ``{stage}_decode``;
* a **forward-shaped** execution (one non-autoregressive forward: encoder
  rerankers, token-pruning compressors, filters, query embedding, FAISS
  search) produces the single bare PerformanceStage ``{stage}``.

The full PerformanceStage catalog is therefore *derived* from
(stage × component shape) — it is not a third hand-maintained table — and is
validated for global uniqueness at import time.

Scope and non-goals:

* This module covers **rag_stack only**. External libraries keep their own
  vocabularies and are bridged by adapters (RAGO ⇄
  ``rag_stack.cost_model.rago_adapter``; FlashRAG monitor events ⇄ the
  ``flashrag.integrations.rag_stack``).
* This module manages **naming only**. Which sub cost model prices a given
  (stage, component) is the cost model's own business
  (``rag_stack.cost_model.dispatch``), validated there against this taxonomy.

Zero heavy dependencies (stdlib only) so every package — rag_ir,
search_space, server, cost_model — can import it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Iterable, Mapping


class Stage(StrEnum):
    """Every pipeline stage name in the system — user-settable stages plus the
    two rag_ir split products of ``semantic_retrieval``. Values ARE the wire
    strings (StrEnum), so members interchange freely with dict/JSON keys."""

    QUERY_EXPANSION = "query_expansion"
    SEMANTIC_RETRIEVAL = "semantic_retrieval"          # user/YAML concept; split before the CM
    SEMANTIC_RETRIEVAL_ENCODE = "semantic_retrieval_encode"            # split stage (rag_ir product)
    SEMANTIC_RETRIEVAL_VECTORSEARCH = "semantic_retrieval_vectorsearch"  # split stage (rag_ir product)
    LEXICAL_RETRIEVAL = "lexical_retrieval"            # bm25 keyword retrieval
    RETRIEVAL_FUSION = "retrieval_fusion"              # explicit CPU result-list fusion
    HYBRID_RETRIEVAL = "hybrid_retrieval"              # logical fusion node; observed operation is retrieval_fusion
    PASSAGE_AUGMENTER = "passage_augmenter"            # recognized algorithm stage (no CM pricing)
    PASSAGE_RERANKER = "passage_reranker"
    PASSAGE_FILTER = "passage_filter"
    PASSAGE_COMPRESSOR = "passage_compressor"
    PROMPT_MAKER = "prompt_maker"                      # no PerformanceStage (not CM-priced)
    GENERATOR = "generator"


#: The stages a user can set in YAML (``stage:`` key). Split stages are NOT
#: user-settable — they exist only downstream of rag_ir normalization.
USER_STAGES: tuple[Stage, ...] = (
    Stage.QUERY_EXPANSION,
    Stage.SEMANTIC_RETRIEVAL,
    Stage.LEXICAL_RETRIEVAL,
    Stage.HYBRID_RETRIEVAL,
    Stage.PASSAGE_RERANKER,
    Stage.PASSAGE_FILTER,
    Stage.PASSAGE_COMPRESSOR,
    Stage.PROMPT_MAKER,
    Stage.GENERATOR,
)

#: The one declared stage-expansion rule. DECLARED here; APPLIED exclusively by
#: rag_ir normalization (the algorithm layer). The cost model never expands —
#: it must receive already-split input.
STAGE_EXPANSION: Mapping[Stage, tuple[Stage, ...]] = MappingProxyType({
    Stage.SEMANTIC_RETRIEVAL: (
        Stage.SEMANTIC_RETRIEVAL_ENCODE,
        Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH,
    ),
})

#: Split stage → owning user stage (engine grouping, gpu_occupants labels).
PARENT_STAGE: Mapping[Stage, Stage] = MappingProxyType({
    Stage.SEMANTIC_RETRIEVAL_ENCODE: Stage.SEMANTIC_RETRIEVAL,
    Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH: Stage.SEMANTIC_RETRIEVAL,
})


def expand_stage(stage: Stage | str) -> tuple[Stage, ...]:
    """Apply the declared stage expansion (identity for every stage except
    ``semantic_retrieval``). The ONLY legitimate caller is rag_ir
    normalization; cost-model code must never call this."""
    s = Stage(stage)
    return STAGE_EXPANSION.get(s, (s,))


# ── shapes: the naming law ──────────────────────────────────────────────────
#: llm-shaped execution → ``{stage}_prefill`` + ``{stage}_decode``.
SHAPE_LLM = "llm"
#: single-forward execution → bare ``{stage}``.
SHAPE_FORWARD = "forward"


@dataclass(frozen=True)
class Component:
    """A selectable implementation of a stage (the pre-refactor "module type").

    Pure naming: ``shape`` states which PerformanceStages this component's
    execution produces under the naming law (``None`` → not CM-priced, e.g.
    prompt makers). Which sub cost model prices it lives in
    ``cost_model.dispatch``, not here.
    """

    name: str
    stage: Stage
    shape: str | None  # SHAPE_LLM | SHAPE_FORWARD | None


@dataclass(frozen=True)
class PerformanceStage:
    """One performance/deployment execution unit (flat, globally unique).

    Produced by sub cost models per the naming law; the resource axis drives
    the measured GPU projection and resource-group membership.
    """

    name: str
    stage: Stage               # owning (possibly split) stage
    kind: str                  # "prefill" | "decode" (GenZ sim kind; one forward = "prefill")
    resource: str = "gpu"      # "gpu" | "cpu_retrieval"


# ── component declarations (the pre-refactor registry table, as naming) ─────

def _components() -> dict[str, Component]:
    def c(name: str, stage: Stage, shape: str | None) -> tuple[str, Component]:
        return name, Component(name, stage, shape)

    generators = [c(n, Stage.GENERATOR, SHAPE_LLM)
                  for n in ("vllm", "vllm_api", "openai_llm", "llama_index_llm")]
    query_expansions = [c(n, Stage.QUERY_EXPANSION, SHAPE_LLM)
                        for n in ("hyde", "query_decompose", "multi_query_expansion")]
    rerankers = [c(n, Stage.PASSAGE_RERANKER, SHAPE_FORWARD) for n in (
        "sentence_transformer_reranker", "tart", "monot5", "colbert_reranker",
        "flag_embedding_reranker", "openvino_reranker", "flashrank_reranker",
        "koreranker", "cohere_reranker", "jina_reranker", "voyageai_reranker",
        "mixedbreadai_reranker",
    )]
    llm_rerankers = [c(n, Stage.PASSAGE_RERANKER, SHAPE_LLM) for n in (
        "rankgpt", "flag_embedding_llm_reranker", "upr",
    )]
    compressors = [
        c("tree_summarize", Stage.PASSAGE_COMPRESSOR, SHAPE_LLM),   # generative
        c("refine", Stage.PASSAGE_COMPRESSOR, SHAPE_LLM),           # generative
        c("longllmlingua", Stage.PASSAGE_COMPRESSOR, SHAPE_FORWARD),  # token pruning
        c("llmlingua2", Stage.PASSAGE_COMPRESSOR, SHAPE_FORWARD),     # BERT classifier
    ]
    retrieval = [
        c("vectordb", Stage.SEMANTIC_RETRIEVAL, SHAPE_FORWARD),
        c("rrf", Stage.RETRIEVAL_FUSION, SHAPE_FORWARD),
    ]
    filters = [c(n, Stage.PASSAGE_FILTER, SHAPE_FORWARD) for n in (
        "similarity_threshold_cutoff", "similarity_percentile_cutoff",
        "threshold_cutoff", "percentile_cutoff", "recency_filter",
    )]
    # Not CM-priced (no PerformanceStage), still part of the naming universe:
    unpriced = [
        c("fstring", Stage.PROMPT_MAKER, None),
        c("long_context_reorder", Stage.PROMPT_MAKER, None),
        c("window_replacement", Stage.PROMPT_MAKER, None),
        c("time_reranker", Stage.PASSAGE_RERANKER, None),  # pure-Python sort
        c("bm25", Stage.LEXICAL_RETRIEVAL, None),          # CPU inverted index
        # Logical configuration names. The evaluator reports the actual RRF
        # operation as retrieval_fusion/rrf; these must not add duplicate cost.
        c("hybrid_rrf", Stage.HYBRID_RETRIEVAL, None),
        c("hybrid_cc", Stage.HYBRID_RETRIEVAL, None),
    ]
    return dict(generators + query_expansions + rerankers + llm_rerankers + compressors
                + retrieval + filters + unpriced)


def _performance_stages_of_component(comp: Component) -> tuple[PerformanceStage, ...]:
    """Derive the PerformanceStages a component produces (naming law)."""
    if comp.shape is None:
        return ()
    out: list[PerformanceStage] = []
    for ds in expand_stage(comp.stage):
        if comp.shape == SHAPE_LLM:
            out.append(PerformanceStage(f"{ds.value}_prefill", ds, "prefill"))
            out.append(PerformanceStage(f"{ds.value}_decode", ds, "decode"))
        elif comp.shape == SHAPE_FORWARD:
            resource = ("cpu_retrieval"
                        if ds in (Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH,
                                  Stage.RETRIEVAL_FUSION) else "gpu")
            out.append(PerformanceStage(ds.value, ds, "prefill", resource))
        else:  # pragma: no cover — unknown shapes are a programming error
            raise ValueError(f"unknown component shape: {comp.shape!r} on {comp.name!r}")
    return tuple(out)


class RagTaxonomy:
    """The central naming authority (singleton :data:`TAXONOMY`).

    Everything in rag_stack reads names from here; strings enter/leave only at
    I/O boundaries. Lookups are table-based — no string parsing anywhere.
    """

    def __init__(self) -> None:
        self.stages: tuple[Stage, ...] = tuple(Stage)
        self.user_stages: tuple[Stage, ...] = USER_STAGES
        self.parent_stage: Mapping[Stage, Stage] = PARENT_STAGE
        self.components: Mapping[str, Component] = MappingProxyType(_components())

        # Derived PerformanceStage catalog: canonical singletons, name-keyed.
        catalog: dict[str, PerformanceStage] = {}
        by_component: dict[str, tuple[PerformanceStage, ...]] = {}
        for comp in self.components.values():
            stages = _performance_stages_of_component(comp)
            deduped: list[PerformanceStage] = []
            for ps in stages:
                existing = catalog.get(ps.name)
                if existing is None:
                    catalog[ps.name] = ps
                    deduped.append(ps)
                elif existing != ps:
                    raise AssertionError(
                        f"PerformanceStage name collision with divergent specs: "
                        f"{ps.name!r} ({existing} vs {ps})"
                    )
                else:
                    deduped.append(existing)
            by_component[comp.name] = tuple(deduped)
        self.performance_stages: Mapping[str, PerformanceStage] = MappingProxyType(catalog)
        self._by_component: Mapping[str, tuple[PerformanceStage, ...]] = (
            MappingProxyType(by_component))

        self.llm_stages: frozenset[Stage] = frozenset(
            {Stage.GENERATOR, Stage.QUERY_EXPANSION})
        self.llm_performance_stages: frozenset[str] = frozenset(
            ps.name
            for component in self.components.values()
            if component.shape == SHAPE_LLM
            for ps in self._by_component[component.name]
        )

        self._validate()

    # ── validation (import time) ────────────────────────────────────────────
    def _validate(self) -> None:
        stage_values = {s.value for s in Stage}
        for name, ps in self.performance_stages.items():
            assert name == ps.name
            # A PerformanceStage may share a name ONLY with its own stage
            # (bare forward units: passage_reranker, semantic_retrieval_encode, …).
            if name in stage_values:
                assert ps.stage.value == name, (
                    f"PerformanceStage {name!r} collides with unrelated stage")
            assert ps.kind in ("prefill", "decode")
            assert ps.resource in ("gpu", "cpu_retrieval")
        for comp in self.components.values():
            assert comp.stage in USER_STAGES or comp.stage in (
                Stage.LEXICAL_RETRIEVAL, Stage.RETRIEVAL_FUSION,
            ), (
                f"component {comp.name!r} attached to non-user stage {comp.stage}")
        for src, targets in STAGE_EXPANSION.items():
            assert src in USER_STAGES
            for t in targets:
                assert PARENT_STAGE[t] is src

    # ── queries (the system-wide read API) ──────────────────────────────────
    def component(self, name: str) -> Component:
        """Component by name — hard KeyError on unknown names."""
        return self.components[name]

    def components_of(self, stage: Stage | str) -> tuple[Component, ...]:
        s = Stage(stage)
        return tuple(c for c in self.components.values() if c.stage is s)

    def performance_stage(self, name: str) -> PerformanceStage:
        """PerformanceStage by flat name — hard KeyError on unknown names."""
        return self.performance_stages[name]

    def performance_stages_of(self, stage: Stage | str) -> tuple[PerformanceStage, ...]:
        s = Stage(stage)
        return tuple(ps for ps in self.performance_stages.values() if ps.stage is s)

    def performance_stages_of_component(self, name: str) -> tuple[PerformanceStage, ...]:
        """The PerformanceStages a component produces (naming law applied)."""
        return self._by_component[name]

    def stage_of(self, name: str) -> Stage:
        """Owning stage for ANY global name — a stage's own value or a
        PerformanceStage name. Table lookup, no string parsing."""
        ps = self.performance_stages.get(name)
        if ps is not None:
            return ps.stage
        return Stage(name)  # raises ValueError on unknown names

    def user_stage_of(self, name: str) -> Stage:
        """Like :meth:`stage_of` but collapses split stages to their user
        stage (engine grouping / gpu-occupant labeling)."""
        s = self.stage_of(name)
        return PARENT_STAGE.get(s, s)

    def gpu_performance_stages_for_component(self, name: str) -> tuple[PerformanceStage, ...]:
        return tuple(ps for ps in self._by_component[name] if ps.resource == "gpu")

    def gpu_performance_stages_for_node(
        self, stage: Stage | str, components: Iterable[str],
    ) -> tuple[PerformanceStage, ...]:
        """Ordered union of GPU PerformanceStages over a node's components —
        the unit the measured placement operates on. Replaces the hand-copied
        ``measured_gpu_stages`` expansion and the ``_DECODE_BEARING_COMPRESSORS``
        special case: decode-bearing-ness falls out of component shapes."""
        s = Stage(stage)
        out: list[PerformanceStage] = []
        for comp_name in components:
            comp = self.components[comp_name]
            if comp.stage is not s:
                raise KeyError(
                    f"component {comp_name!r} belongs to stage {comp.stage.value!r}, "
                    f"not {s.value!r}")
            for ps in self.gpu_performance_stages_for_component(comp_name):
                if ps not in out:
                    out.append(ps)
        return tuple(out)


#: Module-level singleton — the one instance everything imports.
TAXONOMY = RagTaxonomy()
