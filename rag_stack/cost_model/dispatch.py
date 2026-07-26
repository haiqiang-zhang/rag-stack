"""Sub-cost-model dispatch — the cost model's OWN (stage, component) binding
table. Absorbs the old ``registry/cost_model_mapping.json`` plus the two
special cases that used to hide inside the registry table and an inline
``db_type`` branch of the old ``build_stage_cost_models``.

Contract (taxonomy owns naming, dispatch owns pricing):

* Input is the rag_ir-normalized dispatch manifest — ``semantic_retrieval``
  has ALREADY been split into ``semantic_retrieval_encode`` /
  ``semantic_retrieval_vectorsearch`` upstream. Receiving the un-split
  ``semantic_retrieval`` here is a hard error: the cost model never expands
  stages (that is rag_ir's job, declared in :mod:`rag_stack.taxonomy`).
* :func:`cost_model_for` looks at exactly ``(stage, component)`` — plus
  ``db_type`` for the vector-search stage, whose sub CM is the index family
  (``faiss_ivf`` / ``faiss_hnsw``) chosen at runtime by the vectordb instance.
* Unknown names, stage/component mismatches, and unpriced components raise —
  no silent fallbacks.

Validated against :data:`rag_stack.taxonomy.TAXONOMY` at import time: every
priced component reaches a binding for every stage it can be dispatched on.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_stack.taxonomy import (
    TAXONOMY,
    PerformanceStage,
    Stage,
    expand_stage,
)

# ── component → sub-CM bindings (the old cost_model_mapping.json, minus the
#    db_type rows which were never components and minus "vectordb" whose
#    binding is stage-dependent below) ────────────────────────────────────────

_COMPONENT_CM: dict[str, str] = {
    "rrf": "rrf",
    # generator (llm-shaped)
    "vllm": "llm_sim",
    "vllm_api": "llm_sim",
    "openai_llm": "llm_sim",
    "llama_index_llm": "llm_sim",
    # query expansion (llm-shaped)
    "hyde": "query_expansion_sim",
    "query_decompose": "query_expansion_sim",
    "multi_query_expansion": "query_expansion_sim",
    # rerankers (forward-shaped); encoder cross-encoders vs LLM rerankers
    # price differently but share the passage_reranker PerformanceStage.
    "sentence_transformer_reranker": "passage_reranker_sim",
    "tart": "passage_reranker_sim",
    "monot5": "passage_reranker_sim",
    "colbert_reranker": "passage_reranker_sim",
    "flag_embedding_reranker": "passage_reranker_sim",
    "openvino_reranker": "passage_reranker_sim",
    "flashrank_reranker": "passage_reranker_sim",
    "koreranker": "passage_reranker_sim",
    "cohere_reranker": "passage_reranker_sim",
    "jina_reranker": "passage_reranker_sim",
    "voyageai_reranker": "passage_reranker_sim",
    "mixedbreadai_reranker": "passage_reranker_sim",
    "rankgpt": "passage_reranker_llm_sim",
    "flag_embedding_llm_reranker": "passage_reranker_llm_sim",
    "upr": "passage_reranker_llm_sim",
    # compressors
    "tree_summarize": "passage_compressor_sim",              # generative (llm-shaped)
    "refine": "passage_compressor_sim",                      # generative (llm-shaped)
    "longllmlingua": "passage_compressor_token_pruning_sim",  # pruning (forward)
    "llmlingua2": "passage_compressor_llmlingua2_sim",        # pruning (forward)
    # filters (forward-shaped)
    "similarity_threshold_cutoff": "passage_filter_sim",
    "similarity_percentile_cutoff": "passage_filter_sim",
    "threshold_cutoff": "passage_filter_sim",
    "percentile_cutoff": "passage_filter_sim",
    "recency_filter": "passage_filter_sim",
}

#: Index families the vector-search stage can be priced by (db_type values —
#: their registered cost-model names are themselves).
_VECTORSEARCH_DB_TYPES: frozenset[str] = frozenset({"faiss_ivf", "faiss_hnsw"})


@dataclass(frozen=True)
class CostModelBinding:
    """One registry binding, before a concrete workload and deployment exist."""

    stage: Stage
    component: str
    db_type: str | None
    cost_model: str


def cost_model_for(stage: Stage | str, component: str, db_type: str | None = None) -> str:
    """Registered sub-cost-model name for an already-normalized
    ``(stage, component)`` dispatch entry.

    ``db_type`` is required exactly when ``stage`` is the vector-search split
    stage. Hard errors on: unknown component, un-split ``semantic_retrieval``,
    component/stage mismatch, unpriced components, unknown db_type.
    """
    s = Stage(stage)
    comp = TAXONOMY.components[component]  # KeyError on unknown component

    if s is Stage.SEMANTIC_RETRIEVAL:
        raise ValueError(
            "cost_model_for received un-split 'semantic_retrieval' — stage "
            "normalization (rag_ir) must run before CM dispatch; the cost "
            "model never expands stages.")
    allowed = expand_stage(comp.stage)
    if s not in allowed:
        raise KeyError(
            f"component {component!r} (stage {comp.stage.value!r}) cannot be "
            f"dispatched on stage {s.value!r}; allowed: {[a.value for a in allowed]}")

    if s is Stage.SEMANTIC_RETRIEVAL_ENCODE:
        return "retrieval_encode_sim"
    if s is Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH:
        if db_type not in _VECTORSEARCH_DB_TYPES:
            raise KeyError(
                f"vector-search dispatch needs db_type in "
                f"{sorted(_VECTORSEARCH_DB_TYPES)}; got {db_type!r}")
        return db_type

    if comp.shape is None:
        raise KeyError(
            f"component {component!r} is not CM-priced (no PerformanceStage); "
            f"it must not reach the cost-model dispatcher")
    return _COMPONENT_CM[component]


def performance_stages_for(stage: Stage | str, component: str) -> tuple[PerformanceStage, ...]:
    """The PerformanceStages a ``(stage, component)`` dispatch entry produces —
    the taxonomy naming law restricted to this dispatch stage. This is what a
    sub cost model instantiated for the entry must emit."""
    s = Stage(stage)
    return tuple(ps for ps in TAXONOMY.performance_stages_of_component(component)
                 if ps.stage is s)


def component_binding_table() -> dict[str, str]:
    """Flat component → sub-cost-model view of the bindings (read-only),
    plus the db_type-keyed vector-search families — the payload shape the
    frontend mapping table renders. Derived, never persisted."""
    table = dict(_COMPONENT_CM)
    table.update({db: db for db in sorted(_VECTORSEARCH_DB_TYPES)})
    return table


def dispatch_binding_table() -> tuple[CostModelBinding, ...]:
    """Complete stage-aware catalog of the production dispatcher.

    Unlike the historical component-keyed view, this preserves the embedding
    and index-family alternatives of ``vectordb``. It describes possible
    bindings, not an executed path or a calibration-readiness verdict.
    """
    bindings = []
    for component in TAXONOMY.components.values():
        if component.shape is None:
            continue
        for stage in expand_stage(component.stage):
            db_types = (sorted(_VECTORSEARCH_DB_TYPES)
                        if stage is Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH else [None])
            for db_type in db_types:
                bindings.append(CostModelBinding(
                    stage=stage, component=component.name, db_type=db_type,
                    cost_model=cost_model_for(stage, component.name, db_type),
                ))
    return tuple(bindings)


def validate_against_registry() -> None:
    """Every binding must name a registered cost model. Called from tests (not
    at import — the ``@register_cost_model`` registry is populated by the
    cost_model package's auto-discovery, which may not have run yet when this
    module is first imported)."""
    from rag_stack.cost_model import COST_MODEL_REGISTRY

    missing = sorted(
        {cm for cm in _COMPONENT_CM.values() if cm not in COST_MODEL_REGISTRY}
        | {db for db in _VECTORSEARCH_DB_TYPES if db not in COST_MODEL_REGISTRY}
        | ({"retrieval_encode_sim"} - set(COST_MODEL_REGISTRY)))
    if missing:
        raise AssertionError(f"dispatch bindings name unregistered cost models: {missing}")


# ── import-time validation against the taxonomy ─────────────────────────────

def _validate_against_taxonomy() -> None:
    for comp in TAXONOMY.components.values():
        if comp.shape is None:
            assert comp.name not in _COMPONENT_CM, (
                f"unpriced component {comp.name!r} must not carry a CM binding")
            continue
        for ds in expand_stage(comp.stage):
            db = "faiss_ivf" if ds is Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH else None
            name = cost_model_for(ds, comp.name, db_type=db)
            assert isinstance(name, str) and name, (
                f"no sub-CM binding for ({ds.value}, {comp.name})")
            produced = performance_stages_for(ds, comp.name)
            assert produced, (
                f"({ds.value}, {comp.name}) is priced but produces no "
                f"PerformanceStage — naming law violated")
    bound_unknown = set(_COMPONENT_CM) - set(TAXONOMY.components)
    assert not bound_unknown, (
        f"dispatch table binds names missing from the taxonomy: {sorted(bound_unknown)}")


_validate_against_taxonomy()
