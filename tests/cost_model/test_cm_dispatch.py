"""Parity tests for the cost model's (stage, component) dispatch
(:mod:`rag_stack.cost_model.dispatch`).

The old dispatch read ONLY the component (``component``): sub CM =
per-stage override (``encode`` → ``retrieval_encode_sim``) or
``MODULE_TYPE_TO_COST_MODEL[component]`` (db_type-keyed for vectordb) — the
old table literal is copied below and must be reproduced by the new
(stage, component) API through the approved stage translation.
"""

from __future__ import annotations

import pytest

from rag_stack.cost_model.dispatch import (
    cost_model_for,
    performance_stages_for,
    validate_against_registry,
)
from rag_stack.taxonomy import TAXONOMY, Stage, expand_stage

# The OLD registry/cost_model_mapping.json literal (copied, not imported).
_OLD_COST_MODEL_MAPPING = {
    "faiss_ivf": "faiss_ivf",
    "faiss_hnsw": "faiss_hnsw",
    "llama_index_llm": "llm_sim",
    "vllm": "llm_sim",
    "vllm_api": "llm_sim",
    "openai_llm": "llm_sim",
    "hyde": "query_expansion_sim",
    "query_decompose": "query_expansion_sim",
    "multi_query_expansion": "query_expansion_sim",
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
    "tree_summarize": "passage_compressor_sim",
    "refine": "passage_compressor_sim",
    "longllmlingua": "passage_compressor_token_pruning_sim",
    "llmlingua2": "passage_compressor_llmlingua2_sim",
    "similarity_threshold_cutoff": "passage_filter_sim",
    "similarity_percentile_cutoff": "passage_filter_sim",
    "threshold_cutoff": "passage_filter_sim",
    "percentile_cutoff": "passage_filter_sim",
    "recency_filter": "passage_filter_sim",
}


def test_reproduces_old_component_bindings():
    """Every priced non-retrieval component resolves to the exact sub CM the
    old component-keyed table gave it, now on its (identity-expanded) stage."""
    for name, old_cm in _OLD_COST_MODEL_MAPPING.items():
        if name in (
            "faiss_ivf",
            "faiss_hnsw",
            "query_decompose",
            "multi_query_expansion",
        ):
            continue  # db_types or paper-excluded multi-list components
        comp = TAXONOMY.components[name]
        for ds in expand_stage(comp.stage):
            assert cost_model_for(ds, name) == old_cm, (name, ds)


def test_reproduces_old_retrieval_special_cases():
    """The old per-stage override (encode → retrieval_encode_sim) and the old
    inline db_type branch are now plain (stage, component) entries."""
    assert cost_model_for(Stage.SEMANTIC_RETRIEVAL_ENCODE, "vectordb") == "retrieval_encode_sim"
    assert cost_model_for(
        Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH, "vectordb", db_type="faiss_ivf") == "faiss_ivf"
    assert cost_model_for(
        Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH, "vectordb", db_type="faiss_hnsw") == "faiss_hnsw"
    with pytest.raises(KeyError, match="db_type"):
        cost_model_for(Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH, "vectordb")
    with pytest.raises(KeyError, match="db_type"):
        cost_model_for(Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH, "vectordb", db_type="chroma")


def test_unsplit_semantic_retrieval_is_rejected():
    """The CM never expands stages — receiving the user stage is a hard error
    that proves normalization was skipped."""
    with pytest.raises(ValueError, match="never expands"):
        cost_model_for(Stage.SEMANTIC_RETRIEVAL, "vectordb")


def test_stage_component_mismatch_is_rejected():
    """The old dispatch ignored stage entirely (zero validation). The new
    key is the (stage, component) pair — mismatches fail loudly."""
    with pytest.raises(KeyError, match="cannot be dispatched"):
        cost_model_for(Stage.GENERATOR, "hyde")
    with pytest.raises(KeyError, match="cannot be dispatched"):
        cost_model_for(Stage.PASSAGE_RERANKER, "vllm")
    with pytest.raises(KeyError):
        cost_model_for(Stage.GENERATOR, "no_such_component")


def test_unpriced_components_are_rejected():
    with pytest.raises(KeyError, match="not CM-priced"):
        cost_model_for(Stage.PROMPT_MAKER, "fstring")
    with pytest.raises(KeyError, match="not CM-priced"):
        cost_model_for(Stage.PASSAGE_RERANKER, "time_reranker")


def test_performance_stages_for_restricts_to_dispatch_stage():
    assert [ps.name for ps in performance_stages_for(Stage.GENERATOR, "vllm")] == [
        "generator_prefill", "generator_decode"]
    assert [ps.name for ps in performance_stages_for(
        Stage.SEMANTIC_RETRIEVAL_ENCODE, "vectordb")] == ["semantic_retrieval_encode"]
    assert [ps.name for ps in performance_stages_for(
        Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH, "vectordb")] == [
        "semantic_retrieval_vectorsearch"]
    assert [ps.name for ps in performance_stages_for(
        Stage.PASSAGE_COMPRESSOR, "llmlingua2")] == ["passage_compressor"]
    assert [ps.name for ps in performance_stages_for(
        Stage.PASSAGE_COMPRESSOR, "tree_summarize")] == [
        "passage_compressor_prefill", "passage_compressor_decode"]


def test_bindings_name_registered_cost_models():
    """Every binding target must exist in the @register_cost_model registry."""
    validate_against_registry()


def test_incomplete_explicit_registry_fails_closed(monkeypatch):
    """A missing adapter in the explicit composition manifest fails closed."""
    from rag_stack import cost_model

    monkeypatch.delitem(cost_model.COST_MODEL_REGISTRY, "query_expansion_sim")
    with pytest.raises(RuntimeError, match="query_expansion_sim"):
        cost_model._validate_required_registry()


def test_unpriced_components_fail_closed():
    for stage, component in (
        (Stage.LEXICAL_RETRIEVAL, "bm25"),
        (Stage.HYBRID_RETRIEVAL, "vectordb"),
        (Stage.PASSAGE_AUGMENTER, "vectordb"),
    ):
        with pytest.raises(KeyError):
            cost_model_for(stage, component)
    for component in ("query_decompose", "multi_query_expansion"):
        assert cost_model_for(
            Stage.QUERY_EXPANSION, component
        ) == "query_expansion_sim"


# ---------------------------------------------------------------------------
# CM system design space: DEFAULT-ON deployment axes (_expand_deployment_axes)
# ---------------------------------------------------------------------------

def test_deployment_axes_default_expansion():
    import pandas as pd
    from rag_stack.cost_model.engines import _expand_deployment_axes

    base = pd.DataFrame({
        "batch_size_request": [64, 256],
        "placement_policy": ["collocated", "collocated"],
    })
    out = _expand_deployment_axes(
        base.copy(), {"max_batch_size_request": 256},
        agentic=True, decode_pinned=False, timeout_pinned=False,
    )
    # decode ≥ request, closed-loop mirror always included:
    # 64 → {64,128,256}, 256 → {256}; × timeout grid [0.002, 0.01, 0.05].
    assert len(out) == (3 + 1) * 3
    assert (out["batch_size_decode"] >= out["batch_size_request"]).all()
    assert sorted(out["dynamic_batch_timeout_s"].unique()) == [0.002, 0.01, 0.05]
    b64 = out[out["batch_size_request"] == 64]
    assert sorted(b64["batch_size_decode"].unique()) == [64, 128, 256]

    # sequential rows: NO decode axis (decode stays request-mirrored later),
    # timeout axis still applies.
    seq = _expand_deployment_axes(
        base.copy(), {}, agentic=False, decode_pinned=False, timeout_pinned=False,
    )
    assert "batch_size_decode" not in seq.columns
    assert len(seq) == 2 * 3

    # pins suppress each axis independently.
    pinned = _expand_deployment_axes(
        base.copy(), {}, agentic=True, decode_pinned=True, timeout_pinned=True,
    )
    assert len(pinned) == 2
    assert "batch_size_decode" not in pinned.columns
    assert "dynamic_batch_timeout_s" not in pinned.columns

    # explicit grids override the defaults.
    custom = _expand_deployment_axes(
        base.copy(),
        {"batch_size_decode": [128], "dynamic_batch_timeout_s": [0.01]},
        agentic=True, decode_pinned=False, timeout_pinned=False,
    )
    # 64 → {64 (mirror), 128}; 256 → {256 (mirror; 128 < 256 dropped)}.
    assert len(custom) == 3
    assert sorted(custom["dynamic_batch_timeout_s"].unique()) == [0.01]
