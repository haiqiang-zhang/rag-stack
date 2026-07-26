"""Parity and invariant tests for the central naming taxonomy
(:mod:`rag_stack.taxonomy`) and the RAGO name adapter
(:mod:`rag_stack.cost_model.rago_adapter`).

The parity tests copy the OLD registry literals verbatim (from
``rag_stack/registry/cost_model_stage_node_mapping.py`` at commit 93128d7) and
prove the taxonomy derives exactly the same facts under the new names — the
strongest guard that the refactor changed names, not semantics.
"""

from __future__ import annotations

import pytest

from rag_stack.taxonomy import (
    PARENT_STAGE,
    SHAPE_FORWARD,
    SHAPE_LLM,
    STAGE_EXPANSION,
    TAXONOMY,
    USER_STAGES,
    Stage,
    expand_stage,
)


# ── the OLD registry literals (copied, not imported — they must not drift) ──

# component → ordered (old stage name, old stage, kind) triples.
_OLD_MODULE_STAGES = {
    "vllm": [("prefill", "generator", "prefill"), ("decode", "generator", "decode")],
    "vllm_api": [("prefill", "generator", "prefill"), ("decode", "generator", "decode")],
    "openai_llm": [("prefill", "generator", "prefill"), ("decode", "generator", "decode")],
    "llama_index_llm": [("prefill", "generator", "prefill"), ("decode", "generator", "decode")],
    "hyde": [("query_expansion_prefill", "query_expansion", "prefill"),
             ("query_expansion_decode", "query_expansion", "decode")],
    "query_decompose": [("query_expansion_prefill", "query_expansion", "prefill"),
                        ("query_expansion_decode", "query_expansion", "decode")],
    "multi_query_expansion": [("query_expansion_prefill", "query_expansion", "prefill"),
                              ("query_expansion_decode", "query_expansion", "decode")],
    **{r: [("passage_reranker", "passage_reranker", "prefill")] for r in (
        "sentence_transformer_reranker", "tart", "monot5", "colbert_reranker",
        "flag_embedding_reranker", "openvino_reranker", "flashrank_reranker",
        "koreranker", "cohere_reranker", "jina_reranker", "voyageai_reranker",
        "mixedbreadai_reranker", "rankgpt", "flag_embedding_llm_reranker", "upr")},
    "tree_summarize": [("passage_compressor_prefill", "passage_compressor", "prefill"),
                       ("passage_compressor_decode", "passage_compressor", "decode")],
    "refine": [("passage_compressor_prefill", "passage_compressor", "prefill"),
               ("passage_compressor_decode", "passage_compressor", "decode")],
    "longllmlingua": [("passage_compressor_prefill", "passage_compressor", "prefill")],
    "llmlingua2": [("passage_compressor_prefill", "passage_compressor", "prefill")],
    "vectordb": [("encode", "semantic_retrieval", "prefill"),
                 ("retrieval", "semantic_retrieval", "prefill")],
    **{f: [("passage_filter", "passage_filter", "prefill")] for f in (
        "similarity_threshold_cutoff", "similarity_percentile_cutoff",
        "threshold_cutoff", "percentile_cutoff", "recency_filter")},
}

# old stage name → new PerformanceStage name (the 5 renames + identities).
_OLD_TO_NEW = {
    "prefill": "generator_prefill",
    "decode": "generator_decode",
    "encode": "semantic_retrieval_encode",
    "retrieval": "semantic_retrieval_vectorsearch",
}


def _new_name(old: str, component: str) -> str:
    if old == "passage_compressor_prefill" and component in ("llmlingua2", "longllmlingua"):
        return "passage_compressor"  # pruning compressors: bare single unit
    return _OLD_TO_NEW.get(old, old)


# ── parity: taxonomy ≡ old registry (modulo the approved renames) ───────────

def test_component_performance_stages_match_old_module_stages():
    for component, old_specs in _OLD_MODULE_STAGES.items():
        got = TAXONOMY.performance_stages_of_component(component)
        if component in {"rankgpt", "flag_embedding_llm_reranker", "upr"}:
            # Intentional taxonomy correction: these are autoregressive LLM
            # rerankers, not encoder forwards as the old registry claimed.
            expected = [
                ("passage_reranker_prefill", "passage_reranker", "prefill"),
                ("passage_reranker_decode", "passage_reranker", "decode"),
            ]
        else:
            expected = [(_new_name(n, component), nt, k) for (n, nt, k) in old_specs]
        assert [(ps.name, TAXONOMY.user_stage_of(ps.name).value, ps.kind) for ps in got] \
            == expected, f"mismatch for {component}"


def test_old_stage_kind_map_reproduced():
    # Old STAGE_KIND, keyed by the old names, translated per-module where the
    # old name was module-ambiguous (compressor prefill).
    old_stage_kind = {
        "prefill": "prefill", "decode": "decode",
        "query_expansion_prefill": "prefill", "query_expansion_decode": "decode",
        "passage_reranker": "prefill", "passage_filter": "prefill",
        "passage_compressor_prefill": "prefill", "passage_compressor_decode": "decode",
        "encode": "prefill", "retrieval": "prefill",
    }
    for old, kind in old_stage_kind.items():
        new = _OLD_TO_NEW.get(old, old)
        assert TAXONOMY.performance_stage(new).kind == kind
    # The pruning compressor's bare unit keeps the forward kind.
    assert TAXONOMY.performance_stage("passage_compressor").kind == "prefill"


def test_catalog_includes_independent_retrieval_fusion():
    assert set(TAXONOMY.performance_stages) == {
        "generator_prefill", "generator_decode",
        "query_expansion_prefill", "query_expansion_decode",
        "semantic_retrieval_encode", "semantic_retrieval_vectorsearch",
        "retrieval_fusion",
        "passage_reranker", "passage_reranker_prefill",
        "passage_reranker_decode", "passage_filter",
        "passage_compressor", "passage_compressor_prefill", "passage_compressor_decode",
    }


def test_vectorsearch_and_fusion_are_cpu_units():
    cpu = [ps.name for ps in TAXONOMY.performance_stages.values()
           if ps.resource == "cpu_retrieval"]
    assert cpu == ["semantic_retrieval_vectorsearch", "retrieval_fusion"]


# ── measured projection parity (replaces measured_gpu_stages) ───────────────

def test_gpu_projection_matches_old_measured_expansion():
    # Old measured_gpu_stages semantics (placement.py at 93128d7), new names:
    # generator → prefill+decode; query_expansion → its pair;
    # semantic_retrieval → ONE GPU slot (the encode unit — vectorsearch is CPU);
    # reranker → itself; compressor → prefill (+decode iff generative).
    def names(stage, comps):
        return [ps.name for ps in TAXONOMY.gpu_performance_stages_for_node(stage, comps)]

    assert names(Stage.GENERATOR, ["vllm"]) == ["generator_prefill", "generator_decode"]
    assert names(Stage.QUERY_EXPANSION, ["hyde"]) == [
        "query_expansion_prefill", "query_expansion_decode"]
    assert names(Stage.SEMANTIC_RETRIEVAL, ["vectordb"]) == ["semantic_retrieval_encode"]
    assert names(Stage.PASSAGE_RERANKER, ["monot5"]) == ["passage_reranker"]
    assert names(Stage.PASSAGE_RERANKER, ["rankgpt"]) == [
        "passage_reranker_prefill", "passage_reranker_decode"]
    # Decode-bearing-ness falls out of component shape — no special-case set.
    assert names(Stage.PASSAGE_COMPRESSOR, ["llmlingua2"]) == ["passage_compressor"]
    assert names(Stage.PASSAGE_COMPRESSOR, ["tree_summarize"]) == [
        "passage_compressor_prefill", "passage_compressor_decode"]
    assert names(Stage.PASSAGE_COMPRESSOR, ["llmlingua2", "tree_summarize"]) == [
        "passage_compressor", "passage_compressor_prefill", "passage_compressor_decode"]


def test_gpu_projection_rejects_component_stage_mismatch():
    with pytest.raises(KeyError):
        TAXONOMY.gpu_performance_stages_for_node(Stage.GENERATOR, ["hyde"])


# ── structural invariants ───────────────────────────────────────────────────

def test_expansion_is_declared_once_and_consistent():
    assert expand_stage(Stage.SEMANTIC_RETRIEVAL) == (
        Stage.SEMANTIC_RETRIEVAL_ENCODE, Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH)
    assert expand_stage(Stage.GENERATOR) == (Stage.GENERATOR,)
    assert expand_stage("passage_reranker") == (Stage.PASSAGE_RERANKER,)
    for src, targets in STAGE_EXPANSION.items():
        assert src in USER_STAGES
        for t in targets:
            assert PARENT_STAGE[t] is src
            assert t not in USER_STAGES


def test_stage_of_and_user_stage_of_are_table_lookups():
    assert TAXONOMY.stage_of("generator_prefill") is Stage.GENERATOR
    assert TAXONOMY.stage_of("passage_reranker") is Stage.PASSAGE_RERANKER
    assert TAXONOMY.stage_of("semantic_retrieval_encode") is Stage.SEMANTIC_RETRIEVAL_ENCODE
    assert TAXONOMY.user_stage_of("semantic_retrieval_encode") is Stage.SEMANTIC_RETRIEVAL
    assert TAXONOMY.user_stage_of("semantic_retrieval_vectorsearch") is Stage.SEMANTIC_RETRIEVAL
    assert TAXONOMY.user_stage_of("generator_decode") is Stage.GENERATOR
    with pytest.raises(ValueError):
        TAXONOMY.stage_of("no_such_name")
    with pytest.raises(KeyError):
        TAXONOMY.performance_stage("no_such_name")


def test_llm_stages_and_shapes():
    assert TAXONOMY.llm_stages == {Stage.GENERATOR, Stage.QUERY_EXPANSION}
    assert TAXONOMY.component("vllm").shape == SHAPE_LLM
    assert TAXONOMY.component("llmlingua2").shape == SHAPE_FORWARD
    assert TAXONOMY.component("tree_summarize").shape == SHAPE_LLM
    assert TAXONOMY.component("rankgpt").shape == SHAPE_LLM
    assert TAXONOMY.component("monot5").shape == SHAPE_FORWARD
    assert {
        "passage_reranker_prefill", "passage_reranker_decode"
    } <= TAXONOMY.llm_performance_stages
    assert TAXONOMY.component("fstring").shape is None
    assert TAXONOMY.performance_stages_of_component("fstring") == ()


def test_components_of_stage():
    gens = {c.name for c in TAXONOMY.components_of(Stage.GENERATOR)}
    assert gens == {"vllm", "vllm_api", "openai_llm", "llama_index_llm"}
    prompt = {c.name for c in TAXONOMY.components_of(Stage.PROMPT_MAKER)}
    assert prompt == {"fstring", "long_context_reorder", "window_replacement"}


# ── RAGO adapter round-trip ─────────────────────────────────────────────────

def test_rago_mapping_total_and_five_non_identities():
    from rag_stack.cost_model.rago_adapter import TO_RAGO

    assert set(TO_RAGO) == set(TAXONOMY.performance_stages)
    non_identity = {k: v for k, v in TO_RAGO.items() if k != v}
    assert non_identity == {
        "generator_prefill": "prefill",
        "generator_decode": "decode",
        "semantic_retrieval_encode": "encode",
        "semantic_retrieval_vectorsearch": "retrieval",
        "passage_compressor": "passage_compressor_prefill",
    }


def test_rago_round_trip_pruning_pipeline():
    from rag_stack.cost_model.rago_adapter import RagoNameMap, to_rago_groups

    active = ["semantic_retrieval_encode", "semantic_retrieval_vectorsearch",
              "passage_reranker", "passage_compressor",
              "generator_prefill", "generator_decode"]
    m = RagoNameMap(active)
    for name in active:
        assert m.from_rago(m.to_rago(name)) == name
    groups = [["semantic_retrieval_encode", "generator_prefill", "generator_decode"],
              ["semantic_retrieval_vectorsearch"]]
    assert m.from_rago_groups(to_rago_groups(groups)) == groups


def test_rago_round_trip_generative_pipeline():
    from rag_stack.cost_model.rago_adapter import RagoNameMap

    m = RagoNameMap(["passage_compressor_prefill", "passage_compressor_decode",
                     "generator_prefill", "generator_decode"])
    assert m.from_rago("passage_compressor_prefill") == "passage_compressor_prefill"


def test_rago_rejects_mixed_compressor_kinds():
    from rag_stack.cost_model.rago_adapter import RagoNameMap

    with pytest.raises(ValueError, match="pruning and"):
        RagoNameMap(["passage_compressor", "passage_compressor_prefill"])


def test_qualify_result_df_renames_and_guards():
    pd = pytest.importorskip("pandas")
    from rag_stack.cost_model.rago_adapter import RagoNameMap, assert_no_rago_names

    m = RagoNameMap(["semantic_retrieval_encode", "semantic_retrieval_vectorsearch",
                     "generator_prefill", "generator_decode", "passage_reranker"])
    df = pd.DataFrame([{
        "qps": 10.0, "qps_per_chip": 2.5, "num_chips_e2e": 4,
        "batch_size_request": 32, "batch_size_decode": 32,
        "num_chips_prefill": 2, "num_chips_decode": 2,
        "qps_encode": 100.0, "qps_retrieval": 55.0,
        "qps_passage_reranker": 70.0,
        "collocation_strategy": "[['encode', 'prefill', 'decode'], ['retrieval']]",
    }])
    out = m.qualify_result_df(df)
    assert "num_chips_generator_prefill" in out.columns
    assert "num_chips_generator_decode" in out.columns
    assert "qps_semantic_retrieval_encode" in out.columns
    assert "qps_semantic_retrieval_vectorsearch" in out.columns
    assert "batch_size_generator_decode" in out.columns  # RAGO decode-stage batch column
    assert "qps_passage_reranker" in out.columns          # identity name untouched
    assert out["batch_size_request"].iloc[0] == 32        # role bucket untouched
    assert "'generator_prefill'" in out["collocation_strategy"].iloc[0]
    assert "'semantic_retrieval_vectorsearch'" in out["collocation_strategy"].iloc[0]
    assert_no_rago_names(out)
    with pytest.raises(AssertionError):
        assert_no_rago_names(df)
