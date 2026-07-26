from types import SimpleNamespace

import pandas as pd
import pytest


class _WhitespaceTokenizer:
    def encode(self, text, *args, **kwargs):
        return str(text).split()


def _node(stage, **module_param):
    return SimpleNamespace(
        stage=stage,
        module=SimpleNamespace(module_param=module_param),
    )


def test_static_execution_dag_uses_real_dataframe_tokens():
    from rag_stack_evaluator.static_rag_evaluator.trace_builder import (
        build_static_execution_dag,
    )

    df = pd.DataFrame({
        "query": ["foo bar"],
        "queries": [["foo bar", "baz"]],
        "retrieved_contents": [["doc one", "doc two three"]],
        "prompts": ["prompt alpha beta"],
        "generated_tokens": [[1, 2, 3, 4]],
        "generated_texts": ["unused fallback"],
    })
    node_lines = {
        "line": [
            _node("query_expansion", model="qe-model"),
            _node("semantic_retrieval", embedding_model="embedder"),
            _node("passage_reranker", model_name="rr-model"),
            _node("generator", model="gen-model"),
        ]
    }

    dag = build_static_execution_dag(
        df,
        node_lines,
        config={},
        token_stats=SimpleNamespace(_tokenizer=_WhitespaceTokenizer()),
    )

    assert dag["shape"] == "aggregate"
    assert dag["rag_ir_mode"] == "dynamic"
    assert dag["query_expansion_fanout"] == 2.0
    assert dag["token_stats"]["avg_query_tokens"] == 2
    assert dag["token_stats"]["avg_retrieved_context_tokens"] == 5
    assert dag["token_stats"]["avg_generator_input_tokens"] == 3
    assert dag["token_stats"]["avg_generator_output_tokens"] == 4
    assert dag["token_stats"]["avg_query_expansion_input_tokens"] == 2
    assert dag["token_stats"]["avg_query_expansion_output_tokens"] == 1
    assert dag["token_stats"]["avg_passage_reranker_input_tokens"] == 9
    # Retrieval is always the encode + vector-search call pair, one pair per
    # expanded sub-query (fanout 2 here).
    assert [c["stage"] for c in dag["trace_v1"][0]] == [
        "query_expansion",
        "semantic_retrieval_encode",
        "semantic_retrieval_vectorsearch",
        "semantic_retrieval_encode",
        "semantic_retrieval_vectorsearch",
        "passage_reranker",
        "generator",
    ]


def test_token_stats_runtime_overrides_take_precedence():
    from rag_stack.cost_model.token_stats import TokenStats

    ts = TokenStats.from_dict({
        "avg_chunk_tokens": 10,
        "prompt_template_tokens": 5,
        "avg_output_tokens": 4,
        "avg_query_tokens": 2,
        "avg_llm_input_tokens": 123,
        "avg_query_expansion_input_tokens": 7,
        "avg_query_expansion_output_tokens": 8,
        "avg_passage_reranker_input_tokens": 9,
    })

    assert ts.get_llm_input_tokens(top_k=3) == 123
    assert ts.get_query_expansion_input_tokens() == 7
    assert ts.get_query_expansion_output_tokens("hyde") == 8
    assert ts.get_passage_reranker_input_tokens() == 9


def test_static_assembly_injects_rag_ir_token_overrides_without_mutation():
    pytest.importorskip("GenZ")
    from rag_stack.cost_model.sweep_inputs import (
        _resolve_main_dec_steps,
        _with_rag_ir_token_overrides,
    )

    cfg = {
        "_token_stats": {
            "avg_chunk_tokens": 10,
            "prompt_template_tokens": 5,
            "avg_output_tokens": 4,
            "avg_query_tokens": 2,
        }
    }
    dag = {
        "shape": "aggregate",
        "rag_ir_mode": "dynamic",
        "token_stats": {
            "avg_generator_input_tokens": 99,
            "avg_generator_output_tokens": 11,
        },
    }

    out = _with_rag_ir_token_overrides(cfg, dag)

    assert out is not cfg
    assert cfg["_token_stats"]["avg_output_tokens"] == 4
    assert out["_token_stats"]["avg_llm_input_tokens"] == 99
    assert out["_token_stats"]["avg_output_tokens"] == 11
    assert _resolve_main_dec_steps({"dec_steps": 4}, dag) == 11


def test_rag_ir_mode_default_dynamic_and_explicit_static():
    from rag_stack.performance_context import stamp_rag_ir_mode

    # Default (key omitted) is dynamic.
    config = {
        "global": {"eval_backend": "static_gt"},
        "optimizer": {"params": {}},
    }
    pc = {}
    stamp_rag_ir_mode(config, pc)
    assert pc["_runtime"]["rag_ir_mode"] == "dynamic"

    # Static mode remains available explicitly.
    config["global"]["rag_ir_mode"] = "static"
    pc = {}
    stamp_rag_ir_mode(config, pc)
    assert pc["_runtime"]["rag_ir_mode"] == "static"

    # 'auto' was removed; unknown values raise.
    config["global"]["rag_ir_mode"] = "auto"
    with pytest.raises(ValueError, match="not one of"):
        stamp_rag_ir_mode(config, {})

    # The legacy key is a hard error with rename guidance.
    config["global"].pop("rag_ir_mode")
    config["global"]["cm_data_source"] = "rag_ir"
    with pytest.raises(ValueError, match="renamed to `rag_ir_mode`"):
        stamp_rag_ir_mode(config, {})
