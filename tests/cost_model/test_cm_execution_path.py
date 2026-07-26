"""The compiler preserves evaluator-declared calls without inventing operations."""

from copy import deepcopy

import pytest

from rag_stack.rag_ir.execution_path import (
    compile_execution_paths,
    query_expansion_fanout,
)


def _call(stage, index=0, **fields):
    return {
        "stage": stage, "step_idx": index,
        "input_tokens": 12, "output_tokens": 0, **fields,
    }


def test_expansion_fanout_never_invents_fusion_calls():
    stages = ["query_expansion", "semantic_retrieval_encode",
              "semantic_retrieval_vectorsearch", "semantic_retrieval_encode",
              "semantic_retrieval_vectorsearch", "generator"]
    traces = [[_call(stage, index) for index, stage in enumerate(stages)]]
    original = deepcopy(traces)
    for fanout in (1, 2, 4, 17):
        compiled = compile_execution_paths(traces, retrieval_fanout=fanout)
        assert compiled == original
        assert all(call["stage"] != "retrieval_fusion" for call in compiled[0])
    compiled[0][0]["input_tokens"] = 100
    assert traces == original


def test_observed_profiles_are_preserved_and_deep_copied():
    trace = [[
        _call("semantic_retrieval_encode", 0),
        _call("semantic_retrieval_vectorsearch", 1),
        _call("retrieval_fusion", 2, model_id="rrf_pure",
              input_profile={"items": 8, "list_sizes": [3, 5], "unique_items": 6},
              output_profile={"items": 4, "unique_items": 4}),
        _call("generator", 3),
    ]]
    original = deepcopy(trace)
    once = compile_execution_paths(trace, retrieval_fanout=1)
    assert once == original
    assert compile_execution_paths(once, retrieval_fanout=7) == once
    once[0][2]["input_profile"]["list_sizes"][0] = 99
    assert trace == original


def test_ir_preserves_hybrid_while_cm_rejects_missing_lexical_model():
    trace = [[_call(stage, index) for index, stage in enumerate([
        "semantic_retrieval_encode", "semantic_retrieval_vectorsearch",
        "lexical_retrieval", "retrieval_fusion", "generator",
    ])]]
    assert compile_execution_paths(trace, retrieval_fanout=2) == trace
    from rag_stack.cost_model.support import require_cm_trace_support
    with pytest.raises(NotImplementedError, match="BM25 analytical cost"):
        require_cm_trace_support(trace)


def test_compiler_rejects_invalid_observed_profiles():
    trace = [[_call("generator", input_profile={"items": 4, "list_sizes": [2, 3]})]]
    with pytest.raises(ValueError):
        compile_execution_paths(trace)


def test_single_list_and_non_dense_calls_do_not_gain_fusion():
    traces = [[_call("semantic_retrieval_vectorsearch", 7)],
              [_call("lexical_retrieval", 9)], []]
    compiled = compile_execution_paths(traces)
    assert [len(path) for path in compiled] == [1, 1, 0]
    assert all(call["step_idx"] == 0 for path in compiled for call in path)
    assert compile_execution_paths([traces[1]], retrieval_fanout=5) == [compiled[1]]


@pytest.mark.parametrize("fanout", [0, -1, True, 1.5, "2", None])
def test_invalid_fanout_fails_before_lowering(fanout):
    with pytest.raises(ValueError, match="positive integer"):
        compile_execution_paths([], retrieval_fanout=fanout)


def test_query_wrapper_is_accepted_without_mutating_identity_or_calls():
    wrapped = [{"question_id": "q7", "calls": [
        _call("semantic_retrieval_vectorsearch"), _call("generator"),
    ]}]
    original = deepcopy(wrapped)
    compiled = compile_execution_paths(wrapped, retrieval_fanout=2)
    assert wrapped == original
    assert len(compiled[0]) == 2


@pytest.mark.parametrize("component, expected", [
    ("hyde", 1), ("query_decompose", 2), ("multi_query_expansion", 4),
    ("custom", 1),
])
def test_fanout_preserves_existing_defaults_and_runtime_precedence(component, expected):
    config = {"node_lines": [{"nodes": [{
        "stage": "query_expansion", "modules": [{"component": component}],
    }]}]}
    assert query_expansion_fanout(config) == expected
    assert query_expansion_fanout(config, {
        "shape": "aggregate", "query_expansion_fanout": 2.1,
        "retrieval_invocations_per_query": 7,
    }) == 3
    assert query_expansion_fanout(config, {"query_expansion_fanout": 9}) == expected
    assert query_expansion_fanout({}) == 1
