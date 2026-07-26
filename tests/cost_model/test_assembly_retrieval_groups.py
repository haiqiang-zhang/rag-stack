import pandas as pd

from rag_stack.cost_model.assembly import (
    _normalize_retrieval_resource_groups,
    _rago_strategy_from_resource_groups,
)


def test_retrieval_is_exported_as_cpu_resource_group():
    df = pd.DataFrame(
        [
            {
                "collocation_strategy": "[['semantic_retrieval_encode', 'semantic_retrieval_vectorsearch', 'generator_prefill'], ['generator_decode']]",
            }
        ]
    )

    out = _normalize_retrieval_resource_groups(df)

    assert out["collocation_strategy"].iloc[0] == [
        ["semantic_retrieval_encode", "generator_prefill"],
        ["semantic_retrieval_vectorsearch"],
        ["generator_decode"],
    ]
    assert out["pipeline_stage_order"].iloc[0] == [
        "semantic_retrieval_encode",
        "semantic_retrieval_vectorsearch",
        "generator_prefill",
        "generator_decode",
    ]
    assert "rago_collocation_strategy" not in out.columns


def test_resource_groups_convert_to_ordered_rago_strategy():
    resource_groups = [["semantic_retrieval_encode", "generator_prefill"], ["semantic_retrieval_vectorsearch"], ["generator_decode"]]
    stage_order = ["semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "generator_prefill", "generator_decode"]

    assert _rago_strategy_from_resource_groups(resource_groups, stage_order) == [
        ["semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "generator_prefill"],
        ["generator_decode"],
    ]


def test_retrieval_stays_separate_when_gpu_groups_do_not_match_across_it():
    resource_groups = [["semantic_retrieval_encode"], ["semantic_retrieval_vectorsearch"], ["generator_prefill", "generator_decode"]]
    stage_order = ["semantic_retrieval_encode", "semantic_retrieval_vectorsearch", "generator_prefill", "generator_decode"]

    assert _rago_strategy_from_resource_groups(resource_groups, stage_order) == [
        ["semantic_retrieval_encode"],
        ["semantic_retrieval_vectorsearch"],
        ["generator_prefill", "generator_decode"],
    ]
