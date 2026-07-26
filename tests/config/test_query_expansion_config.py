"""Search-space scope stays explicit as CM capabilities evolve."""

from __future__ import annotations

from copy import deepcopy

import pytest

from rag_stack.cost_model.sweep_inputs import (
    build_performance_stage_cost_models,
)
from rag_stack.search_space.rag_search_space import (
    RagSearchSpace,
    pipeline_method_blocks,
)


def _config(stage: str, component: str) -> dict:
    return {
        "algo_search_space": {
            "pipeline": {
                "node_lines": [{
                    "node_line_name": "main",
                    "nodes": [{
                        "stage": stage,
                        "modules": [{"component": component}],
                    }],
                }],
            },
        },
    }


@pytest.mark.parametrize(
    ("stage", "component"),
    [
        ("passage_augmenter", "vectordb"),
    ],
)
def test_search_space_rejects_paper_exclusions(stage, component):
    with pytest.raises(NotImplementedError, match="current paper search space"):
        RagSearchSpace.build_ax_searchspace(_config(stage, component))


def test_prompt_maker_is_not_rejected_by_paper_scope_gate():
    search_space = RagSearchSpace.build_ax_searchspace(
        _config("prompt_maker", "fstring")
    )
    assert search_space is not None


def test_search_space_keeps_multi_list_query_expansion_arms():
    config = _config("query_expansion", "query_decompose")
    modules = config["algo_search_space"]["pipeline"]["node_lines"][0][
        "nodes"
    ][0]["modules"]
    modules.extend([
        {"component": "hyde", "temperature": [0.0, 0.5]},
        {"component": "multi_query_expansion"},
    ])
    original = deepcopy(config)

    search_space = RagSearchSpace.build_ax_searchspace(config)
    parsed_modules = pipeline_method_blocks(config)[0]["node_lines"][0][
        "nodes"
    ][0]["modules"]

    assert [module["component"] for module in parsed_modules] == [
        "query_decompose", "hyde", "multi_query_expansion",
    ]
    component_axis = search_space.parameters[
        "pipeline__sequential__query_expansion__component"
    ]
    assert component_axis.values == [
        "query_decompose", "hyde", "multi_query_expansion",
    ]
    assert config == original


@pytest.mark.parametrize(
    "component",
    ["query_decompose", "multi_query_expansion"],
)
def test_search_space_accepts_query_expansion_when_only_one_fusion_arm_remains(
    component,
):
    search_space = RagSearchSpace.build_ax_searchspace(
        _config("query_expansion", component)
    )
    assert "pipeline__sequential__query_expansion__component" not in (
        search_space.parameters
    )


def test_manual_query_expansion_fusion_override_is_resolved():
    config = _config("query_expansion", "hyde")
    config["algo_search_space"]["pipeline"]["node_lines"][0]["nodes"][0][
        "modules"
    ].append({"component": "multi_query_expansion"})
    adapter = RagSearchSpace.from_yaml(config)

    _, node_lines = adapter.resolve_pipeline_block(config, {
        "pipeline": {
            "sequential": {
                "query_expansion": {
                    "component": "multi_query_expansion",
                },
            },
        },
    })
    assert node_lines[0]["nodes"][0]["modules"] == [
        {"component": "multi_query_expansion"},
    ]


@pytest.mark.parametrize(
    "component", ["query_decompose", "multi_query_expansion"],
)
def test_component_cm_assembly_supports_multi_list_query_expansion(component):
    pipeline = {
        "node_lines": [{
            "nodes": [{
                "stage": "query_expansion",
                "modules": [{"component": component}],
            }],
        }],
    }
    models = build_performance_stage_cost_models(pipeline)
    assert models
