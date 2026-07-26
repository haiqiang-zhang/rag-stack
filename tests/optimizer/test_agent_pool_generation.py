"""Self-contained regressions for hierarchy-aware Agent pool generation."""
import logging

import pytest
import torch

from rag_stack.search_space.rag_search_space import RagSearchSpace


def _self_contained_space() -> RagSearchSpace:
    nodes = [
        {
            "stage": "query_expansion",
            "optional": True,
            "modules": [{"component": "query_decompose", "fanout": [1, 2]}],
        },
        {
            "stage": "semantic_retrieval",
            "top_k": [2, 4, 8],
            "modules": [{"component": "vectordb"}],
        },
        {
            "stage": "passage_reranker",
            "optional": True,
            "modules": [
                {"component": "reranker-a"},
                {"component": "reranker-b"},
            ],
        },
        {
            "stage": "generator",
            "modules": [{
                "component": "vllm",
                "model": {"ordered": ["model-1", "model-2", "model-3"]},
                "temperature": [0.0, 0.5],
            }],
        },
    ]
    return RagSearchSpace.from_yaml({
        "algo_search_space": {
            "pipeline": {
                "rag_dataflow": [
                    "sequential",
                    {"react": {"max_iter": [3, 5]}},
                ],
                "node_lines": [{"node_line_name": "main", "nodes": nodes}],
            },
        },
    })


def _build_agent(**kwargs):
    from rag_stack.optimizer import OPTIMIZER_REGISTRY

    space = _self_contained_space()
    objectives = [
        {"objective_name": "performance", "cost": 1, "ref_point": 0},
        {"objective_name": "quality", "cost": 10, "ref_point": 0},
    ]
    opt = OPTIMIZER_REGISTRY["agent_qnehvi"](
        space.ss, objectives=objectives, seed=46, n_both_init=4,
        ssops=space,
        n_candidates=32,
        **kwargs,
    )
    return opt


@pytest.fixture(scope="module")
def agent():
    opt = _build_agent()
    xs, _ = opt.suggest(4)
    opt.add_objectives(
        xs,
        {"performance": [10.0, 40.0, 5.0, 90.0],
         "quality": [{"combined": 0.2}, {"combined": 0.3},
                     {"combined": 0.5}, {"combined": 0.15}]},
        eval_ids=[0, 1, 2, 3],
    )
    opt._fit_models()   # _propose_pool is called directly in these tests
    return opt


# ------------------------------------------------- (1) hierarchy-aware pool

def test_hier_pool_sobol_decodable_and_deterministic(agent):
    a = agent._hier_pool_sobol(16)
    b = agent._hier_pool_sobol(16)
    assert a is not None and a.shape[0] >= 8
    assert torch.equal(a, b), "same t + seed must reproduce the same draws"
    for i in range(a.shape[0]):
        assert agent.ssops.decode_to_config(a[i].reshape(-1))


def test_hier_pool_fallback_to_raw(agent, caplog):
    orig = agent._get_init_sobol_client
    agent._pool_sobol_ss = None
    agent._get_init_sobol_client = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        with caplog.at_level(logging.WARNING):
            out = agent._hier_pool_sobol(8)
    finally:
        agent._get_init_sobol_client = orig
        agent._pool_sobol_ss = None
    assert out is None
    assert any("hier sobol failed" in r.message for r in caplog.records)


def test_pool_composition_logged(agent, caplog):
    with caplog.at_level(logging.INFO):
        agent._propose_pool(None, forced_due=False, forced_gap=None)
    pool_lines = [r.message for r in caplog.records if r.message.startswith("[pool t=")]
    assert pool_lines, "pool composition line must be emitted every round"
    assert "sobol=" in pool_lines[-1] and "farfield" not in pool_lines[-1]


def test_enabled_doe_failure_does_not_silently_change_the_method(
    agent, monkeypatch,
):
    def fail():
        raise RuntimeError("broken DOE")

    monkeypatch.setattr(agent, "_doe_round_plan", fail)

    with pytest.raises(RuntimeError, match="broken DOE"):
        agent._propose_pool(None, forced_due=False, forced_gap=None)
