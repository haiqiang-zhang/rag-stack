"""Self-contained checkpoint identity regressions for hierarchical configs."""

from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.controller import (
    _parse_checkpoint_csv_cell,
    _restore_checkpoint_selector_choices,
)
from rag_stack.optimizer.greedy import GreedyParetoOptimizer
from rag_stack.search_space import AxSearchSpaceOps, RagSearchSpace


SELECTOR = (
    "pipeline__sequential__semantic_retrieval__vectordb__vectordb"
)
EFS = (
    "pipeline__sequential__semantic_retrieval__vectordb__hnsw__ef_search"
)
NPROBE = (
    "pipeline__sequential__semantic_retrieval__vectordb__ivf__nprobe"
)
M = "vectordb__hnsw__M"
EFC = "vectordb__hnsw__ef_construction"
EMBED = "vectordb__hnsw__embedding_model"
TEMPERATURE = "pipeline__sequential__generator__vllm__temperature"


def _choice(name, parameter_type, values, *, ordered=False, dependents=None):
    return ChoiceParameter(
        name=name,
        parameter_type=parameter_type,
        values=values,
        is_ordered=ordered,
        sort_values=bool(ordered and parameter_type.is_numeric),
        dependents=dependents,
    )


def _hnsw_space() -> RagSearchSpace:
    parameters = [
        _choice(EFS, ParameterType.INT, [32, 64], ordered=True),
        _choice(NPROBE, ParameterType.INT, [8, 16], ordered=True),
        _choice(M, ParameterType.INT, [16, 32, 48], ordered=True),
        _choice(EFC, ParameterType.INT, [100, 400], ordered=True),
        _choice(
            EMBED,
            ParameterType.STRING,
            ["small", "bge-m3"],
        ),
        _choice(
            SELECTOR,
            ParameterType.STRING,
            ["ivf", "hnsw"],
            dependents={
                "ivf": [NPROBE],
                "hnsw": [EFS, M, EFC, EMBED],
            },
        ),
        _choice(
            TEMPERATURE,
            ParameterType.FLOAT,
            [0.0, 0.7],
            ordered=True,
        ),
    ]
    return RagSearchSpace(SearchSpace(parameters=parameters))


def _decoded_hnsw(m, efc, embedding, temperature=0.0):
    return {
        "pipeline": {
            "sequential": {
                "semantic_retrieval": {
                    "component": "vectordb",
                    "vectordb": "hnsw",
                    "ef_search": 32,
                },
                "generator": {
                    "component": "vllm",
                    "temperature": temperature,
                },
            },
        },
        "hnsw": {
            "M": m,
            "ef_construction": efc,
            "embedding_model": embedding,
        },
    }


def _legacy_csv_roundtrip(ssops, decoded):
    legacy = {
        **decoded,
        "pipeline": {
            "sequential": {
                **decoded["pipeline"]["sequential"],
                "semantic_retrieval": {
                    **decoded["pipeline"]["sequential"]["semantic_retrieval"],
                    "vectordb": {
                        "ivf": {"nprobe": [8, 16]},
                        "hnsw": {"ef_search": [32, 64]},
                    },
                },
            },
        },
    }
    csv_cells = {
        key: str(value)
        for key, value in ssops.flatten_2deep(legacy).items()
    }
    parsed = {
        key: _parse_checkpoint_csv_cell(value)
        for key, value in csv_cells.items()
    }
    nested = ssops.decode_to_nested(parsed)
    runtime = {
        "pipeline_runtime": {"mode": "sequential"},
        "node_lines": [
            {
                "nodes": [
                    {
                        "stage": "semantic_retrieval",
                        "modules": [
                            {
                                "component": "vectordb",
                                "vectordb": "hnsw",
                                "ef_search": 32,
                            },
                        ],
                    },
                    {
                        "stage": "generator",
                        "modules": [
                            {
                                "component": "vllm",
                                "temperature": decoded["pipeline"]
                                ["sequential"]["generator"]["temperature"],
                            },
                        ],
                    },
                ],
            },
        ],
    }
    return _restore_checkpoint_selector_choices(
        nested,
        runtime,
        ssops.hierarchical_selector_paths(),
    )


def test_four_hnsw_variants_survive_legacy_csv_roundtrip_as_four_keys():
    ssops = _hnsw_space()
    variants = [
        (16, 100, "small"),
        (48, 100, "small"),
        (32, 400, "small"),
        (32, 100, "bge-m3"),
    ]
    active = [
        ssops.active_flat_config(
            _legacy_csv_roundtrip(ssops, _decoded_hnsw(*variant))
        )
        for variant in variants
    ]

    assert len({tuple(sorted(row.items())) for row in active}) == 4
    for row, (m, efc, embedding) in zip(active, variants):
        assert row[SELECTOR] == "hnsw"
        assert row[M] == m
        assert row[EFC] == efc
        assert row[EMBED] == embedding
        assert NPROBE not in row


def test_greedy_warm_start_keeps_hnsw_variants_and_true_pareto():
    ssops = _hnsw_space()
    rows = [
        (6, _decoded_hnsw(16, 100, "small"), 105.72634427801675, 0.288),
        (8, _decoded_hnsw(48, 100, "small"), 99.2221, 0.267),
        (9, _decoded_hnsw(32, 400, "small"), 98.356588, 0.262),
        (10, _decoded_hnsw(32, 100, "bge-m3"), 82.325924, 0.304),
        (22, _decoded_hnsw(16, 100, "small", 0.7), 90.44797571987391, 0.286),
    ]
    evaluations = [
        {
            "eval_id": eval_id,
            "config": _legacy_csv_roundtrip(ssops, config),
            "objectives": {
                "performance": performance,
                "quality": {"combined": quality},
            },
        }
        for eval_id, config, performance, quality in rows
    ]
    optimizer = GreedyParetoOptimizer(
        ssops.ss,
        [
            {"objective_name": "performance", "cost": 1.0, "ref_point": 0.0},
            {"objective_name": "quality", "cost": 1.0, "ref_point": 0.0},
        ],
        ssops=ssops,
        search_mode="forward",
        n_iterations=5,
    )

    optimizer.warm_start(evaluations)
    assert len(optimizer._replay_queue) == 5
    optimizer._flush_replay("unit-test")
    _, _, pareto_eval_ids = optimizer.get_pareto()
    assert 6 in pareto_eval_ids
    assert 22 not in pareto_eval_ids


def test_component_qualified_leaf_is_not_encoded_into_sibling_branch():
    a_temp = "stage__a__temperature"
    b_temp = "stage__b__temperature"
    parent = "stage__component"
    ssops = AxSearchSpaceOps(SearchSpace(parameters=[
        _choice(a_temp, ParameterType.FLOAT, [0.1, 0.2], ordered=True),
        _choice(b_temp, ParameterType.FLOAT, [0.6, 0.7], ordered=True),
        _choice(
            parent,
            ParameterType.STRING,
            ["a", "b"],
            dependents={"a": [a_temp], "b": [b_temp]},
        ),
    ]))

    active = ssops.active_flat_config({
        "stage": {"component": "b", "temperature": 0.7},
    })
    assert active == {parent: "b", b_temp: 0.7}
