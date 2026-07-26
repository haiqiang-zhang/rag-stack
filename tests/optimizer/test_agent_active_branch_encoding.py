"""Agent regressions for nested configs with colliding sibling leaves.

The search space is intentionally tiny and self-contained: both component
branches expose a runtime ``temperature`` leaf, but Ax stores them in distinct
qualified dimensions.
"""

import sys
from types import ModuleType, SimpleNamespace

import torch
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.optimizer.agent.core import AgentGuidedQNEHVI
from rag_stack.optimizer.agent.llm_agent import LLMAgent
from rag_stack.search_space import AxSearchSpaceOps


A_TEMPERATURE = "stage__a__temperature"
B_TEMPERATURE = "stage__b__temperature"
COMPONENT = "stage__component"


def _choice(name, parameter_type, values, *, ordered=False, dependents=None):
    return ChoiceParameter(
        name=name,
        parameter_type=parameter_type,
        values=values,
        is_ordered=ordered,
        sort_values=bool(ordered and parameter_type.is_numeric),
        dependents=dependents,
    )


def _agent(*, enable_doe=True):
    search_space = SearchSpace(parameters=[
        _choice(
            A_TEMPERATURE,
            ParameterType.FLOAT,
            [0.1, 0.2],
            ordered=True,
        ),
        _choice(
            B_TEMPERATURE,
            ParameterType.FLOAT,
            [0.6, 0.7],
            ordered=True,
        ),
        _choice(
            COMPONENT,
            ParameterType.STRING,
            ["a", "b"],
            dependents={
                "a": [A_TEMPERATURE],
                "b": [B_TEMPERATURE],
            },
        ),
    ])
    ssops = AxSearchSpaceOps(search_space)
    optimizer = AgentGuidedQNEHVI(
        search_space,
        objectives=[
            {"objective_name": "performance", "cost": 1, "ref_point": 0},
            {"objective_name": "quality", "cost": 1, "ref_point": 0},
        ],
        seed=7,
        n_both_init=2,
        init_sobol_k=2,
        ssops=ssops,
        enable_doe=enable_doe,
    )
    return optimizer, ssops


def _active_row(ssops, row):
    decoded = ssops.decode_flat(row.reshape(-1))
    active = ssops.active_params(decoded)
    return {name: decoded[name] for name in active}


def test_agent_prompt_and_ledger_always_hide_inactive_sibling_children():
    _, ssops = _agent(enable_doe=False)
    renderer = LLMAgent(model="unused", ai_client=object())
    history = [
        {
            "flat": {
                COMPONENT: "a",
                A_TEMPERATURE: 0.1,
                B_TEMPERATURE: "DEAD_B_CHILD",
            },
            "correctness": 0.4,
            "performance": 10.0,
            "sub_metrics": {},
        },
        {
            "flat": {
                COMPONENT: "b",
                A_TEMPERATURE: "DEAD_A_CHILD",
                B_TEMPERATURE: 0.7,
            },
            "correctness": 0.3,
            "performance": 20.0,
            "sub_metrics": {},
        },
    ]

    prompt = "\n".join(
        message["content"]
        for message in renderer._build_messages(
            history, ssops, mo={"pareto": []},
        )
    )
    detail = renderer._run_ledger_queries(
        ["detail e00 e01"], history, ssops,
    )

    for dead_value in ("DEAD_A_CHILD", "DEAD_B_CHILD"):
        assert dead_value not in prompt
        assert dead_value not in detail


def test_agent_history_and_infeasible_ingress_keep_selected_sibling_leaf():
    optimizer, ssops = _agent()
    config = {"stage": {"component": "b", "temperature": 0.7}}
    expected = {COMPONENT: "b", B_TEMPERATURE: 0.7}

    assert _active_row(ssops, optimizer._encode_nested_config(config)) == expected

    optimizer.add_objectives(
        [config],
        {
            "performance": [12.0],
            "quality": [{"combined": 0.4}],
        },
        eval_ids=[3],
    )
    assert _active_row(ssops, optimizer._x_rows[0]) == expected

    optimizer.register_infeasible(config, "unit-test")
    assert _active_row(ssops, optimizer._x_bad[0]) == expected


def test_agent_init_and_dls_sobol_route_nested_rows_through_active_encoder(
    monkeypatch,
):
    arm_parameters = [
        {COMPONENT: "a", A_TEMPERATURE: 0.1},
        {COMPONENT: "a", A_TEMPERATURE: 0.2},
        {COMPONENT: "b", B_TEMPERATURE: 0.6},
        {COMPONENT: "b", B_TEMPERATURE: 0.7},
    ]

    class _InitClient:
        experiment = object()

        def __init__(self):
            self.cursor = 0

        def get_next_trial(self):
            params = arm_parameters[self.cursor]
            self.cursor += 1
            return params, self.cursor

        def abandon_trial(self, trial_index):
            del trial_index

    optimizer, ssops = _agent()
    monkeypatch.setattr(optimizer, "_get_init_sobol_client", _InitClient)
    original = optimizer._encode_nested_config
    encoded = []

    def _record(config):
        row = original(config)
        encoded.append(_active_row(ssops, row))
        return row

    monkeypatch.setattr(optimizer, "_encode_nested_config", _record)

    init_configs = optimizer._sobol_batch(4)
    assert len(init_configs) == 4
    assert len(encoded) >= 4
    assert all(COMPONENT in row for row in encoded)
    assert all(
        (A_TEMPERATURE in row) != (B_TEMPERATURE in row)
        for row in encoded
    )

    # The DLS breadth method imports Ax's registry inside the call.  Substitute
    # a deterministic in-memory generator so this remains a CPU-only unit test
    # and exercises only the nested-config ingress under test.
    class _Arm:
        def __init__(self, parameters):
            self.parameters = parameters

    class _Sobol:
        def gen(self, n):
            return SimpleNamespace(
                arms=[_Arm(params) for params in arm_parameters[:n]],
            )

    class _Generators:
        @staticmethod
        def SOBOL(**kwargs):
            del kwargs
            return _Sobol()

    registry = ModuleType("ax.adapter.registry")
    registry.Generators = _Generators
    monkeypatch.setitem(sys.modules, "ax.adapter.registry", registry)

    optimizer, ssops = _agent()
    optimizer._pool_sobol_ss = SimpleNamespace(search_space=ssops.ss)
    original = optimizer._encode_nested_config
    encoded = []

    def _record_pool(config):
        row = original(config)
        encoded.append(_active_row(ssops, row))
        return row

    monkeypatch.setattr(optimizer, "_encode_nested_config", _record_pool)
    pool = optimizer._hier_pool_sobol(4)
    assert pool is not None and pool.shape[0] > 0
    assert encoded
    assert all(COMPONENT in row for row in encoded)
    assert all(
        (A_TEMPERATURE in row) != (B_TEMPERATURE in row)
        for row in encoded
    )


def test_dls_acquisition_canonicalizes_inactive_dims_on_every_call():
    optimizer, ssops = _agent()
    a_row = optimizer._encode_nested_config({
        "stage": {"component": "a", "temperature": 0.1},
    }).reshape(1, 1, -1)
    a_idx = ssops.param_names.index(A_TEMPERATURE)
    b_idx = ssops.param_names.index(B_TEMPERATURE)
    a_row[..., b_idx] = 1.0  # deliberately non-canonical dead sibling value
    b_row = optimizer._encode_nested_config({
        "stage": {"component": "b", "temperature": 0.6},
    }).reshape(1, 1, -1)
    b_row[..., a_idx] = 1.0  # dead sibling must return to the midpoint
    b_row[..., b_idx] = 0.5  # newly active child is not on its 2-level grid
    seen = []

    def probe(X):
        seen.append(X.detach().clone())
        return X.sum(dim=(-1, -2))

    wrapped = optimizer._hierarchy_safe_acqf(probe)
    wrapped(a_row)
    wrapped(b_row)

    assert len(seen) == 2
    assert seen[0].shape == a_row.shape
    assert float(seen[0][0, 0, b_idx]) == 0.5
    assert float(seen[1][0, 0, a_idx]) == 0.5
    assert float(seen[1][0, 0, b_idx]) in ssops.choice_grid(B_TEMPERATURE)
    assert float(seen[1][0, 0, b_idx]) != 0.5


def test_dls_pool_restarts_exclude_evaluated_rows(monkeypatch):
    optimizer, ssops = _agent(enable_doe=False)
    evaluated = {"stage": {"component": "a", "temperature": 0.1}}
    novel = {"stage": {"component": "a", "temperature": 0.2}}
    optimizer.add_objectives(
        [evaluated],
        {"performance": [10.0], "quality": [{"combined": 0.2}]},
        eval_ids=[0],
    )
    evaluated_row = optimizer._encode_nested_config(evaluated).reshape(1, -1)
    novel_row = optimizer._encode_nested_config(novel).reshape(1, -1)
    pool = torch.cat([evaluated_row, novel_row], dim=0)
    captured = {}

    monkeypatch.setattr(optimizer, "_build_acqf", lambda: (
        lambda X: torch.arange(
            X.shape[0], 0, -1, dtype=torch.double,
        )
    ))
    monkeypatch.setattr(optimizer, "_hier_pool_sobol", lambda n: pool)

    def capture_grouped(_acqf, _advice, *, pool_seeds=None, **_kwargs):
        captured["pool_seeds"] = pool_seeds
        return None

    monkeypatch.setattr(optimizer, "_dls_candidates_grouped", capture_grouped)

    config = optimizer._propose_pool(
        None, forced_due=False, forced_gap=None,
    )

    assert config == novel
    assert captured["pool_seeds"].shape[0] == 1
    assert not ssops.is_duplicate_decoded(
        captured["pool_seeds"][0].reshape(1, -1),
        torch.stack(optimizer._x_rows),
    )
