"""Self-contained structured-init identity and Sobol-fill regressions."""

from __future__ import annotations

import json

from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.optimizer.agent.core import AgentGuidedQNEHVI
from rag_stack.optimizer.agent.llm_agent import LLMAgent
from rag_stack.search_space import AxSearchSpaceOps


def _choice(name, parameter_type, values, *, dependents=None):
    return ChoiceParameter(
        name=name,
        parameter_type=parameter_type,
        values=values,
        is_ordered=False,
        dependents=dependents,
    )


def _optimizer(ai_client) -> AgentGuidedQNEHVI:
    search_space = SearchSpace(parameters=[
        _choice("a_knob", ParameterType.INT, [0, 1]),
        _choice("b_knob", ParameterType.INT, [0, 1]),
        _choice(
            "branch",
            ParameterType.STRING,
            ["a", "b"],
            dependents={"a": ["a_knob"], "b": ["b_knob"]},
        ),
    ])
    ssops = AxSearchSpaceOps(search_space)
    optimizer = AgentGuidedQNEHVI(
        search_space,
        objectives=[
            {"objective_name": "performance", "cost": 1, "ref_point": 0},
            {"objective_name": "quality", "cost": 1, "ref_point": 0},
        ],
        seed=43,
        n_both_init=6,
        init_sobol_k=0,
        ssops=ssops,
        n_candidates=8,
    )
    optimizer.agent = LLMAgent(model="unused", ai_client=ai_client)
    return optimizer


def _identity(opt, config):
    _, x = opt._canonical_init_config(config)
    decoded = opt.ssops.decode_flat(x.reshape(-1))
    active = opt.ssops.active_params(decoded)
    return tuple((name, decoded[name]) for name in sorted(active))


def test_llm_init_dedups_canonical_identity_and_sobol_fills_pending_batch(
    monkeypatch,
):
    # Full-config proposals may not smuggle inactive sibling values through a
    # hierarchical branch.  Duplicate only valid active parameterizations;
    # structured init must canonical-dedup them before serving seats.
    proposals = [
        {"branch": "a", "a_knob": 0},
        {"branch": "a", "a_knob": 0},
        {"branch": "a", "a_knob": 1},
        {"branch": "a", "a_knob": 1},
        {"branch": "b", "b_knob": 0},
    ]

    class _AgentClient:
        async def chat(self, messages, **kwargs):
            del messages, kwargs
            return json.dumps({"candidates": proposals})

    opt = _optimizer(_AgentClient())
    history_x = opt.ssops.clamp(
        opt.ssops.encode_arm_params(
            {"branch": "a", "a_knob": 0},
        ).reshape(1, -1)
    ).reshape(-1)
    opt._x_rows = [history_x]
    opt._configs = [opt.ssops.decode_to_config(history_x)]
    opt._perf = [10.0]
    opt._corr = [0.2]

    plan = opt._llm_probe_plan(5)
    assert plan == [
        {"a_knob": 0, "branch": "a"},
        {"a_knob": 0, "branch": "a"},
        {"a_knob": 1, "branch": "a"},
        {"a_knob": 1, "branch": "a"},
        {"b_knob": 0, "branch": "b"},
    ]

    # The hierarchy-aware Sobol source first repeats history and both accepted
    # LLM identities, then supplies the one remaining branch-b configuration.
    arms = [
        {"branch": "a", "a_knob": 0},
        {"branch": "a", "a_knob": 1},
        {"branch": "b", "b_knob": 0},
        {"branch": "b", "b_knob": 1},
    ]

    class _InitClient:
        def __init__(self):
            self.cursor = 0

        def get_next_trial(self):
            params = arms[self.cursor]
            self.cursor += 1
            return params, self.cursor

        def abandon_trial(self, trial_index):
            del trial_index

    init_client = _InitClient()
    monkeypatch.setattr(opt, "_get_init_sobol_client", lambda: init_client)

    batch = opt._structured_init_batch(3)
    assert batch == [
        {"a_knob": 1, "branch": "a"},
        {"b_knob": 0, "branch": "b"},
        {"b_knob": 1, "branch": "b"},
    ]
    identities = [_identity(opt, config) for config in batch]
    assert len(set(identities)) == len(batch)
    assert _identity(opt, opt._configs[0]) not in identities
    assert init_client.cursor == 4, "Sobol duplicates must be skipped, not served"
    assert all(
        ("a_knob" in config) != ("b_knob" in config)
        for config in batch
    ), "served configs must contain only the selected branch's active knobs"
