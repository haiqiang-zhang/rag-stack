"""Fail-fast semantics for attributable LLMAgent paper benchmarks."""

import json
import time
from types import SimpleNamespace

import anyio
import pytest
import torch
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.optimizer.agent import core as core_module
from rag_stack.optimizer.agent.core import AgentGuidedQNEHVI
from rag_stack.optimizer.agent.llm_agent import (
    AgentAdviceError,
    LLMAgent,
    _extract_json,
)
from rag_stack.optimizer.agent.prompts import PromptFeedMixin
from rag_stack.optimizer.botorch.greedy_then_hv_anchor_agent_qnehvi import (
    AnchorAgentQNEHVI,
    GreedyAnchorLLMAgent,
    HVAnchorLLMAgent,
    HVAnchorQNEHVI,
)
from rag_stack.search_space import AxSearchSpaceOps


_HISTORY = [
    {
        "correctness": 0.5,
        "performance": 10.0,
        "flat": {},
        "sub_metrics": {},
    }
]


def _agent(client, *, strict=False, wall_timeout_s=1.0):
    agent = LLMAgent(
        ai_client=client,
        strict=strict,
        wall_timeout_s=wall_timeout_s,
    )
    agent._build_messages = lambda history, ssops, mo: [
        {"role": "user", "content": "test"}
    ]
    return agent


class _RaisingClient:
    async def chat(self, messages, **kwargs):
        raise ConnectionError("provider unavailable")


class _ReplyClient:
    def __init__(self, *replies):
        self.replies = list(replies)

    async def chat(self, messages, **kwargs):
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return reply


def test_extract_json_treats_braces_inside_strings_as_data() -> None:
    parsed = _extract_json(
        'prefix {not-json}\n'
        '{"reasoning":"do not close } the payload or open { another",'
        '"candidates":[{"mode":"middle"}]}'
    )

    assert parsed == {
        "reasoning": "do not close } the payload or open { another",
        "candidates": [{"mode": "middle"}],
    }


def test_extract_json_prefers_agent_payload_over_unrelated_prefix_object() -> None:
    parsed = _extract_json(
        '{"example": true}\n'
        '{"reasoning":"actual", "candidates":[{"mode":"middle"}]}'
    )

    assert parsed == {
        "reasoning": "actual",
        "candidates": [{"mode": "middle"}],
    }


def test_non_strict_chat_failure_preserves_empty_advice_fallback(monkeypatch):
    monkeypatch.delenv("RAG_STACK_AGENT_STRICT", raising=False)
    advice = _agent(_RaisingClient())._advise_oneshot(
        _HISTORY, object(), 3,
    )
    assert advice.n() == 0


def test_default_wall_timeout_only_changes_strict_runs(monkeypatch):
    monkeypatch.delenv("RAG_STACK_AGENT_STRICT", raising=False)
    monkeypatch.delenv("RAG_STACK_AGENT_WALL_TIMEOUT_S", raising=False)
    assert LLMAgent(ai_client=object()).wall_timeout_s is None
    assert LLMAgent(ai_client=object(), strict=True).wall_timeout_s == 150.0


def test_objectives_card_declares_unobserved_configured_submetrics() -> None:
    agent = LLMAgent(ai_client=object())
    ssops = SimpleNamespace(_config={
        "eval_backend_setting": {
            "submetric_query_subset": 15,
            "combined_quality": {
                "mode": "mean",
                "metrics": ["deepeval_answer_correctness"],
            },
            "metrics": [
                {
                    "metric_name": "deepeval_answer_correctness",
                    "description": "optimized target",
                },
                {
                    "metric_name": "deepeval_faithfulness",
                    "description": "grounding diagnostic",
                },
            ],
        },
        "system": {
            "performance_objectives": {"selection": "max_throughput"},
        },
    })

    card = agent._objectives_card(
        ssops,
        {"quality_name": "quality", "perf_name": "performance"},
        ["deepeval_answer_correctness"],
    )

    assert "deepeval_faithfulness" in card
    assert "grounding diagnostic" in card
    assert "when available" in card
    assert "deterministic first 15 queries" in card
    assert "not cohort-aligned" in card
    assert "collocated unified engine" in card


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (ConnectionError("provider unavailable"), "initial chat call failed"),
        ("not JSON", "could not parse candidates"),
    ],
)
def test_strict_chat_and_json_failures_raise(monkeypatch, reply, message):
    monkeypatch.delenv("RAG_STACK_AGENT_STRICT", raising=False)
    agent = _agent(_ReplyClient(reply), strict=True)
    with pytest.raises(AgentAdviceError, match=message):
        agent._advise_oneshot(_HISTORY, object(), 3)


def test_strict_retrieval_round_failure_raises(monkeypatch):
    monkeypatch.delenv("RAG_STACK_AGENT_STRICT", raising=False)
    first = json.dumps({"queries": ["detail e00"]})
    agent = _agent(
        _ReplyClient(first, ConnectionError("follow-up failed")),
        strict=True,
    )
    agent._run_ledger_queries = lambda queries, history, ssops: "evidence"
    with pytest.raises(AgentAdviceError, match="retrieval round failed"):
        agent._advise_oneshot(_HISTORY, object(), 3)


def test_strict_all_candidate_encode_failures_raise(monkeypatch):
    monkeypatch.delenv("RAG_STACK_AGENT_STRICT", raising=False)
    reply = json.dumps({"candidates": [{"unknown": "value"}]})
    agent = _agent(_ReplyClient(reply), strict=True)
    agent._encode_proposal_with_metadata = (
        lambda prop, ssops, best_flat: (None, {"deltas": {}})
    )
    with pytest.raises(AgentAdviceError, match="all proposals failed to encode"):
        agent._advise_oneshot(_HISTORY, object(), 3)


def test_each_chat_has_hard_wall_clock_budget(monkeypatch):
    monkeypatch.delenv("RAG_STACK_AGENT_STRICT", raising=False)

    class _HangingClient:
        async def chat(self, messages, **kwargs):
            await anyio.sleep(10)

    agent = _agent(_HangingClient(), strict=True, wall_timeout_s=0.03)
    started = time.monotonic()
    with pytest.raises(AgentAdviceError, match="initial chat call failed") as exc_info:
        agent._advise_oneshot(_HISTORY, object(), 3)
    assert time.monotonic() - started < 1.0
    assert isinstance(exc_info.value.__cause__, TimeoutError)


class _PromptHarness(PromptFeedMixin):
    pass


def _prompt_harness(agent, *, strict):
    harness = _PromptHarness()
    harness.agent = agent
    harness.agent_strict = strict
    harness._t = 2
    harness.objective_names = ["performance", "quality"]
    harness._perf_idx = 0
    harness._qual_idx = 1
    harness.ref_point = torch.tensor([0.0, 0.0])
    harness._infeasible = []
    harness._advice_log_dir = None
    harness.ssops = object()
    harness._pareto_for_agent = lambda: []
    harness._model_confidence = lambda: {}
    harness._max_hv_rectangle = lambda: {}
    harness._history = lambda: list(_HISTORY)
    return harness


def test_strict_maybe_advise_does_not_swallow_agent_exception():
    class _BrokenAgent:
        def advise(self, history, ssops, t, mo):
            raise ValueError("bad agent response")

    strict = _prompt_harness(_BrokenAgent(), strict=True)
    with pytest.raises(AgentAdviceError, match="agent.advise failed in strict mode"):
        strict._maybe_advise()

    non_strict = _prompt_harness(_BrokenAgent(), strict=False)
    assert non_strict._maybe_advise() is None


def test_maybe_advise_calls_agent_on_every_successive_iteration():
    seen = []

    class _RecordingAgent:
        def advise(self, history, ssops, t, mo):
            seen.append(t)
            return None

    harness = _prompt_harness(_RecordingAgent(), strict=False)
    for iteration in (0, 1, 2, 7):
        harness._t = iteration
        assert harness._maybe_advise() is None

    assert seen == [0, 1, 2, 7]


def _bare_optimizer(*, strict):
    optimizer = object.__new__(AgentGuidedQNEHVI)
    optimizer._configs = []
    optimizer.n_both_init = 0
    optimizer.agent_strict = strict
    optimizer.agent = object()
    optimizer.objective_names = ["performance", "quality"]
    optimizer._t = 0
    optimizer._pending_full_metrics = []
    optimizer.doe = SimpleNamespace(load_file=lambda _path: None)
    optimizer._forced_state_loaded = True
    optimizer._fit_models = lambda: (_ for _ in ()).throw(ValueError("BO failed"))
    optimizer._sobol_batch = lambda n: [{"fallback": True}]
    return optimizer


def test_strict_suggest_never_random_falls_back_on_bo_failure():
    optimizer = _bare_optimizer(strict=True)
    with pytest.raises(ValueError, match="BO failed"):
        optimizer.suggest()
    assert optimizer._pending_full_metrics == []
    assert optimizer._t == 0


def test_non_strict_suggest_keeps_legacy_random_fallback():
    optimizer = _bare_optimizer(strict=False)
    configs, objectives = optimizer.suggest()
    assert configs == [{"fallback": True}]
    assert objectives == ["performance", "quality"]
    assert optimizer._t == 1


def test_constructor_and_environment_wire_strict_policy(monkeypatch):
    class _FakeAgent:
        def __init__(
            self, model, strict, wall_timeout_s,
        ):
            self.strict = strict
            self.wall_timeout_s = wall_timeout_s or 150.0

    config_space = SimpleNamespace(parameters={})
    ssops = SimpleNamespace(
        ss=config_space,
        dim=1,
        cat_dims_set=set(),
        tkwargs={"dtype": torch.double, "device": torch.device("cpu")},
    )
    monkeypatch.setattr(core_module, "LLMAgent", _FakeAgent)
    monkeypatch.setenv("RAG_STACK_AGENT_STRICT", "1")
    monkeypatch.setenv("RAG_STACK_AGENT_WALL_TIMEOUT_S", "12.5")

    optimizer = AgentGuidedQNEHVI(
        config_space,
        objectives=[
            {"objective_name": "performance", "ref_point": 0.0},
            {"objective_name": "quality", "ref_point": 0.0},
        ],
        ssops=ssops,
    )
    assert optimizer.agent_strict is True
    assert optimizer.agent.strict is True
    assert optimizer.agent_wall_timeout_s == 12.5


@pytest.mark.parametrize(
    ("removed_name", "removed_value"),
    [
        ("agent", object()),
        ("agent_kind", "llm"),
        ("inject_candidates", True),
        ("agent_mo_aware", False),
    ],
)
def test_constructor_rejects_removed_agent_options(
    removed_name, removed_value,
):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        AgentGuidedQNEHVI(
            SimpleNamespace(parameters={}),
            objectives=[
                {"objective_name": "performance", "ref_point": 0.0},
                {"objective_name": "quality", "ref_point": 0.0},
            ],
            **{removed_name: removed_value},
        )


@pytest.mark.parametrize("legacy_value", [True, False])
def test_llm_agent_rejects_removed_mo_aware_option(legacy_value):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        LLMAgent(ai_client=object(), mo_aware=legacy_value)


def test_optimizer_families_construct_their_canonical_agent_types():
    search_space = SearchSpace(parameters=[
        ChoiceParameter(
            "x",
            ParameterType.INT,
            [0, 1],
            is_ordered=True,
            sort_values=True,
        ),
    ])
    objectives = [
        {"objective_name": "performance", "ref_point": 0.0},
        {"objective_name": "quality", "ref_point": 0.0},
    ]

    base = AgentGuidedQNEHVI(
        search_space,
        objectives=objectives,
        ssops=AxSearchSpaceOps(search_space),
        enable_doe=False,
    )
    anchor = AnchorAgentQNEHVI(
        search_space,
        objectives=objectives,
        ssops=AxSearchSpaceOps(search_space),
        enable_doe=False,
    )
    hv_anchor = HVAnchorQNEHVI(
        search_space,
        objectives=objectives,
        ssops=AxSearchSpaceOps(search_space),
        enable_doe=False,
    )

    assert type(base.agent) is LLMAgent
    assert type(anchor.agent) is GreedyAnchorLLMAgent
    assert type(hv_anchor.agent) is HVAnchorLLMAgent


def test_invalid_wall_timeout_is_rejected(monkeypatch):
    monkeypatch.setenv("RAG_STACK_AGENT_WALL_TIMEOUT_S", "0")
    with pytest.raises(ValueError, match="positive finite"):
        LLMAgent(ai_client=object())
