"""Self-contained policy tests for the always-available LLM ledger follow-up."""

from __future__ import annotations

import json

from rag_stack.optimizer.agent.llm_agent import AgentAdvice, LLMAgent


class _Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.replies.pop(0)


def _exercise(agent: LLMAgent, monkeypatch):
    monkeypatch.setattr(agent, "_build_messages", lambda *a, **k: [
        {"role": "user", "content": "fixture"},
    ])
    monkeypatch.setattr(agent, "_run_ledger_queries", lambda *a, **k: "evidence")
    monkeypatch.setattr(
        agent,
        "_build_advice_from_parsed",
        lambda parsed, *_args: AgentAdvice(rationale=parsed.get("reasoning", "")),
    )
    history = [{"flat": {}, "correctness": 0.2, "performance": 1.0}]
    return agent._advise_oneshot(history, object(), 0)


def test_final_answer_uses_one_chat(monkeypatch):
    client = _Client([
        json.dumps({"reasoning": "done", "candidates": [{"x": 1}]}),
    ])
    agent = LLMAgent(model="unused", ai_client=client, strict=False)

    advice = _exercise(agent, monkeypatch)

    assert advice.rationale == "done"
    assert len(client.calls) == 1


def test_query_request_always_gets_one_ledger_followup(monkeypatch):
    client = _Client([
        json.dumps({"queries": ["detail e00"]}),
        json.dumps({"reasoning": "done", "candidates": [{"x": 1}]}),
    ])
    agent = LLMAgent(model="unused", ai_client=client, strict=False)

    advice = _exercise(agent, monkeypatch)

    assert advice.rationale == "done"
    assert len(client.calls) == 2
    followup = client.calls[1][0][-1]["content"]
    assert "LEDGER QUERY RESULTS:\nevidence" in followup
    assert "no further queries" in followup


def test_followup_is_bounded_to_one_extra_chat(monkeypatch):
    client = _Client([
        json.dumps({"queries": ["detail e00"]}),
        json.dumps({"queries": ["detail e01"]}),
        json.dumps({"reasoning": "must not be reached", "candidates": [{"x": 1}]}),
    ])
    agent = LLMAgent(model="unused", ai_client=client, strict=False)

    advice = _exercise(agent, monkeypatch)

    assert advice.centers is None
    assert len(client.calls) == 2
    assert len(client.replies) == 1
