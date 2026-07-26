"""Self-contained maximum-Pareto-gap Agent exploration contracts."""

from __future__ import annotations

import json

import pytest
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.optimizer.agent.llm_agent import (
    GapExplorationRequest,
    LLMAgent,
    ParetoGapAnchor,
)
from rag_stack.search_space import AxSearchSpaceOps


def _choice(name, values, *, ordered=False):
    return ChoiceParameter(
        name=name,
        parameter_type=(
            ParameterType.INT
            if all(isinstance(value, int) for value in values)
            else ParameterType.STRING
        ),
        values=values,
        is_ordered=ordered,
        sort_values=ordered,
    )


def _flat_space() -> AxSearchSpaceOps:
    return AxSearchSpaceOps(SearchSpace(parameters=[
        _choice("mode", ["slow", "middle", "fast"]),
        _choice("depth", [1, 2, 3], ordered=True),
        _choice("method", ["plain", "augmented"]),
    ]))


def _request(*, failed_nearby=()) -> GapExplorationRequest:
    return GapExplorationRequest(
        gap_id="gap-e4-e9",
        quality_anchor=ParetoGapAnchor(
            flat={"mode": "slow", "depth": 3, "method": "augmented"},
            normalized_performance=0.2,
            normalized_quality=0.9,
            sub_metrics={"grounded": 0.92, "coverage": 0.81},
            perf_attr={
                "qps": 20.0,
                "stage_qps": {"left": 20.0, "right": 100.0},
                "bound": "left",
                "deploy": "layout-a",
            },
        ),
        performance_anchor=ParetoGapAnchor(
            flat={"mode": "fast", "depth": 1, "method": "plain"},
            normalized_performance=0.9,
            normalized_quality=0.2,
            sub_metrics={"grounded": 0.31, "coverage": 0.42},
            perf_attr={
                "qps": 90.0,
                "stage_qps": {"left": 180.0, "right": 90.0},
                "bound": "right",
                "deploy": "layout-b",
            },
        ),
        failed_nearby=tuple(failed_nearby),
    )


def _history(request):
    return [
        {
            "flat": dict(request.quality_anchor.flat),
            "performance": 20.0,
            "correctness": 0.9,
        },
        {
            "flat": dict(request.performance_anchor.flat),
            "performance": 90.0,
            "correctness": 0.2,
        },
    ]


def test_gap_request_rejects_non_normalized_or_misoriented_geometry():
    with pytest.raises(ValueError, match="finite value in \\[0, 1\\]"):
        ParetoGapAnchor(
            flat={"mode": "slow"},
            normalized_performance=2.0,
            normalized_quality=0.9,
        )

    q_anchor = ParetoGapAnchor(
        flat={"mode": "slow"},
        normalized_performance=0.8,
        normalized_quality=0.9,
    )
    p_anchor = ParetoGapAnchor(
        flat={"mode": "fast"},
        normalized_performance=0.2,
        normalized_quality=0.1,
    )
    with pytest.raises(ValueError, match="internal trade-off rectangle"):
        GapExplorationRequest("bad-gap", q_anchor, p_anchor)


def test_gap_prompt_contains_exact_geometry_evidence_and_strict_schema():
    ssops = _flat_space()
    request = _request(failed_nearby=[{
        "flat": {"mode": "middle", "depth": 3, "method": "plain"},
        "reason": "no realized HV gain",
        "normalized_performance": 0.55,
        "normalized_quality": 0.25,
    }])
    agent = LLMAgent(ai_client=object(), n_propose=3)

    messages = agent._build_gap_exploration_messages(
        request, _history(request), ssops,
    )
    prompt = messages[1]["content"]

    assert "GAP ID: gap-e4-e9" in prompt
    assert "performance=0.200000, quality=0.900000" in prompt
    assert "performance=0.900000, quality=0.200000" in prompt
    assert "ACTIVE KNOB DIFFERENCES BETWEEN THE ANCHORS" in prompt
    assert "mode: quality_anchor='slow'; performance_anchor='fast'" in prompt
    assert '"grounded": 0.92' in prompt
    assert '"bound": "left"' in prompt and '"bound": "right"' in prompt
    assert "no realized HV gain" in prompt
    assert "Return at most 3 diverse candidates" in prompt
    assert '"base": "quality_anchor|performance_anchor"' in prompt
    assert "non-empty set of explicit deltas" in prompt
    assert "1-5" not in prompt
    assert "no queries, markdown, notes, or additional keys" in prompt


def test_gap_agent_accepts_more_than_five_valid_deltas():
    ssops = AxSearchSpaceOps(SearchSpace(parameters=[
        _choice(f"knob_{index}", [0, 1], ordered=True)
        for index in range(6)
    ]))
    quality_flat = {f"knob_{index}": 0 for index in range(6)}
    performance_flat = dict(quality_flat)
    performance_flat["knob_0"] = 1
    request = GapExplorationRequest(
        "wide-gap",
        ParetoGapAnchor(quality_flat, 0.2, 0.8),
        ParetoGapAnchor(performance_flat, 0.8, 0.2),
    )
    deltas = {f"knob_{index}": 1 for index in range(6)}
    parsed = {
        "gap_id": "wide-gap",
        "reasoning": "coordinated cross-stage change",
        "candidates": [{
            "base": "quality_anchor",
            "deltas": deltas,
            "hypothesis": "the joint change targets the gap",
        }],
    }

    advice = LLMAgent(
        ai_client=object(), n_propose=1,
    )._build_gap_advice_from_parsed(
        parsed, request, [], ssops,
    )

    assert advice.n() == 1
    assert advice.proposal_meta[0]["deltas"] == deltas
    assert ssops.decode_flat(advice.centers[0]) == deltas


def test_gap_exploration_is_bounded_novel_and_uses_deterministic_chat(monkeypatch):
    ssops = _flat_space()
    failed = {"flat": {"mode": "middle", "depth": 3, "method": "plain"}}
    request = _request(failed_nearby=[failed])
    agent = LLMAgent(ai_client=object(), n_propose=2, strict=True)
    captured = {}
    response = {
        "gap_id": request.gap_id,
        "reasoning": "probe two distinct interior directions",
        "candidates": [
            # No-op relative to its anchor: rejected by decoded novelty.
            {
                "base": "quality_anchor",
                "deltas": {"mode": "slow"},
                "hypothesis": "control",
            },
            # Explicitly listed failed config: also rejected.
            {
                "base": "performance_anchor",
                "deltas": {"mode": "middle", "depth": 3},
                "hypothesis": "already disproven",
            },
            {
                "base": "quality_anchor",
                "deltas": {"mode": "middle", "depth": 2},
                "hypothesis": "balanced interior",
            },
            {
                "base": "performance_anchor",
                "deltas": {"depth": 2, "method": "augmented"},
                "hypothesis": "quality graft on fast anchor",
            },
        ],
    }

    def _chat(_client, messages, *, temperature=None):
        captured["messages"] = messages
        captured["temperature"] = temperature
        return json.dumps(response)

    monkeypatch.setattr(agent, "_chat_with_wall_timeout", _chat)

    advice = agent.explore_gap(request, _history(request), ssops, t=12)

    assert captured["temperature"] == 0.0
    assert advice.n() == 2
    assert advice.raw["status"] == "ok"
    decoded = [ssops.decode_flat(row) for row in advice.centers]
    assert decoded == [
        {"mode": "middle", "depth": 2, "method": "augmented"},
        {"mode": "fast", "depth": 2, "method": "augmented"},
    ]
    assert [meta["gap_id"] for meta in advice.proposal_meta] == [
        request.gap_id, request.gap_id,
    ]
    assert [meta["hypothesis"] for meta in advice.proposal_meta] == [
        "balanced interior", "quality graft on fast anchor",
    ]


def test_gap_exploration_non_strict_schema_failure_is_fallback_friendly(
    monkeypatch,
):
    ssops = _flat_space()
    request = _request()
    agent = LLMAgent(ai_client=object(), strict=False)
    response = {
        "gap_id": "stale-gap",
        "reasoning": "wrong request",
        "candidates": [{
            "base": "quality_anchor",
            "deltas": {"mode": "middle"},
            "hypothesis": "would otherwise be valid",
        }],
    }
    monkeypatch.setattr(
        agent,
        "_chat_with_wall_timeout",
        lambda *_args, **_kwargs: json.dumps(response),
    )

    advice = agent.explore_gap(request, _history(request), ssops, t=8)

    assert advice.n() == 0
    assert advice.raw["status"] == "unavailable"
    assert advice.raw["gap_id"] == request.gap_id
    assert "gap_id does not match" in advice.raw["error"]


def test_gap_exploration_infers_hierarchical_branch(monkeypatch):
    selector = ChoiceParameter(
        name="selector",
        parameter_type=ParameterType.STRING,
        values=["a", "b"],
        is_ordered=False,
        sort_values=False,
        dependents={"a": ["a_level"], "b": ["b_level"]},
    )
    ssops = AxSearchSpaceOps(SearchSpace(parameters=[
        selector,
        _choice("a_level", [0, 1], ordered=True),
        _choice("b_level", [0, 1], ordered=True),
    ]))
    request = GapExplorationRequest(
        "hier-gap",
        ParetoGapAnchor(
            {"selector": "a", "a_level": 1}, 0.2, 0.8,
        ),
        ParetoGapAnchor(
            {"selector": "b", "b_level": 0}, 0.8, 0.2,
        ),
    )
    response = {
        "gap_id": "hier-gap",
        "reasoning": "try the unobserved level on branch b",
        "candidates": [{
            "base": "quality_anchor",
            "deltas": {"b_level": 1},
            "hypothesis": "branch-safe structural interpolation",
        }],
    }
    agent = LLMAgent(ai_client=object())
    monkeypatch.setattr(
        agent,
        "_chat_with_wall_timeout",
        lambda *_args, **_kwargs: json.dumps(response),
    )

    advice = agent.explore_gap(request, [], ssops, t=8)

    assert advice.n() == 1
    decoded = ssops.decode_flat(advice.centers[0])
    assert decoded["selector"] == "b"
    assert decoded["b_level"] == 1
    assert advice.proposal_meta[0]["deltas"] == {"b_level": 1}
    assert advice.proposal_meta[0]["inferred_ancestors"] == {"selector": "b"}
    assert set(advice.proposal_meta[0]) == {
        "kind",
        "gap_id",
        "base",
        "hypothesis",
        "deltas",
        "inferred_ancestors",
    }
