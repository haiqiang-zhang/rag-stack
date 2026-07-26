"""Self-contained contracts for Agent GP/DOE evidence and strict forced calls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.optimizer.agent.llm_agent import (
    AgentAdviceError,
    GapExplorationRequest,
    LLMAgent,
    ParetoGapAnchor,
)
from rag_stack.optimizer.agent.prompts import PromptFeedMixin
from rag_stack.search_space import AxSearchSpaceOps


def _choice(name, values):
    return ChoiceParameter(
        name=name,
        parameter_type=ParameterType.INT,
        values=list(values),
        is_ordered=True,
        sort_values=True,
    )


def _space() -> AxSearchSpaceOps:
    return AxSearchSpaceOps(SearchSpace(parameters=[
        _choice("factor_a", [0, 1, 2]),
        _choice("factor_b", [0, 1, 2]),
    ]))


def _row(ssops, factor_a, factor_b):
    return ssops.clamp(
        ssops.encode_arm_params({
            "factor_a": factor_a,
            "factor_b": factor_b,
        }).reshape(1, -1),
    ).reshape(-1)


class _AffineModel:
    """Deterministic posterior whose output axes can be affinely transformed."""

    def __init__(
        self,
        *,
        performance_scale=1.0,
        performance_shift=0.0,
        quality_scale=1.0,
        quality_shift=0.0,
        factor_b_performance=-20.0,
    ):
        self.performance_scale = float(performance_scale)
        self.performance_shift = float(performance_shift)
        self.quality_scale = float(quality_scale)
        self.quality_shift = float(quality_shift)
        self.factor_b_performance = float(factor_b_performance)

    def posterior(self, x):
        flat = x.reshape(-1, x.shape[-1])
        factor_a, factor_b = flat[:, 0], flat[:, 1]
        raw_performance = (
            100.0 - 40.0 * factor_a
            + self.factor_b_performance * factor_b
        )
        raw_quality = 0.1 + 0.6 * factor_a + 0.2 * factor_b
        performance = (
            self.performance_scale * raw_performance
            + self.performance_shift
        )
        quality = self.quality_scale * raw_quality + self.quality_shift
        mean = torch.stack((performance, quality), dim=-1).unsqueeze(-2)
        variance = torch.stack((
            torch.full_like(performance, (3.0 * self.performance_scale) ** 2),
            torch.full_like(quality, (0.04 * self.quality_scale) ** 2),
        ), dim=-1).unsqueeze(-2)
        return SimpleNamespace(mean=mean, variance=variance)


class _DOE:
    @staticmethod
    def ablation_card_payload():
        return ({
                "kind": "oat",
                "side": "p",
                "branch_id": "root",
                "changed_knobs": [{
                    "parameter": "factor_a",
                    "anchor_level": 0,
                    "observed_level": 1,
                }],
                "delta_performance": -1 / 3,
                "delta_quality": 0.375,
                "delivery": "forced",
                "eval_id": 4,
                "interaction": None,
            },)


class _Harness(PromptFeedMixin):
    pass


def _harness(
    *,
    performance_scale=1.0,
    performance_shift=0.0,
    quality_scale=1.0,
    quality_shift=0.0,
    factor_b_performance=-20.0,
):
    ssops = _space()
    harness = _Harness()
    harness.ssops = ssops
    harness.tkwargs = ssops.tkwargs
    harness._perf_idx = 0
    harness._qual_idx = 1
    harness._x_rows = [
        _row(ssops, 0, 0),
        _row(ssops, 1, 0),
        _row(ssops, 0, 1),
        _row(ssops, 2, 2),
    ]
    raw_performance = [100.0, 80.0, 90.0, 40.0]
    raw_quality = [0.1, 0.4, 0.2, 0.9]
    harness._perf = [
        performance_scale * value + performance_shift
        for value in raw_performance
    ]
    harness._corr = [
        quality_scale * value + quality_shift
        for value in raw_quality
    ]
    harness.model = _AffineModel(
        performance_scale=performance_scale,
        performance_shift=performance_shift,
        quality_scale=quality_scale,
        quality_shift=quality_shift,
        factor_b_performance=factor_b_performance,
    )
    harness.doe = _DOE()
    return harness


def test_gp_tension_card_is_generic_supported_and_affine_invariant():
    original = _harness()._independent_evidence_payload()
    transformed = _harness(
        performance_scale=7.0,
        performance_shift=-300.0,
        quality_scale=0.125,
        quality_shift=4.0,
    )._independent_evidence_payload()

    assert original["gp_knob_tension"]["status"] == "ok"
    original_rows = original["gp_knob_tension"]["rows"]
    transformed_rows = transformed["gp_knob_tension"]["rows"]
    assert len(original_rows) == len(transformed_rows)
    categorical = {
        "knob",
        "anchor",
        "anchor_level",
        "level",
        "anchor_branch",
        "candidate_branch",
        "support",
        "levels_considered",
    }
    for row, transformed_row in zip(original_rows, transformed_rows):
        assert {key: row[key] for key in categorical} == {
            key: transformed_row[key] for key in categorical
        }
        for key in set(row) - categorical:
            assert row[key] == pytest.approx(transformed_row[key])
        assert row["knob"] in {"factor_a", "factor_b"}
        assert row["support"] >= 0
        assert row["anchor_branch"] == {}
        assert row["candidate_branch"] == {}
        assert row["delta_performance_norm"] * row["delta_quality_norm"] < 0
        assert row["tension"] > 0
        assert row["levels_considered"] == 2


def test_gp_card_keeps_every_eligible_knob_when_tension_is_zero():
    payload = _harness(
        factor_b_performance=20.0,
    )._gp_knob_tension_evidence()

    assert payload["status"] == "ok"
    by_knob = {}
    for row in payload["rows"]:
        by_knob.setdefault(row["knob"], []).append(row)
    assert set(by_knob) == {"factor_a", "factor_b"}
    assert all(row["tension"] > 0.0 for row in by_knob["factor_a"])
    assert all(row["tension"] == 0.0 for row in by_knob["factor_b"])


def _history(harness):
    return [
        {
            "flat": harness.ssops.decode_flat(row),
            "performance": performance,
            "correctness": quality,
            "sub_metrics": {},
        }
        for row, performance, quality in zip(
            harness._x_rows,
            harness._perf,
            harness._corr,
        )
    ]


def _gap_request():
    return GapExplorationRequest(
        gap_id="gap-0",
        quality_anchor=ParetoGapAnchor(
            flat={"factor_a": 2, "factor_b": 2},
            normalized_performance=0.0,
            normalized_quality=1.0,
        ),
        performance_anchor=ParetoGapAnchor(
            flat={"factor_a": 0, "factor_b": 0},
            normalized_performance=1.0,
            normalized_quality=0.0,
        ),
    )


def test_ordinary_and_forced_prompts_render_two_distinct_evidence_sections():
    harness = _harness()
    payload = harness._independent_evidence_payload()
    agent = LLMAgent(ai_client=object())

    ordinary = "\n".join(
        message["content"]
        for message in agent._build_messages(
            _history(harness),
            harness.ssops,
            mo={"pareto": [], "independent_evidence": payload},
        )
    )
    forced = "\n".join(
        message["content"]
        for message in agent._build_gap_exploration_messages(
            _gap_request(),
            _history(harness),
            harness.ssops,
            independent_evidence=payload,
        )
    )

    for prompt in (ordinary, forced):
        assert "GP PER-KNOB MULTIOBJECTIVE TENSION" in prompt
        assert "MARGINAL SURROGATE SCREENING; NOT A LANDED ABLATION" in prompt
        assert "DOE LANDED EXACT CHAMPION ABLATIONS" in prompt
        assert "REALIZED OAT/PAIR EVIDENCE" in prompt
        assert "factor_a" in prompt
        assert "OAT side=p" in prompt
        assert "delivery=forced eval=4" in prompt


def test_missing_evidence_is_explicit_on_every_ordinary_prompt():
    harness = _harness()
    agent = LLMAgent(ai_client=object())
    prompt = "\n".join(
        message["content"]
        for message in agent._build_messages(
            _history(harness), harness.ssops, mo={"pareto": []},
        )
    )

    assert prompt.count("unavailable:") >= 2
    assert "independent GP evidence was not supplied" in prompt
    assert "DOE landed-ablation evidence was not supplied" in prompt


@pytest.mark.parametrize(
    "reply",
    (
        "not JSON",
        json.dumps({
            "gap_id": "wrong-gap",
            "reasoning": "stale",
            "candidates": [{
                "base": "quality_anchor",
                "deltas": {"factor_a": 1},
                "hypothesis": "otherwise valid",
            }],
        }),
        json.dumps({
            "gap_id": "gap-0",
            "reasoning": "no-op",
            "candidates": [{
                "base": "quality_anchor",
                "deltas": {"factor_a": 2},
                "hypothesis": "duplicates its anchor",
            }],
        }),
    ),
)
def test_strict_gap_agent_rejects_parse_schema_and_no_usable_candidate(
    monkeypatch,
    reply,
):
    ssops = _space()
    agent = LLMAgent(ai_client=object(), strict=True)
    monkeypatch.setattr(
        agent,
        "_chat_with_wall_timeout",
        lambda *_args, **_kwargs: reply,
    )

    with pytest.raises(AgentAdviceError, match="gap exploration unavailable"):
        agent.explore_gap(_gap_request(), [], ssops, t=3)


def test_strict_gap_agent_rejects_chat_failure_but_non_strict_is_auditable(
    monkeypatch,
):
    ssops = _space()

    def fail(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    strict = LLMAgent(ai_client=object(), strict=True)
    monkeypatch.setattr(strict, "_chat_with_wall_timeout", fail)
    with pytest.raises(AgentAdviceError, match="chat failed"):
        strict.explore_gap(_gap_request(), [], ssops, t=3)

    non_strict = LLMAgent(ai_client=object(), strict=False)
    monkeypatch.setattr(non_strict, "_chat_with_wall_timeout", fail)
    advice = non_strict.explore_gap(_gap_request(), [], ssops, t=3)
    assert advice.n() == 0
    assert advice.raw["status"] == "unavailable"
    assert advice.raw["gap_id"] == "gap-0"
