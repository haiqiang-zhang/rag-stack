"""CPU-only checks for diagnostic isolation in ordinary and forced prompts."""

from copy import deepcopy
import json
import socket
from types import SimpleNamespace

import pytest
import torch
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.optimizer.agent.llm_agent import (
    AgentAdvice,
    GapExplorationRequest,
    LLMAgent,
    ParetoGapAnchor,
)
from rag_stack.optimizer.agent.prompts import PromptFeedMixin
from rag_stack.search_space import AxSearchSpaceOps


DIAGNOSTIC = "private_stage_metric"
WORKER = "private_des_worker"


@pytest.fixture(autouse=True)
def _no_network_or_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Stage-signal tests must not use network or CUDA")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    for name in ("init", "_lazy_init", "is_available", "device_count"):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def _space():
    space = AxSearchSpaceOps(SearchSpace(parameters=[ChoiceParameter(
        name="depth", parameter_type=ParameterType.INT, values=[1, 2, 3],
        is_ordered=True, sort_values=True,
    )]))
    space._config = {
        "eval_backend_setting": {
            "metrics": [{
                "metric_name": DIAGNOSTIC,
                "description": "private_stage_description",
            }],
            "combined_quality": {"metrics": ["final_judge"], "mode": "mean"},
        },
    }
    return space


def _history():
    return [{
        "flat": {"depth": index + 1},
        "performance": 30.0 - index * 10,
        "correctness": 0.2 + index * 0.3,
        "sub_metrics": {DIAGNOSTIC: 0.123 + index * 0.1},
        "perf_attr": {
            "source": "authoritative_des",
            "qps": 30.0 - index * 10,
            "workers": {WORKER: {"batch_mean": 7.89}},
        },
    } for index in range(3)]


def _pareto(history):
    return [{
        "flat": row["flat"], "perf": row["performance"],
        "corr": row["correctness"], "sub_metrics": row["sub_metrics"],
        "perf_attr": row["perf_attr"],
    } for row in history]


@pytest.mark.parametrize("enabled", [True, False])
def test_ordinary_prompt_masks_all_diagnostic_cards_without_mutation(enabled):
    space = _space()
    history = _history()
    mo = {"pareto": _pareto(history)}
    original_history, original_mo = deepcopy(history), deepcopy(mo)
    agent = LLMAgent(model="unused", enable_stage_signals=enabled)

    prompt = "\n".join(row["content"] for row in agent._build_messages(history, space, mo))

    for token in (DIAGNOSTIC, WORKER, "private_stage_description", "PERF ATTRIBUTION"):
        assert (token in prompt) == enabled
    assert ("INTERMEDIATE SUB-METRICS" in prompt) == enabled
    assert ("SUB-METRIC RELIABILITY" in prompt) == enabled
    assert ("PERFORMANCE DIAGNOSTICS" in prompt) == enabled
    assert "final_judge" in prompt
    assert "MODEL CONFIDENCE" in prompt
    assert "q=0.800" in prompt
    assert history == original_history
    assert mo == original_mo


@pytest.mark.parametrize("enabled", [True, False])
def test_ledger_followup_and_direct_cards_respect_signal_flag(enabled):
    space, history = _space(), _history()
    agent = LLMAgent(model="unused", enable_stage_signals=enabled)
    ledger = agent._run_ledger_queries(["detail e00", "depth>=2"], history, space)
    assert (DIAGNOSTIC in ledger) == enabled
    assert "perf=30" in ledger
    for card in (
        agent._obs_card(history, {"depth"}, space),
        agent._pareto_card(_pareto(history), {"depth"}, space),
    ):
        assert (DIAGNOSTIC in card) == enabled
        assert (WORKER in card) == enabled


@pytest.mark.parametrize("enabled", [True, False])
def test_forced_prompt_masks_anchor_and_nested_nearby_diagnostics(enabled):
    history, space = _history(), _space()
    request = GapExplorationRequest(
        gap_id="test-gap",
        quality_anchor=ParetoGapAnchor(
            flat=history[2]["flat"], normalized_performance=0.2,
            normalized_quality=0.8, sub_metrics=history[2]["sub_metrics"],
            perf_attr=history[2]["perf_attr"],
        ),
        performance_anchor=ParetoGapAnchor(
            flat=history[0]["flat"], normalized_performance=0.8,
            normalized_quality=0.2, sub_metrics=history[0]["sub_metrics"],
            perf_attr=history[0]["perf_attr"],
        ),
        failed_nearby=({
            "flat": {"depth": 2}, "reason": "no HV gain",
            "evidence": {"sub_metrics": {"nested_private_metric": 0.45}},
        },),
    )
    agent = LLMAgent(model="unused", enable_stage_signals=enabled)
    prompt = "\n".join(row["content"] for row in agent._build_gap_exploration_messages(
        request, history, space,
    ))

    for token in (DIAGNOSTIC, WORKER, "nested_private_metric", "CM bottleneck/audit"):
        assert (token in prompt) == enabled
    assert "quality=0.800000" in prompt
    assert "no HV gain" in prompt
    assert request.quality_anchor.sub_metrics == history[2]["sub_metrics"]


def test_missing_flag_preserves_default_rendering():
    agent = LLMAgent(model="unused")
    space, history = _space(), _history()
    expected = agent._build_messages(history, space)
    del agent.enable_stage_signals
    assert agent._build_messages(history, space) == expected


@pytest.mark.parametrize("value", ["false", "true", 0, None])
def test_agent_rejects_non_boolean_signal_flag(value):
    with pytest.raises(ValueError, match="enable_stage_signals must be a boolean"):
        LLMAgent(model="unused", enable_stage_signals=value)


def test_disabled_agent_does_not_build_or_query_prompt_evidence():
    feed = PromptFeedMixin()
    feed.enable_agent = False
    assert feed._maybe_advise() is None


@pytest.mark.parametrize("enabled", [True, False])
def test_resume_restores_notes_only_from_matching_signal_mode(tmp_path, enabled):
    records = [
        {"notes": "legacy diagnostics"},
        {"notes": "objective-only notes", "enable_stage_signals": False},
        {"notes": "stage diagnostics", "enable_stage_signals": True},
    ]
    (tmp_path / "agent_advice.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
    )
    feed = PromptFeedMixin()
    feed.enable_stage_signals = enabled
    feed._qual_idx, feed._perf_idx = 1, 0
    feed.objective_names = ["performance", "quality"]
    feed.ref_point = SimpleNamespace(tolist=lambda: [0.0, 0.0])
    feed._pareto_for_agent = lambda: []
    feed._model_confidence = lambda: {}
    feed._max_hv_rectangle = lambda: {}
    feed._independent_evidence_payload = lambda: {}
    feed._history = lambda: []
    feed._infeasible = []
    feed._advice_log_dir = str(tmp_path)
    feed._t = 10
    feed.ssops = object()
    feed.agent_strict = False
    feed.agent = SimpleNamespace(_notes="", advise=lambda *args: AgentAdvice())

    feed._maybe_advise()

    assert feed.agent._notes == ("stage diagnostics" if enabled else "objective-only notes")


@pytest.mark.parametrize("enabled", [True, False])
def test_optimizer_prompt_feed_hides_values_without_erasing_observations(enabled):
    class Feed(PromptFeedMixin):
        pass

    feed = Feed()
    feed.enable_stage_signals = enabled
    feed.ssops = _space()
    history = _history()
    feed._configs = [row["flat"] for row in history]
    feed._x_rows = [feed.ssops.encode_arm_params(row["flat"]) for row in history]
    feed._perf = [row["performance"] for row in history]
    feed._corr = [row["correctness"] for row in history]
    feed._sub_metrics = [row["sub_metrics"] for row in history]
    feed._eval_ids = list(range(len(history)))
    feed._perf_attr = {i: row["perf_attr"] for i, row in enumerate(history)}
    original_submetrics, original_attrs = deepcopy(feed._sub_metrics), deepcopy(feed._perf_attr)

    for rows in (feed._history(), feed._pareto_for_agent()):
        assert len(rows) == 3
        assert all(bool(row["sub_metrics"]) == enabled for row in rows)
        assert all(bool(row["perf_attr"]) == enabled for row in rows)
    assert feed._sub_metrics == original_submetrics
    assert feed._perf_attr == original_attrs
