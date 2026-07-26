"""Self-contained optimizer initialization, DLS, and forced-state regressions."""

from __future__ import annotations

import json

import pytest
import torch
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.optimizer.agent.core import AgentGuidedQNEHVI
from rag_stack.optimizer.agent.llm_agent import LLMAgent
from rag_stack.search_space import AxSearchSpaceOps


OBJECTIVES = [
    {"objective_name": "performance", "cost": 1, "ref_point": 0},
    {"objective_name": "quality", "cost": 10, "ref_point": 0},
]


class _EmptyAgentClient:
    """Local Agent transport for tests that do not exercise LLM proposals."""

    async def chat(self, messages, **kwargs):
        del messages, kwargs
        return json.dumps({"reasoning": "no proposal", "candidates": []})


def _choice(name: str, values) -> ChoiceParameter:
    return ChoiceParameter(
        name=name,
        parameter_type=ParameterType.INT,
        values=list(values),
        is_ordered=True,
        sort_values=True,
    )


def _make_opt(**overrides) -> AgentGuidedQNEHVI:
    """Build a tiny all-discrete CPU optimizer without repository configs."""
    search_space = SearchSpace(parameters=[
        _choice("a", [0, 1, 2]),
        _choice("b", [0, 1, 2]),
        _choice("c", [0, 1, 2]),
        _choice("d", [0, 1, 2]),
    ])
    ssops = AxSearchSpaceOps(search_space)
    kwargs = {
        "seed": 46,
        "n_both_init": 4,
        "init_sobol_k": 2,
        "ssops": ssops,
        "n_candidates": 32,
        "dls_num_restarts": 4,
    }
    kwargs.update(overrides)
    optimizer = AgentGuidedQNEHVI(
        search_space,
        objectives=OBJECTIVES,
        **kwargs,
    )
    optimizer.agent = LLMAgent(
        model="unused",
        ai_client=_EmptyAgentClient(),
    )
    return optimizer


def _feed(opt, configs, perf, qual, start_id=0) -> None:
    opt.add_objectives(
        configs,
        {
            "performance": list(perf),
            "quality": [{"combined": q} for q in qual],
        },
        eval_ids=list(range(start_id, start_id + len(configs))),
    )


def _warmed(**overrides) -> AgentGuidedQNEHVI:
    opt = _make_opt(init_sobol_k=0, **overrides)
    configs, _ = opt.suggest(4)
    _feed(opt, configs, [10.0, 40.0, 5.0, 90.0], [0.2, 0.3, 0.5, 0.15])
    opt._fit_models()
    return opt


def test_constructor_accepts_only_explicit_agent_policy_options():
    opt = _make_opt(
        enable_doe=False,
    )
    assert opt.doe.enabled is False

    with pytest.raises(
        TypeError,
        match="unexpected keyword argument 'normalize_hv_axes'",
    ):
        _make_opt(normalize_hv_axes=False)

    with pytest.raises(TypeError, match="unexpected keyword argument 'dead_option'"):
        _make_opt(dead_option=True)


def test_init_chunks_progression():
    opt = _make_opt()
    assert opt.init_chunks == [2, 2]
    first, _ = opt.suggest(2)
    assert len(first) == 2
    assert not hasattr(opt, "_init_plan")
    _feed(opt, first, [10.0, 40.0], [0.2, 0.3])
    assert opt.init_chunks == [2]

    second, _ = opt.suggest(2)
    assert len(second) == 2
    _feed(opt, second, [5.0, 90.0], [0.5, 0.15], start_id=2)
    assert opt.init_chunks == []


def test_init_chunks_resume_midway():
    opt = _make_opt()
    first, _ = opt.suggest(2)
    _feed(opt, first, [10.0, 40.0], [0.2, 0.3])
    second, _ = opt.suggest(1)
    _feed(opt, second, [5.0], [0.5], start_id=2)
    assert opt.init_chunks == [1]


def test_sobol_k_clamped_to_budget():
    opt = _make_opt(init_sobol_k=99)
    assert opt.init_sobol_k == opt.n_both_init
    assert opt.init_chunks == [4]


def test_llm_plan_gets_observed_results_and_exact_count():
    opt = _make_opt()
    first, _ = opt.suggest(2)
    _feed(opt, first, [12.5, 250.0], [0.41, 0.11])

    agent = LLMAgent(model="unused")
    captured = {}
    donor = [dict(config) for config in (first * 3)][:5]

    class _FakeClient:
        async def chat(self, messages, **kwargs):
            captured["messages"] = messages
            return json.dumps({"candidates": donor})

    agent._get_client = lambda: _FakeClient()
    opt.agent = agent
    plan = opt._llm_probe_plan(2)
    text = "\n".join(message["content"] for message in captured["messages"])
    assert "OBSERVED RESULTS" in text
    assert "Design EXACTLY 2 probes" in text
    assert "250" in text and "0.41" in text
    assert len(plan) == 2


def test_llm_plan_failure_falls_back_to_sobol():
    opt = _make_opt()
    first, _ = opt.suggest(2)
    _feed(opt, first, [10.0, 40.0], [0.2, 0.3])
    second, _ = opt.suggest(2)
    assert len(second) == 2
    assert all(isinstance(config, dict) and config for config in second)


def test_due_without_internal_gap_keeps_ordinary_agent_active():
    opt = _warmed()
    opt._t = 1
    opt._normal_no_hv_gain = 3
    opt._stagnation_normalization = {
        "p_min": 1.0,
        "p_span": 9.0,
        "q_min": 0.1,
        "q_span": 0.8,
    }
    opt._fit_models = lambda: None
    opt._largest_hv_gap = lambda **_kwargs: None
    calls = []
    advice = object()

    def ordinary_advice():
        calls.append("ordinary")
        return advice

    def propose(received, *, forced_due, forced_gap):
        assert received is advice
        assert forced_due is False
        assert forced_gap is None
        return opt._configs[0]

    opt._maybe_advise = ordinary_advice
    opt._propose_pool = propose

    configs, _ = opt.suggest()

    assert calls == ["ordinary"]
    assert configs == [opt._configs[0]]
    assert opt._normal_no_hv_gain == 3


def test_official_coordinate_dls_keeps_pool_provenance():
    opt = _warmed()
    acqf = opt._build_acqf()
    seeds = opt.ssops.snap_to_grid(torch.rand(4, opt.dim))
    grouped = opt._dls_candidates_grouped(acqf, None, pool_seeds=seeds)
    assert grouped is not None
    _, provenance = grouped
    assert "dls-pool" in provenance


def test_resume_reconciles_forced_eval_after_stale_state_snapshot(tmp_path):
    source = _make_opt()
    source._advice_log_dir = str(tmp_path)
    source._normal_no_hv_gain = 3
    source._stagnation_normalization = {
        "p_min": 1.0,
        "p_span": 9.0,
        "q_min": 0.1,
        "q_span": 0.8,
    }
    source._last_accounted_eval_id = 6
    source._t = 3
    source._save_forced_state()

    configs = [
        {"a": 0, "b": 0, "c": 0, "d": 0},
        {"a": 1, "b": 0, "c": 0, "d": 0},
        {"a": 2, "b": 0, "c": 0, "d": 0},
        {"a": 0, "b": 1, "c": 0, "d": 0},
        {"a": 0, "b": 2, "c": 0, "d": 0},
        {"a": 0, "b": 0, "c": 1, "d": 0},
        {"a": 0, "b": 0, "c": 2, "d": 0},
        {"a": 0, "b": 0, "c": 0, "d": 1},
    ]
    evaluations = [
        {
            "eval_id": eval_id,
            "config": config,
            "objectives": {
                "performance": float(eval_id + 1),
                "quality": {"combined": 1.0 / float(eval_id + 1)},
            },
            "optimizer_metadata": {
                "forced": eval_id == 7,
                "agent_proposal_round": max(0, eval_id - 3),
            },
        }
        for eval_id, config in enumerate(configs)
    ]

    restored = _make_opt()
    restored._advice_log_dir = str(tmp_path)
    restored.warm_start(evaluations)

    assert len(restored._configs) == len(evaluations)
    assert restored._last_accounted_eval_id == 7
    assert restored._normal_no_hv_gain == 0
    assert restored._stagnation_normalization is None


def test_resume_fails_closed_without_exact_optimizer_metadata():
    optimizer = _make_opt()
    evaluation = {
        "eval_id": 0,
        "config": {"a": 0, "b": 0, "c": 0, "d": 0},
        "objectives": {
            "performance": 1.0,
            "quality": {"combined": 0.1},
        },
    }

    with pytest.raises(RuntimeError, match="missing optimizer observation"):
        optimizer.warm_start([evaluation])


def test_corrupt_or_absent_forced_state_keeps_defaults(tmp_path):
    absent = _make_opt()
    absent._advice_log_dir = str(tmp_path / "absent")
    absent._load_forced_state()
    assert absent._normal_no_hv_gain == 0

    (tmp_path / "forced_state.json").write_text("{broken json")
    corrupt = _make_opt()
    corrupt._advice_log_dir = str(tmp_path)
    corrupt._load_forced_state()
    assert corrupt._normal_no_hv_gain == 0

    (tmp_path / "forced_state.json").write_text(json.dumps({
        "last_accounted_eval_id": 100,
        "forced": {
            "normal_no_hv_gain": "not-an-integer",
            "normalization": None,
        },
    }))
    partial = _make_opt()
    partial._advice_log_dir = str(tmp_path)
    partial._load_forced_state()
    assert partial._last_accounted_eval_id == -1


def test_hv_objective_is_always_equal_axis_normalized():
    opt = _warmed()
    acqf = opt._build_acqf()
    weights = acqf.objective.weights
    assert weights[opt._perf_idx].item() == pytest.approx(1 / 85.0, rel=1e-6)
    assert weights[opt._qual_idx].item() == pytest.approx(1 / 0.35, rel=1e-6)
    assert float(acqf.ref_point[opt._perf_idx]) == pytest.approx(5.0 / 85.0)
    assert float(acqf.ref_point[opt._qual_idx]) == pytest.approx(0.15 / 0.35)


def test_constant_axes_still_use_normalized_hv_objective():
    opt = _warmed()
    opt._perf = [10.0] * len(opt._perf)
    opt._corr = [0.5] * len(opt._corr)
    opt._fit_models()
    acqf = opt._build_acqf()
    weights = acqf.objective.weights
    assert weights[opt._perf_idx].item() == pytest.approx(0.1)
    assert weights[opt._qual_idx].item() == pytest.approx(2.0)
    assert float(acqf.ref_point[opt._perf_idx]) == pytest.approx(1.0)
    assert float(acqf.ref_point[opt._qual_idx]) == pytest.approx(1.0)


def test_axes_equal_to_reference_are_neutral_not_raw_scaled():
    opt = _warmed()
    opt._perf = [0.0] * len(opt._perf)
    opt._corr = [0.0] * len(opt._corr)
    opt._fit_models()
    acqf = opt._build_acqf()

    assert torch.equal(acqf.objective.weights, torch.zeros(2, **opt.tkwargs))
    assert torch.equal(acqf.ref_point, -torch.ones(2, **opt.tkwargs))
