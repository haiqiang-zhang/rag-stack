"""Self-contained regressions for LLM proposal deltas and DLS ancestry."""

from __future__ import annotations

import pytest
import torch
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.optimizer.agent.core import AgentGuidedQNEHVI
from rag_stack.optimizer.agent.llm_agent import AgentAdvice, LLMAgent
from rag_stack.search_space import AxSearchSpaceOps


OBJECTIVES = [
    {"objective_name": "performance", "cost": 1, "ref_point": 0},
    {"objective_name": "quality", "cost": 1, "ref_point": 0},
]


def _choice(name, values, parameter_type=ParameterType.INT):
    return ChoiceParameter(
        name=name,
        parameter_type=parameter_type,
        values=values,
        is_ordered=parameter_type.is_numeric,
        sort_values=parameter_type.is_numeric,
    )


def _space() -> SearchSpace:
    return SearchSpace(parameters=[
        _choice("model", ["small", "medium", "large"], ParameterType.STRING),
        _choice("chunk", [128, 256, 512]),
        _choice("top_k", [8, 16, 32]),
        _choice("inherited", [0, 1, 2]),
    ])


def _optimizer() -> AgentGuidedQNEHVI:
    space = _space()
    ssops = AxSearchSpaceOps(space)
    opt = AgentGuidedQNEHVI(
        space,
        objectives=OBJECTIVES,
        seed=11,
        n_both_init=0,
        init_sobol_k=0,
        ssops=ssops,
        n_candidates=8,
        enable_doe=False,
    )
    flats = [
        {"model": "small", "chunk": 128, "top_k": 8, "inherited": 0},
        {"model": "large", "chunk": 512, "top_k": 32, "inherited": 2},
        {"model": "small", "chunk": 512, "top_k": 16, "inherited": 1},
        {"model": "large", "chunk": 128, "top_k": 16, "inherited": 1},
    ]
    opt._x_rows = [
        ssops.clamp(ssops.encode_arm_params(row).reshape(1, -1)).reshape(-1)
        for row in flats
    ]
    opt._configs = [ssops.decode_to_config(row) for row in opt._x_rows]
    opt._perf = [100.0, 10.0, 60.0, 30.0]
    opt._corr = [0.1, 0.9, 0.4, 0.6]
    opt._sub_metrics = [{} for _ in flats]
    opt._eval_ids = list(range(len(flats)))
    return opt


def _history():
    return [
        {
            "flat": {"model": "small", "chunk": 128, "top_k": 8, "inherited": 0},
            "performance": 100.0,
            "correctness": 0.1,
        },
        {
            "flat": {"model": "large", "chunk": 512, "top_k": 32, "inherited": 2},
            "performance": 10.0,
            "correctness": 0.9,
        },
    ]


def test_equal_axis_hv_matches_observed_min_max_geometry():
    """Range scaling must translate the HV reference to the observed floor."""
    opt = _optimizer()
    opt._fit_models()

    acqf = opt._build_acqf()
    weights = acqf.objective.weights
    raw = torch.tensor(
        list(zip(opt._perf, opt._corr)),
        **opt.tkwargs,
    )
    minima = raw.amin(dim=0)
    ranges = raw.amax(dim=0) - minima

    assert torch.allclose(weights, ranges.reciprocal())
    assert torch.allclose(acqf.ref_point, minima / ranges)
    # Weighted samples relative to the weighted reference must be exactly the
    # same geometry as explicit (Y - min) / range normalization with ref=0.
    assert torch.allclose(
        raw * weights - acqf.ref_point,
        (raw - minima) / ranges,
    )


def test_acquisition_sampler_seed_is_round_derived_not_global_rng():
    """Skipping a discarded DOE pool must not perturb later QMC draws."""
    opt = _optimizer()
    opt._fit_models()
    opt._t = 7

    torch.manual_seed(1)
    first = opt._build_acqf().sampler.seed
    torch.manual_seed(999_999)
    second = opt._build_acqf().sampler.seed

    assert first == second == opt.seed + 300_000 + opt._t
    opt._t += 1
    assert opt._build_acqf().sampler.seed == first + 1


def test_parsed_delta_keeps_metadata_and_does_not_mutate_raw():
    ssops = AxSearchSpaceOps(_space())
    agent = LLMAgent(ai_client=object())
    parsed = {
        "reasoning": "fill the middle",
        "candidates": [
            {"base": "perf", "model": "medium", "chunk": 256},
            {"base": "quality", "top_k": 16},
        ],
    }

    advice = agent._build_advice_from_parsed(parsed, _history(), ssops)

    assert advice.centers is not None and advice.centers.shape[0] == 2
    assert advice.proposal_meta[0]["base"] == "perf"
    assert advice.proposal_meta[0]["deltas"] == {
        "model": "medium", "chunk": 256,
    }
    assert advice.proposal_meta[0]["inferred_ancestors"] == {}
    assert set(advice.proposal_meta[0]) == {
        "base", "deltas", "inferred_ancestors",
    }
    assert advice.raw["candidates"][0]["base"] == "perf"
    assert parsed["candidates"][1]["base"] == "quality"


def test_ordinary_agent_accepts_more_than_five_valid_deltas():
    space = SearchSpace(parameters=[
        _choice(f"knob_{index}", [0, 1])
        for index in range(6)
    ])
    ssops = AxSearchSpaceOps(space)
    base = {f"knob_{index}": 0 for index in range(6)}
    deltas = {f"knob_{index}": 1 for index in range(6)}
    history = [{
        "flat": base,
        "performance": 1.0,
        "correctness": 1.0,
    }]
    parsed = {
        "reasoning": "coordinated cross-stage change",
        "candidates": [{"base": "quality", **deltas}],
    }

    advice = LLMAgent(ai_client=object())._build_advice_from_parsed(
        parsed, history, ssops,
    )

    assert advice.n() == 1
    assert advice.proposal_meta[0]["deltas"] == deltas
    assert ssops.decode_flat(advice.centers[0]) == deltas


def test_anchor_agent_parser_preserves_the_same_delta_contract():
    from rag_stack.optimizer.botorch.greedy_then_hv_anchor_agent_qnehvi import (
        GreedyAnchorLLMAgent,
    )

    ssops = AxSearchSpaceOps(_space())
    agent = GreedyAnchorLLMAgent(ai_client=object())
    parsed = {
        "reasoning": "refine perf anchor",
        "candidates": [{"base": "perf", "model": "medium", "top_k": 16}],
    }

    advice = agent._build_advice_from_parsed(parsed, _history(), ssops)

    assert advice.proposal_meta[0]["base"] == "perf"
    assert advice.proposal_meta[0]["deltas"] == {
        "model": "medium",
        "top_k": 16,
    }
    assert advice.proposal_meta[0]["inferred_ancestors"] == {}
    assert advice.raw["candidates"][0]["base"] == "perf"


def test_anchor_agent_parser_requires_base_and_rejects_legacy_anchor_alias():
    from rag_stack.optimizer.botorch.greedy_then_hv_anchor_agent_qnehvi import (
        GreedyAnchorLLMAgent,
    )

    ssops = AxSearchSpaceOps(_space())
    advice = GreedyAnchorLLMAgent(ai_client=object())._build_advice_from_parsed(
        {
            "reasoning": "legacy shape",
            "candidates": [{"anchor": "perf", "model": "medium"}],
        },
        _history(),
        ssops,
    )

    assert advice.n() == 0


def test_advice_proposal_metadata_fails_closed_when_misaligned():
    centers = torch.zeros((2, 4), dtype=torch.double)
    with pytest.raises(ValueError, match="one-to-one"):
        AgentAdvice(centers=centers, proposal_meta=[{"deltas": {}}])


def test_lean_is_not_a_reserved_proposal_key():
    space = SearchSpace(parameters=[
        _choice("lean", [0, 1]),
        _choice("other", [0, 1]),
    ])
    ssops = AxSearchSpaceOps(space)

    center, meta = LLMAgent(ai_client=object())._encode_proposal_with_metadata(
        {"lean": 1},
        ssops,
        {"lean": 0, "other": 0},
    )

    assert center is not None
    assert ssops.decode_flat(center[0])["lean"] == 1
    assert meta == {"deltas": {"lean": 1}, "inferred_ancestors": {}}


def test_explicit_hierarchical_child_preserves_required_selector_ancestor():
    selector = ChoiceParameter(
        name="selector",
        parameter_type=ParameterType.STRING,
        values=["a", "b"],
        is_ordered=False,
        sort_values=False,
        dependents={"a": ["a_knob"], "b": ["b_knob"]},
    )
    space = SearchSpace(parameters=[
        selector,
        _choice("a_knob", [0, 1]),
        _choice("b_knob", [0, 1]),
    ])
    ssops = AxSearchSpaceOps(space)
    history = [
        {
            "flat": {"selector": "a", "a_knob": 0},
            "performance": 50.0,
            "correctness": 0.2,
        },
        {
            "flat": {"selector": "b", "b_knob": 1},
            "performance": 10.0,
            "correctness": 0.8,
        },
    ]
    parsed = {
        "reasoning": "refine branch a",
        "candidates": [{"base": "perf", "a_knob": 1}],
    }

    advice = LLMAgent(ai_client=object())._build_advice_from_parsed(
        parsed, history, ssops,
    )

    assert advice.proposal_meta[0]["deltas"] == {"a_knob": 1}
    assert advice.proposal_meta[0]["inferred_ancestors"] == {}
    assert set(advice.proposal_meta[0]) == {
        "base", "deltas", "inferred_ancestors",
    }
    decoded = ssops.decode_flat(advice.centers[0])
    assert decoded["selector"] == "a"
    assert decoded["a_knob"] == 1


def test_inactive_explicit_child_infers_unique_ancestor_path():
    enabled = ChoiceParameter(
        name="enabled",
        parameter_type=ParameterType.STRING,
        values=["false", "true"],
        is_ordered=False,
        sort_values=False,
        dependents={"false": [], "true": ["component"]},
    )
    component = ChoiceParameter(
        name="component",
        parameter_type=ParameterType.STRING,
        values=["a", "b"],
        is_ordered=False,
        sort_values=False,
        dependents={"a": ["a_knob"], "b": ["b_knob"]},
    )
    space = SearchSpace(parameters=[
        enabled,
        component,
        _choice("a_knob", [0, 1]),
        _choice("b_knob", [0, 1]),
    ])
    ssops = AxSearchSpaceOps(space)
    anchor = {
        "enabled": "false", "component": "b", "a_knob": 0, "b_knob": 0,
    }

    center, meta = LLMAgent(ai_client=object())._encode_proposal_with_metadata(
        {"a_knob": 1}, ssops, anchor,
    )

    assert center is not None
    decoded = ssops.decode_flat(center[0])
    assert decoded["enabled"] == "true"
    assert decoded["component"] == "a"
    assert decoded["a_knob"] == 1
    assert meta["inferred_ancestors"] == {
        "component": "a",
        "enabled": "true",
    }
    assert meta["deltas"] == {"a_knob": 1}
    assert set(meta) == {"deltas", "inferred_ancestors"}


def test_explicit_selector_child_conflict_rejects_proposal():
    enabled = ChoiceParameter(
        name="enabled",
        parameter_type=ParameterType.STRING,
        values=["false", "true"],
        is_ordered=False,
        sort_values=False,
        dependents={"false": [], "true": ["child"]},
    )
    space = SearchSpace(parameters=[enabled, _choice("child", [0, 1])])
    ssops = AxSearchSpaceOps(space)

    center, meta = LLMAgent(ai_client=object())._encode_proposal_with_metadata(
        {"enabled": "false", "child": 1},
        ssops,
        {"enabled": "false", "child": 0},
    )

    assert center is None
    assert meta["inferred_ancestors"] == {}
    assert "disables explicit child" in meta["error"]


def test_children_from_mutually_exclusive_branches_fail_closed():
    selector = ChoiceParameter(
        name="selector",
        parameter_type=ParameterType.STRING,
        values=["a", "b"],
        is_ordered=False,
        sort_values=False,
        dependents={"a": ["a_knob"], "b": ["b_knob"]},
    )
    space = SearchSpace(parameters=[
        selector, _choice("a_knob", [0, 1]), _choice("b_knob", [0, 1]),
    ])
    ssops = AxSearchSpaceOps(space)

    center, meta = LLMAgent(ai_client=object())._encode_proposal_with_metadata(
        {"a_knob": 1, "b_knob": 1},
        ssops,
        {"selector": "a", "a_knob": 0, "b_knob": 0},
    )

    assert center is None
    assert meta["inferred_ancestors"] == {}
    assert "conflicting 'selector' branches" in meta["error"]


def test_llm_centers_are_one_official_multistart_dls(monkeypatch):
    from rag_stack.optimizer.agent import dls as dls_module

    opt = _optimizer()
    opt._fit_models()
    opt._gradient_polish = lambda *_args, **_kwargs: None
    centers = torch.cat([
        opt.ssops.encode_arm_params({
            "model": "medium", "chunk": 256, "top_k": 8, "inherited": 0,
        }).reshape(1, -1),
        opt.ssops.encode_arm_params({
            "model": "small", "chunk": 256, "top_k": 32, "inherited": 2,
        }).reshape(1, -1),
    ])
    advice = AgentAdvice(centers=centers)
    original_centers = advice.centers.clone()
    calls = []

    def fake_dls(
        acq_function, discrete_choices, q, num_restarts,
        batch_initial_conditions, X_avoid, max_batch_size,
    ):
        calls.append((
            discrete_choices,
            q,
            num_restarts,
            batch_initial_conditions.clone(),
        ))
        return batch_initial_conditions[:1, 0, :], None

    monkeypatch.setattr(
        dls_module, "optimize_acqf_discrete_local_search", fake_dls,
    )

    out = opt._dls_candidates_grouped(lambda x: x.sum(dim=-1), advice)

    assert out is not None
    _, provenance = out
    assert provenance.count("dls-llm") == 1
    choices, q, num_restarts, initial_conditions = calls[0]
    assert all(len(levels) > 1 for levels in choices)
    assert q == 1
    assert num_restarts == centers.shape[0] == 2
    assert initial_conditions.shape == (2, 1, opt.dim)
    assert torch.equal(initial_conditions[:, 0, :], centers)
    assert torch.equal(advice.centers, original_centers)


def test_quasi_random_seeds_use_official_multistart_dls(monkeypatch):
    from rag_stack.optimizer.agent import dls as dls_module

    opt = _optimizer()
    opt._fit_models()
    opt._gradient_polish = lambda *_args, **_kwargs: None

    def _rows(configs):
        return torch.cat([
            opt.ssops.encode_arm_params(config).reshape(1, -1)
            for config in configs
        ])

    random_seeds = _rows([
        {"model": "medium", "chunk": 128, "top_k": 8, "inherited": 0},
        {"model": "medium", "chunk": 256, "top_k": 16, "inherited": 1},
        {"model": "medium", "chunk": 512, "top_k": 32, "inherited": 2},
    ])
    pool_seeds = _rows([
        {"model": "small", "chunk": 256, "top_k": 8, "inherited": 1},
        {"model": "large", "chunk": 256, "top_k": 32, "inherited": 0},
    ])
    monkeypatch.setattr(
        opt,
        "_restart_seed_groups",
        lambda _advice: {"random": random_seeds},
    )

    calls = []

    def fake_dls(
        acq_function, discrete_choices, q, num_restarts,
        batch_initial_conditions, X_avoid, max_batch_size,
    ):
        calls.append((
            q, num_restarts, batch_initial_conditions.clone(),
        ))
        return batch_initial_conditions[0, 0].reshape(1, -1), None

    monkeypatch.setattr(
        dls_module, "optimize_acqf_discrete_local_search", fake_dls,
    )

    out = opt._dls_candidates_grouped(
        lambda x: x.sum(dim=-1), None, pool_seeds=pool_seeds,
    )

    assert out is not None
    _, provenance = out
    assert provenance.count("dls-random") == 1
    assert provenance.count("dls-pool") == 1
    assert len(calls) == 1 + 2 + 1  # random + perf/quality + pool
    random_q, random_restarts, random_initial = calls[0]
    pool_q, pool_restarts, pool_initial = calls[-1]
    assert random_q == pool_q == 1
    assert random_restarts == random_seeds.shape[0] == 3
    assert pool_restarts == pool_seeds.shape[0] == 2
    assert torch.equal(random_initial[:, 0, :], random_seeds)
    assert torch.equal(pool_initial[:, 0, :], pool_seeds)
    channels = opt._dls_channel_diagnostics["channels"]
    assert channels["dls-random"]["attempted_starts"] == 3
    assert channels["dls-random"]["returned_endpoints"] == 1
    assert channels["dls-pool"]["attempted_starts"] == 2
    assert channels["dls-pool"]["returned_endpoints"] == 1


def test_multistart_source_endpoints_share_outer_dedup_and_acqf_arbitration(
    monkeypatch,
):
    """Each source returns one best endpoint; the common pool arbitrates them."""
    from rag_stack.optimizer.agent import dls as dls_module

    opt = _optimizer()
    opt.dls_num_restarts = 2
    opt._gradient_polish = lambda *_args, **_kwargs: None

    def _rows(configs):
        return torch.cat([
            opt.ssops.encode_arm_params(config).reshape(1, -1)
            for config in configs
        ])

    agent_seeds = _rows([
        {"model": "medium", "chunk": 128, "top_k": 8, "inherited": 1},
        {"model": "medium", "chunk": 256, "top_k": 16, "inherited": 2},
    ])
    random_seeds = _rows([
        {"model": "small", "chunk": 256, "top_k": 8, "inherited": 2},
        {"model": "large", "chunk": 256, "top_k": 16, "inherited": 0},
    ])
    breadth_seeds = _rows([
        {"model": "small", "chunk": 128, "top_k": 32, "inherited": 2},
        {"model": "large", "chunk": 512, "top_k": 8, "inherited": 0},
    ])
    endpoints = _rows([
        {"model": "medium", "chunk": 512, "top_k": 32, "inherited": 0},
        {"model": "small", "chunk": 256, "top_k": 16, "inherited": 1},
    ])
    # The random source and Agent source converge to the same best endpoint.
    # Both source calls return it; the outer pool removes the duplicate.
    primary_returns = [endpoints[0], endpoints[0], endpoints[1]]

    monkeypatch.setattr(
        opt,
        "_restart_seed_groups",
        lambda advice: {
            "llm": advice.centers,
            "random": random_seeds,
        },
    )
    monkeypatch.setattr(opt, "_hier_pool_sobol", lambda _n: breadth_seeds)

    scored_batches = []

    def lognehvi(X):
        rows = X.reshape(-1, opt.dim)
        scored_batches.append(rows.clone())
        values = []
        for row in rows:
            if torch.equal(row, endpoints[1]):
                values.append(100.0)
            elif torch.equal(row, breadth_seeds[0]):
                values.append(20.0)
            elif torch.equal(row, breadth_seeds[1]):
                values.append(10.0)
            else:
                values.append(0.0)
        return torch.tensor(values, **opt.tkwargs)

    monkeypatch.setattr(opt, "_build_acqf", lambda: lognehvi)

    primary_calls = []

    def fake_dls(
        acq_function, discrete_choices, q, num_restarts,
        batch_initial_conditions, X_avoid, max_batch_size,
    ):
        call = (
            acq_function,
            q,
            num_restarts,
            batch_initial_conditions.clone(),
        )
        if acq_function is lognehvi:
            primary_calls.append(call)
            endpoint = primary_returns[len(primary_calls) - 1]
        else:
            # Keep the grouped perf/quality enrichment out of this wiring
            # test's final pool by making its synthetic endpoint evaluated.
            endpoint = opt._x_rows[0]
        return endpoint.reshape(1, -1), None

    monkeypatch.setattr(
        dls_module, "optimize_acqf_discrete_local_search", fake_dls,
    )
    selected_sources = []
    monkeypatch.setattr(
        opt, "_mark_dls_selected", selected_sources.append,
    )

    result = opt._propose_pool(
        AgentAdvice(centers=agent_seeds),
        forced_due=False,
        forced_gap=None,
    )

    assert len(primary_calls) == 3
    for call, expected_starts in zip(
        primary_calls, (agent_seeds, random_seeds, breadth_seeds),
    ):
        fn, q, num_restarts, initial = call
        assert fn is lognehvi
        assert q == 1
        assert num_restarts == expected_starts.shape[0] == 2
        assert initial.shape == (2, 1, opt.dim)
        assert torch.equal(initial[:, 0, :], expected_starts)

    channels = opt._dls_channel_diagnostics["channels"]
    assert channels["dls-llm"]["returned_endpoints"] == 1
    assert channels["dls-random"]["returned_endpoints"] == 1
    assert channels["dls-pool"]["returned_endpoints"] == 1
    assert (
        channels["dls-random"]["returned_decoded_duplicate_other_channel"]
        == 1
    )

    # One pre-score chooses the two breadth restarts; one final batch is the
    # sole arbitration over raw Sobol, raw Agent, and source-best endpoints.
    assert len(scored_batches) == 2
    expected_final_pool = torch.cat([
        breadth_seeds, agent_seeds, endpoints,
    ])
    assert torch.equal(scored_batches[-1], expected_final_pool)
    assert sum(
        torch.equal(row, endpoints[0]) for row in scored_batches[-1]
    ) == 1
    assert result == opt.ssops.decode_to_config(endpoints[1])
    assert selected_sources == ["dls-pool"]
