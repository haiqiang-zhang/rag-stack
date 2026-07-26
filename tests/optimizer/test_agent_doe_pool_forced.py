"""Self-contained contracts shared by the DOE pool and unified forced set."""

from __future__ import annotations

import json

import torch
from ax.core.parameter import ChoiceParameter, ParameterType, RangeParameter
from ax.core.search_space import SearchSpace

from rag_stack.optimizer.agent.doe import DOEController
from rag_stack.search_space import AxSearchSpaceOps


def _choice(
    name: str,
    values,
    *,
    parameter_type: ParameterType = ParameterType.INT,
    dependents=None,
) -> ChoiceParameter:
    return ChoiceParameter(
        name=name,
        parameter_type=parameter_type,
        values=list(values),
        is_ordered=False,
        sort_values=parameter_type.is_numeric,
        dependents=dependents,
    )


def _factor_space() -> SearchSpace:
    return SearchSpace(parameters=[
        _choice("factor_a", [0, 1, 2]),
        _choice("factor_b", [0, 1, 2]),
        RangeParameter(
            name="context",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
    ])


def _controller(
    search_space: SearchSpace | None = None,
) -> tuple[DOEController, AxSearchSpaceOps]:
    ssops = AxSearchSpaceOps(search_space or _factor_space())
    controller = DOEController(
        ssops,
        0,
        1,
        enabled=True,
    )
    return controller, ssops


def _row(ssops: AxSearchSpaceOps, config: dict) -> torch.Tensor:
    return ssops.clamp(
        ssops.encode_arm_params(config).reshape(1, -1),
    ).reshape(-1)


def _observe(
    controller: DOEController,
    ssops: AxSearchSpaceOps,
    config: dict,
    performance: float,
    quality: float,
    eval_id: int,
) -> None:
    controller.observe(
        _row(ssops, config),
        (performance, quality),
        eval_id,
    )


def _install_pair_ready_history(
    controller: DOEController,
    ssops: AxSearchSpaceOps,
) -> None:
    observations = [
        ({"factor_a": 0, "factor_b": 0, "context": 0.0}, 100, 0.10),
        ({"factor_a": 2, "factor_b": 2, "context": 1.0}, 10, 0.90),
        ({"factor_a": 1, "factor_b": 0, "context": 0.0}, 80, 0.50),
        ({"factor_a": 0, "factor_b": 1, "context": 0.0}, 70, 0.60),
    ]
    for eval_id, (config, performance, quality) in enumerate(observations):
        _observe(
            controller,
            ssops,
            config,
            performance,
            quality,
            eval_id,
        )


def _plan(controller: DOEController, *, t: int = 4, accept=None):
    return controller.plan_round(
        t=t,
        accept_candidate=accept or (lambda *_args: True),
    )


def _signature_set(controller: DOEController, candidates) -> set[str]:
    return {
        controller._row_signature(candidate.x)
        for candidate in candidates
    }


def test_pool_contains_oat_and_exactly_one_bounded_top_two_pair():
    controller, ssops = _controller()
    _install_pair_ready_history(controller, ssops)

    plan = _plan(controller)
    pair_cells = [
        candidate for candidate in plan.pool if candidate.kind == "pair"
    ]
    oat_cells = [
        candidate for candidate in plan.pool if candidate.kind == "oat"
    ]

    assert oat_cells
    assert len(pair_cells) == 1
    assert pair_cells[0] is plan.pool[0]
    champions = [
        controller._observations[index].x
        for index in controller._champion_indices()
    ]
    anchor = champions[0]
    assert controller._active_diff_dims(
        anchor,
        pair_cells[0].x,
    ) == tuple(sorted((
        ssops.param_names.index("factor_a"),
        ssops.param_names.index("factor_b"),
    )))
    assert all(
        min(
            len(controller._active_diff_dims(champion, candidate.x))
            for champion in champions
        ) == 1
        for candidate in oat_cells
    )
    assert plan.audit.oat_cells == len(oat_cells)
    assert plan.audit.pair_cells == 1


def test_pair_waits_for_both_exact_champion_oat_main_effects():
    controller, ssops = _controller()
    observations = [
        ({"factor_a": 0, "factor_b": 0, "context": 0.0}, 100, 0.10),
        ({"factor_a": 2, "factor_b": 2, "context": 1.0}, 10, 0.90),
        # These rows establish marginal tension for both factors, but context
        # also changes, so neither is an exact champion-relative OAT result.
        ({"factor_a": 1, "factor_b": 0, "context": 0.25}, 80, 0.50),
        ({"factor_a": 0, "factor_b": 1, "context": 0.50}, 70, 0.60),
    ]
    for eval_id, (config, performance, quality) in enumerate(observations):
        _observe(
            controller,
            ssops,
            config,
            performance,
            quality,
            eval_id,
        )

    plan = _plan(controller)

    assert plan.audit.oat_cells > 0
    assert plan.audit.pair_cells == 0
    assert all(candidate.kind == "oat" for candidate in plan.pool)
    assert plan.audit.phases == (
        "p:oat_discovery(k=2)",
        "q:oat_discovery(k=2)",
    )


def test_pool_shares_legality_novelty_and_deduplication_across_consumers():
    controller, ssops = _controller()
    _install_pair_ready_history(controller, ssops)
    rejected = {"factor_a": 2, "factor_b": 0, "context": 0.0}

    def accept(_x, config):
        return config != rejected

    plan = _plan(controller, accept=accept)
    configs = [ssops.decode_flat(candidate.x) for candidate in plan.pool]
    signatures = [
        controller._row_signature(candidate.x)
        for candidate in plan.pool
    ]
    observed = {
        controller._row_signature(observation.x)
        for observation in controller._observations
    }

    assert rejected not in configs
    assert len(signatures) == len(set(signatures))
    assert observed.isdisjoint(signatures)


def test_candidates_preserve_the_champion_hierarchy_branch():
    search_space = SearchSpace(parameters=[
        _choice("branch_a_factor", [0, 1, 2]),
        _choice("branch_b_factor", [0, 1, 2]),
        _choice(
            "branch",
            ["a", "b"],
            parameter_type=ParameterType.STRING,
            dependents={
                "a": ["branch_a_factor"],
                "b": ["branch_b_factor"],
            },
        ),
        RangeParameter(
            name="context",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
    ])
    controller, ssops = _controller(search_space)
    observations = [
        (
            {"branch": "a", "branch_a_factor": 0, "context": 0.0},
            100,
            0.10,
        ),
        (
            {"branch": "b", "branch_b_factor": 2, "context": 1.0},
            10,
            0.90,
        ),
        (
            {"branch": "a", "branch_a_factor": 1, "context": 0.5},
            60,
            0.60,
        ),
    ]
    for eval_id, (config, performance, quality) in enumerate(observations):
        _observe(
            controller,
            ssops,
            config,
            performance,
            quality,
            eval_id,
        )

    plan = _plan(controller, t=3)
    champion_branch = controller._branch_id(controller._observations[0].x)

    assert plan.pool
    for candidate in plan.pool:
        config = ssops.decode_flat(candidate.x)
        active = ssops.active_params(config)
        assert controller._branch_id(candidate.x) == champion_branch
        assert config["branch"] == "a"
        assert "branch_a_factor" in active
        assert "branch_b_factor" not in active


def test_landed_pair_remains_bounded_after_resume_and_duplicate_observe():
    controller, ssops = _controller()
    _install_pair_ready_history(controller, ssops)
    pairs = [
        candidate for candidate in _plan(controller).pool
        if candidate.kind == "pair"
    ]
    assert len(pairs) == 1
    pair = pairs[0]

    controller.mark_selected(pair)
    controller.observe(pair.x, (55.0, 0.55), eval_id=4, forced=True)
    # A resume/replay may encounter the same durable result again.  It must
    # neither add GP/DOE weight nor create another pair confirmation.
    controller.observe(pair.x, (55.0, 0.55), eval_id=9, forced=True)
    state = json.loads(json.dumps(controller.state_dict()))

    restored, _ = _controller()
    restored.load_state_dict(state)
    following = _plan(restored, t=5)
    pair_signature = restored._row_signature(pair.x)

    assert len(restored._observations) == 5
    forced_eval_ids = {
        row["eval_id"]
        for row in restored.state_dict()["observations"]
        if row["forced"]
    }
    assert forced_eval_ids == {4}
    assert pair.era_id in restored.state_dict()["pair_landed"]
    assert pair_signature not in _signature_set(restored, following.pool)
    assert all(candidate.kind != "pair" for candidate in following.pool)
