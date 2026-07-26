"""Self-contained contracts for champion-anchored DOE semantics."""

from __future__ import annotations

import pytest
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
    return (
        DOEController(
            ssops,
            0,
            1,
            enabled=True,
        ),
        ssops,
    )


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


def _plan(controller: DOEController, t: int = 0):
    return controller.plan_round(
        t=t,
        accept_candidate=lambda *_args: True,
    )


def _install_two_factor_history(
    controller: DOEController,
    ssops: AxSearchSpaceOps,
    *,
    performance_scale: float = 1.0,
    performance_shift: float = 0.0,
    quality_scale: float = 1.0,
    quality_shift: float = 0.0,
) -> None:
    observations = [
        ({"factor_a": 0, "factor_b": 0, "context": 0.0}, 100.0, 0.10),
        ({"factor_a": 2, "factor_b": 2, "context": 1.0}, 10.0, 0.90),
        ({"factor_a": 1, "factor_b": 0, "context": 0.0}, 80.0, 0.50),
        ({"factor_a": 0, "factor_b": 1, "context": 0.0}, 70.0, 0.60),
    ]
    for eval_id, (config, performance, quality) in enumerate(observations):
        _observe(
            controller,
            ssops,
            config,
            performance_scale * performance + performance_shift,
            quality_scale * quality + quality_shift,
            eval_id,
        )


def _candidate_identity(ssops: AxSearchSpaceOps, candidate) -> tuple:
    return (
        candidate.kind,
        ssops.decode_flat(candidate.x),
    )


def test_objective_champions_are_the_two_doe_anchors():
    controller, ssops = _controller()
    _install_two_factor_history(controller, ssops)

    performance_champion = controller._observations[0]
    quality_champion = controller._observations[1]
    plan = _plan(controller, 4)

    assert controller._champion_indices() == (0, 1)
    assert plan.audit.reason == "champion_tension"
    assert any(phase.startswith("p:") for phase in plan.audit.phases)
    assert any(phase.startswith("q:") for phase in plan.audit.phases)
    assert plan.pool
    for candidate in plan.pool:
        assert min(
            len(controller._active_diff_dims(anchor.x, candidate.x))
            for anchor in (performance_champion, quality_champion)
        ) in (1, 2)


def test_only_history_backed_opposite_sign_level_effect_is_a_factor():
    search_space = SearchSpace(parameters=[
        _choice("tradeoff_factor", [0, 1, 2]),
        _choice("aligned_factor", [0, 1, 2]),
        RangeParameter(
            name="context",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
    ])
    controller, ssops = _controller(search_space)
    observations = [
        ({"tradeoff_factor": 0, "aligned_factor": 0, "context": 0.0}, 100, 0.10),
        ({"tradeoff_factor": 2, "aligned_factor": 0, "context": 1.0}, 10, 0.90),
        ({"tradeoff_factor": 1, "aligned_factor": 0, "context": 0.2}, 70, 0.50),
        # This level improves both normalized objectives relative to the
        # aggregate level-zero mean, so it is not tension evidence.
        ({"tradeoff_factor": 0, "aligned_factor": 1, "context": 0.4}, 80, 0.70),
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

    plan = _plan(controller, 4)
    tradeoff_dim = ssops.param_names.index("tradeoff_factor")
    aligned_dim = ssops.param_names.index("aligned_factor")

    assert plan.pool
    champions = [
        controller._observations[index].x
        for index in controller._champion_indices()
    ]
    assert all(
        (tradeoff_dim,) in {
            controller._active_diff_dims(anchor, candidate.x)
            for anchor in champions
        }
        for candidate in plan.pool
    )
    assert all(
        aligned_dim not in controller._active_diff_dims(anchor, candidate.x)
        for anchor in champions
        for candidate in plan.pool
    )
    assert plan.audit.phases == (
        "p:oat_discovery(k=1)",
        "q:oat_discovery(k=1)",
    )
    assert plan.audit.pair_cells == 0


def test_positive_affine_objective_rescaling_preserves_the_doe_plan():
    original, original_ssops = _controller()
    transformed, transformed_ssops = _controller()
    _install_two_factor_history(original, original_ssops)
    _install_two_factor_history(
        transformed,
        transformed_ssops,
        performance_scale=7.0,
        performance_shift=-300.0,
        quality_scale=0.125,
        quality_shift=4.0,
    )

    original_plan = _plan(original, 4)
    transformed_plan = _plan(transformed, 4)

    assert [
        _candidate_identity(original_ssops, candidate)
        for candidate in original_plan.pool
    ] == [
        _candidate_identity(transformed_ssops, candidate)
        for candidate in transformed_plan.pool
    ]
    assert original_plan.audit.phases == transformed_plan.audit.phases
    assert original_plan.audit.tension_score == pytest.approx(
        transformed_plan.audit.tension_score,
    )


def test_effective_support_selects_one_or_multiple_factors_from_history():
    one_factor_space = SearchSpace(parameters=[
        _choice("tradeoff_factor", [0, 1, 2]),
        _choice("aligned_factor", [0, 1, 2]),
        RangeParameter(
            name="context",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
    ])
    one_factor, one_ssops = _controller(one_factor_space)
    one_factor_history = [
        ({"tradeoff_factor": 0, "aligned_factor": 0, "context": 0.0}, 100, 0.10),
        ({"tradeoff_factor": 2, "aligned_factor": 0, "context": 1.0}, 10, 0.90),
        ({"tradeoff_factor": 1, "aligned_factor": 0, "context": 0.2}, 70, 0.50),
        ({"tradeoff_factor": 0, "aligned_factor": 1, "context": 0.4}, 80, 0.70),
    ]
    for eval_id, (config, performance, quality) in enumerate(one_factor_history):
        _observe(
            one_factor,
            one_ssops,
            config,
            performance,
            quality,
            eval_id,
        )

    two_factor, two_ssops = _controller()
    _install_two_factor_history(two_factor, two_ssops)

    one_plan = _plan(one_factor, 4)
    two_plan = _plan(two_factor, 4)

    assert one_plan.audit.phases == (
        "p:oat_discovery(k=1)",
        "q:oat_discovery(k=1)",
    )
    assert one_plan.audit.pair_cells == 0
    assert two_plan.audit.phases == (
        "p:pair_confirmation(k=2)",
        "q:oat_discovery(k=2)",
    )
    assert two_plan.audit.pair_cells == 1
    pair_cells = [
        candidate for candidate in two_plan.pool
        if candidate.kind == "pair"
    ]
    assert len(pair_cells) == 1
    pair_config = two_ssops.decode_flat(pair_cells[0].x)
    performance_champion = two_factor._observations[
        two_factor._champion_indices()[0]
    ]
    assert {
        two_ssops.param_names[dim]
        for dim in two_factor._active_diff_dims(
            performance_champion.x,
            pair_cells[0].x,
        )
    } == {"factor_a", "factor_b"}
    assert (pair_config["factor_a"], pair_config["factor_b"]) == (1, 1)


def test_same_valued_champions_can_expose_a_useful_third_level():
    search_space = SearchSpace(parameters=[
        _choice("shared_factor", [0, 1, 2]),
        RangeParameter(
            name="context",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
    ])
    controller, ssops = _controller(search_space)
    observations = [
        ({"shared_factor": 0, "context": 0.0}, 100, 0.10),
        ({"shared_factor": 0, "context": 1.0}, 10, 0.90),
        # A confounded historical observation identifies level 1 as a
        # trade-off.  DOE then grafts that level onto the champion context.
        ({"shared_factor": 1, "context": 0.5}, 40, 0.75),
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

    plan = _plan(controller, 3)
    champion_configs = [
        ssops.decode_flat(controller._observations[index].x)
        for index in controller._champion_indices()
    ]
    factor_dim = ssops.param_names.index("shared_factor")

    assert [config["shared_factor"] for config in champion_configs] == [0, 0]
    assert any(
        controller._active_diff_dims(
            controller._observations[0].x,
            candidate.x,
        ) == (factor_dim,)
        and ssops.decode_flat(candidate.x)["shared_factor"] == 1
        for candidate in plan.pool
    )


def test_endpoint_tradeoff_stays_eligible_and_unseen_level_has_no_evidence_score():
    search_space = SearchSpace(parameters=[
        _choice("tradeoff_factor", [0, 1, 2]),
        RangeParameter(
            name="context",
            parameter_type=ParameterType.FLOAT,
            lower=0.0,
            upper=1.0,
        ),
    ])
    controller, ssops = _controller(search_space)
    _observe(
        controller,
        ssops,
        {"tradeoff_factor": 0, "context": 0.0},
        100.0,
        0.10,
        0,
    )
    _observe(
        controller,
        ssops,
        {"tradeoff_factor": 2, "context": 1.0},
        10.0,
        0.90,
        1,
    )

    plan = _plan(controller, 2)
    candidates = [ssops.decode_flat(candidate.x) for candidate in plan.pool]
    evidence = controller._ranked_screening_evidence(
        anchor_idx=controller._champion_indices()[0],
        side="p",
        points=controller._normalized_y(),
    )

    # A complete normalized loss on the preserved endpoint is still the
    # clearest possible trade-off; it must not be multiplied down to zero.
    assert [
        candidate["tradeoff_factor"]
        for candidate in candidates
        if candidate["context"] == 0.0
    ] == [2, 1]
    assert [
        candidate["tradeoff_factor"]
        for candidate in candidates
        if candidate["context"] == 1.0
    ] == [0, 1]
    assert len(evidence) == 1
    observed_utilities = dict(evidence[0].observed_level_utilities)
    endpoint_key = controller._level_key(2)
    unseen_key = controller._level_key(1)
    assert evidence[0].score > 0.0
    assert observed_utilities[endpoint_key] > 0.0
    # Level 1 is a valid exploration cell but has never been observed.  Keeping
    # it in the pool must not make it look like landed level-mean evidence.
    assert unseen_key not in observed_utilities


def test_landed_oat_card_is_exact_champion_ablation_not_screening_mean():
    controller, ssops = _controller()
    _install_two_factor_history(controller, ssops)

    cards = controller.ablation_card_payload()
    cards_by_parameter = {card["parameter"]: card for card in cards}

    assert set(cards_by_parameter) == {"factor_a", "factor_b"}
    assert all(
        card["evidence_type"] == "landed_champion_oat"
        and card["side"] == "p"
        for card in cards
    )
    factor_a = cards_by_parameter["factor_a"]
    assert factor_a["kind"] == "oat"
    assert factor_a["anchor_level"] == 0
    assert factor_a["observed_level"] == 1
    assert factor_a["changed_knobs"] == [{
        "parameter": "factor_a",
        "dim": ssops.param_names.index("factor_a"),
        "anchor_level": 0,
        "observed_level": 1,
    }]
    assert factor_a["delivery"] == "arbitrated"
    assert factor_a["interaction"] is None
    assert factor_a["delta_performance"] == pytest.approx(-20.0 / 90.0)
    assert factor_a["delta_quality"] == pytest.approx(0.40 / 0.80)
    assert factor_a["utility"] == pytest.approx(
        (0.40 / 0.80) / (1.0 + 20.0 / 90.0),
    )


def test_unrelated_two_factor_levels_do_not_suppress_intended_pair():
    controller, ssops = _controller()
    _install_two_factor_history(controller, ssops)
    # This is a landed two-factor row relative to the performance champion, but
    # it is not the intended pair of the two best exact OAT levels (1, 1).
    _observe(
        controller,
        ssops,
        {"factor_a": 2, "factor_b": 2, "context": 0.0},
        45.0,
        0.75,
        4,
    )

    plan = _plan(controller, 5)
    pairs = [candidate for candidate in plan.pool if candidate.kind == "pair"]

    assert len(pairs) == 1
    pair_config = ssops.decode_flat(pairs[0].x)
    assert (pair_config["factor_a"], pair_config["factor_b"]) == (1, 1)


def test_landed_pair_card_reports_exact_second_order_residual():
    controller, ssops = _controller()
    _install_two_factor_history(controller, ssops)
    pairs = [
        candidate for candidate in _plan(controller, 4).pool
        if candidate.kind == "pair"
    ]
    assert len(pairs) == 1
    pair = pairs[0]

    controller.mark_selected(pair)
    controller.observe(pair.x, (55.0, 0.58), eval_id=4, forced=True)
    pair_cards = [
        card for card in controller.ablation_card_payload()
        if card["kind"] == "pair"
    ]

    assert len(pair_cards) == 1
    card = pair_cards[0]
    assert card["delivery"] == "forced"
    assert [
        (
            change["parameter"],
            change["anchor_level"],
            change["observed_level"],
        )
        for change in card["changed_knobs"]
    ] == [
        ("factor_a", 0, 1),
        ("factor_b", 0, 1),
    ]
    interaction = card["interaction"]
    assert interaction["available"] is True
    assert interaction["main_effect_eval_ids"] == [2, 3]
    pair_delta_performance = (55.0 - 100.0) / 90.0
    pair_delta_quality = (0.58 - 0.10) / 0.80
    oat_delta_performance = (
        (80.0 - 100.0) / 90.0 + (70.0 - 100.0) / 90.0
    )
    oat_delta_quality = (
        (0.50 - 0.10) / 0.80 + (0.60 - 0.10) / 0.80
    )
    assert interaction["delta_performance"] == pytest.approx(
        pair_delta_performance - oat_delta_performance,
    )
    assert interaction["delta_quality"] == pytest.approx(
        pair_delta_quality - oat_delta_quality,
    )


def test_pair_card_does_not_infer_missing_main_effect():
    controller, ssops = _controller()
    observations = [
        ({"factor_a": 0, "factor_b": 0, "context": 0.0}, 100, 0.10),
        ({"factor_a": 2, "factor_b": 2, "context": 1.0}, 10, 0.90),
        ({"factor_a": 1, "factor_b": 0, "context": 0.0}, 80, 0.50),
        # Exact top-two pair, but there is no champion-relative factor_b=1 OAT.
        ({"factor_a": 1, "factor_b": 1, "context": 0.0}, 55, 0.65),
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

    pair_cards = [
        card for card in controller.ablation_card_payload()
        if card["kind"] == "pair"
    ]

    assert len(pair_cards) == 1
    interaction = pair_cards[0]["interaction"]
    assert interaction == {
        "available": False,
        "delta_performance": None,
        "delta_quality": None,
        "main_effect_eval_ids": [2],
    }
