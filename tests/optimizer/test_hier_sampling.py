"""Self-contained hierarchy-aware optimizer and DOE invariants."""

import pytest
import torch
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace


from rag_stack.optimizer.agent.doe import DOEController
from rag_stack.search_space import AxSearchSpaceOps


@pytest.fixture(scope="module")
def agent():
    from rag_stack.optimizer import OPTIMIZER_REGISTRY

    a_level = "stage__a__level"
    b_level = "stage__b__level"
    selector = "stage__component"
    search_space = SearchSpace(parameters=[
        ChoiceParameter(
            name=a_level,
            parameter_type=ParameterType.INT,
            values=[0, 1, 2],
            is_ordered=True,
            sort_values=True,
        ),
        ChoiceParameter(
            name=b_level,
            parameter_type=ParameterType.INT,
            values=[0, 1, 2],
            is_ordered=True,
            sort_values=True,
        ),
        ChoiceParameter(
            name="shared",
            parameter_type=ParameterType.INT,
            values=[0, 1, 2],
            is_ordered=True,
            sort_values=True,
        ),
        ChoiceParameter(
            name=selector,
            parameter_type=ParameterType.STRING,
            values=["a", "b"],
            is_ordered=False,
            sort_values=False,
            dependents={"a": [a_level], "b": [b_level]},
        ),
    ])
    space = AxSearchSpaceOps(search_space)
    objectives = [
        {"objective_name": "performance", "cost": 1, "ref_point": 0},
        {"objective_name": "quality", "cost": 10, "ref_point": 0},
    ]
    opt = OPTIMIZER_REGISTRY["agent_qnehvi"](
        search_space, objectives=objectives, seed=46, n_both_init=4,
        ssops=space,
    )
    xs, _ = opt.suggest(4)
    opt.add_objectives(
        xs,
        {"performance": [10.0, 40.0, 5.0, 90.0],
         "quality": [{"combined": 0.2}, {"combined": 0.3},
                     {"combined": 0.5}, {"combined": 0.15}]},
        eval_ids=[0, 1, 2, 3],
    )
    return opt


def test_train_rows_are_canonical(agent):
    X = torch.stack(agent._x_rows, dim=0)
    assert torch.allclose(agent.ssops.clamp(X), X, atol=1e-9)


def test_hier_pool_sobol_rows_canonical_unique_deterministic(agent):
    a = agent._hier_pool_sobol(24)
    assert a is not None and a.shape[0] > 0
    # canonical: clamp is identity on every generated row
    assert torch.allclose(agent.ssops.clamp(a), a, atol=1e-9)
    # unique canonical rows (no dead-knob phantom duplicates)
    assert len({tuple(r.tolist()) for r in a}) == a.shape[0]
    # deterministic at fixed (seed, t); distinct stream for the DLS offset
    b = agent._hier_pool_sobol(24)
    assert torch.equal(a, b)
    c = agent._hier_pool_sobol(24, seed_offset=300_000)
    assert not torch.equal(a, c)


def test_dls_restart_group_is_canonical(agent):
    groups = agent._restart_seed_groups(None)
    rnd = groups["random"]
    assert rnd.shape[0] >= 2
    assert torch.allclose(agent.ssops.clamp(rnd), rnd, atol=1e-9)


# ── DOE: champion-tension shared pool + within-branch factors ────────────

def _choice(name, values, *, parameter_type=ParameterType.INT, dependents=None):
    return ChoiceParameter(
        name=name,
        parameter_type=parameter_type,
        values=values,
        is_ordered=False,
        sort_values=parameter_type.is_numeric,
        dependents=dependents,
    )


def _row(ssops, config):
    return ssops.clamp(
        ssops.encode_arm_params(config).reshape(1, -1),
    ).reshape(-1)


def test_doe_champion_tension_plan_builds_one_shared_candidate_pool():
    ssops = AxSearchSpaceOps(SearchSpace(parameters=[
        _choice("x", [0, 1, 2]),
        _choice("y", [0, 1, 2]),
    ]))
    doe = DOEController(
        ssops,
        0,
        1,
    )
    doe.observe(_row(ssops, {"x": 0, "y": 0}), (100.0, 0.1), 0)
    doe.observe(_row(ssops, {"x": 2, "y": 2}), (10.0, 0.9), 1)
    # Cross-factor history establishes measured trade-off evidence without
    # hard-coding a preferred knob or level into the DOE policy.
    doe.observe(_row(ssops, {"x": 1, "y": 2}), (70.0, 0.5), 2)
    doe.observe(_row(ssops, {"x": 2, "y": 1}), (55.0, 0.65), 3)

    plan = doe.plan_round(
        t=5,
        accept_candidate=lambda *_args: True,
    )

    assert plan.pool
    assert sum(candidate.provenance == "doe-pair" for candidate in plan.pool) == 1
    champions = [
        doe._observations[index].x for index in doe._champion_indices()
    ]
    assert all(
        candidate.provenance == "doe-pair"
        or min(
            len(doe._active_diff_dims(champion, candidate.x))
            for champion in champions
        ) == 1
        for candidate in plan.pool
    )
    assert plan.audit.reason == "champion_tension"
    assert plan.audit.phases == (
        "p:oat_discovery(k=2)",
        "q:pair_confirmation(k=2)",
    )


def test_doe_history_backed_tension_is_scoped_to_champion_branch():
    ssops = AxSearchSpaceOps(SearchSpace(parameters=[
        _choice("shared", [0, 1, 2]),
        _choice("same", [0, 1, 2]),
        _choice("a_only", [0, 1]),
        _choice("b_only", [0, 1]),
        _choice(
            "selector",
            ["a", "b"],
            parameter_type=ParameterType.STRING,
            dependents={"a": ["a_only"], "b": ["b_only"]},
        ),
    ]))
    doe = DOEController(ssops, 0, 1)
    doe.observe(
        _row(
            ssops,
            {"shared": 0, "same": 0, "selector": "a", "a_only": 0},
        ),
        (100.0, 0.1),
        0,
    )
    doe.observe(
        _row(
            ssops,
            {"shared": 2, "same": 0, "selector": "b", "b_only": 1},
        ),
        (10.0, 0.9),
        1,
    )
    doe.observe(
        _row(
            ssops,
            {"shared": 1, "same": 0, "selector": "a", "a_only": 0},
        ),
        (80.0, 0.4),
        2,
    )

    plan = doe.plan_round(
        t=3,
        accept_candidate=lambda *_args: True,
    )

    shared_dim = ssops.param_names.index("shared")
    assert plan.pool
    anchor = doe._observations[0].x
    assert all(
        doe._active_diff_dims(anchor, candidate.x) == (shared_dim,)
        for candidate in plan.pool
    )
    for candidate in plan.pool:
        config = ssops.decode_flat(candidate.x)
        assert config["selector"] == "a"
        assert "a_only" in ssops.active_params(config)
        assert "b_only" not in ssops.active_params(config)


# ── known-invalid hard filter in arbitration (07-08) ──────────────────────

def test_registered_infeasible_config_is_filtered_from_pool(agent):
    """A config registered infeasible must be dropped from the candidate set
    BEFORE ranking/acqf — arbitration never picks what the controller's
    known-invalid seatbelt would refuse (eval_type=DUPLICATE)."""
    import torch

    cand = agent._hier_pool_sobol(12, seed_offset=800_000)
    assert cand is not None and cand.shape[0] >= 4
    prov = ["sobol"] * cand.shape[0]
    victim_cfg = agent.ssops.decode_to_nested(
        agent.ssops.decode_flat(cand[2].reshape(-1)))  # nested — the controller's shape

    # before registration: nothing dropped
    kept, kprov = agent._filter_known_bad(cand, list(prov))
    assert kept.shape[0] == cand.shape[0]

    agent.register_infeasible(victim_cfg, "CONTEXT_OVERFLOW: test")
    kept, kprov = agent._filter_known_bad(cand, list(prov))
    assert kept.shape[0] == cand.shape[0] - 1
    assert len(kprov) == kept.shape[0]
    X_bad = torch.stack(agent._x_bad, dim=0)
    for row in kept:
        assert not agent.ssops.is_duplicate_decoded(row.reshape(1, -1), X_bad)
