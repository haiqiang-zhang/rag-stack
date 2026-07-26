"""CPU-only routing and initialization contracts for Agent mechanism ablations.

No evaluator, model service, API request, benchmark output, or CUDA device is
used. Acquisition values in the routing tests are deterministic local stubs;
initialization tests run both existing phases using a local Agent transport.
"""

from __future__ import annotations

import json
import logging
import socket

import pytest
import torch
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.optimizer.agent.core import AgentGuidedQNEHVI
from rag_stack.optimizer.agent.doe import DOEAudit, DOECandidate, DOEPlan
from rag_stack.optimizer.agent.llm_agent import AgentAdvice, AgentAdviceError
from rag_stack.search_space import AxSearchSpaceOps


CPU = {"dtype": torch.double, "device": torch.device("cpu")}
OBJECTIVES = [
    {"objective_name": "performance", "cost": 1, "ref_point": 0},
    {"objective_name": "quality", "cost": 10, "ref_point": 0},
]
VARIANTS = {
    "no_stage_signals": "enable_stage_signals",
    "no_agent": "enable_agent",
    "no_dls": "enable_dls",
    "no_ptc": "enable_doe",
    "no_forced": "enable_forced",
}
FLAGS = tuple(VARIANTS.values())


def _unexpected(*_args, **_kwargs):
    raise AssertionError("This path must not execute in the ablation test")


@pytest.fixture(autouse=True)
def _no_cuda_or_network(monkeypatch):
    """Fail rather than probing hardware or contacting an Agent provider."""
    original_fork_rng = torch.random.fork_rng

    def cpu_fork_rng(devices=None, *args, **kwargs):
        # Ax's CPU Sobol generator otherwise asks fork_rng to enumerate every
        # GPU merely to save RNG state. Explicit devices=[] avoids that probe
        # without changing environment variables or accelerator visibility.
        if devices:
            raise AssertionError("CUDA RNG devices are forbidden in this test")
        return original_fork_rng(devices=[], *args, **kwargs)

    monkeypatch.setattr(torch.random, "fork_rng", cpu_fork_rng)
    monkeypatch.setattr(torch.cuda, "is_available", _unexpected)
    monkeypatch.setattr(torch.cuda, "device_count", _unexpected)
    monkeypatch.setattr(torch.cuda, "_lazy_init", _unexpected)
    monkeypatch.setattr(socket.socket, "connect", _unexpected)
    monkeypatch.setattr(socket, "create_connection", _unexpected)


class _LocalInitClient:
    def __init__(self, candidates):
        self.candidates = candidates
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return json.dumps({"candidates": self.candidates})


def _make_opt(**overrides):
    space = SearchSpace(parameters=[
        ChoiceParameter(
            name=name, parameter_type=ParameterType.INT,
            values=[0, 1, 2, 3], is_ordered=True, sort_values=True,
        )
        for name in "abcd"
    ])
    params = {
        "seed": 43, "n_both_init": 4, "init_sobol_k": 2,
        "n_candidates": 8, "dls_num_restarts": 2, "agent_strict": True,
    }
    params.update(overrides)
    return AgentGuidedQNEHVI(
        space, OBJECTIVES, ssops=AxSearchSpaceOps(space, CPU), **params,
    )


def _feed(opt, configs):
    start = len(opt._configs)
    opt.add_objectives(
        configs,
        {
            "performance": [10.0 + start + i for i in range(len(configs))],
            "quality": [
                {"combined": 0.8 - 0.01 * (start + i), "stage_metric": 0.31}
                for i in range(len(configs))
            ],
        },
        eval_ids=list(range(start, start + len(configs))),
    )


def _ready(**overrides):
    opt = _make_opt(**overrides)
    _feed(opt, [{"a": i, "b": 0, "c": 0, "d": 0} for i in range(4)])
    opt._fit_models = lambda: None
    return opt


def _row(opt, **values):
    return opt._encode_nested_config(dict.fromkeys("abcd", 0) | values)


def _plan(*cells):
    return DOEPlan(
        pool=tuple(cells),
        audit=DOEAudit(
            round=0, history_size=4, tension_score=0.5, phases=(),
            oat_cells=sum(cell.kind == "oat" for cell in cells),
            pair_cells=sum(cell.kind == "pair" for cell in cells),
            reason="local_test",
        ),
    )


def _local_pool(opt):
    """Use the production filtering/ranking flow with a deterministic scorer."""
    scored = []

    def score(rows):
        scored.append(rows.detach().clone())
        return rows.sum(dim=(-1, -2))

    opt._build_acqf = lambda: score
    opt._hier_pool_sobol = lambda _n, **_kwargs: torch.stack([
        _row(opt, b=1), _row(opt, b=2),
    ])
    opt._doe_round_plan = lambda: _plan()
    return scored


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("bad_value", ["false", "true", 0, 1, None])
def test_ablation_flags_reject_non_boolean_values(flag, bad_value):
    with pytest.raises((TypeError, ValueError), match=flag):
        _make_opt(**{flag: bad_value})


def test_constructor_defaults_preserve_full_and_explicit_cpu():
    opt = _make_opt()
    assert all(getattr(opt, flag) is True for flag in FLAGS)
    assert opt.doe.enabled is True
    assert opt.tkwargs["device"].type == "cpu"
    assert opt.agent.enable_stage_signals is True


@pytest.mark.parametrize("disabled_flag", [None, *FLAGS])
def test_each_ablation_preserves_sobol_then_local_agent_initialization(disabled_flag):
    overrides = {} if disabled_flag is None else {disabled_flag: False}
    opt = _make_opt(**overrides)
    _exercise_both_init_phases(opt)


def _exercise_both_init_phases(opt):
    assert opt.init_chunks == [opt.init_sobol_k, opt.n_both_init - opt.init_sobol_k]
    first, names = opt.suggest(opt.init_sobol_k)
    assert names == ["performance", "quality"]
    assert len(first) == opt.init_sobol_k
    _feed(opt, first)
    remaining = opt.n_both_init - len(first)
    donors = opt._sobol_batch(remaining)
    donor_flats = [opt.ssops.active_flat_config(config) for config in donors]
    client = _LocalInitClient(donor_flats)
    opt.agent._get_client = lambda: client
    second, names = opt.suggest(remaining)
    assert len(client.calls) == 1
    assert len(second) == remaining
    prompt = "\n".join(message["content"] for message in client.calls[0][0])
    assert "OBSERVED RESULTS" in prompt
    assert f"Design EXACTLY {remaining} probes" in prompt
    donor_ids = {json.dumps(config, sort_keys=True) for config in donors}
    assert {json.dumps(config, sort_keys=True) for config in second} == donor_ids
    _feed(opt, second)
    assert opt.init_chunks == []
    assert len(opt._configs) == opt.n_both_init
    assert len({json.dumps(config, sort_keys=True) for config in opt._configs}) == opt.n_both_init
    for config in opt._configs:
        active = opt.ssops.active_flat_config(config, require_all_active=True)
        assert opt.search_space.check_membership(active)
    assert all(row.device.type == "cpu" for row in opt._x_rows)
    assert opt._sobol_calls == 0, "Native Ax Sobol must succeed without raw fallback"


@pytest.mark.parametrize("forced_due", [False, True])
def test_no_agent_skips_both_bo_agent_calls_but_keeps_ptc_forced(forced_due):
    opt = _ready(enable_agent=False)
    gap = object() if forced_due else None
    opt._forced_due = lambda _patience: forced_due
    opt._largest_hv_gap = lambda **_kwargs: gap
    opt._maybe_advise = _unexpected
    opt._maybe_advise_forced_gap = _unexpected
    calls = []

    def propose(advice, **kwargs):
        calls.append((advice, kwargs))
        return {"a": 0, "b": 1, "c": 0, "d": 0}

    opt._propose_pool = propose
    result, _ = opt.suggest()
    assert result
    assert calls == [(None, {"forced_due": forced_due, "forced_gap": gap})]
    assert opt.doe.enabled is True


def test_no_forced_never_requests_gap_advice_and_keeps_ordinary_agent():
    opt = _ready(enable_forced=False)
    opt._normal_no_hv_gain = 99
    opt._largest_hv_gap = _unexpected
    opt._maybe_advise_forced_gap = _unexpected
    advice = AgentAdvice(centers=torch.stack([_row(opt, b=3)]))
    opt._maybe_advise = lambda: advice
    calls = []

    def propose(received, **kwargs):
        calls.append((received, kwargs))
        return {"a": 0, "b": 1, "c": 0, "d": 0}

    opt._propose_pool = propose
    opt.suggest()
    assert calls == [(advice, {"forced_due": False, "forced_gap": None})]


def test_full_preserves_specialized_forced_agent_routing():
    opt = _ready()
    gap = object()
    advice = object()
    opt._forced_due = lambda _patience: True
    opt._largest_hv_gap = lambda **_kwargs: gap
    opt._maybe_advise = _unexpected
    opt._maybe_advise_forced_gap = lambda received: advice if received is gap else _unexpected()
    calls = []

    def propose(received, **kwargs):
        calls.append((received, kwargs))
        return {"a": 0, "b": 1, "c": 0, "d": 0}

    opt._propose_pool = propose
    opt.suggest()
    assert calls == [(advice, {"forced_due": True, "forced_gap": gap})]


def test_no_dls_skips_all_local_search_and_breadth_seed_scoring():
    opt = _ready(enable_dls=False, enable_doe=False)
    scored = _local_pool(opt)
    opt._dls_candidates_grouped = _unexpected
    chosen = opt._propose_pool(None, forced_due=False)
    assert chosen["b"] == 2
    assert len(scored) == 1, "Only final pool arbitration may score candidates"


def test_no_dls_direct_entrypoint_exits_without_preparing_search():
    opt = _ready(enable_dls=False)
    opt._grid_choices = _unexpected
    opt._restart_seed_groups = _unexpected
    assert opt._dls_candidates_grouped(_unexpected, None) is None


def test_no_agent_strips_direct_centers_and_dls_advice(caplog):
    opt = _ready(enable_agent=False, enable_doe=False)
    _local_pool(opt)
    received = []

    def dls(_acqf, advice, **kwargs):
        received.append((advice, kwargs["pool_seeds"]))
        return None

    opt._dls_candidates_grouped = dls
    advice = AgentAdvice(centers=torch.stack([_row(opt, a=3, b=3, c=3, d=3)]))
    with caplog.at_level(logging.INFO, logger="RAG-Stack"):
        chosen = opt._propose_pool(advice, forced_due=False)
    assert chosen["a"] == 0 and chosen["b"] == 2
    assert received[0][0] is None
    assert received[0][1] is not None
    assert not any("source=llm" in record.message for record in caplog.records)
    groups = opt._restart_seed_groups(advice)
    assert "llm" not in groups
    assert {"incumbent", "random"} <= groups.keys()


def test_full_pool_keeps_agent_ptc_and_dls_candidates(caplog):
    opt = _ready()
    scored = _local_pool(opt)
    advice = AgentAdvice(centers=torch.stack([_row(opt, a=3, b=3, c=3, d=3)]))
    cell = DOECandidate(x=_row(opt, b=3), kind="oat", era_id="local")
    opt._doe_round_plan = lambda: _plan(cell)
    calls = []

    def dls(_acqf, received, **kwargs):
        calls.append((received, kwargs["pool_seeds"]))
        return torch.stack([_row(opt, a=2, b=3, c=3, d=3)]), ["dls-llm"]

    opt._dls_candidates_grouped = dls
    with caplog.at_level(logging.INFO, logger="RAG-Stack"):
        chosen = opt._propose_pool(advice, forced_due=False)
    assert chosen == dict.fromkeys("abcd", 3)
    assert calls[0][0] is advice and calls[0][1] is not None
    assert len(scored) == 2
    composition = next(record.message for record in caplog.records if record.message.startswith("[pool t="))
    assert all(source in composition for source in ("sobol=2", "llm=1", "doe-oat=1", "dls-llm=1"))


def test_no_agent_rejects_advice_at_direct_forced_entrypoint():
    opt = _ready(enable_agent=False)
    opt._forced_gap_coverage_scores = _unexpected
    advice = AgentAdvice(centers=torch.stack([_row(opt, b=3)]))
    assert opt._try_forced_exploration(_plan(), advice, object()) is None
    assert opt._forced_candidates_unavailable is True


def test_no_forced_blocks_direct_forced_pool_entry():
    opt = _ready(enable_forced=False, enable_dls=False, enable_doe=False)
    _local_pool(opt)
    opt._try_forced_exploration = _unexpected
    chosen = opt._propose_pool(None, forced_due=True, forced_gap=object())
    assert chosen["b"] == 2
    assert opt._forced_fired is False


def test_disabled_ptc_has_no_candidates_or_evidence():
    opt = _ready(enable_doe=False)
    plan = opt._doe_round_plan()
    assert plan.pool == ()
    assert plan.audit.oat_cells == plan.audit.pair_cells == 0
    assert opt.doe.enabled is False
    assert opt._doe_landed_evidence()["rows"] == []
    opt._forced_gap_coverage_scores = _unexpected
    stale_cell = DOECandidate(x=_row(opt, b=3), kind="oat", era_id="local")
    assert opt._try_forced_exploration(_plan(stale_cell), None, object()) is None


def test_no_agent_exhausted_ptc_forced_set_is_audited_normal_fallback(caplog):
    opt = _ready(enable_agent=False, enable_dls=False)
    _local_pool(opt)
    opt._normal_no_hv_gain = 3
    with caplog.at_level(logging.INFO, logger="RAG-Stack"):
        chosen = opt._propose_pool(None, forced_due=True, forced_gap=object())
    assert chosen["b"] == 2
    assert opt._normal_no_hv_gain == 3
    assert opt._forced_fired is False
    assert any("forced" in record.message and "ablation" in record.message for record in caplog.records)


def test_full_exhausted_forced_set_still_fails_in_strict_mode():
    opt = _ready(enable_dls=False)
    _local_pool(opt)
    with pytest.raises(AgentAdviceError, match="exhausted"):
        opt._propose_pool(None, forced_due=True, forced_gap=object())


def test_no_agent_does_not_hide_real_forced_posterior_failure(caplog):
    opt = _ready(enable_agent=False, enable_dls=False)
    _local_pool(opt)
    cell = DOECandidate(x=_row(opt, b=3), kind="oat", era_id="local")
    opt._doe_round_plan = lambda: _plan(cell)

    def broken_posterior(*_args):
        raise RuntimeError("broken posterior")

    opt._forced_gap_coverage_scores = broken_posterior
    with caplog.at_level(logging.WARNING, logger="RAG-Stack"):
        with pytest.raises(AgentAdviceError, match="exhausted"):
            opt._propose_pool(None, forced_due=True, forced_gap=object())
    assert any("broken posterior" in record.message for record in caplog.records)
    assert opt._forced_candidates_unavailable is False
