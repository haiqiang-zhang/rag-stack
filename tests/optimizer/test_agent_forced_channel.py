"""CPU-only contracts for the unified realized-HV stagnation forced channel.

These tests intentionally build tiny in-memory harnesses.  They do not read a
repository config, benchmark artifact, model checkpoint, or CUDA device.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from rag_stack.controller import Controller
from rag_stack.optimizer.agent.core import AgentGuidedQNEHVI, _LargestHVGap
from rag_stack.optimizer.agent.doe import DOEAudit, DOECandidate, DOEPlan
from rag_stack.optimizer.agent.forced import (
    ForcedChannelMixin,
    _FRONTIER_NEGLECT,
)
from rag_stack.optimizer.agent.llm_agent import (
    AgentAdvice,
    AgentAdviceError,
    GapExplorationRequest,
    ParetoGapAnchor,
)
from rag_stack.types import EvalType


class _ForcedClock(ForcedChannelMixin):
    """Minimal state owner for exercising the forced-channel mixin directly."""

    def __init__(self, state_dir=None):
        self._advice_log_dir = state_dir
        self._forced_state_loaded = False
        self._normal_no_hv_gain = 0
        self._stagnation_normalization = None
        self._last_accounted_eval_id = -1
        self._forced_fired = False
        self._t = 0
        # A non-degenerate realized objective frame with two frontier ends.
        self._perf = [1.0, 2.0]
        self._corr = [2.0, 1.0]

    def land(self, performance, quality, *, eval_id, forced=False):
        gain = self._record_realized_outcome(
            performance,
            quality,
            eval_id=eval_id,
            forced=forced,
        )
        # Production calls the recorder immediately before appending the
        # completed result to its realized history.
        self._perf.append(float(performance))
        self._corr.append(float(quality))
        return gain


def _land_zero_gain_normals(clock, count, *, first_eval_id=2):
    for offset in range(count):
        # Every point lies on/below the frozen quality minimum, so its realized
        # normalized marginal HV is exactly zero.
        gain = clock.land(
            1.1 + 0.01 * offset,
            1.0,
            eval_id=first_eval_id + offset,
        )
        assert gain == pytest.approx(0.0)


def test_forced_is_due_after_three_normal_zero_hv_results_not_two():
    assert _FRONTIER_NEGLECT == 3
    clock = _ForcedClock()

    _land_zero_gain_normals(clock, 2)
    assert clock._normal_no_hv_gain == 2
    assert not clock._forced_due(_FRONTIER_NEGLECT)

    _land_zero_gain_normals(clock, 1, first_eval_id=4)
    assert clock._normal_no_hv_gain == 3
    assert clock._forced_due(_FRONTIER_NEGLECT)


def test_positive_realized_normalized_hv_resets_stagnation_episode():
    clock = _ForcedClock()
    _land_zero_gain_normals(clock, 2)

    gain = clock.land(1.5, 1.5, eval_id=4)

    assert gain is not None and gain > 0.0
    assert clock._normal_no_hv_gain == 0
    assert clock._stagnation_normalization is None
    assert not clock._forced_due(_FRONTIER_NEGLECT)


def test_forced_result_resets_and_requires_three_new_normal_opportunities():
    clock = _ForcedClock()
    _land_zero_gain_normals(clock, 3)
    assert clock._forced_due(_FRONTIER_NEGLECT)

    gain = clock.land(1.0, 1.0, eval_id=5, forced=True)

    assert gain == pytest.approx(0.0)
    assert clock._normal_no_hv_gain == 0
    assert clock._stagnation_normalization is None
    _land_zero_gain_normals(clock, 2, first_eval_id=6)
    assert not clock._forced_due(_FRONTIER_NEGLECT)
    _land_zero_gain_normals(clock, 1, first_eval_id=8)
    assert clock._forced_due(_FRONTIER_NEGLECT)


def test_unscored_forced_attempt_preserves_due_episode():
    clock = _ForcedClock()
    _land_zero_gain_normals(clock, 3)
    frozen = dict(clock._stagnation_normalization)

    assert clock._record_unscored_attempt(eval_id=5, forced=True)

    assert clock._last_accounted_eval_id == 5
    assert clock._normal_no_hv_gain == _FRONTIER_NEGLECT
    assert clock._stagnation_normalization == frozen
    assert clock._forced_due(_FRONTIER_NEGLECT)


def test_forced_state_round_trips_across_resume(tmp_path):
    source = _ForcedClock(tmp_path)
    source._t = 9
    source._normal_no_hv_gain = 3
    source._stagnation_normalization = {
        "p_min": 1.0,
        "p_span": 3.0,
        "q_min": 0.2,
        "q_span": 0.7,
    }
    source._last_accounted_eval_id = 8
    source._save_forced_state()

    restored = _ForcedClock(tmp_path)
    restored._load_forced_state()

    assert restored._normal_no_hv_gain == 3
    assert restored._stagnation_normalization == source._stagnation_normalization
    assert restored._last_accounted_eval_id == 8
    assert restored._forced_due(_FRONTIER_NEGLECT)


def test_eval_record_persists_pending_forced_provenance(tmp_path):
    class _SearchSpace:
        @staticmethod
        def enrich_configs(configs, _base):
            return configs

    controller = Controller.__new__(Controller)
    controller.project_dir = str(tmp_path)
    controller._failure_counts = {}
    controller._ssops = _SearchSpace()
    controller._base_decoded_config = {}
    controller.optimizer = SimpleNamespace(
        pending_observation_metadata=lambda: {
            "forced": True,
            "agent_proposal_round": 4,
        },
    )

    assert controller._save_eval_record(
        eval_id=4,
        config={},
        pipeline_config={},
        eval_type=EvalType.GT,
        objectives={"performance": 2.0, "quality": 0.5},
        quality_metrics=None,
        assembly_result=None,
        obj_columns=["performance", "quality"],
    )
    marker = (
        tmp_path
        / "evaluations"
        / "eval_0004"
        / "optimizer_observation.json"
    )
    assert json.loads(marker.read_text()) == {
        "forced": True,
        "agent_proposal_round": 4,
    }


def test_invalid_rebuilds_filter_without_double_counting_forced_clock():
    optimizer = AgentGuidedQNEHVI.__new__(AgentGuidedQNEHVI)
    optimizer._infeasible = []
    optimizer._x_bad = []
    failed_rows = []
    optimizer.doe = SimpleNamespace(
        mark_failed=lambda row: failed_rows.append(row.detach().clone()),
        save_file=lambda _path: None,
    )
    optimizer._doe_state_path = lambda: None
    optimizer.ssops = SimpleNamespace(
        clamp=lambda row: row,
        snap_to_grid=lambda row: row,
    )
    optimizer._encode_nested_config = lambda _cfg: torch.tensor(
        [0.0], dtype=torch.double,
    )
    optimizer.tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
    optimizer._restoring_history = False
    optimizer._configs = [{}]
    optimizer.n_both_init = 1
    optimizer._normal_no_hv_gain = 3
    optimizer._forced_fired = False
    optimizer._stagnation_normalization = {
        "p_min": 0.0,
        "p_span": 1.0,
        "q_min": 0.0,
        "q_span": 1.0,
    }
    optimizer._save_forced_state = lambda: None

    optimizer.register_infeasible({}, "historical", replay=True)
    assert optimizer._normal_no_hv_gain == 3
    assert len(failed_rows) == 1

    optimizer.register_infeasible({}, "live")
    assert optimizer._normal_no_hv_gain == 3
    assert len(failed_rows) == 2


def test_invalid_penalty_defers_clock_accounting_to_objective_commit(tmp_path):
    calls = []

    class _Optimizer:
        @staticmethod
        def pending_observation_metadata():
            return {"forced": False, "agent_proposal_round": 4}

        @staticmethod
        def register_infeasible(config, reason, **kwargs):
            calls.append((config, reason, kwargs))

    controller = Controller.__new__(Controller)
    controller.project_dir = str(tmp_path)
    controller.optimizer = _Optimizer()
    controller._opt_ref = controller.optimizer
    controller._invalid_configs = []
    controller._invalid_config_keys = set()

    controller._record_invalid(
        7,
        {"candidate": "penalty"},
        "deployment_preset: invalid",
        objectives_will_be_committed=True,
    )
    controller._record_invalid(
        8,
        {"candidate": "unscored"},
        "resolve_config: invalid",
    )

    assert calls[0][2] == {
        "eval_id": None,
        "optimizer_metadata": {
            "forced": False,
            "agent_proposal_round": 4,
        },
    }
    assert calls[1][2] == {
        "eval_id": 8,
        "optimizer_metadata": {
            "forced": False,
            "agent_proposal_round": 4,
        },
    }


class _TinySearchSpaceOps:
    """Two-column decoded identity: [active branch, categorical value]."""

    @staticmethod
    def clamp(rows):
        return rows.detach().clone().to(dtype=torch.double, device="cpu")

    @staticmethod
    def decode_flat(row):
        values = row.reshape(-1)
        return {"branch": int(values[0].item()), "value": int(values[1].item())}

    decode_to_config = decode_flat

    @staticmethod
    def active_params(_flat):
        return {"branch", "value"}


class _PlanDOE:
    def __init__(self):
        self.selected = []

    @staticmethod
    def _row_signature(row):
        return ":".join(str(int(value)) for value in row.reshape(-1).tolist())

    def mark_selected(self, candidate):
        self.selected.append(candidate)

    @staticmethod
    def load_file(_path):
        return None


class _ForcedPosteriorModel:
    """Deterministic two-output posterior for objective-gap ranking.

    The within-branch ``[0, 2]`` row lands at the center of the normalized
    objective rectangle.  The structurally farther ``[1, 3]`` row predicts
    high performance but quality below the rectangle, so its expected fill is
    zero.  Source labels deliberately play no role in these predictions.
    """

    @staticmethod
    def posterior(rows):
        flat = rows.reshape(-1, rows.shape[-1])
        branch, value = flat[:, 0], flat[:, 1]
        within_branch = branch == 0
        performance = torch.where(
            within_branch,
            2.5 * value,
            torch.full_like(value, 9.0),
        )
        quality = torch.where(
            within_branch,
            10.0 - 2.5 * value,
            torch.full_like(value, -1.0),
        )
        mean = torch.stack((performance, quality), dim=-1).unsqueeze(-2)
        return SimpleNamespace(
            mean=mean,
            variance=torch.zeros_like(mean),
        )


def _doe_candidate(
    value,
    *,
    branch=0,
    kind="oat",
    side="p",
):
    return DOECandidate(
        x=torch.tensor([float(branch), float(value)], dtype=torch.double),
        kind=kind,
        era_id=f"{side}-era",
    )


def _doe_plan(*pool):
    return DOEPlan(
        pool=tuple(pool),
        audit=DOEAudit(
            round=9,
            history_size=8,
            tension_score=0.75,
            phases=("p:oat_discovery(k=1)",),
            oat_cells=sum(candidate.kind == "oat" for candidate in pool),
            pair_cells=sum(candidate.kind == "pair" for candidate in pool),
            reason="champion_tension",
        ),
    )


def _selection_harness():
    optimizer = AgentGuidedQNEHVI.__new__(AgentGuidedQNEHVI)
    optimizer.ssops = _TinySearchSpaceOps()
    optimizer.tkwargs = {"dtype": torch.double, "device": torch.device("cpu")}
    optimizer.doe = _PlanDOE()
    optimizer._stagnation_normalization = {
        "p_min": 0.0,
        "p_span": 10.0,
        "q_min": 0.0,
        "q_span": 10.0,
    }
    optimizer._normal_no_hv_gain = _FRONTIER_NEGLECT
    optimizer._x_rows = [
        torch.tensor([0.0, 0.0], dtype=torch.double),
        torch.tensor([0.0, 5.0], dtype=torch.double),
    ]
    optimizer._perf = [0.0, 10.0]
    optimizer._corr = [10.0, 0.0]
    optimizer._eval_ids = [1, 2]
    optimizer._sub_metrics = [{}, {}]
    optimizer._perf_attr = {}
    optimizer._infeasible = []
    optimizer._filter_known_bad = lambda rows, prov: (rows, prov)
    optimizer._drop_evaluated_pool_rows = lambda rows, prov: (rows, prov)
    optimizer._prefilter_pool = lambda rows: rows
    optimizer._accept_candidate = lambda _row, _config: True
    optimizer.model = _ForcedPosteriorModel()
    optimizer._perf_idx = 0
    optimizer._qual_idx = 1
    optimizer._independent_evidence_payload = lambda: {
        "gp_knob_tension": {"status": "ok", "rows": []},
        "doe_landed_ablation": {"status": "ok", "rows": []},
    }
    optimizer._forced_fired = False
    optimizer._t = 9
    return optimizer


def _gap_fixture(optimizer):
    return _LargestHVGap(
        request=GapExplorationRequest(
            gap_id="q:p",
            quality_anchor=ParetoGapAnchor(
                {"branch": 0, "value": 0},
                normalized_performance=0.0,
                normalized_quality=1.0,
            ),
            performance_anchor=ParetoGapAnchor(
                {"branch": 0, "value": 5},
                normalized_performance=1.0,
                normalized_quality=0.0,
            ),
        ),
        quality_index=0,
        performance_index=1,
        quality_x=optimizer._x_rows[0],
        performance_x=optimizer._x_rows[1],
        area=1.0,
        normalization=(0.0, 10.0, 0.0, 10.0),
    )


@pytest.mark.parametrize("center_source", ("agent", "doe"))
def test_forced_gap_ranking_is_source_neutral(center_source, caplog):
    optimizer = _selection_harness()
    if center_source == "doe":
        doe_cell = _doe_candidate(2)
        agent_row = [1.0, 3.0]
    else:
        doe_cell = _doe_candidate(3, branch=1)
        agent_row = [0.0, 2.0]
    plan = _doe_plan(doe_cell)
    advice = AgentAdvice(
        centers=torch.tensor([agent_row], dtype=torch.double),
    )

    with caplog.at_level("INFO"):
        selected = optimizer._try_forced_exploration(
            plan, advice, _gap_fixture(optimizer),
        )

    assert selected == {"branch": 0, "value": 2}
    assert optimizer.doe.selected == (
        [doe_cell] if center_source == "doe" else []
    )
    assert optimizer._forced_fired
    assert any(
        f"source_family={center_source} all_sources={center_source}"
        in record.message
        for record in caplog.records
    )


def test_dedup_records_joint_source_and_marks_only_real_doe_ticket(caplog):
    optimizer = _selection_harness()
    shared = _doe_candidate(
        2,
        kind="pair",
        side="q",
    )
    plan = _doe_plan(shared)
    advice = AgentAdvice(
        centers=torch.tensor(
            [[0.0, 2.0], [0.0, 2.0]],
            dtype=torch.double,
        ),
    )

    with caplog.at_level("INFO"):
        selected = optimizer._try_forced_exploration(
            plan, advice, _gap_fixture(optimizer),
        )

    assert selected == {"branch": 0, "value": 2}
    assert optimizer.doe.selected == [shared]
    assert any(
        "source_family=joint all_sources=agent,doe "
        "source_labels=agent,doe-pair" in record.message
        for record in caplog.records
    )


def test_duplicate_candidate_count_cannot_change_forced_choice():
    def _pick(repeats):
        optimizer = _selection_harness()
        doe_cell = _doe_candidate(2)
        plan = _doe_plan(doe_cell)
        advice = AgentAdvice(
            centers=torch.tensor(
                [[1.0, 3.0]] * repeats,
                dtype=torch.double,
            ),
        )
        selected = optimizer._try_forced_exploration(
            plan, advice, _gap_fixture(optimizer),
        )
        return selected

    assert _pick(1) == _pick(7)


@pytest.mark.parametrize(
    "rejecting_gate",
    ("known_bad", "evaluated", "forced_prefilter", "accept"),
)
def test_forced_common_gates_reject_without_spending_forced_seat(
    rejecting_gate,
):
    optimizer = _selection_harness()
    doe_cell = _doe_candidate(3)
    plan = _doe_plan(doe_cell)

    def reject_rows(rows, _provenance):
        return rows[:0], []

    if rejecting_gate == "known_bad":
        optimizer._filter_known_bad = reject_rows
    elif rejecting_gate == "evaluated":
        optimizer._drop_evaluated_pool_rows = reject_rows
    elif rejecting_gate == "forced_prefilter":
        optimizer._prefilter_forced_pool_with_prov = reject_rows
    else:
        optimizer._accept_candidate = lambda _row, _config: False

    advice = AgentAdvice(
        centers=torch.tensor([[1.0, 3.0]], dtype=torch.double),
    )
    assert optimizer._try_forced_exploration(
        plan, advice, _gap_fixture(optimizer),
    ) is None
    assert optimizer.doe.selected == []
    assert not optimizer._forced_fired
    assert optimizer._normal_no_hv_gain == _FRONTIER_NEGLECT
    assert optimizer._forced_due(_FRONTIER_NEGLECT)


def test_forced_gap_scoring_bypasses_ordinary_hv_shortlist():
    optimizer = _selection_harness()
    calls = []

    def reject_as_ordinary_shortlist(rows, provenance):
        calls.append((rows, provenance))
        return rows[:0], []

    # An HV-anchor subclass may rank/truncate the ordinary qLogNEHVI pool by
    # overriding this wrapper. Forced selection must instead retain the whole
    # legal Agent+DOE set and rank it against the chosen objective-space gap.
    optimizer._prefilter_pool_with_prov = reject_as_ordinary_shortlist
    selected = optimizer._try_forced_exploration(
        _doe_plan(_doe_candidate(2)),
        AgentAdvice(centers=torch.tensor([[1.0, 3.0]], dtype=torch.double)),
        _gap_fixture(optimizer),
    )

    assert selected == {"branch": 0, "value": 2}
    assert calls == []


def test_expected_bounded_extent_handles_limits_and_uncertainty():
    deterministic = AgentGuidedQNEHVI._expected_bounded_extent(
        torch.tensor([-1.0, 0.25, 2.0], dtype=torch.double),
        torch.zeros(3, dtype=torch.double),
        0.0,
        1.0,
    )
    assert deterministic.tolist() == pytest.approx([0.0, 0.25, 1.0])

    # A mean below the rectangle still has positive exploration value when
    # posterior uncertainty places probability mass inside it.
    uncertain = AgentGuidedQNEHVI._expected_bounded_extent(
        torch.tensor([-0.25], dtype=torch.double),
        torch.tensor([1.0], dtype=torch.double),
        0.0,
        1.0,
    )
    assert 0.0 < uncertain.item() < 1.0

    with pytest.raises(ValueError, match="strictly ordered"):
        AgentGuidedQNEHVI._expected_bounded_extent(
            torch.tensor([0.5], dtype=torch.double),
            torch.zeros(1, dtype=torch.double),
            1.0,
            1.0,
        )


def test_gap_geometry_and_agent_request_are_the_same_indexed_object():
    optimizer = _selection_harness()
    optimizer._x_rows = [
        torch.tensor([0.0, float(index)], dtype=torch.double)
        for index in range(4)
    ]
    optimizer._perf = [0.0, 4.0, 10.0, 2.0]
    optimizer._corr = [10.0, 6.0, 0.0, 2.0]
    optimizer._eval_ids = [10, 11, 12, 13]
    optimizer._sub_metrics = [
        {"tag": "quality"},
        {"tag": "middle"},
        {"tag": "performance"},
        {"tag": "dominated"},
    ]

    gap = optimizer._largest_hv_gap()

    assert gap is not None
    assert (gap.quality_index, gap.performance_index) == (1, 2)
    assert gap.area == pytest.approx(0.36)
    assert gap.request.quality_anchor.flat == {"branch": 0, "value": 1}
    assert gap.request.performance_anchor.flat == {"branch": 0, "value": 2}
    assert gap.request.quality_anchor.sub_metrics == {"tag": "middle"}


def test_largest_gap_uses_episode_frozen_normalization():
    optimizer = _selection_harness()
    optimizer._x_rows = [
        torch.tensor([0.0, 2.0], dtype=torch.double),
        torch.tensor([0.0, 8.0], dtype=torch.double),
        torch.tensor([1.0, 9.0], dtype=torch.double),
    ]
    optimizer._perf = [2.0, 8.0, -100.0]
    optimizer._corr = [8.0, 2.0, -100.0]
    optimizer._eval_ids = [10, 11, 12]
    optimizer._sub_metrics = [{}, {}, {}]

    gap = optimizer._largest_hv_gap(
        normalization=optimizer._stagnation_normalization,
    )

    assert gap is not None
    assert gap.normalization == (0.0, 10.0, 0.0, 10.0)
    assert gap.area == pytest.approx(0.36)
    assert (
        gap.request.quality_anchor.normalized_performance,
        gap.request.quality_anchor.normalized_quality,
    ) == pytest.approx((0.2, 0.8))
    assert (
        gap.request.performance_anchor.normalized_performance,
        gap.request.performance_anchor.normalized_quality,
    ) == pytest.approx((0.8, 0.2))


def test_strict_gap_agent_empty_result_aborts_before_forced_gt():
    optimizer = _selection_harness()
    optimizer.agent_strict = True
    optimizer._history = lambda: []
    optimizer.agent = SimpleNamespace(
        explore_gap=lambda *_args, **_kwargs: AgentAdvice(
            raw={"status": "unavailable", "error": "malformed reply"},
        ),
    )

    with pytest.raises(AgentAdviceError, match="malformed reply"):
        optimizer._maybe_advise_forced_gap(_gap_fixture(optimizer))


def test_strict_gap_agent_failure_aborts_suggest_without_spending_seat():
    optimizer = _selection_harness()
    optimizer.agent_strict = True
    optimizer.agent = SimpleNamespace(
        explore_gap=lambda *_args, **_kwargs: AgentAdvice(
            raw={"status": "unavailable", "error": "malformed reply"},
        ),
    )
    optimizer._history = lambda: []
    optimizer._configs = [{}, {}]
    optimizer.n_both_init = 2
    optimizer.objective_names = ["performance", "quality"]
    optimizer._doe_state_path = lambda: None
    optimizer._load_forced_state = lambda: None
    optimizer._fit_models = lambda: None
    gap = _gap_fixture(optimizer)
    optimizer._largest_hv_gap = lambda **_kwargs: gap
    pool_calls = []
    optimizer._propose_pool = lambda *_args, **_kwargs: pool_calls.append(
        (_args, _kwargs)
    )
    before_normalization = dict(optimizer._stagnation_normalization)

    with pytest.raises(AgentAdviceError, match="malformed reply"):
        optimizer.suggest()

    assert pool_calls == []
    assert optimizer._t == 9
    assert not optimizer._forced_fired
    assert optimizer._normal_no_hv_gain == _FRONTIER_NEGLECT
    assert optimizer._stagnation_normalization == before_normalization
    assert optimizer.pending_observation_metadata() == {
        "forced": False,
        "agent_proposal_round": 9,
    }


def test_forced_gap_agent_receives_independent_evidence():
    optimizer = _selection_harness()
    optimizer.agent_strict = True
    optimizer._history = lambda: [{"eval_id": 1}]
    captured = {}

    def _explore(request, history, ssops, t, **kwargs):
        captured.update({
            "request": request,
            "history": history,
            "ssops": ssops,
            "t": t,
            **kwargs,
        })
        return AgentAdvice(
            centers=torch.tensor([[1.0, 3.0]], dtype=torch.double),
        )

    optimizer.agent = SimpleNamespace(explore_gap=_explore)
    gap = _gap_fixture(optimizer)

    advice = optimizer._maybe_advise_forced_gap(gap)

    assert advice is not None and advice.n() == 1
    assert captured["request"] is gap.request
    assert captured["history"] == [{"eval_id": 1}]
    assert captured["ssops"] is optimizer.ssops
    assert captured["t"] == optimizer._t
    assert captured["independent_evidence"] == {
        "gp_knob_tension": {"status": "ok", "rows": []},
        "doe_landed_ablation": {"status": "ok", "rows": []},
    }

def test_second_forced_delivery_in_one_round_fails_closed():
    clock = _ForcedClock()
    clock._mark_forced()

    with pytest.raises(RuntimeError, match="round's only forced seat"):
        clock._mark_forced()

    assert clock._forced_fired
