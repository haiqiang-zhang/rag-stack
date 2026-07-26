"""v3 selection layer: E[ΔHV] scoring and greedy HV-plateau handoff.

Pure-function tests run on ``__new__``-constructed instances with only the
attributes each method reads — no search space / GP required."""

import numpy as np
import pytest
import torch
from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.optimizer.botorch.greedy_then_hv_anchor_agent_qnehvi import (
    AnchorAgentQNEHVI,
    GreedyThenAnchorAgentQNEHVI,
    GreedyThenHVAnchorAgentQNEHVI,
)
from rag_stack.optimizer.botorch.greedy_then_hv_anchor_agent_qnehvi import (
    HVAnchorQNEHVI,
    HVAnchorLLMAgent,
)
from rag_stack.search_space import AxSearchSpaceOps


def _scorer(perf, corr, **attrs):
    """Bare HVAnchorQNEHVI with just the state the scoring math reads."""
    obj = HVAnchorQNEHVI.__new__(HVAnchorQNEHVI)
    obj._perf = list(perf)
    obj._corr = list(corr)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def _deterministic_hv_gain(perf, corr, perf_new, corr_new):
    """Independent staircase-HV oracle for the sigma→0 regression."""
    y = np.asarray(list(zip(perf, corr)), dtype=float)
    front = y[
        ~np.asarray([
            any(
                j != i
                and other[0] >= point[0]
                and other[1] >= point[1]
                and (other[0] > point[0] or other[1] > point[1])
                for j, other in enumerate(y)
            )
            for i, point in enumerate(y)
        ])
    ]
    p_scale = max(1e-9, float(y[:, 0].max()))
    q_scale = max(1e-9, float(y[:, 1].max()))

    def area(rows):
        result = previous_quality = 0.0
        for performance, quality in sorted(
            (
                (float(row[0]) / p_scale, float(row[1]) / q_scale)
                for row in rows
            ),
            reverse=True,
        ):
            if quality > previous_quality:
                result += performance * (quality - previous_quality)
                previous_quality = quality
        return result

    augmented = np.vstack([front, [perf_new, corr_new]])
    keep = ~np.asarray([
        any(
            j != i
            and other[0] >= point[0]
            and other[1] >= point[1]
            and (other[0] > point[0] or other[1] > point[1])
            for j, other in enumerate(augmented)
        )
        for i, point in enumerate(augmented)
    ])
    return area(augmented[keep]) - area(front)


# ── E[ΔHV] ────────────────────────────────────────────────────────────────

def test_expected_hv_gain_sigma_zero_matches_deterministic():
    rng = np.random.default_rng(7)
    for _ in range(20):
        n = int(rng.integers(3, 12))
        perf = rng.uniform(1, 400, n)
        corr = rng.uniform(0.05, 0.6, n)
        s = _scorer(perf, corr)
        p_new = float(rng.uniform(1, 450))
        q_new = float(rng.uniform(0.05, 0.7))
        det = _deterministic_hv_gain(perf, corr, p_new, q_new)
        exp = s._expected_hv_gain(p_new, q_new, 1e-9)
        assert exp is not None
        assert exp == pytest.approx(det, abs=1e-6)


def test_expected_hv_gain_dominated_point_needs_sigma():
    s = _scorer([300.0, 20.0], [0.15, 0.50])
    # (100, 0.10) is dominated by (300, 0.15): zero deterministic gain …
    assert s._expected_hv_gain(100.0, 0.10, 1e-9) == pytest.approx(0.0, abs=1e-6)
    # … but a wide quality posterior keeps beat-the-front probability mass.
    assert s._expected_hv_gain(100.0, 0.10, 0.2) > 0.0


def test_expected_hv_gain_scale_invariant_in_perf():
    s1 = _scorer([300.0, 20.0], [0.15, 0.50])
    s10 = _scorer([3000.0, 200.0], [0.15, 0.50])
    g1 = s1._expected_hv_gain(150.0, 0.30, 0.05)
    g10 = s10._expected_hv_gain(1500.0, 0.30, 0.05)
    assert g1 == pytest.approx(g10, rel=1e-9)


def test_pool_ranking_uses_raw_quality_posterior():
    """The deleted online-calibration path was never populated in production.

    Keep the effective identity behavior explicit: E[ΔHV] receives the GP
    posterior mean and standard deviation without a second transform.
    """

    class FakeModel:
        @staticmethod
        def posterior(_x):
            return type("Posterior", (), {
                "mean": torch.tensor(
                    [[[30.0, 0.2]], [[20.0, 0.3]], [[10.0, 0.4]]],
                    dtype=torch.double,
                ),
                "variance": torch.tensor(
                    [[[1.0, 0.01]], [[1.0, 0.04]], [[1.0, 0.09]]],
                    dtype=torch.double,
                ),
            })()

    s = _scorer([300.0, 20.0], [0.15, 0.50])
    s.rank_floor = 1
    s.rank_top_k = 3
    s.model = FakeModel()
    s._perf_idx, s._qual_idx = 0, 1
    s._t, s._guard_until_t = 10, 3
    seen = []

    def capture(perf, qual, sigma):
        seen.append((perf, qual, sigma))
        return perf

    s._expected_hv_gain = capture
    cand = torch.zeros((3, 2), dtype=torch.double)
    kept, kept_prov = s._prefilter_pool_with_prov(cand, ["a", "b", "c"])

    assert kept.shape[0] == 3
    assert kept_prov == ["a", "b", "c"]
    assert np.asarray(seen) == pytest.approx(np.asarray([
        (30.0, 0.2, 0.1),
        (20.0, 0.3, 0.2),
        (10.0, 0.4, 0.3),
    ]))


def test_hv_wrapper_constructs_exactly_one_phase2_optimizer(monkeypatch):
    calls = []
    original = AnchorAgentQNEHVI.__init__

    def counted(self, *args, **kwargs):
        calls.append(type(self))
        original(self, *args, **kwargs)

    monkeypatch.setattr(AnchorAgentQNEHVI, "__init__", counted)
    search_space = SearchSpace(parameters=[
        ChoiceParameter(
            "x", ParameterType.INT, [0, 1], is_ordered=True,
            sort_values=True,
        ),
    ])
    ssops = AxSearchSpaceOps(search_space)
    optimizer = GreedyThenHVAnchorAgentQNEHVI(
        search_space,
        [
            {"objective_name": "performance", "ref_point": 0, "cost": 1},
            {"objective_name": "quality", "ref_point": 0, "cost": 1},
        ],
        greedy_budget=1,
        n_iterations=2,
        ssops=ssops,
        n_candidates=4,
        dls_num_restarts=1,
    )

    assert calls == [HVAnchorQNEHVI]
    assert type(optimizer.agent) is HVAnchorQNEHVI
    assert type(optimizer.agent.agent) is HVAnchorLLMAgent


# ── cold-start guard ──────────────────────────────────────────────────────

def test_low_perf_width_reject_semantics():
    s = _scorer(
        [300, 200, 100, 20, 5],
        [0.10, 0.15, 0.20, 0.45, 0.50],
        anchor_filter_low_perf_quantile=0.5,
        anchor_filter_quality_margin_frac=0.05,
    )
    rej_slow, _ = s._low_perf_quality_width_reject(3.0, 0.30)
    keep_fast, _ = s._low_perf_quality_width_reject(280.0, 0.12)
    assert rej_slow is True
    assert keep_fast is False


def test_history_duplicate_cannot_mask_all_guarded_novel_pool():
    """Regression for the production 262->1 false-exhaustion handoff.

    The sole unguarded row was already evaluated. Removing it before HV
    ranking must expose the existing all-guarded fail-open and preserve every
    novel row with aligned provenance.
    """
    class ExactDecodedOps:
        @staticmethod
        def is_duplicate_decoded(new_x, existing_x):
            return any(torch.equal(new_x[0], row) for row in existing_x)

    class FakeModel:
        @staticmethod
        def posterior(x):
            n = x.shape[0]
            return type("Posterior", (), {
                "mean": torch.zeros(n, 1, 2),
                "variance": torch.ones(n, 1, 2) * 0.01,
            })()

    s = _scorer([100.0, 20.0], [0.1, 0.5])
    s.rank_floor = 8
    s.rank_top_k = 16
    s.model = FakeModel()
    s._perf_idx, s._qual_idx = 0, 1
    s._t, s._guard_until_t = 20, 30
    s._x_rows = [torch.tensor([9.0, 9.0])]
    s.tkwargs = {"dtype": torch.float32, "device": torch.device("cpu")}
    s.ssops = ExactDecodedOps()
    s._low_perf_quality_width_reject = lambda perf, qual: (True, "guarded")

    novel = torch.arange(20, dtype=torch.float32).reshape(10, 2) / 20
    cand = torch.cat([s._x_rows[0].reshape(1, -1), novel], dim=0)
    prov = ["evaluated"] + [f"novel-{i}" for i in range(10)]

    cand, prov = s._drop_evaluated_pool_rows(cand, prov)
    kept, kept_prov = s._prefilter_pool_with_prov(cand, prov)
    assert torch.equal(kept, novel)
    assert kept_prov == prov
    assert "evaluated" not in kept_prov


# ── wrapper greedy HV-plateau handoff ─────────────────────────────────────

def _wrapper(evals, successes, window=6, eps=0.005):
    w = GreedyThenAnchorAgentQNEHVI.__new__(GreedyThenAnchorAgentQNEHVI)
    w.greedy_plateau_window = window
    w.greedy_plateau_eps = eps
    w._greedy_successes = successes
    w._evaluations = [
        {"objectives": {"performance": p, "quality": {"combined": q}}}
        for p, q in evals
    ]
    return w


def test_plateau_detector_fires_on_flat_tail():
    evals = [(26, 0.17), (304, 0.14), (33, 0.21), (17, 0.31), (20, 0.48),
             (6, 0.55), (273, 0.16)] + [(35 + i, 0.10) for i in range(6)]
    w = _wrapper(evals, successes=len(evals))
    assert w._greedy_hv_plateaued() is True


def test_plateau_detector_quiet_while_front_moves():
    evals = [(26, 0.17), (304, 0.14), (33, 0.21), (17, 0.31), (20, 0.48),
             (6, 0.55), (273, 0.16), (40, 0.20), (50, 0.25), (290, 0.20),
             (310, 0.22), (320, 0.30), (330, 0.35)]  # tail keeps extending the front
    w = _wrapper(evals, successes=len(evals))
    assert w._greedy_hv_plateaued() is False


def test_plateau_detector_needs_history():
    evals = [(26, 0.17), (304, 0.14), (33, 0.21)]
    w = _wrapper(evals, successes=3)
    assert w._greedy_hv_plateaued() is False   # < max(10, w+2) successes


def test_plateau_disabled_by_zero_window():
    evals = [(26, 0.17)] * 20
    w = _wrapper(evals, successes=20, window=0)
    assert w._greedy_hv_plateaued() is False
