"""Measured optimizer observations fail closed at every publication boundary."""

from __future__ import annotations

import csv
import json
from copy import deepcopy

import pytest

from rag_stack.controller import Controller, _eval_retry_decision
from rag_stack_evaluator.static_rag_evaluator.measured import provider as provider_module
from rag_stack_evaluator.static_rag_evaluator.measured.artifact_contract import (
    MeasuredGTInadmissibleError,
    require_measured_gt_admissible,
)
from rag_stack_evaluator.static_rag_evaluator.measured.provider import MeasuredProvider
from rag_stack.types import EvalType


_CHECK_NAMES = (
    "driver_errors_window_zero",
    "measurement_rate_stability_waves_ge_2",
    "population_saturated",
    "qps_subwindow_cv_le_0_15",
    "warmup_rate_stable_not_cap_forced",
)


def _verdict(admissible: bool) -> dict:
    reasons = [] if admissible else [
        "qps_subwindow_cv_missing_or_above_maximum"
    ]
    return {
        "admissible": admissible,
        "reasons": reasons,
        "checks": {
            name: {
                "observed": None,
                "passed": admissible,
                "required": True,
            }
            for name in _CHECK_NAMES
        },
    }


def _performance(admissible: bool) -> dict:
    return {
        "qps": 12.5,
        "population_saturated": True,
        "measured_gt_admissibility": _verdict(admissible),
    }


def test_publication_gate_is_retryable_and_fail_closed():
    assert require_measured_gt_admissible(_performance(True))["admissible"]

    with pytest.raises(
        MeasuredGTInadmissibleError,
        match="qps_subwindow_cv_missing_or_above_maximum",
    ) as caught:
        require_measured_gt_admissible(_performance(False))
    assert _eval_retry_decision(caught.value) == (EvalType.INVALID, True)

    with pytest.raises(
        MeasuredGTInadmissibleError,
        match="invalid measured_gt_admissibility verdict",
    ):
        require_measured_gt_admissible({"qps": 12.5})


class _FakeCache:
    def __init__(self):
        self.hf_evictions = 0
        self.vllm_evictions = 0

    def evict_all_hf_models(self):
        self.hf_evictions += 1

    def evict_vllm(self):
        self.vllm_evictions += 1


class _FakeMeasured:
    def __init__(self, performance: dict):
        self.performance = performance
        self.calls = 0

    def evaluate(self, **_kwargs):
        self.calls += 1
        return {
            "quality": {"answer_correctness": 0.5},
            "performance": deepcopy(self.performance),
            "performance_execution_trace": {"kind": "test"},
        }


def _provider(performance: dict) -> MeasuredProvider:
    provider = object.__new__(MeasuredProvider)
    provider._cache = _FakeCache()
    provider._measured = _FakeMeasured(performance)
    provider._deploy = object()
    provider._n_queries = None
    provider._selection = "max_throughput"
    provider._memo = {}
    provider._dirty = False
    return provider


@pytest.mark.parametrize(
    ("admissible", "diagnostic", "expected_calls", "expected_memo_size"),
    [(False, True, 2, 0), (True, False, 1, 1)],
)
def test_provider_only_memoizes_publishable_results(
    monkeypatch, admissible, diagnostic, expected_calls, expected_memo_size,
):
    monkeypatch.setattr(
        provider_module, "apply_measured_generation_defaults", lambda value: value
    )
    provider = _provider(_performance(admissible))

    first = provider.evaluate({}, {}, require_admissible=not diagnostic)
    second = provider.evaluate({}, {}, require_admissible=not diagnostic)

    assert first.performance_score == pytest.approx(12.5)
    assert second.performance_score == pytest.approx(12.5)
    assert provider._measured.calls == expected_calls
    assert len(provider._memo) == expected_memo_size


def test_provider_strict_call_persists_diagnostic_without_memoizing(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        provider_module, "apply_measured_generation_defaults", lambda value: value
    )
    provider = _provider(_performance(False))

    with pytest.raises(MeasuredGTInadmissibleError):
        provider.evaluate({}, {}, run_dir=str(tmp_path))

    assert provider._measured.calls == 1
    assert provider._memo == {}
    diagnostic = json.loads(
        (tmp_path / "inadmissible_attempt_001.json").read_text()
    )
    assert diagnostic["status"] == "inadmissible"
    assert diagnostic["performance_score"] == pytest.approx(12.5)
    assert diagnostic["performance"]["measured_gt_admissibility"][
        "admissible"
    ] is False


class _SearchSpaceOps:
    @staticmethod
    def hierarchical_selector_paths():
        # This fixture has no hierarchical selectors; checkpoint recovery
        # still consumes the same explicit interface as the production space.
        return set()

    @staticmethod
    def decode_to_nested(flat):
        mode = flat.get("pipeline__mode")
        return {"pipeline": {"mode": mode}} if mode else {}


class _Optimizer:
    objective_costs = {"performance": 1, "quality": 10}

    def __init__(self):
        self.warm_started = None

    def warm_start(self, evaluations):
        self.warm_started = evaluations


def _checkpoint_controller(project_dir) -> tuple[Controller, _Optimizer]:
    optimizer = _Optimizer()
    controller = Controller.__new__(Controller)
    controller.project_dir = str(project_dir)
    controller._failure_counts = {}
    controller._invalid_configs = []
    controller._invalid_config_keys = set()
    controller._ssops = _SearchSpaceOps()
    controller._base_decoded_config = {}
    controller._combined_quality_spec = None
    controller.config = {}
    controller.optimizer = optimizer
    controller._measured_provider = object()
    controller._quality_memo_enabled = False
    controller._quality_memo = {}
    controller._checkpoint_loaded = False
    controller._eval_counter = 0
    controller._cm_count = 0
    controller._gt_count = 0
    controller._gt_memo_hits = 0
    return controller, optimizer


def test_checkpoint_excludes_inadmissible_measured_gt(tmp_path):
    evaluations = tmp_path / "evaluations"
    for eval_id, admissible in enumerate((False, True)):
        eval_dir = evaluations / f"eval_{eval_id:04d}"
        eval_dir.mkdir(parents=True)
        (eval_dir / "pipeline_config.json").write_text(
            json.dumps({"pipeline": {"mode": f"mode-{eval_id}"}})
        )
        (eval_dir / "quality_metrics.json").write_text(
            json.dumps({"answer_correctness": 0.5 + eval_id / 10})
        )
        (eval_dir / "performance.json").write_text(json.dumps({
            "performance_score": 10.0 + eval_id,
            "performance": _performance(admissible),
        }))

    with (tmp_path / "all_evaluations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "pipeline__mode", "performance", "quality__combined",
            "eval_type", "eval_error", "eval_id",
        ])
        writer.writeheader()
        for eval_id in range(2):
            writer.writerow({
                "pipeline__mode": f"mode-{eval_id}",
                "performance": 10.0 + eval_id,
                "quality__combined": 0.5 + eval_id / 10,
                "eval_type": "GT",
                "eval_error": "",
                "eval_id": eval_id,
            })

    controller, optimizer = _checkpoint_controller(tmp_path)
    controller._load_checkpoint()

    assert controller._checkpoint_loaded is True
    assert controller._gt_count == 1
    assert optimizer.warm_started is not None
    assert [item["eval_id"] for item in optimizer.warm_started] == [1]
    assert optimizer.warm_started[0]["objectives"]["performance"] == 11.0
    assert optimizer.warm_started[0]["config"] == {"pipeline": {"mode": "mode-1"}}
