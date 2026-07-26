"""Measured evaluator OOM is a genuine, replayable config penalty."""

from __future__ import annotations

import csv
import json

from rag_stack.controller import Controller, _measured_failure_penalty
from rag_stack_evaluator.static_rag_evaluator.measured.serving_runtime import (
    _request_failure_is_fatal,
)
from rag_stack.types import EvalType


def test_measured_request_cuda_oom_is_fatal_but_transient_error_is_recycled():
    assert _request_failure_is_fatal(
        RuntimeError(
            "aux worker LLMLingua2 failed: OutOfMemoryError: "
            "CUDA out of memory. Tried to allocate 384.00 MiB"
        )
    )
    assert not _request_failure_is_fatal(
        RuntimeError("temporary upstream connection reset")
    )


def test_measured_oom_builds_exact_zero_multi_objective_penalty():
    outcome, objectives, quality = _measured_failure_penalty(
        RuntimeError("CUDA out of memory while allocating KV cache"),
        eval_gt=True,
        metric_names=("answer_correctness", "faithfulness"),
    )

    assert outcome is EvalType.OOM
    assert objectives == {
        "performance": 0.0,
        "quality": {
            "combined": 0.0,
            "answer_correctness": 0.0,
            "faithfulness": 0.0,
        },
    }
    assert quality == objectives["quality"]


def test_measured_perf_only_oom_has_no_synthetic_quality():
    outcome, objectives, quality = _measured_failure_penalty(
        RuntimeError("OutOfMemoryError: engine launch OOM"),
        eval_gt=False,
        metric_names=("answer_correctness",),
    )

    assert outcome is EvalType.OOM
    assert objectives == {"performance": 0.0}
    assert quality is None


def test_measured_context_overflow_is_penalty_but_other_error_is_invalid():
    context = _measured_failure_penalty(
        RuntimeError("This model's maximum context length is 32768 tokens"),
        eval_gt=True,
        metric_names=("answer_correctness",),
    )
    bug = _measured_failure_penalty(
        RuntimeError("judge returned malformed JSON"),
        eval_gt=True,
        metric_names=("answer_correctness",),
    )

    assert context[0] is EvalType.CONTEXT_OVERFLOW
    assert context[1]["performance"] == 0.0
    assert context[1]["quality"]["combined"] == 0.0
    assert bug == (EvalType.INVALID, {}, None)


class _SearchSpaceOps:
    @staticmethod
    def enrich_configs(configs, _base):
        return configs

    @staticmethod
    def decode_to_nested(flat):
        mode = flat.get("pipeline__mode")
        return {"pipeline": {"mode": mode}} if mode else {}

    @staticmethod
    def hierarchical_selector_paths():
        # This fixture has no dict-form hierarchical selectors.  Keep the
        # fake aligned with the production search-space interface without
        # weakening checkpoint selector restoration.
        return set()


class _Optimizer:
    objective_costs = {"performance": 1, "quality": 10}

    def __init__(self):
        self.warm_started = None

    @staticmethod
    def pending_observation_metadata():
        return {"forced": True, "agent_proposal_round": 3}

    def warm_start(self, evaluations):
        self.warm_started = evaluations


def _bare_controller(project_dir, optimizer=None):
    controller = Controller.__new__(Controller)
    controller.project_dir = str(project_dir)
    controller._eval_counter = 0
    controller._failure_counts = {}
    controller._invalid_configs = []
    controller._invalid_config_keys = set()
    controller._ssops = _SearchSpaceOps()
    controller._base_decoded_config = {}
    controller.config = {}
    controller.optimizer = optimizer or _Optimizer()
    controller._quality_memo_enabled = False
    controller._quality_memo = {}
    controller._checkpoint_loaded = False
    controller._cm_count = 0
    controller._gt_count = 0
    controller._gt_memo_hits = 0
    return controller


def test_measured_oom_zero_objectives_persist_and_replay_on_resume(tmp_path):
    config = {"pipeline": {"mode": "react"}}
    _, objectives, _ = _measured_failure_penalty(
        RuntimeError("CUDA out of memory"),
        eval_gt=True,
        metric_names=("answer_correctness",),
    )
    writer = _bare_controller(tmp_path)
    writer._save_eval_record(
        eval_id=0,
        config=config,
        pipeline_config=config,
        eval_type=EvalType.OOM,
        eval_error="measured: RuntimeError: CUDA out of memory",
        objectives=objectives,
        quality_metrics=None,
        assembly_result=None,
        obj_columns=[
            "performance", "quality__combined", "quality__answer_correctness",
        ],
    )

    with open(tmp_path / "all_evaluations.csv", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["eval_type"] == "OOM"
    assert row["performance"] == "0.000000"
    assert row["quality__combined"] == "0.000000"
    assert row["quality__answer_correctness"] == "0.000000"

    # The OOM is the only row: resume must still reach the CSV penalty replay
    # rather than returning early because no performance.json exists.
    with open(tmp_path / "invalid_configs.json", "w") as handle:
        json.dump(
            [
                {
                    "eval_id": 0,
                    "config": config,
                    "reason": "MEASURED: RuntimeError: CUDA out of memory",
                },
                {
                    "eval_id": 1,
                    "config": {"pipeline": {"mode": "sequential"}},
                    "reason": "MEASURED: RuntimeError: judge returned malformed JSON",
                },
            ],
            handle,
        )

    optimizer = _Optimizer()
    resumed = _bare_controller(tmp_path, optimizer)
    resumed._load_checkpoint()

    assert resumed._checkpoint_loaded is True
    assert len(optimizer.warm_started) == 1
    replay = optimizer.warm_started[0]
    assert replay["eval_type"] == "OOM"
    assert replay["objectives"] == {
        "performance": 0.0,
        "quality": {
            "combined": 0.0,
            "answer_correctness": 0.0,
        },
    }
    assert replay["free"] is True
    assert replay["optimizer_metadata"] == {
        "forced": True,
        "agent_proposal_round": 3,
    }
    assert resumed._gt_count == 0
    assert resumed._cm_count == 0
    assert resumed._is_known_invalid(config)
    assert not resumed._is_known_invalid(
        {"pipeline": {"mode": "sequential"}}
    )
