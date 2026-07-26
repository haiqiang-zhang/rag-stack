"""CM-only OOM classification keeps source provenance explicit."""

from rag_stack.controller import _classify_eval_arm_failures
from rag_stack.types import CMOutOfMemoryError, EvalType


GENZ_DEVICE_OOM = RuntimeError(
    "generator_decode: exact-device GenZ repricing failed for ['cuda:3']: "
    "tp=1,pp=1: ValueError: All params would not fit on chip. "
    "System Memory Cap:93.0 GB, Weights:5.75 GB, KV Cache:102.30 GB"
)

LEGACY_EVAL17_OOM = RuntimeError(
    "Exact-device repricing failed for every structurally valid pair of "
    "disaggregated generator: P=(1, 1),D=(1, 1): RuntimeError: "
    "generator_decode: exact-device GenZ repricing failed for ['cuda:3']: "
    "tp=1,pp=1: ValueError: All params would not fit on chip. "
    "System Memory Cap:93.0 GB, Weights:5.75 GB, KV Cache:102.30 GB"
)


def test_untyped_genz_attempt_is_not_an_eval_level_oom():
    # A per-attempt diagnostic is not proof that every candidate failed.
    assert EvalType.classify_failure(GENZ_DEVICE_OOM) is EvalType.INVALID
    assert EvalType.classify_cm_failure(GENZ_DEVICE_OOM) is EvalType.INVALID


def test_typed_cm_capacity_is_oom_only_with_cm_provenance():
    error = CMOutOfMemoryError("every candidate exceeds device memory")
    assert EvalType.classify_failure(error) is EvalType.INVALID
    assert EvalType.classify_cm_failure(error) is EvalType.OOM


def test_legacy_eval17_text_is_not_promoted_without_typed_proof():
    assert EvalType.classify_failure(LEGACY_EVAL17_OOM) is EvalType.INVALID
    assert EvalType.classify_cm_failure(LEGACY_EVAL17_OOM) is EvalType.INVALID


def test_controller_records_cm_genz_capacity_as_budget_free_oom():
    outcome, gt_genuine, cm_genuine, excluded = _classify_eval_arm_failures(
        None, CMOutOfMemoryError("every candidate exceeds device memory"),
    )
    assert outcome is EvalType.OOM
    assert not gt_genuine
    assert cm_genuine
    assert not excluded


def test_same_genz_text_from_gt_is_a_bug_not_an_allowed_oom():
    outcome, gt_genuine, cm_genuine, excluded = _classify_eval_arm_failures(
        GENZ_DEVICE_OOM, None,
    )
    assert outcome is EvalType.INVALID
    assert not gt_genuine
    assert not cm_genuine
    assert excluded


def test_other_cm_failure_remains_invalid():
    outcome, _, cm_genuine, excluded = _classify_eval_arm_failures(
        None, RuntimeError("tuple index out of range"),
    )
    assert outcome is EvalType.INVALID
    assert not cm_genuine
    assert excluded


def test_cm_oom_cannot_hide_a_gt_bug():
    outcome, _, cm_genuine, excluded = _classify_eval_arm_failures(
        RuntimeError("judge returned malformed JSON"),
        CMOutOfMemoryError("every candidate exceeds device memory"),
    )
    assert outcome is EvalType.INVALID
    assert cm_genuine
    assert excluded


def test_oom_persistence_categories_remain_budget_free_penalties():
    assert EvalType.OOM.value in EvalType.excluded_from_data()
    assert EvalType.OOM.value in EvalType.reload_as_penalty()
    assert not EvalType.OOM.is_success
