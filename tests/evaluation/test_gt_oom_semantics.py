"""GT (quality) arm OOM semantics: harness failure, never an INVALID verdict.

The unified retry policy says OOM is deterministic-per-config — true for the
MEASURED arm (the deployment *is* the sampled config) but wrong for the GT
static arm in cost_model mode: quality is deployment-independent, so a CUDA
OOM there is eval-infrastructure pressure. The controller's GT call site
overrides it to retryable, and aborts the run (judge-outage-grade) if the OOM
survives all attempts. These tests pin the pieces that implement that.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_stack.controller import (
    _checkpoint_next_eval_id,
    _checkpoint_penalty_row_is_replayable,
    _eval_retry_decision,
    _invalid_record_is_config_verdict,
    _should_abort_eval_failure,
)
from rag_stack.types import EvalType
from rag_stack.project_config import CONFIG_FILENAME, freeze_project_config


def _oom_exc() -> Exception:
    return RuntimeError(
        "CUDA out of memory. Tried to allocate 512.00 MiB. GPU 0 has a total "
        "capacity of 93.09 GiB of which 445.12 MiB is free."
    )


def test_oom_classified_and_base_policy_nonretryable():
    """Base (measured-arm) policy is unchanged: OOM → deterministic, no retry."""
    et, retryable = _eval_retry_decision(_oom_exc())
    assert et is EvalType.OOM
    assert retryable is False


def test_oom_alone_does_not_trip_global_abort_helper():
    """_should_abort_eval_failure stays judge-outage-only (measured arm relies
    on it); the GT arm's OOM abort is wired at its call site instead."""
    assert _should_abort_eval_failure(_oom_exc()) is False


def test_gt_oom_retries_then_aborts_without_penalty_or_next_candidate(monkeypatch, tmp_path):
    """Exercise the real controller loop with a failing, model-free evaluator."""
    import rag_stack.controller as controller_module
    from rag_stack.controller import Controller

    monkeypatch.setattr(controller_module, "_MAX_EVAL_RETRIES", 2)
    backoff = Mock()
    monkeypatch.setattr(controller_module.time, "sleep", backoff)
    quality = Mock(side_effect=_oom_exc())
    optimizer = SimpleNamespace(
        n_both_init=2,
        objective_names=["performance", "quality"],
        objective_costs={"performance": 1, "quality": 10},
        _pending_full_metrics=[],
        suggest=Mock(return_value=([{"x": 1}, {"x": 2}], ["performance", "quality"])),
        add_objectives=Mock(),
    )
    controller = Controller.__new__(Controller)
    controller.optimizer = optimizer
    controller.project_dir = str(tmp_path)
    controller.config = {}
    (tmp_path / CONFIG_FILENAME).write_text("{}\n")
    controller._project_config_raw = freeze_project_config(tmp_path)
    controller._project_runtime_config = {}
    controller._combined_quality_spec = {}
    controller._eval_counter = 0
    controller._total_cost = 0.0
    controller._gt_count = 0
    controller._quality_memo_enabled = False
    controller._cm_fixed_provider = None
    controller._measured_provider = None
    controller._assembly_results = []
    controller._data = SimpleNamespace(
        resolve_corpus=lambda _params: SimpleNamespace(chunk_hash="fixture"),
        apply_corpus_view=lambda _config, _view: None,
    )
    controller.rag_quality_evaluator = SimpleNamespace(evaluate=quality)
    controller.assembly = SimpleNamespace(evaluate_dynamic=Mock(), evaluate_static=Mock())
    controller._is_known_invalid = lambda _config: False
    controller._resolve_config = lambda config: dict(config)
    controller._configure_rag_ir_mode = lambda _config: None
    controller._record_invalid = Mock()
    controller._save_eval_record = Mock()
    controller._update_summary = Mock()

    with pytest.raises(RuntimeError, match="Stopping optimization.*CUDA out of memory"):
        controller._run_phase_1(optimizer, obj_columns=[])

    assert quality.call_count == 2
    assert all(call.args[0] == {"x": 1} for call in quality.call_args_list)
    backoff.assert_called_once()
    assert controller._eval_counter == 1, "the next candidate must not run after exhausted OOM"
    optimizer.add_objectives.assert_not_called()
    controller._record_invalid.assert_not_called()
    controller.assembly.evaluate_dynamic.assert_not_called()
    controller.assembly.evaluate_static.assert_not_called()
    controller._save_eval_record.assert_called_once()
    saved = controller._save_eval_record.call_args.kwargs
    assert saved["eval_type"] is EvalType.INVALID
    assert saved["objectives"] == {}
    assert saved["quality_metrics"] is None
    assert saved["eval_error"].startswith("GT:") and "CUDA out of memory" in saved["eval_error"]
    controller._update_summary.assert_called_once()


def test_context_overflow_still_nonretryable_config_property():
    """CONTEXT_OVERFLOW stays a config property (top_k × chunk_size floods the
    window every time) — not affected by the OOM override."""
    exc = RuntimeError(
        "This model's maximum context length is 32768 tokens. However, you "
        "requested 40000 tokens in the messages."
    )
    et, retryable = _eval_retry_decision(exc)
    assert et is EvalType.CONTEXT_OVERFLOW
    assert retryable is False


def test_gt_harness_oom_is_not_a_resumed_infeasible_advisory():
    """A legacy GT OOM record is audit evidence only, not a config verdict."""
    assert not _invalid_record_is_config_verdict({"reason": f"GT: {_oom_exc()}"})


def test_persisted_failure_verdict_filter_keeps_only_config_properties():
    assert _invalid_record_is_config_verdict({
        "reason": "GT: maximum context length is 32768 tokens; requested 40000",
    })
    assert _invalid_record_is_config_verdict({
        "reason": "MEASURED: CUDA out of memory while launching sampled deployment",
    })
    assert not _invalid_record_is_config_verdict({
        "reason": "GT: judge failed on ALL test cases",
    })
    assert _invalid_record_is_config_verdict({
        "reason": "resolve_config: invalid preset",
    })


def test_structured_verdict_survives_typed_cm_oom_string_loss():
    """New records do not try to re-prove a typed CM failure from text."""
    legacy_reason = "CM: every concrete deployment is out of memory"
    assert not _invalid_record_is_config_verdict({"reason": legacy_reason})
    assert _invalid_record_is_config_verdict({
        "reason": legacy_reason,
        "config_verdict": True,
    })


def test_failure_only_checkpoint_keeps_raw_eval_id_monotonic():
    assert _checkpoint_next_eval_id(
        ["eval_0000", "eval_0007", "not_an_eval"],
        [{"eval_id": "9"}, {"eval_id": "broken"}],
    ) == 10


def test_legacy_gt_harness_oom_is_never_replayed_as_gp_penalty():
    assert not _checkpoint_penalty_row_is_replayable({
        "eval_type": "OOM",
        "eval_error": f"GT: {_oom_exc()}",
        "performance": "0.0",
        "quality__combined": "0.0",
    })
    assert _checkpoint_penalty_row_is_replayable({
        "eval_type": "CONTEXT_OVERFLOW",
        "eval_error": "GT: maximum context length exceeded",
    })
    assert _checkpoint_penalty_row_is_replayable({
        "eval_type": "OOM",
        "eval_error": "CM: every concrete deployment is out of memory",
    })
