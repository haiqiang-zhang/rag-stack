"""Regression coverage for judge/CM overlap timeout handling."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_stack.project_config import CONFIG_FILENAME, freeze_project_config


def test_overlap_timeout_records_invalid_and_aborts_before_next_eval(
    monkeypatch, tmp_path,
):
    """A stuck overlap is infrastructure noise, never a GP observation.

    The first candidate must be recorded INVALID and the batch must abort
    before candidate two (or ``optimizer.add_objectives``) can run.
    """
    import rag_stack.controller as controller_module
    from rag_stack.controller import Controller, OverlappedCMTimeoutError
    from rag_stack.rag_ir.trace import make_quality_trace_envelope
    from rag_stack.types import EvalType

    monkeypatch.setattr(controller_module, "_CM_OVERLAP_TIMEOUT_S", 0.01)
    monkeypatch.setattr(
        controller_module,
        "build_workflow_schema",
        lambda _pipeline_config: object(),
    )

    trace = make_quality_trace_envelope(
        [[{
            "stage": "generator",
            "input_tokens": 4,
            "output_tokens": 2,
            "step_idx": 0,
        }]],
        question_ids=["q0"],
    )
    cm_started = threading.Event()
    release_cm = threading.Event()
    cm_exited = threading.Event()

    def stuck_cm(*_args, **_kwargs):
        cm_started.set()
        try:
            assert release_cm.wait(5), "test failed to release fake CM worker"
            return SimpleNamespace(performance_score=1.0)
        finally:
            cm_exited.set()

    def quality_eval(_pipeline_config, *, on_trace_ready, **_kwargs):
        on_trace_ready(trace)
        assert cm_started.wait(1)
        return {"score": 0.5, "__execution_dag__": trace}

    controller = Controller.__new__(Controller)
    optimizer = SimpleNamespace(
        n_both_init=2,
        objective_names=["performance", "quality"],
        objective_costs={"performance": 1, "quality": 10},
        _pending_full_metrics=[],
        suggest=Mock(return_value=([{"x": 1}, {"x": 2}], ["performance", "quality"])),
        add_objectives=Mock(),
    )
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
        resolve_corpus=lambda _params: SimpleNamespace(chunk_hash="test"),
        apply_corpus_view=lambda _pipeline_config, _view: None,
    )
    controller.rag_quality_evaluator = SimpleNamespace(evaluate=quality_eval)
    controller.assembly = SimpleNamespace(evaluate_dynamic=stuck_cm)
    controller._is_known_invalid = lambda _cfg: False
    controller._resolve_config = lambda cfg: dict(cfg)
    controller._configure_rag_ir_mode = lambda _pipeline_config: None
    controller._record_invalid = Mock()
    controller._save_eval_record = Mock()
    controller._update_summary = Mock()

    try:
        with pytest.raises(OverlappedCMTimeoutError, match="did not finish"):
            controller._run_phase_1(optimizer, obj_columns=[])
    finally:
        # Python threads are not force-cancellable. Production exits through
        # the CLI reaper; the test releases its fake worker to avoid leakage.
        release_cm.set()
        assert cm_exited.wait(1)

    assert controller._eval_counter == 1, "candidate two must never start"
    optimizer.add_objectives.assert_not_called()
    # Infrastructure noise is audit-only: it must not enter the persistent
    # config-infeasibility blacklist.
    controller._record_invalid.assert_not_called()
    controller._update_summary.assert_called_once()

    saved = controller._save_eval_record.call_args.kwargs
    assert saved["eval_id"] == 0
    assert saved["eval_type"] is EvalType.INVALID
    assert saved["objectives"] == {}
    assert "overlapped CM did not finish" in saved["eval_error"]


def test_judge_failure_after_trace_publish_keeps_dynamic_cm_dispatch(
    monkeypatch, tmp_path,
):
    """A post-pipeline metric failure must not erase an agentic trace.

    Quality and CM are independent arms: the row is still INVALID because the
    judge failed, but the already-published workload is priced dynamically and
    no spurious static-dispatch error is appended to the audit record.
    """
    import rag_stack.controller as controller_module
    from rag_stack.controller import Controller
    from rag_stack.rag_ir.trace import make_quality_trace_envelope
    from rag_stack.types import EvalType

    monkeypatch.setattr(controller_module, "_MAX_EVAL_RETRIES", 1)
    schema = SimpleNamespace(dataflow_type="agentic")
    monkeypatch.setattr(
        controller_module,
        "build_workflow_schema",
        lambda _pipeline_config: schema,
    )

    trace = make_quality_trace_envelope(
        [[{
            "stage": "generator",
            "input_tokens": 4,
            "output_tokens": 2,
            "step_idx": 0,
        }]],
        question_ids=["q0"],
    )

    def quality_eval(_pipeline_config, *, on_trace_ready, **_kwargs):
        on_trace_ready(trace)
        raise ValueError(
            "FaithfulnessMetric: structurally missing retrieved_contents"
        )

    dynamic = Mock(return_value=SimpleNamespace(performance_score=17.0))
    static = Mock(side_effect=AssertionError("static CM must not be called"))
    controller = Controller.__new__(Controller)
    optimizer = SimpleNamespace(
        n_both_init=1,
        objective_names=["performance", "quality"],
        objective_costs={"performance": 1, "quality": 10},
        _pending_full_metrics=[],
        suggest=Mock(return_value=([{"x": 1}], ["performance", "quality"])),
        add_objectives=Mock(),
    )
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
        resolve_corpus=lambda _params: SimpleNamespace(chunk_hash="test"),
        apply_corpus_view=lambda _pipeline_config, _view: None,
    )
    controller.rag_quality_evaluator = SimpleNamespace(evaluate=quality_eval)
    controller.assembly = SimpleNamespace(
        evaluate_dynamic=dynamic,
        evaluate_static=static,
    )
    controller._is_known_invalid = lambda _cfg: False
    controller._resolve_config = lambda cfg: dict(cfg)
    controller._configure_rag_ir_mode = lambda _pipeline_config: None
    controller._record_invalid = Mock()
    controller._save_eval_record = Mock()
    controller._update_summary = Mock()

    with pytest.raises(RuntimeError, match="GT: FaithfulnessMetric"):
        controller._run_phase_1(optimizer, obj_columns=[])

    dynamic.assert_called_once()
    assert dynamic.call_args.kwargs["trace"] is trace
    static.assert_not_called()
    optimizer.add_objectives.assert_not_called()
    saved = controller._save_eval_record.call_args.kwargs
    assert saved["eval_type"] is EvalType.INVALID
    assert saved["execution_dag"] is trace
    assert saved["eval_error"].startswith("GT: FaithfulnessMetric")
    assert " | CM:" not in saved["eval_error"]


@pytest.mark.parametrize("dataflow_type", ["agentic", "iterative"])
def test_nonsequential_workflow_without_trace_never_dispatches_static(
    monkeypatch, tmp_path, dataflow_type,
):
    """Missing agentic/iterative traces fail at dispatch, not in StaticEngine."""
    import rag_stack.controller as controller_module
    from rag_stack.controller import Controller

    schema = SimpleNamespace(dataflow_type=dataflow_type)
    monkeypatch.setattr(
        controller_module,
        "build_workflow_schema",
        lambda _pipeline_config: schema,
    )

    dynamic = Mock()
    static = Mock(side_effect=AssertionError("static CM must not be called"))
    controller = Controller.__new__(Controller)
    optimizer = SimpleNamespace(
        n_both_init=1,
        objective_names=["performance", "quality"],
        objective_costs={"performance": 1, "quality": 10},
        _pending_full_metrics=[],
        suggest=Mock(return_value=([{"x": 1}], ["performance", "quality"])),
        add_objectives=Mock(),
    )
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
        resolve_corpus=lambda _params: SimpleNamespace(chunk_hash="test"),
        apply_corpus_view=lambda _pipeline_config, _view: None,
    )
    controller.rag_quality_evaluator = SimpleNamespace(
        evaluate=lambda *_args, **_kwargs: {"score": 0.5},
    )
    controller.assembly = SimpleNamespace(
        evaluate_dynamic=dynamic,
        evaluate_static=static,
    )
    controller._is_known_invalid = lambda _cfg: False
    controller._resolve_config = lambda cfg: dict(cfg)
    controller._configure_rag_ir_mode = lambda _pipeline_config: None
    controller._record_invalid = Mock()
    controller._save_eval_record = Mock()
    controller._update_summary = Mock()

    with pytest.raises(RuntimeError, match="had an infrastructure failure"):
        controller._run_phase_1(optimizer, obj_columns=[])

    dynamic.assert_not_called()
    static.assert_not_called()
    saved = controller._save_eval_record.call_args.kwargs
    assert (
        "CM: eval_0000: agentic/iterative workflow did not publish a "
        "canonical quality trace; refusing static CM dispatch"
        in saved["eval_error"]
    )
