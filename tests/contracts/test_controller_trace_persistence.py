"""Host controller persists each evaluator trace under its own contract."""
import json

from rag_stack_evaluator.static_rag_evaluator.measured.performance_trace import make_performance_trace_envelope


def _calls(output_tokens: int) -> list[dict]:
    return [
        {
            "stage": "semantic_retrieval_encode",
            "input_tokens": 5,
            "output_tokens": 0,
            "input_bytes": 20,
            "output_bytes": 0,
            "step_idx": 0,
            "model_id": "embed",
        },
        {
            "stage": "semantic_retrieval_vectorsearch",
            "input_tokens": 5,
            "output_tokens": 0,
            "input_bytes": 20,
            "output_bytes": 40,
            "step_idx": 1,
            "model_id": None,
        },
        {
            "stage": "generator",
            "input_tokens": 50,
            "output_tokens": output_tokens,
            "input_bytes": 200,
            "output_bytes": output_tokens * 4,
            "step_idx": 2,
            "model_id": "generator",
        },
    ]


def test_controller_persists_quality_and_performance_traces_separately(tmp_path):
    from rag_stack.controller import Controller
    from rag_stack.rag_ir import make_quality_trace_envelope
    from rag_stack.types import EvalType

    class SearchSpace:
        def enrich_configs(self, configs, _base):
            return configs

    controller = Controller.__new__(Controller)
    controller.project_dir = str(tmp_path)
    controller._failure_counts = {}
    controller._ssops = SearchSpace()
    controller._base_decoded_config = {}
    exact_system_config = {
        "hardware": {"available_gpus": ["cuda:0"]},
        "layout": {"available_devices": ["cuda:0"]},
    }

    quality = make_quality_trace_envelope(
        [_calls(2)], question_ids=["dataset-0"],
    )
    performance = make_performance_trace_envelope(
        [_calls(9)],
        invocation_ids=["measured-7"],
        source_question_ids=["dataset-0"],
        capacity=1,
        population_queries=12,
    )
    controller._save_eval_record(
        eval_id=3,
        config={},
        pipeline_config={},
        eval_type=EvalType.GT,
        objectives={"performance": 1.0},
        quality_metrics={"exact_match": 0.5},
        assembly_result=None,
        obj_columns=["performance"],
        execution_dag=quality,
        performance_execution_trace=performance,
        system_config=exact_system_config,
    )

    eval_dir = tmp_path / "evaluations" / "eval_0003"
    assert json.loads((eval_dir / "execution_dag.json").read_text()) == quality
    assert json.loads(
        (eval_dir / "performance_execution_trace.json").read_text()
    ) == performance
    metrics = json.loads((eval_dir / "quality_metrics.json").read_text())
    assert metrics == {"exact_match": 0.5}
    assert json.loads((eval_dir / "system_config.json").read_text()) == (
        exact_system_config
    )
