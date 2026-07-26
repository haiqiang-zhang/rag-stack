"""Cache recording requests canonical traces regardless of the CM mode."""

from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.evaluation import EvaluationResult, TraceSession
from rag_stack.quality_store import QualityStore, _algo_record_key
from rag_stack.search_space import RagSearchSpace


@pytest.mark.parametrize("backend, cm_mode", [
    ("static_gt", "static"),
    ("custom_backend", "static"),
    ("custom_backend", "dynamic"),
])
def test_store_records_dynamic_trace_when_project_uses_static_cm(
    monkeypatch, tmp_path, backend, cm_mode,
):
    resolver = SimpleNamespace(
        resolve_pipeline_block=lambda config, decoded: ({"mode": "sequential"}, []),
        resolve_vectordbs=lambda config, decoded: None,
    )
    monkeypatch.setattr(RagSearchSpace, "from_yaml", lambda config: resolver)
    store = object.__new__(QualityStore)
    store.config = {"global": {"eval_backend": backend, "rag_ir_mode": cm_mode}}
    store.store_path = str(tmp_path)
    store._data = SimpleNamespace(resolve_corpus_chunker=lambda decoded: {})
    store.qa_data = pd.DataFrame({"qid": ["task-1"]})
    store._evaluated = {}
    store._combined_quality_spec = None
    store._canonical_algo_config = lambda config: dict(config)
    captured = []

    class ModeSensitiveBackend:
        def evaluate(self, config, run_dir=None):
            if config.get("_runtime", {}).get("rag_ir_mode") == "static":
                return EvaluationResult({"quality": 0.75}, runtime_stats={"n_queries": 1})
            session = TraceSession()
            with session.query("task-1"):
                with session.stage("generator", model_id="fixture-generator") as call:
                    call.input(token_count=3)
                    call.output(token_count=2)
            return EvaluationResult(
                {"quality": 0.75}, session.finish(expected_question_ids=["task-1"]),
            )

    store.rag_quality_evaluator = ModeSensitiveBackend()

    def append_record(config, pipeline_config, quality_metrics, execution_dag, **kwargs):
        # The production evaluate_configs path validates the canonical trace
        # before this storage boundary; aggregate results never reach here.
        captured.append((pipeline_config, execution_dag))
        key = _algo_record_key(config)
        store._evaluated[key] = {"quality_metrics": quality_metrics}
        return key, True

    store.append_record = append_record
    store._save_manifest = lambda: None

    def unexpected_failure(key, config, error):
        pytest.fail(f"Canonical QualityStore evaluation failed: {error}")

    store._save_failed = unexpected_failure
    records = store.evaluate_configs([{}])

    assert records == [{"quality_metrics": {"quality": 0.75}}]
    pipeline_config, trace = captured[0]
    assert pipeline_config["_runtime"]["rag_ir_mode"] == "dynamic"
    assert trace["queries"][0]["question_id"] == "task-1"
    assert trace["provenance"]["evaluator_backend"] == backend
    assert store.config["global"]["rag_ir_mode"] == cm_mode
    assert "_runtime" not in store.config
