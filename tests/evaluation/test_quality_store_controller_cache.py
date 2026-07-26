"""Normal ``optimize --quality-store`` is a lookup-only optimizer side cache.

The optimizer still owns candidate selection.  The store is consulted only
after a candidate has been selected, using the fully enriched algorithm
configuration (and never its top-level deployment ``system`` block).
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_stack.controller import Controller, _restored_gt_budget_count
from rag_stack.cost_model.types import AssemblyResult
from rag_stack.quality_store import QualityStore
from rag_stack.rag_ir import make_quality_trace_envelope
from rag_stack.search_space import RagSearchSpace
from rag_stack.types import EvalType


def _merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


class _EnrichingSearchSpace:
    """Small stand-in for the one API the cache-key path consumes."""

    def enrich_configs(self, configs: list[dict], base: dict) -> list[dict]:
        return [_merge(base, config) for config in configs]


def _base_config(*, chat: bool = True) -> dict:
    return {
        "pipeline": {
            "sequential": {
                "rag_dataflow": "sequential",
                "generator": {
                    "component": "vllm",
                    "model": "fixture-generator",
                    "temperature": 0.0,
                    # A fixed (non-swept) semantic must still be in the key.
                    "use_chat_template": chat,
                },
            },
        },
        "corpus": {"chunk_size": 512},
        "system": {
            "deployment_preset": "collocated_tp4",
            "batch_size_request": 16,
        },
    }


def _trace() -> dict:
    return make_quality_trace_envelope(
        [[{
            "stage": "generator",
            "input_tokens": 12,
            "output_tokens": 3,
            "step_idx": 0,
            "model_id": "fixture-generator",
        }]],
        question_ids=["q0"],
    )


def _controller(tmp_path: Path, *, chat: bool = True) -> Controller:
    controller = Controller.__new__(Controller)
    controller.project_dir = str(tmp_path / "project")
    controller.config = {
        "global": {"eval_backend": "static_gt", "rag_ir_mode": "dynamic"},
        "dataset": {
            "qa": str(tmp_path / "qa.parquet"),
            "corpus": str(tmp_path / "corpus.parquet"),
        },
        "eval_backend_setting": {
            "combined_quality": {
                "mode": "mean",
                "metrics": ["answer_correctness"],
            },
            "metrics": [{"metric_name": "answer_correctness"}],
        },
    }
    Path(controller.config["dataset"]["qa"]).touch()
    Path(controller.config["dataset"]["corpus"]).touch()
    controller._combined_quality_spec = controller.config[
        "eval_backend_setting"
    ]["combined_quality"]
    controller._ssops = _EnrichingSearchSpace()
    controller._base_decoded_config = _base_config(chat=chat)
    controller._quality_memo_enabled = True
    controller._quality_memo = {}
    controller._gt_memo_hits = 0
    controller._external_quality_store = None
    return controller


def _fake_store(tmp_path: Path, controller: Controller, trace: dict):
    store_path = tmp_path / "quality-store"
    trace_path = store_path / "traces" / "record.json"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text(json.dumps(trace))
    algo = controller._quality_algo_config({})
    key = QualityStore.key_for_config(algo)
    record = {
        "config_hash": key,
        "quality_metrics": {"answer_correctness": 0.75},
        "execution_dag_path": str(trace_path),
        "full": True,
        "record": {"quality_spec": QualityStore.quality_spec(controller.config)},
    }
    return SimpleNamespace(
        store_path=str(store_path),
        requires_quality_trace=True,
        assert_dataset_compatible=Mock(),
        get_quality_results=Mock(return_value=[record]),
        get_record=Mock(side_effect=lambda h: record if h == key else None),
        append_record=Mock(),
    )


def _prepare_evaluation(controller: Controller, trace: dict) -> None:
    from rag_stack.project_config import freeze_project_config, write_project_config

    write_project_config(controller.project_dir, json.dumps(controller.config))
    controller._project_config_raw = freeze_project_config(controller.project_dir)
    controller._project_runtime_config = deepcopy(controller.config)
    controller.optimizer = SimpleNamespace(
        objective_names=["performance", "quality"],
        objective_costs={"performance": 1, "quality": 10},
        _pending_full_metrics=[],
    )
    controller._eval_counter = 0
    controller._measured_provider = None
    controller._cm_fixed_provider = None
    controller._assembly_results = []
    controller._data = SimpleNamespace(
        resolve_corpus=lambda _params: SimpleNamespace(chunk_hash="fixture"),
        apply_corpus_view=lambda _pipeline_config, _view: None,
    )
    controller._is_known_invalid = lambda _config: False
    controller._resolve_config = lambda _config: {
        "_runtime": {"rag_ir_mode": "dynamic"},
        "corpus_runtime": {"chunker": {}},
    }
    controller._configure_rag_ir_mode = lambda _pipeline_config: None
    controller._record_invalid = Mock()
    controller._save_eval_record = Mock()
    controller._update_summary = Mock()
    controller.rag_quality_evaluator = SimpleNamespace(evaluate=Mock(
        return_value={
            "answer_correctness": 0.75,
            "__execution_dag__": trace,
        },
    ))
    controller.assembly = SimpleNamespace(evaluate_dynamic=Mock(
        return_value=SimpleNamespace(
            performance_score=19.0,
            best_hw_params=None,
        ),
    ))


def test_partial_objective_evaluation_is_rejected(tmp_path: Path):
    controller = _controller(tmp_path)
    _prepare_evaluation(controller, _trace())

    with pytest.raises(ValueError, match="complete objective set"):
        controller._evaluate_objectives([{}], ["performance"], obj_columns=[])


def test_exact_enriched_algo_hit_reuses_metrics_and_v2_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    controller = _controller(tmp_path)
    trace = _trace()
    store = _fake_store(tmp_path, controller, trace)
    monkeypatch.setattr(QualityStore, "load", lambda _path: store)
    monkeypatch.setattr(
        "rag_stack.controller.build_workflow_schema",
        lambda _pipeline_config: object(),
    )

    controller._attach_quality_store(store.store_path)
    _prepare_evaluation(controller, trace)
    result, valid, n_success, free_flags = controller._evaluate_objectives(
        [{}], ["performance", "quality"], obj_columns=[],
    )

    controller.rag_quality_evaluator.evaluate.assert_not_called()
    store.append_record.assert_not_called()
    controller.assembly.evaluate_dynamic.assert_called_once()
    assert controller.assembly.evaluate_dynamic.call_args.kwargs["trace"] == trace
    assert result["quality"] == [{
        "combined": 0.75,
        "answer_correctness": 0.75,
    }]
    assert result["performance"] == [19.0]
    assert valid == [{}]
    assert n_success == 1
    assert free_flags == [False]
    assert controller._gt_memo_hits == 1


def test_transfer_evaluation_persists_full_cm_result_but_retains_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path)
    trace = _trace()
    store = _fake_store(tmp_path, controller, trace)
    monkeypatch.setattr(QualityStore, "load", lambda _path: store)
    monkeypatch.setattr(
        "rag_stack.controller.build_workflow_schema",
        lambda _pipeline_config: object(),
    )
    controller._attach_quality_store(store.store_path)
    _prepare_evaluation(controller, trace)
    controller._retain_portable_assembly_results = True
    full_result = AssemblyResult(
        performance_score=19.0,
        best_hw_params={"qps": 19.0, "pipeline_stage_order": []},
        rago_result_df=object(),
        sweep_dfs={"generator": object()},
        swept_configs=[{"candidate": 1}],
    )
    controller.assembly.evaluate_dynamic.return_value = full_result

    result, valid, _, _ = controller._evaluate_objectives(
        [{}], ["performance", "quality"], obj_columns=[],
    )

    assert result["performance"] == [19.0]
    assert valid == [{}]
    assert controller._save_eval_record.call_args.kwargs["assembly_result"] is full_result
    assert len(controller._assembly_results) == 1
    retained = controller._assembly_results[0]
    assert retained is not full_result
    assert retained.performance_score == full_result.performance_score
    assert retained.best_hw_params == full_result.best_hw_params
    assert retained.rago_result_df is None
    assert retained.sweep_dfs is None
    assert retained.swept_configs is None
    # This synthetic result has no authoritative closed-loop DES diagnostics.
    assert controller._last_perf_attrs == [None]


def test_top_level_system_change_still_hits_same_algo_record(tmp_path: Path):
    controller = _controller(tmp_path)
    collocated = {
        "system": {
            "deployment_preset": "collocated_tp4",
            "batch_size_request": 16,
        },
    }
    disaggregated = {
        "system": {
            "deployment_preset": "disagg_2p2d",
            "batch_size_request": 256,
        },
    }

    assert controller._quality_algo_key(collocated) == controller._quality_algo_key(
        disaggregated
    )


def test_fixed_algo_semantic_change_is_a_cache_miss(tmp_path: Path):
    controller = _controller(tmp_path, chat=True)
    same_chat = {"system": {"deployment_preset": "disagg_2p2d"}}
    raw_prompt = {
        "pipeline": {
            "sequential": {
                "generator": {"use_chat_template": False},
            },
        },
    }

    store_key = controller._quality_algo_key(same_chat)
    assert controller._quality_algo_key(raw_prompt) != store_key
    controller._quality_memo[store_key] = object()
    assert controller._quality_memo.get(controller._quality_algo_key(raw_prompt)) is None


def test_fixed_chunker_defaults_are_part_of_enriched_algo_identity():
    ssops = RagSearchSpace.__new__(RagSearchSpace)

    def base(chunk_size: int) -> dict:
        return ssops.build_base_decoded_config({
            "algo_search_space": {
                "corpus": {
                    "raw_path": "/machine-specific/corpus.parquet",
                    "chunker": {
                        "component": "recursivecharacter",
                        "chunk_size": chunk_size,
                        "chunk_overlap": 0,
                    },
                },
                "pipeline": {
                    "node_lines": [],
                },
            },
        })

    small = base(512)
    large = base(2048)
    assert small["corpus"]["chunker"]["chunk_size"] == 512
    assert "raw_path" not in small["corpus"]
    assert QualityStore.key_for_config(small) != QualityStore.key_for_config(large)


def test_resume_counts_quality_cache_hits_as_consumed_gt_budget():
    evaluations = [
        {"eval_type": EvalType.GT, "free": False, "quality_memo_hit": False},
        {"eval_type": EvalType.GT, "free": False, "quality_memo_hit": True},
        {"eval_type": EvalType.GT_FP, "free": True, "quality_memo_hit": True},
        {"eval_type": EvalType.CM, "free": False, "quality_memo_hit": False},
    ]
    assert _restored_gt_budget_count(evaluations) == 2


def test_attach_does_not_warm_start_or_inject_optimizer_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    controller = _controller(tmp_path)
    optimizer = SimpleNamespace(
        suggest=Mock(),
        warm_start=Mock(),
        add_objectives=Mock(),
    )
    controller.optimizer = optimizer
    store = _fake_store(tmp_path, controller, _trace())
    monkeypatch.setattr(QualityStore, "load", lambda _path: store)

    controller._attach_quality_store(store.store_path)

    optimizer.suggest.assert_not_called()
    optimizer.warm_start.assert_not_called()
    optimizer.add_objectives.assert_not_called()
    assert set(controller._quality_memo) == {
        controller._quality_algo_key({}),
    }
    assert controller._external_quality_store is store


def test_attach_skips_same_metric_names_from_different_judge_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    controller = _controller(tmp_path)
    store = _fake_store(tmp_path, controller, _trace())
    store.get_quality_results.return_value[0]["record"]["quality_spec"] = {
        "combined_quality": {
            "mode": "mean",
            "metrics": ["answer_correctness"],
        },
        "metrics": [{
            "metric_name": "answer_correctness",
            "model": "different-judge",
        }],
    }
    monkeypatch.setattr(QualityStore, "load", lambda _path: store)

    controller._attach_quality_store(store.store_path)

    assert controller._quality_memo == {}


def test_full_cache_hit_is_projected_to_minimal_metric_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    controller = _controller(tmp_path)
    controller.config["eval_backend_setting"]["metrics"] = [
        {"metric_name": "answer_correctness"},
        {"metric_name": "deepeval_auxiliary"},
    ]
    trace = _trace()
    trace_path = tmp_path / "quality-store" / "traces" / "record.json"
    trace_path.parent.mkdir(parents=True)
    trace_path.write_text(json.dumps(trace))
    key = controller._quality_algo_key({})
    record = {
        "config_hash": key,
        "quality_metrics": {
            "answer_correctness": 0.75,
            "deepeval_auxiliary": 0.99,
        },
        "execution_dag_path": str(trace_path),
        "full": True,
        "record": {"quality_spec": QualityStore.quality_spec(controller.config)},
    }
    store = SimpleNamespace(
        store_path=str(tmp_path / "quality-store"),
        requires_quality_trace=True,
        assert_dataset_compatible=Mock(),
        get_quality_results=Mock(return_value=[record]),
        get_record=Mock(return_value=record),
        append_record=Mock(),
    )
    monkeypatch.setattr(QualityStore, "load", lambda _path: store)
    monkeypatch.setattr(
        "rag_stack.controller.build_workflow_schema",
        lambda _pipeline_config: object(),
    )

    controller._attach_quality_store(store.store_path)
    _prepare_evaluation(controller, trace)
    controller.optimizer._pending_full_metrics = [False]
    result, _, _, _ = controller._evaluate_objectives(
        [{}], ["performance", "quality"], obj_columns=[],
    )

    controller.rag_quality_evaluator.evaluate.assert_not_called()
    assert result["quality"] == [{
        "combined": 0.75,
        "answer_correctness": 0.75,
    }]


def test_fresh_gt_miss_writes_metrics_and_trace_as_one_store_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    controller = _controller(tmp_path)
    controller.config["eval_backend_setting"]["submetric_query_subset"] = 15
    trace = _trace()
    _prepare_evaluation(controller, trace)
    store = SimpleNamespace(append_record=Mock())
    controller._external_quality_store = store
    monkeypatch.setattr(
        "rag_stack.controller.build_workflow_schema",
        lambda _pipeline_config: object(),
    )

    candidate = {
        "system": {
            "deployment_preset": "disagg_2p2d",
            "batch_size_request": 128,
        },
    }
    controller._evaluate_objectives(
        [candidate], ["performance", "quality"], obj_columns=[],
    )

    controller.rag_quality_evaluator.evaluate.assert_called_once()
    store.append_record.assert_called_once()
    args = store.append_record.call_args.args
    assert args[0] == controller._quality_algo_config(candidate)
    assert "system" not in args[0]
    assert args[2] == {"answer_correctness": 0.75}
    assert args[3] is trace
    assert store.append_record.call_args.kwargs["full"] is True
    assert store.append_record.call_args.kwargs["source"]["eval_id"] == 0
    assert store.append_record.call_args.kwargs["quality_spec"] == (
        QualityStore.quality_spec(controller.config)
    )


def test_evaluator_preserves_raw_eval_ids_after_skipped_batch_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    controller = _controller(tmp_path)
    trace = _trace()
    _prepare_evaluation(controller, trace)
    controller._is_known_invalid = lambda cfg: bool(cfg.get("skip"))
    monkeypatch.setattr(
        "rag_stack.controller.build_workflow_schema",
        lambda _pipeline_config: object(),
    )

    result, valid, n_success, free_flags = controller._evaluate_objectives(
        [{"skip": True}, {"skip": False}],
        ["performance", "quality"],
        obj_columns=[],
    )

    assert valid == [{"skip": False}]
    assert result["performance"] == [19.0]
    assert n_success == 1
    assert free_flags == [False]
    assert controller._last_valid_eval_ids == [1]


def test_project_memo_export_preserves_complete_quality_spec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from rag_stack.quality_memo import QualityMemoEntry

    controller = _controller(tmp_path)
    controller.config["eval_backend_setting"]["submetric_query_subset"] = 15
    trace = _trace()
    _prepare_evaluation(controller, trace)
    memo_key = controller._quality_algo_key({})
    controller._quality_memo[memo_key] = QualityMemoEntry(
        quality_metrics={"answer_correctness": 0.75},
        execution_dag=trace,
        full=True,
    )
    store = SimpleNamespace(
        get_record=Mock(return_value=None),
        append_record=Mock(),
    )
    controller._external_quality_store = store
    monkeypatch.setattr(
        "rag_stack.controller.build_workflow_schema",
        lambda _pipeline_config: object(),
    )

    controller._evaluate_objectives(
        [{}], ["performance", "quality"], obj_columns=[],
    )

    controller.rag_quality_evaluator.evaluate.assert_not_called()
    store.append_record.assert_called_once()
    assert store.append_record.call_args.kwargs["source"]["kind"] == (
        "project_quality_memo"
    )
    assert store.append_record.call_args.kwargs["quality_spec"] == (
        QualityStore.quality_spec(controller.config)
    )
