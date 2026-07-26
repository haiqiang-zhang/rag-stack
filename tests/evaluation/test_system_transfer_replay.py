"""CPU-only contracts for trace-driven system-transfer replay.

The fixtures deliberately bypass :class:`Controller` construction and mock the
cost model/evaluator.  They pin orchestration semantics without loading a model,
touching CUDA, or running a benchmark sweep.
"""

from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_stack.controller import (
    Controller,
    _TRANSFER_ACTION_RECOMPUTED,
    _TRANSFER_ACTION_REUSED,
    _canonical_payload_sha256,
    _file_sha256,
    _portable_assembly_result,
)
from rag_stack.cost_model.types import AssemblyResult
from rag_stack.project_summary import CONFIG_FILENAME
from rag_stack.quality_store import QualityStore
from rag_stack.types import EvalType


def _bare_controller() -> Controller:
    controller = Controller.__new__(Controller)
    controller._eval_counter = 0
    controller._last_valid_eval_ids = []
    return controller


def test_stored_replay_uses_only_dynamic_cm_and_transplants_target_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    target_system = {
        "system_hardware": "sgs_a100x8_epyc7742",
        "gpu": "A100_80GB_GPU",
        "rag_cm": {"calibration_mode": "uncalibrated_spec"},
        "cm_search_space": {"min_num_gpus": 1, "max_num_gpus": 8},
        "retrieval": {
            "faiss_num_threads": None,
            "faiss_ivf_parallel_mode": 0,
        },
    }
    controller.config = {
        "system": deepcopy(target_system),
        "algo_search_space": {
            "pipeline": {
                "node_lines": [{
                    "nodes": [{
                        "stage": "generator",
                        "modules": [{"component": "vllm"}],
                    }],
                }],
            },
        },
    }

    source_pipeline = {
        "pipeline_runtime": {"mode": "react"},
        "corpus_runtime": {
            "chunker": {"component": "character", "chunk_size": 2048},
        },
        "vectordb": [{"name": "fixture-index", "db_type": "faiss_hnsw"}],
        "node_lines": [{
            "nodes": [{
                "stage": "generator",
                "modules": [{
                    "component": "vllm_api",
                    "uri": "http://old-h100-host:8000",
                    "model": "fixture-generator",
                }],
            }],
        }],
        "system": {
            "system_hardware": "sgs_h100x4",
            "gpu": "H100_GPU",
            "rag_cm": {"calibration_mode": "strict"},
        },
    }
    source_snapshot = deepcopy(source_pipeline)
    canonical_algo = QualityStore.algorithm_config({
        "pipeline": {
            "mode": "react",
            "react": {
                "generator": {
                    "component": "vllm",
                    "model": "fixture-generator",
                },
            },
        },
    })
    algo_hash = QualityStore.key_for_config(canonical_algo)
    controller._resolve_config = Mock(side_effect=AssertionError(
        "Phase 2a must not reinterpret a committed source pipeline",
    ))
    controller._ctx = SimpleNamespace(
        _resolve_system_retrieval=lambda container, _decoded: container[
            "system"
        ]["retrieval"].update({"faiss_num_threads": 128}),
    )
    corpus_view = SimpleNamespace(n_vectors=8_841_823)

    def _apply_corpus_view(pipeline: dict, view) -> None:
        pipeline["vectordb"][0]["N"] = view.n_vectors
        pipeline["_token_stats"] = {"avg_chunk_tokens": 81}

    controller._data = SimpleNamespace(
        resolve_corpus=Mock(return_value=corpus_view),
        apply_corpus_view=Mock(side_effect=_apply_corpus_view),
    )
    trace = {
        "queries": [{
            "question_id": "q0",
            "calls": [{
                "stage": "generator",
                "input_tokens": 12,
                "output_tokens": 3,
                "step_idx": 0,
                "model_id": "fixture-generator",
            }],
        }],
    }
    store = SimpleNamespace(
        get_record=Mock(return_value={"config": canonical_algo}),
        get_pipeline_config=Mock(return_value=source_pipeline),
        get_execution_dag=Mock(return_value=trace),
    )
    schema = object()
    build_schema = Mock(return_value=schema)
    monkeypatch.setattr(
        "rag_stack.controller.build_workflow_schema", build_schema,
    )
    result = SimpleNamespace(performance_score=42.0)
    # There is intentionally no evaluate_static attribute on this mock.  Any
    # static fallback therefore makes the test fail immediately.
    assembly = SimpleNamespace(evaluate_dynamic=Mock(return_value=result))

    pipeline, returned_trace, returned_result = (
        controller._price_stored_quality_record(
            quality_store=store,
            config_hash_value=algo_hash,
            assembly=assembly,
        )
    )

    assert pipeline["system"] is not controller.config["system"]
    assert pipeline["system"]["retrieval"]["faiss_num_threads"] == 128
    for key in ("system_hardware", "gpu", "rag_cm", "cm_search_space"):
        assert pipeline["system"][key] == target_system[key]
    replay_generator = pipeline["node_lines"][0]["nodes"][0]["modules"][0]
    assert replay_generator["component"] == "vllm"
    assert "uri" not in replay_generator
    assert replay_generator["use_chat_template"] is True
    assert source_pipeline == source_snapshot
    store.get_record.assert_called_once_with(algo_hash)
    store.get_pipeline_config.assert_called_once_with(algo_hash)
    controller._resolve_config.assert_not_called()
    controller._data.resolve_corpus.assert_called_once_with({
        "component": "character",
        "chunk_size": 2048,
    })
    controller._data.apply_corpus_view.assert_called_once()
    assert pipeline["vectordb"][0]["N"] == 8_841_823
    assert pipeline["_token_stats"] == {"avg_chunk_tokens": 81}
    assert returned_trace is trace
    assert returned_result is result
    build_schema.assert_called_once_with(pipeline)
    assembly.evaluate_dynamic.assert_called_once_with(
        pipeline,
        workflow_schema=schema,
        trace=trace,
    )


def test_stored_replay_uses_committed_pipeline_instead_of_reinterpreting_algo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Historical observations are not constrained by the new sampling space.

    Canonical optimizer records are identities, not source-exact runtime
    snapshots: fixed module defaults and active-stage pruning can evolve.
    Phase 2a must price the checksum-committed pipeline on the target system;
    re-resolving or dropping it would corrupt the warm-start ledger.
    """
    controller = _bare_controller()
    target_system = {
        "system_hardware": "sgs_a100x8_epyc7742",
        "gpu": "A100_80GB_GPU",
        "rag_cm": {"calibration_mode": "uncalibrated_spec"},
        "cm_search_space": {"min_num_gpus": 1, "max_num_gpus": 8},
        "retrieval": {
            "faiss_num_threads": 128,
            "faiss_ivf_parallel_mode": 0,
        },
    }
    controller.config = {"system": deepcopy(target_system)}
    controller._transfer_target_system = deepcopy(target_system)

    canonical_algo = QualityStore.algorithm_config({
        "pipeline": {
            "mode": "sequential",
            "sequential": {
                "query_expansion": {
                    "enabled": "true",
                    "component": "multi_query_expansion",
                },
            },
        },
    })
    algo_hash = QualityStore.key_for_config(canonical_algo)
    source_pipeline = {
        "pipeline_runtime": {"mode": "sequential"},
        "corpus_runtime": {
            "chunker": {"component": "character", "chunk_size": 2048},
        },
        "vectordb": [{
            "name": "fixture-index",
            "db_type": "faiss_hnsw",
            "N": 123,
        }],
        "node_lines": [{
            "nodes": [{
                "stage": "query_expansion",
                "modules": [{
                    "component": "multi_query_expansion",
                    "generator_backend": "vllm_api",
                    "model": "fixture-generator",
                }],
            }],
        }],
        "system": {
            "system_hardware": "sgs_h100x4",
            "gpu": "H100_GPU",
            "rag_cm": {"calibration_mode": "strict"},
        },
        "_token_stats": {
            "avg_chunk_tokens": 81,
            "prompt_template_tokens": 17,
            "avg_output_tokens": 22,
            "avg_query_tokens": 7,
        },
    }
    source_snapshot = deepcopy(source_pipeline)
    controller._resolve_config = Mock(side_effect=AssertionError(
        "historical pipeline must not be re-resolved",
    ))
    corpus_view = SimpleNamespace(n_vectors=8_841_823)

    def _apply_corpus_view(pipeline: dict, view) -> None:
        pipeline["vectordb"][0]["N"] = view.n_vectors

    controller._data = SimpleNamespace(
        resolve_corpus=Mock(return_value=corpus_view),
        apply_corpus_view=Mock(side_effect=_apply_corpus_view),
    )
    trace = {
        "queries": [{
            "question_id": "q0",
            "calls": [{
                "stage": "query_expansion",
                "input_tokens": 12,
                "output_tokens": 9,
                "step_idx": 0,
                "model_id": "fixture-generator",
            }],
        }],
    }
    store = SimpleNamespace(
        get_record=Mock(return_value={"config": canonical_algo}),
        get_pipeline_config=Mock(return_value=source_pipeline),
        get_execution_dag=Mock(return_value=trace),
    )
    schema = object()
    monkeypatch.setattr(
        "rag_stack.controller.build_workflow_schema",
        Mock(return_value=schema),
    )
    result = SimpleNamespace(performance_score=42.0)
    assembly = SimpleNamespace(evaluate_dynamic=Mock(return_value=result))

    pipeline, returned_trace, returned_result = (
        controller._price_stored_quality_record(
            quality_store=store,
            config_hash_value=algo_hash,
            assembly=assembly,
        )
    )

    controller._resolve_config.assert_not_called()
    store.get_pipeline_config.assert_called_once_with(algo_hash)
    assert source_pipeline == source_snapshot
    assert pipeline is not source_pipeline
    assert pipeline["system"] == target_system
    assert pipeline["node_lines"][0]["nodes"][0]["modules"][0][
        "component"
    ] == "multi_query_expansion"
    assert pipeline["vectordb"][0]["N"] == 8_841_823
    assert returned_trace is trace
    assert returned_result is result
    assembly.evaluate_dynamic.assert_called_once_with(
        pipeline,
        workflow_schema=schema,
        trace=trace,
    )


def test_stored_replay_does_not_skip_unresolvable_record_without_pipeline() -> None:
    controller = _bare_controller()
    controller.config = {"system": {"gpu": "A100_80GB_GPU"}}
    canonical_algo = QualityStore.algorithm_config({
        "pipeline": {"mode": "sequential"},
    })
    algo_hash = QualityStore.key_for_config(canonical_algo)
    controller._resolve_config = Mock(side_effect=AssertionError(
        "target resolver must not be used for historical replay",
    ))
    store = SimpleNamespace(
        get_record=Mock(return_value={"config": canonical_algo}),
        get_pipeline_config=Mock(side_effect=FileNotFoundError("missing")),
    )

    with pytest.raises(Exception, match="missing|outside target search space"):
        controller._price_stored_quality_record(
            quality_store=store,
            config_hash_value=algo_hash,
            assembly=SimpleNamespace(),
        )

    store.get_pipeline_config.assert_called_once_with(algo_hash)
    controller._resolve_config.assert_not_called()


@pytest.mark.parametrize("performance", [float("nan"), float("inf"), float("-inf")])
def test_stored_replay_rejects_nonfinite_performance(
    monkeypatch: pytest.MonkeyPatch,
    performance: float,
) -> None:
    controller = _bare_controller()
    controller.config = {"system": {"gpu": "A100_80GB_GPU"}}
    canonical_algo = QualityStore.algorithm_config({
        "pipeline": {"mode": "react"},
    })
    algo_hash = QualityStore.key_for_config(canonical_algo)
    committed_pipeline = {
        "pipeline_runtime": {"mode": "react"},
        "system": {"gpu": "A100_80GB_GPU"},
    }
    controller._resolve_config = Mock(side_effect=AssertionError(
        "target resolver must not be used for historical replay",
    ))
    controller._data = SimpleNamespace(
        resolve_corpus=Mock(return_value=SimpleNamespace()),
        apply_corpus_view=Mock(),
    )
    trace = {
        "queries": [{"question_id": "q0", "calls": []}],
    }
    store = SimpleNamespace(
        get_record=Mock(return_value={"config": canonical_algo}),
        get_pipeline_config=Mock(return_value=committed_pipeline),
        get_execution_dag=Mock(return_value=trace),
    )
    monkeypatch.setattr(
        "rag_stack.controller.build_workflow_schema", lambda _pipeline: object(),
    )
    assembly = SimpleNamespace(evaluate_dynamic=Mock(return_value=SimpleNamespace(
        performance_score=performance,
    )))

    with pytest.raises(ValueError, match="non-finite performance"):
        controller._price_stored_quality_record(
            quality_store=store,
            config_hash_value=algo_hash,
            assembly=assembly,
        )


def test_portable_assembly_result_drops_only_sweep_sized_payloads() -> None:
    best = {"qps": 42.0, "pipeline_stage_order": ["generator_decode"]}
    full = AssemblyResult(
        performance_score=42.0,
        best_hw_params=best,
        rago_result_df=object(),
        sweep_dfs={"generator_decode": object()},
        swept_configs=[{"placement_policy": "collocated_pd"}],
    )

    retained = _portable_assembly_result(full)

    assert retained is not full
    assert retained.performance_score == 42.0
    assert retained.best_hw_params == best
    assert retained.best_hw_params is not best
    assert retained.rago_result_df is None
    assert retained.sweep_dfs is None
    assert retained.swept_configs is None


def test_partial_optimizer_warm_start_fails_closed() -> None:
    controller = _bare_controller()
    controller._quality_algo_key = lambda cfg: cfg["id"]
    replay = [
        {"config": {"id": "first"}},
        {"config": {"id": "second"}},
        {"config": {"id": "third"}},
    ]
    optimizer = SimpleNamespace(_configs=[])

    def restore_only_a_prefix(evaluations: list[dict]) -> None:
        optimizer._configs.extend(
            item["config"] for item in evaluations[:-1]
        )

    optimizer.warm_start = Mock(side_effect=restore_only_a_prefix)
    controller.optimizer = optimizer

    optimizer.warm_start(replay)
    with pytest.raises(RuntimeError, match=r"partial: restored 2/3"):
        controller._assert_exact_transfer_warm_start(replay)

    optimizer.warm_start.assert_called_once_with(replay)


def test_transfer_optimizer_payload_uses_target_submetrics_only() -> None:
    controller = _bare_controller()
    controller.config = {
        "eval_backend_setting": {
            "combined_quality": {
                "mode": "mean",
                "metrics": ["deepeval_answer_correctness"],
            },
            "metrics": [
                {"metric_name": "deepeval_answer_correctness"},
                {"metric_name": "deepeval_faithfulness"},
                {"metric_name": "deepeval_context_precision"},
                {"metric_name": "deepeval_context_recall"},
            ],
        },
    }
    source_metrics = {
        "deepeval_answer_correctness": 0.7,
        "deepeval_context_precision": 0.8,
        "deepeval_context_recall": 0.6,
        "retrieval_token_precision": 0.2,
        "retrieval_token_recall": 0.3,
    }

    payload = controller._transfer_optimizer_quality_payload(
        combined=0.7,
        quality_metrics=source_metrics,
    )

    assert payload == {
        "combined": 0.7,
        "deepeval_answer_correctness": 0.7,
        "deepeval_context_precision": 0.8,
        "deepeval_context_recall": 0.6,
    }
    assert "deepeval_faithfulness" not in payload
    assert source_metrics["retrieval_token_precision"] == 0.2


def test_transfer_optimizer_payload_without_metric_config_keeps_source_metrics() -> None:
    controller = _bare_controller()
    controller.config = {}

    payload = controller._transfer_optimizer_quality_payload(
        combined=0.4,
        quality_metrics={"legacy_diagnostic": 0.9, "missing": None},
    )

    assert payload == {"combined": 0.4, "legacy_diagnostic": 0.9}


def test_transfer_provenance_uses_current_structural_contract(tmp_path) -> None:
    controller = _bare_controller()
    controller.project_dir = str(tmp_path)

    controller._save_transfer_provenance(
        replay_idx=0,
        config_hash_value="algo-hash",
        source_record={
            "source": {"eval_id": 7},
            "source_eval_type": "OOM",
            "source_eval_error": "CM: source capacity proof",
        },
        target_eval_type=EvalType.GT_FP,
        source_record_marker_sha256="a" * 64,
    )

    payload = json.loads(
        (tmp_path / "evaluations/eval_0000/transfer_provenance.json").read_text()
    )
    assert payload["kind"] == "system_transfer_replay"
    assert payload["cost_model_version"] == 1


def test_refinement_has_no_extra_init_and_lands_exactly_requested_iterations() -> None:
    controller = _bare_controller()
    requested = 4
    suggestions = [
        ([{"candidate": index}], ["performance", "quality"])
        for index in range(requested)
    ]
    optimizer = SimpleNamespace(
        objective_costs={"performance": 1.0, "quality": 10.0},
        # A large init count makes an accidental n_both_init pre-batch obvious.
        n_both_init=99,
        suggest=Mock(side_effect=suggestions),
        add_objectives=Mock(),
    )
    controller.optimizer = optimizer
    controller._eval_counter = 100
    controller._assembly_results = []
    controller._gt_count = 0
    controller._cm_count = 0
    controller._total_cost = 0.0
    controller._push_perf_attribution = Mock()
    controller._update_summary = Mock()
    controller._quality_algo_key = lambda cfg: f"hash-{cfg['candidate']}"
    controller._external_quality_store = SimpleNamespace(
        get_record=Mock(return_value={"committed": True}),
        get_execution_dag=Mock(return_value={"queries": []}),
    )

    def evaluate_one(
        candidates: list[dict], objectives: list[str], _columns: list[str],
    ):
        assert len(candidates) == 1
        assert objectives == ["performance", "quality"]
        index = candidates[0]["candidate"]
        eval_id = controller._eval_counter
        controller._eval_counter += 1
        controller._last_valid_eval_ids = [eval_id]
        controller._assembly_results.append(
            SimpleNamespace(performance_score=100.0 + index),
        )
        return (
            {
                "performance": [100.0 + index],
                "quality": [{"combined": 0.5 + index / 100.0}],
            },
            candidates,
            1,
            [False],
        )

    controller._evaluate_objectives = Mock(side_effect=evaluate_one)
    configs: list[dict] = []
    quality_scores: list[float] = []
    performance_scores: list[float] = []
    config_hashes: list[str] = []
    target_eval_ids: list[int] = []

    gt_count = controller._run_refinement(
        assembly=SimpleNamespace(),
        quality_store=SimpleNamespace(),
        n_iterations=requested,
        configs=configs,
        quality_scores=quality_scores,
        performance_scores=performance_scores,
        config_hashes=config_hashes,
        target_eval_ids=target_eval_ids,
        assembly_results=controller._assembly_results,
        obj_columns=[],
    )

    assert gt_count == requested
    assert optimizer.suggest.call_count == requested
    assert all(
        invocation.kwargs == {"n_suggestions": 1}
        for invocation in optimizer.suggest.call_args_list
    )
    assert optimizer.add_objectives.call_count == requested
    assert len(configs) == requested
    assert quality_scores == [0.5, 0.51, 0.52, 0.53]
    assert performance_scores == [100.0, 101.0, 102.0, 103.0]
    assert config_hashes == [f"hash-{index}" for index in range(requested)]
    assert target_eval_ids == list(range(100, 100 + requested))
    assert controller._eval_counter == 100 + requested
    assert controller._gt_count == requested
    assert controller._cm_count == requested
    assert controller._total_cost == requested * 11.0


def test_selective_refinement_profile_change_aborts_before_optimizer_commit(
    tmp_path,
) -> None:
    profile = tmp_path / "A100_80GB_GPU.json"
    profile.write_text('{"profile_kind": "compressor_llmlingua2"}\n')
    dependency = {
        "logical_path": (
            "calibration_profiles/compressor_llmlingua2/"
            "A100_80GB_GPU.json"
        ),
        "resolved_path": str(profile.resolve()),
        "sha256": _file_sha256(str(profile)),
    }
    controller = _bare_controller()
    controller.optimizer = SimpleNamespace(
        objective_costs={"performance": 1.0, "quality": 10.0},
        suggest=Mock(return_value=(
            [{"candidate": "mixed-profile"}],
            ["performance", "quality"],
        )),
        add_objectives=Mock(),
    )
    controller._eval_counter = 0
    controller._assembly_results = []

    def replace_profile_during_evaluation(*_args):
        profile.write_text('{"profile_kind": "changed"}\n')
        return (
            {
                "performance": [9.0],
                "quality": [{"combined": 0.75}],
            },
            [{"candidate": "mixed-profile"}],
            1,
            [False],
        )

    controller._evaluate_objectives = Mock(
        side_effect=replace_profile_during_evaluation,
    )

    with pytest.raises(RuntimeError, match="dependency changed"):
        controller._run_refinement(
            assembly=SimpleNamespace(),
            quality_store=SimpleNamespace(),
            n_iterations=1,
            configs=[],
            quality_scores=[],
            performance_scores=[],
            config_hashes=[],
            target_eval_ids=[],
            assembly_results=controller._assembly_results,
            obj_columns=[],
            transfer_state={
                "replay_plan": {"llmlingua2_profile": dependency},
            },
        )

    assert controller._evaluate_objectives.call_count == 1
    assert controller.optimizer.add_objectives.call_count == 0


def test_persisted_perf_attribution_registers_stage_card(
    tmp_path,
) -> None:
    controller = _bare_controller()
    controller.project_dir = str(tmp_path)
    eval_dir = tmp_path / "evaluations/eval_0051"
    eval_dir.mkdir(parents=True)
    (eval_dir / "assembly.json").write_text(json.dumps({
        "performance_score": 12.0,
        "best_hw_params": {
            "dynamic_performance_model": "closed_loop_simulator",
            "simulator_status": "ok",
            "simulator_qps": 12.0,
            "simulator_latency_mean_s": 0.25,
            "simulator_engine_fill_mean": {"generator.unified.r0": 1.0},
            "simulator_stage_batch_mean": {"semantic_retrieval": 4.0},
            "simulator_transfer_latency_mean": {},
            "simulator_transfer_queue_wait_mean": {},
            "simulator_saturation_plateau": True,
            "simulator_saturation_stop_reason": "qps_plateau",
            "simulator_populations_tried": [32, 64],
            "load_concurrency": 64,
            "candidate_system_layout": {
                "engines": {"generator": {"pd_serving": "collocated_pd"}},
            },
            "qps": 12.0,
            "latency_s": 0.25,
            "qps_bound": ["generator_decode"],
            "placement_policy": "collocated_pd",
            "num_chips_e2e": 8,
            "batch_size_request": 4,
            "qps_generator_decode": 12.0,
            "qps_per_chip": 1.5,
        },
    }))
    register = Mock()
    optimizer = SimpleNamespace(register_perf_attribution=register)

    controller._restore_persisted_perf_attribution(optimizer)

    register.assert_called_once_with(51, {
        "source": "authoritative_des",
        "qps": 12.0,
        "latency_s": 0.25,
        "deployment": {
            "policy": "collocated_pd",
            "chips": 8,
            "batch_request": 4,
            "batch_decode": 0,
        },
        "saturation": {
            "plateau": True,
            "stop_reason": "qps_plateau",
            "population": 64,
            "populations_tried": [32, 64],
        },
        "engines": {
            "generator": {
                "pd_serving": "collocated_pd",
                "identifiability": "unified_only",
                "replicas": {
                    "generator.unified.r0": {"fill_fraction": 1.0},
                },
            },
        },
        "workers": {"semantic_retrieval": {"batch_mean": 4.0}},
        "transfers": {},
    })


def test_completed_transfer_result_is_atomic_portable_and_structural(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _bare_controller()
    controller.project_dir = str(tmp_path)
    real_replace = os.replace
    replace_spy = Mock(side_effect=real_replace)
    monkeypatch.setattr("rag_stack.controller.os.replace", replace_spy)

    controller._save_system_transfer_result({
        "pareto_configs": [{
            "pipeline": {"mode": "react"},
            "generator": {"api_key": "sk-" + "a" * 48},
        }],
        "pareto_objectives": [{"quality": 0.6, "performance": 42.0}],
        "pareto_eval_ids": [61],
        "pareto_algo_hashes": ["algo-hash"],
        "total_cost": 110.0,
        "cm_count": 62,
        "gt_count": 10,
        # The portable artifact deliberately does not serialize CM objects.
        "assembly_results": [object(), object()],
        "hardware": {"gpu": "A100_80GB_GPU"},
        "system_transfer": {
            "source_observations": 52,
            "repriced_observations": 52,
            "refinement_iterations_requested": 10,
            "refinement_iterations_completed": 10,
            "cost_model_version": 1,
        },
    })

    path = tmp_path / "system_transfer_result.json"
    payload = json.loads(path.read_text())
    assert payload["kind"] == "system_transfer_result"
    assert payload["status"] == "completed"
    assert payload["pareto_eval_ids"] == [61]
    assert payload["pareto_algo_hashes"] == ["algo-hash"]
    assert payload["assembly_results_count"] == 2
    assert "assembly_results" not in payload
    assert "api_key" not in payload["pareto_configs"][0]["generator"]
    assert not list(tmp_path.glob(".system_transfer_result.*.tmp"))

    replace_spy.assert_called_once()
    temp_path, final_path = replace_spy.call_args.args
    assert temp_path != str(path)
    assert final_path == str(path)


def test_inexact_transfer_result_is_not_persisted(tmp_path) -> None:
    controller = _bare_controller()
    controller.project_dir = str(tmp_path)

    with pytest.raises(RuntimeError, match=r"iterations 9/10"):
        controller._save_system_transfer_result({
            "pareto_configs": [],
            "pareto_objectives": [],
            "pareto_eval_ids": [],
            "pareto_algo_hashes": [],
            "assembly_results": [],
            "system_transfer": {
                "refinement_iterations_requested": 10,
                "refinement_iterations_completed": 9,
            },
        })

    assert not (tmp_path / "system_transfer_result.json").exists()


class _ResumeStore:
    def __init__(
        self,
        root,
        records: list[dict],
        traces: dict[str, dict],
        pipelines: dict[str, dict] | None = None,
    ):
        self.store_path = str(root)
        self._records = {record["config_hash"]: record for record in records}
        self._traces = traces
        self._pipelines = pipelines or {}
        (root / "records").mkdir(parents=True)
        for record in records:
            (root / "records" / f"{record['config_hash']}.json").write_text(
                json.dumps({"algo_hash": record["config_hash"]})
            )

    def get_record(self, record_hash: str):
        return self._records.get(record_hash)

    def get_execution_dag(self, record_hash: str):
        return deepcopy(self._traces[record_hash])

    def get_pipeline_config(self, record_hash: str):
        return deepcopy(self._pipelines[record_hash])

    def append_record(
        self, config, pipeline_config, quality_metrics, execution_dag, **kwargs,
    ):
        record_hash = config.get("hash")
        if not record_hash:
            raise AssertionError("fixture recovery config must carry hash")
        self._records[record_hash] = {
            "config_hash": record_hash,
            "config": deepcopy(config),
            "quality_metrics": deepcopy(quality_metrics),
            "quality_combined": float(quality_metrics["answer_correctness"]),
            "record": {"source": deepcopy(kwargs.get("source") or {})},
        }
        self._traces[record_hash] = deepcopy(execution_dag)
        marker = os.path.join(self.store_path, "records", f"{record_hash}.json")
        with open(marker, "w") as stream:
            json.dump({"algo_hash": record_hash}, stream)
        return record_hash, True


def _resume_controller(
    tmp_path, target_system: dict, project_name: str = "target",
) -> Controller:
    controller = _bare_controller()
    controller.project_dir = str(tmp_path / project_name)
    os.makedirs(controller.project_dir)
    controller.config = {"system": deepcopy(target_system), "global": {"seed": 45}}
    controller._transfer_target_system = deepcopy(target_system)
    controller.optimizer = SimpleNamespace(
        _configs=[], objective_costs={"performance": 1.0, "quality": 10.0},
    )
    controller._eval_counter = 0
    return controller


def _source_record(record_hash: str, eval_id: int, score: float) -> dict:
    return {
        "config_hash": record_hash,
        "config": {"pipeline": {"mode": record_hash}},
        "quality_metrics": {"answer_correctness": score},
        "quality_combined": score,
        "record": {
            "source_eval_type": "GT",
            "source": {"eval_id": eval_id, "eval_type": "GT"},
        },
    }


def _commit_completed_refinement_ledger(
    controller: Controller,
    *,
    store: _ResumeStore,
    state: dict,
    target_system: dict,
    trace: dict,
    count: int,
) -> None:
    """Write a complete CPU-only refinement ledger accepted by strict restore."""
    controller._quality_algo_key = lambda config: config["hash"]
    controller._quality_algo_config = lambda config: deepcopy(config)
    source_count = len(state["source_records"])
    csv_rows = [
        "cfg__hash,performance,quality__combined,eval_type,eval_error,eval_id\n"
    ]
    for refinement_index in range(count):
        algo_hash = f"refine-{refinement_index}"
        config = {"hash": algo_hash}
        quality_score = 0.6 + refinement_index / 100.0
        quality_metrics = {"answer_correctness": quality_score}
        performance = 20.0 + refinement_index
        eval_id = source_count + refinement_index
        store._records[algo_hash] = {
            "config_hash": algo_hash,
            "config": deepcopy(config),
            "quality_metrics": deepcopy(quality_metrics),
            "quality_combined": quality_score,
            "record": {"source": {"kind": "target-refinement"}},
        }
        store._traces[algo_hash] = deepcopy(trace)
        (Path(store.store_path) / "records" / f"{algo_hash}.json").write_text(
            json.dumps({"algo_hash": algo_hash})
        )
        eval_dir = (
            Path(controller.project_dir) / "evaluations" / f"eval_{eval_id:04d}"
        )
        eval_dir.mkdir(parents=True)
        (eval_dir / "pipeline_config.json").write_text(json.dumps({
            "system": target_system,
        }))
        (eval_dir / "quality_metrics.json").write_text(
            json.dumps(quality_metrics)
        )
        (eval_dir / "execution_dag.json").write_text(json.dumps(trace))
        (eval_dir / "assembly.json").write_text(json.dumps({
            "performance_score": performance,
            "best_hw_params": {"num_chips": 8},
        }))
        csv_rows.append(
            f"{algo_hash},{performance:.6f},{quality_score:.6f},GT,,{eval_id}\n"
        )
        controller._save_transfer_refinement_commit(
            state=state,
            refinement_index=refinement_index,
            eval_id=eval_id,
            config=config,
            performance=performance,
            quality={
                "combined": quality_score,
                "answer_correctness": quality_score,
            },
            algo_hash=algo_hash,
            eval_type="GT",
            free=False,
            gt_budget_count=1,
            quality_store=store,
        )
    (Path(controller.project_dir) / "all_evaluations.csv").write_text(
        "".join(csv_rows)
    )


def _selective_pipeline(target_system: dict, *, active: bool) -> dict:
    """Resolved graph carries active deps; search space always declares them."""
    nodes = [{
        "stage": "generator",
        "modules": [{"component": "vllm", "model": "fixture"}],
    }]
    if active:
        nodes.insert(0, {
            "stage": "passage_compressor",
            "modules": [{"component": "llmlingua2", "rate": 0.5}],
        })
    return {
        "system": deepcopy(target_system),
        "algo_search_space": {
            "pipeline": {"node_lines": [{"nodes": [{
                "stage": "passage_compressor",
                "modules": [{"component": "llmlingua2"}],
            }]}]},
        },
        "node_lines": [{"nodes": nodes}],
    }


def _commit_prior_replay(
    controller: Controller,
    *,
    state: dict,
    record: dict,
    trace: dict,
    pipeline: dict,
    replay_idx: int,
    performance: float,
) -> None:
    eval_dir = (
        Path(controller.project_dir) / "evaluations" / f"eval_{replay_idx:04d}"
    )
    eval_dir.mkdir(parents=True)
    (eval_dir / "pipeline_config.json").write_text(json.dumps(pipeline))
    (eval_dir / "quality_metrics.json").write_text(
        json.dumps(record["quality_metrics"])
    )
    (eval_dir / "execution_dag.json").write_text(json.dumps(trace))
    (eval_dir / "assembly.json").write_text(json.dumps({
        "performance_score": performance,
        "best_hw_params": {"qps": performance, "num_chips_e2e": 8},
    }))
    controller._save_transfer_provenance(
        replay_idx=replay_idx,
        config_hash_value=record["config_hash"],
        source_record=record["record"],
        target_eval_type=EvalType.GT_FP,
        source_record_marker_sha256=state["source_records"][replay_idx][
            "record_marker_sha256"
        ],
        target_config_sha256=state["target_config_sha256"],
    )


def test_selective_transfer_plan_is_exact_immutable_and_fail_closed(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_system = {
        "system_hardware": "a100x8",
        "gpu": "A100_80GB_GPU",
        "rag_cm": {"calibration_mode": "uncalibrated_spec"},
        "performance_source": "cost_model",
        "cm_search_space": {"min_num_gpus": 1, "max_num_gpus": 8},
    }
    active = [0, 4, 11, 12, 13, 15, 17, 21, 34, 44, 48, 49]
    records = [
        _source_record(f"source-{index:02d}", index, 0.2 + index / 1000)
        for index in range(52)
    ]
    trace = {"queries": [{"question_id": "q"}]}
    traces = {record["config_hash"]: trace for record in records}
    pipelines = {
        record["config_hash"]: _selective_pipeline(
            target_system, active=index in active,
        )
        for index, record in enumerate(records)
    }
    prior_store = _ResumeStore(
        tmp_path / "prior-store", records, traces, pipelines=pipelines,
    )
    fresh_store = _ResumeStore(
        tmp_path / "fresh-store", records, traces, pipelines=pipelines,
    )
    for record in records:
        marker = (
            tmp_path / "fresh-store/records"
            / f"{record['config_hash']}.json"
        )
        marker.write_text(json.dumps({
            "algo_hash": record["config_hash"],
            "updated_at": "fresh-import-transaction",
        }))
    prior = _resume_controller(tmp_path, target_system, "prior")
    target = _resume_controller(tmp_path, target_system, "target")
    for controller in (prior, target):
        (Path(controller.project_dir) / CONFIG_FILENAME).write_text(
            json.dumps(controller.config)
        )
    prior_state, _ = prior._prepare_system_transfer_state(
        quality_store=prior_store,
        quality_results=records,
        n_refine_iterations=10,
    )
    assert all(
        prior_state["source_records"][index]["record_marker_sha256"]
        != target._transfer_record_marker_sha256(
            fresh_store, record["config_hash"],
        )
        for index, record in enumerate(records)
    )

    reused = [index for index in range(52) if index not in active]
    csv_rows = [
        "performance,quality__combined,eval_type,eval_error,eval_id"
    ]
    for index in reused:
        performance = 10.0 + index
        _commit_prior_replay(
            prior,
            state=prior_state,
            record=records[index],
            trace=trace,
            pipeline=pipelines[records[index]["config_hash"]],
            replay_idx=index,
            performance=performance,
        )
        csv_rows.append(
            f"{performance:.6f},{records[index]['quality_combined']:.6f},"
            f"GT_FP,,{index}"
        )
    (tmp_path / "prior/all_evaluations.csv").write_text(
        "\n".join(csv_rows) + "\n"
    )

    profile = tmp_path / "A100_80GB_GPU.json"
    profile.write_text('{"profile_kind": "compressor_llmlingua2"}\n')
    profile_identity = {
        "logical_path": (
            "calibration_profiles/compressor_llmlingua2/"
            "A100_80GB_GPU.json"
        ),
        "resolved_path": str(profile.resolve()),
        "sha256": _file_sha256(str(profile)),
    }
    monkeypatch.setattr(
        target,
        "_llmlingua2_transfer_profile_identity",
        lambda: deepcopy(profile_identity),
    )

    plan = target._build_selective_transfer_replay_plan(
        quality_store=fresh_store,
        quality_results=records,
        reuse_target_project=prior.project_dir,
        retransfer_source_indices=active,
    )
    assert plan["retransfer_source_indices"] == active
    assert plan["llmlingua2_active_source_indices"] == active
    assert plan["reuse_source_indices"] == reused
    assert len(plan["reuse_lineage"]) == 40
    assert plan["llmlingua2_profile"] == profile_identity
    assert plan["plan_sha256"] == _canonical_payload_sha256({
        key: value for key, value in plan.items() if key != "plan_sha256"
    })

    with pytest.raises(RuntimeError, match="active LLMLingua2 set"):
        target._build_selective_transfer_replay_plan(
            quality_store=fresh_store,
            quality_results=records,
            reuse_target_project=prior.project_dir,
            retransfer_source_indices=active[:-1],
        )
    assert not (tmp_path / "target/system_transfer_state.json").exists()

    state, _ = target._prepare_system_transfer_state(
        quality_store=fresh_store,
        quality_results=records,
        n_refine_iterations=10,
        replay_plan=plan,
    )
    assert [entry["source_index"] for entry in state["source_records"]] == list(
        range(52)
    )
    altered_plan = deepcopy(plan)
    altered_plan["prior_target_project"] = str(tmp_path / "other-prior")
    altered_plan["plan_sha256"] = _canonical_payload_sha256({
        key: value
        for key, value in altered_plan.items()
        if key != "plan_sha256"
    })
    with pytest.raises(RuntimeError, match="resume identity mismatch"):
        target._prepare_system_transfer_state(
            quality_store=fresh_store,
            quality_results=records,
            n_refine_iterations=10,
            replay_plan=altered_plan,
        )

    tampered_assembly = (
        tmp_path / "prior/evaluations"
        / f"eval_{reused[0]:04d}" / "assembly.json"
    )
    tampered_assembly.write_text(json.dumps({
        "performance_score": -1.0, "best_hw_params": {},
    }))
    with pytest.raises(RuntimeError, match="artifacts changed"):
        target._build_selective_transfer_replay_plan(
            quality_store=fresh_store,
            quality_results=records,
            reuse_target_project=prior.project_dir,
            retransfer_source_indices=active,
        )

    profile.write_text('{"profile_kind": "changed"}\n')
    with pytest.raises(RuntimeError, match="dependency changed"):
        target._assert_transfer_dependency_unchanged(profile_identity)


def test_selective_transfer_provenance_has_one_unambiguous_origin(
    tmp_path,
) -> None:
    controller = _bare_controller()
    controller.project_dir = str(tmp_path)
    profile = {
        "logical_path": "calibration_profiles/compressor_llmlingua2/a.json",
        "resolved_path": "/profiles/a.json",
        "sha256": "a" * 64,
    }
    lineage = {
        "source_index": 1,
        "prior_target_project": "/prior",
        "prior_transfer_provenance_sha256": "b" * 64,
    }
    controller._save_transfer_provenance(
        replay_idx=0,
        config_hash_value="recomputed",
        source_record={"source": {"eval_id": 0}},
        target_eval_type=EvalType.GT_FP,
        replay_action=_TRANSFER_ACTION_RECOMPUTED,
        llmlingua2_profile=profile,
        source_record_marker_sha256="c" * 64,
    )
    controller._save_transfer_provenance(
        replay_idx=1,
        config_hash_value="reused",
        source_record={"source": {"eval_id": 1}},
        target_eval_type=EvalType.GT_FP,
        replay_action=_TRANSFER_ACTION_REUSED,
        reuse_lineage=lineage,
        source_record_marker_sha256="d" * 64,
    )
    recomputed = json.loads((
        tmp_path / "evaluations/eval_0000/transfer_provenance.json"
    ).read_text())
    reused_payload = json.loads((
        tmp_path / "evaluations/eval_0001/transfer_provenance.json"
    ).read_text())
    assert recomputed["performance_origin"] == _TRANSFER_ACTION_RECOMPUTED
    assert recomputed["llmlingua2_profile"] == profile
    assert "reuse_lineage" not in recomputed
    assert reused_payload["performance_origin"] == _TRANSFER_ACTION_REUSED
    assert reused_payload["reuse_lineage"] == lineage
    assert "llmlingua2_profile" not in reused_payload


def test_selective_transfer_mixed_ledger_is_contiguous_and_resumable(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_system = {
        "system_hardware": "a100x8",
        "gpu": "A100_80GB_GPU",
        "rag_cm": {"calibration_mode": "uncalibrated_spec"},
        "performance_source": "cost_model",
        "cm_search_space": {"min_num_gpus": 1, "max_num_gpus": 8},
    }
    records = [
        _source_record(f"mixed-{index}", index, 0.5 + index / 10)
        for index in range(3)
    ]
    for record in records:
        record["quality_metrics"]["retired_source_diagnostic"] = 0.25
    trace = {"queries": [{"question_id": "q"}]}
    traces = {record["config_hash"]: trace for record in records}
    pipelines = {
        record["config_hash"]: _selective_pipeline(
            target_system, active=index == 1,
        )
        for index, record in enumerate(records)
    }
    store = _ResumeStore(
        tmp_path / "store", records, traces, pipelines=pipelines,
    )
    prior = _resume_controller(tmp_path, target_system, "prior")
    target = _resume_controller(tmp_path, target_system, "target")
    for controller in (prior, target):
        controller.config["eval_backend_setting"] = {
            "combined_quality": {
                "mode": "mean", "metrics": ["answer_correctness"],
            },
            "metrics": [{"metric_name": "answer_correctness"}],
        }
        (Path(controller.project_dir) / CONFIG_FILENAME).write_text(
            json.dumps(controller.config)
        )
    prior_state, _ = prior._prepare_system_transfer_state(
        quality_store=store, quality_results=records, n_refine_iterations=10,
    )
    for index, performance in ((0, 11.0), (2, 13.0)):
        _commit_prior_replay(
            prior,
            state=prior_state,
            record=records[index],
            trace=trace,
            pipeline=pipelines[records[index]["config_hash"]],
            replay_idx=index,
            performance=performance,
        )
    (tmp_path / "prior/all_evaluations.csv").write_text(
        "performance,quality__combined,eval_type,eval_error,eval_id\n"
        "11.000000,0.500000,GT_FP,,0\n"
        "13.000000,0.700000,GT_FP,,2\n"
    )
    profile = tmp_path / "A100_80GB_GPU.json"
    profile.write_text('{"profile_kind": "compressor_llmlingua2"}\n')
    profile_identity = {
        "logical_path": (
            "calibration_profiles/compressor_llmlingua2/"
            "A100_80GB_GPU.json"
        ),
        "resolved_path": str(profile.resolve()),
        "sha256": _file_sha256(str(profile)),
    }
    monkeypatch.setattr(
        target,
        "_llmlingua2_transfer_profile_identity",
        lambda: deepcopy(profile_identity),
    )
    plan = target._build_selective_transfer_replay_plan(
        quality_store=store,
        quality_results=records,
        reuse_target_project=prior.project_dir,
        retransfer_source_indices=[1],
    )
    # Reproduce a plan written before refinement budgets became mutable.  Those
    # plans included the original budget in their prior-lineage hash and must
    # remain resumable after both the code upgrade and a prior 10 -> 20 extension.
    prior_identity, canonical_identity_sha = (
        target._prior_transfer_state_identity(prior_state)
    )
    assert plan["prior_system_transfer_identity_sha256"] == canonical_identity_sha
    legacy_identity = deepcopy(prior_identity)
    legacy_identity["refinement_iterations_requested"] = 10
    legacy_identity_sha = _canonical_payload_sha256(legacy_identity)
    plan["prior_system_transfer_identity_sha256"] = legacy_identity_sha
    for lineage in plan["reuse_lineage"]:
        lineage["prior_system_transfer_identity_sha256"] = legacy_identity_sha
    plan["plan_sha256"] = _canonical_payload_sha256({
        key: value for key, value in plan.items() if key != "plan_sha256"
    })
    state, records = target._prepare_system_transfer_state(
        quality_store=store,
        quality_results=records,
        n_refine_iterations=10,
        replay_plan=plan,
    )

    # Runtime completion fields may advance while this child replay is live.
    # They are not part of the immutable prior lineage identity.
    prior_state_path = tmp_path / "prior/system_transfer_state.json"
    persisted_prior_state = json.loads(prior_state_path.read_text())
    _, identity_before = target._prior_transfer_state_identity(
        persisted_prior_state,
    )
    persisted_prior_state.update({
        "status": "completed",
        "refinement_iterations_completed": 10,
        "result_sha256": "c" * 64,
    })
    prior_state_path.write_text(json.dumps(persisted_prior_state))
    _, identity_after = target._prior_transfer_state_identity(
        persisted_prior_state,
    )
    assert identity_after == identity_before
    assert legacy_identity_sha in (
        target._compatible_prior_transfer_identity_sha256s(
            persisted_prior_state,
        )
    )

    # The cumulative refinement budget is mutable too: extending the prior
    # target from 10 to 20 must not invalidate a child's frozen Phase-2a reuse
    # lineage.  The actual replay loads below exercise the child-resume check.
    persisted_prior_state.update({
        "status": "running",
        "refinement_iterations_requested": 20,
        "refinement_budget_extensions": [{
            "from": 10,
            "to": 20,
            "prior_status": "completed",
            "prior_completed_iterations": 10,
            "prior_result_sha256": "mutable-runtime-result",
        }],
    })
    persisted_prior_state.pop("refinement_iterations_completed")
    persisted_prior_state.pop("result_sha256")
    prior_state_path.write_text(json.dumps(persisted_prior_state))
    _, identity_after_extension = target._prior_transfer_state_identity(
        persisted_prior_state,
    )
    assert identity_after_extension == identity_before
    assert legacy_identity_sha in (
        target._compatible_prior_transfer_identity_sha256s(
            persisted_prior_state,
        )
    )

    target._ssops = SimpleNamespace(
        enrich_configs=lambda configs, _base: deepcopy(configs),
    )
    target._base_decoded_config = {}
    target._failure_counts = {}
    target._strict_eval_persistence = True
    target._ctx = SimpleNamespace()
    selected_result = AssemblyResult(
        performance_score=99.0,
        best_hw_params={"qps": 99.0, "num_chips_e2e": 8},
    )
    price = Mock(return_value=(
        pipelines[records[1]["config_hash"]], trace, selected_result,
    ))
    target._price_stored_quality_record = price
    obj_columns = [
        "performance", "quality__combined", "quality__answer_correctness",
    ]
    for replay_idx, record in enumerate(records):
        action, dependency = target._transfer_replay_action(state, replay_idx)
        if action == _TRANSFER_ACTION_REUSED:
            (
                pipeline, replay_trace, result, target_type, _,
                performance,
            ) = target._load_validated_prior_target_replay(
                replay_idx=replay_idx,
                quality_store=store,
                quality_record=record,
                replay_plan=plan,
            )
            lineage = target._transfer_reuse_lineage(plan, replay_idx)
        else:
            target._assert_transfer_dependency_unchanged(dependency)
            pipeline, replay_trace, result = target._price_stored_quality_record(
                quality_store=store,
                config_hash_value=record["config_hash"],
                assembly=SimpleNamespace(),
            )
            target_type = EvalType.GT_FP
            performance = float(result.performance_score)
            lineage = None
        quality_payload = {
            "combined": record["quality_combined"],
            **record["quality_metrics"],
        }
        target._save_eval_record(
            eval_id=replay_idx,
            config=record["config"],
            pipeline_config=pipeline,
            eval_type=target_type,
            objectives={
                "performance": performance, "quality": quality_payload,
            },
            quality_metrics=record["quality_metrics"],
            assembly_result=result,
            obj_columns=obj_columns,
            execution_dag=replay_trace,
        )
        target._save_transfer_provenance(
            replay_idx=replay_idx,
            config_hash_value=record["config_hash"],
            source_record=record["record"],
            target_eval_type=target_type,
            source_record_marker_sha256=state["source_records"][replay_idx][
                "record_marker_sha256"
            ],
            target_config_sha256=state["target_config_sha256"],
            replay_action=action,
            llmlingua2_profile=dependency,
            reuse_lineage=lineage,
        )

    assert price.call_count == 1
    assert [int(row["eval_id"]) for row in target._transfer_csv_rows().values()] == [
        0, 1, 2,
    ]
    eval_dirs = sorted((tmp_path / "target/evaluations").glob("eval_*"))
    assert [path.name for path in eval_dirs] == [
        "eval_0000", "eval_0001", "eval_0002",
    ]
    assert not any((path / "rago_sweep.csv").exists() for path in eval_dirs)
    assert not any((path / "sweep_dfs").exists() for path in eval_dirs)

    # Crash window: artifacts+CSV are durable but the first reuse marker is not.
    first_marker = eval_dirs[0] / "transfer_provenance.json"
    first_marker.unlink()
    restored, assemblies = target._restore_transfer_replays(
        state=state, quality_store=store, quality_results=records,
    )
    assert len(restored) == 3
    assert len(assemblies) == 3
    assert all(
        "retired_source_diagnostic"
        not in observation["objectives"]["quality"]
        for observation in restored
    )
    assert json.loads(
        (eval_dirs[0] / "quality_metrics.json").read_text()
    )["retired_source_diagnostic"] == 0.25
    assert price.call_count == 1
    recovered = json.loads(first_marker.read_text())
    assert recovered["performance_origin"] == _TRANSFER_ACTION_REUSED
    assert recovered["reuse_lineage"] == plan["reuse_lineage"][0]
    selected_marker = json.loads(
        (eval_dirs[1] / "transfer_provenance.json").read_text()
    )
    assert selected_marker["performance_origin"] == _TRANSFER_ACTION_RECOMPUTED
    assert selected_marker["llmlingua2_profile"] == profile_identity
    assert "reuse_lineage" not in selected_marker


def test_transfer_state_freezes_source_order_and_reuses_a_grown_store(
    tmp_path,
) -> None:
    target_system = {
        "system_hardware": "a100x8", "rag_cm": {"calibration_mode": "uncalibrated_spec"},
    }
    controller = _resume_controller(tmp_path, target_system)
    trace = {"queries": [{"question_id": "q"}]}
    initial = [_source_record("source-a", 0, 0.5), _source_record("source-b", 1, 0.6)]
    store = _ResumeStore(
        tmp_path / "store", initial,
        {"source-a": trace, "source-b": trace},
    )

    state, selected = controller._prepare_system_transfer_state(
        quality_store=store, quality_results=initial, n_refine_iterations=10,
    )
    assert [item["config_hash"] for item in selected] == ["source-a", "source-b"]
    assert state["kind"] == "system_transfer_state"
    assert state["cost_model_version"] == 1

    refinement = _source_record("new-refinement-cache", 99, 0.7)
    store._records["new-refinement-cache"] = refinement
    store._traces["new-refinement-cache"] = trace
    (tmp_path / "store/records/new-refinement-cache.json").write_text(
        '{"algo_hash": "new-refinement-cache"}'
    )
    _, selected_after_growth = controller._prepare_system_transfer_state(
        quality_store=store,
        quality_results=initial + [refinement],
        n_refine_iterations=10,
    )
    assert [item["config_hash"] for item in selected_after_growth] == [
        "source-a", "source-b",
    ]

    (tmp_path / "store/records/source-a.json").write_text(
        '{"algo_hash": "source-a", "changed": true}'
    )
    with pytest.raises(RuntimeError, match="frozen source record changed"):
        controller._prepare_system_transfer_state(
            quality_store=store,
            quality_results=initial + [refinement],
            n_refine_iterations=10,
        )


def test_transfer_state_identity_ignores_runtime_config_enrichment(tmp_path) -> None:
    target_system = {
        "system_hardware": "a100x8",
        "rag_cm": {"calibration_mode": "uncalibrated_spec"},
    }
    controller = _resume_controller(tmp_path, target_system)
    persisted = {
        "global": {"rag_ir_mode": "dynamic"},
        "system": deepcopy(target_system),
        "algo_search_space": {"vectordb": []},
    }
    (tmp_path / "target" / CONFIG_FILENAME).write_text(json.dumps(persisted))
    records = [_source_record("source-a", 0, 0.5)]
    trace = {"queries": [{"question_id": "q"}]}
    store = _ResumeStore(tmp_path / "store", records, {"source-a": trace})

    state, _ = controller._prepare_system_transfer_state(
        quality_store=store, quality_results=records, n_refine_iterations=10,
    )
    controller.config["_token_stats"] = {"avg_chunk_tokens": 999}
    controller.config.setdefault("algo_search_space", {})["vectordb"] = [
        {"N": 8_841_823},
    ]

    resumed, _ = controller._prepare_system_transfer_state(
        quality_store=store, quality_results=records, n_refine_iterations=10,
    )
    assert resumed["target_config_sha256"] == state["target_config_sha256"]
    assert resumed["target_system_sha256"] == state["target_system_sha256"]

    controller._save_transfer_provenance(
        replay_idx=0,
        config_hash_value="source-a",
        source_record=records[0]["record"],
        target_eval_type=EvalType.GT_FP,
        source_record_marker_sha256=state["source_records"][0][
            "record_marker_sha256"
        ],
    )
    marker = json.loads(
        (tmp_path / "target/evaluations/eval_0000/transfer_provenance.json")
        .read_text()
    )
    assert marker["target_config_sha256"] == state["target_config_sha256"]


def test_replay_resume_migrates_only_the_exact_legacy_mutable_identity(
    tmp_path,
) -> None:
    target_system = {
        "system_hardware": "a100x8",
        "rag_cm": {"calibration_mode": "uncalibrated_spec"},
    }
    controller = _resume_controller(tmp_path, target_system)
    persisted = {
        "global": {"rag_ir_mode": "dynamic"},
        "system": deepcopy(target_system),
    }
    (tmp_path / "target" / CONFIG_FILENAME).write_text(json.dumps(persisted))
    controller.config["_token_stats"] = {"avg_chunk_tokens": 999}
    legacy_mutable_sha = _canonical_payload_sha256(controller.config)
    records = [_source_record("source-a", 0, 0.5)]
    trace = {"queries": [{"question_id": "q"}]}
    store = _ResumeStore(tmp_path / "store", records, {"source-a": trace})
    state, records = controller._prepare_system_transfer_state(
        quality_store=store, quality_results=records, n_refine_iterations=10,
    )
    assert legacy_mutable_sha != state["target_config_sha256"]

    eval_dir = tmp_path / "target/evaluations/eval_0000"
    eval_dir.mkdir(parents=True)
    (eval_dir / "pipeline_config.json").write_text(
        json.dumps({"system": target_system})
    )
    (eval_dir / "quality_metrics.json").write_text(
        json.dumps(records[0]["quality_metrics"])
    )
    (eval_dir / "execution_dag.json").write_text(json.dumps(trace))
    (eval_dir / "assembly.json").write_text(json.dumps({
        "performance_score": 12.0,
        "best_hw_params": {"num_chips": 8},
    }))
    (tmp_path / "target/all_evaluations.csv").write_text(
        "performance,quality__combined,eval_type,eval_error,eval_id\n"
        "12.000000,0.500000,GT_FP,,0\n"
    )
    controller._save_transfer_provenance(
        replay_idx=0,
        config_hash_value="source-a",
        source_record=records[0]["record"],
        target_eval_type=EvalType.GT_FP,
        source_record_marker_sha256=state["source_records"][0][
            "record_marker_sha256"
        ],
        # Reproduce the pre-fix writer exactly.
        target_config_sha256=legacy_mutable_sha,
    )
    artifact_hashes = {
        path.name: _file_sha256(str(path))
        for path in eval_dir.iterdir()
        if path.name != "transfer_provenance.json"
    }

    restored, _ = controller._restore_transfer_replays(
        state=state, quality_store=store, quality_results=records,
    )
    assert len(restored) == 1
    migrated = json.loads(
        (eval_dir / "transfer_provenance.json").read_text()
    )
    assert migrated["target_config_sha256"] == state["target_config_sha256"]
    assert {
        path.name: _file_sha256(str(path))
        for path in eval_dir.iterdir()
        if path.name != "transfer_provenance.json"
    } == artifact_hashes

    migrated["target_config_sha256"] = "unrelated-target"
    # Simulate external corruption. The production writer itself rejects this
    # malformed identity before touching disk.
    (eval_dir / "transfer_provenance.json").write_text(
        json.dumps(migrated),
    )
    with pytest.raises(ValueError, match="invalid target_config_sha256"):
        controller._restore_transfer_replays(
            state=state, quality_store=store, quality_results=records,
        )


def test_completed_transfer_state_pins_result_hash(tmp_path) -> None:
    target_system = {
        "system_hardware": "a100x8", "rag_cm": {"calibration_mode": "uncalibrated_spec"},
    }
    controller = _resume_controller(tmp_path, target_system)
    trace = {"queries": [{"question_id": "q"}]}
    records = [_source_record("source-a", 0, 0.5)]
    store = _ResumeStore(tmp_path / "store", records, {"source-a": trace})
    state, _ = controller._prepare_system_transfer_state(
        quality_store=store, quality_results=records, n_refine_iterations=10,
    )
    result_path = tmp_path / "target/system_transfer_result.json"
    result_path.write_text('{"status": "completed"}\n')
    state.update({
        "status": "completed",
        "refinement_iterations_completed": 10,
        "result_sha256": _file_sha256(str(result_path)),
    })
    controller._atomic_transfer_json(
        str(tmp_path / "target/system_transfer_state.json"), state,
    )
    controller._prepare_system_transfer_state(
        quality_store=store, quality_results=records, n_refine_iterations=10,
    )

    result_path.write_text('{"status": "tampered"}\n')
    with pytest.raises(RuntimeError, match="state/result integrity"):
        controller._prepare_system_transfer_state(
            quality_store=store, quality_results=records, n_refine_iterations=10,
        )


def test_completed_transfer_state_allows_monotonic_budget_extension(
    tmp_path,
) -> None:
    target_system = {
        "system_hardware": "a100x8",
        "rag_cm": {"calibration_mode": "uncalibrated_spec"},
    }
    controller = _resume_controller(tmp_path, target_system)
    trace = {"queries": [{"question_id": "q"}]}
    records = [_source_record("source-a", 0, 0.5)]
    store = _ResumeStore(tmp_path / "store", records, {"source-a": trace})
    state, _ = controller._prepare_system_transfer_state(
        quality_store=store, quality_results=records, n_refine_iterations=10,
    )
    _commit_completed_refinement_ledger(
        controller,
        store=store,
        state=state,
        target_system=target_system,
        trace=trace,
        count=10,
    )
    result_path = tmp_path / "target/system_transfer_result.json"
    result_path.write_text('{"status": "completed"}\n')
    prior_result_sha256 = _file_sha256(str(result_path))
    state.update({
        "status": "completed",
        "refinement_iterations_completed": 10,
        "result_sha256": prior_result_sha256,
    })
    controller._atomic_transfer_json(
        str(tmp_path / "target/system_transfer_state.json"), state,
    )

    extended, selected = controller._prepare_system_transfer_state(
        quality_store=store, quality_results=records, n_refine_iterations=20,
    )

    assert [item["config_hash"] for item in selected] == ["source-a"]
    assert extended["status"] == "running"
    assert extended["refinement_iterations_requested"] == 20
    assert "refinement_iterations_completed" not in extended
    assert "result_sha256" not in extended
    assert extended["refinement_budget_extensions"] == [{
        "from": 10,
        "to": 20,
        "prior_status": "completed",
        "prior_completed_iterations": 10,
        "prior_result_sha256": prior_result_sha256,
    }]
    assert _file_sha256(str(result_path)) == prior_result_sha256

    resumed, _ = controller._prepare_system_transfer_state(
        quality_store=store, quality_results=records, n_refine_iterations=20,
    )
    assert resumed == extended
    with pytest.raises(RuntimeError, match="budget cannot be decreased"):
        controller._prepare_system_transfer_state(
            quality_store=store,
            quality_results=records,
            n_refine_iterations=19,
        )


@pytest.mark.parametrize("ledger_error", ["missing", "surplus"])
def test_completed_transfer_budget_extension_rejects_inexact_old_ledger(
    tmp_path, ledger_error: str,
) -> None:
    target_system = {
        "system_hardware": "a100x8",
        "rag_cm": {"calibration_mode": "uncalibrated_spec"},
    }
    controller = _resume_controller(tmp_path, target_system)
    trace = {"queries": [{"question_id": "q"}]}
    records = [_source_record("source-a", 0, 0.5)]
    store = _ResumeStore(tmp_path / "store", records, {"source-a": trace})
    state, _ = controller._prepare_system_transfer_state(
        quality_store=store, quality_results=records, n_refine_iterations=10,
    )
    _commit_completed_refinement_ledger(
        controller,
        store=store,
        state=state,
        target_system=target_system,
        trace=trace,
        count=10,
    )
    result_path = tmp_path / "target/system_transfer_result.json"
    result_path.write_text('{"status": "completed"}\n')
    state.update({
        "status": "completed",
        "refinement_iterations_completed": 10,
        "result_sha256": _file_sha256(str(result_path)),
    })
    state_path = tmp_path / "target/system_transfer_state.json"
    controller._atomic_transfer_json(str(state_path), state)
    sealed_state = state_path.read_bytes()

    last_marker = (
        tmp_path / "target/evaluations/eval_0010/transfer_refinement.json"
    )
    if ledger_error == "missing":
        last_marker.unlink()
    else:
        surplus = json.loads(last_marker.read_text())
        surplus["refinement_index"] = 10
        surplus["eval_id"] = 11
        surplus_dir = tmp_path / "target/evaluations/eval_0011"
        surplus_dir.mkdir()
        (surplus_dir / "transfer_refinement.json").write_text(
            json.dumps(surplus)
        )

    with pytest.raises(RuntimeError, match="marker ledger"):
        controller._prepare_system_transfer_state(
            quality_store=store,
            quality_results=records,
            n_refine_iterations=20,
        )
    assert state_path.read_bytes() == sealed_state


def test_replay_resume_restores_committed_prefix_and_rejects_partial_tail(
    tmp_path,
) -> None:
    target_system = {
        "system_hardware": "a100x8", "rag_cm": {"calibration_mode": "uncalibrated_spec"},
    }
    controller = _resume_controller(tmp_path, target_system)
    trace = {"queries": [{"question_id": "q"}]}
    records = [_source_record("source-a", 0, 0.5), _source_record("source-b", 1, 0.6)]
    store = _ResumeStore(
        tmp_path / "store", records,
        {"source-a": trace, "source-b": trace},
    )
    state, records = controller._prepare_system_transfer_state(
        quality_store=store, quality_results=records, n_refine_iterations=10,
    )

    eval_dir = tmp_path / "target/evaluations/eval_0000"
    eval_dir.mkdir(parents=True)
    (eval_dir / "pipeline_config.json").write_text(json.dumps({"system": target_system}))
    (eval_dir / "quality_metrics.json").write_text(
        json.dumps(records[0]["quality_metrics"])
    )
    (eval_dir / "execution_dag.json").write_text(json.dumps(trace))
    (eval_dir / "assembly.json").write_text(json.dumps({
        "performance_score": 12.0, "best_hw_params": {"num_chips": 8},
    }))
    controller._save_transfer_provenance(
        replay_idx=0,
        config_hash_value="source-a",
        source_record=records[0]["record"],
        target_eval_type=EvalType.GT_FP,
        source_record_marker_sha256=state["source_records"][0]["record_marker_sha256"],
    )
    # Simulate SIGKILL after the expensive eval+CSV transaction but before
    # its tiny provenance marker became durable.
    (eval_dir / "transfer_provenance.json").unlink()
    (tmp_path / "target/all_evaluations.csv").write_text(
        "performance,quality__combined,eval_type,eval_error,eval_id\n"
        "12.000000,0.500000,GT_FP,,0\n"
    )

    restored, assemblies = controller._restore_transfer_replays(
        state=state, quality_store=store, quality_results=records,
    )
    assert len(restored) == 1
    assert restored[0]["free"] is False  # source GT history remains a paid phase seat
    assert restored[0]["quality_reused"] is True
    assert assemblies[0].performance_score == 12.0
    assert controller._eval_counter == 1
    assert (eval_dir / "transfer_provenance.json").is_file()

    (tmp_path / "target/evaluations/eval_0001").mkdir()
    restored_again, _ = controller._restore_transfer_replays(
        state=state, quality_store=store, quality_results=records,
    )
    assert len(restored_again) == 1
    assert not (tmp_path / "target/evaluations/eval_0001").exists()


def test_refinement_resume_restores_only_landed_commits_and_counters(
    tmp_path,
) -> None:
    target_system = {
        "system_hardware": "a100x8", "rag_cm": {"calibration_mode": "uncalibrated_spec"},
    }
    controller = _resume_controller(tmp_path, target_system)
    controller._quality_algo_key = lambda config: config["hash"]
    controller._quality_algo_config = lambda config: deepcopy(config)
    controller._ssops = SimpleNamespace(
        decode_to_nested=lambda flat: {"hash": flat["cfg__hash"]},
    )
    trace = {"queries": [{"question_id": "q"}]}
    source = _source_record("source-a", 0, 0.5)
    store = _ResumeStore(tmp_path / "store", [source], {"source-a": trace})
    state, _ = controller._prepare_system_transfer_state(
        quality_store=store, quality_results=[source], n_refine_iterations=10,
    )

    refinement = {
        "config_hash": "refine-a",
        "config": {"hash": "refine-a"},
        "quality_metrics": {"answer_correctness": 0.75},
        "quality_combined": 0.75,
        "record": {"source": {"kind": "target-refinement"}},
    }
    store._records["refine-a"] = refinement
    store._traces["refine-a"] = trace
    (tmp_path / "store/records/refine-a.json").write_text(
        '{"algo_hash": "refine-a"}'
    )
    eval_dir = tmp_path / "target/evaluations/eval_0002"
    eval_dir.mkdir(parents=True)
    (eval_dir / "pipeline_config.json").write_text(json.dumps({"system": target_system}))
    (eval_dir / "quality_metrics.json").write_text(json.dumps(refinement["quality_metrics"]))
    (eval_dir / "execution_dag.json").write_text(json.dumps(trace))
    (eval_dir / "assembly.json").write_text(json.dumps({
        "performance_score": 9.0, "best_hw_params": {"num_chips": 8},
    }))
    (tmp_path / "target/all_evaluations.csv").write_text(
        "cfg__hash,performance,quality__combined,eval_type,eval_error,eval_id\n"
        "source-a,12.000000,0.500000,GT_FP,,0\n"
        "bad,0.000000,0.000000,CONTEXT_OVERFLOW,prompt too long,1\n"
        "refine-a,9.000000,0.750000,GT,,2\n"
    )
    controller._save_transfer_refinement_commit(
        state=state,
        refinement_index=0,
        eval_id=2,
        config={"hash": "refine-a"},
        performance=9.0,
        quality={"combined": 0.75, "answer_correctness": 0.75},
        algo_hash="refine-a",
        eval_type="GT",
        free=False,
        gt_budget_count=1,
        quality_store=store,
    )
    # Complete eval/QS/CSV survived, only the final landed marker did not.
    (eval_dir / "transfer_refinement.json").unlink()
    # Simulate a transient failure of the mandatory QualityStore append too;
    # durable target artifacts are sufficient to reconstruct it on resume.
    store._records.pop("refine-a")
    store._traces.pop("refine-a")
    (tmp_path / "store/records/refine-a.json").unlink()

    evaluations, assemblies, gt_count, total_cost = (
        controller._restore_transfer_refinements(
            state=state, quality_store=store, source_count=1,
        )
    )
    assert [item["eval_id"] for item in evaluations] == [2]
    assert assemblies[0].performance_score == 9.0
    assert gt_count == 1
    assert total_cost == 11.0
    assert controller._eval_counter == 3
    assert (eval_dir / "transfer_refinement.json").is_file()
    assert store.get_record("refine-a") is not None

    with open(tmp_path / "target/all_evaluations.csv", "a") as stream:
        stream.write("refine-b,8.000000,0.800000,GT,,3\n")
    recovered_again, _, _, _ = controller._restore_transfer_refinements(
        state=state, quality_store=store, source_count=1,
    )
    assert [item["eval_id"] for item in recovered_again] == [2]
    assert 3 not in controller._transfer_csv_rows()


def test_selective_refinement_without_profile_attestation_is_repriced(
    tmp_path,
) -> None:
    target_system = {
        "system_hardware": "a100x8",
        "rag_cm": {"calibration_mode": "uncalibrated_spec"},
    }
    controller = _resume_controller(tmp_path, target_system)
    source = _source_record("source-a", 0, 0.5)
    trace = {"queries": [{"question_id": "q"}]}
    store = _ResumeStore(tmp_path / "store", [source], {"source-a": trace})
    state, _ = controller._prepare_system_transfer_state(
        quality_store=store,
        quality_results=[source],
        n_refine_iterations=10,
    )
    state["replay_plan"] = {
        "llmlingua2_profile": {
            "logical_path": "calibration_profiles/test.json",
            "resolved_path": str(tmp_path / "test.json"),
            "sha256": "a" * 64,
        },
    }

    refinement = {
        "config_hash": "refine-a",
        "config": {"hash": "refine-a"},
        "quality_metrics": {"answer_correctness": 0.75},
        "quality_combined": 0.75,
        "record": {"source": {"kind": "target-refinement"}},
    }
    store._records["refine-a"] = refinement
    store._traces["refine-a"] = trace
    (tmp_path / "store/records/refine-a.json").write_text(
        '{"algo_hash": "refine-a"}'
    )
    eval_dir = tmp_path / "target/evaluations/eval_0001"
    eval_dir.mkdir(parents=True)
    (eval_dir / "pipeline_config.json").write_text(
        json.dumps({"system": target_system})
    )
    (eval_dir / "quality_metrics.json").write_text(
        json.dumps(refinement["quality_metrics"])
    )
    (eval_dir / "execution_dag.json").write_text(json.dumps(trace))
    (eval_dir / "assembly.json").write_text(json.dumps({
        "performance_score": 9.0,
        "best_hw_params": {"num_chips": 8},
    }))
    (tmp_path / "target/all_evaluations.csv").write_text(
        "cfg__hash,performance,quality__combined,eval_type,eval_error,eval_id\n"
        "source-a,12.000000,0.500000,GT_FP,,0\n"
        "refine-a,9.000000,0.750000,GT,,1\n"
    )

    evaluations, assemblies, gt_count, total_cost = (
        controller._restore_transfer_refinements(
            state=state,
            quality_store=store,
            source_count=1,
        )
    )

    assert evaluations == []
    assert assemblies == []
    assert gt_count == 0
    assert total_cost == 0.0
    assert not eval_dir.exists()
    assert set(controller._transfer_csv_rows()) == {0}
    assert store.get_record("refine-a") == refinement
    assert store.get_execution_dag("refine-a") == trace


def test_live_refinement_excludes_context_overflow_and_lands_next_candidate(
    tmp_path,
) -> None:
    controller = _bare_controller()
    controller.project_dir = str(tmp_path)
    controller.optimizer = SimpleNamespace(
        objective_costs={"performance": 1.0, "quality": 10.0},
        suggest=Mock(side_effect=[
            ([{"candidate": "overflow"}], ["performance", "quality"]),
            ([{"candidate": "landed"}], ["performance", "quality"]),
        ]),
        add_objectives=Mock(),
    )
    controller._eval_counter = 0
    controller._assembly_results = []
    controller._gt_count = 0
    controller._cm_count = 0
    controller._total_cost = 0.0
    controller._push_perf_attribution = Mock()
    controller._update_summary = Mock()
    controller._quality_algo_key = lambda cfg: cfg["candidate"]
    rows: dict[int, dict] = {}
    controller._transfer_csv_rows = Mock(side_effect=lambda: deepcopy(rows))
    controller._save_transfer_refinement_commit = Mock()
    store = SimpleNamespace(
        get_record=lambda record_hash: (
            {"config": {"candidate": "landed"}}
            if record_hash == "landed" else None
        ),
        get_execution_dag=Mock(return_value={"queries": []}),
    )
    controller._external_quality_store = store

    def evaluate(configs, objectives, columns):
        cfg = configs[0]
        eval_id = controller._eval_counter
        controller._eval_counter += 1
        controller._last_valid_eval_ids = [eval_id]
        controller._assembly_results.append(AssemblyResult(
            performance_score=7.0, best_hw_params={},
        ))
        if cfg["candidate"] == "overflow":
            rows[eval_id] = {
                "eval_id": str(eval_id), "eval_type": "CONTEXT_OVERFLOW",
                "performance": "0.000000", "quality__combined": "0.000000",
            }
            return (
                {"performance": [0.0], "quality": [{"combined": 0.0}]},
                configs, 0, [True],
            )
        rows[eval_id] = {
            "eval_id": str(eval_id), "eval_type": "GT",
            "performance": "7.000000", "quality__combined": "0.800000",
        }
        return (
            {"performance": [7.0], "quality": [{"combined": 0.8}]},
            configs, 1, [False],
        )

    controller._evaluate_objectives = Mock(side_effect=evaluate)
    configs: list[dict] = []
    qualities: list[float] = []
    performances: list[float] = []
    hashes: list[str] = []
    eval_ids: list[int] = []

    gt_count = controller._run_refinement(
        assembly=SimpleNamespace(),
        quality_store=store,
        n_iterations=1,
        configs=configs,
        quality_scores=qualities,
        performance_scores=performances,
        config_hashes=hashes,
        target_eval_ids=eval_ids,
        assembly_results=controller._assembly_results,
        obj_columns=[],
        transfer_state={"target_config_sha256": "target"},
    )

    assert gt_count == 1
    assert configs == [{"candidate": "landed"}]
    assert eval_ids == [1]  # overflow remains a recorded/blacklisted gap
    assert hashes == ["landed"]
    assert controller.optimizer.add_objectives.call_count == 1
    assert len(controller._assembly_results) == 1
