import json
import sys

import pytest

from rag_stack.cost_model.llm_sim.calibration.main_llm.vllm_probe import build_key
from rag_stack_evaluator.static_rag_evaluator.measured.evaluator import MeasuredEvaluator
from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import (
    VllmDeploymentManager,
)
from rag_stack_evaluator.static_rag_evaluator.measured.vllm_subprocess import (
    VllmStartupKey,
    _prefix_caching_args,
    _vllm_server_command_prefix,
    resolve_vllm_api_server_count,
)
from scripts.paper.replay_msmarco_agent_v2_old_pareto_measured import (
    is_final_result,
    measured_gt_rejection,
    measured_runtime_policy,
    result_path,
    trial_invalid_record_status,
)


def test_api_server_count_env_defaults_to_one_and_requires_positive_int(monkeypatch):
    monkeypatch.delenv("RAG_STACK_VLLM_API_SERVER_COUNT", raising=False)
    assert resolve_vllm_api_server_count() == 1

    monkeypatch.setenv("RAG_STACK_VLLM_API_SERVER_COUNT", "4")
    assert resolve_vllm_api_server_count() == 4

    for invalid in ("0", "-2", "many"):
        monkeypatch.setenv("RAG_STACK_VLLM_API_SERVER_COUNT", invalid)
        with pytest.raises(ValueError, match="must be a positive integer"):
            resolve_vllm_api_server_count()


def test_replay_runtime_policy_tracks_api_server_count(monkeypatch):
    monkeypatch.setenv("RAG_STACK_VLLM_API_SERVER_COUNT", "4")
    assert measured_runtime_policy()["vllm_api_server_count"] == 4


def test_replay_runtime_policy_fingerprints_disagg_role_admission():
    collocated = measured_runtime_policy("collocated_pd")
    disaggregated = measured_runtime_policy("disagg_pd")

    assert collocated["version"] == (
        "closed_loop_stage_services_v7_aux_dp_saturation_gated_performance_trace"
    )
    assert collocated["llm_http_admission"] == (
        "shared_engine_max_num_seqs_plus_20pct"
    )
    assert disaggregated["version"] == (
        "closed_loop_stage_services_v8_pd_prefill_feeder_aux_dp_"
        "saturation_gated_performance_trace"
    )
    assert collocated["hf_aux_data_parallel"] == (
        "one_replica_per_declared_device_v1"
    )
    assert collocated["hf_aux_aggregate_batch"] == (
        "replicas_x_batch_size_request_v1"
    )
    assert collocated["measured_window_gate"] == (
        "standing_backlog_plus_rate_stable_v2"
    )
    assert collocated["measured_gt_required"] == (
        "measured_gt_admissible_true"
    )
    assert disaggregated["llm_http_admission"] == (
        "prefill_engine_plus_one_feeder_decode_strict_outer_sum"
    )
    assert disaggregated["pd_prefill_feeder_slots"] == 1
    assert disaggregated["pd_handoff_backpressure"] == (
        "hold_prefill_until_decode_admitted_v1"
    )


def _admissibility(admissible: bool, reasons: list[str]) -> dict:
    names = (
        "driver_errors_window_zero",
        "measurement_rate_stability_waves_ge_2",
        "population_saturated",
        "qps_subwindow_cv_le_0_15",
        "warmup_rate_stable_not_cap_forced",
    )
    return {
        "admissible": admissible,
        "reasons": reasons,
        "checks": {
            name: {"passed": admissible, "observed": None, "required": True}
            for name in names
        },
    }


def test_replay_resume_rejects_different_runtime_policy(tmp_path):
    point = {
        "pid": "s43_eval_0041",
        "quality": "0.331000",
        "overlay_version": "overlay",
        "measured_runtime_policy": {
            "version": "policy",
            "vllm_api_server_count": 4,
        },
    }
    record = {
        **point,
        "status": "ok",
        "measured_performance": 321.0,
        "raw_performance": {
            "measured_gt_admissibility": _admissibility(True, []),
        },
    }
    path = result_path(tmp_path, point)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record))

    assert is_final_result(tmp_path, point)
    one_frontend = {
        **point,
        "measured_runtime_policy": {
            "version": "policy",
            "vllm_api_server_count": 1,
        },
    }
    assert not is_final_result(tmp_path, one_frontend)


def test_replay_rejects_positive_qps_when_measured_gt_is_inadmissible():
    raw = {
        "qps": 2.786069,
        "measured_gt_admissibility": _admissibility(
            False,
            [
                "warmup_rate_stability_gate_not_met",
                "measurement_waves_below_minimum",
            ],
        ),
    }

    assert measured_gt_rejection(raw) == (
        "measured GT inadmissible: warmup_rate_stability_gate_not_met, "
        "measurement_waves_below_minimum"
    )
    raw["measured_gt_admissibility"] = _admissibility(True, [])
    assert measured_gt_rejection(raw) is None
    raw["measured_gt_admissibility"]["obsolete_header"] = 1
    assert measured_gt_rejection(raw).startswith(
        "invalid measured_gt_admissibility verdict"
    )


def test_replay_retries_warmup_proof_timeout_but_not_bad_deployment():
    assert trial_invalid_record_status(
        "measured_warmup_gate_timeout: cap=240s"
    ) == "inadmissible"
    assert trial_invalid_record_status(
        "measured_population_not_saturated: cap=4096"
    ) == "inadmissible"
    assert trial_invalid_record_status(
        "aux-stage CUDA OOM under the planned vLLM tenancy"
    ) == "invalid"


def test_single_api_server_preserves_legacy_launcher():
    key = VllmStartupKey(model="Qwen/Test", api_server_count=1)
    assert _vllm_server_command_prefix(key) == [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        "Qwen/Test",
    ]


def test_prefix_cache_policy_preserves_measured_default_and_can_be_pinned_off():
    assert _prefix_caching_args(VllmStartupKey(model="Qwen/Test")) == []
    assert _prefix_caching_args(
        VllmStartupKey(model="Qwen/Test", enable_prefix_caching=False)
    ) == ["--no-enable-prefix-caching"]


def test_multiple_api_servers_use_vllm_serve_dispatcher():
    key = VllmStartupKey(model="Qwen/Test", api_server_count=4)
    assert _vllm_server_command_prefix(key) == [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "Qwen/Test",
        "--api-server-count",
        "4",
    ]
    assert key != VllmStartupKey(model="Qwen/Test", api_server_count=1)


def test_stage_calibration_telemetry_uses_single_frontend_wrapper(monkeypatch):
    monkeypatch.setenv(
        "RAG_STACK_STAGE_TELEMETRY_PATH", "/tmp/stage-cycles.jsonl"
    )
    monkeypatch.setenv("RAG_STACK_VLLM_CALIBRATION_RUN_ID", "stage-test")
    key = VllmStartupKey(model="Qwen/Test", api_server_count=1)
    assert _vllm_server_command_prefix(key) == [
        sys.executable,
        "-m",
        "rag_stack_evaluator.vllm_instrumentation.serving_curves.calibration.vllm_server",
        "--model",
        "Qwen/Test",
    ]

    with pytest.raises(ValueError, match="exactly one API frontend"):
        _vllm_server_command_prefix(
            VllmStartupKey(model="Qwen/Test", api_server_count=2)
        )


def test_stage_calibration_telemetry_path_cannot_be_empty(monkeypatch):
    monkeypatch.setenv("RAG_STACK_STAGE_TELEMETRY_PATH", "")
    with pytest.raises(ValueError, match="cannot be empty"):
        _vllm_server_command_prefix(VllmStartupKey(model="Qwen/Test"))


def test_v1_production_timing_uses_shared_single_frontend_wrapper(monkeypatch):
    monkeypatch.delenv("RAG_STACK_STAGE_TELEMETRY_PATH", raising=False)
    monkeypatch.setenv("RAG_STACK_V1_TIMING_DISPATCH_COUNTER", "1")
    key = VllmStartupKey(model="Qwen/Test", api_server_count=1)
    assert _vllm_server_command_prefix(key) == [
        sys.executable,
        "-m",
        "rag_stack_evaluator.vllm_instrumentation.serving_curves.calibration.vllm_server",
        "--model",
        "Qwen/Test",
    ]

    with pytest.raises(ValueError, match="exactly one API frontend"):
        _vllm_server_command_prefix(
            VllmStartupKey(model="Qwen/Test", api_server_count=2)
        )


def test_stage_proof_and_production_timing_cannot_share_a_server(monkeypatch):
    monkeypatch.setenv(
        "RAG_STACK_STAGE_TELEMETRY_PATH", "/tmp/stage-cycles.jsonl"
    )
    monkeypatch.setenv("RAG_STACK_V1_TIMING_DISPATCH_COUNTER", "1")
    with pytest.raises(ValueError, match="separate servers"):
        _vllm_server_command_prefix(VllmStartupKey(model="Qwen/Test"))


def test_deployment_and_calibration_resolve_the_same_api_server_count(monkeypatch):
    monkeypatch.setenv("RAG_STACK_VLLM_API_SERVER_COUNT", "4")

    class Cache:
        key = None

        def register_main_vllm(self, key):
            self.key = key

    cache = Cache()
    manager = VllmDeploymentManager(["cuda:0"])
    manager._register_engine(
        cache=cache,
        role="main",
        model="Qwen/Test",
        dtype="bfloat16",
        util=0.85,
        devices=["cuda:0"],
        disagg=False,
        prefill_devices=None,
        decode_devices=None,
        knobs={
            "max_num_seqs": 64,
            "max_num_batched_tokens": None,
            "max_model_len": -1,
            "kv_cache_dtype": "auto",
        },
    )

    calibration_key = build_key("Qwen/Test", 1, 1, {})
    assert cache.key.api_server_count == 4
    assert calibration_key.api_server_count == 4


def test_summary_exports_generator_api_server_count(monkeypatch):
    monkeypatch.setenv("RAG_STACK_VLLM_API_SERVER_COUNT", "4")
    collocated = {
        "layout": {
            "engines": {
                "generator": {
                    "pd_serving": "collocated_pd",
                    "devices": ["cuda:0"],
                    "tp": 1,
                    "pp": 1,
                }
            }
        }
    }
    summary = {}
    MeasuredEvaluator._add_deployment_metadata(summary, {}, collocated)
    assert summary["generator_api_server_count"] == 4

    disaggregated = {
        "layout": {
            "engines": {
                "generator": {
                    "pd_serving": "disagg_pd",
                    "generator_prefill": {
                        "devices": ["cuda:0"],
                        "tp": 1,
                        "pp": 1,
                    },
                    "generator_decode": {
                        "devices": ["cuda:1"],
                        "tp": 1,
                        "pp": 1,
                    },
                }
            }
        }
    }
    summary = {}
    MeasuredEvaluator._add_deployment_metadata(summary, {}, disaggregated)
    assert summary["generator_api_server_count"] == 1
