from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

import rag_stack.cost_model as cost_model
from rag_stack.cost_model.llm_sim import calibration
from rag_stack.cost_model.llm_sim.calibration import _efficiency_identifiability
from rag_stack.cost_model.llm_sim.calibration import _predict as predict_module
from rag_stack.cost_model.llm_sim.calibration.main_llm import generator
from rag_stack.cost_model.llm_sim.calibration.main_llm import (
    generator_request_contract,
)
from rag_stack.cost_model.llm_sim.calibration.main_llm import vllm_probe


def _generator_request_contracts(prompt_tokens) -> dict:
    contracts = {}
    for prompt_tokens_value in sorted({int(value) for value in prompt_tokens}):
        token_ids = list(range(prompt_tokens_value))
        contract = {
            "schema": generator_request_contract.REQUEST_CONTRACT_SCHEMA,
            "endpoint": generator_request_contract.ENDPOINT,
            "request_format": generator_request_contract.REQUEST_FORMAT,
            "messages": [{"role": "user", "content": "fixture question"}],
            "chat_template": {
                "source": generator_request_contract.CHAT_TEMPLATE_SOURCE,
                "tokenize": True,
                "add_generation_prompt": True,
                "chat_template_sha256": hashlib.sha256(
                    b"fixture-chat-template"
                ).hexdigest(),
            },
            "rendered_prompt_tokens": prompt_tokens_value,
            "rendered_prompt_token_ids_sha256": (
                generator_request_contract._sha256(token_ids)
            ),
            "temperature": generator_request_contract.CALIBRATION_TEMPERATURE,
            "seed": generator_request_contract.CALIBRATION_SEED,
            "stream": False,
            "ignore_eos": True,
            "ignore_eos_purpose": generator_request_contract.IGNORE_EOS_PURPOSE,
            "logprobs": True,
            "top_logprobs": 0,
            "n": 1,
            "max_tokens_policy": generator_request_contract.MAX_TOKENS_POLICY,
            "truncate_prompt_tokens_policy": (
                generator_request_contract.TRUNCATE_PROMPT_TOKENS_POLICY
            ),
            "truncation_side": generator_request_contract.TRUNCATION_SIDE,
        }
        contracts[str(prompt_tokens_value)] = contract
    return contracts


def _independent_latency(
    batch_size: int,
    seq_len: int,
    compute_efficiency: float,
    memory_efficiency: float,
) -> float:
    # Long sequences expose a compute-dominant regime; large batches expose a
    # memory-dominant regime.  Both terms remain non-zero, as in a real model.
    compute_work = 0.002 * float(seq_len) + 0.01
    memory_work = 0.004 * float(batch_size) + 0.01
    return (
        compute_work / float(compute_efficiency)
        + memory_work / float(memory_efficiency)
    )


def _predict_from_pipeline(
    _cost_model,
    pipeline_config,
    _hardware_dict,
    compute_eff=1.0,
    memory_eff=1.0,
    bits="bf16",
    per_call_overhead_s=0.0,
):
    del bits
    module = pipeline_config["node_lines"][0]["nodes"][0]["modules"][0]
    batch_size = int(module["batch_size"])
    seq_len = int(pipeline_config["_token_stats"]["avg_chunk_tokens"])
    return (
        _independent_latency(batch_size, seq_len, compute_eff, memory_eff)
        + float(per_call_overhead_s)
    )


class _CostModel:
    @staticmethod
    def _get_model_name_for_df(_pipeline_config):
        return "synthetic_model"


@pytest.fixture
def fit_environment(monkeypatch):
    monkeypatch.setenv("RAG_STACK_FAST_FIT", "0")
    monkeypatch.setattr(calibration, "detect_hardware_key", lambda _hardware: "test_gpu")
    monkeypatch.setattr(calibration, "_hardware_to_dict", lambda _hardware: {})
    # Avoid importing/probing torch.cuda in this CPU-only regression.
    monkeypatch.setattr(calibration, "get_cuda_name", lambda: "")
    monkeypatch.setattr(predict_module, "predict_latency", _predict_from_pipeline)
    return {
        "node_lines": [{
            "nodes": [{
                "modules": [{"batch_size": 1}],
            }],
        }],
        "_token_stats": {
            "avg_chunk_tokens": 1,
            "avg_output_tokens": 1,
            "avg_query_tokens": 0,
            "prompt_template_tokens": 0,
        },
    }


def _evidence(points, ceff=0.55, meff=0.78):
    return {
        point: _independent_latency(point[0], point[1], ceff, meff)
        for point in points
    }


def test_default_generator_recipe_has_disjoint_fit_and_validation_evidence():
    assert len(generator.POINTS) == 4
    assert len(generator.VALIDATION_POINTS) == 2
    assert set(generator.POINTS).isdisjoint(generator.VALIDATION_POINTS)
    assert max(point[0] for point in generator.POINTS) == 128
    assert any(point[0] == 128 for point in generator.POINTS)
    assert any(point[0] == 256 for point in generator.VALIDATION_POINTS)
    assert not any(point[0] == 256 for point in generator.POINTS)
    assert (128, 64, 128) in generator.POINTS
    assert (256, 64, 128) in generator.VALIDATION_POINTS
    coverage = generator._resident_batch_coverage_contract(
        generator.POINTS,
        generator.VALIDATION_POINTS,
        reference_max_num_seqs=256,
    )
    assert coverage["required_batches"] == [128, 256]
    assert coverage["fit_batches"] == [4, 32, 128]
    assert coverage["held_out_batches"] == [16, 256]


def test_generator_recipe_fails_closed_without_b256_or_with_cap128():
    with pytest.raises(
        generator.GeneratorComponentCoverageError,
        match="missing=\\[256\\]",
    ):
        generator._resident_batch_coverage_contract(
            generator.POINTS,
            [(16, 128, 128), (64, 64, 256)],
            reference_max_num_seqs=128,
        )

    with pytest.raises(
        generator.GeneratorComponentCoverageError,
        match="max_num_seqs cannot cover required resident batch 256",
    ):
        generator._resident_batch_coverage_contract(
            generator.POINTS,
            generator.VALIDATION_POINTS,
            reference_max_num_seqs=128,
        )


def test_generator_recipe_rejects_ambiguous_fit_heldout_batch_roles():
    with pytest.raises(
        generator.GeneratorComponentCoverageError,
        match="roles are ambiguous",
    ):
        generator._resident_batch_coverage_contract(
            generator.POINTS,
            [*generator.VALIDATION_POINTS, (128, 128, 128)],
            reference_max_num_seqs=256,
        )


def test_generator_recipe_requires_same_shape_b128_b256_batch_pair():
    with pytest.raises(
        generator.GeneratorComponentCoverageError,
        match="same-token-shape pair",
    ):
        generator._resident_batch_coverage_contract(
            [
                point
                for point in generator.POINTS
                if point != (128, 64, 128)
            ],
            generator.VALIDATION_POINTS,
            reference_max_num_seqs=256,
        )


def test_generator_calibration_uses_only_tp1_component_service_and_keeps_raw_evidence(
    monkeypatch, tmp_path,
):
    class _GeneratorCostModel:
        @staticmethod
        def _get_params(pipeline_config):
            token_stats = pipeline_config["_token_stats"]
            return {
                "input_tokens": int(token_stats["avg_chunk_tokens"]),
                "output_tokens": int(token_stats["avg_output_tokens"]),
            }

    captured = {}
    fake_hardware = object()
    monkeypatch.setenv("RAG_STACK_CALIBRATION_DATA_DIR", str(tmp_path))
    monkeypatch.setitem(
        cost_model.COST_MODEL_REGISTRY,
        "llm_sim",
        _GeneratorCostModel,
    )
    monkeypatch.setattr(
        calibration,
        "hardware_from_config",
        lambda _config: (fake_hardware, {"available_gpus": ["cuda:7", "cuda:8"]}),
    )
    monkeypatch.setattr(
        calibration,
        "detect_hardware_key",
        lambda _hardware: "test_gpu",
    )

    def fake_build_key(
        model,
        tp,
        pp,
        sysd,
        devices=None,
        *,
        max_num_seqs=None,
        **kwargs,
    ):
        captured["build_key"] = {
            "model": model,
            "tp": tp,
            "pp": pp,
            "sysd": sysd,
            "devices": devices,
            "max_num_seqs": max_num_seqs,
            **kwargs,
        }
        return SimpleNamespace(
            model=model,
            device=",".join(devices or ["cuda:7"]),
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=kwargs["max_num_batched_tokens"],
            enable_prefix_caching=kwargs["enable_prefix_caching"],
            api_server_count=kwargs["api_server_count"],
            max_model_len=kwargs["max_model_len"],
            dtype="bfloat16",
            kv_cache_dtype="auto",
        )

    def fake_measure_component_grid(
        key,
        shapes,
        *,
        telemetry_path,
        repeats,
        request_contracts,
    ):
        captured["measure_key"] = key
        captured["shapes"] = list(shapes)
        captured["repeats"] = repeats
        captured["request_contracts"] = request_contracts
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        telemetry_path.write_text('{"raw_worker_telemetry":true}\n')
        results = {
            shape: (
                10_000.0 + shape[0],       # offline cohort QPS diagnostic
                1_000.0 + shape[0],        # deliberately absurd cohort wall
                0.01 + shape[0] / 10_000,  # pure GPU component service
            )
            for shape in shapes
        }
        captured["results"] = results
        return results

    def fake_audit_component_raw(
        _path,
        *,
        requested_shapes,
        **_kwargs,
    ):
        audited = {}
        for shape in requested_shapes:
            component_service_s = captured["results"][shape][2]
            measured = (
                component_service_s * 0.99,
                component_service_s,
                component_service_s * 1.01,
            )
            audited[shape] = vllm_probe._OfflineComponentRawPoint(
                run_id="component-run",
                warmup_component_services_s=(
                    component_service_s * 2.0,
                    component_service_s * 1.5,
                ),
                measurement_component_services_s=measured,
                component_service_s=component_service_s,
                measurement_relative_spread=(
                    max(measured) - min(measured)
                ) / component_service_s,
                cohort_cycle_ids=tuple(
                    (f"component-run:{index}:{shape}",)
                    for index in range(5)
                ),
            )
        return audited

    def fake_fit_and_save(**kwargs):
        captured["fit"] = kwargs
        return SimpleNamespace(
            compute_efficiency=0.6,
            memory_efficiency=0.7,
            fit_mape=0.01,
            validation_mape=0.02,
            correlation=0.99,
        )

    monkeypatch.setattr(vllm_probe, "build_key", fake_build_key)
    monkeypatch.setattr(
        generator_request_contract,
        "build_generator_request_contracts",
        lambda _model, prompt_tokens: _generator_request_contracts(
            prompt_tokens
        ),
    )
    monkeypatch.setattr(
        vllm_probe,
        "measure_component_grid",
        fake_measure_component_grid,
    )
    monkeypatch.setattr(
        vllm_probe,
        "_audit_offline_component_raw",
        fake_audit_component_raw,
    )
    monkeypatch.setattr(calibration, "fit_and_save", fake_fit_and_save)

    assert generator.calibrate(
        "Qwen/reference",
        "unused.yaml",
        tp=1,
        pp=1,
        devices=["cuda:7", "cuda:8"],
    ) == 0

    build = captured["build_key"]
    assert (build["tp"], build["pp"]) == (1, 1)
    assert build["devices"] == ["cuda:7"]
    expected_cap = max(
        point[0] for point in generator.POINTS + generator.VALIDATION_POINTS
    )
    assert build["max_num_seqs"] == expected_cap
    assert build["max_num_batched_tokens"] == 65_536
    assert build["enable_prefix_caching"] is False
    assert build["api_server_count"] == 1
    assert build["max_model_len"] == 4096
    assert captured["measure_key"].max_num_seqs == expected_cap

    fit = captured["fit"]
    expected_fit = {}
    expected_validation = {}
    validation_specs = set(generator.VALIDATION_POINTS)
    for measurement_key, result in captured["results"].items():
        c, input_tokens, output_tokens = measurement_key
        calibration_key = (c, (input_tokens, output_tokens))
        target = (
            expected_validation
            if (c, input_tokens, output_tokens) in validation_specs
            else expected_fit
        )
        target[calibration_key] = result[2]
    assert fit["measured"] == expected_fit
    assert fit["validation_points"] == expected_validation
    assert all(value < 1.0 for value in fit["measured"].values())
    assert all(value < 1.0 for value in fit["validation_points"].values())
    generator_module = (
        fit["pipeline_config"]["node_lines"][0]["nodes"][1]["modules"][0]
    )
    assert generator_module["tensor_parallel"] == 1
    assert generator_module["pipeline_parallel"] == 1

    telemetry_files = list(tmp_path.rglob("*.jsonl"))
    manifest_files = list(tmp_path.rglob("*.manifest.json"))
    assert len(telemetry_files) == 1
    assert len(manifest_files) == 1
    assert telemetry_files[0].read_text() == '{"raw_worker_telemetry":true}\n'
    manifest = json.loads(manifest_files[0].read_text())
    assert manifest["fit_latency_source"] == (
        calibration.GENERATOR_COMPONENT_FIT_LATENCY_SOURCE
    )
    assert "measurement_evidence" not in fit
    assert manifest["run_id"] == "component-run"
    assert manifest["cohort_contract"] == {
        "max_selected_relative_spread": 0.1,
        "selected_measurement_cohorts": 3,
        "selection": "median_of_selected_measurement_cohort_totals",
        "total_cohorts": 5,
        "warmup_cohorts": 2,
    }
    assert manifest["reference_execution"] == {
        "dtype": "bfloat16",
        "enable_prefix_caching": False,
        "kv_cache_dtype": "auto",
        "max_model_len": 4096,
        "max_num_batched_tokens": 65_536,
        "max_num_seqs": expected_cap,
        "mode": "offline_inprocess_engine_full_cohort_enqueue",
        "pipeline_parallel_size": 1,
        "tensor_parallel_size": 1,
    }
    assert len(manifest["raw_telemetry_sha256"]) == 64
    assert manifest["raw_telemetry_path"] == str(telemetry_files[0].resolve())
    expected_coverage = {
        "schema": "rag_stack.generator_component_resident_batch_coverage",
        "batch_unit": "resident_sequences",
        "required_batches": [128, 256],
        "fit_batches": [4, 32, 128],
        "held_out_batches": [16, 256],
        "reference_max_num_seqs": 256,
        "status": "complete",
    }
    assert manifest["resident_batch_coverage"] == expected_coverage
    assert manifest["request_contracts"] == captured["request_contracts"]
    expected_request_sha = (
        generator_request_contract.generator_request_contracts_sha256(
            captured["request_contracts"]
        )
    )
    assert manifest["request_contracts_sha256"] == expected_request_sha
    assert {point["split"] for point in manifest["points"]} == {
        "fit", "validation",
    }
    assert all(
        point["offline_cohort_wall_s_diagnostic"] > 1_000
        and point["component_service_s"] < 1
        and len(point["cohorts"]) == 5
        and [cohort["role"] for cohort in point["cohorts"]]
        == [
            "warmup",
            "warmup",
            "selected_measurement",
            "selected_measurement",
            "selected_measurement",
        ]
        for point in manifest["points"]
    )


def test_generator_cli_cannot_weaken_fixed_component_cohort_contract():
    with pytest.raises(SystemExit, match="fixed at 5"):
        generator.main(["--config", "unused.yaml", "--repeats", "3"])


def test_generator_component_calibration_rejects_deployment_topology():
    with pytest.raises(ValueError, match="requires the TP1/PP1 reference"):
        generator.calibrate(
            "Qwen/reference",
            "unused.yaml",
            tp=2,
            pp=2,
        )


def test_identifiability_distinguishes_independent_and_collinear_coefficients():
    independent = lambda bs, sl, ce, me, _oh: _independent_latency(
        bs, sl, ce, me,
    )
    rank, condition = _efficiency_identifiability(
        independent,
        [(1, 256), (64, 16), (16, 64)],
        0.55,
        0.78,
    )
    assert rank == 2
    assert condition < 100.0

    def collinear(bs, sl, ce, me, _oh):
        scale = float(bs + sl)
        return scale * (1.0 / ce + 1.0 / me)

    rank, condition = _efficiency_identifiability(
        collinear,
        [(1, 256), (64, 16), (16, 64)],
        0.6,
        0.6,
    )
    assert rank == 1
    assert condition == float("inf")


def test_fit_is_zero_intercept_and_persists_held_out_diagnostics(
    fit_environment, tmp_path,
):
    training = _evidence([(1, 256), (64, 16), (16, 64)])
    validation = _evidence([(4, 128), (128, 8)])

    record = calibration.fit_and_save(
        _CostModel(),
        object(),
        fit_environment,
        training,
        component="agnostic",
        output_dir=tmp_path,
        validation_points=validation,
    )

    assert record.compute_efficiency == pytest.approx(0.55, rel=2e-3)
    assert record.memory_efficiency == pytest.approx(0.78, rel=2e-3)
    assert record.per_call_overhead_s == 0.0
    assert record.identifiability_rank == 2
    assert record.identifiability_condition < 100.0
    assert record.validation_mape == pytest.approx(0.0, abs=2e-4)
    assert {(p["batch_size"], p["seq_len"]) for p in record.grid} == set(training)
    assert {
        (p["batch_size"], p["seq_len"]) for p in record.validation_grid
    } == set(validation)

    payload = json.loads((tmp_path / "test_gpu.json").read_text())
    stored = payload["profiles"]["agnostic__synthetic_model__bf16"]
    assert stored["per_call_overhead_s"] == 0.0
    assert stored["identifiability_rank"] == 2
    assert len(stored["validation_grid"]) == 2
    assert not {
        "measurement_evidence",
        "single_role_measurement_evidence",
        "cuda_name",
        "timestamp",
    }.intersection(stored)


def test_zero_intercept_fit_balances_all_sparse_regimes(
    fit_environment, tmp_path,
):
    """A sparse fit must not make one anchor the L1 sacrificial point."""

    ceff, meff = 0.8, 0.4
    fixed_unmodelled_s = 0.08
    training_points = [(4, 64), (64, 512), (256, 64)]
    validation_points = [(16, 128), (64, 64)]
    training = {
        point: _independent_latency(*point, ceff, meff) + fixed_unmodelled_s
        for point in training_points
    }
    validation = {
        point: _independent_latency(*point, ceff, meff) + fixed_unmodelled_s
        for point in validation_points
    }

    record = calibration.fit_and_save(
        _CostModel(),
        object(),
        fit_environment,
        training,
        component="agnostic",
        output_dir=tmp_path,
        validation_points=validation,
    )

    training_errors = [abs(float(point["rel_err"])) for point in record.grid]
    assert record.per_call_overhead_s == 0.0
    assert max(training_errors) < 0.30
    assert record.fit_mape < 0.20
    assert record.validation_mape < 0.20


def test_fit_rejects_bad_held_out_points_without_writing_profile(
    fit_environment, tmp_path,
):
    training = _evidence([(1, 256), (64, 16), (16, 64)])
    validation = {
        point: latency * 2.0
        for point, latency in _evidence([(4, 128), (128, 8)]).items()
    }

    with pytest.raises(RuntimeError, match="held-out validation failed"):
        calibration.fit_and_save(
            _CostModel(),
            object(),
            fit_environment,
            training,
            component="agnostic",
            output_dir=tmp_path,
            validation_points=validation,
        )

    assert not (tmp_path / "test_gpu.json").exists()


def test_fit_rejects_training_validation_overlap(fit_environment, tmp_path):
    training = _evidence([(1, 256), (64, 16), (16, 64)])
    with pytest.raises(ValueError, match="must be disjoint"):
        calibration.fit_and_save(
            _CostModel(),
            object(),
            fit_environment,
            training,
            component="agnostic",
            output_dir=tmp_path,
            validation_points={(1, 256): training[(1, 256)]},
        )


def test_fit_rejects_rank_deficient_grid_without_writing_profile(
    fit_environment, monkeypatch, tmp_path,
):
    def collinear_predictor(
        _cost_model,
        pipeline_config,
        _hardware_dict,
        compute_eff=1.0,
        memory_eff=1.0,
        bits="bf16",
        per_call_overhead_s=0.0,
    ):
        del bits
        module = pipeline_config["node_lines"][0]["nodes"][0]["modules"][0]
        bs = int(module["batch_size"])
        sl = int(pipeline_config["_token_stats"]["avg_chunk_tokens"])
        return (bs + sl) * (1.0 / compute_eff + 1.0 / memory_eff)

    monkeypatch.setattr(predict_module, "predict_latency", collinear_predictor)
    training = {
        (1, 256): 257.0 * (1.0 / 0.6 + 1.0 / 0.6),
        (64, 16): 80.0 * (1.0 / 0.6 + 1.0 / 0.6),
        (16, 64): 80.0 * (1.0 / 0.6 + 1.0 / 0.6),
    }

    with pytest.raises(RuntimeError, match="not separately identifiable"):
        calibration.fit_and_save(
            _CostModel(),
            object(),
            fit_environment,
            training,
            component="agnostic",
            output_dir=tmp_path,
        )

    assert not (tmp_path / "test_gpu.json").exists()


def test_load_fails_closed_on_legacy_model_keyed_intercept(tmp_path):
    payload = {
        "component": "agnostic",
        "model_key": "synthetic_model",
        "hardware_key": "test_gpu",
        "compute_efficiency": 0.55,
        "memory_efficiency": 0.78,
        "per_call_overhead_s": 0.01,
        "calibration_contract": (
            calibration.ZERO_INTERCEPT_ROOFLINE_CONTRACT
        ),
        "fit_mape": 0.02,
        "correlation": 0.99,
        "bits": "bf16",
        "grid": [],
    }
    (tmp_path / "test_gpu.json").write_text(json.dumps({
        "profiles": {"agnostic__synthetic_model__bf16": payload},
    }))
    calibration.invalidate_cache()

    with pytest.raises(ValueError, match="non-zero per_call_overhead_s"):
        calibration.load_record(
            "agnostic",
            "synthetic_model",
            "test_gpu",
            bits="bf16",
            directory=tmp_path,
        )


def test_legacy_auxiliary_internal_profile_remains_loadable(tmp_path):
    payload = {
        "component": "sentence_transformer_reranker",
        "model_key": "auxiliary_model",
        "hardware_key": "test_gpu",
        "compute_efficiency": 0.55,
        "memory_efficiency": 0.78,
        "per_call_overhead_s": 0.01,
        "fit_mape": 0.02,
        "correlation": 0.99,
        "bits": "f32",
        "grid": [],
    }
    (tmp_path / "test_gpu.json").write_text(json.dumps({
        "profiles": {
            "sentence_transformer_reranker__auxiliary_model__f32": payload,
        },
    }))
    calibration.invalidate_cache()

    record = calibration.load_record(
        "sentence_transformer_reranker",
        "auxiliary_model",
        "test_gpu",
        bits="f32",
        directory=tmp_path,
    )
    assert record is not None
    assert record.calibration_contract == (
        calibration.LEGACY_INTERNAL_COMPONENT_CONTRACT
    )
    assert record.per_call_overhead_s == pytest.approx(0.01)


def test_save_revalidates_mutated_record_before_writing(tmp_path):
    record = calibration.CalibrationRecord(
        component="agnostic",
        model_key="synthetic_model",
        hardware_key="test_gpu",
        compute_efficiency=0.55,
        memory_efficiency=0.78,
        bits="bf16",
        calibration_contract=calibration.ZERO_INTERCEPT_ROOFLINE_CONTRACT,
        identifiability_rank=2,
        identifiability_condition=2.0,
        validation_mape=0.01,
        validation_grid=[{
            "batch_size": 1,
            "seq_len": 1,
            "measured_s": 1.0,
            "predicted_s": 1.0,
            "rel_err": 0.0,
        }],
    )
    record.per_call_overhead_s = 0.01

    with pytest.raises(ValueError, match="non-zero per_call_overhead_s"):
        calibration.save_record(record, tmp_path)
    assert not (tmp_path / "test_gpu.json").exists()


def test_roofline_record_rejects_training_mape_above_publication_gate():
    with pytest.raises(ValueError, match="training MAPE exceeds 20%"):
        calibration.CalibrationRecord(
            component="agnostic",
            model_key="synthetic_model",
            hardware_key="test_gpu",
            compute_efficiency=0.55,
            memory_efficiency=0.78,
            fit_mape=0.201,
            bits="bf16",
            calibration_contract=(
                calibration.ZERO_INTERCEPT_ROOFLINE_CONTRACT
            ),
            identifiability_rank=2,
            identifiability_condition=2.0,
            validation_mape=0.01,
            validation_grid=[{
                "batch_size": 1,
                "seq_len": 1,
                "measured_s": 1.0,
                "predicted_s": 1.0,
                "rel_err": 0.0,
            }],
        )


def test_generic_component_grid_reserves_small_disjoint_holdout() -> None:
    evidence = {(batch, 128): float(batch) for batch in (1, 2, 4, 8, 16, 32)}

    training, validation = calibration.split_fit_validation_evidence(evidence)

    assert set(training).isdisjoint(validation)
    assert {**training, **validation} == evidence
    assert len(validation) == 2
    assert (32, 128) in validation
