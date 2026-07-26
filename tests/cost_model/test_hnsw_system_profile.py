"""CPU-only contract tests for exact-CPU HNSW system profiles."""

from __future__ import annotations

from tests import REPO_ROOT

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from rag_stack.calibration_inventory import build_calibration_inventory
from rag_stack.cost_model.faiss_hnsw_sim.coefficients import (
    HNSWProfileError,
    build_hardware_from_profile,
    load_profile,
    load_profile_for_preset,
    save_profile,
)
from rag_stack.cost_model.faiss_hnsw_sim.adapter import FaissHNSWCostModel
from rag_stack.cost_model.faiss_hnsw_sim.cost_model import HNSWCostModel
from rag_stack.cost_model.faiss_hnsw_sim.hardware import (
    system_config_to_faiss_hardware,
)
from rag_stack.cost_model.faiss_hnsw_sim.hardware_calibrator import (
    HNSWHardwareCalibrator,
)
from rag_stack.cost_model.faiss_hnsw_sim.hnsw_config import HNSWConfig
from rag_stack.cost_model.faiss_hnsw_sim.system_presets import (
    cpu_amd_epyc_9124,
)
from rag_stack.cost_model.faiss_hnsw_sim.workload_predictor import (
    WorkloadProfile,
)
from rag_stack.hnsw_profile_schema import hnsw_profile_validation_error
from rag_stack.types import SystemConfig




def _document(cpu_key: str = "epyc_9124") -> dict:
    path = (
        REPO_ROOT / "calibration_profiles" / "faiss_hnsw" / f"{cpu_key}.json"
    )
    return json.loads(path.read_text())


def _custom_7742_system() -> SystemConfig:
    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "calibrate" / "sgs_a100_8x.yaml").read_text()
    )
    return SystemConfig.from_dict(config["system"])


def _pipeline_config() -> dict:
    return {
        "dataset": {"dataset_name": "fixture"},
        "vectordb": [{
            "name": "hnsw",
            "N": 100_000,
            "embedding_dim": 128,
            "M": 16,
            "ef_construction": 200,
        }],
        "node_lines": [{
            "nodes": [{
                "stage": "semantic_retrieval",
                "top_k": 10,
                "modules": [{
                    "vectordb": "hnsw",
                    "ef_search": 64,
                    "batch_size": 1,
                }],
            }],
        }],
        "system": {"retrieval": {"faiss_num_threads": 1}},
    }


@pytest.mark.parametrize(
    ("cpu_key", "expected_hardware", "expected_coefficients", "expected_fit"),
    [
        (
            "epyc_9124",
            {
                "simd_efficiency": 0.5,
                "dram_random_access_latency_ns": 120.0,
                "scattered_bw_ceiling": 0.65,
                "numa_local_bytes": 0,
                "numa_remote_bw_ceiling": 0.06,
            },
            {
                "hops_per_upper_layer": 8.0,
                "cache_random_access_latency_multiplier": 1.4719063224636333,
                "cycles_per_visited_probe": 7.743702858648601,
                "cycles_per_heap_op": 48.93659658479141,
            },
            {
                "mape_latency_pct": 14.830426005910784,
                "max_ape_latency_pct": 66.39799474537547,
                "num_trials": 120,
                "dataset_configs": [
                    "glove", "cifar", "dragonball", "sift", "gist", "deep",
                ],
            },
        ),
        (
            "epyc_7742_2s",
            {
                "simd_efficiency": 0.55,
                "dram_random_access_latency_ns": 100.0,
                "scattered_bw_ceiling": 0.35,
                "numa_local_bytes": 0,
                "numa_remote_bw_ceiling": 0.06,
            },
            {
                "hops_per_upper_layer": 46.56989270691491,
                "cache_random_access_latency_multiplier": 5.460465841818943,
                "cycles_per_visited_probe": 48.662741781324804,
                "cycles_per_heap_op": 503.63916103121295,
            },
            {
                "mape_latency_pct": 20.054,
                "max_ape_latency_pct": 131.883,
                "num_trials": 120,
                "dataset_configs": [
                    "glove", "cifar", "dragonball", "sift", "gist", "deep",
                ],
            },
        ),
    ],
)
def test_migrated_profile_contains_every_previous_runtime_value(
    cpu_key, expected_hardware, expected_coefficients, expected_fit,
):
    profile = load_profile_for_preset(cpu_key)
    assert (
        profile.hardware_model_parameters.to_hardware_kwargs()
        == expected_hardware
    )
    document = profile.to_document()
    assert set(document) == {
        "profile_kind",
        "hardware_key",
        "hardware_model_parameters",
        "empirical_coefficients",
        "fit_metadata",
    }
    assert set(document["fit_metadata"]) == {
        "mape_latency_pct",
        "max_ape_latency_pct",
        "num_trials",
        "dataset_configs",
    }
    assert document["empirical_coefficients"] == expected_coefficients
    assert document["fit_metadata"] == expected_fit
    assert hnsw_profile_validation_error(document, cpu_key) is None


@pytest.mark.parametrize("cpu_key", ["epyc_9124", "epyc_7742_2s"])
def test_profile_driven_hardware_preserves_pre_migration_prediction(cpu_key):
    profile = load_profile_for_preset(cpu_key)
    if cpu_key == "epyc_9124":
        global_system = SystemConfig(cpu_preset=cpu_key)
        previous_hardware = cpu_amd_epyc_9124(active_threads=1)
    else:
        global_system = _custom_7742_system()
        previous_hardware = system_config_to_faiss_hardware(
            global_system,
            active_threads=1,
            **profile.hardware_model_parameters.to_hardware_kwargs(),
        )
    profile_hardware = build_hardware_from_profile(
        profile,
        active_threads=1,
        system_config=global_system,
    )

    config = HNSWConfig(
        N=1_000_000,
        d=128,
        M=16,
        ef_search=64,
        ef_construction=200,
        k=10,
        batch_size=1,
    )
    workload = WorkloadProfile(ndis_per_query=1400.0, nhops_per_query=80.0)
    previous = HNSWCostModel(
        config,
        previous_hardware,
        workload,
        coefficients=profile.empirical_coefficients,
    ).estimate()
    migrated = HNSWCostModel(
        config,
        profile_hardware,
        workload,
        coefficients=profile.empirical_coefficients,
    ).estimate()
    assert migrated.latency_ms == previous.latency_ms


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(obsolete_field=0), "extra"),
        (
            lambda d: d["hardware_model_parameters"].pop("simd_efficiency"),
            "missing",
        ),
        (
            lambda d: d["empirical_coefficients"].update(invented=1.0),
            "extra",
        ),
        (
            lambda d: d.update(provenance={"source_path": "legacy.parquet"}),
            "extra",
        ),
        (
            lambda d: d["fit_metadata"].update(
                fitted_at="2026-07-16T00:00:00Z"
            ),
            "extra",
        ),
        (
            lambda d: d.update(evidence_scope="standalone_component"),
            "extra",
        ),
    ],
)
def test_loader_rejects_legacy_missing_unknown_and_mismatched_fields(
    tmp_path, mutate, message,
):
    document = copy.deepcopy(_document())
    mutate(document)
    path = tmp_path / "epyc_9124.json"
    path.write_text(json.dumps(document))
    with pytest.raises(HNSWProfileError, match=message):
        load_profile(path, expected_hardware_key="epyc_9124")


def test_loader_rejects_requested_cpu_identity_mismatch(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_document()))
    with pytest.raises(HNSWProfileError, match="does not match requested CPU"):
        load_profile(path, expected_hardware_key="epyc_7742_2s")


def test_custom_hardware_converter_has_no_generic_algorithm_defaults():
    with pytest.raises(ValueError, match="missing .*simd_efficiency"):
        system_config_to_faiss_hardware(_custom_7742_system())


def test_new_custom_calibrator_requires_all_explicit_parameters(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("RAG_STACK_CALIBRATION_PROFILE_DIR", str(tmp_path))
    calibrator = HNSWHardwareCalibrator(system="new_custom_cpu")
    with pytest.raises(HNSWProfileError, match="generic algorithm defaults"):
        calibrator._resolve_hardware_model_parameters()

    explicit = {
        "simd_efficiency": 0.4,
        "dram_random_access_latency_ns": 90.0,
        "scattered_bw_ceiling": 0.3,
        "numa_local_bytes": 0,
        "numa_remote_bw_ceiling": 0.05,
    }
    configured = HNSWHardwareCalibrator(
        system="new_custom_cpu",
        hardware_model_parameters=explicit,
    )
    assert (
        configured._resolve_hardware_model_parameters().to_hardware_kwargs()
        == explicit
    )


def test_calibrator_does_not_hide_existing_legacy_profile_with_preset_seed(
    tmp_path, monkeypatch,
):
    profile = tmp_path / "faiss_hnsw" / "epyc_9124.json"
    profile.parent.mkdir(parents=True)
    profile.write_text(json.dumps({"hardware_preset": "epyc_9124"}))
    monkeypatch.setenv("RAG_STACK_CALIBRATION_PROFILE_DIR", str(tmp_path))
    calibrator = HNSWHardwareCalibrator(system="epyc_9124")
    with pytest.raises(HNSWProfileError, match="fields mismatch"):
        calibrator._resolve_hardware_model_parameters()


def test_adapter_fails_before_workload_resolution_when_exact_profile_is_missing(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("RAG_STACK_CALIBRATION_PROFILE_DIR", str(tmp_path))
    adapter = FaissHNSWCostModel()
    monkeypatch.setattr(
        adapter,
        "_resolve_predictor",
        lambda *args: pytest.fail("workload predictor should not be resolved"),
    )
    with pytest.raises(HNSWProfileError, match="missing HNSW CPU profile"):
        adapter.estimate(
            _pipeline_config(), SystemConfig(cpu_preset="unprofiled_cpu")
        )


def test_adapter_applies_profile_values_instead_of_preset_defaults(
    tmp_path, monkeypatch,
):
    profile_root = tmp_path / "profiles"
    path = profile_root / "faiss_hnsw" / "epyc_9124.json"
    path.parent.mkdir(parents=True)
    document = _document()
    document["hardware_model_parameters"]["simd_efficiency"] = 0.25
    path.write_text(json.dumps(document))
    monkeypatch.setenv("RAG_STACK_CALIBRATION_PROFILE_DIR", str(profile_root))

    adapter = FaissHNSWCostModel()
    monkeypatch.setattr(
        adapter,
        "_resolve_predictor",
        lambda *args: SimpleNamespace(
            predict=lambda *unused: WorkloadProfile(
                ndis_per_query=1000.0, nhops_per_query=50.0,
            )
        ),
    )
    captured = {}

    class CapturingModel:
        def __init__(self, config, hardware, workload, coefficients):
            captured["hardware"] = hardware
            captured["coefficients"] = coefficients

        def estimate(self):
            return SimpleNamespace(
                total_time_s=0.001, throughput_qps=1000.0, operators=[],
            )

    monkeypatch.setattr(
        "rag_stack.cost_model.faiss_hnsw_sim.adapter.HNSWCostModel",
        CapturingModel,
    )
    adapter.estimate(_pipeline_config(), SystemConfig(cpu_preset="epyc_9124"))
    assert captured["hardware"].simd_efficiency == 0.25
    assert captured["hardware"].dram_random_access_latency_ns == 120.0
    assert captured["coefficients"].hops_per_upper_layer == 8.0


def test_save_profile_is_atomic_and_keeps_exact_schema(tmp_path):
    profile = load_profile_for_preset("epyc_9124")
    target = tmp_path / "epyc_9124.json"
    target.write_text("{}\n")
    target.chmod(0o640)
    save_profile(profile, target)
    assert target.stat().st_mode & 0o777 == 0o640
    assert json.loads(target.read_text()) == profile.to_document()
    assert not list(tmp_path.glob(".epyc_9124.json.*.tmp"))


def test_calibrator_profile_captures_hardware_values_it_used(monkeypatch):
    calibrator = HNSWHardwareCalibrator(system="epyc_9124", maxiter=1)
    frozen = pd.DataFrame([
        {
            "dataset": "fixture",
            "N": 100_000,
            "d": 128,
            "M": 16,
            "ef_search": 64,
            "ef_construction": 200,
            "batch_size": 1,
            "k": 10,
            "ndis_per_query": 1000.0,
            "nhops_per_query": 50.0,
            "latency_ms": 1.0,
        }
    ])
    best = np.array([2.0, 3.0, 40.0, 75.0])
    monkeypatch.setattr(
        "rag_stack.cost_model.faiss_hnsw_sim.hardware_calibrator."
        "differential_evolution",
        lambda *args, **kwargs: SimpleNamespace(x=best),
    )
    monkeypatch.setattr(
        calibrator,
        "_predict_ms",
        lambda point, coefficients, hardware: float(point.latency_ms),
    )

    profile, _ = calibrator._fit(frozen)
    applied = calibrator._hw(profile.hardware_model_parameters)
    assert profile.hardware_model_parameters.to_hardware_kwargs() == {
        "simd_efficiency": applied.simd_efficiency,
        "dram_random_access_latency_ns": applied.dram_random_access_latency_ns,
        "scattered_bw_ceiling": applied.scattered_bw_ceiling,
        "numa_local_bytes": applied.numa_local_bytes,
        "numa_remote_bw_ceiling": applied.numa_remote_bw_ceiling,
    }
    assert profile.fit_metadata.mape_latency_pct == 0.0
    assert profile.fit_metadata.max_ape_latency_pct == 0.0
    assert profile.fit_metadata.num_trials == len(frozen)
    assert profile.fit_metadata.dataset_configs == (
        "glove",
        "cifar",
        "dragonball",
        "sift",
        "gist",
        "deep",
    )


def test_inventory_uses_same_strict_hnsw_schema(tmp_path):
    config = tmp_path / "system.yaml"
    config.write_text("system:\n  cpu: epyc_9124\n")
    profile_root = tmp_path / "profiles"
    profile_path = profile_root / "faiss_hnsw" / "epyc_9124.json"
    profile_path.parent.mkdir(parents=True)
    document = _document()
    document["hardware_model_parameters"]["invented"] = 1.0
    profile_path.write_text(json.dumps(document))

    inventory = build_calibration_inventory(
        config,
        data_dir=tmp_path / "data",
        profile_dir=profile_root,
        scopes=("hnsw",),
    )
    assert inventory.items[0].state == "invalid"
    assert "extra=['invented']" in inventory.items[0].detail
