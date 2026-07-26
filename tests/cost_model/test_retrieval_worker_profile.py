"""Strict CPU-keyed retrieval-worker calibration contract."""

from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy

import pytest

from rag_stack.calibration_paths import calibration_profile_root
from rag_stack.cost_model.retrieval_worker_profile import (
    RetrievalWorkerProfileError,
    load_rrf_fusion_profile,
)
from rag_stack.cost_model.sweep_inputs import rrf_fusion_per_user_row_s
from rag_stack.types import SystemConfig


def _document(cpu_key: str = "test_cpu") -> dict:
    grid = [
        {
            "k": k,
            "top_k": top_k,
            "overlap": overlap,
            "items": k * top_k,
            "seconds": 1e-3 + 1e-6 * k * top_k,
        }
        for k in (2, 3, 4, 6)
        for top_k in (4, 8, 16, 32, 64)
        for overlap in (0.0, 0.5)
    ]
    return {
        "profile_kind": "retrieval_worker",
        "hardware_key": cpu_key,
        "components": {
            "rrf_fusion": {
                "operator": (
                    "rag_stack_evaluator.static_rag_evaluator.nodes.hybridretrieval."
                    "hybrid_rrf.rrf_pure"
                ),
                "service_unit": "seconds_per_user_row",
                "demand_axis": "input_items",
                "fit_form": "nonnegative_affine",
                "equation": (
                    "service_s = per_call_s + per_item_s * "
                    "(fanout * top_k)"
                ),
                "per_call_s": 1e-3,
                "per_item_s": 1e-6,
                "fit_mape": 0.0,
                "grid": grid,
            },
        },
    }


def _write(tmp_path, document: dict):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(document))
    return path


def _pipeline(qe_component=None, top_k=16):
    nodes = []
    if qe_component:
        nodes.append({
            "stage": "query_expansion",
            "modules": [{"component": qe_component}],
        })
    nodes.append({
        "stage": "semantic_retrieval",
        "top_k": top_k,
        "modules": [{"component": "vectordb"}],
    })
    return {"node_lines": [{"nodes": nodes}]}


def test_exact_profile_load_needs_only_deployable_profile(
    tmp_path,
    monkeypatch,
):
    missing_raw_root = tmp_path / "raw-calibration-tree-does-not-exist"
    monkeypatch.setenv(
        "RAG_STACK_CALIBRATION_DATA_DIR",
        str(missing_raw_root),
    )
    profile = load_rrf_fusion_profile(
        "test_cpu", profile_path=_write(tmp_path, _document()),
    )
    assert not missing_raw_root.exists()
    assert profile.per_call_s == 1e-3
    assert profile.per_item_s == 1e-6
    assert profile.service_s(64) == pytest.approx(1.064e-3)


def test_checked_in_profile_matches_the_runtime_contract():
    document = json.loads(
        (
            calibration_profile_root()
            / "retrieval_worker"
            / "epyc_9124.json"
        ).read_text()
    )
    assert set(document) == {"profile_kind", "hardware_key", "components"}
    component = document["components"]["rrf_fusion"]
    assert component["equation"] == (
        "service_s = per_call_s + per_item_s * (fanout * top_k)"
    )
    assert component["fit_mape"] == pytest.approx(0.0844)
    assert {
        (row["k"], row["top_k"], row["overlap"])
        for row in component["grid"]
    } == {
        (k, top_k, overlap)
        for k in (2, 3, 4, 6)
        for top_k in (4, 8, 16, 32, 64)
        for overlap in (0.0, 0.5)
    }

    profile = load_rrf_fusion_profile("epyc_9124")
    assert profile.per_call_s == pytest.approx(0.00091323)
    assert profile.per_item_s == pytest.approx(0.0000033252)
    assert profile.service_s(256) > profile.service_s(8)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("profile_kind",), "assembly"),
        (("hardware_key",), "other_cpu"),
        (("components", "rrf_fusion", "service_unit"), "seconds_per_batch"),
        (("components", "rrf_fusion", "demand_axis"), "measured_batch"),
        (("components", "rrf_fusion", "fit_form"), "e2e_scale"),
        (("components", "rrf_fusion", "equation"), "service_s = 0"),
        (("components", "rrf_fusion", "per_call_s"), -1.0),
        (("components", "rrf_fusion", "per_item_s"), "nan"),
        (("components", "rrf_fusion", "fit_mape"), -1.0),
    ],
)
def test_invalid_or_mismatched_profiles_fail_closed(tmp_path, path, value):
    document = deepcopy(_document())
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(RetrievalWorkerProfileError):
        load_rrf_fusion_profile(
            "test_cpu", profile_path=_write(tmp_path, document),
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("evidence_scope",), "standalone_component_only_no_rag_e2e_gt"),
        (("source_path",), "calibration_data/retrieval_worker/raw.json"),
        (("timestamp",), "2026-07-11T00:00:00+00:00"),
        (("components", "rrf_fusion", "provenance"), {"source": "raw"}),
    ],
)
def test_legacy_provenance_fields_are_rejected(tmp_path, path, value):
    document = deepcopy(_document())
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(RetrievalWorkerProfileError):
        load_rrf_fusion_profile(
            "test_cpu", profile_path=_write(tmp_path, document),
        )


def test_missing_profile_and_missing_cpu_identity_fail_closed(tmp_path):
    with pytest.raises(RetrievalWorkerProfileError, match="missing"):
        load_rrf_fusion_profile(
            "test_cpu", profile_path=tmp_path / "missing.json",
        )
    with pytest.raises(RetrievalWorkerProfileError, match="hardware_key"):
        load_rrf_fusion_profile("")


def test_rrf_charge_requires_observation_instead_of_structural_fanout():
    profile = load_rrf_fusion_profile("epyc_9124")
    hardware = SystemConfig(cpu_preset="epyc_9124")
    for component in (None, "hyde", "multi_query_expansion", "query_decompose"):
        assert rrf_fusion_per_user_row_s(_pipeline(component), hardware) == 0.0
    config = _pipeline("multi_query_expansion")
    config["_evaluator_fusion_calls"] = [{
        "stage": "retrieval_fusion", "model_id": "rrf_pure",
        "input_profile": {"items": 11, "list_sizes": [7, 4]},
        "output_profile": {"items": 5, "unique_items": 5},
    }]
    assert rrf_fusion_per_user_row_s(config, hardware) == pytest.approx(profile.service_s(11))


def test_production_cost_model_registry_import_does_not_read_profiles(
    tmp_path,
    monkeypatch,
):
    profile_root = tmp_path / "empty_profiles"
    profile_root.mkdir()
    monkeypatch.setenv("RAG_STACK_CALIBRATION_PROFILE_DIR", str(profile_root))
    script = """
import os
import sys
from pathlib import Path

profile_root = Path(os.environ['RAG_STACK_CALIBRATION_PROFILE_DIR']).resolve()
profile_reads = []

def record_profile_open(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).resolve()
    if path.is_relative_to(profile_root):
        profile_reads.append(str(path))

# Audit attempted opens, including missing-file errors swallowed by discovery.
sys.addaudithook(record_profile_open)
import rag_stack.cost_model
from rag_stack.cost_model.dispatch import validate_against_registry
validate_against_registry()
assert not profile_reads, profile_reads
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_system_config_has_no_assembly_coefficient_override_surface():
    hardware = SystemConfig.from_dict({
        "cpu": "epyc_9124",
        "assembly": {"obsolete": 99.0},
        "serving": {"obsolete": 99.0},
    })
    assert not hasattr(hardware, "assembly")
    assert not hasattr(hardware, "serving")
    assert "assembly" not in hardware.to_dict()
    assert "serving" not in hardware.to_dict()
