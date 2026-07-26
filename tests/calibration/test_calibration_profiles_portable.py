"""Repository-wide portability contract for deployable CM profiles."""

from __future__ import annotations

from tests import REPO_ROOT

import json
from pathlib import Path
from typing import Any


PROFILE_ROOT = REPO_ROOT / "calibration_profiles"

# These fields bind a pricing profile to the machine, checkout, campaign, or
# raw files that produced it.  A deployable profile is selected by its exact
# runtime key and validated from its self-contained coefficients/diagnostics.
LEGACY_PROVENANCE_FIELDS = frozenset({
    "provenance",
    "provenance_sha256",
    "profile_provenance_sha256",
    "measurement_evidence",
    "single_role_measurement_evidence",
    "evidence_scope",
    "source_path",
    "source_sha256",
    "raw_hardware_source",
    "raw_hardware_source_sha256",
    "source_kind",
    "calibration_runner",
    "calibration_runner_sha256",
    "source_commit",
    "measured_hardware_key",
    "measured_at",
    "manifest_path",
    "manifest_sha256",
    "raw_telemetry_path",
    "raw_telemetry_sha256",
    "component_source_path",
    "component_source_sha256",
    "stage_contract_path",
    "stage_contract_sha256",
    "component_contract_path",
    "component_contract_sha256",
    "request_contract",
    "request_contract_sha256",
    "request_contracts",
    "request_contracts_sha256",
    "parameter_origins",
    "timestamp",
    "fitted_at",
    "cuda_name",
    "fit_method",
})


def _assert_no_provenance(value: Any, *, location: str) -> None:
    if isinstance(value, dict):
        forbidden = sorted(set(value).intersection(LEGACY_PROVENANCE_FIELDS))
        assert not forbidden, f"{location} contains retired fields: {forbidden}"
        for key, child in value.items():
            _assert_no_provenance(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_provenance(child, location=f"{location}[{index}]")


def test_every_serialized_calibration_profile_is_provenance_free() -> None:
    profile_paths = sorted(
        path
        for path in PROFILE_ROOT.rglob("*")
        if path.is_file() and ".json" in path.name
    )
    assert profile_paths
    for path in profile_paths:
        document = json.loads(path.read_text(encoding="utf-8"))
        _assert_no_provenance(
            document,
            location=str(path.relative_to(PROFILE_ROOT)),
        )


def test_hnsw_workload_profiles_have_only_runtime_models() -> None:
    import joblib

    paths = sorted((PROFILE_ROOT / "faiss_hnsw").glob("*.joblib"))
    assert paths
    for path in paths:
        document = joblib.load(path)
        assert set(document) == {"ndis_model", "nhops_model"}


def test_a100_pipeline_is_ready_with_profiles_and_no_calibration_data(
    tmp_path: Path,
) -> None:
    from rag_stack.calibration_inventory import build_calibration_inventory

    missing_data_root = tmp_path / "calibration-data-does-not-exist"
    config = (
        REPO_ROOT
        / "configs"
        / "system_transfer"
        / "baseline_from_scratch_sgs_a100x8_msmarco_agent_v2_s45.yaml"
    )
    inventory = build_calibration_inventory(
        config,
        data_dir=missing_data_root,
        profile_dir=PROFILE_ROOT,
        scopes=("pipeline",),
    )

    assert not missing_data_root.exists()
    assert inventory.deploy_ready
    assert inventory.items
    assert all(item.state == "ready" for item in inventory.items)
