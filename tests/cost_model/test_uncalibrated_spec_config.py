from __future__ import annotations

from tests import REPO_ROOT

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rag_stack.cli import (
    _add_optimize_args,
    _apply_optimize_config_overrides,
)
from rag_stack.config_validator import ConfigValidator
from rag_stack.cost_model.calibration_mode import resolve_calibration_mode
from rag_stack.types import SystemConfig


def test_system_config_round_trip_preserves_complete_rag_cm_mapping() -> None:
    source = {
        "system_hardware": "new_host",
        "cpu": "epyc_9124",
        "gpu": "H100_GPU",
        "rag_cm": {
            "calibration_mode": "uncalibrated_spec",
        },
    }

    hardware = SystemConfig.from_dict(source)
    assert hardware.rag_cm == source["rag_cm"]

    # The hardware binding owns its copy; neither input nor serialized output
    # may mutate a resumed replay's policy by aliasing nested dictionaries.
    source["rag_cm"]["calibration_mode"] = "strict"
    assert hardware.rag_cm["calibration_mode"] == "uncalibrated_spec"

    serialized = hardware.to_dict()
    assert serialized["rag_cm"] == {
        "calibration_mode": "uncalibrated_spec",
    }
    serialized["rag_cm"]["calibration_mode"] = "strict"
    assert hardware.rag_cm["calibration_mode"] == "uncalibrated_spec"


def _system_only_errors(rag_cm) -> list[str]:
    config = {"system": {"cpu": "epyc_9124"}}
    if rag_cm is not None:
        config["system"]["rag_cm"] = rag_cm
    return ConfigValidator(
        config, check_model_cache=False,
    ).validate_system_only(require_cpu=True)


def test_rag_cm_calibration_mode_defaults_to_strict_and_rejects_store_keys() -> None:
    assert _system_only_errors(None) == []
    assert _system_only_errors({"calibration_mode": "strict"}) == []
    for field in ("component_calibration_store", "calibration_store"):
        errors = _system_only_errors({field: "/tmp/forbidden-profile-root"})
        assert len(errors) == 1
        assert f"[system.rag_cm.{field}]" in errors[0]
        assert "cost_model" in errors[0]
        assert "RAG_STACK_CALIBRATION_PROFILE_DIR" in errors[0]


def test_runtime_rejects_store_keys_even_without_global_validation() -> None:
    for field in ("component_calibration_store", "calibration_store"):
        with pytest.raises(ValueError, match="unsupported by cost_model"):
            resolve_calibration_mode({
                "system": {"rag_cm": {field: "/tmp/forbidden-profile-root"}},
            })


def test_shipped_configs_do_not_select_calibration_profile_roots() -> None:
    root = REPO_ROOT / "configs"
    forbidden_fields = {"component_calibration_store", "calibration_store"}
    for path in sorted(root.rglob("*.yaml")):
        document = yaml.safe_load(path.read_text()) or {}
        system = document.get("system")
        rag_cm = system.get("rag_cm") if isinstance(system, dict) else None
        if isinstance(rag_cm, dict):
            assert forbidden_fields.isdisjoint(rag_cm), path


def test_rag_cm_calibration_mode_rejects_bad_shape_or_value() -> None:
    assert _system_only_errors("uncalibrated_spec") == [
        "[system.rag_cm] must be a mapping."
    ]
    assert _system_only_errors({"calibration_mode": "best_effort"}) == [
        "[system.rag_cm.calibration_mode] must be 'strict' or "
        "'uncalibrated_spec'."
    ]
    assert _system_only_errors({"calibration_mode": True}) == [
        "[system.rag_cm.calibration_mode] must be 'strict' or "
        "'uncalibrated_spec'."
    ]


def test_optimize_parser_accepts_only_declared_calibration_modes() -> None:
    parser = argparse.ArgumentParser()
    _add_optimize_args(parser)
    args = parser.parse_args([
        "--cm-calibration-mode", "uncalibrated_spec",
    ])
    assert args.cm_calibration_mode == "uncalibrated_spec"


def test_optimize_parser_keeps_refinement_optional_and_non_negative() -> None:
    parser = argparse.ArgumentParser()
    _add_optimize_args(parser)

    assert parser.parse_args([]).n_refine_iterations is None
    assert parser.parse_args([
        "--n-refine-iterations", "10",
    ]).n_refine_iterations == 10
    with pytest.raises(SystemExit):
        parser.parse_args(["--n-refine-iterations", "-1"])


def test_cli_override_persists_mode(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "rag_stack_config.yaml"
    config_path.write_text(yaml.safe_dump({
        "system": {
            "performance_source": "cost_model",
            "rag_cm": {"calibration_mode": "strict"},
        },
    }, sort_keys=False))

    _apply_optimize_config_overrides(
        str(config_path),
        SimpleNamespace(
            performance_source=None,
            cm_calibration_mode="uncalibrated_spec",
        ),
    )
    updated = yaml.safe_load(config_path.read_text())
    assert updated["system"]["rag_cm"] == {
        "calibration_mode": "uncalibrated_spec",
    }

    _apply_optimize_config_overrides(
        str(config_path),
        SimpleNamespace(
            performance_source="measured",
            cm_calibration_mode="strict",
        ),
    )
    updated = yaml.safe_load(config_path.read_text())
    assert updated["system"]["performance_source"] == "measured"
    assert updated["system"]["rag_cm"]["calibration_mode"] == "strict"
