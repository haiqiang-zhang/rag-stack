"""Shared config loading for calibration-only hardware specs."""

from __future__ import annotations

from rag_stack.config_validator import ConfigValidator
from rag_stack.utils.config import load_yaml_config


def load_calibration_config(
    config_path: str,
    *,
    require_cpu: bool = False,
    require_gpu: bool = False,
    require_available_gpus: bool = False,
) -> dict:
    """Load a calibration config with RAG-Stack YAML handling and validation.

    Calibration configs under ``configs/calibrate`` usually contain only a
    ``system:`` block. They intentionally use a system-only validator rather
    than the full optimizer-config validator.
    """
    cfg = load_yaml_config(config_path) or {}
    errors = ConfigValidator(cfg, check_model_cache=False).validate_system_only(
        require_cpu=require_cpu,
        require_gpu=require_gpu,
        require_available_gpus=require_available_gpus,
    )
    if errors:
        details = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"{config_path}: calibration config validation failed:\n{details}")
    return cfg
