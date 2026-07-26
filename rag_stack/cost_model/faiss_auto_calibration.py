"""Read-only FAISS system-profile locators for benchmark tools.

This module intentionally handles only host/system artifacts:

    calibration_profiles/faiss_<component>/<system>.json

It does not create data/workload artifacts such as IVF imbalance profiles or
HNSW per-corpus+embedding workload predictors.
"""

from __future__ import annotations

import os

from rag_stack.calibration_paths import system_profile_path


def faiss_system_profile_path(component: str, system: str) -> str:
    if component not in {"ivf", "hnsw"}:
        raise ValueError(f"unknown FAISS component {component!r}")
    return str(system_profile_path(f"faiss_{component}", system))


def faiss_system_from_calibration_config(component: str, config_path: str) -> str:
    """Read/register the FAISS CPU system key from a calibration YAML.

    ``system.cpu`` may be either a built-in preset string or an inline CPU spec
    carrying ``hardware_key``. Inline specs are registered into the selected
    FAISS component's preset table for this process.
    """
    from rag_stack.cost_model.calibration_config import load_calibration_config
    from rag_stack.cost_model.faiss_system_registry import require_configured_faiss_preset
    from rag_stack.types import SystemConfig as GlobalSystemConfig

    cfg = load_calibration_config(config_path, require_cpu=True)
    sysd = cfg.get("system") or {}
    cpu = sysd.get("cpu")
    system_config = GlobalSystemConfig.from_dict(sysd)
    if isinstance(cpu, dict):
        key = str(cpu.get("hardware_key") or "")
        if not key:
            raise ValueError(f"{config_path}: inline system.cpu needs hardware_key")
        require_configured_faiss_preset(component, key, system_config)
        return key
    key = str(cpu or "")
    require_configured_faiss_preset(component, key)
    return key


def require_faiss_system_profile(component: str, system: str) -> str:
    """Return an existing exact profile; never measure or write on miss."""
    from rag_stack.cost_model.faiss_system_registry import require_configured_faiss_preset
    require_configured_faiss_preset(component, system)

    profile = faiss_system_profile_path(component, system)
    if os.path.exists(profile):
        return profile
    raise FileNotFoundError(
        f"missing FAISS {component.upper()} system profile for {system!r}: "
        f"{profile}. Run `rag_stack calibrate status --config <hardware.yaml> "
        f"--scope {component}` and perform the staged measure/fit workflow."
    )
