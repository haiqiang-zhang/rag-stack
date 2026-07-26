"""Config-backed FAISS CPU spec registry.

Built-in CPU presets live in ``cpu_specs.py`` plus each FAISS algorithm's
``system_presets.py``. For a new machine, a calibration config may instead
define an inline CPU spec:

    configs/calibrate/my_box.yaml
      system:
        cpu:
          hardware_key: my_box
          ...

Important boundary: ``configs/calibrate`` is input-only. It carries hardware
specs and stable keys. Generated profiles are always written under
``calibration_profiles`` by the calibrators and are not managed here.

This module lets runtime code resolve that ``hardware_key`` across processes
without requiring users to edit generated profile files.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from rag_stack.types import SystemConfig as GlobalSystemConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_CALIBRATE_DIR = _REPO_ROOT / "configs" / "calibrate"


def _iter_calibrate_configs() -> list[Path]:
    if not CONFIG_CALIBRATE_DIR.exists():
        return []
    return sorted(
        p for p in CONFIG_CALIBRATE_DIR.iterdir()
        if p.is_file() and p.suffix in {".yaml", ".yml"}
    )


@lru_cache(maxsize=1)
def _configured_cpu_systems() -> dict[str, tuple[GlobalSystemConfig, str]]:
    """Return inline CPU specs declared under ``configs/calibrate/*.yaml``.

    Only configs whose ``system.cpu`` is a dict with ``hardware_key`` are part of
    this registry. Bare preset strings such as ``cpu: epyc_9124`` keep using the
    built-in preset tables. Profile JSON files under ``calibration_profiles`` are
    produced by calibration commands and are intentionally not scanned here.
    """
    from rag_stack.cost_model.calibration_config import load_calibration_config

    systems: dict[str, tuple[GlobalSystemConfig, str]] = {}
    for path in _iter_calibrate_configs():
        cfg = load_calibration_config(str(path))
        sysd = cfg.get("system") or {}
        cpu = sysd.get("cpu")
        if not isinstance(cpu, dict):
            continue
        key = str(cpu.get("hardware_key") or "")
        if not key:
            continue
        if key in systems:
            prev = systems[key][1]
            raise ValueError(
                f"duplicate FAISS CPU hardware_key {key!r} in {prev} and {path}"
            )
        systems[key] = (GlobalSystemConfig.from_dict(sysd), str(path))
    return systems


def configured_cpu_system(cpu_key: str) -> GlobalSystemConfig | None:
    item = _configured_cpu_systems().get(cpu_key)
    return item[0] if item else None


def configured_cpu_system_source(cpu_key: str) -> str | None:
    item = _configured_cpu_systems().get(cpu_key)
    return item[1] if item else None


def register_configured_faiss_preset(
    component: str,
    cpu_key: str,
    system_config: GlobalSystemConfig | None = None,
) -> bool:
    """Register a config-backed CPU spec into IVF/HNSW ``SYSTEM_PRESETS``.

    Returns True when ``cpu_key`` is now available. Built-in presets also return
    True and are left untouched.
    """
    if component == "ivf":
        from rag_stack.cost_model.faiss_ivf_sim.system_presets import SYSTEM_PRESETS
        from rag_stack.cost_model.faiss_ivf_sim.systems import system_config_to_faiss_hardware
    elif component == "hnsw":
        from rag_stack.cost_model.faiss_hnsw_sim.system_presets import SYSTEM_PRESETS
        from rag_stack.cost_model.faiss_hnsw_sim.hardware import system_config_to_faiss_hardware
    else:
        raise ValueError(f"unknown FAISS component {component!r}")

    if cpu_key in SYSTEM_PRESETS:
        if system_config is not None and not system_config.cpu_preset:
            from rag_stack.cost_model.cpu_specs import CPU_SPECS
            if cpu_key in CPU_SPECS:
                raise ValueError(
                    f"system.cpu.hardware_key {cpu_key!r} is a built-in CPU preset; "
                    "use the bare preset string or choose a new hardware_key."
                )
        return True

    hw = system_config or configured_cpu_system(cpu_key)
    if hw is None:
        return False

    def factory(**kwargs):
        return system_config_to_faiss_hardware(hw, **kwargs)

    SYSTEM_PRESETS[cpu_key] = factory
    return True


def require_configured_faiss_preset(
    component: str,
    cpu_key: str,
    system_config: GlobalSystemConfig | None = None,
) -> None:
    if register_configured_faiss_preset(component, cpu_key, system_config):
        return
    raise KeyError(
        f"unknown FAISS {component} CPU preset {cpu_key!r}. Add a built-in preset "
        f"or define system.cpu.hardware_key: {cpu_key} in configs/calibrate/*.yaml."
    )
