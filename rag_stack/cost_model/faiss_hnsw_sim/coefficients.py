"""Strict exact-CPU profile persistence for the FAISS HNSW cost model.

System profiles live at ``calibration_profiles/faiss_hnsw/<cpu_key>.json``.
They contain every non-raw-hardware value consumed by production HNSW pricing:
five CPU-specific effective hardware-model parameters and four empirical graph-
search coefficients. Raw CPU facts remain in ``cpu_specs`` or the inline YAML
CPU block and are joined with the profile only after exact identity validation.

Only the current exact shape is accepted. There is no unknown-key filtering,
legacy backfill, cross-CPU fallback, or production use of dataclass/preset
algorithm defaults.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re as _re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from rag_stack.calibration_paths import (
    atomic_write_json,
    calibration_profile_root,
    system_profile_path,
)
from rag_stack.hnsw_profile_schema import (
    HNSW_PROFILE_KIND,
    hnsw_profile_validation_error,
    is_hnsw_hardware_key,
)

from .cost_model import HNSWCoefficients

logger = logging.getLogger(__name__)


class HNSWProfileError(ValueError):
    """An exact latest-schema HNSW CPU profile is missing or malformed."""


@dataclass(frozen=True)
class HNSWHardwareModelParameters:
    """CPU-specific effective HNSW inputs, distinct from raw hardware facts.

    These values affect the HNSW analytical model but are not manufacturer CPU
    specifications. In particular, the efficiency and bandwidth ceilings are
    effective model parameters.
    """

    simd_efficiency: float
    dram_random_access_latency_ns: float
    scattered_bw_ceiling: float
    numa_local_bytes: int
    numa_remote_bw_ceiling: float

    def __post_init__(self) -> None:
        for field in (
            "simd_efficiency",
            "scattered_bw_ceiling",
            "numa_remote_bw_ceiling",
        ):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= 1.0
            ):
                raise HNSWProfileError(f"{field} must be in (0, 1]")
        latency = self.dram_random_access_latency_ns
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(float(latency))
            or float(latency) <= 0.0
        ):
            raise HNSWProfileError(
                "dram_random_access_latency_ns must be finite and positive"
            )
        if (
            isinstance(self.numa_local_bytes, bool)
            or not isinstance(self.numa_local_bytes, int)
            or self.numa_local_bytes < 0
        ):
            raise HNSWProfileError(
                "numa_local_bytes must be a non-negative integer"
            )

    @classmethod
    def from_hardware(cls, hardware) -> "HNSWHardwareModelParameters":
        """Capture the exact values applied to a calibrator hardware object."""

        return cls(
            simd_efficiency=float(hardware.simd_efficiency),
            dram_random_access_latency_ns=float(
                hardware.dram_random_access_latency_ns
            ),
            scattered_bw_ceiling=float(hardware.scattered_bw_ceiling),
            numa_local_bytes=int(hardware.numa_local_bytes),
            numa_remote_bw_ceiling=float(hardware.numa_remote_bw_ceiling),
        )

    def to_hardware_kwargs(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class HNSWFitMetadata:
    mape_latency_pct: float
    max_ape_latency_pct: float
    num_trials: int
    dataset_configs: tuple[str, ...]


@dataclass(frozen=True)
class HNSWSystemProfile:
    """The complete non-raw-hardware HNSW profile for one exact CPU key."""

    hardware_key: str
    hardware_model_parameters: HNSWHardwareModelParameters
    empirical_coefficients: HNSWCoefficients
    fit_metadata: HNSWFitMetadata

    def to_document(self) -> dict:
        return {
            "profile_kind": HNSW_PROFILE_KIND,
            "hardware_key": self.hardware_key,
            "hardware_model_parameters": asdict(
                self.hardware_model_parameters
            ),
            "empirical_coefficients": asdict(self.empirical_coefficients),
            "fit_metadata": {
                "mape_latency_pct": self.fit_metadata.mape_latency_pct,
                "max_ape_latency_pct": self.fit_metadata.max_ape_latency_pct,
                "num_trials": self.fit_metadata.num_trials,
                "dataset_configs": list(self.fit_metadata.dataset_configs),
            },
        }


def profile_path_for_preset(preset: str) -> str:
    """Return the current profile-root path for an exact CPU key."""

    if not is_hnsw_hardware_key(preset):
        raise HNSWProfileError(f"invalid HNSW CPU hardware key: {preset!r}")
    return str(system_profile_path("faiss_hnsw", preset))


# Per-dataset workload-predictor paths retain their existing independent
# contract. They are deliberately not part of the exact-CPU system profile.
def _sani(value: str) -> str:
    return _re.sub(r"[^A-Za-z0-9]+", "-", str(value or "")).strip("-")


def dataset_predictor_key(dataset_name: str, embedding_model: str) -> str:
    """Return ``<dataset>__<embedding>`` using the historical sanitization."""

    return f"{_sani(dataset_name)}__{_sani(embedding_model)}"


def dataset_predictor_path(key: str) -> str:
    """Return the independent per-dataset HNSW workload-predictor path."""

    return str(calibration_profile_root() / "faiss_hnsw" / f"{key}.joblib")


def save_profile(
    profile: HNSWSystemProfile,
    path: Optional[str | os.PathLike[str]] = None,
) -> str:
    """Validate and atomically write one complete latest-schema profile."""

    if not isinstance(profile, HNSWSystemProfile):
        raise TypeError("save_profile requires an HNSWSystemProfile")
    target = Path(path) if path is not None else Path(
        profile_path_for_preset(profile.hardware_key)
    )
    if target.stem != profile.hardware_key:
        raise HNSWProfileError(
            f"profile filename {target.name!r} must match exact hardware key "
            f"{profile.hardware_key!r}"
        )
    document = profile.to_document()
    error = hnsw_profile_validation_error(document, profile.hardware_key)
    if error is not None:
        raise HNSWProfileError(f"refusing to save invalid HNSW profile: {error}")
    atomic_write_json(target, document)
    logger.info("Saved exact-CPU HNSW profile to %s", target)
    return str(target)


def _profile_from_document(document: dict) -> HNSWSystemProfile:
    hardware = document["hardware_model_parameters"]
    coefficients = document["empirical_coefficients"]
    fit = document["fit_metadata"]
    return HNSWSystemProfile(
        hardware_key=document["hardware_key"],
        hardware_model_parameters=HNSWHardwareModelParameters(**hardware),
        empirical_coefficients=HNSWCoefficients(**coefficients),
        fit_metadata=HNSWFitMetadata(
            mape_latency_pct=float(fit["mape_latency_pct"]),
            max_ape_latency_pct=float(fit["max_ape_latency_pct"]),
            num_trials=int(fit["num_trials"]),
            dataset_configs=tuple(fit["dataset_configs"]),
        ),
    )


def load_profile(
    path: str | os.PathLike[str],
    *,
    expected_hardware_key: str | None = None,
) -> HNSWSystemProfile:
    """Load one complete latest-schema profile without legacy backfilling."""

    source = Path(path)
    try:
        with source.open() as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HNSWProfileError(f"cannot read HNSW CPU profile {source}: {exc}") from exc
    error = hnsw_profile_validation_error(document, expected_hardware_key)
    if error is not None:
        raise HNSWProfileError(f"{source}: {error}")
    return _profile_from_document(document)


def load_profile_for_preset(preset: str) -> HNSWSystemProfile:
    """Load the exact CPU-keyed system profile, or fail closed."""

    raw_key = str(preset or "")
    key = raw_key.strip()
    if not key:
        raise HNSWProfileError(
            "HNSW pricing requires system.cpu.hardware_key (or a CPU preset)"
        )
    if key != raw_key:
        raise HNSWProfileError(
            f"HNSW CPU hardware key must not contain surrounding whitespace: "
            f"{raw_key!r}"
        )
    if not is_hnsw_hardware_key(key):
        raise HNSWProfileError(f"invalid HNSW CPU hardware key: {key!r}")
    path = Path(profile_path_for_preset(key))
    if not path.is_file():
        raise HNSWProfileError(
            f"missing HNSW CPU profile for hardware {key!r}: {path}"
        )
    return load_profile(path, expected_hardware_key=key)


def _require_system_preset(preset: str) -> None:
    from .system_presets import SYSTEM_PRESETS

    if preset in SYSTEM_PRESETS:
        return
    from rag_stack.cost_model.faiss_system_registry import (
        require_configured_faiss_preset,
    )

    require_configured_faiss_preset("hnsw", preset)


def build_hardware_from_profile(
    profile: HNSWSystemProfile,
    *,
    active_threads: int = 0,
    system_config=None,
):
    """Join raw CPU facts with all profile-owned HNSW runtime parameters.

    ``system_config`` is required in production and supplies either a built-in
    preset identity or an inline exact CPU spec. A direct calibrator/test caller
    may omit it and resolve a registered preset by ``profile.hardware_key``.
    Every algorithm-specific value is passed explicitly from the profile.
    """

    from .hardware import system_config_to_faiss_hardware
    from .system_presets import SYSTEM_PRESETS

    kwargs = profile.hardware_model_parameters.to_hardware_kwargs()
    kwargs["active_threads"] = int(active_threads)

    if system_config is not None:
        configured_key = str(
            getattr(system_config, "cpu_key", "")
            or getattr(system_config, "cpu_preset", "")
            or ""
        )
        if not configured_key.strip():
            raise HNSWProfileError(
                "HNSW pricing requires an exact system.cpu hardware key"
            )
        if configured_key != configured_key.strip():
            raise HNSWProfileError(
                f"system CPU hardware key must not contain surrounding "
                f"whitespace: {configured_key!r}"
            )
        if configured_key != profile.hardware_key:
            raise HNSWProfileError(
                f"CPU identity mismatch: system CPU is {configured_key!r} but "
                f"the HNSW profile belongs to {profile.hardware_key!r}"
            )
        if getattr(system_config, "cpu_preset", ""):
            _require_system_preset(configured_key)
            hardware = SYSTEM_PRESETS[configured_key](**kwargs)
        elif getattr(system_config, "peak_flops", 0.0) > 0.0:
            hardware = system_config_to_faiss_hardware(system_config, **kwargs)
        else:
            raise HNSWProfileError(
                f"system CPU {configured_key!r} has no raw hardware facts"
            )
    else:
        _require_system_preset(profile.hardware_key)
        hardware = SYSTEM_PRESETS[profile.hardware_key](**kwargs)

    # Fail if a future factory signature accidentally accepts but drops a
    # profile-owned parameter.
    for field, expected in profile.hardware_model_parameters.to_hardware_kwargs().items():
        actual = getattr(hardware, field, None)
        if actual != expected:
            raise HNSWProfileError(
                f"HNSW hardware builder did not apply profile field {field}: "
                f"expected {expected!r}, got {actual!r}"
            )
    return hardware


# Compatibility names for internal callers that only need the four numerical
# values. They still validate the *entire* latest-schema document first.
def load_coefficients(path: str | os.PathLike[str]) -> HNSWCoefficients:
    source = Path(path)
    return load_profile(
        source, expected_hardware_key=source.stem,
    ).empirical_coefficients


def load_coefficients_for_preset(preset: str) -> HNSWCoefficients:
    return load_profile_for_preset(preset).empirical_coefficients


def save_coefficients(
    profile: HNSWSystemProfile,
    path: Optional[str | os.PathLike[str]] = None,
) -> str:
    """Compatibility alias; partial coefficient-only writes are forbidden."""

    return save_profile(profile, path)
