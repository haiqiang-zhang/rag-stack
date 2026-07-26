"""Strict runtime loader for standalone retrieval-worker calibrations.

Retrieval-worker profiles are keyed by the CPU that executes the operator,
not by a co-located GPU.  They contain only measurements of isolated worker
operators.  End-to-end stage residuals, deployment corrections and measured
benchmark throughput are deliberately outside this profile contract.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from rag_stack.calibration_paths import system_profile_path
from rag_stack.cost_model.calibration_mode import is_uncalibrated_spec_mode


logger = logging.getLogger("RAG-Stack")
_SPEC_FALLBACK_WARNED: set[str] = set()


class RetrievalWorkerProfileError(ValueError):
    """A required retrieval-worker profile is missing or malformed."""


@dataclass(frozen=True)
class RRFFusionProfile:
    """Affine service demand for one user row's multi-list RRF operation."""

    per_call_s: float
    per_item_s: float

    def service_s(self, input_items: int | float) -> float:
        try:
            items = float(input_items)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"RRF input_items must be a finite non-negative number, got "
                f"{input_items!r}"
            ) from exc
        if not math.isfinite(items) or items < 0.0:
            raise ValueError(
                f"RRF input_items must be a finite non-negative number, got "
                f"{input_items!r}"
            )
        return self.per_call_s + self.per_item_s * items


@dataclass(frozen=True)
class RRFFusionSpecModel:
    """Coefficient-free CPU roofline lower bound for multi-list RRF.

    Pandas/Python dispatch is not inferable from a processor datasheet.  This
    model therefore counts only structural table materialization and the two
    rank sorts in ``rrf_pure`` and prices them from one-core integer throughput
    and single-thread memory bandwidth.  It is intentionally available only in
    explicit ``uncalibrated_spec`` mode and must not be presented as calibrated
    worker service.
    """

    cpu_key: str
    single_thread_bandwidth_bytes_s: float
    single_core_int_ops_s: float

    def service_s(self, input_items: int | float) -> float:
        try:
            items = float(input_items)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "RRF input_items must be a finite non-negative number, got "
                f"{input_items!r}"
            ) from exc
        if not math.isfinite(items) or items < 0.0:
            raise ValueError(
                "RRF input_items must be a finite non-negative number, got "
                f"{input_items!r}"
            )
        if items == 0.0:
            return 0.0

        sort_levels = max(1, int(math.ceil(math.log2(max(2.0, items)))))
        # Four minimum-width table columns (id pointer, score, rank, fused
        # score), one write + one read, followed by two comparison sorts.
        structural_bytes = items * (
            4.0 * 8.0 * 2.0 + 2.0 * sort_levels * 16.0
        )
        structural_int_ops = items * (8.0 + 2.0 * sort_levels)
        return max(
            structural_bytes / self.single_thread_bandwidth_bytes_s,
            structural_int_ops / self.single_core_int_ops_s,
        )


def _rrf_spec_model(hardware: Any, cpu_key: str) -> RRFFusionSpecModel:
    """Build the RRF lower bound from inline facts or an exact CPU preset."""

    num_cores = int(getattr(hardware, "num_cores", 0) or 0)
    bandwidth = float(getattr(hardware, "mem_bandwidth_1t", 0.0) or 0.0)
    peak_int_ops = float(getattr(hardware, "peak_int_ops", 0.0) or 0.0)

    if (
        (bandwidth <= 0.0 or peak_int_ops <= 0.0 or num_cores <= 0)
        and getattr(hardware, "cpu_preset", "")
    ):
        from rag_stack.cost_model.faiss_ivf_sim.system_presets import (
            SYSTEM_PRESETS,
        )

        factory = SYSTEM_PRESETS.get(cpu_key)
        if factory is not None:
            preset = factory(active_threads=1)
            num_cores = int(preset.num_cores)
            bandwidth = float(preset.mem_bandwidth_1t_bytes_s)
            peak_int_ops = float(preset.peak_int_ops)

    if (
        not math.isfinite(bandwidth)
        or bandwidth <= 0.0
        or not math.isfinite(peak_int_ops)
        or peak_int_ops <= 0.0
        or num_cores <= 0
    ):
        raise RetrievalWorkerProfileError(
            "uncalibrated_spec RRF pricing requires physical CPU "
            "num_cores, peak_int_ops, and mem_bandwidth_1t"
        )
    return RRFFusionSpecModel(
        cpu_key=cpu_key,
        single_thread_bandwidth_bytes_s=bandwidth,
        single_core_int_ops_s=peak_int_ops / float(num_cores),
    )


def _require_mapping(value: Any, *, field: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RetrievalWorkerProfileError(
            f"{path}: {field} must be a JSON object"
        )
    return value


def _nonnegative_finite(value: Any, *, field: str, path: Path) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RetrievalWorkerProfileError(
            f"{path}: {field} must be a finite non-negative number"
        ) from exc
    if not math.isfinite(resolved) or resolved < 0.0:
        raise RetrievalWorkerProfileError(
            f"{path}: {field} must be a finite non-negative number"
        )
    return resolved


def load_rrf_fusion_profile(
    cpu_key: str,
    *,
    profile_path: str | Path | None = None,
) -> RRFFusionProfile:
    """Load an exact CPU-keyed RRF component profile, or fail closed.

    A missing profile is not silently interpreted as zero work: doing so would
    make MQE/query-decomposition candidates look artificially cheap.  Schema
    and unit tags are checked before any coefficient can enter the simulator.
    """

    key = str(cpu_key or "").strip()
    if not key:
        raise RetrievalWorkerProfileError(
            "RRF fusion requires system.cpu.hardware_key (or a CPU preset)"
        )
    path = (
        Path(profile_path)
        if profile_path is not None
        else system_profile_path("retrieval_worker", key)
    )
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RetrievalWorkerProfileError(
            f"missing retrieval-worker profile for CPU {key!r}: {path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RetrievalWorkerProfileError(
            f"cannot read retrieval-worker profile {path}: {exc}"
        ) from exc

    root = _require_mapping(document, field="profile", path=path)
    # Reuse the metadata-only inventory contract so production pricing cannot
    # bypass the profile schema merely because ``status`` was skipped.
    from rag_stack.calibration_inventory import (
        retrieval_worker_validation_error,
    )
    contract_error = retrieval_worker_validation_error(root, key)
    if contract_error is not None:
        raise RetrievalWorkerProfileError(f"{path}: {contract_error}")
    if root.get("profile_kind") != "retrieval_worker":
        raise RetrievalWorkerProfileError(
            f"{path}: profile_kind must be 'retrieval_worker'"
        )
    if root.get("hardware_key") != key:
        raise RetrievalWorkerProfileError(
            f"{path}: hardware_key {root.get('hardware_key')!r} does not "
            f"match requested CPU {key!r}"
        )
    components = _require_mapping(
        root.get("components"), field="components", path=path,
    )
    rrf = _require_mapping(
        components.get("rrf_fusion"),
        field="components.rrf_fusion",
        path=path,
    )
    expected_operator = (
        "rag_stack_evaluator.static_rag_evaluator.nodes.hybridretrieval."
        "hybrid_rrf.rrf_pure"
    )
    if rrf.get("operator") != expected_operator:
        raise RetrievalWorkerProfileError(
            f"{path}: components.rrf_fusion.operator must be "
            f"{expected_operator!r}"
        )
    if rrf.get("service_unit") != "seconds_per_user_row":
        raise RetrievalWorkerProfileError(
            f"{path}: components.rrf_fusion.service_unit must be "
            "'seconds_per_user_row'"
        )
    if rrf.get("demand_axis") != "input_items":
        raise RetrievalWorkerProfileError(
            f"{path}: components.rrf_fusion.demand_axis must be 'input_items'"
        )
    if rrf.get("fit_form") != "nonnegative_affine":
        raise RetrievalWorkerProfileError(
            f"{path}: components.rrf_fusion.fit_form must be "
            "'nonnegative_affine'"
        )
    return RRFFusionProfile(
        per_call_s=_nonnegative_finite(
            rrf.get("per_call_s"),
            field="components.rrf_fusion.per_call_s",
            path=path,
        ),
        per_item_s=_nonnegative_finite(
            rrf.get("per_item_s"),
            field="components.rrf_fusion.per_item_s",
            path=path,
        ),
    )


def resolve_rrf_fusion_profile(
    hardware: Any,
    pipeline_config: Mapping[str, Any] | None = None,
) -> RRFFusionProfile | RRFFusionSpecModel:
    """Resolve exact RRF evidence or the explicit CPU-spec lower bound."""

    cpu_key = (
        getattr(hardware, "cpu_key", "")
        or getattr(hardware, "cpu_preset", "")
    )
    key = str(cpu_key or "")
    path = system_profile_path("retrieval_worker", key)
    if path.exists() or path.is_symlink():
        return load_rrf_fusion_profile(key, profile_path=path)
    if is_uncalibrated_spec_mode(pipeline_config, hardware):
        model = _rrf_spec_model(hardware, key)
        if key not in _SPEC_FALLBACK_WARNED:
            _SPEC_FALLBACK_WARNED.add(key)
            logger.warning(
                "No exact retrieval-worker RRF profile for CPU %r; "
                "uncalibrated_spec is using a coefficient-free CPU roofline "
                "lower bound. Pandas/Python dispatch is not calibrated.",
                key,
            )
        return model
    return load_rrf_fusion_profile(key, profile_path=path)
