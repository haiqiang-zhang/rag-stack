"""Read-only calibration readiness and migration planning.

The inventory deliberately uses only YAML/JSON metadata and file existence.
It never imports :mod:`rag_stack.cost_model`, opens parquet/joblib payloads, or
probes CUDA.  This makes ``rag_stack calibrate status`` safe on a shared box
and usable before the target device has any profile at all.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from rag_stack.calibration_paths import (
    calibration_data_root,
    calibration_profile_root,
)
from rag_stack.hnsw_profile_schema import (
    hnsw_profile_validation_error,
)
from rag_stack.ivf_profile_schema import (
    IVF_RUNTIME_COEFFICIENT_FIELDS,
    ivf_profile_validation_error,
)
from rag_stack.ivf_imbalance_profile_schema import unwrap_ivf_imbalance_profile
from rag_stack.llm_sim_profile_schema import unwrap_llm_sim_profile_document

CANONICAL_GENERATOR_MODELS = (
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
)

CORE_SCOPES = ("ivf", "hnsw", "generator", "generator-tp")
VALID_SCOPES = (*CORE_SCOPES, "retrieval-worker", "pipeline")

# Pipeline inventory intentionally duplicates only the small identity tables
# needed to resolve serialized profile keys.  Importing the owning cost-model
# modules here would pull in numpy/pandas/GenZ (and, on some installations,
# CUDA-aware packages), defeating the metadata-only status command.
RERANKER_DEFAULT_MODELS = {
    "tart": "tart_flan_t5_xl",
    "monot5": "monot5-3b",
    "flag_embedding_reranker": "bge_reranker_large",
    "openvino_reranker": "bge_reranker_large",
    "sentence_transformer_reranker": "ms_marco_minilm_l2",
    "colbert_reranker": "colbertv2",
    "flag_embedding_llm_reranker": "bge-reranker-v2-gemma",
    "rankgpt": "llama-2-7b",
    "upr": "t5-large",
}
RERANKER_COST_MODELS = {
    **{
        component: "passage_reranker_sim"
        for component in (
            "tart",
            "monot5",
            "flag_embedding_reranker",
            "openvino_reranker",
            "sentence_transformer_reranker",
            "colbert_reranker",
        )
    },
    "flag_embedding_llm_reranker": "passage_reranker_llm_sim",
    "rankgpt": "passage_reranker_llm_sim",
    "upr": "passage_reranker_llm_sim",
}
UNPRICED_RERANKERS = {"time_reranker"}
QUERY_EXPANSION_DEFAULT_FANOUT = {
    "hyde": 1,
    "query_decompose": 2,
    "multi_query_expansion": 4,
}

LLMLINGUA2_ALLOWED_FIELDS = {
    "profile_kind", "service_unit",
    "demand_axes", "model", "hardware_key", "cpu_hardware_key",
    "fixed_ms", "per_chunk_ms", "per_token_ms", "fit_mape",
    "correlation", "grid", "note",
}
LLMLINGUA2_GRID_FIELDS = {
    "n_chunks", "chunk_tokens", "measured_ms", "predicted_ms", "rel_err",
}

_RAG_CM_FIELDS = frozenset({"calibration_mode"})
_CALIBRATION_MODES = frozenset({"strict", "uncalibrated_spec"})
RETRIEVAL_WORKER_ROOT_FIELDS = {
    "profile_kind",
    "hardware_key",
    "components",
}
RRF_COMPONENT_FIELDS = {
    "operator",
    "service_unit",
    "demand_axis",
    "fit_form",
    "equation",
    "per_call_s",
    "per_item_s",
    "fit_mape",
    "grid",
}
RRF_GRID_FIELDS = {"k", "top_k", "overlap", "items", "seconds"}
RRF_K_GRID = (2, 3, 4, 6)
RRF_TOPK_GRID = (4, 8, 16, 32, 64)
RRF_OVERLAP_GRID = (0.0, 0.5)

# Backward-compatible test/public alias: runtime coefficients plus numeric fit
# diagnostics. The authoritative exact schema lives in ivf_profile_schema.py.
IVF_REQUIRED_NUMERIC_FIELDS = (
    *IVF_RUNTIME_COEFFICIENT_FIELDS,
    "mape_latency",
    "max_ape_latency",
    "num_trials",
)
LLM_RECORD_ALLOWED_FIELDS = {
    "component",
    "model_key",
    "hardware_key",
    "calibration_contract",
    "bits",
    "compute_efficiency",
    "memory_efficiency",
    "fit_mape",
    "per_call_overhead_s",
    "correlation",
    "grid",
    "linear_law",
    "max_seq_len",
    "identifiability_rank",
    "identifiability_condition",
    "validation_mape",
    "validation_grid",
}
_GENERATOR_ROOFLINE_CALIBRATION_CONTRACT = (
    "standalone_ceff_meff_zero_intercept_v1"
)
_LEGACY_INTERNAL_COMPONENT_CONTRACT = "legacy_internal_component_v0"
_LINEAR_LAW_COMPONENT_CONTRACT = "standalone_linear_law_v1"
_CAPACITY_HINGE_LAW_COMPONENT_CONTRACT = "standalone_capacity_hinge_law_v1"


@dataclass(frozen=True)
class CalibrationItem:
    """One independently actionable calibration capability."""

    name: str
    state: str
    required_for_deploy: bool
    profile_path: str
    frozen_path: str | None
    detail: str
    commands: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["commands"] = list(self.commands)
        return value


@dataclass(frozen=True)
class CalibrationInventory:
    config_path: str
    system_hardware: str | None
    cpu_key: str | None
    gpu_key: str | None
    gpu_count: int | None
    data_root: str
    profile_root: str
    scopes: tuple[str, ...]
    items: tuple[CalibrationItem, ...]

    @property
    def deploy_ready(self) -> bool:
        return all(
            item.state == "ready"
            for item in self.items
            if item.required_for_deploy
        )

    @property
    def pending_commands(self) -> tuple[str, ...]:
        commands: list[str] = []
        for item in self.items:
            if item.state == "ready" or not item.required_for_deploy:
                continue
            for command in item.commands:
                if command not in commands:
                    commands.append(command)
        return tuple(commands)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "hardware": {
                "system_hardware": self.system_hardware,
                "cpu_key": self.cpu_key,
                "gpu_key": self.gpu_key,
                "gpu_count": self.gpu_count,
            },
            "roots": {
                "data": self.data_root,
                "profiles": self.profile_root,
            },
            "scopes": list(self.scopes),
            "deploy_ready": self.deploy_ready,
            "deploy_ready_semantic": (
                "deploy_ready means every required item in the selected scopes "
                "is ready; it means full-pipeline readiness only when scope "
                "'pipeline' is selected"
            ),
            "items": [item.to_dict() for item in self.items],
            "pending_commands": list(self.pending_commands),
        }


def _hardware_key(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        key = value.strip()
        if key in (".", "..") or "/" in key or "\\" in key:
            raise ValueError(f"config.system.{label} hardware key is not path-safe: {key!r}")
        return key
    if isinstance(value, Mapping):
        key = value.get("hardware_key")
        if isinstance(key, str) and key.strip():
            key = key.strip()
            if key in (".", "..") or "/" in key or "\\" in key:
                raise ValueError(f"config.system.{label} hardware key is not path-safe: {key!r}")
            return key
    raise ValueError(
        f"config.system.{label} must be a preset string or a mapping with "
        "a non-empty hardware_key"
    )


def _system_hardware_key(value: Any) -> str | None:
    """Return the host+accelerator serving identity used by stage profiles."""

    if value is None:
        return None
    from rag_stack.cost_model.stage_batch_curve import (
        validate_system_hardware,
    )

    try:
        return validate_system_hardware(value)
    except ValueError as exc:
        raise ValueError(f"config.system.{exc}") from exc


def _load_config_document(config_path: str | Path) -> Mapping[str, Any]:
    """Load and environment-expand YAML without importing runtime config code."""
    try:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"{config_path}: cannot load calibration YAML: {exc}") from exc

    env_pattern = re.compile(r"\$\{([^}]+)}")

    def expand(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            return env_pattern.sub(lambda match: os.environ.get(match.group(1), ""), value)
        return value

    cfg = expand(cfg)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"{config_path}: YAML root must be a mapping")
    return cfg


def _load_hardware_identity(
    config_path: str | Path,
    document: Mapping[str, Any] | None = None,
    *,
    strict_system_hardware: bool = True,
    strict_cpu: bool = True,
    strict_gpu: bool = True,
) -> tuple[str | None, str | None, str | None, int | None]:
    """Parse only the identity fields needed by inventory; never probe presets."""
    cfg = document or _load_config_document(config_path)
    system = cfg.get("system")
    if not isinstance(system, Mapping):
        raise ValueError(f"{config_path}: missing mapping `system`")

    def optional_identity(loader, value: Any, *args: Any) -> Any:
        try:
            return loader(value, *args)
        except ValueError:
            return None

    system_hardware = (
        _system_hardware_key(system.get("system_hardware"))
        if strict_system_hardware
        else optional_identity(
            _system_hardware_key, system.get("system_hardware")
        )
    )
    cpu_key = (
        _hardware_key(system.get("cpu"), "cpu")
        if strict_cpu
        else optional_identity(_hardware_key, system.get("cpu"), "cpu")
    )
    gpu_key = (
        _hardware_key(system.get("gpu"), "gpu")
        if strict_gpu
        else optional_identity(_hardware_key, system.get("gpu"), "gpu")
    )
    available = system.get("available_gpus")
    design: Mapping[str, Any] | None = None
    if not available:
        design = system.get("system_design_space")
        if not isinstance(design, Mapping):
            design = system.get("measured_search_space")
        if not isinstance(design, Mapping):
            design = system.get("cm_search_space")
        available = design.get("available_gpus") if isinstance(design, Mapping) else None
    gpu_count = (
        len(available)
        if isinstance(available, Sequence) and not isinstance(available, str)
        else None
    )
    if gpu_count is None and isinstance(design, Mapping):
        declared_max = design.get("max_num_gpus")
        if isinstance(declared_max, int) and not isinstance(declared_max, bool):
            gpu_count = declared_max if declared_max > 0 else None
    return system_hardware, cpu_key, gpu_key, gpu_count


def _load_hardware_identity_lenient(
    document: Mapping[str, Any],
) -> tuple[str | None, str | None, str | None, int | None]:
    """Best-effort identities so pipeline status can report, not crash, gaps."""
    system = document.get("system")
    if not isinstance(system, Mapping):
        return None, None, None, None

    def identity(field: str) -> str | None:
        try:
            return _hardware_key(system.get(field), field)
        except ValueError:
            return None

    available = system.get("available_gpus")
    design: Mapping[str, Any] | None = None
    if not available:
        for key in (
            "system_design_space", "measured_search_space", "cm_search_space",
        ):
            candidate = system.get(key)
            if isinstance(candidate, Mapping):
                design = candidate
                available = candidate.get("available_gpus")
                if available:
                    break
    count = (
        len(available)
        if isinstance(available, Sequence) and not isinstance(available, str)
        else None
    )
    if count is None and isinstance(design, Mapping) and _positive_int(
        design.get("max_num_gpus")
    ):
        count = int(design["max_num_gpus"])
    try:
        system_hardware = _system_hardware_key(system.get("system_hardware"))
    except ValueError:
        system_hardware = None
    return system_hardware, identity("cpu"), identity("gpu"), count


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "file is absent"
    try:
        with path.open() as stream:
            value = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(value, dict):
        return None, "JSON root is not an object"
    return value, None


def _read_llm_sim_profiles(
    path: Path,
) -> tuple[dict[str, Any] | None, str | None]:
    """Read only the strict current LLM component envelope."""

    document, error = _read_json(path)
    if document is None:
        return None, error
    try:
        return unwrap_llm_sim_profile_document(document, source=str(path)), None
    except ValueError as exc:
        return None, str(exc)


def _validate_calibration_mode_config(document: Mapping[str, Any]) -> None:
    """Metadata-only mirror of the production ``rag_cm`` policy gate.

    Importing ``rag_stack.cost_model`` would auto-discover the full cost
    model registry, which the read-only status command intentionally avoids.
    Keep this tiny serialized-config boundary exact and fail closed instead.
    """

    system = document.get("system")
    if system is None:
        return
    if not isinstance(system, Mapping):
        raise ValueError("system must be a mapping")
    rag_cm = system.get("rag_cm")
    if rag_cm is None:
        return
    if not isinstance(rag_cm, Mapping):
        raise ValueError("system.rag_cm must be a mapping")
    unsupported = sorted(set(rag_cm) - _RAG_CM_FIELDS)
    if unsupported:
        raise ValueError(
            "system.rag_cm contains fields unsupported by cost_model: "
            f"{unsupported}. Remove them; the CM reads the canonical "
            "calibration_profiles root (or the process-wide "
            "RAG_STACK_CALIBRATION_PROFILE_DIR override)."
        )
    mode = rag_cm.get("calibration_mode", "strict")
    if not isinstance(mode, str) or mode not in _CALIBRATION_MODES:
        raise ValueError(
            "system.rag_cm.calibration_mode must be one of "
            f"{sorted(_CALIBRATION_MODES)}; got {mode!r}"
        )


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    number = float(value)
    return math.isfinite(number) and (number > 0 if positive else number >= 0)


def _finite_any_number(value: Any) -> bool:
    return bool(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _frozen_exists(path: Path | None) -> bool:
    if not path or not path.is_file() or path.stat().st_size < 8:
        return False
    try:
        with path.open("rb") as stream:
            header = stream.read(4)
            stream.seek(-4, 2)
            footer = stream.read(4)
        return header == b"PAR1" and footer == b"PAR1"
    except OSError:
        return False


def _stage_frozen_cohort_paths(stage_raw: Path) -> tuple[Path, ...]:
    """Return the complete offline-fit cohort for one system stage campaign."""

    from rag_stack.cost_model.llm_sim.calibration.main_llm.vllm_probe import (
        _run_contract_path,
    )

    component_raw = stage_raw.with_name(
        stage_raw.name.removesuffix(".jsonl") + ".component.jsonl"
    )
    return (
        stage_raw,
        component_raw,
        _run_contract_path(stage_raw),
        _run_contract_path(component_raw),
    )


def _stage_frozen_cohort_status(stage_raw: Path) -> tuple[bool, str | None]:
    """Validate the metadata needed before advertising CPU-only stage fit.

    The two telemetry streams are deliberately acquired in independent runs.
    Their shared startup/workload contract, distinct run IDs, exact raw SHA,
    frozen target IDs, raw row schemas, exact stage shapes, and independent
    component join are validated through the same readers used by the fitter.
    """

    artifacts = _stage_frozen_cohort_paths(stage_raw)
    has_any_artifact = any(path.exists() for path in artifacts)
    missing_or_empty: list[str] = []
    for path in artifacts:
        try:
            valid_file = path.is_file() and path.stat().st_size > 0
        except OSError:
            valid_file = False
        if not valid_file:
            missing_or_empty.append(path.name)
    if missing_or_empty:
        return has_any_artifact, (
            "stage frozen cohort is absent/incomplete; missing or empty: "
            + ", ".join(missing_or_empty)
        )

    from rag_stack.cost_model.llm_sim.calibration.main_llm.vllm_probe import (
        _load_run_contract,
        _validate_comparable_run_contracts,
        read_component_cycle_services,
        read_stage_cycle_observations,
    )

    component_raw = artifacts[1]
    try:
        stage_contract = _load_run_contract(
            stage_raw, expected_scope="stage_only"
        )
        component_contract = _load_run_contract(
            component_raw, expected_scope="component_only"
        )
        _validate_comparable_run_contracts(
            component_contract, stage_contract
        )
        stage_observations = read_stage_cycle_observations(stage_raw)
        _validate_stage_frozen_target_shapes(
            stage_contract, stage_observations,
        )
        component_services = read_component_cycle_services(
            component_raw,
            stage_observations,
            stage_telemetry_path=stage_raw,
        )
        expected_stage_keys = {
            (observation.run_id, observation.cycle_id)
            for observation in stage_observations
        }
        if set(component_services) != expected_stage_keys:
            raise ValueError(
                "component telemetry join does not cover every frozen stage cycle"
            )
    except (OSError, ValueError) as exc:
        return True, f"invalid stage frozen cohort: {exc}"
    return True, None


def _validate_stage_frozen_target_shapes(
    stage_contract: Mapping[str, Any],
    stage_observations: Sequence[Any],
) -> None:
    """Bind every frozen target ID to its exact contracted phase shape."""

    from rag_stack.cost_model.stage_batch_curve import (
        ANCHOR_BATCHES,
        HELD_OUT_BATCH,
    )

    workload = stage_contract["workload"]
    expected_batches = tuple(sorted((*ANCHOR_BATCHES, HELD_OUT_BATCH)))
    actual_batches = tuple(int(value) for value in workload["target_batches"])
    if actual_batches != expected_batches:
        raise ValueError(
            "stage frozen target batches must exactly cover anchors plus "
            f"held-out validation: expected={expected_batches!r}, "
            f"actual={actual_batches!r}"
        )
    prompt_tokens = int(workload["prompt_tokens"])
    output_tokens = int(workload["output_tokens"])
    if output_tokens < 2:
        raise ValueError(
            "stage frozen workload must contain at least one exact decode cycle"
        )
    by_cycle_id = {
        str(observation.cycle_id): observation
        for observation in stage_observations
    }
    for raw_batch in workload["target_batches"]:
        batch = int(raw_batch)
        subruns = stage_contract["subruns"][str(batch)]
        if any(
            int(subruns[role]["startup"]["max_num_seqs"]) != batch
            for role in ("proof", "timing")
        ):
            raise ValueError(
                f"stage frozen B{batch} subrun engine cap must equal its batch"
            )
        cycle_ids = stage_contract["target_cycle_ids"][str(batch)]
        for index, cycle_id in enumerate(cycle_ids):
            observation = by_cycle_id.get(str(cycle_id))
            if observation is None:
                raise ValueError(
                    f"stage telemetry is missing frozen target cycle {cycle_id!r}"
                )
            if index == 0:
                expected = (
                    "vllm_prefill",
                    batch,
                    batch * prompt_tokens,
                    batch * prompt_tokens,
                    batch * prompt_tokens * prompt_tokens,
                )
            else:
                context = batch * (prompt_tokens + index)
                expected = (
                    "vllm_decode",
                    batch,
                    batch,
                    context,
                    context,
                )
            actual = (
                str(observation.performance_stage),
                int(observation.batch_size),
                int(observation.scheduled_tokens),
                int(observation.scheduled_context_tokens),
                int(observation.token_context_product),
            )
            if actual != expected:
                raise ValueError(
                    "stage frozen target shape mismatch for "
                    f"B{batch} cycle {cycle_id!r}: "
                    f"expected={expected!r}, actual={actual!r}"
                )
    expected_count = len(expected_batches) * output_tokens
    if len(by_cycle_id) != expected_count:
        raise ValueError(
            "stage telemetry frozen target cycle count disagrees with workload"
        )


def _stage_measurement_blocker(
    gpu_key: str | None,
    gpu_count: int | None,
) -> str | None:
    """Explain acquisition-only config gaps without changing profile identity."""

    missing: list[str] = []
    if not gpu_key:
        missing.append("config.system.gpu")
    if not gpu_count:
        missing.append(
            "a non-empty config.system.available_gpus (or search-space "
            "equivalent)"
        )
    if not missing:
        return None
    return (
        "stage measurement is blocked until the TP1/PP1 reference can be "
        "acquired: " + "; ".join(missing)
    )


def _root_flags(data_root: Path, profile_root: Path) -> str:
    return (
        f"--data-dir {shlex.quote(str(data_root))} "
        f"--profile-dir {shlex.quote(str(profile_root))}"
    )


def _phase_commands(
    component: str,
    config_path: Path,
    data_root: Path,
    profile_root: Path,
    *,
    frozen_ready: bool,
) -> tuple[str, ...]:
    flags = _root_flags(data_root, profile_root)
    if frozen_ready:
        return (
            f"rag_stack calibrate fit {component} --config {shlex.quote(str(config_path))} {flags}",
        )
    return (
        f"rag_stack calibrate measure {component} --config {shlex.quote(str(config_path))} {flags}",
        f"rag_stack calibrate fit {component} --config {shlex.quote(str(config_path))} {flags}",
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def retrieval_worker_validation_error(
    document: Mapping[str, Any] | None,
    cpu_key: str,
) -> str | None:
    """Validate the standalone RRF profile without importing numeric stacks."""

    if not isinstance(document, Mapping):
        return "profile root must be a JSON object"
    if set(document) != RETRIEVAL_WORKER_ROOT_FIELDS:
        return "profile root fields do not match the retrieval-worker contract"
    if document.get("profile_kind") != "retrieval_worker":
        return "profile_kind must be 'retrieval_worker'"
    if document.get("hardware_key") != cpu_key:
        return f"hardware_key must match CPU {cpu_key!r}"
    components = document.get("components")
    if not isinstance(components, Mapping) or set(components) != {"rrf_fusion"}:
        return "components must contain exactly rrf_fusion"
    rrf = components.get("rrf_fusion")
    if not isinstance(rrf, Mapping) or set(rrf) != RRF_COMPONENT_FIELDS:
        return "components.rrf_fusion fields do not match the exact contract"
    expected_tags = {
        "operator": (
            "rag_stack_evaluator.static_rag_evaluator.nodes.hybridretrieval."
            "hybrid_rrf.rrf_pure"
        ),
        "service_unit": "seconds_per_user_row",
        "demand_axis": "input_items",
        "fit_form": "nonnegative_affine",
        "equation": (
            "service_s = per_call_s + per_item_s * (fanout * top_k)"
        ),
    }
    for field, expected in expected_tags.items():
        if rrf.get(field) != expected:
            return f"components.rrf_fusion.{field} must be {expected!r}"
    for field in ("per_call_s", "per_item_s", "fit_mape"):
        if not _finite_number(rrf.get(field)):
            return f"components.rrf_fusion.{field} must be finite and non-negative"
    if float(rrf["per_call_s"]) + float(rrf["per_item_s"]) <= 0.0:
        return "RRF service coefficients cannot both be zero"
    rows = rrf.get("grid")
    expected_grid = {
        (k, top_k, overlap)
        for k in RRF_K_GRID
        for top_k in RRF_TOPK_GRID
        for overlap in RRF_OVERLAP_GRID
    }
    if not isinstance(rows, list) or len(rows) != len(expected_grid):
        return "RRF grid must contain the complete 40-point calibration design"
    observed_grid: set[tuple[int, int, float]] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != RRF_GRID_FIELDS:
            return "every RRF grid row must contain k, top_k, overlap, items, seconds"
        k, top_k, overlap = row.get("k"), row.get("top_k"), row.get("overlap")
        if not _positive_int(k) or k not in RRF_K_GRID:
            return "RRF grid contains an unsupported k"
        if not _positive_int(top_k) or top_k not in RRF_TOPK_GRID:
            return "RRF grid contains an unsupported top_k"
        if not _finite_number(overlap) or float(overlap) not in RRF_OVERLAP_GRID:
            return "RRF grid contains an unsupported overlap"
        if not _positive_int(row.get("items")) or row["items"] != k * top_k:
            return "RRF grid items must equal k * top_k"
        if not _finite_number(row.get("seconds"), positive=True):
            return "RRF grid seconds must be finite and positive"
        observed_grid.add((k, top_k, float(overlap)))
    if observed_grid != expected_grid:
        return "RRF grid is duplicated or incomplete"
    return None


def _retrieval_worker_item(
    cpu_key: str,
    config_path: Path,
    data_root: Path,
    profile_root: Path,
) -> CalibrationItem:
    profile = profile_root / "retrieval_worker" / f"{cpu_key}.json"
    document, read_error = _read_json(profile)
    validation_error = (
        read_error
        if document is None
        else retrieval_worker_validation_error(document, cpu_key)
    )
    flags = _root_flags(data_root, profile_root)
    command = (
        f"rag_stack calibrate retrieval-worker "
        f"--config {shlex.quote(str(config_path))} {flags}"
    )
    if validation_error is None:
        state = "ready"
        detail = "CPU identity, units, coefficients, and fit grid are valid"
        commands: tuple[str, ...] = ()
    elif profile.is_file():
        state = "invalid"
        detail = validation_error
        commands = (command,)
    else:
        state = "needs_measure"
        detail = read_error or "retrieval-worker profile is absent"
        commands = (command,)
    return CalibrationItem(
        name="retrieval-worker",
        state=state,
        required_for_deploy=True,
        profile_path=str(profile),
        frozen_path=None,
        detail=detail,
        commands=commands,
    )


def _faiss_item(
    component: str,
    cpu_key: str,
    config_path: Path,
    data_root: Path,
    profile_root: Path,
    *,
    custom_hnsw_parameters_required: bool = False,
) -> CalibrationItem:
    frozen = data_root / component / f"{cpu_key}.parquet"
    profile = profile_root / f"faiss_{component}" / f"{cpu_key}.json"
    document, error = _read_json(profile)
    hnsw_error: str | None = None
    ivf_error: str | None = None
    if component == "ivf":
        ivf_error = error or ivf_profile_validation_error(document, cpu_key)
        valid = ivf_error is None
    else:
        hnsw_error = error or hnsw_profile_validation_error(document, cpu_key)
        valid = hnsw_error is None
    frozen_ready = _frozen_exists(frozen)
    if valid:
        state = "ready"
        detail = (
            "latest exact-CPU HNSW runtime profile is valid"
            if component == "hnsw"
            else "latest exact-CPU IVF runtime profile is valid"
        )
        commands: tuple[str, ...] = ()
    elif profile.is_file():
        state = "invalid"
        detail = (
            hnsw_error
            if component == "hnsw"
            else ivf_error
        )
        commands = _phase_commands(
            component, config_path, data_root, profile_root,
            frozen_ready=frozen_ready,
        )
    elif (
        component == "hnsw"
        and custom_hnsw_parameters_required
        and not profile.is_file()
    ):
        if frozen_ready:
            state = "blocked"
            detail = (
                "frozen HNSW measurements exist, but this custom CPU has no "
                "reviewed hardware-model parameters; supply all five "
                "--hnsw-* values documented in docs/rag_cm/calibration.md"
            )
            commands = ()
        else:
            state = "needs_measure"
            detail = (
                "profile and frozen measurements are absent; raw measurement "
                "can run now, but fit requires all five reviewed --hnsw-* "
                "hardware-model values"
            )
            commands = _phase_commands(
                component,
                config_path,
                data_root,
                profile_root,
                frozen_ready=False,
            )[:1]
    elif frozen_ready:
        state = "ready_to_fit"
        detail = "frozen measurements exist; fitting needs no hardware"
        commands = _phase_commands(
            component, config_path, data_root, profile_root, frozen_ready=True
        )
    else:
        state = "needs_measure"
        detail = error or "profile and frozen measurements are absent"
        commands = _phase_commands(
            component, config_path, data_root, profile_root, frozen_ready=False
        )
    return CalibrationItem(
        name=component,
        state=state,
        required_for_deploy=True,
        profile_path=str(profile),
        frozen_path=str(frozen),
        detail=detail,
        commands=commands,
    )


def _builtin_cpu_spec_keys() -> frozenset[str]:
    """Read built-in CPU registry keys without importing the cost model."""
    source = Path(__file__).resolve().parent / "cost_model" / "cpu_specs.py"
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ValueError(f"cannot inspect production CPU registry: {exc}") from exc
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        if (
            isinstance(target, ast.Name)
            and target.id == "CPU_SPECS"
            and isinstance(value, ast.Dict)
        ):
            keys = {
                key.value
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if keys:
                return frozenset(keys)
    raise ValueError("production CPU_SPECS registry has no static keys")


def _custom_hnsw_parameters_required(
    document: Mapping[str, Any],
    cpu_key: str | None,
) -> bool:
    system = document.get("system")
    cpu = system.get("cpu") if isinstance(system, Mapping) else None
    return (
        isinstance(cpu, Mapping)
        and bool(cpu_key)
        and cpu_key not in _builtin_cpu_spec_keys()
    )


def generator_record_model_key(model: str) -> str:
    """Return the GenZ record identity used by the generator cost model."""
    return model.removesuffix("-Instruct")


def _generator_record_inventory_error(
    record: Any,
    *,
    gpu_key: str,
    model_key: str,
    data_root: Path | None,
) -> str | None:
    return _llm_record_validation_error(
        record,
        component="agnostic",
        model_key=model_key,
        gpu_key=gpu_key,
        bits="bf16",
    )


def _valid_llm_record(
    record: Any,
    gpu_key: str,
    model_key: str,
    data_root: Path | None = None,
) -> bool:
    return _generator_record_inventory_error(
        record,
        gpu_key=gpu_key,
        model_key=model_key,
        data_root=data_root,
    ) is None


def _generator_model_validation_errors(
    document: Mapping[str, Any] | None,
    gpu_key: str,
    models: Sequence[str],
    data_root: Path | None,
) -> dict[str, str]:
    errors: dict[str, str] = {}
    for model in models:
        model_key = generator_record_model_key(model)
        record = (document or {}).get(
            f"agnostic__{model_key}__bf16"
        )
        error = _generator_record_inventory_error(
            record,
            gpu_key=gpu_key,
            model_key=model_key,
            data_root=data_root,
        )
        if error is not None:
            errors[model] = error
    return errors


def _missing_generator_models(
    document: Mapping[str, Any] | None,
    gpu_key: str,
    models: Sequence[str],
    data_root: Path | None = None,
) -> list[str]:
    return list(_generator_model_validation_errors(
        document, gpu_key, models, data_root,
    ))


def _generator_commands(
    models: Sequence[str],
    config_path: Path,
    data_root: Path,
    profile_root: Path,
) -> tuple[str, ...]:
    flags = _root_flags(data_root, profile_root)
    return tuple(
        f"rag_stack calibrate generator --model {shlex.quote(model)} "
        f"--config {shlex.quote(str(config_path))} {flags}"
        for model in models
    )


def _generator_item(
    gpu_key: str,
    models: Sequence[str],
    config_path: Path,
    data_root: Path,
    profile_root: Path,
    llm_document: dict[str, Any] | None,
    llm_error: str | None,
) -> CalibrationItem:
    profile = profile_root / "llm_sim" / f"{gpu_key}.json"
    validation_errors = _generator_model_validation_errors(
        llm_document, gpu_key, models, data_root,
    )
    missing = list(validation_errors)
    if not missing:
        state, detail, commands = (
            "ready",
            f"{len(models)} requested generator record(s) are valid",
            (),
        )
    else:
        state = "invalid" if profile.is_file() else "needs_measure"
        detail = llm_error if profile.is_file() and llm_error else (
            "missing/invalid generator records: "
            + "; ".join(
                f"{model}: {validation_errors[model]}" for model in missing
            )
            if profile.is_file()
            else llm_error or "LLM profile is absent"
        )
        commands = _generator_commands(
            missing, config_path, data_root, profile_root
        )
    return CalibrationItem(
        name="generator",
        state=state,
        required_for_deploy=True,
        profile_path=str(profile),
        frozen_path=None,
        detail=detail,
        commands=commands,
    )


_REQUIRED_VLLM_STAGE_BATCH_PROFILES = (
    "generator",
    "query_expansion",
)
_DISAGG_STAGE_BATCH_PROFILES = (
    "generator_prefill",
    "generator_decode",
    "query_expansion_prefill",
    "query_expansion_decode",
)
_COMPLETE_STAGE_BATCH_PROFILES = (
    *_REQUIRED_VLLM_STAGE_BATCH_PROFILES,
    *_DISAGG_STAGE_BATCH_PROFILES,
)
def _stage_batch_block_error(
    block: Any,
    system_hardware: str,
    *,
    frozen_stage_raw: Path | None = None,
    require_disagg: bool = False,
) -> str | None:
    """Validate the authoritative hardware×stage×batch generation."""

    if not isinstance(block, Mapping):
        return "_stage_batch_curves is absent or not an object"
    if set(block) != {"system_hardware", "profiles"}:
        return "stage-batch block fields do not match the exact schema"
    if block.get("system_hardware") != system_hardware:
        return "stage-batch block hardware identity mismatch"
    profiles = block.get("profiles")
    if not isinstance(profiles, Mapping):
        return "stage-batch profiles must be an object"
    actual_profiles = set(profiles)
    collocated_profiles = set(_REQUIRED_VLLM_STAGE_BATCH_PROFILES)
    complete_profiles = set(_COMPLETE_STAGE_BATCH_PROFILES)
    if require_disagg and actual_profiles == collocated_profiles:
        return (
            "pipeline permits disaggregated placement but the stage profile "
            "contains only collocated curves; all six collocated/direct-P/D "
            "profiles are required"
        )
    if actual_profiles not in (collocated_profiles, complete_profiles):
        expected_profiles = (
            complete_profiles
            if actual_profiles.intersection(_DISAGG_STAGE_BATCH_PROFILES)
            else collocated_profiles
        )
        missing = sorted(expected_profiles - actual_profiles)
        extra = sorted(actual_profiles - expected_profiles)
        return (
            "stage-batch profiles must exactly cover either the collocated "
            "pair or complete six-profile generation "
            f"(missing={missing}, extra={extra})"
        )
    try:
        from rag_stack.cost_model.stage_batch_curve import (
            StageBatchCurveProfile,
            validate_stage_batch_curve_generation,
        )

        parsed_profiles = []
        for stage in sorted(actual_profiles):
            profile = StageBatchCurveProfile.from_dict(profiles[stage])
            if (
                profile.key.system_hardware != system_hardware
                or profile.key.stage != stage
            ):
                return f"stage-batch profile identity mismatch for {stage}"
            parsed_profiles.append(profile)
        validate_stage_batch_curve_generation(parsed_profiles)
    except (TypeError, ValueError) as exc:
        return f"invalid stage-batch profile: {exc}"
    return None


def _stage_batch_document_error(
    document: Any,
    system_hardware: str,
    *,
    frozen_stage_raw: Path | None = None,
    require_disagg: bool = False,
) -> str | None:
    """Mirror the runtime's exact standalone stage-profile root schema."""

    if not isinstance(document, Mapping):
        return "stage-batch profile root is absent or not an object"
    if set(document) != {"_stage_batch_curves"}:
        return (
            "stage-batch profile root must contain exactly "
            "'_stage_batch_curves'"
        )
    return _stage_batch_block_error(
        document.get("_stage_batch_curves"),
        system_hardware,
        frozen_stage_raw=frozen_stage_raw,
        require_disagg=require_disagg,
    )


def _vllm_stage_batch_item(
    system_hardware: str,
    config_path: Path,
    data_root: Path,
    profile_root: Path,
    stage_document: Mapping[str, Any] | None,
    stage_error: str | None,
    measurement_blocker: str | None,
    *,
    name: str,
    require_disagg: bool = False,
) -> CalibrationItem:
    frozen = (
        data_root
        / "generator_stage_batch"
        / f"{system_hardware}.jsonl"
    )
    profile = profile_root / "stage_batch" / f"{system_hardware}.json"
    error = _stage_batch_document_error(
        stage_document,
        system_hardware,
        frozen_stage_raw=frozen,
        require_disagg=require_disagg,
    )
    measure = (
        f"rag_stack calibrate generator-tp measure --config "
        f"{shlex.quote(str(config_path))} {_root_flags(data_root, profile_root)}"
    )
    force_measure = measure + " --force"
    fit = (
        f"rag_stack calibrate generator-tp fit --config "
        f"{shlex.quote(str(config_path))} {_root_flags(data_root, profile_root)}"
    )
    disagg_measure = (
        f"rag_stack calibrate generator-disagg measure --config "
        f"{shlex.quote(str(config_path))} {_root_flags(data_root, profile_root)}"
    )
    disagg_fit = (
        f"rag_stack calibrate generator-disagg fit --config "
        f"{shlex.quote(str(config_path))} {_root_flags(data_root, profile_root)}"
    )
    if error is None:
        state = "ready"
        detail = (
            "verified sparse hardware×PerformanceStage×resident-batch curves "
            "cover B1/8/32/64/256 with held-out B128"
        )
        commands: tuple[str, ...] = ()
    else:
        has_frozen_artifact, frozen_error = _stage_frozen_cohort_status(frozen)
        collocated_error = _stage_batch_document_error(
            stage_document,
            system_hardware,
            frozen_stage_raw=frozen,
            require_disagg=False,
        )
        if require_disagg and collocated_error is None:
            state = "needs_measure"
            detail = str(error)
            commands = (disagg_measure, disagg_fit)
        elif frozen_error is None:
            state = "ready_to_fit"
            detail = (
                f"{error}; verified independent stage/component frozen cohort "
                "is ready for CPU-only fit"
            )
            commands = (
                (fit, disagg_measure, disagg_fit)
                if require_disagg else (fit,)
            )
        elif measurement_blocker is not None:
            state = "needs_measure"
            detail = f"{error}; {frozen_error}; {measurement_blocker}"
            commands = ()
        elif has_frozen_artifact:
            state = "needs_measure"
            detail = f"{error}; {frozen_error}"
            # generator_parallel refuses to overwrite any cohort member unless
            # force is explicit; fitting a partial/mismatched cohort is invalid.
            commands = (force_measure,)
        else:
            state = "invalid" if profile.is_file() else "needs_measure"
            detail = f"{stage_error or error}; {frozen_error}"
            commands = (
                (measure, fit, disagg_measure, disagg_fit)
                if require_disagg else (measure, fit)
            )
    return CalibrationItem(
        name=name,
        state=state,
        required_for_deploy=True,
        profile_path=str(profile),
        frozen_path=str(frozen),
        detail=str(detail),
        commands=commands,
    )


def _generator_tp_item(
    system_hardware: str,
    config_path: Path,
    data_root: Path,
    profile_root: Path,
    stage_document: dict[str, Any] | None,
    stage_error: str | None,
    measurement_blocker: str | None,
    *,
    include_scheduler: bool = True,
) -> CalibrationItem:
    # The legacy scope name is retained for CLI compatibility.  It now means
    # one sparse hardware×PerformanceStage×resident-batch generation; no
    # model×TP×PP calibration artifact is accepted or required.
    del include_scheduler
    return _vllm_stage_batch_item(
        system_hardware,
        config_path,
        data_root,
        profile_root,
        stage_document,
        stage_error,
        measurement_blocker,
        name="generator-tp",
    )


def _scheduler_item(
    system_hardware: str,
    config_path: Path,
    data_root: Path,
    profile_root: Path,
    stage_document: Mapping[str, Any] | None,
    stage_error: str | None,
    measurement_blocker: str | None,
    *,
    require_disagg: bool,
) -> CalibrationItem:
    """Compatibility inventory item for the coarse vLLM stage curves."""
    return _vllm_stage_batch_item(
        system_hardware,
        config_path,
        data_root,
        profile_root,
        stage_document,
        stage_error,
        measurement_blocker,
        name="scheduler-vllm",
        require_disagg=require_disagg,
    )


def _candidate_strings(value: Any) -> tuple[str, ...]:
    """Flatten the small categorical shapes used by full RAG YAML configs."""
    values: list[str] = []

    def add(candidate: Any) -> None:
        if isinstance(candidate, str) and candidate.strip():
            text = candidate.strip()
            if text not in values:
                values.append(text)
        elif isinstance(candidate, Sequence) and not isinstance(
            candidate, (str, bytes)
        ):
            for item in candidate:
                add(item)
        elif isinstance(candidate, Mapping):
            # ``ordered`` is the paper model ladder; values/choices cover the
            # generic categorical spellings accepted by config tooling.
            for key in ("ordered", "values", "choices", "value"):
                if key in candidate:
                    add(candidate[key])

    add(value)
    return tuple(values)


def _pipeline_requires_disagg_stage_profiles(
    document: Mapping[str, Any],
) -> bool:
    """Mirror the CM sweep's placement-policy default and normalization."""
    system = document.get("system")
    search_space = (
        system.get("cm_search_space")
        if isinstance(system, Mapping)
        else None
    )
    if not isinstance(search_space, Mapping):
        return True
    policies = _candidate_strings(
        search_space.get("placement_policy", "disaggregated")
    )
    return "disaggregated" in policies


def _genz_model_map_keys() -> frozenset[str]:
    """Read registry keys from source AST; never import GenZ/ModelConfig."""
    path = (
        Path(__file__).resolve().parent
        / "cost_model/llm_sim/genz_model_map.py"
    )
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, SyntaxError):
        return frozenset()
    for node in tree.body:
        target = node.target if isinstance(node, ast.AnnAssign) else None
        value = node.value if isinstance(node, ast.AnnAssign) else None
        if (
            isinstance(target, ast.Name)
            and target.id == "GENZ_MODEL_MAP"
            and isinstance(value, ast.Dict)
        ):
            return frozenset(
                str(key.value).lower()
                for key in value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            )
    return frozenset()


def _pipeline_node_lines(pipeline: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Collect direct and method-nested node lines without runtime imports."""
    result: list[Mapping[str, Any]] = []

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            node_lines = value.get("node_lines")
            if isinstance(node_lines, list):
                result.extend(item for item in node_lines if isinstance(item, Mapping))
            for key, child in value.items():
                if key != "node_lines":
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(pipeline)
    return tuple(result)


def _pipeline_nodes(pipeline: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        node
        for node_line in _pipeline_node_lines(pipeline)
        for node in (node_line.get("nodes") or [])
        if isinstance(node, Mapping)
    )


def _module_options(node: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    modules = node.get("modules") or []
    if isinstance(modules, Mapping):
        modules = [modules]
    return tuple(item for item in modules if isinstance(item, Mapping))


def _vectordb_selector_names(value: Any) -> tuple[str, ...]:
    direct = _candidate_strings(value)
    if direct:
        return direct
    if not isinstance(value, Mapping):
        return ()
    # Dict-form selectors use the vectordb names as keys and nest each
    # family's query-time knobs below it.
    return tuple(
        str(key).strip() for key in value
        if isinstance(key, str) and key.strip()
    )


def _pipeline_gpu_cap(document: Mapping[str, Any], gpu_count: int | None) -> int | None:
    system = document.get("system")
    if not isinstance(system, Mapping):
        return gpu_count
    blocks = (
        system.get("cm_search_space"),
        system.get("system_design_space"),
        system.get("measured_search_space"),
        system.get("search_space"),
    )
    for field in ("max_num_gpus_per_stage", "max_num_gpus"):
        for block in blocks:
            if not isinstance(block, Mapping):
                continue
            value = block.get(field)
            if _positive_int(value):
                return int(value)
    return gpu_count


def _llm_record_validation_error(
    record: Any,
    *,
    component: str,
    model_key: str,
    gpu_key: str,
    bits: str,
) -> str | None:
    """Validate one self-contained serialized pricing record."""

    if not isinstance(record, Mapping):
        return "record is absent or not an object"
    unknown = sorted(set(record) - LLM_RECORD_ALLOWED_FIELDS)
    if unknown:
        return f"record has unknown fields: {unknown}"
    for field, expected in {
        "component": component,
        "model_key": model_key,
        "hardware_key": gpu_key,
        "bits": bits,
    }.items():
        if record.get(field) != expected:
            return f"record {field} must be {expected!r}"
    if not _finite_number(record.get("fit_mape")):
        return "record fit_mape must be finite and non-negative"
    if not _finite_any_number(record.get("correlation")) or abs(
        float(record["correlation"])
    ) > 1.0:
        return "record correlation must be finite and in [-1, 1]"
    if "max_seq_len" in record and record.get("max_seq_len") is not None and not _positive_int(
        record.get("max_seq_len")
    ):
        return "record max_seq_len must be a positive integer"

    grid_fields = {
        "batch_size", "seq_len", "measured_s", "predicted_s", "rel_err",
    }

    def grid_error(name: str, points: Any, *, required: bool) -> str | None:
        if not isinstance(points, list) or (required and not points):
            return f"record {name} must contain standalone measurements"
        for index, point in enumerate(points):
            if not isinstance(point, Mapping) or set(point) != grid_fields:
                return (
                    f"record {name}[{index}] fields do not match standalone schema"
                )
            batch = point.get("batch_size")
            if (
                isinstance(batch, bool)
                or not isinstance(batch, (int, float))
                or not math.isfinite(float(batch))
                or float(batch) != float(int(batch))
                or int(batch) < 1
            ):
                return (
                    f"record {name}[{index}].batch_size must be a positive integer"
                )
            seq_len = point.get("seq_len")
            axes = seq_len if isinstance(seq_len, (list, tuple)) else (seq_len,)
            if not axes or any(
                isinstance(axis, bool)
                or not isinstance(axis, (int, float))
                or not math.isfinite(float(axis))
                or float(axis) != float(int(axis))
                or int(axis) < 1
                for axis in axes
            ):
                return (
                    f"record {name}[{index}].seq_len must be a positive integer "
                    "or integer tuple"
                )
            for field in ("measured_s", "predicted_s"):
                if not _finite_number(point.get(field), positive=True):
                    return (
                        f"record {name}[{index}].{field} must be finite and positive"
                    )
            if not _finite_any_number(point.get("rel_err")):
                return f"record {name}[{index}].rel_err must be finite"
        return None

    error = grid_error("grid", record.get("grid"), required=True)
    if error is not None:
        return error
    validation_grid = record.get("validation_grid") or []
    error = grid_error("validation_grid", validation_grid, required=False)
    if error is not None:
        return error

    contract = (
        record.get("calibration_contract")
        or _LEGACY_INTERNAL_COMPONENT_CONTRACT
    )
    has_compute = record.get("compute_efficiency") is not None
    has_memory = record.get("memory_efficiency") is not None
    linear = record.get("linear_law")
    if has_compute != has_memory:
        return "record must provide both compute/memory efficiencies"

    if has_compute:
        if (
            component == "agnostic"
            and contract != _GENERATOR_ROOFLINE_CALIBRATION_CONTRACT
        ):
            return (
                "generator roofline record calibration_contract must be "
                f"{_GENERATOR_ROOFLINE_CALIBRATION_CONTRACT!r}"
            )
        if float(record["fit_mape"]) > 0.20:
            return "roofline record fit_mape must be <= 0.20"
        for field in ("compute_efficiency", "memory_efficiency"):
            if not _finite_number(record.get(field), positive=True) or not (
                0.05 <= float(record[field]) <= 1.0
            ):
                return f"record {field} is outside fitter bounds [0.05, 1]"
        if linear is not None:
            return "roofline record cannot also carry linear_law"
        overhead = record.get("per_call_overhead_s", 0.0)
        if not _finite_number(overhead):
            return "record per_call_overhead_s must be finite and non-negative"
        if (
            contract == _GENERATOR_ROOFLINE_CALIBRATION_CONTRACT
            and float(overhead) != 0.0
        ):
            return "zero-intercept roofline record requires zero per-call overhead"
        rank = record.get("identifiability_rank")
        condition = record.get("identifiability_condition")
        if rank != 2:
            return "roofline record must prove identifiability_rank=2"
        if (
            not _finite_number(condition, positive=True)
            or float(condition) < 1.0
            or float(condition) > 100.0
        ):
            return "roofline record identifiability_condition must be in [1, 100]"
        validation_mape = record.get("validation_mape")
        if not _finite_number(validation_mape) or float(validation_mape) > 0.20:
            return "roofline record held-out validation_mape must be <= 0.20"
        if not validation_grid:
            return "roofline record requires held-out validation_grid"
    else:
        required = {"fixed_ms", "per_item_ms", "per_token_ms"}
        if contract == _LINEAR_LAW_COMPONENT_CONTRACT:
            required.add("fit_mape")
        elif contract == _CAPACITY_HINGE_LAW_COMPONENT_CONTRACT:
            required |= {
                "capacity_tokens",
                "overload_step_ms",
                "overload_per_token_ms",
                "fit_mape",
            }
        exact_contract = contract in {
            _LINEAR_LAW_COMPONENT_CONTRACT,
            _CAPACITY_HINGE_LAW_COMPONENT_CONTRACT,
        }
        valid_fields = (
            isinstance(linear, Mapping)
            and (
                set(linear) == required
                if exact_contract
                else {"fixed_ms", "per_item_ms", "per_token_ms"}.issubset(
                    linear
                )
            )
        )
        if not valid_fields:
            return (
                f"{contract} linear_law fields must be exactly "
                f"{sorted(required)}"
                if exact_contract
                else "record without efficiencies requires a complete linear_law"
            )
        for field in ("fixed_ms", "per_item_ms", "per_token_ms"):
            if not _finite_number(linear.get(field)):
                return f"linear_law.{field} must be finite and non-negative"
        if contract == _LINEAR_LAW_COMPONENT_CONTRACT:
            if not _finite_number(linear.get("fit_mape")):
                return "linear_law.fit_mape must be finite and non-negative"
            if float(linear["fit_mape"]) != float(record["fit_mape"]):
                return "linear_law.fit_mape must equal record fit_mape"
            if float(record["fit_mape"]) > 0.20:
                return "linear-law record fit_mape must be <= 0.20"
            if float(record.get("per_call_overhead_s", 0.0) or 0.0) != 0.0:
                return "linear-law record per_call_overhead_s must be zero"
            if record.get("identifiability_rank") is not None or record.get(
                "identifiability_condition"
            ) is not None:
                return "linear-law record cannot carry roofline identifiability"
            validation_mape = record.get("validation_mape")
            if (
                not _finite_number(validation_mape)
                or float(validation_mape) > 0.20
                or not validation_grid
            ):
                return (
                    "linear-law record requires held-out validation_mape "
                    "<= 0.20 and validation_grid"
                )
        elif contract == _CAPACITY_HINGE_LAW_COMPONENT_CONTRACT:
            for field in (
                "capacity_tokens",
                "overload_step_ms",
                "overload_per_token_ms",
            ):
                if not _finite_number(linear.get(field), positive=True):
                    return f"linear_law.{field} must be finite and positive"
            if float(linear["fit_mape"]) != float(record["fit_mape"]):
                return "capacity-hinge linear_law.fit_mape must equal record fit_mape"
            if float(record["fit_mape"]) > 0.20:
                return "capacity-hinge record fit_mape must be <= 0.20"
            if record.get("identifiability_rank") is not None or record.get(
                "identifiability_condition"
            ) is not None:
                return "capacity-hinge record cannot carry roofline identifiability"
            validation_mape = record.get("validation_mape")
            if (
                not _finite_number(validation_mape)
                or float(validation_mape) > 0.20
                or not validation_grid
            ):
                return (
                    "capacity-hinge record requires held-out validation_mape "
                    "<= 0.20 and validation_grid"
                )
        elif "fit_mape" in linear and not _finite_number(
            linear.get("fit_mape")
        ):
            return "linear_law.fit_mape must be finite and non-negative"

    def point_identity(point: Mapping[str, Any]) -> tuple[int, tuple[int, ...]]:
        seq_len = point["seq_len"]
        axes = seq_len if isinstance(seq_len, (list, tuple)) else (seq_len,)
        return int(point["batch_size"]), tuple(int(axis) for axis in axes)

    overlap = {
        point_identity(point) for point in record["grid"]
    }.intersection(point_identity(point) for point in validation_grid)
    if overlap:
        return "record training and held-out grids overlap"
    if (
        has_compute
        and component == "agnostic"
        and contract == _GENERATOR_ROOFLINE_CALIBRATION_CONTRACT
    ):
        b128_shapes = {
            point_identity(point)[1]
            for point in record["grid"]
            if int(point["batch_size"]) == 128
        }
        b256_shapes = {
            point_identity(point)[1]
            for point in validation_grid
            if int(point["batch_size"]) == 256
        }
        if not b128_shapes.intersection(b256_shapes):
            return (
                "generator roofline profile requires an exact same-token-shape "
                "B128 fit/B256 held-out pair"
            )
    return None


def _find_exact_llm_record(
    document: Mapping[str, Any] | None,
    *,
    component: str,
    model_key: str,
    gpu_key: str,
    bits: str | None,
) -> tuple[str | None, str | None]:
    """Return ``(record_key, error)`` using production's precision semantics."""
    if not isinstance(document, Mapping):
        return None, "LLM profile is absent or not an object"
    prefix = f"{component}__{model_key}__"
    if bits is not None:
        keys = [prefix + bits]
    else:
        keys = sorted(
            key for key in document
            if isinstance(key, str)
            and key.startswith(prefix)
            and "__" not in key[len(prefix):]
        )
        if len(keys) > 1:
            return None, (
                f"ambiguous precision for {component}/{model_key}: {keys}; "
                "declare module.bits"
            )
    if not keys or keys[0] not in document:
        precision = bits if bits is not None else "an unambiguous precision"
        return None, (
            f"missing exact record {component}/{model_key}@{gpu_key} "
            f"with {precision}"
        )
    key = keys[0]
    actual_bits = key[len(prefix):]
    error = _llm_record_validation_error(
        document.get(key), component=component, model_key=model_key,
        gpu_key=gpu_key, bits=actual_bits,
    )
    return key, error


def _pipeline_llm_item(
    *,
    kind: str,
    component: str,
    model_key: str,
    bits: str | None,
    cost_model: str,
    gpu_key: str,
    config_path: Path,
    data_root: Path,
    profile_root: Path,
    llm_document: Mapping[str, Any] | None,
) -> CalibrationItem:
    profile = profile_root / "llm_sim" / f"{gpu_key}.json"
    record_key, error = _find_exact_llm_record(
        llm_document, component=component, model_key=model_key,
        gpu_key=gpu_key, bits=bits,
    )
    display_bits = bits or "unique-precision"
    name = f"{kind}:{component}:{model_key}@{display_bits}"
    flags = _root_flags(data_root, profile_root)
    grid_flags = (
        "--batch-sizes 1,8,32,64,128,256 "
        + (
            "--seq-lens 16,32,64,128,256"
            if kind == "embedding"
            else "--seq-lens 128,256,512"
        )
    )
    command = (
        f"rag_stack calibrate llm --config {shlex.quote(str(config_path))} "
        f"--cost-model {shlex.quote(cost_model)} "
        f"--module {shlex.quote(model_key if kind == 'embedding' else component)} "
        f"{grid_flags} {flags}"
    )
    if error is None:
        state = "ready"
        detail = f"exact GPU/component record is valid: {record_key}"
        commands: tuple[str, ...] = ()
    else:
        state = "invalid" if profile.is_file() else "needs_measure"
        detail = error
        commands = (command,)
    return CalibrationItem(
        name=name,
        state=state,
        required_for_deploy=True,
        profile_path=str(profile),
        frozen_path=None,
        detail=detail,
        commands=commands,
    )


def _pipeline_generator_item(
    gpu_key: str,
    models: Sequence[str],
    config_path: Path,
    data_root: Path,
    profile_root: Path,
    llm_document: Mapping[str, Any] | None,
    llm_error: str | None,
) -> CalibrationItem:
    """Strict full-pipeline variant; legacy selected generator stays unchanged."""
    profile = profile_root / "llm_sim" / f"{gpu_key}.json"
    errors: list[str] = []
    missing: list[str] = []
    record_keys: list[str] = []
    for model in models:
        model_key = generator_record_model_key(model)
        key, error = _find_exact_llm_record(
            llm_document,
            component="agnostic",
            model_key=model_key,
            gpu_key=gpu_key,
            bits="bf16",
        )
        if error is not None:
            missing.append(model)
            errors.append(f"{model}: {error}")
        elif key is not None:
            record_keys.append(key)
    if not errors:
        state = "ready"
        detail = f"{len(record_keys)} exact generator record(s) pass runtime schema"
        commands: tuple[str, ...] = ()
    else:
        state = "invalid" if profile.is_file() else "needs_measure"
        detail = (
            llm_error or "; ".join(errors)
            if profile.is_file()
            else llm_error or "LLM profile is absent"
        )
        commands = _generator_commands(
            missing, config_path, data_root, profile_root,
        )
    return CalibrationItem(
        name="generator",
        state=state,
        required_for_deploy=True,
        profile_path=str(profile),
        frozen_path=None,
        detail=detail,
        commands=commands,
    )


def llmlingua2_profile_validation_error(
    document: Mapping[str, Any] | None,
    *,
    gpu_key: str,
    cpu_key: str,
) -> str | None:
    """Mirror the aggregate schema/identity checks using JSON metadata only."""
    if not isinstance(document, Mapping):
        return "profile root must be a JSON object"
    unknown = sorted(set(document) - LLMLINGUA2_ALLOWED_FIELDS)
    if unknown:
        return f"unknown LLMLingua2 profile fields: {unknown}"
    expected = {
        "profile_kind": "compressor_llmlingua2",
        "service_unit": "milliseconds_per_fused_compressor_call",
        "model": "llmlingua2",
        "hardware_key": gpu_key,
        "cpu_hardware_key": cpu_key,
    }
    for field, value in expected.items():
        if document.get(field) != value:
            return f"{field} must be {value!r}, got {document.get(field)!r}"
    if tuple(document.get("demand_axes") or ()) != ("n_chunks", "chunk_tokens"):
        return "demand_axes must be ['n_chunks', 'chunk_tokens']"
    for field, positive in (
        ("fixed_ms", False), ("per_chunk_ms", True),
        ("per_token_ms", False), ("fit_mape", False),
    ):
        if not _finite_number(document.get(field), positive=positive):
            return f"{field} must be finite and {'positive' if positive else 'non-negative'}"
    if not _finite_any_number(document.get("correlation")) or abs(
        float(document["correlation"])
    ) > 1.0:
        return "correlation must be finite and in [-1, 1]"
    grid = document.get("grid")
    if not isinstance(grid, list) or not grid:
        return "grid must contain standalone measurements"
    for index, row in enumerate(grid):
        if not isinstance(row, Mapping):
            return f"grid[{index}] must be an object"
        unknown_grid = sorted(set(row) - LLMLINGUA2_GRID_FIELDS)
        if unknown_grid:
            return f"grid[{index}] has unknown fields: {unknown_grid}"
        for field, positive in (
            ("n_chunks", True), ("chunk_tokens", True),
            ("measured_ms", True), ("predicted_ms", False),
        ):
            if not _finite_number(row.get(field), positive=positive):
                return f"grid[{index}].{field} is invalid"
        if not _finite_any_number(row.get("rel_err")):
            return f"grid[{index}].rel_err must be finite"
    return None


def _llmlingua2_item(
    cpu_key: str,
    gpu_key: str,
    config_path: Path,
    data_root: Path,
    profile_root: Path,
) -> CalibrationItem:
    profile = profile_root / "compressor_llmlingua2" / f"{gpu_key}.json"
    document, read_error = _read_json(profile)
    error = read_error or llmlingua2_profile_validation_error(
        document, gpu_key=gpu_key, cpu_key=cpu_key,
    )
    command = (
        f"rag_stack calibrate llmlingua2 --config "
        f"{shlex.quote(str(config_path))} {_root_flags(data_root, profile_root)}"
    )
    if error is None:
        state = "ready"
        detail = "exact GPU+CPU LLMLingua2 schema and fit grid are valid"
        commands: tuple[str, ...] = ()
    else:
        state = "invalid" if profile.is_file() else "needs_measure"
        detail = error
        commands = (command,)
    return CalibrationItem(
        name="llmlingua2",
        state=state,
        required_for_deploy=True,
        profile_path=str(profile),
        frozen_path=None,
        detail=detail,
        commands=commands,
    )


def _ivf_workload_key(dataset_name: str, embedding_model: str) -> str:
    def sanitize(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_]+", "-", value).strip("-_") or "x"
    return f"{sanitize(dataset_name)}__{sanitize(embedding_model)}"


def _hnsw_workload_key(dataset_name: str, embedding_model: str) -> str:
    def sanitize(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    return f"{sanitize(dataset_name)}__{sanitize(embedding_model)}"


def _ivf_workload_validation_error(document: Mapping[str, Any] | None) -> str | None:
    try:
        profile = unwrap_ivf_imbalance_profile(document)
    except ValueError as exc:
        return str(exc)
    if not _positive_int(profile.get("d")) or not _positive_int(profile.get("N")):
        return "IVF workload profile requires positive d and N"
    curves = profile.get("nlist")
    if not isinstance(curves, Mapping) or not curves:
        return "IVF workload profile has no nlist curve"
    for nlist, curve in curves.items():
        try:
            valid_nlist = int(nlist) > 0
        except (TypeError, ValueError):
            valid_nlist = False
        if not valid_nlist or not isinstance(curve, Mapping) or not curve:
            return "IVF workload nlist curve identity/content is invalid"
        for nprobe, factor in curve.items():
            try:
                valid_nprobe = int(nprobe) > 0
            except (TypeError, ValueError):
                valid_nprobe = False
            if not valid_nprobe or not _finite_number(factor, positive=True):
                return "IVF workload curve contains invalid nprobe/factor"
    return None


def _workload_item(
    family: str,
    dataset_name: str,
    embedding_model: str,
    corpus_identity: str | None,
    config_path: Path,
    data_root: Path,
    profile_root: Path,
    qa_data_path: str | None,
    corpus_data_path: str | None,
) -> CalibrationItem:
    if family == "ivf":
        key = _ivf_workload_key(dataset_name, embedding_model)
        profile = profile_root / "faiss_ivf" / "imbalance" / f"{key}.json"
        document, read_error = _read_json(profile)
        error = read_error or _ivf_workload_validation_error(document)
    else:
        key = _hnsw_workload_key(dataset_name, embedding_model)
        profile = profile_root / "faiss_hnsw" / f"{key}.joblib"
        if not profile.is_file():
            error = "file is absent"
        else:
            try:
                error = None if profile.stat().st_size > 0 else "file is empty"
            except OSError as exc:
                error = f"cannot stat profile: {exc}"
    project = data_root / "workload_projects" / re.sub(
        r"[^A-Za-z0-9_.-]+", "-", dataset_name
    ).strip("-.")
    source_flags = ""
    if qa_data_path and corpus_data_path:
        repo_root = Path(__file__).resolve().parents[1]

        def absolute_source(value: str) -> Path:
            path = Path(value).expanduser()
            if path.is_absolute():
                return path.resolve()
            repo_candidate = (repo_root / path).resolve()
            config_candidate = (config_path.parent / path).resolve()
            return (
                repo_candidate
                if repo_candidate.exists() or not config_candidate.exists()
                else config_candidate
            )

        source_flags = (
            f"--qa-data {shlex.quote(str(absolute_source(qa_data_path)))} "
            f"--corpus-data {shlex.quote(str(absolute_source(corpus_data_path)))} "
        )
    command = (
        f"rag_stack calibrate workload --config {shlex.quote(str(config_path))} "
        f"--project-dir {shlex.quote(str(project))} {source_flags}"
        f"{_root_flags(data_root, profile_root)}"
    )
    identity = (
        f"dataset={dataset_name!r}, corpus={corpus_identity or '<unresolved>'!r}, "
        f"embedding={embedding_model!r}, key={key!r}"
    )
    if error is None:
        state = "ready"
        detail = f"exact corpus×embedding identity exists ({identity})"
        commands: tuple[str, ...] = ()
    else:
        state = "invalid" if profile.is_file() else "needs_measure"
        detail = f"{error}; required exact identity: {identity}; no generic fallback"
        commands = (command,)
    return CalibrationItem(
        name=f"{family}-workload:{key}",
        state=state,
        required_for_deploy=True,
        profile_path=str(profile),
        frozen_path=None,
        detail=detail,
        commands=commands,
    )


def _pipeline_config_item(
    config_path: Path,
    detail: str,
    *,
    name: str = "pipeline-config",
) -> CalibrationItem:
    return CalibrationItem(
        name=name,
        state="invalid",
        required_for_deploy=True,
        profile_path=str(config_path),
        frozen_path=None,
        detail=detail,
        commands=(),
    )


def _missing_lexical_profile_item(
    component: str,
    cpu_key: str | None,
    profile_root: Path,
) -> CalibrationItem:
    identity = cpu_key or "missing-cpu-key"
    return CalibrationItem(
        name=f"lexical-retrieval:{component}",
        state="invalid",
        required_for_deploy=True,
        profile_path=str(
            profile_root / "lexical_retrieval" / f"{identity}.json"
        ),
        frozen_path=None,
        detail=(
            f"selectable lexical component {component!r} has no standalone "
            "exact-CPU calibration/profile contract; hardcoded service is not "
            "deployment evidence"
        ),
        commands=(),
    )


def _pipeline_items(
    document: Mapping[str, Any],
    *,
    system_hardware: str | None,
    cpu_key: str | None,
    gpu_key: str | None,
    gpu_count: int | None,
    config_path: Path,
    data_root: Path,
    profile_root: Path,
    llm_document: Mapping[str, Any] | None,
    llm_error: str | None,
    stage_document: Mapping[str, Any] | None,
    stage_error: str | None,
    custom_hnsw_parameters_required: bool,
) -> tuple[CalibrationItem, ...]:
    """Expand every selectable production path in a full RAG config."""
    items: list[CalibrationItem] = []
    errors: list[str] = []
    algo = document.get("algo_search_space")
    dataset = document.get("dataset")
    if not isinstance(algo, Mapping):
        return (_pipeline_config_item(
            config_path, "pipeline scope requires mapping algo_search_space",
        ),)
    if not isinstance(dataset, Mapping):
        return (_pipeline_config_item(
            config_path, "pipeline scope requires mapping dataset",
        ),)
    pipeline = algo.get("pipeline")
    if not isinstance(pipeline, Mapping):
        return (_pipeline_config_item(
            config_path, "pipeline scope requires algo_search_space.pipeline",
        ),)
    dataset_names = _candidate_strings(dataset.get("dataset_name"))
    if not dataset_names:
        errors.append("dataset.dataset_name has no auditable identity")
    corpus_candidates = _candidate_strings(dataset.get("corpus"))
    corpus_identity = corpus_candidates[0] if len(corpus_candidates) == 1 else None
    if corpus_identity is None:
        errors.append("dataset.corpus has no single auditable source identity")
    qa_candidates = _candidate_strings(dataset.get("qa"))
    qa_identity = qa_candidates[0] if len(qa_candidates) == 1 else None
    if qa_identity is None:
        errors.append("dataset.qa has no single auditable source identity")
    raw_vectordbs = algo.get("vectordb")
    if not isinstance(raw_vectordbs, list):
        errors.append("algo_search_space.vectordb must be a list")
        raw_vectordbs = []
    vectordbs: dict[str, Mapping[str, Any]] = {}
    for entry in raw_vectordbs:
        if not isinstance(entry, Mapping):
            errors.append("algo_search_space.vectordb contains a non-object entry")
            continue
        names = _candidate_strings(entry.get("name"))
        if len(names) != 1:
            errors.append("each vectordb definition needs one scalar name")
            continue
        vectordbs[names[0]] = entry

    nodes = _pipeline_nodes(pipeline)
    if not nodes:
        errors.append("pipeline contains no readable node_lines")
    lexical_components: set[str] = set()
    for node in nodes:
        if node.get("stage") not in ("lexical_retrieval", "hybrid_retrieval"):
            continue
        for module in _module_options(node):
            lexical_components.update(
                component
                for component in _candidate_strings(module.get("component"))
                if component != "vectordb"
            )
    if lexical_components and not cpu_key:
        errors.append("lexical retrieval paths require config.system.cpu hardware_key")
    for component in sorted(lexical_components):
        items.append(_missing_lexical_profile_item(
            component, cpu_key, profile_root,
        ))
    selected_names: list[str] = []
    for node in nodes:
        if node.get("stage") not in ("semantic_retrieval", "hybrid_retrieval"):
            continue
        for module in _module_options(node):
            if module.get("component") != "vectordb" and "vectordb" not in module:
                continue
            for name in _vectordb_selector_names(module.get("vectordb")):
                if name not in selected_names:
                    selected_names.append(name)
    if not selected_names and len(vectordbs) == 1:
        selected_names.extend(vectordbs)
    if not selected_names:
        errors.append("pipeline retrieval does not select an auditable vectordb")

    selected_vdbs: list[Mapping[str, Any]] = []
    for name in selected_names:
        entry = vectordbs.get(name)
        if entry is None:
            errors.append(f"pipeline selects undefined vectordb {name!r}")
        else:
            selected_vdbs.append(entry)

    if (selected_vdbs or any(node.get("stage") == "passage_reranker" for node in nodes)) and not gpu_key:
        errors.append("pipeline GPU components require config.system.gpu hardware_key")
    if any(
        str(entry.get("db_type") or "").startswith("faiss_")
        for entry in selected_vdbs
    ) and not cpu_key:
        errors.append("FAISS pipeline paths require config.system.cpu hardware_key")

    embeddings: set[str] = set()
    family_workloads: set[tuple[str, str, str]] = set()
    required_faiss: set[str] = set()
    for entry in selected_vdbs:
        db_type = str(entry.get("db_type") or "")
        models = _candidate_strings(entry.get("embedding_model"))
        if not models:
            errors.append(
                f"vectordb {entry.get('name')!r} has no auditable embedding_model"
            )
        embeddings.update(models)
        if db_type == "faiss_ivf":
            family = "ivf"
        elif db_type == "faiss_hnsw":
            family = "hnsw"
        else:
            errors.append(
                f"selected vectordb {entry.get('name')!r} has unsupported "
                f"cost-model db_type {db_type!r}"
            )
            continue
        required_faiss.add(family)
        family_workloads.update(
            (family, dataset_name, model)
            for dataset_name in dataset_names
            for model in models
        )

    mapped_models = _genz_model_map_keys()
    for embedding_model in sorted(embeddings):
        if embedding_model.lower() not in mapped_models:
            errors.append(
                f"embedding_model {embedding_model!r} has no GENZ_MODEL_MAP "
                "entry; an exact record cannot make an unresolved encoder "
                "production-ready"
            )

    if cpu_key:
        for family in sorted(required_faiss):
            items.append(
                _faiss_item(
                    family,
                    cpu_key,
                    config_path,
                    data_root,
                    profile_root,
                    custom_hnsw_parameters_required=(
                        custom_hnsw_parameters_required and family == "hnsw"
                    ),
                )
            )
    for family, dataset_name, embedding_model in sorted(family_workloads):
        items.append(_workload_item(
            family, dataset_name, embedding_model, corpus_identity,
            config_path, data_root, profile_root,
            qa_identity, corpus_identity,
        ))
    if gpu_key:
        for embedding_model in sorted(embeddings):
            items.append(_pipeline_llm_item(
                kind="embedding", component="agnostic",
                model_key=embedding_model, bits=None,
                cost_model="retrieval_encode_sim", gpu_key=gpu_key,
                config_path=config_path, data_root=data_root,
                profile_root=profile_root, llm_document=llm_document,
            ))

    reranker_requirements: set[tuple[str, str, str | None, str]] = set()
    generator_models: set[str] = set()
    query_expansion_models: set[str] = set()
    needs_retrieval_worker = False
    needs_llmlingua2 = False
    generator_stage_seen = False
    for node in nodes:
        stage = node.get("stage")
        modules = _module_options(node)
        if stage == "passage_reranker":
            for module in modules:
                components = _candidate_strings(module.get("component"))
                for component in components:
                    if component in UNPRICED_RERANKERS:
                        continue
                    cost_model = RERANKER_COST_MODELS.get(component)
                    if cost_model is None:
                        errors.append(
                            f"reranker {component!r} has no metadata-only exact "
                            "profile identity mapping"
                        )
                        continue
                    explicit_models = _candidate_strings(module.get("model"))
                    if explicit_models:
                        models = explicit_models
                    else:
                        default = RERANKER_DEFAULT_MODELS.get(component)
                        models = (default,) if default else ()
                    if not models:
                        errors.append(
                            f"reranker {component!r} has no resolved model identity"
                        )
                        continue
                    bit_values = _candidate_strings(module.get("bits")) or (None,)
                    for model in models:
                        for bits in bit_values:
                            reranker_requirements.add(
                                (component, model, bits, cost_model)
                            )
        elif stage == "passage_compressor":
            for module in modules:
                for component in _candidate_strings(module.get("component")):
                    if component == "llmlingua2":
                        needs_llmlingua2 = True
                    else:
                        errors.append(
                            f"compressor {component!r} is selectable but pipeline "
                            "inventory has no exact profile contract for it"
                        )
        elif stage == "query_expansion":
            for module in modules:
                components = _candidate_strings(module.get("component"))
                for component in components:
                    fanout = QUERY_EXPANSION_DEFAULT_FANOUT.get(component)
                    if fanout is None:
                        errors.append(
                            f"query expansion {component!r} has unknown retrieval fanout"
                        )
                    elif fanout > 1:
                        needs_retrieval_worker = True
                models = _candidate_strings(module.get("model"))
                if not models:
                    errors.append("query-expansion module has no resolved model")
                configured_bits = _candidate_strings(module.get("bits"))
                if configured_bits and configured_bits != ("bf16",):
                    errors.append(
                        "query-expansion precision search is not covered by "
                        f"the bf16 generator profile contract: {configured_bits}"
                    )
                query_expansion_models.update(models)
        elif stage == "generator":
            generator_stage_seen = True
            for module in modules:
                models = _candidate_strings(module.get("model"))
                if not models:
                    errors.append("generator module has no resolved model")
                configured_bits = _candidate_strings(module.get("bits"))
                if configured_bits and configured_bits != ("bf16",):
                    errors.append(
                        "generator precision search is not covered by the "
                        f"bf16 generator profile contract: {configured_bits}"
                    )
                generator_models.update(models)

    if not generator_stage_seen:
        errors.append("pipeline contains no generator stage")
    if gpu_key:
        for component, model, bits, cost_model in sorted(
            reranker_requirements,
            key=lambda value: tuple("" if item is None else item for item in value),
        ):
            items.append(_pipeline_llm_item(
                kind="reranker", component=component, model_key=model,
                bits=bits, cost_model=cost_model, gpu_key=gpu_key,
                config_path=config_path, data_root=data_root,
                profile_root=profile_root, llm_document=llm_document,
            ))
    if needs_llmlingua2 and cpu_key and gpu_key:
        items.append(_llmlingua2_item(
            cpu_key, gpu_key, config_path, data_root, profile_root,
        ))
    if needs_retrieval_worker and cpu_key:
        items.append(_retrieval_worker_item(
            cpu_key, config_path, data_root, profile_root,
        ))

    all_llm_models = tuple(sorted(generator_models | query_expansion_models))
    if gpu_key and all_llm_models:
        items.append(_pipeline_generator_item(
            gpu_key, all_llm_models, config_path, data_root, profile_root,
            llm_document, llm_error,
        ))
    if generator_models and system_hardware:
        # TP/PP is analytical and never changes calibration readiness.  One
        # stage-batch campaign is required even on a single-GPU box because it
        # prices runtime service rather than a multi-GPU deployment.
        items.append(_scheduler_item(
            system_hardware, config_path, data_root, profile_root,
            stage_document, stage_error,
            _stage_measurement_blocker(gpu_key, gpu_count),
            require_disagg=_pipeline_requires_disagg_stage_profiles(document),
        ))
    elif generator_models:
        errors.append(
            "generator stage calibration requires "
            "config.system.system_hardware"
        )

    if errors:
        items.append(_pipeline_config_item(config_path, "; ".join(errors)))
    return tuple(items)


def _normalize_scopes(scopes: Iterable[str] | None) -> tuple[str, ...]:
    requested = tuple(scopes or CORE_SCOPES)
    expanded: list[str] = []
    for scope in requested:
        for value in (CORE_SCOPES if scope == "core" else (scope,)):
            if value not in VALID_SCOPES:
                raise ValueError(
                    f"unknown calibration scope {value!r}; choose from ('core', {', '.join(VALID_SCOPES)})"
                )
            if value not in expanded:
                expanded.append(value)
    return tuple(expanded)


def build_calibration_inventory(
    config_path: str | Path,
    *,
    data_dir: str | Path | None = None,
    profile_dir: str | Path | None = None,
    scopes: Iterable[str] | None = None,
    generator_models: Sequence[str] | None = None,
) -> CalibrationInventory:
    """Build a pure, read-only readiness inventory for one hardware config."""
    config = Path(config_path).expanduser().resolve()
    data_root = calibration_data_root(data_dir)
    profile_root = calibration_profile_root(profile_dir)
    selected = _normalize_scopes(scopes)
    document = _load_config_document(config)
    # Status is a production entrypoint too. Validate the same minimal rag_cm
    # policy surface before inspecting any profile.
    _validate_calibration_mode_config(document)
    hardware_identity_error: str | None = None
    try:
        system_hardware, cpu_key, gpu_key, gpu_count = _load_hardware_identity(
            config,
            document,
            strict_system_hardware="generator-tp" in selected,
            strict_cpu=any(
                scope in selected
                for scope in ("ivf", "hnsw", "retrieval-worker")
            ),
            strict_gpu="generator" in selected,
        )
    except ValueError as exc:
        if selected != ("pipeline",):
            raise
        hardware_identity_error = str(exc)
        system_hardware, cpu_key, gpu_key, gpu_count = (
            _load_hardware_identity_lenient(document)
        )

    if any(
        scope in selected for scope in ("ivf", "hnsw", "retrieval-worker")
    ) and not cpu_key:
        raise ValueError("selected CPU calibration scope requires config.system.cpu")
    if "generator" in selected and not gpu_key:
        raise ValueError("selected GPU scope requires config.system.gpu")
    if "generator" in selected and not gpu_count:
        raise ValueError(
            "selected generator scope requires a non-empty "
            "config.system.available_gpus (or system_design_space/legacy "
            "measured_search_space equivalent)"
        )
    if "generator-tp" in selected and not system_hardware:
        raise ValueError(
            "selected stage calibration scope requires "
            "config.system.system_hardware"
        )

    llm_document: dict[str, Any] | None = None
    llm_error: str | None = None
    if gpu_key and any(scope in selected for scope in ("generator", "pipeline")):
        llm_document, llm_error = _read_llm_sim_profiles(
            profile_root / "llm_sim" / f"{gpu_key}.json"
        )

    stage_document: dict[str, Any] | None = None
    stage_error: str | None = None
    if system_hardware and any(
        scope in selected for scope in ("generator-tp", "pipeline")
    ):
        stage_document, stage_error = _read_json(
            profile_root / "stage_batch" / f"{system_hardware}.json"
        )

    requested_generator_models = (
        tuple(generator_models or CANONICAL_GENERATOR_MODELS)
        if "generator" in selected
        else ()
    )
    if "generator" in selected and not requested_generator_models:
        raise ValueError("generator scope requires at least one model")

    items: list[CalibrationItem] = []
    custom_hnsw_parameters_required = _custom_hnsw_parameters_required(
        document, cpu_key
    )
    for scope in selected:
        if scope in ("ivf", "hnsw"):
            items.append(
                _faiss_item(
                    scope,
                    cpu_key or "",
                    config,
                    data_root,
                    profile_root,
                    custom_hnsw_parameters_required=(
                        custom_hnsw_parameters_required and scope == "hnsw"
                    ),
                )
            )
        elif scope == "retrieval-worker":
            items.append(
                _retrieval_worker_item(
                    cpu_key or "", config, data_root, profile_root
                )
            )
        elif scope == "generator":
            items.append(
                _generator_item(
                    gpu_key or "", requested_generator_models, config, data_root, profile_root,
                    llm_document, llm_error,
                )
            )
        elif scope == "generator-tp":
            items.append(
                _generator_tp_item(
                    system_hardware or "", config, data_root, profile_root,
                    stage_document, stage_error,
                    _stage_measurement_blocker(gpu_key, gpu_count),
                )
            )
        elif scope == "pipeline":
            items.extend(_pipeline_items(
                document,
                system_hardware=system_hardware,
                cpu_key=cpu_key,
                gpu_key=gpu_key,
                gpu_count=gpu_count,
                config_path=config,
                data_root=data_root,
                profile_root=profile_root,
                llm_document=llm_document,
                llm_error=llm_error,
                stage_document=stage_document,
                stage_error=stage_error,
                custom_hnsw_parameters_required=(
                    custom_hnsw_parameters_required
                ),
            ))
            if hardware_identity_error is not None:
                items.append(_pipeline_config_item(
                    config,
                    "pipeline calibration identity is incomplete: "
                    + hardware_identity_error,
                    name="pipeline-hardware-identity",
                ))
    return CalibrationInventory(
        config_path=str(config),
        system_hardware=system_hardware,
        cpu_key=cpu_key,
        gpu_key=gpu_key,
        gpu_count=gpu_count,
        data_root=str(data_root),
        profile_root=str(profile_root),
        scopes=selected,
        items=tuple(items),
    )
