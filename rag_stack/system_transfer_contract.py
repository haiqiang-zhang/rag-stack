"""Versionless structural contracts for system-transfer artifacts."""

from __future__ import annotations

from typing import Any


_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "system_transfer_launch": (
        frozenset({
            "generator_backend", "kind", "n_refine_iterations",
            "n_refine_iterations_source", "source_project", "system_yaml",
            "vllm_chat_projection",
        }),
        frozenset(),
    ),
    "system_transfer_state": (
        frozenset({
            "cost_model_version", "kind", "quality_store_path",
            "refinement_iterations_requested", "source_records", "status",
            "target_config_sha256", "target_system_sha256",
        }),
        frozenset({
            "refinement_budget_extensions", "refinement_iterations_completed",
            "replay_plan", "result_sha256",
        }),
    ),
    "selective_system_transfer_replay_plan": (
        frozenset({
            "kind", "llmlingua2_active_source_indices", "llmlingua2_profile",
            "plan_sha256", "prior_system_transfer_identity_sha256",
            "prior_target_project", "retransfer_source_indices",
            "reuse_lineage", "reuse_source_indices",
        }),
        frozenset(),
    ),
    "system_transfer_replay": (
        frozenset({
            "artifact_sha256", "cost_model_version", "kind",
            "performance_origin", "quality_reused", "source",
            "source_algo_hash", "source_eval_error", "source_eval_type",
            "source_index", "source_record_marker_sha256",
            "target_config_sha256", "target_eval_type",
        }),
        frozenset({"llmlingua2_profile", "reuse_lineage"}),
    ),
    "system_transfer_refinement": (
        frozenset({
            "algo_hash", "artifact_sha256", "config", "cost_model_version",
            "eval_id", "eval_type", "gt_budget_count", "kind", "objectives",
            "objective_cost", "optimizer_history_free", "refinement_index",
            "quality_store_record_marker_sha256", "target_config_sha256",
        }),
        frozenset({"llmlingua2_profile"}),
    ),
    "system_transfer_result": (
        frozenset({
            "assembly_results_count", "cm_count", "gt_count", "hardware",
            "kind", "pareto_algo_hashes", "pareto_configs", "pareto_eval_ids",
            "pareto_objectives", "status", "system_transfer", "total_cost",
        }),
        frozenset(),
    ),
}

_SHA_FIELDS = frozenset({
    "plan_sha256", "prior_system_transfer_identity_sha256", "result_sha256",
    "source_record_marker_sha256", "target_config_sha256",
    "target_system_sha256", "quality_store_record_marker_sha256",
})


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_system_transfer_artifact(
    payload: Any,
    *,
    expected_kind: str | None = None,
    source: str = "system-transfer artifact",
) -> dict:
    """Return a mapping only when it has the exact current shape."""

    if not isinstance(payload, dict):
        raise TypeError(f"{source} must be an object")
    kind = payload.get("kind")
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(f"{source} requires kind={expected_kind!r}, got {kind!r}")
    if kind not in _FIELDS:
        raise ValueError(f"{source} has unsupported kind={kind!r}")
    required, optional = _FIELDS[kind]
    unknown = set(payload) - required - optional
    if unknown:
        raise ValueError(f"{source} has unsupported fields: {sorted(unknown)}")
    missing = required - set(payload)
    if missing:
        raise ValueError(f"{source} is missing fields: {sorted(missing)}")
    for field in _SHA_FIELDS & set(payload):
        if not _is_sha256(payload[field]):
            raise ValueError(f"{source} has invalid {field}")

    if kind == "system_transfer_state":
        if payload["status"] not in {"running", "completed"}:
            raise ValueError(f"{source} has invalid status")
        requested = payload["refinement_iterations_requested"]
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
            raise ValueError(f"{source} has invalid refinement budget")
        records = payload["source_records"]
        if not isinstance(records, list) or not records:
            raise ValueError(f"{source} has no source records")
        for index, record in enumerate(records):
            if not isinstance(record, dict) or set(record) != {
                "algo_hash", "record_marker_sha256", "source_index",
            }:
                raise ValueError(f"{source} source record {index} has invalid shape")
            if record["source_index"] != index or not _is_sha256(
                record["record_marker_sha256"]
            ):
                raise ValueError(f"{source} source record {index} has invalid identity")
    elif kind == "system_transfer_launch":
        requested = payload["n_refine_iterations"]
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 0:
            raise ValueError(f"{source} has invalid refinement budget")
    elif kind == "system_transfer_replay":
        if payload["quality_reused"] is not True:
            raise ValueError(f"{source} must declare reused quality")
    elif kind == "system_transfer_result":
        if payload["status"] != "completed":
            raise ValueError(f"{source} must be completed")

    return payload


__all__ = ["validate_system_transfer_artifact"]
