"""Structural contracts for QualityStore manifests and commit markers."""

from __future__ import annotations

from typing import Any


QUALITY_STORE_KIND = "quality_store"
QUALITY_STORE_KEY_KIND = "enriched_algo_config_without_system_and_transport"

MANIFEST_REQUIRED_FIELDS = frozenset({
    "combined_quality", "config_hashes", "config_path", "created_at",
    "dataset", "failed_hashes", "key_kind", "kind", "n_configs",
    "n_configs_requested", "n_grid", "pid", "status", "strategy",
    "token_stats", "updated_at",
})
MANIFEST_OPTIONAL_FIELDS = frozenset({
    "data_storage", "last_duplicate_hashes", "last_imported_hashes",
    "last_imported_project_dir", "last_replaced_hashes",
    "last_trace_rejections", "source_project_dir", "source_project_dirs",
})

_RECORD_REQUIRED_FIELDS = frozenset({
    "algo_hash", "artifact_sha256", "full", "metric_names", "quality_spec",
    "source", "source_mtime_ns", "updated_at",
})
_RECORD_OPTIONAL_FIELDS = frozenset({
    "legacy_sparse_memo_hash", "source_eval_error", "source_eval_type",
})
_RECORD_ARTIFACT_FIELDS = frozenset({
    "config", "pipeline_config", "quality", "trace",
})


def validate_quality_store_manifest(manifest: Any, *, source: str) -> None:
    """Fail closed unless a manifest has the exact current store shape."""

    if not isinstance(manifest, dict):
        raise TypeError(f"QualityStore manifest at {source} must be an object")
    unknown = set(manifest) - MANIFEST_REQUIRED_FIELDS - MANIFEST_OPTIONAL_FIELDS
    if unknown:
        raise ValueError(
            f"QualityStore manifest at {source} has unsupported fields: "
            f"{sorted(unknown)}"
        )
    missing = MANIFEST_REQUIRED_FIELDS - set(manifest)
    if missing:
        raise ValueError(
            f"QualityStore manifest at {source} is missing fields: {sorted(missing)}"
        )
    if manifest["kind"] != QUALITY_STORE_KIND:
        raise ValueError(f"QualityStore manifest at {source} has invalid kind")
    if manifest["key_kind"] != QUALITY_STORE_KEY_KIND:
        raise ValueError(f"QualityStore manifest at {source} has invalid key_kind")
    dataset = manifest.get("dataset") or {}
    if not isinstance(dataset.get("qa_rows"), int) or dataset["qa_rows"] <= 0:
        raise ValueError(
            f"QualityStore manifest at {source} has no valid QA row count"
        )
    hashes = manifest.get("config_hashes")
    if (
        not isinstance(hashes, list)
        or any(not isinstance(value, str) or not value for value in hashes)
        or hashes != sorted(set(hashes))
    ):
        raise ValueError(
            f"QualityStore manifest at {source} has invalid config_hashes"
        )
    if manifest.get("n_configs") != len(hashes):
        raise ValueError(
            f"QualityStore manifest at {source} has inconsistent n_configs"
        )


def validate_quality_store_record_marker(
    marker: Any,
    *,
    source: str = "QualityStore record marker",
) -> None:
    """Reject any marker that is not the exact current record shape."""

    if not isinstance(marker, dict):
        raise TypeError(f"{source} must be a JSON object")
    unknown = set(marker) - _RECORD_REQUIRED_FIELDS - _RECORD_OPTIONAL_FIELDS
    if unknown:
        raise ValueError(f"{source} has unsupported fields: {sorted(unknown)}")
    missing = _RECORD_REQUIRED_FIELDS - set(marker)
    if missing:
        raise ValueError(f"{source} is missing fields: {sorted(missing)}")

    algo_hash = marker["algo_hash"]
    if not isinstance(algo_hash, str) or not algo_hash:
        raise ValueError(f"{source} has invalid algo_hash")
    if type(marker["full"]) is not bool:
        raise ValueError(f"{source} full must be a boolean")
    metric_names = marker["metric_names"]
    if (
        not isinstance(metric_names, list)
        or any(not isinstance(name, str) or not name for name in metric_names)
        or metric_names != sorted(set(metric_names))
    ):
        raise ValueError(f"{source} has invalid metric_names")
    if not isinstance(marker["quality_spec"], dict):
        raise ValueError(f"{source} quality_spec must be an object")
    if not isinstance(marker["source"], dict):
        raise ValueError(f"{source} source must be an object")
    source_mtime_ns = marker["source_mtime_ns"]
    if (
        isinstance(source_mtime_ns, bool)
        or not isinstance(source_mtime_ns, int)
        or source_mtime_ns < 0
    ):
        raise ValueError(f"{source} source_mtime_ns must be a non-negative integer")
    if not isinstance(marker["updated_at"], str) or not marker["updated_at"]:
        raise ValueError(f"{source} updated_at must be a non-empty string")

    artifacts = marker["artifact_sha256"]
    if not isinstance(artifacts, dict) or set(artifacts) != _RECORD_ARTIFACT_FIELDS:
        raise ValueError(f"{source} has incomplete artifact_sha256")
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
        for digest in artifacts.values()
    ):
        raise ValueError(f"{source} has invalid artifact SHA256")


__all__ = [
    "MANIFEST_OPTIONAL_FIELDS",
    "QUALITY_STORE_KEY_KIND",
    "QUALITY_STORE_KIND",
    "validate_quality_store_manifest",
    "validate_quality_store_record_marker",
]
