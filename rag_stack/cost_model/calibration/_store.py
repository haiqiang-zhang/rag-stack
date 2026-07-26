"""Path layout and integrity-checked readers for the calibration store."""

from __future__ import annotations

import json
from pathlib import Path

from ._io import (
    canonical_json_bytes,
    read_json_mapping,
    require_sha256,
    sha256_bytes,
)
from .errors import CalibrationIntegrityError
from .manifest import ArtifactDescriptor, ArtifactRef, GenerationManifest


def _addressed_path(root: Path, family: str, digest: str, suffix: str = "") -> Path:
    digest = require_sha256(digest, field=f"{family} digest")
    return root / family / "sha256" / digest[:2] / f"{digest}{suffix}"


def blob_path(root: Path, digest: str) -> Path:
    return _addressed_path(root, "blobs", digest)


def artifact_path(root: Path, digest: str) -> Path:
    return _addressed_path(root, "artifacts", digest, ".json")


def generation_path(root: Path, digest: str) -> Path:
    return _addressed_path(root, "generations", digest, ".json")


def active_pointer_path(root: Path) -> Path:
    return root / "active.json"


def load_artifact_descriptor(root: Path, reference: ArtifactRef) -> ArtifactDescriptor:
    path = artifact_path(root, reference.artifact_sha256)
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise CalibrationIntegrityError(
            f"missing artifact descriptor {reference.artifact_sha256}"
        ) from exc
    if sha256_bytes(raw) != reference.artifact_sha256:
        raise CalibrationIntegrityError("artifact descriptor digest mismatch")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationIntegrityError("invalid artifact descriptor JSON") from exc
    if not isinstance(document, dict):
        raise CalibrationIntegrityError("artifact descriptor must be a JSON object")
    if canonical_json_bytes(document) != raw:
        raise CalibrationIntegrityError("artifact descriptor is not canonical JSON")
    descriptor = ArtifactDescriptor.from_document(document)
    if descriptor.artifact_sha256 != reference.artifact_sha256:
        raise CalibrationIntegrityError("artifact descriptor digest mismatch")
    if descriptor.to_ref() != reference:
        raise CalibrationIntegrityError("artifact descriptor does not match manifest")
    return descriptor


def read_artifact_payload(root: Path, reference: ArtifactRef) -> bytes:
    load_artifact_descriptor(root, reference)
    path = blob_path(root, reference.payload_sha256)
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise CalibrationIntegrityError(
            f"missing calibration payload {reference.payload_sha256}"
        ) from exc
    if len(payload) != reference.size_bytes:
        raise CalibrationIntegrityError("calibration payload size mismatch")
    if sha256_bytes(payload) != reference.payload_sha256:
        raise CalibrationIntegrityError("calibration payload digest mismatch")
    return payload


def load_generation(root: Path, digest: str) -> GenerationManifest:
    digest = require_sha256(digest, field="generation_sha256")
    path = generation_path(root, digest)
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise CalibrationIntegrityError(
            f"missing generation manifest {digest}"
        ) from exc
    if sha256_bytes(raw) != digest:
        raise CalibrationIntegrityError("generation manifest digest mismatch")
    document = read_json_mapping(path, description="generation manifest")
    # Reject non-canonical rewrites even if a caller somehow chose a matching
    # filename; every immutable JSON object has one byte representation.
    if canonical_json_bytes(document) != raw:
        raise CalibrationIntegrityError("generation manifest is not canonical JSON")
    return GenerationManifest.from_document(document, generation_sha256=digest)
