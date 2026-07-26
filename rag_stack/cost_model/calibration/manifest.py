"""Immutable references and complete-generation calibration manifests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from ._io import (
    canonical_json_bytes,
    require_logical_key,
    require_sha256,
    sha256_bytes,
)
from .errors import CalibrationIntegrityError, CalibrationValidationError


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A manifest reference to one immutable artifact descriptor and payload."""

    logical_key: str
    artifact_sha256: str
    payload_sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "logical_key", require_logical_key(self.logical_key))
        for field in ("artifact_sha256", "payload_sha256"):
            object.__setattr__(
                self,
                field,
                require_sha256(getattr(self, field), field=field),
            )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise CalibrationValidationError("size_bytes must be non-negative")
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise CalibrationValidationError("media_type must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_sha256": self.artifact_sha256,
            "logical_key": self.logical_key,
            "media_type": self.media_type,
            "payload_sha256": self.payload_sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ArtifactRef":
        expected = {
            "artifact_sha256",
            "logical_key",
            "media_type",
            "payload_sha256",
            "size_bytes",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise CalibrationIntegrityError("invalid artifact reference fields")
        try:
            return cls(**dict(value))
        except (TypeError, ValueError) as exc:
            raise CalibrationIntegrityError("invalid artifact reference") from exc


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    """Runtime-relevant metadata durably bound to one payload digest."""

    logical_key: str
    payload_sha256: str
    size_bytes: int
    media_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "logical_key", require_logical_key(self.logical_key))
        object.__setattr__(
            self,
            "payload_sha256",
            require_sha256(self.payload_sha256, field="payload_sha256"),
        )
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes < 0
        ):
            raise CalibrationValidationError("size_bytes must be non-negative")
        if not isinstance(self.media_type, str) or not self.media_type.strip():
            raise CalibrationValidationError("media_type must be non-empty")

    def to_document(self) -> dict[str, Any]:
        return {
            "logical_key": self.logical_key,
            "media_type": self.media_type,
            "payload_sha256": self.payload_sha256,
            "size_bytes": self.size_bytes,
        }

    @property
    def artifact_sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.to_document()))

    def to_ref(self) -> ArtifactRef:
        return ArtifactRef(
            logical_key=self.logical_key,
            artifact_sha256=self.artifact_sha256,
            payload_sha256=self.payload_sha256,
            size_bytes=self.size_bytes,
            media_type=self.media_type,
        )

    @classmethod
    def from_document(cls, value: Mapping[str, Any]) -> "ArtifactDescriptor":
        expected = {
            "logical_key",
            "media_type",
            "payload_sha256",
            "size_bytes",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise CalibrationIntegrityError("invalid artifact descriptor fields")
        try:
            return cls(
                logical_key=value["logical_key"],
                payload_sha256=value["payload_sha256"],
                size_bytes=value["size_bytes"],
                media_type=value["media_type"],
            )
        except (TypeError, ValueError) as exc:
            raise CalibrationIntegrityError("invalid artifact descriptor") from exc


@dataclass(frozen=True, slots=True)
class GenerationManifest:
    """One exact, immutable set of calibration artifacts used by a run."""

    generation_sha256: str
    artifacts: Mapping[str, ArtifactRef]
    expected_artifact_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "generation_sha256",
            require_sha256(self.generation_sha256, field="generation_sha256"),
        )
        entries = dict(self.artifacts)
        if not entries:
            raise CalibrationValidationError(
                "a calibration generation must contain at least one artifact"
            )
        for key, reference in entries.items():
            require_logical_key(key, field="artifact key")
            if not isinstance(reference, ArtifactRef) or reference.logical_key != key:
                raise CalibrationValidationError(
                    f"generation entry {key!r} does not match its ArtifactRef"
                )
        expected = tuple(sorted(set(self.expected_artifact_keys)))
        if tuple(self.expected_artifact_keys) != expected:
            raise CalibrationValidationError(
                "expected_artifact_keys must be unique and sorted"
            )
        if set(entries) != set(expected):
            raise CalibrationValidationError(
                "generation artifacts do not exactly match expected_artifact_keys"
            )
        object.__setattr__(self, "artifacts", MappingProxyType(entries))
        object.__setattr__(self, "expected_artifact_keys", expected)

    def to_document(self) -> dict[str, Any]:
        return {
            "artifacts": {
                key: reference.to_dict()
                for key, reference in sorted(self.artifacts.items())
            },
            "expected_artifact_keys": list(self.expected_artifact_keys),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_document(),
            "generation_sha256": self.generation_sha256,
        }

    @classmethod
    def build(
        cls,
        artifacts: Mapping[str, ArtifactRef],
        *,
        expected_artifact_keys: Iterable[str],
    ) -> "GenerationManifest":
        expected = tuple(sorted(expected_artifact_keys))
        document = {
            "artifacts": {
                key: reference.to_dict()
                for key, reference in sorted(dict(artifacts).items())
            },
            "expected_artifact_keys": list(expected),
        }
        digest = sha256_bytes(canonical_json_bytes(document))
        return cls(
            generation_sha256=digest,
            artifacts=dict(artifacts),
            expected_artifact_keys=expected,
        )

    @classmethod
    def from_document(
        cls,
        value: Mapping[str, Any],
        *,
        generation_sha256: str,
    ) -> "GenerationManifest":
        expected_fields = {
            "artifacts",
            "expected_artifact_keys",
        }
        if not isinstance(value, Mapping) or set(value) != expected_fields:
            raise CalibrationIntegrityError("invalid generation manifest fields")
        artifacts_value = value.get("artifacts")
        expected_keys = value.get("expected_artifact_keys")
        if not isinstance(artifacts_value, Mapping) or not isinstance(
            expected_keys, list
        ):
            raise CalibrationIntegrityError("invalid generation artifact set")
        try:
            manifest = cls(
                generation_sha256=generation_sha256,
                artifacts={
                    str(key): ArtifactRef.from_dict(reference)
                    for key, reference in artifacts_value.items()
                },
                expected_artifact_keys=tuple(expected_keys),
            )
        except (TypeError, ValueError) as exc:
            raise CalibrationIntegrityError("invalid generation manifest") from exc
        actual = sha256_bytes(canonical_json_bytes(manifest.to_document()))
        if actual != generation_sha256:
            raise CalibrationIntegrityError("generation manifest digest mismatch")
        return manifest
