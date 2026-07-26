"""Transactional publisher for isolated calibration artifacts and generations."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ._io import (
    atomic_replace_bytes,
    canonical_json_bytes,
    key_lock,
    publish_immutable,
    require_logical_key,
    sha256_bytes,
)
from ._store import (
    active_pointer_path,
    artifact_path,
    blob_path,
    generation_path,
    load_artifact_descriptor,
    load_generation,
    read_artifact_payload,
)
from .errors import CalibrationValidationError, IncompleteGenerationError
from .manifest import ArtifactDescriptor, ArtifactRef, GenerationManifest


ArtifactValidationHook = Callable[[str, Any], None]
GenerationValidationHook = Callable[[GenerationManifest], None]


class CalibrationPublisher:
    """Publish immutable artifacts, then atomically select one complete set.

    Artifact publication never changes the active runtime view. Only
    :meth:`activate_generation` replaces ``active.json``, and that pointer can
    reference only a fully validated, content-addressed generation manifest.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        artifact_validator: ArtifactValidationHook | None = None,
        generation_validator: GenerationValidationHook | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self._artifact_validator = artifact_validator
        self._generation_validator = generation_validator

    @contextmanager
    def publication_lock(self, key: str) -> Iterator[None]:
        """Expose the same cross-process lock for multi-step component builds."""
        require_logical_key(key, field="publication lock key")
        with key_lock(self.root, f"publisher:{key}"):
            yield

    def payload_path(self, reference_or_digest: ArtifactRef | str) -> Path:
        digest = (
            reference_or_digest.payload_sha256
            if isinstance(reference_or_digest, ArtifactRef)
            else reference_or_digest
        )
        return blob_path(self.root, digest)

    def descriptor_path(self, reference_or_digest: ArtifactRef | str) -> Path:
        digest = (
            reference_or_digest.artifact_sha256
            if isinstance(reference_or_digest, ArtifactRef)
            else reference_or_digest
        )
        return artifact_path(self.root, digest)

    def manifest_path(self, manifest_or_digest: GenerationManifest | str) -> Path:
        digest = (
            manifest_or_digest.generation_sha256
            if isinstance(manifest_or_digest, GenerationManifest)
            else manifest_or_digest
        )
        return generation_path(self.root, digest)

    @property
    def active_pointer_path(self) -> Path:
        return active_pointer_path(self.root)

    def publish_json(
        self,
        logical_key: str,
        document: Any,
        *,
        validator: ArtifactValidationHook | None = None,
    ) -> ArtifactRef:
        """Validate and publish one canonical JSON calibration artifact."""
        payload = canonical_json_bytes(document)
        # Detach the value seen by validators from mutable caller objects.
        validated_value = json.loads(payload.decode("utf-8"))
        return self._publish(
            logical_key,
            payload,
            validated_value=validated_value,
            media_type="application/json",
            validator=validator,
        )

    def publish_bytes(
        self,
        logical_key: str,
        payload: bytes,
        *,
        media_type: str,
        validator: ArtifactValidationHook | None = None,
    ) -> ArtifactRef:
        """Validate and publish one arbitrary binary calibration artifact."""
        if not isinstance(payload, bytes):
            raise CalibrationValidationError("payload must be bytes")
        return self._publish(
            logical_key,
            payload,
            validated_value=payload,
            media_type=media_type,
            validator=validator,
        )

    def _publish(
        self,
        logical_key: str,
        payload: bytes,
        *,
        validated_value: Any,
        media_type: str,
        validator: ArtifactValidationHook | None,
    ) -> ArtifactRef:
        logical_key = require_logical_key(logical_key)
        if not isinstance(media_type, str) or not media_type.strip():
            raise CalibrationValidationError("media_type must be non-empty")
        for hook in (self._artifact_validator, validator):
            if hook is None:
                continue
            try:
                hook(logical_key, validated_value)
            except Exception as exc:
                raise CalibrationValidationError(
                    f"artifact validation failed for {logical_key!r}: {exc}"
                ) from exc

        descriptor = ArtifactDescriptor(
            logical_key=logical_key,
            payload_sha256=sha256_bytes(payload),
            size_bytes=len(payload),
            media_type=media_type,
        )
        reference = descriptor.to_ref()
        descriptor_payload = canonical_json_bytes(descriptor.to_document())

        # The logical lock serializes two publishers for the same independently
        # calibrated key. Address locks additionally deduplicate equal content
        # shared by different keys without allowing in-place mutation.
        with key_lock(self.root, f"artifact-key:{logical_key}"):
            publish_immutable(
                blob_path(self.root, reference.payload_sha256),
                payload,
                root=self.root,
                key=f"blob:{reference.payload_sha256}",
            )
            publish_immutable(
                artifact_path(self.root, reference.artifact_sha256),
                descriptor_payload,
                root=self.root,
                key=f"artifact:{reference.artifact_sha256}",
            )
        # Read back through the same strict path used by runtime snapshots.
        load_artifact_descriptor(self.root, reference)
        read_artifact_payload(self.root, reference)
        return reference

    def publish_generation(
        self,
        artifacts: Mapping[str, ArtifactRef],
        *,
        expected_artifact_keys: Iterable[str],
        activate: bool = False,
    ) -> GenerationManifest:
        """Publish one exact artifact set and optionally select it atomically."""
        entries = dict(artifacts)
        expected = tuple(sorted(expected_artifact_keys))
        if set(entries) != set(expected):
            missing = sorted(set(expected) - set(entries))
            unexpected = sorted(set(entries) - set(expected))
            raise IncompleteGenerationError(
                "generation is incomplete: "
                f"missing={missing}, unexpected={unexpected}"
            )
        # Verify every addressed descriptor and payload before the generation
        # itself becomes publishable.
        for key, reference in entries.items():
            if not isinstance(reference, ArtifactRef) or reference.logical_key != key:
                raise CalibrationValidationError(
                    f"generation entry {key!r} does not match its ArtifactRef"
                )
            read_artifact_payload(self.root, reference)

        manifest = GenerationManifest.build(
            entries,
            expected_artifact_keys=expected,
        )
        if self._generation_validator is not None:
            try:
                self._generation_validator(manifest)
            except Exception as exc:
                raise CalibrationValidationError(
                    f"generation validation failed: {exc}"
                ) from exc
        payload = canonical_json_bytes(manifest.to_document())
        publish_immutable(
            generation_path(self.root, manifest.generation_sha256),
            payload,
            root=self.root,
            key=f"generation:{manifest.generation_sha256}",
        )
        load_generation(self.root, manifest.generation_sha256)
        if activate:
            self.activate_generation(manifest)
        return manifest

    def activate_generation(
        self, manifest_or_digest: GenerationManifest | str
    ) -> GenerationManifest:
        """Atomically replace the single active pointer with a complete manifest."""
        digest = (
            manifest_or_digest.generation_sha256
            if isinstance(manifest_or_digest, GenerationManifest)
            else manifest_or_digest
        )
        with key_lock(self.root, "active-generation"):
            manifest = load_generation(self.root, digest)
            for reference in manifest.artifacts.values():
                read_artifact_payload(self.root, reference)
            pointer = canonical_json_bytes({
                "generation_sha256": manifest.generation_sha256,
            })
            atomic_replace_bytes(active_pointer_path(self.root), pointer)
        return manifest

    def snapshot(self) -> "CalibrationSnapshot":
        """Resolve and freeze the active generation exactly once."""
        from .snapshot import CalibrationSnapshot

        return CalibrationSnapshot.from_active(self.root)
