"""Read-only, generation-pinned runtime view of calibration artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._io import read_json_mapping, require_sha256
from ._store import (
    active_pointer_path,
    load_generation,
    read_artifact_payload,
)
from .errors import (
    CalibrationIntegrityError,
    NoActiveCalibrationError,
)
from .manifest import ArtifactRef, GenerationManifest


@dataclass(frozen=True, slots=True)
class CalibrationSnapshot:
    """A complete manifest captured once and unaffected by later publication."""

    root: Path
    generation: GenerationManifest

    @classmethod
    def from_active(cls, root: str | Path) -> "CalibrationSnapshot":
        root_path = Path(root).expanduser().resolve()
        pointer_path = active_pointer_path(root_path)
        try:
            pointer = read_json_mapping(
                pointer_path, description="active calibration pointer"
            )
        except FileNotFoundError as exc:
            raise NoActiveCalibrationError(
                f"no active calibration generation at {pointer_path}"
            ) from exc
        if set(pointer) != {"generation_sha256"}:
            raise CalibrationIntegrityError("invalid active calibration pointer")
        digest = require_sha256(
            pointer.get("generation_sha256"), field="generation_sha256"
        )
        generation = load_generation(root_path, digest)
        # Eager verification means a successfully constructed snapshot is a
        # complete runtime view, never a pointer to a partly published set.
        for reference in generation.artifacts.values():
            read_artifact_payload(root_path, reference)
        return cls(root=root_path, generation=generation)

    @property
    def generation_sha256(self) -> str:
        return self.generation.generation_sha256

    @property
    def artifacts(self) -> Mapping[str, ArtifactRef]:
        return MappingProxyType(dict(self.generation.artifacts))

    def __contains__(self, logical_key: object) -> bool:
        return logical_key in self.generation.artifacts

    def __iter__(self) -> Iterator[str]:
        return iter(self.generation.expected_artifact_keys)

    def reference(self, logical_key: str) -> ArtifactRef:
        try:
            return self.generation.artifacts[logical_key]
        except KeyError as exc:
            raise KeyError(
                f"calibration artifact {logical_key!r} is not in generation "
                f"{self.generation_sha256}"
            ) from exc

    def read_bytes(self, logical_key: str) -> bytes:
        return read_artifact_payload(self.root, self.reference(logical_key))

    def read_json(self, logical_key: str) -> Any:
        reference = self.reference(logical_key)
        if reference.media_type != "application/json" and not (
            reference.media_type.endswith("+json")
        ):
            raise TypeError(
                f"calibration artifact {logical_key!r} is not JSON "
                f"({reference.media_type})"
            )
        try:
            return json.loads(self.read_bytes(logical_key).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CalibrationIntegrityError(
                f"calibration artifact {logical_key!r} contains invalid JSON"
            ) from exc
