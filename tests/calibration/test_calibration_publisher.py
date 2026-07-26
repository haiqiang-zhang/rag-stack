"""Pure-CPU contracts for immutable calibration publication and snapshots."""

from __future__ import annotations

import json
import multiprocessing
from types import MappingProxyType

import pytest

from rag_stack.cost_model.calibration import (
    ArtifactDescriptor,
    ArtifactRef,
    CalibrationIntegrityError,
    CalibrationPublisher,
    CalibrationValidationError,
    GenerationManifest,
    IncompleteGenerationError,
    NoActiveCalibrationError,
)


def _wait_for_publication_lock(root: str, connection) -> None:
    publisher = CalibrationPublisher(root)
    with publisher.publication_lock("components/faiss_ivf/workload/msmarco"):
        connection.send("acquired")
    connection.close()


def test_publish_complete_generation_and_load_snapshot(tmp_path):
    publisher = CalibrationPublisher(tmp_path)
    key = "components/faiss_ivf/workload/msmarco"
    reference = publisher.publish_json(
        key,
        {"imbalance_factor": 1.25},
    )
    same_reference = publisher.publish_json(
        key,
        {"imbalance_factor": 1.25},
    )
    assert same_reference == reference
    assert publisher.payload_path(reference).is_file()
    assert publisher.descriptor_path(reference).is_file()
    descriptor = json.loads(publisher.descriptor_path(reference).read_text())
    assert set(descriptor) == {
        "logical_key",
        "media_type",
        "payload_sha256",
        "size_bytes",
    }

    with pytest.raises(NoActiveCalibrationError):
        publisher.snapshot()

    manifest = publisher.publish_generation(
        {key: reference},
        expected_artifact_keys=[key],
        activate=True,
    )
    snapshot = publisher.snapshot()
    assert snapshot.generation_sha256 == manifest.generation_sha256
    assert snapshot.read_json(key) == {"imbalance_factor": 1.25}
    assert isinstance(snapshot.artifacts, MappingProxyType)
    assert not hasattr(snapshot, "artifact_provenance")
    generation = json.loads(publisher.manifest_path(manifest).read_text())
    assert set(generation) == {"artifacts", "expected_artifact_keys"}

    pointer = json.loads(publisher.active_pointer_path.read_text())
    assert pointer == {"generation_sha256": manifest.generation_sha256}


def test_artifact_and_generation_identity_use_only_runtime_content(tmp_path):
    publisher = CalibrationPublisher(tmp_path)
    first_key = "components/faiss_ivf/workload/msmarco"
    second_key = "components/faiss_ivf/workload/nq"

    first = publisher.publish_json(first_key, {"b": 2, "a": 1})
    same = publisher.publish_json(first_key, {"a": 1, "b": 2})
    different_key = publisher.publish_json(second_key, {"a": 1, "b": 2})
    different_media_type = publisher.publish_bytes(
        first_key,
        publisher.payload_path(first).read_bytes(),
        media_type="application/octet-stream",
    )

    assert same == first
    assert different_key.payload_sha256 == first.payload_sha256
    assert different_key.artifact_sha256 != first.artifact_sha256
    assert different_media_type.payload_sha256 == first.payload_sha256
    assert different_media_type.artifact_sha256 != first.artifact_sha256

    artifacts = {first_key: first, second_key: different_key}
    forward = publisher.publish_generation(
        artifacts,
        expected_artifact_keys=[first_key, second_key],
    )
    reverse = publisher.publish_generation(
        dict(reversed(list(artifacts.items()))),
        expected_artifact_keys=[second_key, first_key],
    )
    assert reverse.generation_sha256 == forward.generation_sha256


def test_store_envelopes_reject_unknown_fields(tmp_path):
    publisher = CalibrationPublisher(tmp_path)
    key = "components/faiss_ivf/workload/msmarco"
    reference = publisher.publish_json(
        key,
        {"imbalance_factor": 1.25},
    )
    manifest = publisher.publish_generation(
        {key: reference},
        expected_artifact_keys=[key],
        activate=True,
    )

    descriptor_document = json.loads(
        publisher.descriptor_path(reference).read_text()
    )
    descriptor_document["provenance"] = {"source": "/retired/local/path"}
    with pytest.raises(CalibrationIntegrityError, match="artifact descriptor"):
        ArtifactDescriptor.from_document(descriptor_document)

    reference_document = reference.to_dict()
    reference_document["provenance_sha256"] = "a" * 64
    with pytest.raises(CalibrationIntegrityError, match="artifact reference"):
        ArtifactRef.from_dict(reference_document)

    generation_document = manifest.to_document()
    generation_document["provenance"] = {"campaign": "retired"}
    with pytest.raises(CalibrationIntegrityError, match="manifest fields"):
        GenerationManifest.from_document(
            generation_document,
            generation_sha256=manifest.generation_sha256,
        )

    pointer = json.loads(publisher.active_pointer_path.read_text())
    pointer["retired_marker"] = 2
    publisher.active_pointer_path.write_text(json.dumps(pointer))
    with pytest.raises(CalibrationIntegrityError, match="pointer"):
        publisher.snapshot()


def test_generation_requires_exact_declared_artifact_set(tmp_path):
    publisher = CalibrationPublisher(tmp_path)
    key = "components/faiss_hnsw/workload/msmarco"
    reference = publisher.publish_json(
        key,
        {"ef_search": 64},
    )

    with pytest.raises(IncompleteGenerationError, match="missing"):
        publisher.publish_generation(
            {key: reference},
            expected_artifact_keys=[key, "components/llm_sim/genz/generator"],
            activate=True,
        )
    assert not publisher.active_pointer_path.exists()


def test_snapshot_remains_pinned_when_new_generation_is_activated(tmp_path):
    publisher = CalibrationPublisher(tmp_path)
    key = "serving_curves/generator_decode"
    first = publisher.publish_json(
        key,
        {"overhead_s": 0.001},
    )
    first_generation = publisher.publish_generation(
        {key: first},
        expected_artifact_keys=[key],
        activate=True,
    )
    frozen = publisher.snapshot()

    second = publisher.publish_json(
        key,
        {"overhead_s": 0.002},
    )
    second_generation = publisher.publish_generation(
        {key: second},
        expected_artifact_keys=[key],
        activate=True,
    )

    assert first_generation.generation_sha256 != second_generation.generation_sha256
    assert frozen.generation_sha256 == first_generation.generation_sha256
    assert frozen.read_json(key) == {"overhead_s": 0.001}
    assert publisher.snapshot().read_json(key) == {"overhead_s": 0.002}


def test_schema_hooks_run_before_any_artifact_write(tmp_path):
    seen: list[str] = []

    def validate_artifact(logical_key, document):
        seen.append(logical_key)
        if document.get("active_batch") != 8:
            raise ValueError("active batch telemetry is required")

    publisher = CalibrationPublisher(
        tmp_path,
        artifact_validator=validate_artifact,
    )
    with pytest.raises(CalibrationValidationError, match="active batch"):
        publisher.publish_json(
            "serving_curves/generator_prefill",
            {"client_concurrency": 8},
        )
    assert seen == ["serving_curves/generator_prefill"]
    assert not (tmp_path / "blobs").exists()
    assert not (tmp_path / "artifacts").exists()


def test_snapshot_detects_payload_tampering(tmp_path):
    publisher = CalibrationPublisher(tmp_path)
    key = "components/llm_sim/genz/generator"
    reference = publisher.publish_json(
        key,
        {"ceff": 0.8},
    )
    publisher.publish_generation(
        {key: reference},
        expected_artifact_keys=[key],
        activate=True,
    )
    publisher.payload_path(reference).write_bytes(b"tampered")
    with pytest.raises(CalibrationIntegrityError, match="payload"):
        publisher.snapshot()


def test_logical_keys_fail_closed_and_legacy_provenance_api_is_removed(tmp_path):
    publisher = CalibrationPublisher(tmp_path)
    with pytest.raises(CalibrationValidationError, match="unsafe path"):
        publisher.publish_json(
            "../escape",
            {},
        )
    with pytest.raises(TypeError, match="provenance"):
        publisher.publish_json(
            "components/faiss_ivf/system/h100x4",
            {},
            provenance={"source": "retired"},  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError, match="provenance_validator"):
        CalibrationPublisher(  # type: ignore[call-arg]
            tmp_path,
            provenance_validator=lambda _value: None,
        )


def test_publication_lock_serializes_separate_processes(tmp_path):
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    publisher = CalibrationPublisher(tmp_path)
    with publisher.publication_lock("components/faiss_ivf/workload/msmarco"):
        process = context.Process(
            target=_wait_for_publication_lock,
            args=(str(tmp_path), child_connection),
        )
        process.start()
        child_connection.close()
        assert not parent_connection.poll(0.25)
    # ``spawn`` re-imports the package in a fresh interpreter. On the shared
    # NFS benchmark host that cold import can exceed five seconds even though
    # the child acquires the lock immediately after the parent releases it.
    assert parent_connection.poll(20.0)
    assert parent_connection.recv() == "acquired"
    process.join(timeout=20.0)
    assert process.exitcode == 0
