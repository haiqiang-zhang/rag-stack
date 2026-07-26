"""Sparse hardware x vLLM-engine-stage x batch overhead curves for CM.

This module intentionally has a very small calibration surface.  A runtime
profile is identified by exactly two values -- the physical system hardware
(GPU + host identity) and one canonical vLLM engine boundary -- and its only
independent variable is the real resident batch observed inside that engine.
Model identity, TP/PP, topology, and deployment are neither profile keys nor
interpolation inputs.

Five sparse batches are mandatory direct anchor observations.  Batch 128 is
measured as a held-out validation point and is never fitted or inserted into
the curve::

    required anchors: 1, 8, 32, 64, 256
    held out: 128

Every anchor is mandatory.  Any negative or otherwise invalid observation,
including an interior point, fails the complete fit so a measurement/contract
bug cannot be hidden by interpolation.

There are no fitted coefficients.  Non-negative stage overhead between
anchors is piecewise-linear in resident batch, and requests above the measured
maximum fail closed.  A collocated scheduler cycle first sums its analytical
prefill and decode component work and then applies its owning-stage overhead
exactly once.  Direct-D has the same engine-cycle boundary.  Direct-P evidence
instead measures a completed-prefill frontier: one B-resident cohort completes
B logical prefills.  Analytical accounting charges that frontier once per
logical P call and divides shared resource demand by B; DES distributes one
frontier overhead across a step as ``completed_prefills / resident_batch``.
A disaggregated P/D deployment reads only these independently measured role
curves.  A collocated profile is never reused for either role.

A hardware generation is either the exact collocated pair (while direct P/D
evidence is not yet available), or the exact complete six-profile set.  A
partial role publication is invalid.  Neither publication nor runtime clamps
negative evidence; malformed evidence fails closed before it can become a
profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from rag_stack.calibration_paths import (
    atomic_write_json,
    calibration_profile_root,
)
from rag_stack.taxonomy import TAXONOMY


ANCHOR_BATCHES: tuple[int, ...] = (1, 8, 32, 64, 256)
HELD_OUT_BATCH = 128
MAX_BATCH_SIZE = ANCHOR_BATCHES[-1]
# This is a publication-quality gate, not a fitted coefficient.  The sparse
# campaign deliberately reserves B128 for validation, so the gate allows the
# measured curvature between the B64 and B256 anchors without adding another
# anchor or any model/deployment-specific correction.
DEFAULT_MAX_HELD_OUT_RELATIVE_ERROR = 0.35
BATCH_UNIT = "resident_sequences"
INTERPOLATION = (
    "piecewise_linear_nonnegative_engine_cycle_overhead_in_resident_batch"
)
PROFILE_BLOCK_KEY = "_stage_batch_curves"
PROFILE_DIRECTORY = "stage_batch"
RUNTIME_SELECTION_MODE = "uncalibrated_spec"
REQUIRED_COLLOCATED_ENGINE_STAGES: tuple[str, ...] = (
    "generator",
    "query_expansion",
)
DISAGG_ROLE_STAGE_KEYS: tuple[str, ...] = (
    "generator_prefill",
    "generator_decode",
    "query_expansion_prefill",
    "query_expansion_decode",
)
COMPLETE_ENGINE_STAGE_KEYS: tuple[str, ...] = (
    *REQUIRED_COLLOCATED_ENGINE_STAGES,
    *DISAGG_ROLE_STAGE_KEYS,
)
_SYSTEM_HARDWARE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_FORBIDDEN_DIMENSION_FIELDS = frozenset({
    "hardware_key",
    "model",
    "model_key",
    "tp",
    "pp",
    "tensor_parallel",
    "tensor_parallel_size",
    "pipeline_parallel",
    "pipeline_parallel_size",
    "topology",
    "deployment",
    "deployment_id",
})


class StageBatchCurveError(ValueError):
    """Base error for malformed or inadmissible stage-batch evidence."""


class StageBatchCurveDomainError(StageBatchCurveError):
    """The requested active batch is outside the calibrated domain."""


class StageBatchCurveValidationError(StageBatchCurveError):
    """Calibration evidence failed the fixed publication contract."""


def validate_system_hardware(value: Any) -> str:
    """Return one canonical path-safe host+accelerator identity.

    The identity is used both as profile content and as a single filename
    component.  Normalizing whitespace or accepting path separators would give
    one physical system multiple spellings or let a runtime lookup escape the
    authoritative ``stage_batch`` directory, so invalid values fail closed.
    """

    if not isinstance(value, str) or not value:
        raise StageBatchCurveValidationError(
            "system_hardware must be a non-empty path-safe string"
        )
    if value != value.strip() or _SYSTEM_HARDWARE_RE.fullmatch(value) is None:
        raise StageBatchCurveValidationError(
            f"system_hardware is not path-safe: {value!r}"
        )
    if len(value.encode("utf-8")) > 255:
        raise StageBatchCurveValidationError(
            "system_hardware exceeds the filesystem component limit"
        )
    return value


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise StageBatchCurveValidationError(
            "stage-batch profile values must be finite JSON data"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value).decode("utf-8"))


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise StageBatchCurveValidationError(
            f"{label} fields do not match the exact schema "
            f"(missing={missing}, extra={extra})"
        )


def _reject_forbidden_dimensions(value: Any, *, path: str = "profile") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            if key.lower() in _FORBIDDEN_DIMENSION_FIELDS:
                raise StageBatchCurveValidationError(
                    f"{path}.{key} is forbidden: stage-batch profiles are keyed "
                    "only by system_hardware and stage"
                )
            _reject_forbidden_dimensions(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_dimensions(child, path=f"{path}[{index}]")


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise StageBatchCurveValidationError(f"{label} must be a positive integer")
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StageBatchCurveValidationError(
            f"{label} must be a positive integer"
        ) from exc
    if normalized < 1 or normalized != value:
        raise StageBatchCurveValidationError(f"{label} must be a positive integer")
    return normalized


def _nonnegative_finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise StageBatchCurveValidationError(
            f"{label} must be finite and non-negative"
        )
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StageBatchCurveValidationError(
            f"{label} must be finite and non-negative"
        ) from exc
    if not math.isfinite(normalized) or normalized < 0.0:
        raise StageBatchCurveValidationError(
            f"{label} must be finite and non-negative"
        )
    return normalized


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise StageBatchCurveValidationError(f"{label} must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StageBatchCurveValidationError(f"{label} must be finite") from exc
    if not math.isfinite(normalized):
        raise StageBatchCurveValidationError(f"{label} must be finite")
    return normalized


def _positive_finite(value: Any, *, label: str) -> float:
    normalized = _finite(value, label=label)
    if normalized <= 0.0:
        raise StageBatchCurveValidationError(
            f"{label} must be finite and positive"
        )
    return normalized


def _validate_service_decomposition(
    *,
    observed_total_service_s: Any,
    reference_component_service_s: Any,
    overhead_s: Any,
    label: str,
) -> tuple[float, float, float]:
    total = _positive_finite(
        observed_total_service_s,
        label=f"{label}.observed_total_service_s",
    )
    component = _positive_finite(
        reference_component_service_s,
        label=f"{label}.reference_component_service_s",
    )
    overhead = _nonnegative_finite(
        overhead_s,
        label=f"{label}.overhead_s",
    )
    expected = component + overhead
    if not math.isclose(
        total, expected, rel_tol=1e-12, abs_tol=1e-15,
    ):
        raise StageBatchCurveValidationError(
            f"{label} must prove observed_total_service_s = "
            "reference_component_service_s + non-negative overhead_s"
        )
    return total, component, overhead


@dataclass(frozen=True)
class StageBatchCurveKey:
    """The complete identity: system hardware and serving-stage boundary."""

    system_hardware: str
    stage: str

    def __post_init__(self) -> None:
        system_hardware = validate_system_hardware(self.system_hardware)
        stage = str(self.stage).strip()
        if not stage:
            raise StageBatchCurveValidationError(
                "stage must be non-empty"
            )
        allowed = {
            *REQUIRED_COLLOCATED_ENGINE_STAGES,
            *DISAGG_ROLE_STAGE_KEYS,
        }
        if stage not in allowed:
            raise StageBatchCurveValidationError(
                f"unknown vLLM serving-stage boundary {stage!r}"
            )
        # Keep the taxonomy as the source of truth for the two bare owning
        # stages and every reserved disaggregated role key.
        if stage in REQUIRED_COLLOCATED_ENGINE_STAGES:
            if stage not in TAXONOMY.stages:
                raise StageBatchCurveValidationError(
                    f"unknown RAG stage {stage!r}"
                )
        else:
            try:
                TAXONOMY.performance_stage(stage)
            except KeyError as exc:
                raise StageBatchCurveValidationError(
                    f"unknown PerformanceStage {stage!r}"
                ) from exc
        object.__setattr__(self, "system_hardware", system_hardware)
        object.__setattr__(self, "stage", stage)

    def to_dict(self) -> dict[str, str]:
        return {
            "system_hardware": self.system_hardware,
            "stage": self.stage,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageBatchCurveKey":
        if not isinstance(value, Mapping):
            raise StageBatchCurveValidationError("stage-batch key must be an object")
        _require_exact_fields(
            value,
            {"system_hardware", "stage"},
            label="stage-batch key",
        )
        return cls(
            system_hardware=value["system_hardware"],
            stage=str(value["stage"]),
        )


@dataclass(frozen=True)
class StageBatchMeasurement:
    """One already-aggregated measurement at an exact real active batch.

    ``target_batch`` is the acquisition target.  Publication accepts the row
    only if the engine cap can admit it and telemetry reports exactly that same
    resident batch.  Client population is deliberately absent from the schema.
    """

    key: StageBatchCurveKey
    target_batch: int
    engine_max_num_seqs: int
    observed_resident_batch: int
    observed_total_service_s: float
    reference_component_service_s: float
    overhead_s: float
    sample_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, StageBatchCurveKey):
            raise StageBatchCurveValidationError(
                "measurement key must be a StageBatchCurveKey"
            )
        target = _positive_int(self.target_batch, label="target_batch")
        cap = _positive_int(
            self.engine_max_num_seqs, label="engine_max_num_seqs"
        )
        resident = _positive_int(
            self.observed_resident_batch, label="observed_resident_batch"
        )
        samples = _positive_int(self.sample_count, label="sample_count")
        total, component, overhead = _validate_service_decomposition(
            observed_total_service_s=self.observed_total_service_s,
            reference_component_service_s=self.reference_component_service_s,
            overhead_s=self.overhead_s,
            label="measurement",
        )
        if cap < target:
            raise StageBatchCurveValidationError(
                f"engine cap {cap} cannot admit target active batch {target}"
            )
        if resident != target:
            raise StageBatchCurveValidationError(
                "observed resident batch must exactly equal the calibration "
                f"target (target={target}, observed={resident})"
            )
        object.__setattr__(self, "target_batch", target)
        object.__setattr__(self, "engine_max_num_seqs", cap)
        object.__setattr__(self, "observed_resident_batch", resident)
        object.__setattr__(self, "observed_total_service_s", total)
        object.__setattr__(self, "reference_component_service_s", component)
        object.__setattr__(self, "overhead_s", overhead)
        object.__setattr__(self, "sample_count", samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "target_batch": self.target_batch,
            "engine_max_num_seqs": self.engine_max_num_seqs,
            "observed_resident_batch": self.observed_resident_batch,
            "observed_total_service_s": self.observed_total_service_s,
            "reference_component_service_s": self.reference_component_service_s,
            "overhead_s": self.overhead_s,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageBatchMeasurement":
        if not isinstance(value, Mapping):
            raise StageBatchCurveValidationError("measurement must be an object")
        _require_exact_fields(
            value,
            {
                "key",
                "target_batch",
                "engine_max_num_seqs",
                "observed_resident_batch",
                "observed_total_service_s",
                "reference_component_service_s",
                "overhead_s",
                "sample_count",
            },
            label="measurement",
        )
        _reject_forbidden_dimensions(value, path="measurement")
        return cls(
            key=StageBatchCurveKey.from_dict(value["key"]),
            target_batch=value["target_batch"],
            engine_max_num_seqs=value["engine_max_num_seqs"],
            observed_resident_batch=value["observed_resident_batch"],
            observed_total_service_s=value["observed_total_service_s"],
            reference_component_service_s=value["reference_component_service_s"],
            overhead_s=value["overhead_s"],
            sample_count=value["sample_count"],
        )


@dataclass(frozen=True)
class StageBatchAnchor:
    batch_size: int
    observed_total_service_s: float
    reference_component_service_s: float
    overhead_s: float
    engine_max_num_seqs: int
    observed_resident_batch: int
    sample_count: int

    def __post_init__(self) -> None:
        batch = _positive_int(self.batch_size, label="anchor.batch_size")
        cap = _positive_int(
            self.engine_max_num_seqs, label="anchor.engine_max_num_seqs"
        )
        resident = _positive_int(
            self.observed_resident_batch,
            label="anchor.observed_resident_batch",
        )
        samples = _positive_int(self.sample_count, label="anchor.sample_count")
        total, component, overhead = _validate_service_decomposition(
            observed_total_service_s=self.observed_total_service_s,
            reference_component_service_s=self.reference_component_service_s,
            overhead_s=self.overhead_s,
            label="anchor",
        )
        if cap < batch or resident != batch:
            raise StageBatchCurveValidationError(
                "anchor does not prove its exact active batch and engine cap"
            )
        object.__setattr__(self, "batch_size", batch)
        object.__setattr__(self, "engine_max_num_seqs", cap)
        object.__setattr__(self, "observed_resident_batch", resident)
        object.__setattr__(self, "sample_count", samples)
        object.__setattr__(self, "observed_total_service_s", total)
        object.__setattr__(self, "reference_component_service_s", component)
        object.__setattr__(self, "overhead_s", overhead)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "observed_total_service_s": self.observed_total_service_s,
            "reference_component_service_s": self.reference_component_service_s,
            "overhead_s": self.overhead_s,
            "engine_max_num_seqs": self.engine_max_num_seqs,
            "observed_resident_batch": self.observed_resident_batch,
            "sample_count": self.sample_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageBatchAnchor":
        if not isinstance(value, Mapping):
            raise StageBatchCurveValidationError("anchor must be an object")
        _require_exact_fields(
            value,
            {
                "batch_size",
                "observed_total_service_s",
                "reference_component_service_s",
                "overhead_s",
                "engine_max_num_seqs",
                "observed_resident_batch",
                "sample_count",
            },
            label="anchor",
        )
        return cls(**dict(value))


@dataclass(frozen=True)
class HeldOutValidation:
    batch_size: int
    observed_total_service_s: float
    reference_component_service_s: float
    observed_overhead_s: float
    predicted_overhead_s: float
    predicted_total_service_s: float
    absolute_error_s: float
    relative_error: float
    max_relative_error: float
    engine_max_num_seqs: int
    observed_resident_batch: int
    sample_count: int
    passed: bool

    def __post_init__(self) -> None:
        batch = _positive_int(self.batch_size, label="validation.batch_size")
        if batch != HELD_OUT_BATCH:
            raise StageBatchCurveValidationError(
                f"held-out validation batch must be {HELD_OUT_BATCH}"
            )
        observed, component, observed_overhead = (
            _validate_service_decomposition(
                observed_total_service_s=self.observed_total_service_s,
                reference_component_service_s=(
                    self.reference_component_service_s
                ),
                overhead_s=self.observed_overhead_s,
                label="validation",
            )
        )
        predicted_overhead = _nonnegative_finite(
            self.predicted_overhead_s,
            label="validation.predicted_overhead_s",
        )
        predicted_total = _positive_finite(
            self.predicted_total_service_s,
            label="validation.predicted_total_service_s",
        )
        expected_predicted_total = component + predicted_overhead
        if not math.isclose(
            predicted_total,
            expected_predicted_total,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise StageBatchCurveValidationError(
                "held-out predicted_total_service_s must equal "
                "reference_component_service_s + non-negative "
                "predicted_overhead_s"
            )
        absolute = _nonnegative_finite(
            self.absolute_error_s,
            label="validation.absolute_error_s",
        )
        relative = _nonnegative_finite(
            self.relative_error,
            label="validation.relative_error",
        )
        maximum = _nonnegative_finite(
            self.max_relative_error,
            label="validation.max_relative_error",
        )
        cap = _positive_int(
            self.engine_max_num_seqs,
            label="validation.engine_max_num_seqs",
        )
        resident = _positive_int(
            self.observed_resident_batch,
            label="validation.observed_resident_batch",
        )
        samples = _positive_int(
            self.sample_count, label="validation.sample_count"
        )
        if cap < batch or resident != batch:
            raise StageBatchCurveValidationError(
                "held-out row does not prove its exact active batch and engine cap"
            )
        expected_absolute = abs(predicted_total - observed)
        expected_relative = _relative_error(predicted_total, observed)
        if not math.isclose(absolute, expected_absolute, rel_tol=1e-12, abs_tol=1e-15):
            raise StageBatchCurveValidationError(
                "held-out absolute error does not match its observations"
            )
        if not math.isclose(relative, expected_relative, rel_tol=1e-12, abs_tol=1e-15):
            raise StageBatchCurveValidationError(
                "held-out relative error does not match its observations"
            )
        expected_passed = relative <= maximum
        if not isinstance(self.passed, bool) or self.passed != expected_passed:
            raise StageBatchCurveValidationError(
                "held-out pass flag does not match the fixed error gate"
            )
        if not self.passed:
            raise StageBatchCurveValidationError(
                f"held-out batch {batch} relative error {relative:.6f} exceeds "
                f"the publication limit {maximum:.6f}"
            )
        object.__setattr__(self, "batch_size", batch)
        object.__setattr__(self, "observed_total_service_s", observed)
        object.__setattr__(self, "reference_component_service_s", component)
        object.__setattr__(self, "observed_overhead_s", observed_overhead)
        object.__setattr__(self, "predicted_overhead_s", predicted_overhead)
        object.__setattr__(self, "predicted_total_service_s", predicted_total)
        object.__setattr__(self, "absolute_error_s", absolute)
        object.__setattr__(self, "relative_error", relative)
        object.__setattr__(self, "max_relative_error", maximum)
        object.__setattr__(self, "engine_max_num_seqs", cap)
        object.__setattr__(self, "observed_resident_batch", resident)
        object.__setattr__(self, "sample_count", samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "observed_total_service_s": self.observed_total_service_s,
            "reference_component_service_s": self.reference_component_service_s,
            "observed_overhead_s": self.observed_overhead_s,
            "predicted_overhead_s": self.predicted_overhead_s,
            "predicted_total_service_s": self.predicted_total_service_s,
            "absolute_error_s": self.absolute_error_s,
            "relative_error": self.relative_error,
            "max_relative_error": self.max_relative_error,
            "engine_max_num_seqs": self.engine_max_num_seqs,
            "observed_resident_batch": self.observed_resident_batch,
            "sample_count": self.sample_count,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HeldOutValidation":
        if not isinstance(value, Mapping):
            raise StageBatchCurveValidationError(
                "held-out validation must be an object"
            )
        _require_exact_fields(
            value,
            {
                "batch_size",
                "observed_total_service_s",
                "reference_component_service_s",
                "observed_overhead_s",
                "predicted_overhead_s",
                "predicted_total_service_s",
                "absolute_error_s",
                "relative_error",
                "max_relative_error",
                "engine_max_num_seqs",
                "observed_resident_batch",
                "sample_count",
                "passed",
            },
            label="held-out validation",
        )
        return cls(**dict(value))


def _relative_error(predicted: float, observed: float) -> float:
    if observed == 0.0:
        return 0.0 if predicted == 0.0 else float("inf")
    return abs(predicted - observed) / observed


def _interpolate_anchors(
    anchors: Sequence[StageBatchAnchor], batch_size: int
) -> float:
    if batch_size == 0:
        return 0.0
    if batch_size < 0:
        raise StageBatchCurveDomainError("active batch must be non-negative")
    if batch_size > MAX_BATCH_SIZE:
        raise StageBatchCurveDomainError(
            f"active batch {batch_size} exceeds calibrated maximum {MAX_BATCH_SIZE}"
        )
    first = anchors[0]
    if batch_size <= first.batch_size:
        return first.overhead_s * float(batch_size) / first.batch_size
    for left, right in zip(anchors, anchors[1:]):
        if batch_size <= right.batch_size:
            fraction = (
                (batch_size - left.batch_size)
                / float(right.batch_size - left.batch_size)
            )
            return left.overhead_s + fraction * (
                right.overhead_s - left.overhead_s
            )
    raise AssertionError("validated stage-batch anchors did not cover the domain")


@dataclass(frozen=True)
class StageBatchCurveProfile:
    """Portable sparse curve with self-verifying numeric diagnostics."""

    key: StageBatchCurveKey
    anchors: tuple[StageBatchAnchor, ...]
    validation: HeldOutValidation
    content_sha256: str
    batch_unit: str = BATCH_UNIT
    interpolation: str = INTERPOLATION
    max_batch_size: int = MAX_BATCH_SIZE

    def __post_init__(self) -> None:
        if self.batch_unit != BATCH_UNIT or self.interpolation != INTERPOLATION:
            raise StageBatchCurveValidationError(
                "stage-batch units/interpolation do not match the fixed contract"
            )
        if int(self.max_batch_size) != MAX_BATCH_SIZE:
            raise StageBatchCurveValidationError(
                f"stage-batch maximum must be {MAX_BATCH_SIZE}"
            )
        if not isinstance(self.key, StageBatchCurveKey):
            raise StageBatchCurveValidationError(
                "profile key must be a StageBatchCurveKey"
            )
        anchors = tuple(self.anchors)
        anchor_batches = tuple(anchor.batch_size for anchor in anchors)
        if (
            anchor_batches != ANCHOR_BATCHES
        ):
            raise StageBatchCurveValidationError(
                "anchor batches must exactly equal the complete required "
                f"set {ANCHOR_BATCHES}"
            )
        if not isinstance(self.validation, HeldOutValidation):
            raise StageBatchCurveValidationError(
                "profile validation must be a HeldOutValidation"
            )
        if self.validation.batch_size in ANCHOR_BATCHES:
            raise StageBatchCurveValidationError(
                "held-out validation must not be inserted into curve anchors"
            )
        if (
            self.validation.max_relative_error
            != DEFAULT_MAX_HELD_OUT_RELATIVE_ERROR
        ):
            raise StageBatchCurveValidationError(
                "held-out max_relative_error must equal the fixed publication "
                f"limit {DEFAULT_MAX_HELD_OUT_RELATIVE_ERROR}"
            )
        expected_held_out_overhead = _interpolate_anchors(
            anchors, HELD_OUT_BATCH,
        )
        if not math.isclose(
            self.validation.predicted_overhead_s,
            expected_held_out_overhead,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise StageBatchCurveValidationError(
                "held-out predicted_overhead_s does not match the fixed "
                "anchor interpolation"
            )
        object.__setattr__(self, "anchors", anchors)
        expected_content = _sha256(self._content_payload())
        if str(self.content_sha256) != expected_content:
            raise StageBatchCurveValidationError("stage-batch content digest mismatch")

    def _content_payload(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "batch_unit": self.batch_unit,
            "interpolation": self.interpolation,
            "max_batch_size": int(self.max_batch_size),
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "validation": self.validation.to_dict(),
        }

    def corrected_service_s(
        self,
        resident_batch: int,
        component_service_s: float,
    ) -> float:
        """Return component plus interpolated non-negative stage overhead."""

        if (
            isinstance(resident_batch, bool)
            or not isinstance(resident_batch, int)
        ):
            raise StageBatchCurveDomainError("resident batch must be an integer")
        try:
            component = _positive_finite(
                component_service_s,
                label="runtime.component_service_s",
            )
        except StageBatchCurveValidationError as exc:
            raise StageBatchCurveDomainError(str(exc)) from exc
        overhead = self.overhead_s(resident_batch)
        service = component + overhead
        if not math.isfinite(service) or service <= 0.0:
            # This is an invariant check, not a clamp. A validated positive
            # component and non-negative overhead cannot reach this branch.
            raise StageBatchCurveDomainError(
                "stage component plus overhead produced invalid service: "
                f"batch={resident_batch}, component={component:.9g}, "
                f"overhead={overhead:.9g}, service={service:.9g}"
            )
        return float(service)

    def overhead_s(self, resident_batch: int) -> float:
        """Return the interpolated non-negative overhead at resident batch.

        This method deliberately does not assign a scheduling boundary.  The
        owning applicator uses it once per collocated/direct-D engine cycle or
        once per completed direct-P frontier, respectively.
        """

        if (
            isinstance(resident_batch, bool)
            or not isinstance(resident_batch, int)
        ):
            raise StageBatchCurveDomainError("resident batch must be an integer")
        return float(_interpolate_anchors(self.anchors, resident_batch))

    def to_dict(self) -> dict[str, Any]:
        return {**self._content_payload(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StageBatchCurveProfile":
        if not isinstance(value, Mapping):
            raise StageBatchCurveValidationError("stage-batch profile must be an object")
        _reject_forbidden_dimensions(value)
        _require_exact_fields(
            value,
            {
                "key",
                "batch_unit",
                "interpolation",
                "max_batch_size",
                "anchors",
                "validation",
                "content_sha256",
            },
            label="stage-batch profile",
        )
        anchors_value = value["anchors"]
        if not isinstance(anchors_value, list):
            raise StageBatchCurveValidationError("anchors must be a list")
        return cls(
            key=StageBatchCurveKey.from_dict(value["key"]),
            batch_unit=str(value["batch_unit"]),
            interpolation=str(value["interpolation"]),
            max_batch_size=value["max_batch_size"],
            anchors=tuple(
                StageBatchAnchor.from_dict(anchor) for anchor in anchors_value
            ),
            validation=HeldOutValidation.from_dict(value["validation"]),
            content_sha256=str(value["content_sha256"]),
        )


def build_stage_batch_curve(
    *,
    key: StageBatchCurveKey,
    measurements: Sequence[StageBatchMeasurement],
    max_held_out_relative_error: float = DEFAULT_MAX_HELD_OUT_RELATIVE_ERROR,
) -> StageBatchCurveProfile:
    """Validate direct evidence and construct a publishable immutable profile.

    This function performs no regression and estimates no coefficients.  Every
    anchor is copied from the measurement at the same exact active batch.  All
    five anchors are mandatory; an invalid interior observation must fail the
    complete fit.  The B128 row is evaluated only after the anchors have been
    fixed.
    """

    if not isinstance(key, StageBatchCurveKey):
        raise StageBatchCurveValidationError("key must be a StageBatchCurveKey")
    limit = _nonnegative_finite(
        max_held_out_relative_error,
        label="max_held_out_relative_error",
    )
    if limit != DEFAULT_MAX_HELD_OUT_RELATIVE_ERROR:
        raise StageBatchCurveValidationError(
            "max_held_out_relative_error is fixed at "
            f"{DEFAULT_MAX_HELD_OUT_RELATIVE_ERROR} and cannot be overridden"
        )

    by_batch: dict[int, StageBatchMeasurement] = {}
    for measurement in measurements:
        if not isinstance(measurement, StageBatchMeasurement):
            raise StageBatchCurveValidationError(
                "measurements must contain StageBatchMeasurement values"
            )
        if measurement.key != key:
            raise StageBatchCurveValidationError(
                "all measurements must have the exact published hardware/stage key"
            )
        if measurement.target_batch in by_batch:
            raise StageBatchCurveValidationError(
                f"duplicate measurement for batch {measurement.target_batch}"
            )
        by_batch[measurement.target_batch] = measurement

    observed_batches = set(by_batch)
    allowed = {*ANCHOR_BATCHES, HELD_OUT_BATCH}
    extra = observed_batches - allowed
    missing_required = allowed - observed_batches
    if extra or missing_required:
        raise StageBatchCurveValidationError(
            "measurement batches must contain every required sparse anchor "
            f"{ANCHOR_BATCHES} and held-out "
            f"batch {HELD_OUT_BATCH} (missing={sorted(missing_required)}, "
            f"extra={sorted(extra)})"
        )
    if by_batch[MAX_BATCH_SIZE].engine_max_num_seqs < MAX_BATCH_SIZE:
        raise StageBatchCurveValidationError(
            f"the maximum anchor requires engine cap >= {MAX_BATCH_SIZE}"
        )

    anchors = tuple(
        StageBatchAnchor(
            batch_size=batch,
            observed_total_service_s=(
                by_batch[batch].observed_total_service_s
            ),
            reference_component_service_s=(
                by_batch[batch].reference_component_service_s
            ),
            overhead_s=by_batch[batch].overhead_s,
            engine_max_num_seqs=by_batch[batch].engine_max_num_seqs,
            observed_resident_batch=by_batch[batch].observed_resident_batch,
            sample_count=by_batch[batch].sample_count,
        )
        for batch in ANCHOR_BATCHES
    )
    held_out = by_batch[HELD_OUT_BATCH]
    predicted_overhead = _interpolate_anchors(anchors, HELD_OUT_BATCH)
    predicted_total = (
        held_out.reference_component_service_s + predicted_overhead
    )
    if not math.isfinite(predicted_total) or predicted_total <= 0.0:
        raise StageBatchCurveValidationError(
            "held-out interpolated overhead produces invalid service"
        )
    observed_total = held_out.observed_total_service_s
    absolute = abs(predicted_total - observed_total)
    relative = _relative_error(predicted_total, observed_total)
    validation = HeldOutValidation(
        batch_size=HELD_OUT_BATCH,
        observed_total_service_s=observed_total,
        reference_component_service_s=(
            held_out.reference_component_service_s
        ),
        observed_overhead_s=held_out.overhead_s,
        predicted_overhead_s=predicted_overhead,
        predicted_total_service_s=predicted_total,
        absolute_error_s=absolute,
        relative_error=relative,
        max_relative_error=limit,
        engine_max_num_seqs=held_out.engine_max_num_seqs,
        observed_resident_batch=held_out.observed_resident_batch,
        sample_count=held_out.sample_count,
        passed=relative <= limit,
    )

    content_payload = {
        "key": key.to_dict(),
        "batch_unit": BATCH_UNIT,
        "interpolation": INTERPOLATION,
        "max_batch_size": MAX_BATCH_SIZE,
        "anchors": [anchor.to_dict() for anchor in anchors],
        "validation": validation.to_dict(),
    }
    return StageBatchCurveProfile(
        key=key,
        anchors=anchors,
        validation=validation,
        content_sha256=_sha256(content_payload),
    )


def _system_hardware(hardware: Any) -> str:
    # Serving overhead includes CPU scheduler/frontend behavior around the
    # GPU engine boundary. A GPU SKU alone is therefore not a sufficient
    # hardware identity: two H100 systems with different host CPUs must not
    # share it.
    return validate_system_hardware(
        getattr(hardware, "system_hardware", None)
    )


def _block_from_document(
    document: Mapping[str, Any],
    *,
    system_hardware: str,
) -> Mapping[str, Any]:
    block = document.get(PROFILE_BLOCK_KEY)
    if not isinstance(block, Mapping):
        raise StageBatchCurveValidationError(
            f"{PROFILE_BLOCK_KEY} is absent or not an object"
        )
    _require_exact_fields(
        block,
        {"system_hardware", "profiles"},
        label="stage-batch profile block",
    )
    if block.get("system_hardware") != system_hardware:
        raise StageBatchCurveValidationError(
            "stage-batch profile block hardware identity mismatch"
        )
    profiles = block.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise StageBatchCurveValidationError(
            "stage-batch profile block has no profiles"
        )
    return profiles


def load_stage_batch_curve(
    hardware: Any,
    stage: str,
) -> StageBatchCurveProfile:
    """Load one exact hardware+serving-boundary curve.

    In particular, ``generator`` never falls back to
    ``generator_prefill``/``generator_decode`` and the reverse is also
    forbidden.  That makes missing direct disaggregated evidence fail closed.
    """

    generation = resolve_frozen_stage_batch_generation(
        hardware,
        calibration_mode="strict",
    )
    key = StageBatchCurveKey(generation.system_hardware, stage)
    profile = generation.materialize_profiles().get(key.stage)
    if profile is None:
        if key.stage in DISAGG_ROLE_STAGE_KEYS:
            raise StageBatchCurveValidationError(
                f"no direct disaggregated engine curve for {key.stage!r}; "
                "collocated profiles cannot satisfy a P/D role boundary"
            )
        raise StageBatchCurveValidationError(
            "no stage-batch curve for "
            f"({generation.system_hardware!r}, {key.stage!r})"
        )
    return profile


def _runtime_selection(
    *,
    system_hardware: str,
    enabled: bool,
    reason: str,
    profiles: Mapping[str, StageBatchCurveProfile] | None = None,
) -> dict[str, Any]:
    """Build one JSON-safe, run-frozen uncalibrated-spec selection.

    A selection deliberately contains either the complete six-profile
    generation or no profiles at all.  It is passed to spawned DES workers,
    so it contains only plain JSON values.
    """

    resolved_profiles = dict(profiles or {})
    documents = {
        stage: resolved_profiles[stage].to_dict()
        for stage in sorted(resolved_profiles)
    }
    generation_sha256 = (
        _sha256({
            "system_hardware": system_hardware,
            "profiles": documents,
        })
        if enabled else None
    )
    return {
        "mode": RUNTIME_SELECTION_MODE,
        "system_hardware": system_hardware,
        "enabled": bool(enabled),
        "reason": str(reason),
        "required_stages": list(COMPLETE_ENGINE_STAGE_KEYS),
        "generation_sha256": generation_sha256,
        "profiles": documents,
    }


def resolve_uncalibrated_spec_stage_batch_selection(
    hardware: Any,
) -> dict[str, Any]:
    """Freeze the all-or-nothing stage generation for an uncalibrated spec.

    Special mode permits a hardware specification with no stage calibration,
    but it must never give only some deployment families a calibrated
    scheduler overhead.  Consequently, only a complete and valid
    :data:`COMPLETE_ENGINE_STAGE_KEYS` generation is enabled.  An absent file,
    a valid collocated-only pair, or another known-key incomplete generation
    disables *all* stage curves without error.

    Presence is not validity.  A file that exists but has malformed JSON,
    schema, hardware identity, embedded profile identity, digest, or unknown
    stage keys still fails closed.  Existing profiles in an
    incomplete generation are individually parsed before disabling so damaged
    evidence cannot be hidden by deleting a sibling key.
    """

    system_hardware = _system_hardware(hardware)
    path = (
        calibration_profile_root()
        / PROFILE_DIRECTORY
        / f"{system_hardware}.json"
    )
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError:
        return _runtime_selection(
            system_hardware=system_hardware,
            enabled=False,
            reason="profile_file_absent",
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBatchCurveValidationError(
            f"cannot read stage-batch profile file: {path}"
        ) from exc

    if not isinstance(document, Mapping):
        raise StageBatchCurveValidationError(
            "stage-batch profile document must be an object"
        )
    if set(document) != {PROFILE_BLOCK_KEY}:
        raise StageBatchCurveValidationError(
            "stage-batch profile file must contain only its authoritative block"
        )
    block = document.get(PROFILE_BLOCK_KEY)
    if not isinstance(block, Mapping):
        raise StageBatchCurveValidationError(
            f"{PROFILE_BLOCK_KEY} is absent or not an object"
        )
    _require_exact_fields(
        block,
        {"system_hardware", "profiles"},
        label="stage-batch profile block",
    )
    if block.get("system_hardware") != system_hardware:
        raise StageBatchCurveValidationError(
            "stage-batch profile block hardware identity mismatch"
        )
    raw_profiles = block.get("profiles")
    if not isinstance(raw_profiles, Mapping):
        raise StageBatchCurveValidationError(
            "stage-batch profiles must be an object"
        )

    actual_stages = set(raw_profiles)
    complete_stages = set(COMPLETE_ENGINE_STAGE_KEYS)
    unknown_stages = actual_stages - complete_stages
    if unknown_stages:
        raise StageBatchCurveValidationError(
            "stage-batch profile block contains unknown stages: "
            f"{sorted(unknown_stages)}"
        )

    parsed: dict[str, StageBatchCurveProfile] = {}
    for map_stage in sorted(actual_stages):
        raw = raw_profiles.get(map_stage)
        if not isinstance(raw, Mapping):
            raise StageBatchCurveValidationError(
                f"stage-batch curve {map_stage!r} is not an object"
            )
        profile = StageBatchCurveProfile.from_dict(raw)
        if profile.key != StageBatchCurveKey(system_hardware, map_stage):
            raise StageBatchCurveValidationError(
                "stage-batch profile map key disagrees with embedded identity"
            )
        parsed[map_stage] = profile

    collocated_stages = set(REQUIRED_COLLOCATED_ENGINE_STAGES)
    if collocated_stages.issubset(actual_stages):
        # Validate the independently complete collocated sub-generation even
        # when one or more direct-role siblings are missing.  Otherwise an
        # incomplete file could hide mismatched shared evidence/anchors in the
        # pair that is already present.
        validate_stage_batch_curve_generation(tuple(
            parsed[stage] for stage in REQUIRED_COLLOCATED_ENGINE_STAGES
        ))
    if actual_stages == collocated_stages:
        # A pair is a valid strict-mode generation but remains insufficient
        # for special mode's all-topology fairness rule.
        return _runtime_selection(
            system_hardware=system_hardware,
            enabled=False,
            reason="collocated_only_generation",
        )
    if actual_stages != complete_stages:
        return _runtime_selection(
            system_hardware=system_hardware,
            enabled=False,
            reason="incomplete_generation",
        )

    validate_stage_batch_curve_generation(tuple(parsed.values()))
    return _runtime_selection(
        system_hardware=system_hardware,
        enabled=True,
        reason="complete_six_profile_generation",
        profiles=parsed,
    )


def stage_batch_profiles_from_runtime_selection(
    selection: Mapping[str, Any],
    hardware: Any,
) -> dict[str, StageBatchCurveProfile]:
    """Validate and materialize a frozen JSON-safe runtime selection.

    The empty mapping is the explicit analytical-only state.  A non-empty
    result always contains all six profiles and has passed the same generation
    validator as a direct disk load.
    """

    if not isinstance(selection, Mapping):
        raise StageBatchCurveValidationError(
            "stage-batch runtime selection must be an object"
        )
    _require_exact_fields(
        selection,
        {
            "mode",
            "system_hardware",
            "enabled",
            "reason",
            "required_stages",
            "generation_sha256",
            "profiles",
        },
        label="stage-batch runtime selection",
    )
    if selection.get("mode") != RUNTIME_SELECTION_MODE:
        raise StageBatchCurveValidationError(
            "stage-batch runtime selection mode mismatch"
        )
    system_hardware = _system_hardware(hardware)
    if selection.get("system_hardware") != system_hardware:
        raise StageBatchCurveValidationError(
            "stage-batch runtime selection hardware identity mismatch"
        )
    required_stages = selection.get("required_stages")
    if required_stages != list(COMPLETE_ENGINE_STAGE_KEYS):
        raise StageBatchCurveValidationError(
            "stage-batch runtime selection must require the complete six stages"
        )
    enabled = selection.get("enabled")
    if type(enabled) is not bool:
        raise StageBatchCurveValidationError(
            "stage-batch runtime selection enabled must be boolean"
        )
    reason = selection.get("reason")
    if not isinstance(reason, str) or not reason:
        raise StageBatchCurveValidationError(
            "stage-batch runtime selection reason must be non-empty"
        )
    raw_profiles = selection.get("profiles")
    if not isinstance(raw_profiles, Mapping):
        raise StageBatchCurveValidationError(
            "stage-batch runtime selection profiles must be an object"
        )

    if not enabled:
        if raw_profiles or selection.get("generation_sha256") is not None:
            raise StageBatchCurveValidationError(
                "disabled stage-batch runtime selection must contain no generation"
            )
        return {}

    if set(raw_profiles) != set(COMPLETE_ENGINE_STAGE_KEYS):
        raise StageBatchCurveValidationError(
            "enabled stage-batch runtime selection must contain all six profiles"
        )
    generation_sha256 = selection.get("generation_sha256")
    if (
        not isinstance(generation_sha256, str)
        or _SHA256_RE.fullmatch(generation_sha256) is None
    ):
        raise StageBatchCurveValidationError(
            "enabled stage-batch runtime selection has invalid generation digest"
        )
    expected_generation_sha256 = _sha256({
        "system_hardware": system_hardware,
        "profiles": dict(raw_profiles),
    })
    if generation_sha256 != expected_generation_sha256:
        raise StageBatchCurveValidationError(
            "stage-batch runtime selection generation digest mismatch"
        )

    parsed: dict[str, StageBatchCurveProfile] = {}
    for map_stage in sorted(raw_profiles):
        raw = raw_profiles[map_stage]
        if not isinstance(raw, Mapping):
            raise StageBatchCurveValidationError(
                f"stage-batch runtime profile {map_stage!r} is not an object"
            )
        profile = StageBatchCurveProfile.from_dict(raw)
        if profile.key != StageBatchCurveKey(system_hardware, map_stage):
            raise StageBatchCurveValidationError(
                "stage-batch runtime profile map key disagrees with identity"
            )
        parsed[map_stage] = profile
    validate_stage_batch_curve_generation(tuple(parsed.values()))
    return parsed


def _load_strict_stage_batch_profiles(
    hardware: Any,
) -> dict[str, StageBatchCurveProfile]:
    """Read and validate one whole strict-mode hardware generation once."""

    system_hardware = _system_hardware(hardware)
    path = (
        calibration_profile_root()
        / PROFILE_DIRECTORY
        / f"{system_hardware}.json"
    )
    try:
        document = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise StageBatchCurveValidationError(
            f"stage-batch profile file is absent: {path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBatchCurveValidationError(
            f"cannot read stage-batch profile file: {path}"
        ) from exc
    if not isinstance(document, Mapping):
        raise StageBatchCurveValidationError(
            "stage-batch profile document must be an object"
        )
    if set(document) != {PROFILE_BLOCK_KEY}:
        raise StageBatchCurveValidationError(
            "stage-batch profile file must contain only its authoritative block"
        )
    raw_profiles = _block_from_document(
        document,
        system_hardware=system_hardware,
    )
    actual_stages = set(raw_profiles)
    allowed_stage_sets = {
        frozenset(REQUIRED_COLLOCATED_ENGINE_STAGES),
        frozenset(COMPLETE_ENGINE_STAGE_KEYS),
    }
    if frozenset(actual_stages) not in allowed_stage_sets:
        raise StageBatchCurveValidationError(
            "stage-batch profile block must contain either the exact "
            "collocated pair or the exact complete six-profile generation "
            f"(present={sorted(actual_stages)})"
        )

    parsed: dict[str, StageBatchCurveProfile] = {}
    for map_stage in sorted(actual_stages):
        raw = raw_profiles.get(map_stage)
        if not isinstance(raw, Mapping):
            raise StageBatchCurveValidationError(
                f"stage-batch curve {map_stage!r} is not an object"
            )
        profile = StageBatchCurveProfile.from_dict(raw)
        if profile.key != StageBatchCurveKey(system_hardware, map_stage):
            raise StageBatchCurveValidationError(
                "stage-batch profile map key disagrees with embedded identity"
            )
        parsed[map_stage] = profile
    validate_stage_batch_curve_generation(tuple(parsed.values()))
    return parsed


@dataclass(frozen=True)
class FrozenStageBatchGeneration:
    """Immutable, JSON-round-trippable stage-curve generation snapshot.

    Profile documents are retained as canonical JSON strings so every dataclass
    field is itself JSON-safe and immutable.  :meth:`to_dict` exposes the plain
    document used for process dispatch.  The snapshot binds both each profile's
    content digests and the digest of the complete selected
    generation; consumers never need to reopen the calibration store.
    """

    calibration_mode: str
    system_hardware: str
    enabled: bool
    reason: str
    generation_sha256: str | None
    profiles_json: tuple[tuple[str, str], ...]
    profile_content_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.calibration_mode not in {"strict", "uncalibrated_spec"}:
            raise StageBatchCurveValidationError(
                "frozen stage-batch calibration_mode must be strict or "
                "uncalibrated_spec"
            )
        system_hardware = validate_system_hardware(self.system_hardware)
        if type(self.enabled) is not bool:
            raise StageBatchCurveValidationError(
                "frozen stage-batch enabled must be boolean"
            )
        if not isinstance(self.reason, str) or not self.reason:
            raise StageBatchCurveValidationError(
                "frozen stage-batch reason must be non-empty"
            )
        if tuple(sorted(self.profiles_json)) != self.profiles_json:
            raise StageBatchCurveValidationError(
                "frozen stage-batch profiles must use unique sorted stage keys"
            )
        if tuple(sorted(self.profile_content_sha256)) != (
            self.profile_content_sha256
        ):
            raise StageBatchCurveValidationError(
                "frozen profile content digests must use sorted stage keys"
            )
        if not self.enabled:
            if self.calibration_mode != "uncalibrated_spec":
                raise StageBatchCurveValidationError(
                    "strict frozen stage-batch generation cannot be disabled"
                )
            if (
                self.generation_sha256 is not None
                or self.profiles_json
                or self.profile_content_sha256
            ):
                raise StageBatchCurveValidationError(
                    "disabled frozen stage-batch generation must contain no "
                    "profiles or digests"
                )
            object.__setattr__(self, "system_hardware", system_hardware)
            return

        if (
            not isinstance(self.generation_sha256, str)
            or _SHA256_RE.fullmatch(self.generation_sha256) is None
        ):
            raise StageBatchCurveValidationError(
                "enabled frozen stage-batch generation has invalid digest"
            )
        profiles = self.materialize_profiles()
        stages = frozenset(profiles)
        collocated = frozenset(REQUIRED_COLLOCATED_ENGINE_STAGES)
        complete = frozenset(COMPLETE_ENGINE_STAGE_KEYS)
        if self.calibration_mode == "strict":
            if stages not in {collocated, complete}:
                raise StageBatchCurveValidationError(
                    "strict frozen generation must contain the collocated pair "
                    "or complete six profiles"
                )
        elif stages != complete:
            raise StageBatchCurveValidationError(
                "enabled uncalibrated-spec frozen generation must contain the "
                "complete six profiles"
            )
        validate_stage_batch_curve_generation(tuple(profiles.values()))
        documents = {
            stage: profiles[stage].to_dict() for stage in sorted(profiles)
        }
        expected_generation = _sha256({
            "system_hardware": system_hardware,
            "profiles": documents,
        })
        if self.generation_sha256 != expected_generation:
            raise StageBatchCurveValidationError(
                "frozen stage-batch generation digest mismatch"
            )
        expected_content = tuple(
            (stage, profiles[stage].content_sha256)
            for stage in sorted(profiles)
        )
        if self.profile_content_sha256 != expected_content:
            raise StageBatchCurveValidationError(
                "frozen stage-batch profile content digests mismatch"
            )
        object.__setattr__(self, "system_hardware", system_hardware)

    @classmethod
    def from_profiles(
        cls,
        *,
        calibration_mode: str,
        system_hardware: str,
        enabled: bool,
        reason: str,
        profiles: Mapping[str, StageBatchCurveProfile] | None = None,
    ) -> "FrozenStageBatchGeneration":
        resolved = dict(profiles or {})
        if not enabled and resolved:
            raise StageBatchCurveValidationError(
                "disabled frozen generation cannot retain profiles"
            )
        documents = {
            stage: resolved[stage].to_dict() for stage in sorted(resolved)
        }
        generation_sha256 = (
            _sha256({
                "system_hardware": system_hardware,
                "profiles": documents,
            })
            if enabled else None
        )
        return cls(
            calibration_mode=str(calibration_mode),
            system_hardware=str(system_hardware),
            enabled=bool(enabled),
            reason=str(reason),
            generation_sha256=generation_sha256,
            profiles_json=tuple(
                (stage, _canonical_json(document).decode("utf-8"))
                for stage, document in sorted(documents.items())
            ),
            profile_content_sha256=tuple(
                (stage, resolved[stage].content_sha256)
                for stage in sorted(resolved)
            ),
        )

    def profile_documents(self) -> dict[str, dict[str, Any]]:
        documents: dict[str, dict[str, Any]] = {}
        for stage, encoded in self.profiles_json:
            try:
                document = json.loads(encoded)
            except (TypeError, json.JSONDecodeError) as exc:
                raise StageBatchCurveValidationError(
                    f"frozen stage-batch profile {stage!r} is not canonical JSON"
                ) from exc
            if not isinstance(document, dict):
                raise StageBatchCurveValidationError(
                    f"frozen stage-batch profile {stage!r} is not an object"
                )
            if encoded != _canonical_json(document).decode("utf-8"):
                raise StageBatchCurveValidationError(
                    f"frozen stage-batch profile {stage!r} is not canonical JSON"
                )
            if stage in documents:
                raise StageBatchCurveValidationError(
                    f"duplicate frozen stage-batch profile {stage!r}"
                )
            documents[stage] = document
        return documents

    def materialize_profiles(self) -> dict[str, StageBatchCurveProfile]:
        profiles: dict[str, StageBatchCurveProfile] = {}
        for stage, document in self.profile_documents().items():
            profile = StageBatchCurveProfile.from_dict(document)
            if profile.key != StageBatchCurveKey(self.system_hardware, stage):
                raise StageBatchCurveValidationError(
                    "frozen stage-batch profile map key disagrees with identity"
                )
            profiles[stage] = profile
        return profiles

    def to_dict(self) -> dict[str, Any]:
        """Return the complete plain-JSON process-dispatch document."""

        return {
            "calibration_mode": self.calibration_mode,
            "system_hardware": self.system_hardware,
            "enabled": self.enabled,
            "reason": self.reason,
            "generation_sha256": self.generation_sha256,
            "profiles": self.profile_documents(),
            "profile_content_sha256": dict(self.profile_content_sha256),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "FrozenStageBatchGeneration":
        if not isinstance(value, Mapping):
            raise StageBatchCurveValidationError(
                "frozen stage-batch generation must be an object"
            )
        _require_exact_fields(
            value,
            {
                "calibration_mode",
                "system_hardware",
                "enabled",
                "reason",
                "generation_sha256",
                "profiles",
                "profile_content_sha256",
            },
            label="frozen stage-batch generation",
        )
        profiles = value.get("profiles")
        content = value.get("profile_content_sha256")
        if not isinstance(profiles, Mapping):
            raise StageBatchCurveValidationError(
                "frozen stage-batch profiles must be an object"
            )
        if not isinstance(content, Mapping):
            raise StageBatchCurveValidationError(
                "frozen stage-batch digest maps must be objects"
            )
        return cls(
            calibration_mode=str(value["calibration_mode"]),
            system_hardware=str(value["system_hardware"]),
            enabled=value["enabled"],
            reason=str(value["reason"]),
            generation_sha256=value["generation_sha256"],
            profiles_json=tuple(
                (str(stage), _canonical_json(document).decode("utf-8"))
                for stage, document in sorted(
                    profiles.items(), key=lambda item: str(item[0]),
                )
            ),
            profile_content_sha256=tuple(
                (str(stage), str(digest))
                for stage, digest in sorted(
                    content.items(), key=lambda item: str(item[0]),
                )
            ),
        )


def resolve_frozen_stage_batch_generation(
    hardware: Any,
    *,
    calibration_mode: str,
) -> FrozenStageBatchGeneration:
    """Resolve one run-frozen whole-file generation without per-stage reads."""

    mode = str(calibration_mode)
    system_hardware = _system_hardware(hardware)
    if mode == "strict":
        profiles = _load_strict_stage_batch_profiles(hardware)
        reason = (
            "strict_complete_six_profile_generation"
            if set(profiles) == set(COMPLETE_ENGINE_STAGE_KEYS)
            else "strict_collocated_pair_generation"
        )
        return FrozenStageBatchGeneration.from_profiles(
            calibration_mode=mode,
            system_hardware=system_hardware,
            enabled=True,
            reason=reason,
            profiles=profiles,
        )
    if mode == "uncalibrated_spec":
        # Reuse the existing all-or-nothing fairness resolver verbatim.  It
        # performs one whole-file read and returns either complete-six profile
        # documents or the explicit analytical-only empty state.
        selection = resolve_uncalibrated_spec_stage_batch_selection(hardware)
        profiles = stage_batch_profiles_from_runtime_selection(
            selection,
            hardware,
        )
        generation = FrozenStageBatchGeneration.from_profiles(
            calibration_mode=mode,
            system_hardware=system_hardware,
            enabled=bool(selection["enabled"]),
            reason=str(selection["reason"]),
            profiles=profiles,
        )
        if generation.generation_sha256 != selection["generation_sha256"]:
            raise StageBatchCurveValidationError(
                "uncalibrated-spec frozen generation digest disagrees with "
                "runtime selection"
            )
        return generation
    raise StageBatchCurveValidationError(
        "stage-batch calibration_mode must be strict or uncalibrated_spec"
    )


@dataclass(frozen=True)
class EngineCycleCurveApplicator:
    """The only runtime application boundary for frozen stage curves.

    ``component_service_s`` must already be the sum of every analytical
    component scheduled in this one cycle.  A unified engine therefore applies
    its bare owning-stage curve once to a mixed P+D cycle.  Direct-D does the
    same for a decode cycle.  Direct-P is intentionally different: its raw
    parent window measures a cohort completion frontier, so
    :meth:`prefill_frontier_corrected_service_s` distributes that overhead by
    the number of logical prefills completed in a DES step.  Disaggregated P
    and D engines select only their direct role profile and never fall back to
    the collocated curve.
    """

    generation: FrozenStageBatchGeneration

    def __post_init__(self) -> None:
        if not isinstance(self.generation, FrozenStageBatchGeneration):
            raise StageBatchCurveValidationError(
                "engine-cycle applicator requires a frozen generation"
            )
        object.__setattr__(
            self,
            "_profiles",
            self.generation.materialize_profiles(),
        )

    def corrected_service_s(
        self,
        *,
        engine_stage: str,
        role: str,
        prefill_stage: str,
        decode_stage: str,
        resident_batch: int,
        component_service_s: float,
    ) -> float:
        if not self.generation.enabled:
            return float(component_service_s)

        normalized_role = str(role)
        engine = str(engine_stage)
        if normalized_role == "unified":
            if engine not in REQUIRED_COLLOCATED_ENGINE_STAGES:
                raise StageBatchCurveValidationError(
                    f"unknown collocated engine-stage boundary {engine!r}"
                )
            curve_stage = engine
        elif normalized_role in {"prefill", "decode"}:
            curve_stage = str(
                prefill_stage if normalized_role == "prefill" else decode_stage
            )
            try:
                performance_stage = TAXONOMY.performance_stage(curve_stage)
                owner = TAXONOMY.user_stage_of(curve_stage).value
            except (KeyError, ValueError) as exc:
                raise StageBatchCurveValidationError(
                    f"unknown direct disaggregated role {curve_stage!r}"
                ) from exc
            if performance_stage.kind != normalized_role or owner != engine:
                raise StageBatchCurveValidationError(
                    f"direct disaggregated role {curve_stage!r} does not match "
                    f"engine={engine!r}, role={normalized_role!r}"
                )
        else:
            raise StageBatchCurveValidationError(
                f"invalid engine-cycle role {normalized_role!r}"
            )

        profile = self._profiles.get(curve_stage)
        if profile is None:
            if normalized_role in {"prefill", "decode"}:
                raise StageBatchCurveValidationError(
                    f"no direct disaggregated engine curve for {curve_stage!r}; "
                    "collocated profiles cannot satisfy a P/D role boundary"
                )
            raise StageBatchCurveValidationError(
                f"no collocated engine curve for {curve_stage!r}"
            )
        return profile.corrected_service_s(
            resident_batch,
            component_service_s,
        )

    def prefill_frontier_corrected_service_s(
        self,
        *,
        engine_stage: str,
        prefill_stage: str,
        decode_stage: str,
        resident_batch: int,
        completed_prefills: int,
        component_service_s: float,
    ) -> float:
        """Apply the direct-P completion-frontier overhead to one DES step."""

        if not self.generation.enabled:
            return float(component_service_s)
        if (
            isinstance(completed_prefills, bool)
            or not isinstance(completed_prefills, int)
            or completed_prefills < 0
        ):
            raise StageBatchCurveDomainError(
                "completed prefills must be a non-negative integer"
            )
        if completed_prefills > resident_batch:
            raise StageBatchCurveDomainError(
                "completed prefills cannot exceed resident batch"
            )
        # Reuse the canonical direct-role resolver and its fail-closed profile
        # selection, but extract the bare frontier delta before distributing
        # it across the B logical completions represented by the calibration.
        baseline = self.corrected_service_s(
            engine_stage=engine_stage,
            role="prefill",
            prefill_stage=prefill_stage,
            decode_stage=decode_stage,
            resident_batch=resident_batch,
            component_service_s=1.0,
        )
        overhead = float(baseline) - 1.0
        component = _positive_finite(
            component_service_s,
            label="runtime.component_service_s",
        )
        service = component + overhead * (
            float(completed_prefills) / float(resident_batch)
        )
        if not math.isfinite(service) or service <= 0.0:
            raise StageBatchCurveDomainError(
                "direct-P frontier produced invalid service: "
                f"batch={resident_batch}, completed={completed_prefills}, "
                f"component={component:.9g}, overhead={overhead:.9g}, "
                f"service={service:.9g}"
            )
        return float(service)


def validate_stage_batch_curve_generation(
    curves: Sequence[StageBatchCurveProfile],
) -> tuple[StageBatchCurveProfile, ...]:
    """Validate one exact collocated-only or complete P/D generation.

    The collocated pair binds one complete collocated engine-cycle boundary.
    When P/D curves are present, all four role keys are mandatory. The two
    query-expansion boundaries must be exact numerical aliases of their
    corresponding generator boundaries. Acquisition evidence is deliberately
    absent from this runtime contract.
    """

    resolved = tuple(curves)
    if not resolved or any(
        not isinstance(curve, StageBatchCurveProfile) for curve in resolved
    ):
        raise StageBatchCurveValidationError(
            "stage-batch generation must contain profile objects"
        )
    stages = [curve.key.stage for curve in resolved]
    stage_set = set(stages)
    collocated_stage_set = set(REQUIRED_COLLOCATED_ENGINE_STAGES)
    complete_stage_set = set(COMPLETE_ENGINE_STAGE_KEYS)
    if (
        len(stages) != len(stage_set)
        or stage_set not in (collocated_stage_set, complete_stage_set)
    ):
        raise StageBatchCurveValidationError(
            "stage-batch generation must exactly cover either the two "
            "collocated vLLM engine stages or all six collocated/direct-P/D "
            "engine stages"
        )
    system_hardware = {curve.key.system_hardware for curve in resolved}
    if len(system_hardware) != 1:
        raise StageBatchCurveValidationError(
            "stage-batch generation must use one system_hardware"
        )
    by_stage = {curve.key.stage: curve for curve in resolved}
    generator = by_stage["generator"]
    query_expansion = by_stage["query_expansion"]
    if (
        generator.anchors != query_expansion.anchors
        or generator.validation != query_expansion.validation
    ):
        raise StageBatchCurveValidationError(
            "query_expansion anchors/validation must exactly equal generator"
        )
    if stage_set == complete_stage_set:
        for role in ("prefill", "decode"):
            source = by_stage[f"generator_{role}"]
            alias = by_stage[f"query_expansion_{role}"]
            if (
                source.anchors != alias.anchors
                or source.validation != alias.validation
            ):
                raise StageBatchCurveValidationError(
                    f"query-expansion {role} anchors/validation must exactly equal "
                    f"generator {role}"
                )
    return resolved


def publish_stage_batch_curves(
    profile_path: str | Path,
    curves: Sequence[StageBatchCurveProfile],
) -> None:
    """Atomically replace the complete hardware stage-curve generation."""

    resolved = tuple(curves)
    if not resolved:
        raise StageBatchCurveValidationError(
            "at least one stage-batch curve is required"
        )
    system_hardware_values = {
        validate_system_hardware(curve.key.system_hardware)
        for curve in resolved
    }
    stages = [curve.key.stage for curve in resolved]
    if len(system_hardware_values) != 1 or len(stages) != len(set(stages)):
        raise StageBatchCurveValidationError(
            "published curves must have one system_hardware and unique stages"
        )
    stage_set = set(stages)
    if stage_set not in (
        set(REQUIRED_COLLOCATED_ENGINE_STAGES),
        set(COMPLETE_ENGINE_STAGE_KEYS),
    ):
        raise StageBatchCurveValidationError(
            "published curves must exactly cover either the collocated pair "
            "or the complete six collocated/direct-P/D engine stages"
        )
    validate_stage_batch_curve_generation(resolved)
    system_hardware = next(iter(system_hardware_values))
    path = Path(profile_path).resolve()
    expected_path = (
        calibration_profile_root()
        / PROFILE_DIRECTORY
        / f"{system_hardware}.json"
    ).resolve()
    if path != expected_path:
        raise StageBatchCurveValidationError(
            "stage-batch publication path must be canonical for its "
            f"system_hardware: expected {expected_path}, got {path}"
        )
    document = {
        PROFILE_BLOCK_KEY: {
            "system_hardware": system_hardware,
            "profiles": {
                curve.key.stage: curve.to_dict()
                for curve in sorted(
                    resolved, key=lambda item: item.key.stage
                )
            },
        }
    }
    atomic_write_json(path, document)


__all__ = [
    "ANCHOR_BATCHES",
    "BATCH_UNIT",
    "DEFAULT_MAX_HELD_OUT_RELATIVE_ERROR",
    "HELD_OUT_BATCH",
    "INTERPOLATION",
    "MAX_BATCH_SIZE",
    "PROFILE_BLOCK_KEY",
    "PROFILE_DIRECTORY",
    "RUNTIME_SELECTION_MODE",
    "COMPLETE_ENGINE_STAGE_KEYS",
    "DISAGG_ROLE_STAGE_KEYS",
    "EngineCycleCurveApplicator",
    "FrozenStageBatchGeneration",
    "REQUIRED_COLLOCATED_ENGINE_STAGES",
    "HeldOutValidation",
    "StageBatchAnchor",
    "StageBatchCurveDomainError",
    "StageBatchCurveError",
    "StageBatchCurveKey",
    "StageBatchCurveProfile",
    "StageBatchCurveValidationError",
    "StageBatchMeasurement",
    "build_stage_batch_curve",
    "load_stage_batch_curve",
    "publish_stage_batch_curves",
    "resolve_uncalibrated_spec_stage_batch_selection",
    "resolve_frozen_stage_batch_generation",
    "stage_batch_profiles_from_runtime_selection",
    "validate_stage_batch_curve_generation",
    "validate_system_hardware",
]
