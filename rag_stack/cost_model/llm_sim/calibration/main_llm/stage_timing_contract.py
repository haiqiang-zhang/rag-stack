"""Production parent-completion interdeparture contract for CM stages.

Stage shape proof and stage timing are deliberately separate acquisitions.
The timing server runs with telemetry disabled.  While its scheduler is paused,
the acquisition queues eight identical request cohorts of size ``B`` and
freezes the acknowledged in-memory dispatch counts ``[B, 2B, ..., 8B]``.  The
engine cap is exactly ``B``.  After resume, cohort zero is a warmup and seven
parent-completion
interdeparture intervals between cohort endpoints are the production samples.

This module consumes only parent request timings plus dispatch-count proof; no
instrumentation duration or fitted overhead enters service.  The complete
out=N collocated-engine interval is the only publishable timing evidence.  An
out=1 interval may be retained as optional raw acquisition diagnostics, but it
is never subtracted from the full interval and never identifies a P/D service
or residual.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time
from typing import Any, Callable, Mapping, Sequence


CLOCK_DOMAIN = "parent_process_clock_gettime_CLOCK_MONOTONIC"
PARENT_RESPONSE_CLOCK_SOURCE = "time.monotonic"

COHORT_SCHEMA = "rag_stack.production_parent_completion_cohort"
INTERDEPARTURE_SCHEMA = "rag_stack.production_parent_interdeparture"
TIMING_SCHEMA = "rag_stack.production_stage_batch_timing"

PRODUCTION_COHORT_COUNT = 8
PRODUCTION_INTERVAL_COUNT = PRODUCTION_COHORT_COUNT - 1
MAX_RELATIVE_INTERVAL_MAD = 0.20
ENGINECORE_DISPATCH_FENCE = (
    "async_llm_add_request_return_then_fifo_is_paused_ack"
)


class StageTimingContractError(ValueError):
    """Frozen production timing evidence is incomplete or inconsistent."""


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise StageTimingContractError(
            f"{label} fields do not match the exact schema "
            f"(missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)})"
        )


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StageTimingContractError(f"{label} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StageTimingContractError(
            f"{label} must be a non-negative integer"
        )
    return int(value)


def _finite_float(
    value: Any,
    *,
    label: str,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageTimingContractError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise StageTimingContractError(f"{label} must be {qualifier}")
    return result


def _same_float(left: float, right: float) -> bool:
    return math.isclose(
        float(left), float(right), rel_tol=1e-12, abs_tol=1e-15
    )


@dataclass(frozen=True)
class ClockDomainProof:
    """Frozen proof for every parent request timestamp in one timing run."""

    domain: str
    parent_response_source: str
    implementation: str
    monotonic: bool
    adjustable: bool
    resolution_s: float

    def __post_init__(self) -> None:
        if self.domain != CLOCK_DOMAIN:
            raise StageTimingContractError("clock domain is not CLOCK_MONOTONIC")
        if self.parent_response_source != PARENT_RESPONSE_CLOCK_SOURCE:
            raise StageTimingContractError("parent response clock source is invalid")
        if (
            not isinstance(self.implementation, str)
            or "CLOCK_MONOTONIC" not in self.implementation
        ):
            raise StageTimingContractError(
                "clock implementation must be clock_gettime(CLOCK_MONOTONIC)"
            )
        if self.monotonic is not True or self.adjustable is not False:
            raise StageTimingContractError(
                "production timing clock must be monotonic and non-adjustable"
            )
        resolution = _finite_float(
            self.resolution_s,
            label="clock.resolution_s",
            positive=True,
        )
        object.__setattr__(self, "resolution_s", resolution)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "parent_response_source": self.parent_response_source,
            "implementation": self.implementation,
            "monotonic": self.monotonic,
            "adjustable": self.adjustable,
            "resolution_s": self.resolution_s,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClockDomainProof":
        if not isinstance(value, Mapping):
            raise StageTimingContractError("clock proof must be an object")
        _require_exact_fields(
            value,
            {
                "domain",
                "parent_response_source",
                "implementation",
                "monotonic",
                "adjustable",
                "resolution_s",
            },
            label="clock proof",
        )
        return cls(
            domain=value["domain"],
            parent_response_source=value["parent_response_source"],
            implementation=value["implementation"],
            monotonic=value["monotonic"],
            adjustable=value["adjustable"],
            resolution_s=value["resolution_s"],
        )


def capture_clock_domain_proof(
    *,
    get_clock_info: Callable[[str], Any] = time.get_clock_info,
    sample_clock: Callable[[], float] = time.monotonic,
) -> ClockDomainProof:
    """Capture the only clock used by production parent timing."""

    info = get_clock_info("monotonic")
    implementation = getattr(info, "implementation", None)
    if (
        not isinstance(implementation, str)
        or "CLOCK_MONOTONIC" not in implementation
    ):
        raise StageTimingContractError(
            "time.monotonic is not clock_gettime(CLOCK_MONOTONIC)"
        )
    if (
        getattr(info, "monotonic", None) is not True
        or getattr(info, "adjustable", None) is not False
    ):
        raise StageTimingContractError(
            "time.monotonic must be monotonic and non-adjustable"
        )
    resolution = _finite_float(
        getattr(info, "resolution", None),
        label="time.monotonic resolution",
        positive=True,
    )
    before = _finite_float(sample_clock(), label="clock sample before")
    after = _finite_float(sample_clock(), label="clock sample after")
    if after < before:
        raise StageTimingContractError("time.monotonic moved backwards")
    return ClockDomainProof(
        domain=CLOCK_DOMAIN,
        parent_response_source=PARENT_RESPONSE_CLOCK_SOURCE,
        implementation=implementation,
        monotonic=True,
        adjustable=False,
        resolution_s=resolution,
    )


@dataclass(frozen=True)
class RequestTiming:
    """Parent-clock send and complete-response timestamps for one request."""

    request_index: int
    sent_monotonic_s: float
    completed_monotonic_s: float

    def __post_init__(self) -> None:
        index = _nonnegative_int(
            self.request_index,
            label="request_timing.request_index",
        )
        sent = _finite_float(
            self.sent_monotonic_s,
            label="request_timing.sent_monotonic_s",
        )
        completed = _finite_float(
            self.completed_monotonic_s,
            label="request_timing.completed_monotonic_s",
        )
        if completed <= sent:
            raise StageTimingContractError(
                "parent response completion must follow request send"
            )
        object.__setattr__(self, "request_index", index)
        object.__setattr__(self, "sent_monotonic_s", sent)
        object.__setattr__(self, "completed_monotonic_s", completed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_index": self.request_index,
            "sent_monotonic_s": self.sent_monotonic_s,
            "completed_monotonic_s": self.completed_monotonic_s,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RequestTiming":
        if not isinstance(value, Mapping):
            raise StageTimingContractError("request timing must be an object")
        _require_exact_fields(
            value,
            {
                "request_index",
                "sent_monotonic_s",
                "completed_monotonic_s",
            },
            label="request timing",
        )
        return cls(
            request_index=value["request_index"],
            sent_monotonic_s=value["sent_monotonic_s"],
            completed_monotonic_s=value["completed_monotonic_s"],
        )


def _request_timing(value: Any, *, index: int) -> RequestTiming:
    if isinstance(value, RequestTiming):
        if value.request_index != index:
            raise StageTimingContractError(
                "request timing indices disagree with cohort task order"
            )
        return value
    if isinstance(value, Mapping):
        try:
            sent = value["request_send_ts"]
            completed = value["last_token_ts"]
        except KeyError as exc:
            raise StageTimingContractError(
                "request timing mapping lacks send/completion timestamps"
            ) from exc
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        sent, completed = value
    else:
        raise StageTimingContractError(
            "request timing must be RequestTiming, mapping, or (send, completion)"
        )
    return RequestTiming(
        request_index=index,
        sent_monotonic_s=sent,
        completed_monotonic_s=completed,
    )


@dataclass(frozen=True)
class ProductionCohortTiming:
    """One of eight identical B-request production cohorts."""

    target_batch: int
    output_tokens: int
    group_index: int
    request_timings: tuple[RequestTiming, ...]
    endpoint_monotonic_s: float
    completion_spread_s: float

    def __post_init__(self) -> None:
        batch = _positive_int(self.target_batch, label="target_batch")
        output_tokens = _positive_int(self.output_tokens, label="output_tokens")
        group_index = _nonnegative_int(self.group_index, label="group_index")
        timings = tuple(self.request_timings)
        if (
            len(timings) != batch
            or any(not isinstance(item, RequestTiming) for item in timings)
            or tuple(item.request_index for item in timings) != tuple(range(batch))
        ):
            raise StageTimingContractError(
                "production cohort must freeze exactly B parent request timings"
            )
        endpoint = _finite_float(
            self.endpoint_monotonic_s,
            label="cohort.endpoint_monotonic_s",
        )
        if not _same_float(
            endpoint,
            max(item.completed_monotonic_s for item in timings),
        ):
            raise StageTimingContractError(
                "cohort endpoint must equal max parent response completion"
            )
        completion_spread = _finite_float(
            self.completion_spread_s,
            label="cohort.completion_spread_s",
        )
        expected_spread = endpoint - min(
            item.completed_monotonic_s for item in timings
        )
        if completion_spread < 0.0 or not _same_float(
            completion_spread, expected_spread
        ):
            raise StageTimingContractError(
                "cohort completion spread must equal max minus min completion"
            )
        object.__setattr__(self, "target_batch", batch)
        object.__setattr__(self, "output_tokens", output_tokens)
        object.__setattr__(self, "group_index", group_index)
        object.__setattr__(self, "request_timings", timings)
        object.__setattr__(self, "endpoint_monotonic_s", endpoint)
        object.__setattr__(self, "completion_spread_s", completion_spread)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COHORT_SCHEMA,
            "target_batch": self.target_batch,
            "output_tokens": self.output_tokens,
            "group_index": self.group_index,
            "request_timings": [
                item.to_dict() for item in self.request_timings
            ],
            "endpoint_monotonic_s": self.endpoint_monotonic_s,
            "completion_spread_s": self.completion_spread_s,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductionCohortTiming":
        if not isinstance(value, Mapping):
            raise StageTimingContractError("production cohort must be an object")
        _require_exact_fields(
            value,
            {
                "schema",
                "target_batch",
                "output_tokens",
                "group_index",
                "request_timings",
                "endpoint_monotonic_s",
                "completion_spread_s",
            },
            label="production cohort",
        )
        if value["schema"] != COHORT_SCHEMA:
            raise StageTimingContractError("unexpected production cohort kind")
        if not isinstance(value["request_timings"], list):
            raise StageTimingContractError(
                "production cohort request_timings must be a list"
            )
        return cls(
            target_batch=value["target_batch"],
            output_tokens=value["output_tokens"],
            group_index=value["group_index"],
            request_timings=tuple(
                RequestTiming.from_dict(item)
                for item in value["request_timings"]
            ),
            endpoint_monotonic_s=value["endpoint_monotonic_s"],
            completion_spread_s=value["completion_spread_s"],
        )


@dataclass(frozen=True)
class ProductionInterdeparture:
    """Eight queued same-shape cohorts and seven steady-state samples."""

    timing_run_id: str
    engine_batch_cap: int
    target_batch: int
    input_tokens: int
    output_tokens: int
    telemetry_enabled: bool
    enginecore_dispatch_fence: str
    enginecore_fence_ack_count: int
    observed_dispatch_counts: tuple[int, ...]
    cohorts: tuple[ProductionCohortTiming, ...]
    interdeparture_s: tuple[float, ...]
    median_service_s: float
    interval_mad_s: float
    relative_interval_mad: float
    clock: ClockDomainProof

    def __post_init__(self) -> None:
        if not isinstance(self.timing_run_id, str) or not self.timing_run_id:
            raise StageTimingContractError("timing_run_id is missing")
        cap = _positive_int(self.engine_batch_cap, label="engine_batch_cap")
        batch = _positive_int(self.target_batch, label="target_batch")
        input_tokens = _positive_int(self.input_tokens, label="input_tokens")
        output_tokens = _positive_int(self.output_tokens, label="output_tokens")
        if cap != batch:
            raise StageTimingContractError(
                "production timing requires engine_batch_cap == target_batch"
            )
        if self.telemetry_enabled is not False:
            raise StageTimingContractError(
                "production timing server must have telemetry disabled"
            )
        if self.enginecore_dispatch_fence != ENGINECORE_DISPATCH_FENCE:
            raise StageTimingContractError(
                "production timing lacks the EngineCore FIFO dispatch fence"
            )
        fence_acks = _positive_int(
            self.enginecore_fence_ack_count,
            label="enginecore_fence_ack_count",
        )
        if fence_acks != PRODUCTION_COHORT_COUNT:
            raise StageTimingContractError(
                "production timing requires one FIFO EngineCore fence ack per cohort"
            )
        if not isinstance(self.clock, ClockDomainProof):
            raise StageTimingContractError(
                "production interdeparture clock proof is missing"
            )

        dispatch_counts = tuple(
            _positive_int(value, label="observed_dispatch_count")
            for value in self.observed_dispatch_counts
        )
        expected_dispatch = tuple(
            batch * (index + 1) for index in range(PRODUCTION_COHORT_COUNT)
        )
        if dispatch_counts != expected_dispatch:
            raise StageTimingContractError(
                "paused FIFO dispatch must prove counts [B, 2B, ..., 8B]"
            )

        cohorts = tuple(self.cohorts)
        if (
            len(cohorts) != PRODUCTION_COHORT_COUNT
            or any(not isinstance(item, ProductionCohortTiming) for item in cohorts)
            or tuple(item.group_index for item in cohorts)
            != tuple(range(PRODUCTION_COHORT_COUNT))
            or any(
                item.target_batch != batch or item.output_tokens != output_tokens
                for item in cohorts
            )
        ):
            raise StageTimingContractError(
                "production timing requires eight indexed identical-shape cohorts"
            )
        for previous, current in zip(cohorts, cohorts[1:]):
            if max(
                item.sent_monotonic_s for item in previous.request_timings
            ) > min(item.sent_monotonic_s for item in current.request_timings):
                raise StageTimingContractError(
                    "cohort request groups were not dispatched in index order"
                )
        if max(
            timing.sent_monotonic_s
            for cohort in cohorts
            for timing in cohort.request_timings
        ) >= min(
            timing.completed_monotonic_s
            for cohort in cohorts
            for timing in cohort.request_timings
        ):
            raise StageTimingContractError(
                "all eight cohorts must be queued before production completion"
            )
        endpoints = tuple(item.endpoint_monotonic_s for item in cohorts)
        if any(right <= left for left, right in zip(endpoints, endpoints[1:])):
            raise StageTimingContractError(
                "cohort completion endpoints must be strictly increasing"
            )
        expected_interdepartures = tuple(
            right - left for left, right in zip(endpoints, endpoints[1:])
        )
        intervals = tuple(
            _finite_float(value, label="interdeparture_s", positive=True)
            for value in self.interdeparture_s
        )
        if (
            len(intervals) != PRODUCTION_INTERVAL_COUNT
            or any(
                not _same_float(observed, expected)
                for observed, expected in zip(
                    intervals,
                    expected_interdepartures,
                    strict=True,
                )
            )
        ):
            raise StageTimingContractError(
                "interdeparture samples must equal consecutive cohort endpoints"
            )
        service = _finite_float(
            self.median_service_s,
            label="median_service_s",
            positive=True,
        )
        if not _same_float(service, statistics.median(intervals)):
            raise StageTimingContractError(
                "phase service must equal median steady-state interdeparture"
            )
        mad = _finite_float(self.interval_mad_s, label="interval_mad_s")
        expected_mad = statistics.median(
            abs(value - service) for value in intervals
        )
        if mad < 0.0 or not _same_float(mad, expected_mad):
            raise StageTimingContractError(
                "interval MAD must be recomputable from interdepartures"
            )
        relative_mad = _finite_float(
            self.relative_interval_mad,
            label="relative_interval_mad",
        )
        if relative_mad < 0.0 or not _same_float(
            relative_mad, mad / service
        ):
            raise StageTimingContractError(
                "relative interval MAD must equal MAD divided by median service"
            )
        if relative_mad > MAX_RELATIVE_INTERVAL_MAD:
            raise StageTimingContractError(
                "relative interval MAD exceeds the 20% stability gate"
            )

        object.__setattr__(self, "engine_batch_cap", cap)
        object.__setattr__(self, "target_batch", batch)
        object.__setattr__(self, "input_tokens", input_tokens)
        object.__setattr__(self, "output_tokens", output_tokens)
        object.__setattr__(self, "enginecore_fence_ack_count", fence_acks)
        object.__setattr__(self, "observed_dispatch_counts", dispatch_counts)
        object.__setattr__(self, "cohorts", cohorts)
        object.__setattr__(self, "interdeparture_s", intervals)
        object.__setattr__(self, "median_service_s", service)
        object.__setattr__(self, "interval_mad_s", mad)
        object.__setattr__(self, "relative_interval_mad", relative_mad)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": INTERDEPARTURE_SCHEMA,
            "timing_run_id": self.timing_run_id,
            "engine_batch_cap": self.engine_batch_cap,
            "target_batch": self.target_batch,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "telemetry_enabled": self.telemetry_enabled,
            "enginecore_dispatch_fence": self.enginecore_dispatch_fence,
            "enginecore_fence_ack_count": self.enginecore_fence_ack_count,
            "observed_dispatch_counts": list(self.observed_dispatch_counts),
            "cohorts": [item.to_dict() for item in self.cohorts],
            "interdeparture_s": list(self.interdeparture_s),
            "median_service_s": self.median_service_s,
            "interval_mad_s": self.interval_mad_s,
            "relative_interval_mad": self.relative_interval_mad,
            "clock": self.clock.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProductionInterdeparture":
        if not isinstance(value, Mapping):
            raise StageTimingContractError(
                "production interdeparture must be an object"
            )
        _require_exact_fields(
            value,
            {
                "schema",
                "timing_run_id",
                "engine_batch_cap",
                "target_batch",
                "input_tokens",
                "output_tokens",
                "telemetry_enabled",
                "enginecore_dispatch_fence",
                "enginecore_fence_ack_count",
                "observed_dispatch_counts",
                "cohorts",
                "interdeparture_s",
                "median_service_s",
                "interval_mad_s",
                "relative_interval_mad",
                "clock",
            },
            label="production interdeparture",
        )
        if value["schema"] != INTERDEPARTURE_SCHEMA:
            raise StageTimingContractError(
                "unexpected production interdeparture kind"
            )
        for field in ("observed_dispatch_counts", "cohorts", "interdeparture_s"):
            if not isinstance(value[field], list):
                raise StageTimingContractError(
                    f"production interdeparture {field} must be a list"
                )
        return cls(
            timing_run_id=value["timing_run_id"],
            engine_batch_cap=value["engine_batch_cap"],
            target_batch=value["target_batch"],
            input_tokens=value["input_tokens"],
            output_tokens=value["output_tokens"],
            telemetry_enabled=value["telemetry_enabled"],
            enginecore_dispatch_fence=value["enginecore_dispatch_fence"],
            enginecore_fence_ack_count=value["enginecore_fence_ack_count"],
            observed_dispatch_counts=tuple(value["observed_dispatch_counts"]),
            cohorts=tuple(
                ProductionCohortTiming.from_dict(item)
                for item in value["cohorts"]
            ),
            interdeparture_s=tuple(value["interdeparture_s"]),
            median_service_s=value["median_service_s"],
            interval_mad_s=value["interval_mad_s"],
            relative_interval_mad=value["relative_interval_mad"],
            clock=ClockDomainProof.from_dict(value["clock"]),
        )


def build_production_cohort_service_evidence(
    *,
    timing_run_id: str,
    engine_batch_cap: int,
    target_batch: int,
    input_tokens: int,
    output_tokens: int,
    cohort_request_timings: Sequence[Sequence[Any]],
    observed_dispatch_counts: Sequence[int],
    enginecore_fence_ack_count: int,
    clock: ClockDomainProof,
) -> ProductionInterdeparture:
    """Freeze one out=1 or out=N eight-cohort production timing phase."""

    if len(cohort_request_timings) != PRODUCTION_COHORT_COUNT:
        raise StageTimingContractError(
            "production timing requires exactly eight request cohorts"
        )
    cohorts: list[ProductionCohortTiming] = []
    for group_index, raw_timings in enumerate(cohort_request_timings):
        timings = tuple(
            _request_timing(value, index=index)
            for index, value in enumerate(raw_timings)
        )
        if not timings:
            raise StageTimingContractError("production cohort is empty")
        cohorts.append(
            ProductionCohortTiming(
                target_batch=target_batch,
                output_tokens=output_tokens,
                group_index=group_index,
                request_timings=timings,
                endpoint_monotonic_s=max(
                    item.completed_monotonic_s for item in timings
                ),
                completion_spread_s=(
                    max(item.completed_monotonic_s for item in timings)
                    - min(item.completed_monotonic_s for item in timings)
                ),
            )
        )
    endpoints = tuple(item.endpoint_monotonic_s for item in cohorts)
    intervals = tuple(
        right - left for left, right in zip(endpoints, endpoints[1:])
    )
    median_service = statistics.median(intervals)
    interval_mad = statistics.median(
        abs(value - median_service) for value in intervals
    )
    return ProductionInterdeparture(
        timing_run_id=timing_run_id,
        engine_batch_cap=engine_batch_cap,
        target_batch=target_batch,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        telemetry_enabled=False,
        enginecore_dispatch_fence=ENGINECORE_DISPATCH_FENCE,
        enginecore_fence_ack_count=enginecore_fence_ack_count,
        observed_dispatch_counts=tuple(observed_dispatch_counts),
        cohorts=tuple(cohorts),
        interdeparture_s=intervals,
        median_service_s=median_service,
        interval_mad_s=interval_mad,
        relative_interval_mad=interval_mad / median_service,
        clock=clock,
    )


@dataclass(frozen=True)
class CollocatedEngineTimingEvidence:
    """Full collocated-engine timing plus optional out=1 raw diagnostics.

    ``full_engine_interdeparture`` is the sole publishable service boundary.
    ``out_one_diagnostic`` is deliberately opaque raw interdeparture evidence:
    this contract never computes or serializes a P/D service split.
    """

    target_batch: int
    input_tokens: int
    output_tokens: int
    full_engine_interdeparture: ProductionInterdeparture
    out_one_diagnostic: ProductionInterdeparture | None = None

    def __post_init__(self) -> None:
        batch = _positive_int(self.target_batch, label="target_batch")
        input_tokens = _positive_int(self.input_tokens, label="input_tokens")
        output_tokens = _positive_int(self.output_tokens, label="output_tokens")
        if output_tokens < 2:
            raise StageTimingContractError(
                "full collocated-engine timing must generate at least two tokens"
            )
        full = self.full_engine_interdeparture
        if not isinstance(full, ProductionInterdeparture):
            raise StageTimingContractError(
                "full collocated-engine interdeparture evidence is required"
            )
        if (
            full.engine_batch_cap != batch
            or full.target_batch != batch
            or full.input_tokens != input_tokens
            or full.output_tokens != output_tokens
        ):
            raise StageTimingContractError(
                "full collocated-engine evidence differs from its batch/shape"
            )

        diagnostic = self.out_one_diagnostic
        if diagnostic is not None:
            if not isinstance(diagnostic, ProductionInterdeparture):
                raise StageTimingContractError(
                    "out=1 diagnostic must be production interdeparture evidence"
                )
            if (
                diagnostic.timing_run_id != full.timing_run_id
                or diagnostic.clock != full.clock
                or diagnostic.engine_batch_cap != batch
                or diagnostic.target_batch != batch
                or diagnostic.input_tokens != input_tokens
                or diagnostic.output_tokens != 1
            ):
                raise StageTimingContractError(
                    "out=1 diagnostic must share the full engine run, clock and shape"
                )
            diagnostic_start = min(
                timing.sent_monotonic_s
                for cohort in diagnostic.cohorts
                for timing in cohort.request_timings
            )
            diagnostic_end = max(
                timing.completed_monotonic_s
                for cohort in diagnostic.cohorts
                for timing in cohort.request_timings
            )
            full_start = min(
                timing.sent_monotonic_s
                for cohort in full.cohorts
                for timing in cohort.request_timings
            )
            full_end = max(
                timing.completed_monotonic_s
                for cohort in full.cohorts
                for timing in cohort.request_timings
            )
            if not (diagnostic_end < full_start or full_end < diagnostic_start):
                raise StageTimingContractError(
                    "out=1 diagnostic and full engine timing must not overlap"
                )

        object.__setattr__(self, "target_batch", batch)
        object.__setattr__(self, "input_tokens", input_tokens)
        object.__setattr__(self, "output_tokens", output_tokens)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TIMING_SCHEMA,
            "target_batch": self.target_batch,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "full_engine_interdeparture": (
                self.full_engine_interdeparture.to_dict()
            ),
            "out_one_diagnostic": (
                None
                if self.out_one_diagnostic is None
                else self.out_one_diagnostic.to_dict()
            ),
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
    ) -> "CollocatedEngineTimingEvidence":
        if not isinstance(value, Mapping):
            raise StageTimingContractError(
                "collocated engine timing evidence must be an object"
            )
        _require_exact_fields(
            value,
            {
                "schema",
                "target_batch",
                "input_tokens",
                "output_tokens",
                "full_engine_interdeparture",
                "out_one_diagnostic",
            },
            label="collocated engine timing evidence",
        )
        if value["schema"] != TIMING_SCHEMA:
            raise StageTimingContractError(
                "unexpected collocated engine timing kind"
            )
        diagnostic = value["out_one_diagnostic"]
        return cls(
            target_batch=value["target_batch"],
            input_tokens=value["input_tokens"],
            output_tokens=value["output_tokens"],
            full_engine_interdeparture=ProductionInterdeparture.from_dict(
                value["full_engine_interdeparture"]
            ),
            out_one_diagnostic=(
                None
                if diagnostic is None
                else ProductionInterdeparture.from_dict(diagnostic)
            ),
        )


def build_collocated_engine_timing_evidence(
    full_engine_interdeparture: ProductionInterdeparture,
    *,
    out_one_diagnostic: ProductionInterdeparture | None = None,
) -> CollocatedEngineTimingEvidence:
    """Build the full-engine boundary without deriving a phase service."""

    if not isinstance(full_engine_interdeparture, ProductionInterdeparture):
        raise StageTimingContractError(
            "full collocated-engine interdeparture evidence is required"
        )
    return CollocatedEngineTimingEvidence(
        target_batch=full_engine_interdeparture.target_batch,
        input_tokens=full_engine_interdeparture.input_tokens,
        output_tokens=full_engine_interdeparture.output_tokens,
        full_engine_interdeparture=full_engine_interdeparture,
        out_one_diagnostic=out_one_diagnostic,
    )


__all__ = [
    "CLOCK_DOMAIN",
    "ENGINECORE_DISPATCH_FENCE",
    "ClockDomainProof",
    "CollocatedEngineTimingEvidence",
    "ProductionCohortTiming",
    "ProductionInterdeparture",
    "RequestTiming",
    "StageTimingContractError",
    "build_collocated_engine_timing_evidence",
    "build_production_cohort_service_evidence",
    "capture_clock_domain_proof",
]
