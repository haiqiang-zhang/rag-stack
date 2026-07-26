"""Strict NIXL transfer evidence from two vLLM Prometheus snapshots.

This module deliberately produces *physical evidence*, not a cost-model fit.
It reads the cumulative NIXL histograms/counters exposed by vLLM, validates a
single before/after cohort, and reports transferred bytes and elapsed NIXL
time.  No correction, residual, deployment factor, or other tunable value is
derived here.

The parser is intentionally small and dependency-free.  It accepts vLLM's
normal ``vllm:nixl_*`` exposition, the backslash-escaped spelling sometimes
seen after transporting metric text through another format, and the
underscore spelling produced by Prometheus name-escaping gateways.  All
spellings are canonicalized before duplicate and delta validation.  Label
sets are retained so that a counter reset, disappeared engine, or newly
appeared engine cannot be hidden by aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping


NIXL_PHYSICAL_EVIDENCE_SCHEMA = "rag_stack.nixl_physical_evidence"
NIXL_PHYSICAL_EVIDENCE_SCOPE = (
    "direct_prometheus_delta_physical_only_no_fit_no_correction"
)

XFER_TIME_SUM = "vllm:nixl_xfer_time_seconds_sum"
XFER_TIME_COUNT = "vllm:nixl_xfer_time_seconds_count"
POST_TIME_SUM = "vllm:nixl_post_time_seconds_sum"
POST_TIME_COUNT = "vllm:nixl_post_time_seconds_count"
BYTES_SUM = "vllm:nixl_bytes_transferred_sum"
BYTES_COUNT = "vllm:nixl_bytes_transferred_count"
FAILED_TRANSFERS = "vllm:nixl_num_failed_transfers_total"
FAILED_NOTIFICATIONS = "vllm:nixl_num_failed_notifications_total"

_REQUIRED_METRICS = (
    XFER_TIME_SUM,
    XFER_TIME_COUNT,
    POST_TIME_SUM,
    POST_TIME_COUNT,
    BYTES_SUM,
    BYTES_COUNT,
    FAILED_TRANSFERS,
    FAILED_NOTIFICATIONS,
)
_COUNT_METRICS = frozenset(
    {
        XFER_TIME_COUNT,
        POST_TIME_COUNT,
        BYTES_COUNT,
        FAILED_TRANSFERS,
        FAILED_NOTIFICATIONS,
    }
)
_COUNTER_BASE_NAMES = {
    "vllm:nixl_num_failed_transfers": FAILED_TRANSFERS,
    "vllm:nixl_num_failed_notifications": FAILED_NOTIFICATIONS,
}

_SAMPLE_RE = re.compile(
    r"^(?P<name>[^\s{]+)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[^\s]+)(?:\s+(?P<timestamp>[^\s]+))?\s*$"
)
_LABEL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

LabelSet = tuple[tuple[str, str], ...]


class NixlPhysicalEvidenceError(ValueError):
    """The NIXL Prometheus evidence is incomplete or inconsistent."""


@dataclass(frozen=True, order=True)
class NixlMetricSample:
    """One canonical Prometheus series sample."""

    metric: str
    labels: LabelSet
    value: float

    def __post_init__(self) -> None:
        if self.metric not in _REQUIRED_METRICS:
            raise NixlPhysicalEvidenceError(
                f"unsupported NIXL metric {self.metric!r}"
            )
        if tuple(sorted(self.labels)) != self.labels:
            raise NixlPhysicalEvidenceError("metric labels are not canonical")
        if len({name for name, _ in self.labels}) != len(self.labels):
            raise NixlPhysicalEvidenceError("metric labels contain duplicates")
        value = float(self.value)
        if not math.isfinite(value) or value < 0.0:
            raise NixlPhysicalEvidenceError(
                f"{self.metric} must be finite and non-negative"
            )
        if self.metric in _COUNT_METRICS and not value.is_integer():
            raise NixlPhysicalEvidenceError(
                f"{self.metric} must be an integer counter"
            )
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class NixlPrometheusSnapshot:
    """The exact labeled NIXL series needed for physical delta evidence."""

    samples: tuple[NixlMetricSample, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.samples))
        if ordered != self.samples:
            raise NixlPhysicalEvidenceError(
                "NIXL snapshot samples are not canonical"
            )
        keys = [(sample.metric, sample.labels) for sample in self.samples]
        if len(set(keys)) != len(keys):
            raise NixlPhysicalEvidenceError(
                "NIXL snapshot contains duplicate metric series"
            )
        present = {sample.metric for sample in self.samples}
        missing = set(_REQUIRED_METRICS) - present
        if missing:
            raise NixlPhysicalEvidenceError(
                f"NIXL snapshot is missing metrics: {sorted(missing)}"
            )
        for sum_metric, count_metric in (
            (XFER_TIME_SUM, XFER_TIME_COUNT),
            (POST_TIME_SUM, POST_TIME_COUNT),
            (BYTES_SUM, BYTES_COUNT),
        ):
            if set(self.metric_values(sum_metric)) != set(
                self.metric_values(count_metric)
            ):
                raise NixlPhysicalEvidenceError(
                    f"{sum_metric} and {count_metric} label sets differ"
                )

    def metric_values(self, metric: str) -> dict[LabelSet, float]:
        """Return labeled values for one canonical metric."""

        if metric not in _REQUIRED_METRICS:
            raise NixlPhysicalEvidenceError(
                f"unsupported NIXL metric {metric!r}"
            )
        return {
            sample.labels: sample.value
            for sample in self.samples
            if sample.metric == metric
        }


@dataclass(frozen=True)
class NixlPhysicalEvidence:
    """Validated cohort-level NIXL physical measurements.

    ``effective_bytes_per_s`` is ``total_bytes / total_xfer_time_s``.  Post
    time remains a separately reported physical observation and is not folded
    into a fitted service correction.
    """

    expected_transfer_count: int
    transfer_count: int
    total_bytes: int
    total_xfer_time_s: float
    total_post_time_s: float
    effective_bytes_per_s: float
    failed_transfer_delta: int = 0
    failed_notification_delta: int = 0

    def __post_init__(self) -> None:
        for name in (
            "expected_transfer_count",
            "transfer_count",
            "total_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise NixlPhysicalEvidenceError(
                    f"{name} must be a positive integer"
                )
        if self.transfer_count != self.expected_transfer_count:
            raise NixlPhysicalEvidenceError(
                "NIXL transfer count does not match the requested cohort"
            )
        for name in (
            "total_xfer_time_s",
            "total_post_time_s",
            "effective_bytes_per_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise NixlPhysicalEvidenceError(
                    f"{name} must be finite and positive"
                )
            object.__setattr__(self, name, float(value))
        if self.failed_transfer_delta != 0:
            raise NixlPhysicalEvidenceError(
                "NIXL failed-transfer counter changed during the cohort"
            )
        if self.failed_notification_delta != 0:
            raise NixlPhysicalEvidenceError(
                "NIXL failed-notification counter changed during the cohort"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize only immutable physical observations, never CM knobs."""

        return {
            "schema": NIXL_PHYSICAL_EVIDENCE_SCHEMA,
            "evidence_scope": NIXL_PHYSICAL_EVIDENCE_SCOPE,
            "expected_transfer_count": self.expected_transfer_count,
            "transfer_count": self.transfer_count,
            "total_bytes": self.total_bytes,
            "total_xfer_time_s": self.total_xfer_time_s,
            "total_post_time_s": self.total_post_time_s,
            "effective_bytes_per_s": self.effective_bytes_per_s,
            "failed_transfer_delta": self.failed_transfer_delta,
            "failed_notification_delta": self.failed_notification_delta,
        }


def _canonical_metric_name(raw_name: str) -> str | None:
    # ``\:`` can appear when raw exposition is embedded in an escaped text
    # field.  Some gateways instead use Prometheus' underscore name-escaping.
    name = raw_name.replace(r"\:", ":")
    if name.startswith("vllm_nixl_"):
        name = "vllm:nixl_" + name[len("vllm_nixl_") :]
    name = _COUNTER_BASE_NAMES.get(name, name)
    if name in _REQUIRED_METRICS:
        return name
    return None


def _unescape_label_value(raw: str) -> str:
    result: list[str] = []
    position = 0
    while position < len(raw):
        char = raw[position]
        if char != "\\":
            result.append(char)
            position += 1
            continue
        position += 1
        if position >= len(raw):
            raise NixlPhysicalEvidenceError(
                "Prometheus label ends with an incomplete escape"
            )
        escaped = raw[position]
        replacements = {"\\": "\\", '"': '"', "n": "\n"}
        if escaped not in replacements:
            raise NixlPhysicalEvidenceError(
                f"unsupported Prometheus label escape \\{escaped}"
            )
        result.append(replacements[escaped])
        position += 1
    return "".join(result)


def _parse_labels(raw: str | None) -> LabelSet:
    if raw is None or not raw.strip():
        return ()
    labels: dict[str, str] = {}
    position = 0
    while position < len(raw):
        while position < len(raw) and raw[position].isspace():
            position += 1
        match = _LABEL_NAME_RE.match(raw, position)
        if match is None:
            raise NixlPhysicalEvidenceError(
                f"invalid Prometheus label list near {raw[position:]!r}"
            )
        name = match.group(0)
        position = match.end()
        while position < len(raw) and raw[position].isspace():
            position += 1
        if position >= len(raw) or raw[position] != "=":
            raise NixlPhysicalEvidenceError(
                f"Prometheus label {name!r} lacks '='"
            )
        position += 1
        while position < len(raw) and raw[position].isspace():
            position += 1
        if position >= len(raw) or raw[position] != '"':
            raise NixlPhysicalEvidenceError(
                f"Prometheus label {name!r} is not quoted"
            )
        position += 1
        value_start = position
        escaped = False
        while position < len(raw):
            char = raw[position]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                break
            position += 1
        if position >= len(raw) or raw[position] != '"':
            raise NixlPhysicalEvidenceError(
                f"Prometheus label {name!r} has no closing quote"
            )
        value = _unescape_label_value(raw[value_start:position])
        position += 1
        if name in labels:
            raise NixlPhysicalEvidenceError(
                f"duplicate Prometheus label {name!r}"
            )
        labels[name] = value
        while position < len(raw) and raw[position].isspace():
            position += 1
        if position == len(raw):
            break
        if raw[position] != ",":
            raise NixlPhysicalEvidenceError(
                f"invalid Prometheus label separator near {raw[position:]!r}"
            )
        position += 1
        if position == len(raw):
            raise NixlPhysicalEvidenceError(
                "Prometheus label list has a trailing comma"
            )
    return tuple(sorted(labels.items()))


def parse_nixl_prometheus_snapshot(payload: str) -> NixlPrometheusSnapshot:
    """Parse the required labeled NIXL series from one ``/metrics`` body."""

    if not isinstance(payload, str):
        raise NixlPhysicalEvidenceError("Prometheus payload must be text")
    samples: list[NixlMetricSample] = []
    seen: set[tuple[str, LabelSet]] = set()
    for line_number, source_line in enumerate(payload.splitlines(), start=1):
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        raw_name = re.split(r"[\s{]", line, maxsplit=1)[0]
        metric = _canonical_metric_name(raw_name)
        if metric is None:
            continue
        match = _SAMPLE_RE.fullmatch(line)
        if match is None:
            raise NixlPhysicalEvidenceError(
                f"malformed NIXL Prometheus sample on line {line_number}"
            )
        if _canonical_metric_name(match.group("name")) != metric:
            raise NixlPhysicalEvidenceError(
                f"ambiguous NIXL metric name on line {line_number}"
            )
        labels = _parse_labels(match.group("labels"))
        try:
            value = float(match.group("value"))
        except ValueError as exc:
            raise NixlPhysicalEvidenceError(
                f"invalid NIXL metric value on line {line_number}"
            ) from exc
        key = (metric, labels)
        if key in seen:
            raise NixlPhysicalEvidenceError(
                "duplicate NIXL series after Prometheus name canonicalization: "
                f"{metric} labels={dict(labels)!r}"
            )
        seen.add(key)
        samples.append(NixlMetricSample(metric, labels, value))
    return NixlPrometheusSnapshot(tuple(sorted(samples)))


def _strict_metric_delta(
    before: NixlPrometheusSnapshot,
    after: NixlPrometheusSnapshot,
    metric: str,
) -> dict[LabelSet, float]:
    before_values = before.metric_values(metric)
    after_values = after.metric_values(metric)
    if set(before_values) != set(after_values):
        raise NixlPhysicalEvidenceError(
            f"{metric} label set changed between snapshots"
        )
    result: dict[LabelSet, float] = {}
    for labels, before_value in before_values.items():
        delta = after_values[labels] - before_value
        if not math.isfinite(delta) or delta < 0.0:
            raise NixlPhysicalEvidenceError(
                f"{metric} decreased or became non-finite for labels "
                f"{dict(labels)!r}"
            )
        if metric in _COUNT_METRICS and not delta.is_integer():
            raise NixlPhysicalEvidenceError(
                f"{metric} has a non-integer delta for labels {dict(labels)!r}"
            )
        result[labels] = delta
    return result


def _validate_histogram_delta(
    *,
    label: str,
    counts: Mapping[LabelSet, float],
    sums: Mapping[LabelSet, float],
) -> None:
    if set(counts) != set(sums):
        raise NixlPhysicalEvidenceError(
            f"{label} histogram sum/count labels differ"
        )
    for labels, count in counts.items():
        total = sums[labels]
        if count == 0.0 and total != 0.0:
            raise NixlPhysicalEvidenceError(
                f"{label} sum changed without observations for labels "
                f"{dict(labels)!r}"
            )
        if count > 0.0 and total <= 0.0:
            raise NixlPhysicalEvidenceError(
                f"{label} observations have no positive sum for labels "
                f"{dict(labels)!r}"
            )


def derive_nixl_physical_evidence(
    before: NixlPrometheusSnapshot,
    after: NixlPrometheusSnapshot,
    *,
    expected_transfer_count: int,
) -> NixlPhysicalEvidence:
    """Validate one direct cohort and derive non-fitted physical evidence."""

    if (
        isinstance(expected_transfer_count, bool)
        or not isinstance(expected_transfer_count, int)
        or expected_transfer_count <= 0
    ):
        raise NixlPhysicalEvidenceError(
            "expected_transfer_count must be a positive integer"
        )

    xfer_counts = _strict_metric_delta(before, after, XFER_TIME_COUNT)
    post_counts = _strict_metric_delta(before, after, POST_TIME_COUNT)
    byte_counts = _strict_metric_delta(before, after, BYTES_COUNT)
    if xfer_counts != post_counts or xfer_counts != byte_counts:
        raise NixlPhysicalEvidenceError(
            "NIXL histogram observation counts differ by metric or label"
        )

    xfer_sums = _strict_metric_delta(before, after, XFER_TIME_SUM)
    post_sums = _strict_metric_delta(before, after, POST_TIME_SUM)
    byte_sums = _strict_metric_delta(before, after, BYTES_SUM)
    _validate_histogram_delta(
        label="NIXL transfer time", counts=xfer_counts, sums=xfer_sums
    )
    _validate_histogram_delta(
        label="NIXL post time", counts=post_counts, sums=post_sums
    )
    _validate_histogram_delta(
        label="NIXL transferred bytes", counts=byte_counts, sums=byte_sums
    )

    transfer_count_float = sum(xfer_counts.values())
    if (
        not transfer_count_float.is_integer()
        or int(transfer_count_float) != expected_transfer_count
    ):
        raise NixlPhysicalEvidenceError(
            "NIXL histogram count delta does not match expected transfers: "
            f"expected={expected_transfer_count}, "
            f"observed={transfer_count_float}"
        )

    failed_transfer_deltas = _strict_metric_delta(
        before, after, FAILED_TRANSFERS
    )
    failed_notification_deltas = _strict_metric_delta(
        before, after, FAILED_NOTIFICATIONS
    )
    failed_transfer_delta = int(sum(failed_transfer_deltas.values()))
    failed_notification_delta = int(sum(failed_notification_deltas.values()))
    if failed_transfer_delta != 0 or failed_notification_delta != 0:
        raise NixlPhysicalEvidenceError(
            "NIXL failure counters changed during the direct cohort: "
            f"failed_transfers={failed_transfer_delta}, "
            f"failed_notifications={failed_notification_delta}"
        )

    total_bytes_float = sum(byte_sums.values())
    if not total_bytes_float.is_integer() or total_bytes_float <= 0.0:
        raise NixlPhysicalEvidenceError(
            "NIXL transferred-byte delta must be a positive integer"
        )
    total_xfer_time_s = sum(xfer_sums.values())
    total_post_time_s = sum(post_sums.values())
    if (
        not math.isfinite(total_xfer_time_s)
        or total_xfer_time_s <= 0.0
        or not math.isfinite(total_post_time_s)
        or total_post_time_s <= 0.0
    ):
        raise NixlPhysicalEvidenceError(
            "NIXL transfer/post time deltas must be finite and positive"
        )
    total_bytes = int(total_bytes_float)
    effective_bytes_per_s = total_bytes / total_xfer_time_s
    return NixlPhysicalEvidence(
        expected_transfer_count=expected_transfer_count,
        transfer_count=int(transfer_count_float),
        total_bytes=total_bytes,
        total_xfer_time_s=total_xfer_time_s,
        total_post_time_s=total_post_time_s,
        effective_bytes_per_s=effective_bytes_per_s,
        failed_transfer_delta=failed_transfer_delta,
        failed_notification_delta=failed_notification_delta,
    )


def derive_nixl_physical_evidence_from_text(
    before_payload: str,
    after_payload: str,
    *,
    expected_transfer_count: int,
) -> NixlPhysicalEvidence:
    """Convenience wrapper for two raw vLLM ``/metrics`` responses."""

    return derive_nixl_physical_evidence(
        parse_nixl_prometheus_snapshot(before_payload),
        parse_nixl_prometheus_snapshot(after_payload),
        expected_transfer_count=expected_transfer_count,
    )
