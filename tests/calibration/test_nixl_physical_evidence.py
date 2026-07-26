from __future__ import annotations

import pytest

from rag_stack.cost_model.llm_sim.calibration.main_llm.nixl_physical_evidence import (
    NIXL_PHYSICAL_EVIDENCE_SCOPE,
    NixlPhysicalEvidenceError,
    derive_nixl_physical_evidence_from_text,
    parse_nixl_prometheus_snapshot,
)


def _payload(
    *,
    prefix: str = "vllm:nixl_",
    xfer_count: tuple[int, int] = (10, 20),
    xfer_sum: tuple[float, float] = (1.0, 2.0),
    post_count: tuple[int, int] | None = None,
    post_sum: tuple[float, float] = (0.1, 0.2),
    byte_count: tuple[int, int] | None = None,
    byte_sum: tuple[int, int] = (1000, 2000),
    failed_transfers: tuple[int, int] = (0, 0),
    failed_notifications: tuple[int, int] = (0, 0),
    reverse_labels: bool = False,
) -> str:
    post_count = xfer_count if post_count is None else post_count
    byte_count = xfer_count if byte_count is None else byte_count

    def labels(engine: int) -> str:
        if reverse_labels:
            return f'engine="{engine}",model_name="Qwen\\\"2.5\\\\pd"'
        return f'model_name="Qwen\\\"2.5\\\\pd",engine="{engine}"'

    lines = ["# arbitrary non-NIXL metrics remain ignorable", "vllm:uptime 1"]
    for suffix, values in (
        ("xfer_time_seconds_sum", xfer_sum),
        ("xfer_time_seconds_count", xfer_count),
        ("post_time_seconds_sum", post_sum),
        ("post_time_seconds_count", post_count),
        ("bytes_transferred_sum", byte_sum),
        ("bytes_transferred_count", byte_count),
        ("num_failed_transfers_total", failed_transfers),
        ("num_failed_notifications_total", failed_notifications),
    ):
        for engine, value in enumerate(values):
            lines.append(f"{prefix}{suffix}{{{labels(engine)}}} {value}")
    return "\n".join(lines) + "\n"


def test_labeled_counter_deltas_produce_physical_evidence_only():
    before = _payload()
    # Three transfers: one on engine 0 and two on engine 1.  Labels are emitted
    # in a different order to prove series identity is label-order invariant.
    after = _payload(
        xfer_count=(11, 22),
        xfer_sum=(1.25, 2.75),
        post_sum=(0.12, 0.28),
        byte_sum=(5096, 10192),
        reverse_labels=True,
    )

    evidence = derive_nixl_physical_evidence_from_text(
        before, after, expected_transfer_count=3
    )

    assert evidence.transfer_count == 3
    assert evidence.total_bytes == 12_288
    assert evidence.total_xfer_time_s == pytest.approx(1.0)
    assert evidence.total_post_time_s == pytest.approx(0.1)
    assert evidence.effective_bytes_per_s == pytest.approx(12_288.0)
    payload = evidence.to_dict()
    assert payload["evidence_scope"] == NIXL_PHYSICAL_EVIDENCE_SCOPE
    assert payload["failed_transfer_delta"] == 0
    assert payload["failed_notification_delta"] == 0
    assert not any(
        token in key
        for key in payload
        for token in ("correction", "residual", "coefficient", "factor")
    )


@pytest.mark.parametrize("prefix", ["vllm_nixl_", r"vllm\:nixl_"])
def test_prometheus_colon_escape_spellings_are_canonicalized(prefix: str):
    before = _payload(prefix=prefix)
    after = _payload(
        prefix=prefix,
        xfer_count=(11, 20),
        xfer_sum=(1.5, 2.0),
        post_sum=(0.15, 0.2),
        byte_sum=(3048, 2000),
    )

    evidence = derive_nixl_physical_evidence_from_text(
        before, after, expected_transfer_count=1
    )
    assert evidence.total_bytes == 2048
    assert evidence.effective_bytes_per_s == pytest.approx(4096.0)


def test_counter_base_name_exporter_variant_is_supported():
    before = _payload().replace(
        "num_failed_transfers_total", "num_failed_transfers"
    ).replace("num_failed_notifications_total", "num_failed_notifications")
    after = _payload(
        xfer_count=(11, 20),
        xfer_sum=(1.5, 2.0),
        post_sum=(0.15, 0.2),
        byte_sum=(3048, 2000),
    ).replace("num_failed_transfers_total", "num_failed_transfers").replace(
        "num_failed_notifications_total", "num_failed_notifications"
    )

    evidence = derive_nixl_physical_evidence_from_text(
        before, after, expected_transfer_count=1
    )
    assert evidence.total_bytes == 2048


@pytest.mark.parametrize(
    "after,match",
    [
        (
            _payload(
                xfer_count=(12, 20),
                xfer_sum=(1.5, 2.0),
                post_sum=(0.15, 0.2),
                byte_sum=(3048, 2000),
            ),
            "does not match expected transfers",
        ),
        (
            _payload(
                xfer_count=(11, 20),
                xfer_sum=(1.5, 2.0),
                post_count=(12, 20),
                post_sum=(0.15, 0.2),
                byte_sum=(3048, 2000),
            ),
            "observation counts differ",
        ),
        (
            _payload(
                xfer_count=(11, 20),
                xfer_sum=(1.5, 2.0),
                post_sum=(0.15, 0.2),
                byte_sum=(3048, 2000),
                failed_transfers=(1, 0),
            ),
            "failure counters changed",
        ),
        (
            _payload(
                xfer_count=(11, 20),
                xfer_sum=(1.5, 2.0),
                post_sum=(0.15, 0.2),
                byte_sum=(3048, 2000),
                failed_notifications=(0, 1),
            ),
            "failure counters changed",
        ),
    ],
)
def test_incomplete_or_failed_cohort_fails_closed(after: str, match: str):
    with pytest.raises(NixlPhysicalEvidenceError, match=match):
        derive_nixl_physical_evidence_from_text(
            _payload(), after, expected_transfer_count=1
        )


def test_counter_reset_fails_closed():
    before = _payload()
    after = _payload(
        xfer_count=(9, 22),
        xfer_sum=(0.9, 2.75),
        post_sum=(0.09, 0.28),
        byte_sum=(900, 10192),
    )
    with pytest.raises(NixlPhysicalEvidenceError, match="decreased"):
        derive_nixl_physical_evidence_from_text(
            before, after, expected_transfer_count=1
        )


def test_label_set_change_fails_closed():
    before = _payload()
    after = _payload(
        xfer_count=(11, 20),
        xfer_sum=(1.5, 2.0),
        post_sum=(0.15, 0.2),
        byte_sum=(3048, 2000),
    ).replace('engine="1"', 'engine="9"')
    with pytest.raises(NixlPhysicalEvidenceError, match="label set changed"):
        derive_nixl_physical_evidence_from_text(
            before, after, expected_transfer_count=1
        )


@pytest.mark.parametrize("bad_value", ["NaN", "+Inf", "-1"])
def test_nonfinite_or_negative_metric_fails_closed(bad_value: str):
    lines = _payload().splitlines()
    sample_index = next(
        index
        for index, line in enumerate(lines)
        if line.startswith("vllm:nixl_xfer_time_seconds_sum")
    )
    lines[sample_index] = lines[sample_index].rsplit(" ", 1)[0] + f" {bad_value}"
    payload = "\n".join(lines) + "\n"
    with pytest.raises(NixlPhysicalEvidenceError):
        parse_nixl_prometheus_snapshot(payload)


def test_alias_duplicate_after_canonicalization_fails_closed():
    payload = _payload()
    duplicate = next(
        line
        for line in payload.splitlines()
        if line.startswith("vllm:nixl_xfer_time_seconds_sum")
    ).replace("vllm:nixl_", "vllm_nixl_")
    with pytest.raises(NixlPhysicalEvidenceError, match="duplicate NIXL series"):
        parse_nixl_prometheus_snapshot(payload + duplicate + "\n")


def test_positive_time_and_byte_deltas_are_required():
    after = _payload(xfer_count=(11, 20))
    with pytest.raises(NixlPhysicalEvidenceError, match="no positive sum"):
        derive_nixl_physical_evidence_from_text(
            _payload(), after, expected_transfer_count=1
        )
