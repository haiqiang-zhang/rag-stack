import math

import pandas as pd
import pytest

from rag_stack.cost_model.selection import apply_selection
from rag_stack.types import PerformanceSelection


def test_dynamic_selection_ignores_screened_and_infeasible_rows():
    frame = pd.DataFrame([
        {
            "id": "screened",
            "qps": 1000.0,
            "latency_s": 0.001,
            "simulator_status": "screened_out",
        },
        {
            "id": "infeasible",
            "qps": 2000.0,
            "latency_s": 0.0005,
            "simulator_status": "infeasible",
        },
        {
            "id": "des",
            "qps": 20.0,
            "latency_s": 1.0,
            "simulator_status": "ok",
        },
    ])

    winner, score = apply_selection(
        frame,
        PerformanceSelection(selection="max_throughput"),
    )

    assert winner["id"] == "des"
    assert score == 20.0


def test_weighted_selection_does_not_normalize_infinite_screened_latency():
    frame = pd.DataFrame([
        {
            "id": "screened",
            "qps": 0.0,
            "latency_s": math.inf,
            "simulator_status": "screened_out",
        },
        {
            "id": "des-a",
            "qps": 10.0,
            "latency_s": 2.0,
            "simulator_status": "ok",
        },
        {
            "id": "des-b",
            "qps": 20.0,
            "latency_s": 1.0,
            "simulator_status": "ok",
        },
    ])

    winner, score = apply_selection(
        frame,
        PerformanceSelection(
            selection="weighted",
            weights={"throughput_qps": 1.0, "latency_s": 1.0},
        ),
    )

    assert winner["id"] == "des-b"
    assert math.isfinite(score)


def test_dynamic_selection_fails_when_des_has_no_feasible_row():
    frame = pd.DataFrame({
        "qps": [0.0],
        "latency_s": [math.inf],
        "simulator_status": ["infeasible"],
    })

    with pytest.raises(RuntimeError, match="no closed-loop simulator-feasible"):
        apply_selection(frame, None)
