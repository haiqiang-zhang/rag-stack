"""Shared result types for the cost model subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Per-operator result
# ---------------------------------------------------------------------------

@dataclass
class OperatorBreakdown:
    """Cost result for a single pipeline operator."""
    name: str
    compute_time_s: float = 0.0
    memory_time_s: float = 0.0
    time_s: float = 0.0            # max(compute, memory)
    bottleneck: str = ""            # "compute" or "memory"


# ---------------------------------------------------------------------------
# Per-module result
# ---------------------------------------------------------------------------

@dataclass
class CostModelResult:
    """Aggregate result returned by a cost model module."""
    latency_s: float = 0.0
    throughput_qps: float = 0.0
    operators: list[OperatorBreakdown] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sweep-level results
# ---------------------------------------------------------------------------

@dataclass
class SweepPoint:
    """One point in the hardware sweep."""
    hw_params: dict[str, Any]
    latency_s: float = 0.0
    throughput_qps: float = 0.0
    score: float = 0.0  # scalar for MOBO, set by selection policy
    per_stage: dict[str, CostModelResult] = field(default_factory=dict)


@dataclass
class AssemblyResult:
    """Result from RAGCMAssembly.evaluate()."""
    performance_score: float                    # scalar for MOBO
    best_hw_params: dict[str, Any]              # the selected hw_config
    rago_result_df: pd.DataFrame | None = None  # raw RAGO sweep output
    sweep_dfs: dict[str, pd.DataFrame] | None = None  # per-stage input sweep DataFrames
    # All swept configs when the BATCHING axis was searched (agentic engine): the
    # compact per-config records (placement × batch × scheduling
    # → qps/latency), retained so the chosen batching policy is auditable and the
    # whole sweep is persisted, not just the selected row. None when not swept.
    swept_configs: list[dict] | None = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dict."""
        n_sweep_points = (
            len(self.rago_result_df) if self.rago_result_df is not None else 0
        )
        return {
            "performance_score": self.performance_score,
            "best_hw_params": self.best_hw_params,
            "n_sweep_points": n_sweep_points,
            "n_swept_configs": len(self.swept_configs) if self.swept_configs else 0,
        }
