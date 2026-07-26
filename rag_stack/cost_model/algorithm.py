"""Hardware-independent extension contract between algorithm and system models.

Algorithm models describe work, data volumes, and dependencies.  They do not
receive hardware or return seconds.  A system model selects execution resources
and maps these demands to time.  This module has only standard-library imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


def _nonnegative(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class DataProfile:
    """A logical input/output collection; unknown byte volume stays unknown."""

    items: int
    bytes: int | None = None
    properties: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (("items", self.items), ("bytes", self.bytes)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class AlgorithmWorkload:
    """Named input ports and implementation-specific workload statistics.

    Each instance describes one logical invocation, not an entire cohort.
    Statistics can come from observed RAG-IR data or a workload predictor; the
    algorithm model must identify any assumptions in its output metadata.
    """

    inputs: Mapping[str, DataProfile]
    statistics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseWork:
    """One phase's logical work, before hardware/cache/parallelism selection.

    ``work`` keeps operation kinds separate (comparisons, integer operations,
    floating-point divisions, etc.).  Byte counts describe accesses and unique
    logical data, not placement in a particular hardware cache tier.
    """

    name: str
    work: Mapping[str, float]
    total_bytes: float
    unique_bytes: float | None = None
    working_set_bytes: float | None = None
    compute_mode: str = "scalar"
    access_block_bytes: float | None = None
    parallel_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("phase name must not be empty")
        for kind, count in self.work.items():
            _nonnegative(count, f"work[{kind!r}]")
        for name in ("total_bytes", "unique_bytes", "working_set_bytes",
                     "access_block_bytes", "parallel_fraction"):
            value = getattr(self, name)
            if value is not None:
                _nonnegative(value, name)
        if self.parallel_fraction > 1:
            raise ValueError("parallel_fraction must be between zero and one")


@dataclass(frozen=True)
class OperatorWorkProfile:
    """The portable output of one stage's algorithm model.

    Dependencies express required order, not a promise of parallel execution.
    Outputs describe data delivered to downstream stages.  Metadata explains
    estimation provenance; hardware timing coefficients do not belong here.
    """

    phases: tuple[PhaseWork, ...]
    dependencies: tuple[tuple[str, str], ...]
    outputs: Mapping[str, DataProfile]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = {phase.name for phase in self.phases}
        if len(names) != len(self.phases):
            raise ValueError("phase names must be unique")
        successors = {name: set() for name in names}
        indegree = dict.fromkeys(names, 0)
        for source, target in self.dependencies:
            if source not in names or target not in names:
                raise ValueError("dependency refers to an unknown phase")
            if target not in successors[source]:
                successors[source].add(target)
                indegree[target] += 1
        ready = [name for name, degree in indegree.items() if degree == 0]
        visited = 0
        while ready:
            source = ready.pop()
            visited += 1
            for target in successors[source]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
        if visited != len(names):
            raise ValueError("operator work dependencies must be acyclic")


class AlgorithmModel(Protocol):
    def build(
        self, config: Mapping[str, Any], workload: AlgorithmWorkload,
    ) -> OperatorWorkProfile:
        """Describe work without consuming a hardware specification."""
        ...


ALGORITHM_MODEL_REGISTRY: dict[str, type[AlgorithmModel]] = {}


def register_algorithm_model(name: str):
    """Register an imported algorithm-model class under its component name."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("algorithm model name must not be empty")

    def register(cls: type[AlgorithmModel]) -> type[AlgorithmModel]:
        if not callable(getattr(cls, "build", None)):
            raise TypeError("algorithm model must implement build(config, workload)")
        if name in ALGORITHM_MODEL_REGISTRY and ALGORITHM_MODEL_REGISTRY[name] is not cls:
            raise ValueError(f"algorithm model {name!r} is already registered")
        ALGORITHM_MODEL_REGISTRY[name] = cls
        return cls

    return register
