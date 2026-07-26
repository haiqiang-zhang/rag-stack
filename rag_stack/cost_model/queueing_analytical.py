"""Deterministic analytical solver for small closed queueing networks.

The production discrete-event simulator is authoritative for scheduling
details.  This module provides a deliberately smaller, coefficient-free model
for screening and diagnostics: callers describe the number of visits and the
mean service time at each resource, plus its parallel service capacity, and
the solver applies mean-value analysis (MVA) to a fixed closed population.

For a single-server product-form center the MVA recurrence is exact.  A
multi-server or load-dependent center uses the standard arrival-theorem
approximation

``R_i(n) = D_i * max(1, (1 + Q_i(n-1)) / c_i(1 + Q_i(n-1)))``

where ``D_i`` is service demand per completed cycle and ``c_i`` is service
capacity in single-server equivalents.  The approximation is exact for an
isolated work-conserving center with constant capacity.  It preserves the
important closed-network limits without fitted constants: no queueing below
available parallelism and asymptotic throughput ``capacity / demand``.

The implementation is intentionally independent from assembly, trace, and
DES protocols.  It performs no I/O and owns no global mutable state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Sequence


@dataclass(frozen=True)
class QueueResource:
    """One service center in a closed request cycle.

    Parameters
    ----------
    name:
        Stable diagnostic identity.  Names must be unique within a solve.
    visits:
        Mean number of visits to this center per completed request cycle.
    service_s:
        Mean service time of one visit in seconds.
    capacity:
        Constant parallel service capacity in single-server equivalents.
        ``1`` is a single server and ``m`` is an ``m``-server center.
    capacity_by_occupancy:
        Optional load-dependent total capacity.  Entry ``j - 1`` is the
        capacity when ``j`` jobs occupy the center.  Values between integer
        occupancies are linearly interpolated; the final value is retained
        beyond the table.  When supplied, it replaces ``capacity`` for the
        MVA residence-time approximation.
    """

    name: str
    visits: float
    service_s: float
    capacity: float = 1.0
    capacity_by_occupancy: tuple[float, ...] = ()

    @property
    def service_demand_s(self) -> float:
        """Mean service demand contributed by this center per cycle."""

        return float(self.visits) * float(self.service_s)

    def effective_capacity(self, occupancy: float) -> float:
        """Return deterministic capacity at an expected occupancy."""

        if not self.capacity_by_occupancy:
            return float(self.capacity)

        table = self.capacity_by_occupancy
        jobs = max(1.0, float(occupancy))
        if jobs <= 1.0:
            return float(table[0])
        if jobs >= len(table):
            return float(table[-1])

        lower_jobs = int(math.floor(jobs))
        upper_jobs = lower_jobs + 1
        lower = float(table[lower_jobs - 1])
        upper = float(table[upper_jobs - 1])
        fraction = jobs - lower_jobs
        return lower + fraction * (upper - lower)


@dataclass(frozen=True)
class QueueMetrics:
    """Steady-state diagnostics for one resource."""

    name: str
    visits: float
    service_s: float
    service_demand_s: float
    mean_occupancy: float
    residence_s_per_cycle: float
    residence_s_per_visit: float
    wait_s_per_cycle: float
    wait_s_per_visit: float
    visit_throughput_per_s: float
    effective_capacity: float
    utilization: float


@dataclass(frozen=True)
class ClosedQueueResult:
    """Result of :func:`solve_closed_queue`."""

    population: int
    throughput_per_s: float
    cycle_time_s: float
    think_time_s: float
    queues: tuple[QueueMetrics, ...]
    bottlenecks: tuple[str, ...]
    approximation: str
    status: str = "ok"

    @property
    def throughput(self) -> float:
        """Short alias for ``throughput_per_s``."""

        return self.throughput_per_s

    def queue(self, name: str) -> QueueMetrics:
        """Return metrics for ``name`` or raise ``KeyError``."""

        for metrics in self.queues:
            if metrics.name == name:
                return metrics
        raise KeyError(name)


def _finite_nonnegative(value: float, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return parsed


def _finite_positive(value: float, label: str) -> float:
    parsed = _finite_nonnegative(value, label)
    if parsed <= 0.0:
        raise ValueError(f"{label} must be positive")
    return parsed


def _validate_resources(
    resources: Sequence[QueueResource],
) -> tuple[QueueResource, ...]:
    normalized: list[QueueResource] = []
    names: set[str] = set()
    for index, resource in enumerate(resources):
        if not isinstance(resource, QueueResource):
            raise TypeError(
                "resources must contain QueueResource instances; "
                f"item {index} is {type(resource).__name__}"
            )
        name = str(resource.name).strip()
        if not name:
            raise ValueError(f"resources[{index}].name must be non-empty")
        if name in names:
            raise ValueError(f"duplicate queue resource name {name!r}")
        names.add(name)

        visits = _finite_nonnegative(
            resource.visits, f"resource {name!r} visits",
        )
        service_s = _finite_nonnegative(
            resource.service_s, f"resource {name!r} service_s",
        )
        capacity = _finite_positive(
            resource.capacity, f"resource {name!r} capacity",
        )
        capacity_table = tuple(
            _finite_positive(
                value,
                f"resource {name!r} capacity_by_occupancy[{table_index}]",
            )
            for table_index, value in enumerate(
                resource.capacity_by_occupancy,
            )
        )
        normalized.append(QueueResource(
            name=name,
            visits=visits,
            service_s=service_s,
            capacity=capacity,
            capacity_by_occupancy=capacity_table,
        ))
    return tuple(normalized)


def _zero_result(
    resources: tuple[QueueResource, ...],
    *,
    population: int,
    think_time_s: float,
    approximation: str,
    status: str,
) -> ClosedQueueResult:
    queues = tuple(
        QueueMetrics(
            name=resource.name,
            visits=resource.visits,
            service_s=resource.service_s,
            service_demand_s=resource.service_demand_s,
            mean_occupancy=0.0,
            residence_s_per_cycle=0.0,
            residence_s_per_visit=0.0,
            wait_s_per_cycle=0.0,
            wait_s_per_visit=0.0,
            visit_throughput_per_s=0.0,
            effective_capacity=resource.effective_capacity(1.0),
            utilization=0.0,
        )
        for resource in resources
    )
    return ClosedQueueResult(
        population=population,
        throughput_per_s=0.0,
        cycle_time_s=0.0,
        think_time_s=think_time_s,
        queues=queues,
        bottlenecks=(),
        approximation=approximation,
        status=status,
    )


def solve_closed_queue(
    resources: Sequence[QueueResource],
    population: int,
    *,
    think_time_s: float = 0.0,
) -> ClosedQueueResult:
    """Solve a closed queueing network with deterministic MVA.

    ``population`` is the fixed number of circulating requests.  ``think_time``
    is a delay-center residence per request cycle and consumes no queue
    capacity.  Zero-demand resources remain in the result with zero occupancy.

    Invalid model declarations raise immediately.  Valid degenerate systems
    are safe: zero population returns zero throughput, and a positive
    population with neither service demand nor think time returns the explicit
    ``"degenerate_zero_cycle"`` status instead of infinity or NaN.
    """

    if isinstance(population, bool) or not isinstance(population, Integral):
        raise TypeError("population must be a non-negative integer")
    population = int(population)
    if population < 0:
        raise ValueError("population must be a non-negative integer")
    think = _finite_nonnegative(think_time_s, "think_time_s")
    centers = _validate_resources(resources)
    load_dependent = any(
        resource.capacity_by_occupancy for resource in centers
    )
    multi_server = any(
        resource.capacity != 1.0 for resource in centers
    )
    approximation = (
        "mva_load_dependent_capacity"
        if load_dependent
        else "mva_multiserver_approximation"
        if multi_server
        else "mva_single_server_exact"
    )

    if population == 0:
        return _zero_result(
            centers,
            population=0,
            think_time_s=think,
            approximation=approximation,
            status="zero_population",
        )

    demands = tuple(resource.service_demand_s for resource in centers)
    if not any(demand > 0.0 for demand in demands):
        if think <= 0.0:
            return _zero_result(
                centers,
                population=population,
                think_time_s=think,
                approximation=approximation,
                status="degenerate_zero_cycle",
            )
        throughput = population / think
        zero = _zero_result(
            centers,
            population=population,
            think_time_s=think,
            approximation=approximation,
            status="delay_center_only",
        )
        return ClosedQueueResult(
            population=zero.population,
            throughput_per_s=throughput,
            cycle_time_s=think,
            think_time_s=think,
            queues=zero.queues,
            bottlenecks=(),
            approximation=zero.approximation,
            status=zero.status,
        )

    occupancies = [0.0] * len(centers)
    residences = [0.0] * len(centers)
    throughput = 0.0
    cycle_time = 0.0

    for jobs in range(1, population + 1):
        for index, (resource, demand) in enumerate(zip(centers, demands)):
            if demand <= 0.0:
                residences[index] = 0.0
                continue
            jobs_present = 1.0 + max(0.0, occupancies[index])
            capacity = resource.effective_capacity(jobs_present)
            queueing_factor = max(1.0, jobs_present / capacity)
            residences[index] = demand * queueing_factor

        cycle_time = think + math.fsum(residences)
        if cycle_time <= 0.0 or not math.isfinite(cycle_time):
            return _zero_result(
                centers,
                population=population,
                think_time_s=think,
                approximation=approximation,
                status="degenerate_nonfinite_cycle",
            )
        throughput = jobs / cycle_time
        occupancies = [
            max(0.0, throughput * residence)
            for residence in residences
        ]

    queue_metrics: list[QueueMetrics] = []
    asymptotic_rates: list[tuple[str, float]] = []
    for resource, demand, occupancy, residence in zip(
        centers, demands, occupancies, residences,
    ):
        effective_capacity = resource.effective_capacity(
            max(1.0, occupancy),
        )
        wait_cycle = max(0.0, residence - demand)
        visits = resource.visits
        residence_per_visit = residence / visits if visits > 0.0 else 0.0
        wait_per_visit = wait_cycle / visits if visits > 0.0 else 0.0
        visit_throughput = throughput * visits
        utilization = (
            min(1.0, max(0.0, throughput * demand / effective_capacity))
            if demand > 0.0 else 0.0
        )
        queue_metrics.append(QueueMetrics(
            name=resource.name,
            visits=visits,
            service_s=resource.service_s,
            service_demand_s=demand,
            mean_occupancy=occupancy,
            residence_s_per_cycle=residence,
            residence_s_per_visit=residence_per_visit,
            wait_s_per_cycle=wait_cycle,
            wait_s_per_visit=wait_per_visit,
            visit_throughput_per_s=visit_throughput,
            effective_capacity=effective_capacity,
            utilization=utilization,
        ))
        if demand > 0.0:
            asymptotic_capacity = resource.effective_capacity(
                float(population),
            )
            asymptotic_rates.append((
                resource.name, asymptotic_capacity / demand,
            ))

    bottlenecks: tuple[str, ...] = ()
    if asymptotic_rates:
        limiting_rate = min(rate for _name, rate in asymptotic_rates)
        tolerance = 1e-12 * max(1.0, abs(limiting_rate))
        bottlenecks = tuple(
            name for name, rate in asymptotic_rates
            if abs(rate - limiting_rate) <= tolerance
        )

    return ClosedQueueResult(
        population=population,
        throughput_per_s=throughput,
        cycle_time_s=cycle_time,
        think_time_s=think,
        queues=tuple(queue_metrics),
        bottlenecks=bottlenecks,
        approximation=approximation,
        status="ok",
    )


__all__ = [
    "ClosedQueueResult",
    "QueueMetrics",
    "QueueResource",
    "solve_closed_queue",
]
