import math

import pytest

from rag_stack.cost_model.queueing_analytical import (
    QueueResource,
    solve_closed_queue,
)


def test_single_server_is_exact_for_an_isolated_closed_center():
    result = solve_closed_queue(
        [QueueResource("generator", visits=1, service_s=0.2)],
        population=5,
    )

    queue = result.queue("generator")
    assert result.approximation == "mva_single_server_exact"
    assert result.throughput == pytest.approx(5.0)
    assert result.cycle_time_s == pytest.approx(1.0)
    assert queue.mean_occupancy == pytest.approx(5.0)
    assert queue.residence_s_per_cycle == pytest.approx(1.0)
    assert queue.wait_s_per_cycle == pytest.approx(0.8)
    assert queue.wait_s_per_visit == pytest.approx(0.8)
    assert queue.utilization == pytest.approx(1.0)
    assert result.bottlenecks == ("generator",)


def test_multiserver_center_respects_parallel_capacity():
    underfilled = solve_closed_queue(
        [QueueResource("retrieval", visits=1, service_s=0.1, capacity=2)],
        population=1,
    )
    saturated = solve_closed_queue(
        [QueueResource("retrieval", visits=1, service_s=0.1, capacity=2)],
        population=5,
    )

    assert saturated.approximation == "mva_multiserver_approximation"
    assert underfilled.throughput == pytest.approx(10.0)
    assert underfilled.queue("retrieval").wait_s_per_cycle == 0.0
    assert saturated.throughput == pytest.approx(20.0)
    assert saturated.queue("retrieval").mean_occupancy == pytest.approx(5.0)
    assert saturated.queue("retrieval").residence_s_per_cycle == pytest.approx(0.25)
    assert saturated.queue("retrieval").wait_s_per_cycle == pytest.approx(0.15)


def test_visits_scale_demand_throughput_and_per_visit_wait():
    result = solve_closed_queue(
        [QueueResource("iterative", visits=2, service_s=0.1)],
        population=3,
    )

    queue = result.queue("iterative")
    assert queue.service_demand_s == pytest.approx(0.2)
    assert result.throughput == pytest.approx(5.0)
    assert queue.visit_throughput_per_s == pytest.approx(10.0)
    assert queue.residence_s_per_cycle == pytest.approx(0.6)
    assert queue.wait_s_per_cycle == pytest.approx(0.4)
    assert queue.wait_s_per_visit == pytest.approx(0.2)


def test_load_dependent_capacity_is_deterministic_and_saturates():
    resource = QueueResource(
        "batched_worker",
        visits=1,
        service_s=0.1,
        capacity_by_occupancy=(1, 2, 2),
    )
    first = solve_closed_queue([resource], population=3)
    second = solve_closed_queue([resource], population=3)

    assert first == second
    assert first.approximation == "mva_load_dependent_capacity"
    assert first.throughput == pytest.approx(20.0)
    assert first.queue("batched_worker").mean_occupancy == pytest.approx(3.0)
    assert first.queue("batched_worker").effective_capacity == pytest.approx(2.0)
    assert first.queue("batched_worker").wait_s_per_cycle == pytest.approx(0.05)


def test_capacity_table_interpolates_expected_occupancy():
    resource = QueueResource(
        "worker",
        visits=1,
        service_s=1,
        capacity_by_occupancy=(1, 2, 4),
    )

    assert resource.effective_capacity(1.0) == pytest.approx(1.0)
    assert resource.effective_capacity(1.5) == pytest.approx(1.5)
    assert resource.effective_capacity(2.5) == pytest.approx(3.0)
    assert resource.effective_capacity(100.0) == pytest.approx(4.0)


def test_two_center_network_obeys_interactive_response_law():
    population = 8
    think_time = 0.4
    result = solve_closed_queue(
        [
            QueueResource("generator", visits=1, service_s=0.08),
            QueueResource("retrieval", visits=2, service_s=0.01, capacity=4),
        ],
        population=population,
        think_time_s=think_time,
    )

    queue_population = math.fsum(
        queue.mean_occupancy for queue in result.queues
    )
    assert queue_population + result.throughput * think_time == pytest.approx(
        population,
    )
    assert result.throughput > 0.0
    assert result.throughput <= 1.0 / 0.08
    assert result.bottlenecks == ("generator",)


def test_throughput_is_monotone_and_bounded_by_service_capacity():
    resources = [
        QueueResource("a", visits=1, service_s=0.1),
        QueueResource("b", visits=1, service_s=0.02, capacity=2),
    ]
    throughputs = [
        solve_closed_queue(resources, population=n).throughput
        for n in range(1, 33)
    ]

    assert all(
        later + 1e-12 >= earlier
        for earlier, later in zip(throughputs, throughputs[1:])
    )
    assert all(value <= 10.0 + 1e-12 for value in throughputs)
    assert throughputs[-1] > 9.9


def test_zero_population_and_zero_demand_are_safe():
    resources = [QueueResource("noop", visits=0, service_s=0)]

    zero = solve_closed_queue(resources, population=0)
    degenerate = solve_closed_queue(resources, population=5)
    delay_only = solve_closed_queue(
        resources, population=5, think_time_s=2.0,
    )

    assert zero.status == "zero_population"
    assert zero.throughput == 0.0
    assert degenerate.status == "degenerate_zero_cycle"
    assert degenerate.throughput == 0.0
    assert delay_only.status == "delay_center_only"
    assert delay_only.throughput == pytest.approx(2.5)
    assert delay_only.cycle_time_s == pytest.approx(2.0)
    assert all(queue.mean_occupancy == 0.0 for queue in delay_only.queues)


def test_zero_demand_center_does_not_change_an_active_network():
    active = QueueResource("active", visits=1, service_s=0.1)
    expected = solve_closed_queue([active], population=4)
    actual = solve_closed_queue(
        [active, QueueResource("noop", visits=0, service_s=10)],
        population=4,
    )

    assert actual.throughput == pytest.approx(expected.throughput)
    assert actual.queue("noop").mean_occupancy == 0.0


def test_large_population_remains_finite():
    result = solve_closed_queue(
        [QueueResource("server", visits=1, service_s=0.1)],
        population=4096,
    )

    assert result.throughput == pytest.approx(10.0)
    assert result.queue("server").mean_occupancy == pytest.approx(4096.0)
    assert math.isfinite(result.cycle_time_s)
    assert math.isfinite(result.queue("server").wait_s_per_cycle)


@pytest.mark.parametrize(
    ("resource", "message"),
    [
        (QueueResource("x", visits=-1, service_s=1), "visits"),
        (QueueResource("x", visits=1, service_s=float("nan")), "service_s"),
        (QueueResource("x", visits=1, service_s=1, capacity=0), "capacity"),
        (
            QueueResource(
                "x", visits=1, service_s=1,
                capacity_by_occupancy=(1, float("inf")),
            ),
            "capacity_by_occupancy",
        ),
    ],
)
def test_invalid_resource_values_fail_closed(resource, message):
    with pytest.raises(ValueError, match=message):
        solve_closed_queue([resource], population=1)


def test_invalid_population_and_duplicate_names_fail_closed():
    resource = QueueResource("x", visits=1, service_s=1)

    with pytest.raises(TypeError, match="population"):
        solve_closed_queue([resource], population=1.5)
    with pytest.raises(ValueError, match="population"):
        solve_closed_queue([resource], population=-1)
    with pytest.raises(ValueError, match="duplicate"):
        solve_closed_queue([resource, resource], population=1)
    with pytest.raises(KeyError):
        solve_closed_queue([resource], population=1).queue("missing")
