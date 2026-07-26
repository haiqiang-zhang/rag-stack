"""Explicit RRF costs retain their units at the external RAGO boundary."""

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from rag_stack.cost_model.rago_adapter import (
    RagoNameMap,
    to_rago_chips,
    to_rago_groups,
    to_rago_physical_mapping,
    to_rago_stage,
    to_rago_sweep_dfs,
)


def _search_table(batches=(1, 2, 8, 16)):
    return pd.DataFrame({
        "db.name": ["retrieval"] * len(batches),
        "hardware.name": ["system"] * len(batches),
        "hardware.num_servers": [1] * len(batches),
        "batch_size": list(batches),
        "time_ms": [10.0] * len(batches),
        "faiss_ivf_parallel_mode": [0] * len(batches),
    })


def _fusion_table():
    # Each visit merges four lists and costs six milliseconds. Batch size is
    # the number of visits, independent of the search backend's batch grid.
    return pd.DataFrame({"batch_size": [1, 4, 16], "time_ms": [6.0, 24.0, 96.0]})


def test_lower_fusion_only_at_boundary_without_mutating_component_tables():
    search, fusion = _search_table(), _fusion_table()
    before_search, before_fusion = search.copy(deep=True), fusion.copy(deep=True)
    generator = object()

    converted = to_rago_sweep_dfs({
        "semantic_retrieval_vectorsearch": search,
        "retrieval_fusion": fusion,
        "generator_prefill": generator,
        "generator_decode": None,
    }, retrieval_fanout=4)

    assert set(converted) == {"retrieval", "prefill", "decode"}
    assert converted["prefill"] is generator
    assert converted["decode"] is None
    assert converted["retrieval"]["time_ms"].tolist() == [11.5, 13.0, 22.0, 34.0]
    assert converted["retrieval"]["faiss_ivf_parallel_mode"].tolist() == [0] * 4
    pd.testing.assert_frame_equal(search, before_search)
    pd.testing.assert_frame_equal(fusion, before_fusion)


@pytest.mark.parametrize("backend_batch", [2, 8])
def test_rago_work_to_request_conversion_charges_each_retrieval_visit(backend_batch):
    from rago.data import preprocess_data
    from rago.rago_sweep import RAGSweep

    # R=2 visits per request and K=4 lists per visit. Both a partial backend
    # batch and a batch holding the complete request must charge R*f = 12 ms.
    table = to_rago_sweep_dfs({
        "semantic_retrieval_vectorsearch": _search_table((backend_batch,)),
        "retrieval_fusion": _fusion_table(),
    }, retrieval_fanout=4)["retrieval"]
    with_fusion = RAGSweep._apply_fanout(
        preprocess_data(table, stage="retrieval", return_pareto=False),
        fanout=8,
    )
    without_fusion = RAGSweep._apply_fanout(
        preprocess_data(_search_table((backend_batch,)), stage="retrieval", return_pareto=False),
        fanout=8,
    )

    assert with_fusion["batch_size"].tolist() == [1]
    assert with_fusion["latency_s"].iloc[0] - without_fusion["latency_s"].iloc[0] == pytest.approx(0.012)


@pytest.mark.parametrize("server_limit", [2, 3, 4, 8])
def test_search_replication_does_not_replicate_shared_fusion_worker(server_limit):
    from rago.data import preprocess_data
    from rago.rago_sweep import RAGSweep

    lowered = to_rago_sweep_dfs({
        "semantic_retrieval_vectorsearch": _search_table((4,)),
        "retrieval_fusion": _fusion_table(),
    }, retrieval_fanout=4, max_retrieval_servers=server_limit)["retrieval"]
    assert set(range(1, server_limit + 1)) <= set(lowered["hardware.num_servers"])
    for row in lowered.itertuples(index=False, name=None):
        values = dict(zip(lowered.columns, row))
        servers = values["hardware.num_servers"]
        assert values["batch_size"] == 4 * servers
        assert values["time_ms"] == 10.0 + 6.0 * servers

    processed = preprocess_data(lowered, stage="retrieval", return_pareto=False)
    before = processed.copy(deep=True)
    # Exercise RAGO's real scaler: all candidate server counts are already
    # represented, including its power-of-two coverage check for a bound of 3.
    sweep = SimpleNamespace(sweep_df={"retrieval": processed})
    RAGSweep.scale_up_performance_to_more_chips(
        sweep, "retrieval", max_num_retrieval_servers=server_limit,
    )
    pd.testing.assert_frame_equal(sweep.sweep_df["retrieval"], before)
    user_units = RAGSweep._apply_fanout(sweep.sweep_df["retrieval"], fanout=4)
    # Six ms per visit limits one shared fusion worker to 166.67 visits/s.
    # RAGO's cached qps column retains backend-query units in this branch;
    # capacity in user units comes from its converted batch and service time.
    request_capacity = user_units["batch_size"] / user_units["latency_s"]
    assert (request_capacity <= 1.0 / 0.006).all()


def test_existing_multi_server_search_measurements_survive_expansion():
    measured = _search_table((8,))
    measured["hardware.num_servers"] = 2
    measured["time_ms"] = 8.0
    search = pd.concat([_search_table((4,)), measured], ignore_index=True)
    before = search.copy(deep=True)

    lowered = to_rago_sweep_dfs({
        "semantic_retrieval_vectorsearch": search,
        "retrieval_fusion": _fusion_table(),
    }, retrieval_fanout=4, max_retrieval_servers=4)["retrieval"]

    rows = lowered.set_index("hardware.num_servers")
    assert rows.loc[2, "batch_size"] == 8
    assert rows.loc[2, "time_ms"] == 20.0  # Measured 8 ms search + 2*6 ms fusion.
    assert rows.loc[4, "batch_size"] == 16
    assert rows.loc[4, "time_ms"] == 34.0  # Replicated 10 ms search + 4*6 ms fusion.
    pd.testing.assert_frame_equal(search, before)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, float("inf"), "bad"])
def test_server_replication_bound_must_be_positive_integer(limit):
    with pytest.raises(ValueError, match="max_retrieval_servers"):
        to_rago_sweep_dfs({}, max_retrieval_servers=limit)


def test_no_fusion_passes_search_table_through_unchanged():
    search = _search_table()
    for fusion in (None, pd.DataFrame()):
        converted = to_rago_sweep_dfs({
            "semantic_retrieval_vectorsearch": search,
            "retrieval_fusion": fusion,
        })
        assert converted == {"retrieval": search}
        assert converted["retrieval"] is search


@pytest.mark.parametrize("fanout", [0, -1, 1.5, float("nan"), float("inf"), True, "bad"])
def test_fusion_rejects_invalid_per_visit_fanout(fanout):
    with pytest.raises(ValueError, match="retrieval_fanout"):
        to_rago_sweep_dfs({
            "semantic_retrieval_vectorsearch": _search_table(),
            "retrieval_fusion": _fusion_table(),
        }, retrieval_fanout=fanout)


def test_fusion_without_search_table_is_not_silently_discarded():
    with pytest.raises(ValueError, match="requires a vector-search"):
        to_rago_sweep_dfs({"retrieval_fusion": _fusion_table()}, retrieval_fanout=4)


@pytest.mark.parametrize("column, value, message", [
    ("time_ms", 25.0, "scale linearly"),
    ("time_ms", float("nan"), "finite and nonnegative"),
    ("time_ms", -1.0, "finite and nonnegative"),
    ("batch_size", 0, "positive integer"),
    ("batch_size", 1.5, "positive integer"),
])
def test_invalid_fusion_table_fails_closed(column, value, message):
    fusion = _fusion_table().astype(float)
    fusion.loc[1, column] = value
    with pytest.raises(ValueError, match=message):
        to_rago_sweep_dfs({
            "semantic_retrieval_vectorsearch": _search_table(),
            "retrieval_fusion": fusion,
        }, retrieval_fanout=4)


def test_fusion_is_not_exposed_as_an_unsupported_rago_gpu_stage():
    with pytest.raises(ValueError, match="no independent RAGO stage"):
        to_rago_stage("retrieval_fusion")
    assert to_rago_groups([
        ["semantic_retrieval_vectorsearch", "retrieval_fusion"],
        ["retrieval_fusion"],
        ["generator_prefill", "generator_decode"],
    ]) == [["retrieval"], ["prefill", "decode"]]
    assert to_rago_chips({"retrieval_fusion": 0, "generator_prefill": 1}) == {"prefill": 1}
    names = RagoNameMap(["semantic_retrieval_vectorsearch", "retrieval_fusion"])
    assert names.active == ("semantic_retrieval_vectorsearch", "retrieval_fusion")
    assert names.from_rago("retrieval") == "semantic_retrieval_vectorsearch"
    with pytest.raises(KeyError):
        names.from_rago("retrieval_fusion")


def test_physical_mapping_removes_fusion_only_group_and_matching_device_slot():
    from rago.config import PhysicalMapping

    original = PhysicalMapping(
        num_chips={"retrieval_fusion": 0, "generator_prefill": 1, "e2e": 1},
        collocation_strategy=[["retrieval_fusion"], ["generator_prefill"]],
        available_devices=["cuda:0"],
        resource_group_devices=[[], ["cuda:0"]],
        stage_devices={"retrieval_fusion": [], "generator_prefill": ["cuda:0"]},
    )
    before = deepcopy(original)

    converted = to_rago_physical_mapping(original)

    assert converted.collocation_strategy == [["prefill"]]
    assert converted.resource_group_devices == [["cuda:0"]]
    assert converted.stage_devices == {"prefill": ["cuda:0"]}
    assert converted.num_chips == {"prefill": 1, "e2e": 1}
    assert vars(original) == vars(before)
