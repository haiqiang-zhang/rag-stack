from rago.config import PhysicalMapping, enumerate_concrete_device_mappings


DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
LINKS = {
    tuple(sorted((left, right))): (
        (300.0, 0.4)
        if {left, right} in ({"cuda:0", "cuda:1"}, {"cuda:2", "cuda:3"})
        else (54.0, 1.0)
    )
    for index, left in enumerate(DEVICES)
    for right in DEVICES[index + 1:]
}


def _pair_kind(devices):
    return LINKS[tuple(sorted(devices))][0]


def test_rago_enumerates_same_pair_and_cross_pair_2_plus_2_layouts():
    mapping = PhysicalMapping(
        num_chips={"prefill": 2, "decode": 2, "e2e": 4},
        collocation_strategy=[["prefill"], ["decode"]],
        placement_policy="disaggregated",
    )

    layouts = enumerate_concrete_device_mappings(
        mapping,
        available_devices=DEVICES,
        device_links=LINKS,
    )

    kinds = {
        tuple(_pair_kind(layout.stage_devices[stage]) for stage in ("prefill", "decode"))
        for layout in layouts
    }
    assert (300.0, 300.0) in kinds
    assert (54.0, 54.0) in kinds
    assert len({layout.device_layout_id for layout in layouts}) == len(layouts)


def test_rago_preserves_topology_distinct_tp4_rank_orders():
    mapping = PhysicalMapping(
        num_chips={"prefill": 4, "decode": 4, "e2e": 4},
        collocation_strategy=[["prefill", "decode"]],
        placement_policy="collocated",
    )

    layouts = enumerate_concrete_device_mappings(
        mapping,
        available_devices=DEVICES,
        device_links=LINKS,
    )

    orders = {tuple(layout.stage_devices["prefill"]) for layout in layouts}
    assert tuple(DEVICES) in orders
    assert ("cuda:0", "cuda:2", "cuda:1", "cuda:3") in orders
    assert all(
        layout.stage_devices["prefill"] == layout.stage_devices["decode"]
        for layout in layouts
    )


def test_rago_fixed_mapping_keeps_exact_resolved_devices():
    mapping = PhysicalMapping(
        num_chips={"prefill": 2, "decode": 2, "e2e": 2},
        collocation_strategy=[["prefill", "decode"]],
        placement_policy="collocated",
        available_devices=DEVICES,
        resource_group_devices=[["cuda:1", "cuda:2"]],
        stage_devices={
            "prefill": ["cuda:1", "cuda:2"],
            "decode": ["cuda:1", "cuda:2"],
        },
        device_layout_id="fixed_resolved",
    )

    layouts = enumerate_concrete_device_mappings(
        mapping,
        available_devices=DEVICES,
        device_links=LINKS,
    )

    assert layouts == [mapping]
    assert layouts[0].stage_devices["prefill"] == ["cuda:1", "cuda:2"]


def test_rago_balances_small_riders_within_a_larger_group():
    mapping = PhysicalMapping(
        num_chips={"rider_a": 1, "rider_b": 1, "engine": 2, "e2e": 2},
        collocation_strategy=[["rider_a", "rider_b", "engine"]],
        placement_policy="collocated",
    )

    layout = enumerate_concrete_device_mappings(
        mapping,
        available_devices=DEVICES[:2],
        device_links={tuple(sorted(DEVICES[:2])): (300.0, 0.4)},
    )[0]

    assert layout.stage_devices["rider_a"] == ["cuda:0"]
    assert layout.stage_devices["rider_b"] == ["cuda:1"]
    assert layout.stage_devices["engine"] == ["cuda:0", "cuda:1"]


def test_rago_canonical_rider_balance_matches_measured_resolver():
    """Rider assignment is a canonical policy, not another search axis."""
    from rag_stack.search_space.placement import derive_layout

    rago_layout = enumerate_concrete_device_mappings(
        PhysicalMapping(
            num_chips={
                "encode": 1,
                "passage_reranker": 1,
                "prefill": 2,
                "decode": 2,
                "e2e": 2,
            },
            collocation_strategy=[[
                "encode", "passage_reranker", "prefill", "decode",
            ]],
            placement_policy="collocated",
        ),
        available_devices=DEVICES[:2],
        device_links={tuple(sorted(DEVICES[:2])): (300.0, 0.4)},
    )[0]
    measured_layout = derive_layout(
        [
            "semantic_retrieval_encode",
            "passage_reranker",
            "generator_prefill",
            "generator_decode",
        ],
        {
            "cut_after_semantic_retrieval_encode": 0,
            "cut_after_passage_reranker": 0,
            "cut_after_generator_prefill": 0,
        },
        {
            "semantic_retrieval": 1,
            "passage_reranker": 1,
            "generator": 2,
        },
        DEVICES[:2],
    )

    assert rago_layout.stage_devices["encode"] == measured_layout["engines"][
        "semantic_retrieval"
    ]["devices"]
    assert rago_layout.stage_devices["passage_reranker"] == measured_layout[
        "engines"
    ]["passage_reranker"]["devices"]
    assert rago_layout.stage_devices["prefill"] == measured_layout["engines"][
        "generator"
    ]["devices"]
    assert rago_layout.stage_devices["decode"] == measured_layout["engines"][
        "generator"
    ]["devices"]


def test_rag_stack_fixed_adapter_preserves_resolved_cross_pair_order():
    from rag_stack.cost_model.rago_adapter import to_rago_physical_mapping

    mapping = PhysicalMapping(
        num_chips={
            "generator_prefill": 2,
            "generator_decode": 2,
        },
        collocation_strategy=[["generator_prefill", "generator_decode"]],
        placement_policy="collocated",
    )
    system = {
        "hardware": {"available_gpus": DEVICES},
        "layout": {
            "performance_stages": {
                "generator_prefill": {
                    "kind": "gpu", "devices": ["cuda:1", "cuda:2"],
                },
                "generator_decode": {
                    "kind": "gpu", "devices": ["cuda:1", "cuda:2"],
                },
            }
        },
    }

    converted = to_rago_physical_mapping(mapping, system_config=system)

    assert converted.available_devices == DEVICES
    assert converted.stage_devices == {
        "prefill": ["cuda:1", "cuda:2"],
        "decode": ["cuda:1", "cuda:2"],
    }
    assert converted.resource_group_devices == [["cuda:1", "cuda:2"]]
    assert converted.device_layout_id == "fixed_resolved"
