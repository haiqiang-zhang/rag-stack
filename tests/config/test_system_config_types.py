"""Focused serialization contracts for :class:`rag_stack.types.SystemConfig`."""

from __future__ import annotations

from rag_stack.types import SystemConfig


def test_inline_cpu_hardware_key_survives_round_trip() -> None:
    source = {
        "cpu": {
            "hardware_key": "epyc_7742_2s",
            "num_cores": 128,
            "peak_flops": 1.39264e13,
            "peak_int_ops": 6.9632e12,
            "mem_bandwidth": 227400300000,
            "mem_bandwidth_1t": 21993800000,
        },
    }

    first = SystemConfig.from_dict(source)
    serialized = first.to_dict()
    second = SystemConfig.from_dict(serialized)

    assert first.cpu_key == "epyc_7742_2s"
    assert serialized["cpu"]["hardware_key"] == "epyc_7742_2s"
    assert second.cpu_key == first.cpu_key


def test_cm_search_space_round_trips_under_canonical_wire_key() -> None:
    search_space = {
        "placement_policy": ["collocated", "disaggregated"],
        "min_num_gpus": 1,
        "max_num_gpus": 8,
        "batch_size_decode": [1, 8, 32, 256],
        "dynamic_batch_timeout_s": [0.002, 0.01],
    }
    source = {
        "system_hardware": "sgs_a100x8_epyc7742",
        "rag_cm": {"calibration_mode": "uncalibrated_spec"},
        "cm_search_space": search_space,
    }

    first = SystemConfig.from_dict(source)
    serialized = first.to_dict()
    second = SystemConfig.from_dict(serialized)

    assert serialized["cm_search_space"] == search_space
    assert "search_space" not in serialized
    assert second.search_space == first.search_space == search_space
    assert second.rag_cm == first.rag_cm == source["rag_cm"]


def test_legacy_search_space_input_is_reemitted_canonically() -> None:
    legacy = {
        "search_space": {
            "max_batch_size_request": 256,
            "max_num_gpus": 8,
        },
    }

    hardware = SystemConfig.from_dict(legacy)
    serialized = hardware.to_dict()

    assert hardware.search_space == legacy["search_space"]
    assert serialized["cm_search_space"] == legacy["search_space"]
    assert "search_space" not in serialized


def test_empty_cm_search_space_survives_round_trip() -> None:
    hardware = SystemConfig.from_dict({"cm_search_space": {}})

    serialized = hardware.to_dict()

    assert hardware.search_space == {}
    assert serialized["cm_search_space"] == {}
    assert SystemConfig.from_dict(serialized).search_space == {}


def test_serialized_search_space_does_not_alias_live_config() -> None:
    hardware = SystemConfig(
        search_space={"placement_policy": ["collocated", "disaggregated"]},
    )

    serialized = hardware.to_dict()
    serialized["cm_search_space"]["placement_policy"].append("invalid")

    assert hardware.search_space == {
        "placement_policy": ["collocated", "disaggregated"],
    }
