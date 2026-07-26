from dataclasses import FrozenInstanceError

import pytest

from rag_stack.cost_model.hardware import (
    CPURecord,
    CacheLevelRecord,
    GPURecord,
    HardwareCatalog,
    HardwareConflictError,
    HardwareDigestMismatchError,
    HardwareRecordKind,
    InterconnectRecord,
    InvalidTopologyError,
    TopologyLink,
    TopologyNode,
    TopologyRecord,
    UnknownHardwareError,
)
def _cpu(*, memory_bandwidth_bytes_s=308e9):
    return CPURecord(
        hardware_key="epyc_9124_2s",
        vendor="AMD",
        model="EPYC 9124",
        architecture="x86_64-zen4",
        socket_count=2,
        physical_core_count=32,
        logical_core_count=32,
        memory_bytes=512 * 1024**3,
        peak_flops_per_s=3.6864e12,
        peak_int_ops_per_s=1.8432e12,
        memory_bandwidth_bytes_s=memory_bandwidth_bytes_s,
        single_thread_memory_bandwidth_bytes_s=38e9,
        simd_width_bits=512,
        cache_levels=(
            CacheLevelRecord(
                name="l3",
                size_bytes=16 * 1024**2,
                instance_count=8,
                cores_per_instance=4,
                bandwidth_bytes_s=1e12,
                single_thread_bandwidth_bytes_s=67e9,
                latency_s=15e-9,
            ),
        ),
    )


def _gpu():
    return GPURecord(
        hardware_key="h100_nvl_94gb",
        vendor="NVIDIA",
        model="H100 NVL",
        architecture="hopper",
        compute_dtype="fp16_tensor_dense",
        peak_tensor_flops_per_s=835e12,
        memory_bytes=93 * 1024**3,
        memory_bandwidth_bytes_s=3938e9,
    )


def _nvlink():
    return InterconnectRecord(
        hardware_key="nvlink4_nvl_pair",
        transport="NVLink 4",
        bandwidth_bytes_s=300e9,
        latency_s=0.4e-6,
        directionality="full_duplex",
    )


def _pcie():
    return InterconnectRecord(
        hardware_key="pcie5_pxb",
        transport="PCIe 5 PXB",
        bandwidth_bytes_s=54e9,
        latency_s=1e-6,
        directionality="full_duplex",
    )


def _topology(*, reverse_order=False):
    nodes = [
        TopologyNode("host", HardwareRecordKind.CPU, "epyc_9124_2s"),
        *[
            TopologyNode(
                f"cuda.{index}", HardwareRecordKind.GPU, "h100_nvl_94gb"
            )
            for index in range(4)
        ],
    ]
    links = [
        TopologyLink("cuda.0", "cuda.1", "nvlink4_nvl_pair"),
        TopologyLink("cuda.2", "cuda.3", "nvlink4_nvl_pair"),
        TopologyLink("cuda.0", "cuda.2", "pcie5_pxb"),
        TopologyLink("cuda.0", "host", "pcie5_pxb"),
        TopologyLink("cuda.1", "host", "pcie5_pxb"),
        TopologyLink("cuda.2", "host", "pcie5_pxb"),
        TopologyLink("cuda.3", "host", "pcie5_pxb"),
    ]
    if reverse_order:
        nodes.reverse()
        links.reverse()
    return TopologyRecord(
        hardware_key="sgs_h100x4_epyc9124",
        nodes=tuple(nodes),
        links=tuple(links),
    )


def _catalog(*, topology=None):
    return HardwareCatalog(
        cpus=(_cpu(),),
        gpus=(_gpu(),),
        interconnects=(_nvlink(), _pcie()),
        topologies=(topology or _topology(),),
    )


def test_records_are_immutable_and_digest_is_canonical():
    cpu = _cpu()
    same_cpu = _cpu()
    assert cpu.digest == same_cpu.digest

    with pytest.raises(FrozenInstanceError):
        cpu.model = "different"  # type: ignore[misc]

    topology = _topology()
    reordered = _topology(reverse_order=True)
    assert topology.nodes == reordered.nodes
    assert topology.links == reordered.links
    assert topology.digest == reordered.digest


def test_lookup_is_exact_and_can_be_digest_pinned():
    catalog = _catalog()
    cpu = catalog.require_cpu("epyc_9124_2s")
    assert catalog.require_cpu(
        "epyc_9124_2s", expected_digest=cpu.digest
    ) is cpu

    identity = catalog.identity(HardwareRecordKind.CPU, "epyc_9124_2s")
    assert catalog.resolve(identity) is cpu

    with pytest.raises(UnknownHardwareError, match="no fallback is allowed"):
        catalog.require_cpu("epyc_9124")
    with pytest.raises(ValueError, match="canonical lowercase ID"):
        catalog.require_cpu("EPYC_9124_2S")
    with pytest.raises(HardwareDigestMismatchError):
        catalog.require_cpu("epyc_9124_2s", expected_digest="0" * 64)


def test_conflicting_registration_is_rejected_without_replacing_original():
    catalog = HardwareCatalog(cpus=(_cpu(),))
    original = catalog.require_cpu("epyc_9124_2s")

    assert catalog.register_cpu(_cpu()) is original
    with pytest.raises(HardwareConflictError, match="conflicting cpu hardware_key"):
        catalog.register_cpu(_cpu(memory_bandwidth_bytes_s=300e9))

    assert catalog.require_cpu("epyc_9124_2s") is original


def test_topology_registration_requires_every_exact_dependency():
    topology = _topology()
    catalog = HardwareCatalog(cpus=(_cpu(),), gpus=(_gpu(),))

    with pytest.raises(InvalidTopologyError, match="unregistered hardware record"):
        catalog.register_topology(topology)

    assert catalog.register_interconnect(_nvlink())
    assert catalog.register_interconnect(_pcie())
    assert catalog.register_topology(topology) is topology


def test_resolved_topology_digest_binds_only_referenced_physical_facts():
    catalog = _catalog(topology=_topology())
    reordered = _catalog(topology=_topology(reverse_order=True))
    digest = catalog.resolved_topology_digest("sgs_h100x4_epyc9124")

    assert digest == reordered.resolved_topology_digest(
        "sgs_h100x4_epyc9124"
    )

    # Adding an unrelated record changes the whole-catalog digest, but not the
    # physical identity of this topology.
    old_catalog_digest = catalog.digest
    catalog.register_gpu(
        GPURecord(
            hardware_key="unrelated_gpu",
            vendor="example",
            model="unrelated",
            architecture="test",
            compute_dtype="fp16",
            peak_tensor_flops_per_s=1.0,
            memory_bytes=1,
            memory_bandwidth_bytes_s=1.0,
        )
    )
    assert catalog.digest != old_catalog_digest
    assert catalog.resolved_topology_digest("sgs_h100x4_epyc9124") == digest


def test_catalog_manifests_use_current_unversioned_shape():
    catalog = _catalog()

    assert set(catalog.to_manifest()) == {"cpu", "gpu", "interconnect", "topology"}
    assert set(catalog.resolved_topology_manifest(
        "sgs_h100x4_epyc9124"
    )) == {"topology", "cpu", "gpu", "interconnect"}


def test_records_reject_missing_or_placeholder_physical_facts():
    with pytest.raises(ValueError, match="finite positive"):
        InterconnectRecord(
            hardware_key="missing_bandwidth",
            transport="unknown",
            bandwidth_bytes_s=0.0,
            latency_s=0.0,
            directionality="full_duplex",
        )

    with pytest.raises(TypeError):
        GPURecord(  # type: ignore[call-arg]
            hardware_key="incomplete_gpu",
            vendor="example",
            model="incomplete",
            architecture="test",
            compute_dtype="fp16",
            peak_tensor_flops_per_s=1.0,
            memory_bytes=1,
            # memory_bandwidth_bytes_s is intentionally not defaulted.
        )
