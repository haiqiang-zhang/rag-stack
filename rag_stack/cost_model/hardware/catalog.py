"""Immutable hardware records and an exact, conflict-detecting catalog.

The catalog contains physical facts only.  Calibration coefficients,
deployment choices, placement, TP/PP, replica counts and scheduler settings do
not belong here.  This module intentionally depends only on the Python standard
library so calibration publishers and cost-model contracts can use it without
importing CUDA, GenZ, FAISS or the optimizer.

There are no fallback lookup APIs.  Every lookup is exact, and callers may pin
the expected record digest.  A topology's resolved digest binds the topology
record together with every CPU, GPU and interconnect record it references; it
is therefore suitable as an immutable physical-system content identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, ClassVar, Iterable, Mapping, TypeAlias



_HARDWARE_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class HardwareCatalogError(RuntimeError):
    """Base class for catalog registration and resolution failures."""


class UnknownHardwareError(KeyError, HardwareCatalogError):
    """An exact catalog key was requested but is not registered."""


class HardwareConflictError(HardwareCatalogError):
    """The same kind/key was registered with different physical facts."""


class HardwareDigestMismatchError(HardwareCatalogError):
    """A caller's pinned digest does not match the catalog record."""


class InvalidTopologyError(HardwareCatalogError):
    """A topology is internally invalid or references absent records."""


class HardwareRecordKind(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    INTERCONNECT = "interconnect"
    TOPOLOGY = "topology"


def _require_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty, trimmed string")
    return value


def _require_key(value: object, *, field: str = "hardware_key") -> str:
    key = _require_text(value, field=field)
    if _HARDWARE_KEY_RE.fullmatch(key) is None:
        raise ValueError(
            f"{field} must be a canonical lowercase ID containing only "
            "letters, digits, '.', '_' and '-'"
        )
    return key


def _require_positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _require_positive_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite positive number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{field} must be a finite positive number")
    return resolved


def _require_nonnegative_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite non-negative number")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return resolved


def _require_digest(value: object, *, field: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _canonical_value(value: Any) -> Any:
    """Convert a supported value into deterministic JSON data."""

    if isinstance(value, _CanonicalRecord):
        return value.to_manifest()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        resolved: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("canonical mappings require string keys")
            resolved[key] = _canonical_value(item)
        return resolved
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not allow NaN or infinity")
        return value
    raise TypeError(f"unsupported canonical value {type(value).__name__}")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _canonical_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for supported canonical JSON data."""

    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


class _CanonicalRecord:
    RECORD_KIND: ClassVar[str]

    def to_manifest(self) -> dict[str, Any]:
        result = {"record_kind": self.RECORD_KIND}
        for item in fields(self):
            result[item.name] = _canonical_value(getattr(self, item.name))
        return result

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_manifest())


@dataclass(frozen=True, slots=True)
class CacheLevelRecord(_CanonicalRecord):
    """One explicit CPU cache tier; all bandwidths are sustained facts."""

    RECORD_KIND: ClassVar[str] = "cpu_cache_level"

    name: str
    size_bytes: int
    instance_count: int
    cores_per_instance: int
    bandwidth_bytes_s: float
    single_thread_bandwidth_bytes_s: float
    latency_s: float

    def __post_init__(self) -> None:
        _require_text(self.name, field="cache.name")
        _require_positive_int(self.size_bytes, field="cache.size_bytes")
        _require_positive_int(self.instance_count, field="cache.instance_count")
        _require_positive_int(
            self.cores_per_instance, field="cache.cores_per_instance"
        )
        object.__setattr__(
            self,
            "bandwidth_bytes_s",
            _require_positive_float(
                self.bandwidth_bytes_s, field="cache.bandwidth_bytes_s"
            ),
        )
        object.__setattr__(
            self,
            "single_thread_bandwidth_bytes_s",
            _require_positive_float(
                self.single_thread_bandwidth_bytes_s,
                field="cache.single_thread_bandwidth_bytes_s",
            ),
        )
        object.__setattr__(
            self,
            "latency_s",
            _require_nonnegative_float(self.latency_s, field="cache.latency_s"),
        )


@dataclass(frozen=True, slots=True)
class CPURecord(_CanonicalRecord):
    """Physical facts for one catalogued host CPU configuration."""

    RECORD_KIND: ClassVar[str] = "cpu"

    hardware_key: str
    vendor: str
    model: str
    architecture: str
    socket_count: int
    physical_core_count: int
    logical_core_count: int
    memory_bytes: int
    peak_flops_per_s: float
    peak_int_ops_per_s: float
    memory_bandwidth_bytes_s: float
    single_thread_memory_bandwidth_bytes_s: float
    simd_width_bits: int
    cache_levels: tuple[CacheLevelRecord, ...]

    def __post_init__(self) -> None:
        _require_key(self.hardware_key)
        for field_name in ("vendor", "model", "architecture"):
            _require_text(getattr(self, field_name), field=f"cpu.{field_name}")
        _require_positive_int(self.socket_count, field="cpu.socket_count")
        _require_positive_int(
            self.physical_core_count, field="cpu.physical_core_count"
        )
        _require_positive_int(
            self.logical_core_count, field="cpu.logical_core_count"
        )
        if self.logical_core_count < self.physical_core_count:
            raise ValueError(
                "cpu.logical_core_count cannot be smaller than physical_core_count"
            )
        if self.physical_core_count < self.socket_count:
            raise ValueError(
                "cpu.physical_core_count cannot be smaller than socket_count"
            )
        _require_positive_int(self.memory_bytes, field="cpu.memory_bytes")
        _require_positive_int(self.simd_width_bits, field="cpu.simd_width_bits")
        for field_name in (
            "peak_flops_per_s",
            "peak_int_ops_per_s",
            "memory_bandwidth_bytes_s",
            "single_thread_memory_bandwidth_bytes_s",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_positive_float(
                    getattr(self, field_name), field=f"cpu.{field_name}"
                ),
            )
        cache_levels = tuple(self.cache_levels)
        if any(not isinstance(level, CacheLevelRecord) for level in cache_levels):
            raise TypeError("cpu.cache_levels must contain CacheLevelRecord values")
        names = [level.name for level in cache_levels]
        if len(names) != len(set(names)):
            raise ValueError("cpu.cache_levels contains duplicate cache names")
        object.__setattr__(
            self, "cache_levels", tuple(sorted(cache_levels, key=lambda item: item.name))
        )


@dataclass(frozen=True, slots=True)
class GPURecord(_CanonicalRecord):
    """Physical facts for one GPU model; topology declares device count."""

    RECORD_KIND: ClassVar[str] = "gpu"

    hardware_key: str
    vendor: str
    model: str
    architecture: str
    compute_dtype: str
    peak_tensor_flops_per_s: float
    memory_bytes: int
    memory_bandwidth_bytes_s: float

    def __post_init__(self) -> None:
        _require_key(self.hardware_key)
        for field_name in (
            "vendor",
            "model",
            "architecture",
            "compute_dtype",
        ):
            _require_text(getattr(self, field_name), field=f"gpu.{field_name}")
        object.__setattr__(
            self,
            "peak_tensor_flops_per_s",
            _require_positive_float(
                self.peak_tensor_flops_per_s,
                field="gpu.peak_tensor_flops_per_s",
            ),
        )
        _require_positive_int(self.memory_bytes, field="gpu.memory_bytes")
        object.__setattr__(
            self,
            "memory_bandwidth_bytes_s",
            _require_positive_float(
                self.memory_bandwidth_bytes_s,
                field="gpu.memory_bandwidth_bytes_s",
            ),
        )


@dataclass(frozen=True, slots=True)
class InterconnectRecord(_CanonicalRecord):
    """One measured, directional payload link class."""

    RECORD_KIND: ClassVar[str] = "interconnect"

    hardware_key: str
    transport: str
    bandwidth_bytes_s: float
    latency_s: float
    directionality: str

    def __post_init__(self) -> None:
        _require_key(self.hardware_key)
        _require_text(self.transport, field="interconnect.transport")
        if self.directionality not in {
            "full_duplex",
            "half_duplex",
            "unidirectional",
        }:
            raise ValueError(
                "interconnect.directionality must be full_duplex, half_duplex "
                "or unidirectional"
            )
        object.__setattr__(
            self,
            "bandwidth_bytes_s",
            _require_positive_float(
                self.bandwidth_bytes_s, field="interconnect.bandwidth_bytes_s"
            ),
        )
        object.__setattr__(
            self,
            "latency_s",
            _require_nonnegative_float(
                self.latency_s, field="interconnect.latency_s"
            ),
        )


@dataclass(frozen=True, slots=True)
class TopologyNode(_CanonicalRecord):
    """One named CPU or GPU instance in a physical topology."""

    RECORD_KIND: ClassVar[str] = "topology_node"

    node_id: str
    kind: HardwareRecordKind
    hardware_key: str

    def __post_init__(self) -> None:
        _require_key(self.node_id, field="topology.node_id")
        try:
            kind = HardwareRecordKind(self.kind)
        except ValueError as exc:
            raise ValueError("topology node kind must be cpu or gpu") from exc
        if kind not in {HardwareRecordKind.CPU, HardwareRecordKind.GPU}:
            raise ValueError("topology node kind must be cpu or gpu")
        object.__setattr__(self, "kind", kind)
        _require_key(self.hardware_key, field="topology.node.hardware_key")


@dataclass(frozen=True, slots=True)
class TopologyLink(_CanonicalRecord):
    """A link between two named topology nodes."""

    RECORD_KIND: ClassVar[str] = "topology_link"

    source_node: str
    target_node: str
    interconnect_key: str

    def __post_init__(self) -> None:
        _require_key(self.source_node, field="topology.link.source_node")
        _require_key(self.target_node, field="topology.link.target_node")
        if self.source_node == self.target_node:
            raise ValueError("topology links cannot connect a node to itself")
        _require_key(
            self.interconnect_key, field="topology.link.interconnect_key"
        )


@dataclass(frozen=True, slots=True)
class TopologyRecord(_CanonicalRecord):
    """A complete physical system graph, with no implicit fallback links."""

    RECORD_KIND: ClassVar[str] = "topology"

    hardware_key: str
    nodes: tuple[TopologyNode, ...]
    links: tuple[TopologyLink, ...]

    def __post_init__(self) -> None:
        _require_key(self.hardware_key)
        nodes = tuple(self.nodes)
        links = tuple(self.links)
        if not nodes:
            raise ValueError("topology.nodes must contain at least one node")
        if any(not isinstance(node, TopologyNode) for node in nodes):
            raise TypeError("topology.nodes must contain TopologyNode values")
        if any(not isinstance(link, TopologyLink) for link in links):
            raise TypeError("topology.links must contain TopologyLink values")
        node_ids = [node.node_id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("topology.nodes contains duplicate node IDs")
        node_set = set(node_ids)
        for link in links:
            missing = {
                endpoint
                for endpoint in (link.source_node, link.target_node)
                if endpoint not in node_set
            }
            if missing:
                raise ValueError(
                    "topology link references unknown node(s): "
                    + ", ".join(sorted(missing))
                )
        link_keys = [
            (link.source_node, link.target_node, link.interconnect_key)
            for link in links
        ]
        if len(link_keys) != len(set(link_keys)):
            raise ValueError("topology.links contains duplicate links")
        object.__setattr__(
            self, "nodes", tuple(sorted(nodes, key=lambda item: item.node_id))
        )
        object.__setattr__(
            self,
            "links",
            tuple(
                sorted(
                    links,
                    key=lambda item: (
                        item.source_node,
                        item.target_node,
                        item.interconnect_key,
                    ),
                )
            ),
        )


HardwareRecord: TypeAlias = CPURecord | GPURecord | InterconnectRecord | TopologyRecord


@dataclass(frozen=True, slots=True)
class HardwareIdentity:
    """An exact record reference pinned to its canonical digest."""

    kind: HardwareRecordKind
    hardware_key: str
    digest: str

    def __post_init__(self) -> None:
        try:
            kind = HardwareRecordKind(self.kind)
        except ValueError as exc:
            raise ValueError(f"unknown hardware record kind {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        _require_key(self.hardware_key)
        _require_digest(self.digest, field="digest")


class HardwareCatalog:
    """Exact registry of immutable physical hardware records.

    Registration is idempotent only when kind, key and canonical content all
    match.  A conflicting redefinition is rejected and leaves the original
    record installed.  Topologies may be registered only after every record
    they reference is available.
    """

    def __init__(
        self,
        *,
        cpus: Iterable[CPURecord] = (),
        gpus: Iterable[GPURecord] = (),
        interconnects: Iterable[InterconnectRecord] = (),
        topologies: Iterable[TopologyRecord] = (),
    ) -> None:
        self._records: dict[HardwareRecordKind, dict[str, HardwareRecord]] = {
            kind: {} for kind in HardwareRecordKind
        }
        for record in cpus:
            self.register_cpu(record)
        for record in gpus:
            self.register_gpu(record)
        for record in interconnects:
            self.register_interconnect(record)
        for record in topologies:
            self.register_topology(record)

    @staticmethod
    def _kind_for(record: HardwareRecord) -> HardwareRecordKind:
        if isinstance(record, CPURecord):
            return HardwareRecordKind.CPU
        if isinstance(record, GPURecord):
            return HardwareRecordKind.GPU
        if isinstance(record, InterconnectRecord):
            return HardwareRecordKind.INTERCONNECT
        if isinstance(record, TopologyRecord):
            return HardwareRecordKind.TOPOLOGY
        raise TypeError(f"unsupported hardware record {type(record).__name__}")

    def _register(self, record: HardwareRecord) -> HardwareRecord:
        kind = self._kind_for(record)
        records = self._records[kind]
        existing = records.get(record.hardware_key)
        if existing is None:
            records[record.hardware_key] = record
            return record
        if existing.digest != record.digest:
            raise HardwareConflictError(
                f"conflicting {kind.value} hardware_key {record.hardware_key!r}: "
                f"registered digest {existing.digest}, new digest {record.digest}"
            )
        return existing

    def register(self, record: HardwareRecord) -> HardwareRecord:
        """Register any supported record, validating topology dependencies."""

        if isinstance(record, TopologyRecord):
            return self.register_topology(record)
        return self._register(record)

    def register_cpu(self, record: CPURecord) -> CPURecord:
        if not isinstance(record, CPURecord):
            raise TypeError("register_cpu requires CPURecord")
        return self._register(record)  # type: ignore[return-value]

    def register_gpu(self, record: GPURecord) -> GPURecord:
        if not isinstance(record, GPURecord):
            raise TypeError("register_gpu requires GPURecord")
        return self._register(record)  # type: ignore[return-value]

    def register_interconnect(
        self, record: InterconnectRecord
    ) -> InterconnectRecord:
        if not isinstance(record, InterconnectRecord):
            raise TypeError("register_interconnect requires InterconnectRecord")
        return self._register(record)  # type: ignore[return-value]

    def register_topology(self, record: TopologyRecord) -> TopologyRecord:
        if not isinstance(record, TopologyRecord):
            raise TypeError("register_topology requires TopologyRecord")
        try:
            for node in record.nodes:
                if node.kind is HardwareRecordKind.CPU:
                    self.require_cpu(node.hardware_key)
                else:
                    self.require_gpu(node.hardware_key)
            for link in record.links:
                self.require_interconnect(link.interconnect_key)
        except UnknownHardwareError as exc:
            raise InvalidTopologyError(
                f"topology {record.hardware_key!r} references an unregistered "
                f"hardware record: {exc.args[0]}"
            ) from exc
        return self._register(record)  # type: ignore[return-value]

    def _require(
        self,
        kind: HardwareRecordKind,
        hardware_key: str,
        *,
        expected_digest: str | None = None,
    ) -> HardwareRecord:
        key = _require_key(hardware_key)
        record = self._records[kind].get(key)
        if record is None:
            raise UnknownHardwareError(
                f"unknown {kind.value} hardware_key {key!r}; no fallback is allowed"
            )
        if expected_digest is not None:
            digest = _require_digest(expected_digest, field="expected_digest")
            if record.digest != digest:
                raise HardwareDigestMismatchError(
                    f"{kind.value} hardware_key {key!r} has digest "
                    f"{record.digest}, expected {digest}"
                )
        return record

    def require_cpu(
        self, hardware_key: str, *, expected_digest: str | None = None
    ) -> CPURecord:
        return self._require(  # type: ignore[return-value]
            HardwareRecordKind.CPU,
            hardware_key,
            expected_digest=expected_digest,
        )

    def require_gpu(
        self, hardware_key: str, *, expected_digest: str | None = None
    ) -> GPURecord:
        return self._require(  # type: ignore[return-value]
            HardwareRecordKind.GPU,
            hardware_key,
            expected_digest=expected_digest,
        )

    def require_interconnect(
        self, hardware_key: str, *, expected_digest: str | None = None
    ) -> InterconnectRecord:
        return self._require(  # type: ignore[return-value]
            HardwareRecordKind.INTERCONNECT,
            hardware_key,
            expected_digest=expected_digest,
        )

    def require_topology(
        self, hardware_key: str, *, expected_digest: str | None = None
    ) -> TopologyRecord:
        return self._require(  # type: ignore[return-value]
            HardwareRecordKind.TOPOLOGY,
            hardware_key,
            expected_digest=expected_digest,
        )

    def identity(
        self, kind: HardwareRecordKind | str, hardware_key: str
    ) -> HardwareIdentity:
        resolved_kind = HardwareRecordKind(kind)
        record = self._require(resolved_kind, hardware_key)
        return HardwareIdentity(resolved_kind, record.hardware_key, record.digest)

    def resolve(self, identity: HardwareIdentity) -> HardwareRecord:
        if not isinstance(identity, HardwareIdentity):
            raise TypeError("resolve requires HardwareIdentity")
        return self._require(
            identity.kind,
            identity.hardware_key,
            expected_digest=identity.digest,
        )

    def to_manifest(self) -> dict[str, Any]:
        manifest: dict[str, Any] = {
            kind.value: [
                records[key].to_manifest() for key in sorted(records)
            ]
            for kind, records in self._records.items()
        }
        return manifest

    @property
    def digest(self) -> str:
        """Digest of the complete catalog, including unrelated systems."""

        return canonical_digest(self.to_manifest())

    def resolved_topology_manifest(self, hardware_key: str) -> dict[str, Any]:
        """Return one topology plus the exact physical records it references."""

        topology = self.require_topology(hardware_key)
        cpu_keys = sorted(
            {
                node.hardware_key
                for node in topology.nodes
                if node.kind is HardwareRecordKind.CPU
            }
        )
        gpu_keys = sorted(
            {
                node.hardware_key
                for node in topology.nodes
                if node.kind is HardwareRecordKind.GPU
            }
        )
        link_keys = sorted({link.interconnect_key for link in topology.links})
        return {
            "topology": topology.to_manifest(),
            "cpu": [self.require_cpu(key).to_manifest() for key in cpu_keys],
            "gpu": [self.require_gpu(key).to_manifest() for key in gpu_keys],
            "interconnect": [
                self.require_interconnect(key).to_manifest() for key in link_keys
            ],
        }

    def resolved_topology_digest(self, hardware_key: str) -> str:
        """Return the canonical digest of one resolved physical system."""

        return canonical_digest(self.resolved_topology_manifest(hardware_key))
