"""Dependency-light, explicit hardware catalog contracts.

This package deliberately does not adapt legacy ``SystemConfig`` objects yet.
Callers must construct audited records explicitly; lookup never guesses a
nearby preset or supplies a default link.
"""

from .catalog import (
    CPURecord,
    CacheLevelRecord,
    GPURecord,
    HardwareCatalog,
    HardwareCatalogError,
    HardwareConflictError,
    HardwareDigestMismatchError,
    HardwareIdentity,
    HardwareRecordKind,
    InterconnectRecord,
    InvalidTopologyError,
    TopologyLink,
    TopologyNode,
    TopologyRecord,
    UnknownHardwareError,
    canonical_digest,
)

__all__ = [
    "CPURecord",
    "CacheLevelRecord",
    "GPURecord",
    "HardwareCatalog",
    "HardwareCatalogError",
    "HardwareConflictError",
    "HardwareDigestMismatchError",
    "HardwareIdentity",
    "HardwareRecordKind",
    "InterconnectRecord",
    "InvalidTopologyError",
    "TopologyLink",
    "TopologyNode",
    "TopologyRecord",
    "UnknownHardwareError",
    "canonical_digest",
]
