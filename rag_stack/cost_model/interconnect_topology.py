"""Pair-aware GPU interconnect helpers.

The legacy CM accepts one homogeneous ``system.interconnect.gpu_gpu`` link. That
is enough for full-mesh systems, but not for boxes such as 4x H100 NVL where
GPU0-GPU1 and GPU2-GPU3 are NVLink pairs while cross-pair traffic goes through
PCIe/PXB. This module keeps the old homogeneous behavior as the default and
uses optional per-link topology when present.
"""
from __future__ import annotations

from itertools import combinations, product
from typing import Any, Iterable

_DEFAULT_ICN_GBS = 150.0
_DEFAULT_ICN_LL_US = 1.0


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return {"type": value}
    return {}


def _norm_device(dev: Any) -> str:
    s = str(dev).strip()
    if not s:
        return s
    if s.isdigit():
        return f"cuda:{s}"
    if s.startswith("GPU"):
        rest = s[3:]
        if rest.isdigit():
            return f"cuda:{rest}"
    return s


def _devices(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",")]
    return [_norm_device(v) for v in value if str(v).strip()]


def _gbps_us(entry: dict[str, Any], default_gbps: float, default_us: float) -> tuple[float, float]:
    bw = entry.get("bandwidth_gbps", default_gbps)
    lat = entry.get("latency_us", default_us)
    return float(bw), float(lat)


def _base_gpu_link(
    system_or_hw: Any,
    default_gbps: float | None = None,
    default_us: float | None = None,
) -> tuple[dict[str, Any], float, float]:
    if isinstance(system_or_hw, dict):
        interconnect = system_or_hw.get("interconnect") or {}
        gpu = system_or_hw.get("gpu")
    else:
        interconnect = getattr(system_or_hw, "interconnect", None) or {}
        gpu = getattr(system_or_hw, "gpu", None)

    if default_gbps is None:
        default_gbps = _DEFAULT_ICN_GBS
        if isinstance(gpu, dict) and gpu.get("ICN") is not None:
            default_gbps = float(gpu["ICN"])
    if default_us is None:
        default_us = _DEFAULT_ICN_LL_US
        if isinstance(gpu, dict) and gpu.get("ICN_LL") is not None:
            default_us = float(gpu["ICN_LL"])

    gg = _as_dict((interconnect or {}).get("gpu_gpu"))
    base_gbps, base_us = _gbps_us(gg, float(default_gbps), float(default_us))
    return gg, base_gbps, base_us


def _pair_key(a: str, b: str) -> frozenset[str]:
    return frozenset((_norm_device(a), _norm_device(b)))


def _add_pair(
    out: dict[frozenset[str], tuple[float, float]],
    a: str,
    b: str,
    entry: dict[str, Any],
    default_gbps: float,
    default_us: float,
) -> None:
    aa, bb = _norm_device(a), _norm_device(b)
    if not aa or not bb or aa == bb:
        return
    out[_pair_key(aa, bb)] = _gbps_us(entry, default_gbps, default_us)


def _explicit_pairs(gg: dict[str, Any], default_gbps: float, default_us: float) -> dict[frozenset[str], tuple[float, float]]:
    pairs: dict[frozenset[str], tuple[float, float]] = {}

    raw_pairs = gg.get("pairs")
    if isinstance(raw_pairs, dict):
        for key, value in raw_pairs.items():
            devs = _devices(key)
            if len(devs) != 2:
                continue
            _add_pair(pairs, devs[0], devs[1], _as_dict(value), default_gbps, default_us)
    elif isinstance(raw_pairs, list):
        for item in raw_pairs:
            entry = _as_dict(item)
            devs = _devices(entry.get("devices"))
            if len(devs) == 2:
                _add_pair(pairs, devs[0], devs[1], entry, default_gbps, default_us)

    for key in ("links", "groups"):
        raw = gg.get(key)
        if not isinstance(raw, list):
            continue
        for item in raw:
            entry = _as_dict(item)
            src = _devices(entry.get("from") or entry.get("src") or entry.get("source"))
            dst = _devices(entry.get("to") or entry.get("dst") or entry.get("target"))
            if src and dst:
                for a, b in product(src, dst):
                    _add_pair(pairs, a, b, entry, default_gbps, default_us)
                continue
            devs = _devices(entry.get("devices"))
            for a, b in combinations(devs, 2):
                _add_pair(pairs, a, b, entry, default_gbps, default_us)

    return pairs


def _components_from_explicit_links(
    devices: list[str],
    explicit: dict[frozenset[str], tuple[float, float]],
) -> list[list[str]]:
    parent = {d: d for d in devices}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    devset = set(devices)
    for pair in explicit:
        vals = [d for d in pair if d in devset]
        if len(vals) == 2:
            union(vals[0], vals[1])

    out: dict[str, list[str]] = {}
    for dev in devices:
        out.setdefault(find(dev), []).append(dev)
    return list(out.values())


def _lookup_pair_link(
    explicit: dict[frozenset[str], tuple[float, float]],
    fallback_gbps: float,
    fallback_us: float,
    a: str,
    b: str,
) -> tuple[float, float]:
    return explicit.get(_pair_key(a, b), (fallback_gbps, fallback_us))


def _aggregate_transfer_gbps_us(
    system_or_hw: Any,
    left: list[str],
    right: list[str],
    *,
    default_gbps: float,
    default_us: float,
) -> tuple[float, float]:
    """Approximate bisection bandwidth between two GPU groups.

    For LLM collectives/stage transfers, multiple GPU pairs can communicate
    across the same topology cut. A 2+2 H100 NVL box therefore has about two
    concurrent PCIe/PXB cross-pair lanes for a 2-GPU island talking to the other
    2-GPU island, not just the single slowest pair. We conservatively sum the
    strongest disjoint pair lanes across the cut and keep worst-pair latency.
    """
    gg, base_gbps, base_us = _base_gpu_link(system_or_hw, default_gbps, default_us)
    fallback = _as_dict(gg.get("fallback") or gg.get("default"))
    fallback_gbps, fallback_us = _gbps_us(fallback, base_gbps, base_us)
    explicit = _explicit_pairs(gg, base_gbps, base_us)
    pairs = [
        _lookup_pair_link(explicit, fallback_gbps, fallback_us, a, b)
        for a, b in product(left, right)
        if _norm_device(a) != _norm_device(b)
    ]
    if not pairs:
        return base_gbps, base_us
    lanes = max(1, min(len(left), len(right)))
    bws = sorted((bw for bw, _ in pairs), reverse=True)[:lanes]
    return sum(bws), max(lat for _, lat in pairs)


def _aggregate_collective_cut_gbps_us(
    system_or_hw: Any,
    left: list[str],
    right: list[str],
    *,
    default_gbps: float,
    default_us: float,
) -> tuple[float, float]:
    """Approximate the cross-island bisection seen by a TP collective.

    Point-to-point transfers are limited by disjoint endpoint pairs, but NCCL
    collectives can pipeline reduce-scatter/all-gather traffic around the ring.
    On a 2+2 H100 NVL topology this exposes roughly three PCIe/PXB-equivalent
    cross-pair lanes to GenZ rather than two strictly disjoint transfers. Keep
    the ordinary transfer model conservative for PD/PP handoffs.
    """
    gg, base_gbps, base_us = _base_gpu_link(system_or_hw, default_gbps, default_us)
    fallback = _as_dict(gg.get("fallback") or gg.get("default"))
    fallback_gbps, fallback_us = _gbps_us(fallback, base_gbps, base_us)
    explicit = _explicit_pairs(gg, base_gbps, base_us)
    pairs = [
        _lookup_pair_link(explicit, fallback_gbps, fallback_us, a, b)
        for a, b in product(left, right)
        if _norm_device(a) != _norm_device(b)
    ]
    if not pairs:
        return base_gbps, base_us
    if len(left) > 1 and len(right) > 1:
        lanes = min(len(pairs), len(left) + len(right) - 1)
    else:
        lanes = 1
    bws = sorted((bw for bw, _ in pairs), reverse=True)[:lanes]
    return sum(bws), max(lat for _, lat in pairs)


def _aggregate_group_collective_gbps_us(
    system_or_hw: Any,
    devices: list[str],
    *,
    default_gbps: float,
    default_us: float,
) -> tuple[float, float]:
    gg, base_gbps, base_us = _base_gpu_link(system_or_hw, default_gbps, default_us)
    fallback = _as_dict(gg.get("fallback") or gg.get("default"))
    fallback_gbps, fallback_us = _gbps_us(fallback, base_gbps, base_us)
    explicit = _explicit_pairs(gg, base_gbps, base_us)
    comps = _components_from_explicit_links(devices, explicit)
    if len(comps) <= 1:
        return effective_gpu_link_gbps_us(
            system_or_hw, devices, default_gbps=base_gbps, default_us=base_us
        )
    candidates = [
        effective_gpu_link_gbps_us(
            system_or_hw, comp, default_gbps=base_gbps, default_us=base_us
        )
        for comp in comps
        if len(comp) > 1
    ]
    candidates.extend(
        _aggregate_collective_cut_gbps_us(
            system_or_hw,
            a,
            b,
            default_gbps=base_gbps,
            default_us=base_us,
        )
        for a, b in combinations(comps, 2)
    )
    if not candidates:
        return fallback_gbps, fallback_us
    return min(bw for bw, _ in candidates), max(lat for _, lat in candidates)


def has_gpu_topology(system_or_hw: Any) -> bool:
    """Return true when ``gpu_gpu`` carries per-pair/group topology metadata."""
    gg, _, _ = _base_gpu_link(system_or_hw)
    return any(k in gg for k in ("pairs", "links", "groups", "fallback", "default"))


def topology_gpu_devices(system_or_hw: Any) -> list[str]:
    """Ordered device inventory explicitly named by the GPU topology."""
    gg, _, _ = _base_gpu_link(system_or_hw)
    found: list[str] = []

    def _add(values: Any) -> None:
        for device in _devices(values):
            if device and device not in found:
                found.append(device)

    raw_pairs = gg.get("pairs")
    if isinstance(raw_pairs, dict):
        for key in raw_pairs:
            _add(key)
    elif isinstance(raw_pairs, list):
        for entry in raw_pairs:
            if isinstance(entry, dict):
                _add(entry.get("devices"))
                _add(entry.get("from") or entry.get("src") or entry.get("source"))
                _add(entry.get("to") or entry.get("dst") or entry.get("target"))

    for key in ("links", "groups"):
        for entry in gg.get(key, []) or []:
            if not isinstance(entry, dict):
                continue
            _add(entry.get("devices"))
            _add(entry.get("from") or entry.get("src") or entry.get("source"))
            _add(entry.get("to") or entry.get("dst") or entry.get("target"))
    return found


def available_gpu_devices(
    system_or_hw: Any,
    *,
    required_count: int | None = None,
) -> list[str]:
    """Resolve the physical GPU inventory without inspecting the local host.

    This is deliberately config-only: CM may run on a 3090 host while pricing a
    4xH100 target.  A heterogeneous topology must name its devices either in an
    explicit available-device list or in its links; it never falls back to a
    fabricated homogeneous inventory.
    """
    explicit: list[str] = []
    if isinstance(system_or_hw, dict):
        system = system_or_hw
        hardware = system.get("hardware") or {}
        design = system.get("system_design_space") or {}
        explicit = _devices(
            system.get("available_gpus")
            or (hardware.get("available_gpus") if isinstance(hardware, dict) else None)
            or (design.get("available_gpus") if isinstance(design, dict) else None)
        )
        search = system.get("cm_search_space") or system.get("search_space") or {}
    else:
        system = system_or_hw
        search = getattr(system_or_hw, "search_space", None) or {}

    topology_devices = topology_gpu_devices(system_or_hw)
    devices = explicit or topology_devices
    if explicit:
        missing = [device for device in topology_devices if device not in explicit]
        if missing:
            raise ValueError(
                "GPU topology names devices outside the declared available inventory: "
                f"{missing} not in {explicit}"
            )

    if required_count is None:
        for key in ("gpus_per_server", "max_num_gpus"):
            raw = search.get(key) if isinstance(search, dict) else None
            if raw is not None:
                if isinstance(raw, list):
                    raw = max(raw)
                required_count = int(raw)
                break
    required = max(0, int(required_count or 0))

    if has_gpu_topology(system_or_hw) and not devices:
        raise ValueError(
            "system.interconnect.gpu_gpu declares a heterogeneous topology but "
            "names no devices and no available_gpus inventory is present"
        )
    if devices and required > len(devices):
        raise ValueError(
            f"CM target requires {required} GPUs but its declared inventory has "
            f"only {len(devices)}: {devices}"
        )
    if not devices:
        # Homogeneous legacy configs may omit physical names.  The identities
        # are abstract but deterministic and never derived from torch/CUDA.
        devices = [f"cuda:{index}" for index in range(max(1, required))]
    return devices


def rago_device_topology(
    system_or_hw: Any,
    *,
    required_count: int | None = None,
) -> tuple[list[str], dict[tuple[str, str], tuple[float, float]]]:
    """RAGO input contract: device universe plus unordered link classes."""
    devices = available_gpu_devices(system_or_hw, required_count=required_count)
    if not has_gpu_topology(system_or_hw):
        return devices, {}
    links = {
        tuple(sorted((left, right))): effective_gpu_link_gbps_us(
            system_or_hw, [left], [right]
        )
        for left, right in combinations(devices, 2)
    }
    return devices, links


def effective_gpu_link_gbps_us(
    system_or_hw: Any,
    devices: Iterable[Any],
    peer_devices: Iterable[Any] | None = None,
    *,
    default_gbps: float | None = None,
    default_us: float | None = None,
) -> tuple[float, float]:
    """Effective GPU-GPU link for one device group or a cross-group transfer.

    Without topology metadata this returns the homogeneous ``gpu_gpu`` link. With
    topology metadata, every required pair is looked up and the effective link is
    the bottleneck bandwidth plus the worst pair latency. Missing pairs use
    ``gpu_gpu.fallback``/``gpu_gpu.default`` when declared, otherwise the
    homogeneous ``gpu_gpu`` value.
    """
    gg, base_gbps, base_us = _base_gpu_link(system_or_hw, default_gbps, default_us)
    left = _devices(list(devices))
    right = _devices(list(peer_devices)) if peer_devices is not None else []
    if peer_devices is None:
        needed = [_pair_key(a, b) for a, b in combinations(left, 2)]
    else:
        needed = [_pair_key(a, b) for a, b in product(left, right) if _norm_device(a) != _norm_device(b)]
    if not needed:
        return base_gbps, base_us

    fallback = _as_dict(gg.get("fallback") or gg.get("default"))
    fallback_gbps, fallback_us = _gbps_us(fallback, base_gbps, base_us)
    explicit = _explicit_pairs(gg, base_gbps, base_us)

    links = [explicit.get(pair, (fallback_gbps, fallback_us)) for pair in needed]
    return min(bw for bw, _ in links), max(lat for _, lat in links)


def effective_gpu_link_bytes_s(
    system_or_hw: Any,
    devices: Iterable[Any],
    peer_devices: Iterable[Any] | None = None,
    *,
    default_gbps: float | None = None,
    default_us: float | None = None,
) -> tuple[float, float]:
    """Like :func:`effective_gpu_link_gbps_us`, returning bytes/s and seconds."""
    gbps, us = effective_gpu_link_gbps_us(
        system_or_hw,
        devices,
        peer_devices,
        default_gbps=default_gbps,
        default_us=default_us,
    )
    return gbps * 1e9, us * 1e-6


def effective_sharded_gpu_transfer_gbps_us(
    system_or_hw: Any,
    source_devices: Iterable[Any],
    target_devices: Iterable[Any],
    *,
    default_gbps: float | None = None,
    default_us: float | None = None,
) -> tuple[float, float]:
    """Effective bandwidth for one sharded tensor/KV transfer.

    ``effective_gpu_link_gbps_us`` intentionally returns the worst single link;
    that is appropriate for a scalar dependency, but not for a TP-sharded KV
    cache whose rank lanes transfer concurrently.  This helper makes the shard
    contract explicit:

    * equal source/target widths use rank-aligned flows (rank ``i`` -> rank
      ``i``), each carrying ``1 / width`` of the full payload;
    * asymmetric widths require reshaping the shards, represented as an
      all-to-all flow matrix where each edge carries
      ``1 / (source_width * target_width)`` of the payload;
    * flows sharing a source or target endpoint serialize.  Independent
      endpoint lanes run concurrently, so completion time is the maximum
      normalized work on any source or target.  This is conservative for an
      asymmetric fast+slow cut and never substitutes the strongest link for a
      required slow-rank flow.

    A homogeneous fabric keeps the historical single-bandwidth interpretation;
    only an explicitly declared topology enables rank-lane aggregation.
    """
    gg, base_gbps, base_us = _base_gpu_link(
        system_or_hw, default_gbps, default_us,
    )
    source = _devices(list(source_devices))
    target = _devices(list(target_devices))
    if not source or not target or not has_gpu_topology(system_or_hw):
        return base_gbps, base_us

    fallback = _as_dict(gg.get("fallback") or gg.get("default"))
    fallback_gbps, fallback_us = _gbps_us(fallback, base_gbps, base_us)
    explicit = _explicit_pairs(gg, base_gbps, base_us)

    if len(source) == len(target):
        flows = [
            (src, dst, 1.0 / len(source))
            for src, dst in zip(source, target)
        ]
    else:
        fraction = 1.0 / (len(source) * len(target))
        flows = [
            (src, dst, fraction)
            for src, dst in product(source, target)
        ]

    source_work = {device: 0.0 for device in source}
    target_work = {device: 0.0 for device in target}
    latencies: list[float] = []
    network_fraction = 0.0
    for src, dst, fraction in flows:
        if _norm_device(src) == _norm_device(dst):
            continue
        bandwidth, latency = _lookup_pair_link(
            explicit, fallback_gbps, fallback_us, src, dst,
        )
        if bandwidth <= 0:
            return 0.0, max(latency, fallback_us)
        work = fraction / bandwidth
        source_work[src] += work
        target_work[dst] += work
        network_fraction += fraction
        latencies.append(latency)

    if network_fraction <= 0:
        return base_gbps, base_us
    completion_per_full_byte = max(
        max(source_work.values(), default=0.0),
        max(target_work.values(), default=0.0),
    )
    if completion_per_full_byte <= 0:
        return base_gbps, max(latencies, default=base_us)
    return 1.0 / completion_per_full_byte, max(latencies, default=base_us)


def effective_sharded_gpu_transfer_bytes_s(
    system_or_hw: Any,
    source_devices: Iterable[Any],
    target_devices: Iterable[Any],
    *,
    default_gbps: float | None = None,
    default_us: float | None = None,
) -> tuple[float, float]:
    """Sharded-transfer bandwidth in bytes/s and latency in seconds."""
    gbps, us = effective_sharded_gpu_transfer_gbps_us(
        system_or_hw,
        source_devices,
        target_devices,
        default_gbps=default_gbps,
        default_us=default_us,
    )
    return gbps * 1e9, us * 1e-6


def effective_llm_parallel_link_gbps_us(
    system_or_hw: Any,
    devices: Iterable[Any],
    *,
    tp: int,
    pp: int,
    default_gbps: float | None = None,
    default_us: float | None = None,
) -> tuple[float, float]:
    """Effective GenZ ICN for a concrete TP/PP placement.

    Devices are interpreted in vLLM order: contiguous chunks of size ``tp`` form
    one tensor-parallel group per pipeline stage. The result covers intra-TP
    collectives plus adjacent PP stage transfers, using the slowest required
    physical path. If there is no declared topology, this is the legacy global
    link.
    """
    gg, base_gbps, base_us = _base_gpu_link(system_or_hw, default_gbps, default_us)
    devs = _devices(list(devices))
    tp = max(1, int(tp))
    pp = max(1, int(pp))
    if not has_gpu_topology(system_or_hw) or len(devs) <= 1:
        return base_gbps, base_us
    if len(devs) < tp * pp:
        return effective_gpu_link_gbps_us(
            system_or_hw, devs, default_gbps=base_gbps, default_us=base_us
        )

    chunks = [devs[i * tp:(i + 1) * tp] for i in range(pp)]
    candidates: list[tuple[float, float]] = []
    if tp > 1:
        candidates.extend(
            _aggregate_group_collective_gbps_us(
                system_or_hw, chunk, default_gbps=base_gbps, default_us=base_us
            )
            for chunk in chunks
        )
    if pp > 1:
        candidates.extend(
            _aggregate_transfer_gbps_us(
                system_or_hw, a, b, default_gbps=base_gbps, default_us=base_us
            )
            for a, b in zip(chunks, chunks[1:])
        )
    if not candidates:
        return base_gbps, base_us
    return min(bw for bw, _ in candidates), max(lat for _, lat in candidates)
