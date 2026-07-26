"""Concrete physical layout contract for CM candidate rows.

RAGO owns chip-group → physical-device materialization and emits
``stage_devices``/``resource_group_devices``.  This module qualifies that
placement into the resolved-layout vocabulary consumed by GenZ,
communication, and the closed-loop simulator.  The stable row column is
``candidate_system_layout`` (a dict in memory, JSON after CSV round-trips).
"""
from __future__ import annotations

import ast
import copy
import json
from typing import Any, Mapping

import pandas as pd

CANDIDATE_LAYOUT_COLUMN = "candidate_system_layout"
_CPU_RETRIEVAL_STAGES = frozenset({
    "semantic_retrieval_vectorsearch", "retrieval_fusion",
})


def _parse(value: Any, *, expected: type | tuple[type, ...]) -> Any:
    if isinstance(value, expected):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Cannot parse candidate layout value: {value!r}") from exc
        if isinstance(parsed, expected):
            return parsed
    raise TypeError(f"Expected {expected}, got {type(value).__name__}")


def candidate_system_layout_from_row(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    """Read and minimally validate the stable per-candidate layout column."""
    value = row.get(CANDIDATE_LAYOUT_COLUMN)
    layout = _parse(value, expected=dict)
    for required in ("available_devices", "resource_groups", "performance_stages", "engines"):
        if required not in layout:
            raise KeyError(f"candidate_system_layout[{required!r}] is required")
    return layout


def _groups(value: Any) -> list[list[str]]:
    if value is None:
        return []
    return [list(group) for group in _parse(value, expected=(list, tuple))]


def _stage_devices(value: Any) -> dict[str, list[str]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    raw = _parse(value, expected=dict)
    return {
        str(stage): [str(device) for device in devices]
        for stage, devices in raw.items()
    }


def _winning_tp_pp(
    row: Mapping[str, Any],
    stage: str,
    devices: list[str],
    sweep_dfs: Mapping[str, pd.DataFrame] | None,
) -> tuple[int, int, int]:
    n = max(1, len(devices))
    sdf = (sweep_dfs or {}).get(stage)
    if sdf is None or not len(sdf) or "winning_tp" not in sdf.columns:
        return n, 1, 1
    nc_col = (
        "hardware.num_chips" if "hardware.num_chips" in sdf.columns
        else "num_chips" if "num_chips" in sdf.columns else None
    )
    sub = sdf[sdf[nc_col] == n] if nc_col else sdf
    batch_col = f"batch_size_{stage}"
    if len(sub) and batch_col in row and "batch_size" in sub.columns:
        exact = sub[sub["batch_size"] == int(row[batch_col])]
        if len(exact):
            sub = exact
    if not len(sub):
        return n, 1, 1
    picked = sub.loc[sub["batch_size"].idxmax()] if "batch_size" in sub.columns else sub.iloc[0]
    tp = max(1, int(picked["winning_tp"]))
    pp = max(1, int(picked.get("winning_pp", 1)))
    width = tp * pp
    if width > n or n % width:
        raise ValueError(
            f"Stage {stage!r} winner TP*PP={tp}*{pp} does not divide devices={n}"
        )
    return tp, pp, n // width


def _fixed_parallelism(layout: dict[str, Any], stage: str) -> tuple[int, int, int] | None:
    from rag_stack.taxonomy import TAXONOMY

    if stage in _CPU_RETRIEVAL_STAGES:
        return None
    perf = layout.get("performance_stages") or {}
    info = perf.get(stage) or {}
    engine = info.get("engine")
    if not engine:
        try:
            engine = TAXONOMY.user_stage_of(stage).value
        except (KeyError, ValueError):
            return None
    eng = (layout.get("engines") or {}).get(engine) or {}
    record = eng
    if eng.get("pd_serving") == "disagg_pd":
        record = eng.get(stage) or {}
    if "tp" not in record or "pp" not in record:
        try:
            kinds = {
                ps.kind for ps in TAXONOMY.performance_stages_of(engine)
            }
        except (KeyError, ValueError):
            return None
        if not {"prefill", "decode"}.issubset(kinds):
            return None
        devices = [str(device) for device in info.get("devices", [])]
        return (len(devices), 1, 1) if devices else None
    devices = [str(device) for device in info.get("devices", [])]
    tp, pp = max(1, int(record["tp"])), max(1, int(record["pp"]))
    replicas = max(1, len(devices) // max(1, tp * pp))
    return tp, pp, replicas


def _enrich_resolved_layout(system_config: dict[str, Any]) -> dict[str, Any]:
    layout = copy.deepcopy(system_config["layout"])
    available = list(((system_config.get("hardware") or {}).get("available_gpus") or []))
    if not available:
        from rag_stack.cost_model.interconnect_topology import available_gpu_devices
        available = available_gpu_devices(system_config)
    layout["source"] = "resolved_system_layout"
    layout["available_devices"] = [str(device) for device in available]

    from rag_stack.taxonomy import TAXONOMY
    parallel_updates: list[tuple[str, int, int]] = []
    for stage, info in (layout.get("performance_stages") or {}).items():
        if not isinstance(info, dict):
            continue
        try:
            ps = TAXONOMY.performance_stage(stage)
            info.setdefault("role", ps.kind)
            info.setdefault("engine", TAXONOMY.user_stage_of(stage).value)
        except (KeyError, ValueError):
            pass
        parallel = _fixed_parallelism(layout, stage)
        if parallel is not None:
            info["tp"], info["pp"], info["replicas"] = parallel
            width = info["tp"] * info["pp"]
            devices = list(info.get("devices") or [])
            info["replica_devices"] = [
                devices[index:index + width]
                for index in range(0, len(devices), width)
            ]
            parallel_updates.append((stage, int(info["tp"]), int(info["pp"])))
    for stage, tp, pp in parallel_updates:
        update_candidate_parallelism(layout, stage, tp=tp, pp=pp)
    return layout


def _layout_from_rago_row(
    row: Mapping[str, Any],
    *,
    sweep_dfs: Mapping[str, pd.DataFrame] | None,
) -> dict[str, Any]:
    from rag_stack.taxonomy import TAXONOMY

    groups = _groups(row.get("collocation_strategy"))
    devices_by_stage = _stage_devices(row.get("stage_devices"))
    available = row.get("available_devices") or []
    if isinstance(available, str):
        available = _parse(available, expected=(list, tuple))
    available = [str(device) for device in available]

    performance_stages: dict[str, dict[str, Any]] = {}
    resource_groups: list[dict[str, Any]] = []
    stage_to_resource: dict[str, str] = {}
    for index, group in enumerate(groups):
        is_cpu = bool(group) and all(stage in _CPU_RETRIEVAL_STAGES for stage in group)
        if is_cpu:
            resource_type = (
                "vectorsearch" if "semantic_retrieval_vectorsearch" in group
                else "fusion"
            )
        else:
            resource_type = "gpu"
        resource_id = f"{resource_type}:{index}"
        group_devices: list[str] = []
        for stage in group:
            for device in devices_by_stage.get(stage, []):
                if device not in group_devices:
                    group_devices.append(device)
        resource = {
            "id": resource_id,
            "kind": "cpu_retrieval" if is_cpu else "gpu",
            "devices": [] if is_cpu else group_devices,
            "performance_stages": list(group),
        }
        if is_cpu and "semantic_retrieval_vectorsearch" in group:
            resource["num_servers"] = max(1, int(row.get("num_retrieval_servers", 1) or 1))
        resource_groups.append(resource)
        for stage in group:
            stage_to_resource[stage] = resource_id

    order = row.get("pipeline_stage_order") or [stage for group in groups for stage in group]
    if isinstance(order, str):
        order = _parse(order, expected=(list, tuple))
    order = [str(stage) for stage in order]

    for stage in order:
        devices = list(devices_by_stage.get(stage, []))
        is_cpu = stage in _CPU_RETRIEVAL_STAGES
        try:
            ps = TAXONOMY.performance_stage(stage)
            engine = TAXONOMY.user_stage_of(stage).value
            role = ps.kind
        except (KeyError, ValueError):
            engine, role = stage, "stage"
        record: dict[str, Any] = {
            "kind": "cpu_retrieval" if is_cpu else "gpu",
            "resource": stage_to_resource.get(stage),
            "devices": [] if is_cpu else devices,
            "num_chips": 0 if is_cpu else len(devices),
            "engine": engine,
            "role": role,
        }
        if is_cpu:
            if stage == "semantic_retrieval_vectorsearch":
                record["num_servers"] = max(1, int(row.get("num_retrieval_servers", 1) or 1))
        elif stage in TAXONOMY.llm_performance_stages:
            tp, pp, replicas = _winning_tp_pp(row, stage, devices, sweep_dfs)
            record.update({"tp": tp, "pp": pp, "replicas": replicas})
            width = tp * pp
            record["replica_devices"] = [
                devices[index:index + width]
                for index in range(0, len(devices), width)
            ]
        else:
            # Taxonomy kind="prefill" also represents one-shot encoder
            # forwards.  Only concrete autoregressive LLM stages use TP/PP;
            # RAGO scales every other GPU stage as independent DP replicas.
            record.update({"tp": 1, "pp": 1, "replicas": len(devices)})
            record["replica_devices"] = [[device] for device in devices]
        performance_stages[stage] = record

    engines: dict[str, dict[str, Any]] = {}
    active_engines = sorted({
        info["engine"] for info in performance_stages.values()
        if info.get("engine")
    })
    for engine in active_engines:
        owned = [
            stage for stage in order
            if (performance_stages.get(stage) or {}).get("engine") == engine
        ]
        gpu_owned = [stage for stage in owned if performance_stages[stage]["kind"] == "gpu"]
        if not gpu_owned:
            continue
        pre = next((stage for stage in gpu_owned if performance_stages[stage]["role"] == "prefill"), None)
        dec = next((stage for stage in gpu_owned if performance_stages[stage]["role"] == "decode"), None)
        disagg = bool(
            pre and dec
            and performance_stages[pre]["resource"] != performance_stages[dec]["resource"]
        )
        all_devices: list[str] = []
        for stage in gpu_owned:
            for device in performance_stages[stage]["devices"]:
                if device not in all_devices:
                    all_devices.append(device)
        entry: dict[str, Any] = {
            "pd_serving": "disagg_pd" if disagg else "collocated_pd",
            "devices": all_devices,
            "num_chips": len(all_devices),
            "performance_stages": owned,
        }
        if disagg:
            for stage in (pre, dec):
                if stage:
                    info = performance_stages[stage]
                    entry[stage] = {
                        key: copy.deepcopy(info[key])
                        for key in ("devices", "num_chips", "tp", "pp", "replicas", "replica_devices")
                        if key in info
                    }
        elif pre or dec:
            # Unified vLLM must use one TP/PP.  The topology-aware repricer
            # resolves a common winner and overwrites this provisional value.
            info = performance_stages[dec or pre]
            for key in ("tp", "pp", "replicas", "replica_devices"):
                if key in info:
                    entry[key] = copy.deepcopy(info[key])
        engines[engine] = entry

    gpu_occupants: dict[str, list[str]] = {}
    for stage, info in performance_stages.items():
        for device in info.get("devices", []):
            gpu_occupants.setdefault(device, []).append(stage)

    return {
        "source": "rago_physical_mapping",
        "device_layout_id": row.get("device_layout_id"),
        "available_devices": available,
        "performance_stage_order": order,
        "total_gpu_slots": int(row.get("num_chips_e2e", 0) or 0),
        "resource_groups": resource_groups,
        "performance_stages": performance_stages,
        "engines": engines,
        "gpu_occupants": gpu_occupants,
    }


def _with_legacy_homogeneous_devices(
    row: Mapping[str, Any],
    system: dict[str, Any],
) -> dict[str, Any]:
    """Materialize old fixed callers only when their fabric is homogeneous."""
    from rag_stack.cost_model.interconnect_topology import (
        available_gpu_devices,
        has_gpu_topology,
    )

    if has_gpu_topology(system):
        raise ValueError(
            "Fixed placement on a heterogeneous GPU topology requires a resolved "
            "system.layout (or explicit stage_devices); refusing to guess devices."
        )
    groups = _groups(row.get("collocation_strategy"))
    needed = max(1, int(row.get("num_chips_e2e", 1) or 1))
    available = available_gpu_devices(system, required_count=needed)
    cursor = 0
    stage_devices: dict[str, list[str]] = {}
    group_devices: list[list[str]] = []
    for group in groups:
        size = max((
            int(row.get(f"num_chips_{stage}", 0) or 0)
            for stage in group
            if stage not in _CPU_RETRIEVAL_STAGES
        ), default=0)
        assigned = list(available[cursor:cursor + size])
        cursor += size
        group_devices.append(assigned)
        for stage in group:
            if stage in _CPU_RETRIEVAL_STAGES:
                continue
            n = int(row.get(f"num_chips_{stage}", size) or size)
            stage_devices[stage] = list(assigned[:n])
    enriched = dict(row)
    enriched["available_devices"] = available
    enriched["resource_group_devices"] = group_devices
    enriched["stage_devices"] = stage_devices
    enriched["device_layout_id"] = "fixed_homogeneous"
    return enriched


def materialize_candidate_system_layouts(
    df: pd.DataFrame,
    *,
    pipeline_config: dict[str, Any],
    sweep_dfs: Mapping[str, pd.DataFrame] | None = None,
    prefer_system_layout: bool = False,
) -> pd.DataFrame:
    """Attach the stable concrete-layout dict and flat provenance columns."""
    if df is None or not len(df):
        return df
    system = (pipeline_config or {}).get("system") or {}
    fixed_layout = system.get("layout") if isinstance(system, dict) else None
    out = df.copy()
    layouts: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        if prefer_system_layout and isinstance(fixed_layout, dict):
            layout = _enrich_resolved_layout(system)
        else:
            row_value: Mapping[str, Any] = row
            if prefer_system_layout and not _stage_devices(row.get("stage_devices")):
                row_value = _with_legacy_homogeneous_devices(row, system)
            layout = _layout_from_rago_row(row_value, sweep_dfs=sweep_dfs)
        layouts.append(layout)
    out[CANDIDATE_LAYOUT_COLUMN] = pd.Series(layouts, index=out.index, dtype=object)

    stages = sorted({
        stage for layout in layouts
        for stage in layout.get("performance_stages", {})
    })
    for stage in stages:
        infos = [(layout.get("performance_stages") or {}).get(stage) or {} for layout in layouts]
        out[f"candidate_devices_{stage}"] = pd.Series(
            [list(info.get("devices") or []) for info in infos], index=out.index, dtype=object
        )
        out[f"candidate_tp_{stage}"] = [info.get("tp") for info in infos]
        out[f"candidate_pp_{stage}"] = [info.get("pp") for info in infos]
        out[f"candidate_replicas_{stage}"] = [info.get("replicas") for info in infos]
    return out


def update_candidate_parallelism(
    layout: dict[str, Any],
    stage: str,
    *,
    tp: int,
    pp: int,
) -> None:
    """Mutate one candidate layout after exact-device GenZ repricing."""
    info = layout["performance_stages"][stage]
    devices = list(info.get("devices") or [])
    width = int(tp) * int(pp)
    if width < 1 or len(devices) % width:
        raise ValueError(
            f"{stage}: TP*PP={tp}*{pp} does not divide devices={len(devices)}"
        )
    replicas = len(devices) // width
    replica_devices = [
        devices[index:index + width]
        for index in range(0, len(devices), width)
    ]
    info.update({
        "tp": int(tp), "pp": int(pp), "replicas": replicas,
        "replica_devices": replica_devices,
    })
    engine = info.get("engine")
    entry = (layout.get("engines") or {}).get(engine) or {}
    if entry.get("pd_serving") == "disagg_pd":
        role = entry.get(stage)
        if isinstance(role, dict):
            role.update({
                "tp": int(tp), "pp": int(pp), "replicas": replicas,
                "replica_devices": replica_devices,
            })
    else:
        entry.update({
            "tp": int(tp), "pp": int(pp), "replicas": replicas,
            "replica_devices": replica_devices,
        })
