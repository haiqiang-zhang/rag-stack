"""RAGO name adapter — the ONLY module where RAGO stage ids may appear.

RAGO is an external library with its own stage vocabulary (``prefill``,
``decode``, ``encode``, ``retrieval``, ``passage_compressor_prefill``, …).
rag_stack speaks :mod:`rag_stack.taxonomy` PerformanceStage names everywhere.
Every interaction with RAGO crosses this adapter:

* inbound  (rag_stack → RAGO): :func:`to_rago_stage`, :func:`to_rago_groups`,
  :func:`to_rago_sweep_dfs`, :func:`to_rago_chips` — called immediately before
  handing data to ``rago.*``;
* outbound (RAGO → rag_stack): :class:`RagoNameMap` (``from_rago``) and
  :func:`qualify_rago_result_df` — called immediately on every RAGO return
  value, so no RAGO id survives past the call site.

For directly represented stages, only five names differ (the remaining names
are identical):

====================================  ==============================
PerformanceStage (rag_stack)          RAGO stage id
====================================  ==============================
generator_prefill                     prefill
generator_decode                      decode
semantic_retrieval_encode             encode
semantic_retrieval_vectorsearch       retrieval
passage_compressor  (pruning)         passage_compressor_prefill
====================================  ==============================

The pruning-compressor row makes ``from_rago`` ambiguous for the RAGO id
``passage_compressor_prefill`` (it is also the identity image of the
GENERATIVE compressor's prefill unit). A single pipeline never runs both
compressor kinds, so inversion is resolved per run against the active
PerformanceStage set — see :class:`RagoNameMap`.

``retrieval_fusion`` is a separate host CPU component with no independent RAGO
slot. Its table is folded into a copied retrieval table by
:func:`to_rago_sweep_dfs`; it is excluded from RAGO placement/chip maps.
"""

from __future__ import annotations

import ast
import json
import math
from typing import Iterable, Mapping

from rag_stack.taxonomy import TAXONOMY

# RAGO has one CPU retrieval table and no independent fusion worker. Keep
# fusion explicit throughout RAG-CM and lower it only at this boundary.
_FOLDED_STAGES = frozenset({"retrieval_fusion"})

# ── the static rag_stack → RAGO direction (total over the catalog) ──────────

_NON_IDENTITY: dict[str, str] = {
    "generator_prefill": "prefill",
    "generator_decode": "decode",
    "semantic_retrieval_encode": "encode",
    "semantic_retrieval_vectorsearch": "retrieval",
    "passage_compressor": "passage_compressor_prefill",
}

#: PerformanceStage name → RAGO stage id, total over the taxonomy catalog.
TO_RAGO: Mapping[str, str] = {
    name: _NON_IDENTITY.get(name, name) for name in TAXONOMY.performance_stages
}

# Import-time guard: the adapter must cover the catalog exactly — a taxonomy
# edit without an adapter decision is an error, not a silent identity.
assert set(TO_RAGO) == set(TAXONOMY.performance_stages), (
    "rago_adapter mapping out of sync with the taxonomy catalog: "
    f"{set(TO_RAGO) ^ set(TAXONOMY.performance_stages)}"
)

#: Every RAGO id this adapter can emit (used by :func:`assert_no_rago_names`
#: to detect leaks; identity-mapped names are legitimately shared).
_RAGO_ONLY_IDS: frozenset[str] = frozenset(
    rago for name, rago in TO_RAGO.items() if rago != name
)


def to_rago_stage(name: str) -> str:
    """Map a stage that RAGO represents directly; fusion needs table lowering."""
    if name in _FOLDED_STAGES:
        raise ValueError(
            "retrieval_fusion has no independent RAGO stage; lower its cost "
            "table with to_rago_sweep_dfs"
        )
    return TO_RAGO[name]


def to_rago_groups(groups: Iterable[Iterable[str]]) -> list[list[str]]:
    """Convert groups, omitting the CPU fusion work lowered into retrieval."""
    converted = [
        [to_rago_stage(n) for n in group if n not in _FOLDED_STAGES]
        for group in groups
    ]
    return [group for group in converted if group]


def to_rago_chips(num_chips: Mapping[str, int]) -> dict[str, int]:
    """Per-PerformanceStage chip counts → RAGO-id-keyed dict."""
    return {
        to_rago_stage(n): int(v) for n, v in num_chips.items()
        if n not in _FOLDED_STAGES
    }


def to_rago_sweep_dfs(
    sweep_dfs: Mapping[str, object],
    *,
    retrieval_fanout: int = 1,
    max_retrieval_servers: int = 1,
) -> dict[str, object]:
    """Convert stage tables and lower explicit CPU fusion for legacy RAGO.

    Search table batches count backend queries. Fusion table batches count
    retrieval visits (one list merge per user row), with a fixed ``K`` input
    lists per visit. For a search row of size ``B``, its compatibility charge
    is ``B / K * fusion_ms_per_visit``. ``retrieval_fanout`` is that ``K``,
    not the number of retrieval visits per request. RAGO applies the latter
    multiplicity separately when converting work batches to request batches.

    The fusion table must be linear in visit batch size, as the current RRF
    model executes one independent merge per visit. A nonlinear future model
    needs an explicit RAGO representation; this adapter refuses to invent an
    interpolation or silently change its units. Input tables are not mutated.
    Other tables, including ``None`` entries, pass through unchanged.

    Search-server replication is materialized BEFORE adding fusion. Otherwise
    RAGO would replicate the merged table and incorrectly replicate the shared
    local fusion service once per search server. Missing server counts use
    ideal data-parallel search scaling; existing rows keep their search costs.
    Coverage extends to the next power of two because RAGO preprocessing may
    discard non-power-of-two rows before checking whether it needs to scale.
    """
    def positive_integer(value, name: str) -> int:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if (
            isinstance(value, bool)
            or not math.isfinite(number)
            or number < 1
            or not number.is_integer()
        ):
            raise ValueError(f"{name} must be a positive integer")
        return int(number)

    server_limit = positive_integer(max_retrieval_servers, "max_retrieval_servers")
    converted = {
        to_rago_stage(n): df for n, df in sweep_dfs.items()
        if n not in _FOLDED_STAGES
    }
    fusion = sweep_dfs.get("retrieval_fusion")
    if fusion is None or len(fusion) == 0:
        return converted
    fanout = positive_integer(retrieval_fanout, "retrieval_fanout")

    search = converted.get("retrieval")
    if search is None or len(search) == 0:
        raise ValueError("retrieval_fusion requires a vector-search cost table")

    def rows(table, stage: str):
        missing = {"batch_size", "time_ms"} - set(table.columns)
        if missing:
            raise ValueError(f"{stage} table is missing columns: {sorted(missing)}")
        for batch_raw, time_raw in zip(table["batch_size"], table["time_ms"]):
            try:
                batch, time_ms = float(batch_raw), float(time_raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{stage} table has nonnumeric demand or time") from exc
            if not math.isfinite(batch) or batch < 1 or not batch.is_integer():
                raise ValueError(f"{stage} batch_size must be a positive integer")
            if not math.isfinite(time_ms) or time_ms < 0:
                raise ValueError(f"{stage} time_ms must be finite and nonnegative")
            yield batch, time_ms

    fusion_per_visit = [time_ms / batch for batch, time_ms in rows(fusion, "retrieval_fusion")]
    per_visit_ms = fusion_per_visit[0]
    if not all(
        math.isclose(value, per_visit_ms, rel_tol=1e-9, abs_tol=1e-12)
        for value in fusion_per_visit
    ):
        raise ValueError("retrieval_fusion table must scale linearly with retrieval visits")
    # RAGO's default table scaler doubles search throughput and batch size
    # while retaining service time. Reproduce that only for the search part,
    # leaving fusion as serial work proportional to all completed visits.
    search_values = list(rows(search, "semantic_retrieval_vectorsearch"))
    if "hardware.num_servers" not in search.columns:
        raise ValueError("vector-search table is missing hardware.num_servers")
    server_counts = [
        positive_integer(value, "hardware.num_servers")
        for value in search["hardware.num_servers"]
    ]
    lowered = search.copy()
    lowered["batch_size"] = [int(batch) for batch, _ in search_values]
    lowered["time_ms"] = [time_ms for _, time_ms in search_values]
    lowered["hardware.num_servers"] = server_counts
    present_servers = set(server_counts)
    coverage_limit = 1 << (server_limit - 1).bit_length()
    missing_servers = set(range(1, coverage_limit + 1)) - present_servers
    if missing_servers:
        if 1 not in present_servers:
            raise ValueError("search-server expansion requires a one-server cost table")
        import pandas as pd

        baseline = lowered.loc[lowered["hardware.num_servers"] == 1]
        tables = [lowered]
        for count in sorted(missing_servers):
            replicated = baseline.copy()
            replicated["hardware.num_servers"] = count
            replicated["batch_size"] = replicated["batch_size"] * count
            tables.append(replicated)
        lowered = pd.concat(tables, ignore_index=True)
    lowered["time_ms"] = [
        time_ms + batch / fanout * per_visit_ms
        for batch, time_ms in rows(lowered, "semantic_retrieval_vectorsearch")
    ]
    converted["retrieval"] = lowered
    return converted


def to_rago_physical_mapping(pm, system_config: dict | None = None):
    """A PerformanceStage-named ``rago.config.PhysicalMapping`` → a RAGO-id
    copy for handing to RAGO's assemble/sweep calls. rag_stack keeps its own
    (qualified) instance; only the converted copy crosses the boundary.

    ``num_chips`` may carry RAGO's own ``"e2e"`` aggregate key — passed through
    verbatim (it is RAGO bookkeeping, not a stage name)."""
    from rago.config import PhysicalMapping

    chips = getattr(pm, "num_chips", {}) or {}
    converted_chips = {
        (k if k == "e2e" else to_rago_stage(str(k))): int(v)
        for k, v in chips.items()
        if k not in _FOLDED_STAGES
    }
    groups = getattr(pm, "collocation_strategy", None)
    converted_groups = to_rago_groups(groups) if groups else groups
    available_devices = list(getattr(pm, "available_devices", None) or [])
    stage_devices = {
        to_rago_stage(str(stage)): list(devices)
        for stage, devices in (getattr(pm, "stage_devices", None) or {}).items()
        if stage not in _FOLDED_STAGES
    }
    resource_group_devices = [
        list(devices)
        for index, devices in enumerate(getattr(pm, "resource_group_devices", None) or [])
        if not groups or index >= len(groups)
        or any(stage not in _FOLDED_STAGES for stage in groups[index])
    ]
    device_layout_id = getattr(pm, "device_layout_id", None)

    # Fixed replay: preserve the resolved deployment's exact physical ranks.
    layout = (
        (system_config or {}).get("layout")
        if isinstance(system_config, dict) else None
    )
    if isinstance(layout, dict):
        perf = layout.get("performance_stages") or {}
        stage_devices = {
            to_rago_stage(str(stage)): [str(device) for device in info.get("devices", [])]
            for stage, info in perf.items()
            if stage in TO_RAGO and stage not in _FOLDED_STAGES
            and isinstance(info, dict) and info.get("kind") != "cpu_retrieval"
        }
        hardware = (system_config or {}).get("hardware") or {}
        available_devices = [str(device) for device in hardware.get("available_gpus", [])]
        if not available_devices:
            from rag_stack.cost_model.interconnect_topology import available_gpu_devices
            available_devices = available_gpu_devices(system_config)

        resource_group_devices = []
        for group in groups or []:
            if not any(stage not in _FOLDED_STAGES for stage in group):
                continue
            merged: list[str] = []
            for stage in group:
                if stage in _FOLDED_STAGES:
                    continue
                info = perf.get(stage)
                stage_group_devices = (
                    info.get("devices", []) if isinstance(info, dict) else []
                )
                for device in stage_group_devices:
                    device = str(device)
                    if device not in merged:
                        merged.append(device)
            resource_group_devices.append(merged)
        device_layout_id = "fixed_resolved"

    return PhysicalMapping(
        num_chips=converted_chips,
        num_retrieval_servers=int(getattr(pm, "num_retrieval_servers", 1) or 1),
        placement_policy=getattr(pm, "placement_policy", "collocated"),
        collocation_strategy=converted_groups,
        available_devices=available_devices,
        resource_group_devices=resource_group_devices,
        stage_devices=stage_devices,
        device_layout_id=device_layout_id,
    )


class RagoNameMap:
    """Bidirectional name map for one run, resolved against the pipeline's
    ACTIVE PerformanceStage set (needed because the pruning and generative
    compressors share the RAGO id ``passage_compressor_prefill``)."""

    def __init__(self, active_performance_stages: Iterable[str]):
        active = list(active_performance_stages)
        unknown = [n for n in active if n not in TAXONOMY.performance_stages]
        if unknown:
            raise KeyError(f"unknown PerformanceStage name(s): {unknown}")
        self.active: tuple[str, ...] = tuple(active)
        self._from_rago: dict[str, str] = {}
        for name in active:
            if name in _FOLDED_STAGES:
                continue
            rago = TO_RAGO[name]
            clash = self._from_rago.get(rago)
            if clash is not None and clash != name:
                raise ValueError(
                    f"RAGO id {rago!r} is claimed by two active PerformanceStages "
                    f"({clash!r} and {name!r}) — a pipeline cannot mix pruning and "
                    f"generative compressors")
            self._from_rago[rago] = name

    def to_rago(self, name: str) -> str:
        return to_rago_stage(name)

    def from_rago(self, rago_id: str) -> str:
        """RAGO stage id → active PerformanceStage name (hard KeyError)."""
        return self._from_rago[rago_id]

    def from_rago_groups(self, groups: Iterable[Iterable[str]]) -> list[list[str]]:
        return [[self.from_rago(r) for r in group] for group in groups]

    # ── result-dataframe qualification ──────────────────────────────────────
    #: Stage-keyed column prefixes RAGO emits; suffixes get name-mapped back.
    _STAGE_COLUMN_PREFIXES = ("num_chips_", "qps_", "batch_size_")
    #: Role-bucket / knob columns that intentionally KEEP their bare names —
    #: they are resource buckets or request knobs, not stage keys. Explicit
    #: allowlist so a new bare column is a conscious decision, not an accident.
    _ROLE_BUCKET_COLUMNS = frozenset({
        "batch_size_request", "batch_size_iterative",
        "num_chips_e2e", "qps", "qps_per_chip",
    })
    #: List-valued columns whose ELEMENTS are RAGO stage ids.
    _STAGE_LIST_COLUMNS = ("collocation_strategy", "pipeline_stage_order")
    _STAGE_DICT_COLUMNS = ("stage_devices",)

    def qualify_result_df(self, df):
        """Rename every RAGO-stage-keyed column/value of a RAGO result df back
        to PerformanceStage names, in place-safe copy. Raises if an
        unrecognized stage suffix is found (fail fast, no silent leaks)."""
        rename: dict[str, str] = {}
        for col in df.columns:
            if col in self._ROLE_BUCKET_COLUMNS:
                continue
            for prefix in self._STAGE_COLUMN_PREFIXES:
                if col.startswith(prefix):
                    suffix = col[len(prefix):]
                    if suffix in self._from_rago:
                        rename[col] = prefix + self.from_rago(suffix)
                    break
        out = df.rename(columns=rename) if rename else df
        for col in self._STAGE_LIST_COLUMNS:
            if col in out.columns:
                out = out.copy() if out is df else out
                out[col] = out[col].map(self._map_stage_list_value)
        for col in self._STAGE_DICT_COLUMNS:
            if col in out.columns:
                out = out.copy() if out is df else out
                out[col] = out[col].map(self._map_stage_dict_value)
        return out

    def _map_stage_list_value(self, value):
        if isinstance(value, str):
            # RAGO frequently str()-reprs the strategy; translate id tokens.
            for rago, name in sorted(self._from_rago.items(), key=lambda kv: -len(kv[0])):
                if rago != name:
                    value = value.replace(f"'{rago}'", f"'{name}'")
            return value
        if isinstance(value, (list, tuple)):
            return [self._map_stage_list_value(v) for v in value]
        return value

    def _map_stage_dict_value(self, value):
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                try:
                    value = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    return value
        if not isinstance(value, dict):
            return value
        return {
            self.from_rago(str(stage)): list(devices)
            for stage, devices in value.items()
            if str(stage) in self._from_rago
        }


def assert_no_rago_names(df) -> None:
    """Guard: no RAGO-only stage id survives in a qualified result df's
    columns. Used by tests and at adapter exits."""
    leaked = []
    for col in getattr(df, "columns", []):
        for prefix in RagoNameMap._STAGE_COLUMN_PREFIXES:
            if col.startswith(prefix) and col[len(prefix):] in _RAGO_ONLY_IDS:
                if col not in RagoNameMap._ROLE_BUCKET_COLUMNS:
                    leaked.append(col)
    if leaked:
        raise AssertionError(f"RAGO stage ids leaked past the adapter: {leaked}")
