"""Assemble feasible system candidates and select their final performance.

The trace path prepares one workload, enumerates candidates (or constructs a
fixed candidate), then calls dynamic_pipeline for analytical scoring,
objective-aware shortlisting and authoritative closed-loop simulation.
The static path retains RAGO's config-driven performance estimates.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import json
import logging
import threading
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # runtime import is inline (_evaluate_fixed_placement) to avoid the rago dep at load
    from rago.config import PhysicalMapping

from rag_stack.types import SystemConfig
from rag_stack.cost_model.base import CostModelModule
from rag_stack.cost_model.types import AssemblyResult
from rag_stack.cost_model.selection import apply_selection
from rag_stack.rag_ir.trace import (
    QualityTraceInput,
    QualityTraceNormalization,
    is_quality_trace_candidate,
    is_trace_payload,
)
from rag_stack.rag_ir.workflow_schema import WorkflowSchema, build_workflow_schema
from rago.config import RetrievalPolicy, EncodeDBPolicy
from rago.rago_sweep import RAGSweep
from rago.data import preprocess_data

logger = logging.getLogger("RAG-Stack")


def _attach_quality_trace_diagnostics(
    df: pd.DataFrame,
    normalization: QualityTraceNormalization,
    *,
    priced_queries: int,
) -> pd.DataFrame:
    """Publish the workload projection without feeding it back into pricing."""
    out = df.copy()
    diagnostics = normalization.diagnostics()
    diagnostics["priced_queries"] = int(priced_queries)
    for name, value in diagnostics.items():
        out[f"quality_trace_{name}"] = value
    return out


def _cacheable(obj: Any) -> Any:
    """Convert dataclasses and nested configs into JSON-stable cache payloads."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _cacheable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_cacheable(v) for v in obj]
    return obj


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _fixed_batch_from_config(
    pipeline_config: dict,
    batch_size_request: int | None,
) -> int:
    if batch_size_request is not None:
        return int(batch_size_request)
    system = pipeline_config.get("system") or {}
    resolved = _optional_int(system.get("batch_size_request"))
    if resolved is None:
        raise ValueError(
            "The fixed CM entry requires batch_size_request. Pass it in the "
            "resolved system_config (evaluate_fixed) or set "
            "pipeline_config['system']['batch_size_request']."
        )
    return resolved


def _fixed_decode_batch_from_config(
    pipeline_config: dict,
    batch_size_decode: int | None,
    batch_size_request: int,
) -> int:
    if batch_size_decode is not None:
        return int(batch_size_decode)
    system = pipeline_config.get("system") or {}
    return _optional_int(system.get("batch_size_decode")) or int(batch_size_request)


def _fixed_stage_user_batch(stage: str, request_batch: int, decode_batch: int) -> int:
    return int(decode_batch if stage == "generator_decode" else request_batch)


def _fixed_stage_resource(stage: str, fixed_placement: "PhysicalMapping") -> int | None:
    if stage == "retrieval_fusion":
        return 1
    if stage == "semantic_retrieval_vectorsearch":
        return int(getattr(fixed_placement, "num_retrieval_servers", 1) or 1)
    chips = getattr(fixed_placement, "num_chips", {}) or {}
    if stage not in chips:
        return None
    return int(chips[stage])


def _filter_stage_sweep_df_for_fixed(
    df: pd.DataFrame | None,
    *,
    stage: str,
    fixed_placement: "PhysicalMapping",
    batch_size: int,
) -> pd.DataFrame | None:
    if df is None or len(df) == 0:
        return df
    out = df
    resource = _fixed_stage_resource(stage, fixed_placement)
    if resource is not None:
        if stage in ("semantic_retrieval_vectorsearch", "retrieval_fusion") and "hardware.num_servers" in out.columns:
            out = out[out["hardware.num_servers"] == resource]
        elif "hardware.num_chips" in out.columns:
            out = out[out["hardware.num_chips"] == resource]
        elif "num_chips" in out.columns:
            out = out[out["num_chips"] == resource]
    # Deliberately NOT filtered to the exact pinned batch.  Forward/single-role
    # sweeps may Pareto-drop batches and rely on RAGO's latency-table extension;
    # two-role LLM sweeps retain the complete feasible batch grid because their
    # component surface cannot establish cross-batch dominance.  Handing RAGO
    # every surviving row preserves sweep/fixed parity in both cases; the
    # pinned batch is enforced downstream by assemble_cost(fixed_batch_size_*).
    if len(out) == 0:
        available_batches = (
            sorted(df["batch_size"].dropna().unique().tolist())
            if "batch_size" in df.columns else []
        )
        raise RuntimeError(
            f"Fixed placement has no stage cost row for stage={stage!r}, "
            f"batch={batch_size}, resource={resource}. "
            f"Available batches: {available_batches[:12]}"
        )
    return out.copy()


def _filter_fixed_result_df(
    df: pd.DataFrame,
    *,
    batch_size_request: int,
    batch_size_decode: int,
) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise RuntimeError("Fixed placement produced no rows.")
    out = df
    if "batch_size_request" not in out.columns:
        raise RuntimeError("Fixed placement result is missing batch_size_request.")
    out = out[out["batch_size_request"] == int(batch_size_request)]
    if len(out) == 0:
        available = sorted(df["batch_size_request"].dropna().unique().tolist())
        raise RuntimeError(
            f"Fixed placement produced no row for batch_size_request={batch_size_request}. "
            f"Available: {available}"
        )
    if "batch_size_decode" in out.columns:
        dec = out[out["batch_size_decode"] == int(batch_size_decode)]
        if len(dec) == 0:
            available = sorted(out["batch_size_decode"].dropna().unique().tolist())
            raise RuntimeError(
                f"Fixed placement produced no row for batch_size_decode={batch_size_decode}. "
                f"Available: {available}"
            )
        out = dec
    sort_cols = [
        col for col in (
            "batch_size_request",
            "batch_size_decode",
            "num_chips_e2e",
            "placement_and_resource_id",
        )
        if col in out.columns
    ]
    if sort_cols:
        out = out.sort_values(sort_cols)
    return out.head(1).copy()


def _materialize_rago_decode_axis(
    df: pd.DataFrame,
    pipeline_config: dict,
) -> pd.DataFrame:
    """Make RAGO's decode sweep the single deployed decode dimension.

    With ``no_microbatching=true`` RAGO already enumerates every legal
    ``(batch_size_request, batch_size_generator_decode)`` pair.  The latter is
    the generator's independently deployed decode concurrency.  DynamicEngine
    historically saw no public ``batch_size_decode`` column and expanded a
    second copy of the same grid, producing ``D_rago × D_runtime`` pseudo
    configurations which signature grouping later hid.  Materialize RAGO's
    value before DynamicEngine so every RAGO config is represented exactly
    once. Non-RAGO callers retain DynamicEngine's compatibility expansion.
    """

    if df is None or len(df) == 0:
        return df
    request_column = "batch_size_request"
    rago_decode_column = "batch_size_generator_decode"
    if request_column not in df.columns or rago_decode_column not in df.columns:
        return df

    out = df.copy()
    required = out[[request_column, rago_decode_column]]
    if bool(required.isna().any().any()):
        raise ValueError(
            "RAGO produced a missing request or generator-decode batch"
        )
    request = out[request_column].astype(int)
    decode = out[rago_decode_column].astype(int)
    if "batch_size_decode" in out.columns:
        public_decode = pd.to_numeric(out["batch_size_decode"], errors="coerce")
        mismatch = public_decode.isna() | (
            public_decode.fillna(-1).astype(int) != decode
        )
        if bool(mismatch.any()):
            bad = out.loc[
                mismatch,
                [request_column, rago_decode_column, "batch_size_decode"],
            ].head(5).to_dict("records")
            raise ValueError(
                "RAGO generator-decode and deployed decode axes diverged: "
                f"{bad}"
            )
    invalid = decode < request
    if bool(invalid.any()):
        bad = out.loc[
            invalid,
            [request_column, rago_decode_column],
        ].head(5).to_dict("records")
        raise ValueError(
            "RAGO produced decode batches below the request batch: "
            f"{bad}"
        )

    system = pipeline_config.get("system") or {}
    search_space = system.get("cm_search_space") or {}
    pinned_decode = _optional_int(system.get("batch_size_decode"))
    if pinned_decode is not None:
        expected = np.maximum(request, int(pinned_decode)).astype(int)
        keep = decode == expected
    else:
        configured_grid = search_space.get("batch_size_decode")
        if configured_grid is None:
            keep = pd.Series(True, index=out.index)
        else:
            configured = {int(value) for value in configured_grid}
            keep = (decode == request) | decode.isin(configured)

    out = out.loc[keep].copy()
    if len(out) == 0:
        target = (
            f"pinned batch_size_decode={pinned_decode}"
            if pinned_decode is not None
            else "configured batch_size_decode grid"
        )
        raise ValueError(f"RAGO produced no row for the {target}")
    out["batch_size_decode"] = out[rago_decode_column].astype(int)
    return out.reset_index(drop=True)


def _requested_decode_batch_grid(
    pipeline_config: dict,
    max_batch_size_request: int,
) -> list[int]:
    """Return the complete generator-decode grid RAGO must enumerate.

    RAGO's component Pareto may stop before a still-deployable vLLM resident
    depth. Candidate generation is a system-design contract, so it must come
    from the public CM search space rather than a Pareto truncation. RAGO may
    use its existing extrapolation to construct the row; the exact-device
    repricer and DES then price the requested batch on the concrete topology.
    """

    maximum = _validate_pow2(
        int(max_batch_size_request),
        "cm_search_space.max_batch_size_request",
    )
    request_grid: list[int] = []
    value = 1
    while value <= maximum:
        request_grid.append(value)
        value *= 2

    system = pipeline_config.get("system") or {}
    pinned = _optional_int(system.get("batch_size_decode"))
    if pinned is not None:
        pinned = _validate_pow2(pinned, "system.batch_size_decode")
        return sorted({max(request, pinned) for request in request_grid})

    configured = (system.get("cm_search_space") or {}).get(
        "batch_size_decode"
    )
    if configured is None:
        return request_grid
    if not isinstance(configured, (list, tuple, set)):
        raise ValueError(
            "cm_search_space.batch_size_decode must be a list of positive "
            "power-of-two values"
        )
    explicit = {
        _validate_pow2(value, "cm_search_space.batch_size_decode")
        for value in configured
    }
    if not explicit:
        raise ValueError("cm_search_space.batch_size_decode cannot be empty")
    # Per request, the deployment space is {request} plus explicit D > request.
    # RAGO enumerates the global Cartesian grid and the outbound boundary
    # performs that row-local filter.
    return sorted(set(request_grid) | explicit)


def _parse_strategy(strategy: Any) -> list[list[str]]:
    if strategy is None:
        return []
    if isinstance(strategy, str):
        try:
            strategy = ast.literal_eval(strategy)
        except Exception:  # noqa: BLE001
            return []
    return [list(group) for group in strategy]


def _strategy_stage_order(strategy: Any) -> list[str]:
    return [stage for group in _parse_strategy(strategy) for stage in group]


def _restore_fusion_stage(df: pd.DataFrame, stages: list[str]) -> pd.DataFrame:
    """Restore the explicit host stage after RAGO's combined CPU cost slot."""
    if "retrieval_fusion" not in stages or df.empty:
        return df
    out = df.copy()

    def insert(order):
        order = list(order)
        if "retrieval_fusion" not in order:
            index = order.index("semantic_retrieval_vectorsearch") + 1
            order.insert(index, "retrieval_fusion")
        return order

    if "collocation_strategy" in out:
        raw = out["collocation_strategy"].map(_parse_strategy)
        if "pipeline_stage_order" in out:
            out["pipeline_stage_order"] = out["pipeline_stage_order"].map(insert)
        else:
            out["pipeline_stage_order"] = raw.map(lambda groups: insert(_strategy_stage_order(groups)))
        out["collocation_strategy"] = raw.map(
            lambda groups: groups if any("retrieval_fusion" in g for g in groups)
            else groups + [["retrieval_fusion"]]
        )
    else:
        out["pipeline_stage_order"] = [list(stages) for _ in range(len(out))]
    if "batch_size_request" in out:
        out["batch_size_retrieval_fusion"] = out["batch_size_request"]
    return out


def _cpu_retrieval_resource_strategy(strategy: Any) -> list[list[str]]:
    """Return resource groups with CPU retrieval split out.

    ``collocation_strategy`` is a resource grouping, not a serialized execution
    order. CPU vectordb retrieval is never part of a GPU group, so split it into
    its own resource group and keep GPU stages grouped separately. The execution
    order is preserved in ``pipeline_stage_order``.
    """
    out: list[list[str]] = []
    cpu_seen: set[str] = set()
    for group in _parse_strategy(strategy):
        gpu_group = [stage for stage in group if stage not in ("semantic_retrieval_vectorsearch", "retrieval_fusion")]
        if gpu_group:
            out.append(gpu_group)
        for stage in group:
            if stage in ("semantic_retrieval_vectorsearch", "retrieval_fusion") and stage not in cpu_seen:
                out.append([stage])
                cpu_seen.add(stage)
    return out


def _normalize_retrieval_resource_groups(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or len(df) == 0 or "collocation_strategy" not in df.columns:
        return df
    out = df.copy()
    raw = out["collocation_strategy"]
    if "pipeline_stage_order" not in out.columns:
        out["pipeline_stage_order"] = raw.map(_strategy_stage_order)
    out["collocation_strategy"] = raw.map(_cpu_retrieval_resource_strategy)
    return out


def _rago_strategy_from_resource_groups(
    resource_groups: Any,
    pipeline_stage_order: list[str],
) -> list[list[str]]:
    """Convert public resource groups into RAGO's ordered partition format.

    The public API treats ``collocation_strategy`` as hardware resource groups:
    CPU ``retrieval`` is its own group, and GPU stages may share a resource even
    when retrieval sits between them in execution order. RAGO's assemble code
    expects ordered contiguous partitions. This helper bridges the two without
    changing the exported CM semantics.
    """
    groups = _parse_strategy(resource_groups)
    if not groups or not pipeline_stage_order:
        return groups

    stage_to_group: dict[str, int] = {}
    for gi, group in enumerate(groups):
        for stage in group:
            stage_to_group[stage] = gi

    def _next_gpu_group(start: int) -> int | None:
        for stage in pipeline_stage_order[start + 1:]:
            if stage in ("semantic_retrieval_vectorsearch", "retrieval_fusion"):
                continue
            return stage_to_group.get(stage)
        return None

    out: list[list[str]] = []
    current: list[str] = []
    current_gpu_group: int | None = None

    def _flush() -> None:
        nonlocal current, current_gpu_group
        if current:
            out.append(current)
        current = []
        current_gpu_group = None

    for idx, stage in enumerate(pipeline_stage_order):
        if stage == "retrieval_fusion":
            continue  # RAGO's CPU table already includes fusion at this boundary.
        if stage not in stage_to_group:
            continue
        if stage == "semantic_retrieval_vectorsearch":
            next_gid = _next_gpu_group(idx)
            if current and current_gpu_group is not None and next_gid == current_gpu_group:
                current.append(stage)
            else:
                _flush()
                out.append([stage])
            continue

        gid = stage_to_group[stage]
        if not current:
            current = [stage]
            current_gpu_group = gid
        elif current_gpu_group == gid:
            current.append(stage)
        else:
            _flush()
            current = [stage]
            current_gpu_group = gid

    _flush()
    return out


def _coerce_workflow_schema(schema: WorkflowSchema | dict | None) -> WorkflowSchema | None:
    if schema is None:
        return None
    if isinstance(schema, WorkflowSchema):
        return schema
    if (
        isinstance(schema, list)          # bare trace list — never a schema
        or is_trace_payload(schema)
        or is_quality_trace_candidate(schema)
    ):
        raise TypeError(
            "A quality trace payload was passed in the workflow_schema slot. "
            "The legacy positional payload form is removed: pass "
            "workflow_schema=..., trace=... explicitly (evaluate_dynamic), or "
            "use evaluate_fixed(pipeline_config, system_config, trace=...)."
        )
    if isinstance(schema, dict) and "dataflow_type" not in schema:
        raise TypeError(
            "An aggregate/runtime-profile dict was passed in the "
            "workflow_schema slot. The legacy positional payload form is "
            "removed: build the schema (build_workflow_schema) and carry the "
            "aggregate on schema.runtime_stats, then pass workflow_schema=... "
            "explicitly."
        )
    if isinstance(schema, dict):
        return WorkflowSchema.from_dict(schema)
    raise TypeError(
        "workflow_schema must be a WorkflowSchema, a WorkflowSchema dict, or None"
    )


def _resolve_workflow_schema(
    pipeline_config: dict,
    workflow_schema: WorkflowSchema | dict | None,
    stages: list[str] | None = None,
) -> WorkflowSchema:
    schema = _coerce_workflow_schema(workflow_schema)
    if schema is not None:
        return schema
    return build_workflow_schema(pipeline_config, stages)


def _two_role_llm_engines_in_schema(
    schema: WorkflowSchema,
) -> tuple[str, ...]:
    """Return active LLM engines whose prefill+decode need cycle composition."""

    from rag_stack.taxonomy import TAXONOMY

    active = {str(stage) for stage in schema.stages}
    engines: list[str] = []
    for engine in sorted(TAXONOMY.llm_stages, key=lambda stage: stage.value):
        role_names = {
            performance_stage.kind: performance_stage.name
            for performance_stage in TAXONOMY.performance_stages_of(engine)
            if performance_stage.kind in {"prefill", "decode"}
        }
        if (
            "prefill" in role_names
            and "decode" in role_names
            and {role_names["prefill"], role_names["decode"]}.issubset(active)
        ):
            engines.append(engine.value)
    return tuple(engines)


def _materialize_corpus_stats(
    pipeline_config: dict,
    schema: WorkflowSchema | None,
) -> dict:
    """Enforce the frozen input contract at the CM boundary.

    Corpus statistics enter the CM ONLY through ``schema.corpus_stats``: the
    config's ``_token_stats`` / ``vectordb[i]["N"]`` side channels are wiped
    and re-materialized from the schema, so every internal consumer
    (``TokenStats.resolve``, the FAISS adapters, communication, genz) can only
    ever see schema-provided values. ``CorpusStats.token_stats_dict()``
    reproduces the producer's ``_token_stats`` dict byte-for-byte, so a schema
    lifted from this very config makes this a bit-exact identity.

    * ``schema is None`` → the internally-built schema would lift from this
      config anyway; materialize against that self-lift (identity).
    * schema carries no stats and the config has none → config unchanged
      (downstream raises exactly as before when it actually needs stats).
    * config HAS ``_token_stats`` but the schema carries none → the caller
      hand-built a schema without the sanctioned lift: contract violation.
    """
    from rag_stack.rag_ir.workflow_schema import (
        CorpusStats,
        corpus_stats_from_pipeline_config,
    )

    if schema is None:
        stats = corpus_stats_from_pipeline_config(pipeline_config)
    else:
        stats = schema.corpus_stats
        if isinstance(stats, dict):
            stats = CorpusStats.from_dict(stats)
    if stats is None:
        if pipeline_config.get("_token_stats") is not None:
            raise ValueError(
                "pipeline_config carries _token_stats but the supplied "
                "workflow_schema has corpus_stats=None. Build the schema via "
                "build_workflow_schema(pipeline_config) (it lifts the "
                "producer-stamped stats) or set schema.corpus_stats "
                "explicitly — the CM no longer reads config side channels "
                "past this boundary."
            )
        return pipeline_config

    # Per-position vectordb names, computed on the ORIGINAL entries with the
    # exact rule of corpus_stats_from_pipeline_config (the sanctioned lift),
    # so materialization is its precise inverse.
    original_vdbs = pipeline_config.get("vectordb") or []
    names: list[str | None] = []
    for vdb in original_vdbs:
        if not isinstance(vdb, dict):
            names.append(None)
            continue
        name = vdb.get("name") or vdb.get("vectordb")
        if not name:
            name = f"vectordb_{original_vdbs.index(vdb)}"
        names.append(str(name))

    cfg = deepcopy(pipeline_config)
    cfg.pop("_token_stats", None)
    index_sizes = stats.vectordb_index_sizes or {}
    for vdb, name in zip(cfg.get("vectordb") or [], names):
        if not isinstance(vdb, dict):
            continue
        vdb.pop("N", None)
        if name is not None and name in index_sizes:
            vdb["N"] = index_sizes[name]
    cfg["_token_stats"] = stats.token_stats_dict()
    return cfg


def _fixed_num_chips_e2e(
    stages: list[str],
    fixed_placement: "PhysicalMapping",
) -> int:
    chips = getattr(fixed_placement, "num_chips", {}) or {}
    groups = getattr(fixed_placement, "collocation_strategy", None) or [
        [stage] for stage in stages
    ]
    total = 0
    for group in groups:
        group_chips = [
            int(chips.get(stage, 0) or 0)
            for stage in group
            if stage not in ("semantic_retrieval_vectorsearch", "retrieval_fusion")
        ]
        if group_chips:
            total += max(group_chips)
    if total <= 0:
        total = sum(
            int(chips.get(stage, 0) or 0)
            for stage in stages
            if stage not in ("semantic_retrieval_vectorsearch", "retrieval_fusion")
        )
    return max(1, int(total))


def _build_fixed_candidate_df(
    *,
    stages: list[str],
    fixed_placement: "PhysicalMapping",
    batch_size_request: int,
    batch_size_decode: int,
    scheduling_policy: str,
) -> pd.DataFrame:
    """Build the single candidate row for trace-driven fixed-config costing.

    DynamicEngine needs placement/resource/batch metadata, not RAGO's swept
    assembly. The initial qps is a large placeholder so communication can apply
    a bandwidth cap before DynamicEngine recomputes fixed-config capacity from
    trace-derived per-resource services.
    """
    chips = getattr(fixed_placement, "num_chips", {}) or {}
    raw_strategy = getattr(
        fixed_placement, "collocation_strategy", [[stage] for stage in stages]
    )
    num_chips_e2e = _fixed_num_chips_e2e(stages, fixed_placement)
    qps_placeholder = 1e30
    row: dict[str, Any] = {
        "latency_s": 0.0,
        "latency_s_ttft": 0.0,
        "latency_s_tpot": 0.0,
        "qps": qps_placeholder,
        "qps_per_chip": qps_placeholder / num_chips_e2e,
        "qps_bound": [],
        "batch_size_request": int(batch_size_request),
        "num_chips_e2e": num_chips_e2e,
        "num_retrieval_servers": int(
            getattr(fixed_placement, "num_retrieval_servers", 1) or 1
        ),
        "placement_policy": getattr(fixed_placement, "placement_policy", "collocated"),
        "collocation_strategy": _cpu_retrieval_resource_strategy(raw_strategy),
        "pipeline_stage_order": _strategy_stage_order(raw_strategy) or list(stages),
        "scheduling_policy": scheduling_policy,
        "placement_and_resource_id": "fixed",
        "available_devices": list(
            getattr(fixed_placement, "available_devices", None) or []
        ),
        "resource_group_devices": [
            list(devices)
            for devices in (
                getattr(fixed_placement, "resource_group_devices", None) or []
            )
        ],
        "stage_devices": {
            str(stage): list(devices)
            for stage, devices in (
                getattr(fixed_placement, "stage_devices", None) or {}
            ).items()
        },
        "device_layout_id": getattr(
            fixed_placement, "device_layout_id", None,
        ),
    }
    for stage in stages:
        stage_batch = _fixed_stage_user_batch(
            stage, batch_size_request, batch_size_decode,
        )
        row[f"batch_size_{stage}"] = stage_batch
        row[f"qps_{stage}"] = None
        if stage not in ("semantic_retrieval_vectorsearch", "retrieval_fusion"):
            row[f"num_chips_{stage}"] = int(chips.get(stage, 1) or 1)
    # ALWAYS stamp the resolved decode batch on the row: DynamicEngine's walk
    # prefers this column and only falls back to pipeline_config's
    # system.batch_size_decode when it is absent — and for a replay whose
    # saved system_config predates the decode-consistency rule, that stale
    # config value would silently override the caller's explicit
    # batch_size_decode (the 0047 replay priced decode at 2 instead of the
    # measured 128 this way). max(request, ·) mirrors _decode_batch_of.
    row["batch_size_decode"] = max(int(batch_size_decode), int(batch_size_request))
    return _restore_fusion_stage(pd.DataFrame([row]), stages)


# ---------------------------------------------------------------------------
# Per-stage sweep DataFrame generation
# ---------------------------------------------------------------------------


# Pure sweep-INPUT helpers live in sweep_inputs.py (shared with assembly.py +
# the engines). Imported back here so the sweep orchestration + StaticAssembly
# below — and existing `from ...static_assembly import <helper>` call sites — keep working.
from rag_stack.cost_model.sweep_inputs import (  # noqa: F401
    _generate_stage_sweep_df,
    _ALLOWED_NODE_SYSTEM_KEYS,
    _get_query_expansion_fanout,
    _get_retrieval_backend_fanout,
    _get_prefill_fanout,
    _get_num_retrievals,
    _rag_ir_token_stats,
    _with_rag_ir_token_overrides,
    _resolve_main_prefill_tokens,
    _resolve_main_dec_steps,
    _get_reranker_input_fanout,
    _get_stage_fanout,
    _get_compressor_input_fanout,
    _validate_pow2,
    _resolve_max_batch_size_for_stage,
    build_performance_stage_cost_models,
)

# ---------------------------------------------------------------------------
# Parallel RAGO sweep helper
# ---------------------------------------------------------------------------

_COLLOC_POOL = None
_COLLOC_POOL_WORKERS = 0
_COLLOC_POOL_SWAP_LOCK = threading.Lock()
_CM_WORKER_THREAD_LIMITER = None
_CM_RESERVED_CPUS = 2
_CM_NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

# CM pricing serialization (07-05, root-caused 07-06): concurrent evaluate_*
# calls used to corrupt shared state and surface as "tuple index out of range"
# deep in the sweep. The 07-06 forensics pinned two real hazards — (a) GenZ's
# ``get_inference_system`` mutated a caller-shared System instance (fixed in
# GenZ: it now configures a copy) and (b) ``_collocation_pool`` below KILLED a
# live pool (cancel_futures=True) when another thread asked for more workers,
# destroying the first thread's in-flight strategy futures (fixed below:
# retire-without-cancel + swap lock). With both fixed, this lock is defense in
# depth for the remaining unaudited GenZ/RAGO surface — it is uncontended in
# the current single-pricer architecture and costs nothing.
_CM_PRICING_LOCK = threading.RLock()


def _cm_affinity_worker_cap() -> int:
    """Return the system-wide CM process budget: visible CPUs minus two.

    ``sched_getaffinity`` is authoritative inside cpusets/containers; falling
    back to ``cpu_count`` keeps the same contract on platforms without it.
    At least one worker remains available on very small systems.
    """
    import os

    try:
        visible_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        visible_cpus = os.cpu_count() or 1
    return max(1, int(visible_cpus) - _CM_RESERVED_CPUS)


def _cm_in_child_process() -> bool:
    """True when CM is already running inside another process worker."""
    import multiprocessing as mp

    try:
        return mp.parent_process() is not None
    except (AttributeError, AssertionError):
        return mp.current_process().name != "MainProcess"


def _cm_process_worker_count(task_count: int) -> int:
    """Use the CM process budget without creating workers with no task."""
    if _cm_in_child_process():
        return 1
    return max(1, min(max(1, int(task_count)), _cm_affinity_worker_cap()))


def _cm_serialized(fn):
    @functools.wraps(fn)
    def _wrapped(*args, **kwargs):
        with _CM_PRICING_LOCK:
            return fn(*args, **kwargs)
    return _wrapped


def _configure_cm_native_threads() -> None:
    """Use process parallelism, never nested BLAS/OpenMP thread teams.

    Spawn children must inherit these values *before* importing NumPy/GenZ;
    their worker initializer then keeps the limits for the process lifetime.
    """
    import os

    for variable in _CM_NATIVE_THREAD_VARIABLES:
        os.environ[variable] = "1"


@contextlib.contextmanager
def _cm_native_thread_environment():
    """Temporarily seed spawn children without leaking into the optimizer."""
    import os

    previous = {
        variable: os.environ.get(variable)
        for variable in _CM_NATIVE_THREAD_VARIABLES
    }
    _configure_cm_native_threads()
    try:
        yield
    finally:
        for variable, value in previous.items():
            if value is None:
                os.environ.pop(variable, None)
            else:
                os.environ[variable] = value


def _cm_worker_initializer(startup_barrier=None) -> None:
    """Reassert one native thread and join the deterministic startup barrier."""
    global _CM_WORKER_THREAD_LIMITER

    _configure_cm_native_threads()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        pass
    else:
        _CM_WORKER_THREAD_LIMITER = threadpool_limits(limits=1)
    if startup_barrier is not None:
        # ProcessPoolExecutor starts workers lazily.  Making every initializer
        # rendezvous forces the complete requested pool to inherit the scoped
        # one-thread environment before any pricing task can run.  This avoids
        # the probabilistic "did every PID happen to consume a probe?" test.
        startup_barrier.wait(timeout=120)


def _cm_worker_ready() -> int:
    """Force lazy spawn while the parent has the scoped thread environment."""
    import os
    import time

    time.sleep(0.05)
    return os.getpid()


def _collocation_pool(workers: int):
    """Lazy persistent spawn pool for collocation-strategy sweeps (07-05).
    Spawn context: judge/CM-overlap safe. Persistent: avoids re-importing
    GenZ in fresh workers on every CM invocation.

    THREAD SAFETY (07-06 root-cause fix): growing the pool used to
    ``shutdown(cancel_futures=True)`` the LIVE pool — cancelling another
    thread's in-flight strategy futures mid-sweep, which returned a
    corrupted/partial result set to that thread. The old pool is now
    retired WITHOUT cancelling (its pending futures run to completion; it
    just refuses new work), and the check-and-swap is locked so two
    threads cannot race the growth itself."""
    global _COLLOC_POOL, _COLLOC_POOL_WORKERS
    from concurrent.futures import ProcessPoolExecutor as _PPE
    import multiprocessing as _mp
    if _cm_in_child_process():
        raise RuntimeError("CM process workers may not create nested pools")
    workers = _cm_process_worker_count(workers)
    with _COLLOC_POOL_SWAP_LOCK:
        if _COLLOC_POOL is None or _COLLOC_POOL_WORKERS < workers:
            if _COLLOC_POOL is not None:
                # Retire, don't kill: pending futures keep running on the old
                # pool's workers; only NEW submissions are refused.
                _COLLOC_POOL.shutdown(wait=False)
            # ProcessPoolExecutor spawns lazily on submit. Warm every worker
            # inside the scoped environment so children import NumPy/GenZ with
            # one native thread, then restore the optimizer parent's variables.
            with _cm_native_thread_environment():
                spawn_context = _mp.get_context("spawn")
                startup_barrier = spawn_context.Barrier(workers)
                _COLLOC_POOL = _PPE(
                    max_workers=workers,
                    mp_context=spawn_context,
                    initializer=_cm_worker_initializer,
                    initargs=(startup_barrier,),
                )
                # Futures complete only after all initializers cross the
                # barrier.  The task results are intentionally irrelevant;
                # they merely make ProcessPoolExecutor start lazily.
                try:
                    ready = [
                        _COLLOC_POOL.submit(_cm_worker_ready)
                        for _ in range(workers)
                    ]
                    for future in ready:
                        future.result()
                except BaseException:
                    broken_pool = _COLLOC_POOL
                    _COLLOC_POOL = None
                    _COLLOC_POOL_WORKERS = 0
                    broken_pool.shutdown(wait=False, cancel_futures=True)
                    raise
            _COLLOC_POOL_WORKERS = workers
        return _COLLOC_POOL


def _collocation_pool_reset():
    global _COLLOC_POOL, _COLLOC_POOL_WORKERS
    if _COLLOC_POOL is not None:
        _COLLOC_POOL.shutdown(wait=False, cancel_futures=True)
    _COLLOC_POOL, _COLLOC_POOL_WORKERS = None, 0


def _bucketed_dag_for_key(dag, bucket: int):
    """KEY-ONLY view of the runtime dag with token counts rounded to
    ``bucket`` multiples (see CM_SWEEP_CACHE_TOKEN_BUCKET above). Only the
    ``token_stats`` subtree is bucketed — fanouts and flags are small numbers
    where rounding would collapse genuinely different workloads. Per-query
    dag lists (FlashRAG trace shape) are returned unchanged: they are not
    consumed by the sweep generator, so exact keying is already correct."""
    if bucket <= 0 or not isinstance(dag, dict):
        return dag

    def _round(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return int(round(float(v) / bucket)) * bucket
        if isinstance(v, dict):
            return {k: _round(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_round(x) for x in v]
        return v

    out = dict(dag)
    if isinstance(out.get("token_stats"), dict):
        out["token_stats"] = _round(out["token_stats"])
    return out


def _sweep_one_strategy(args):
    """Process a single collocation strategy in a subprocess.

    Receives a pickled RAGSweep copy + strategy, runs assemble_cost for all
    hardware mappings of that strategy, returns list of result dicts.
    """
    import copy
    sweep_obj, strategy, kwargs = args
    sweep_copy = copy.deepcopy(sweep_obj)
    # Override strategies so run_sweep_across_systems only processes this one
    sweep_copy.all_placement_strategies = [strategy]
    result_df = sweep_copy.run_sweep_across_systems(**kwargs)
    return result_df


def _run_sweep_parallel(
    rago_sweep,
    placement_policy: str,
    min_num_chips: int,
    max_num_chips: int,
    max_num_chips_per_stage: int,
    scheduling_policy: str,
    max_batch_size_request: int,
    no_microbatching: bool,
    possible_batch_sizes_decode: list[int],
    allow_decode_collocation: bool = True,
):
    """Run RAGO sweep with strategies parallelized across processes."""
    from concurrent.futures import as_completed

    # Propagate the decode-collocation policy onto the sweep instance so it
    # reaches both the single-strategy path (run_sweep_across_systems below) and
    # every subprocess copy (_sweep_one_strategy deepcopies this object). This is
    # the single channel that keeps strategy *generation* and the multi-GPU
    # decode-group *assertion* in run_sweep_across_systems consistent.
    rago_sweep.allow_decode_collocation = allow_decode_collocation

    sweep_kwargs = dict(
        min_num_chips=min_num_chips,
        max_num_chips=max_num_chips,
        max_num_chips_per_stage=max_num_chips_per_stage,
        placement_policy=placement_policy,
        scheduling_policy=scheduling_policy,
        max_batch_size_request=max_batch_size_request,
        no_microbatching=no_microbatching,
        possible_batch_sizes_decode=possible_batch_sizes_decode,
    )

    # Reset the cached strategies set by a previous placement_policy iteration;
    # otherwise get_collocation_combinations() returns the stale value (e.g. the
    # full-disaggregation strategy from the previous pass) because it caches on
    # self.all_placement_strategies.
    rago_sweep.all_placement_strategies = None

    # Strategy enumeration. Three branches, each with a distinct rationale:
    #
    #   1. collocated + single GPU:
    #      Structurally there is exactly one strategy — all stages share the
    #      one GPU. RAGO's heuristics (disaggregate_decode,
    #      filter_retrieval_at_beginning_or_end) all assume ≥2 groups and
    #      cannot apply, so we bypass get_collocation_combinations entirely
    #      instead of trying to disable its flags case-by-case.
    #
    #   2. collocated + multi-GPU:
    #      Defer to RAGO's prescriptive placement heuristics (its own defaults
    #      for disaggregate_decode, filter_retrieval_at_beginning_or_end,
    #      filter_full_disaggregation). Two structural feasibility bounds:
    #        - upper: len(s) <= max_num_chips (need at least 1 chip per group)
    #        - lower: len(s) * max_num_chips_per_stage >= min_num_chips
    #          (max chips reachable by a strategy is len(s) * per_stage cap;
    #           if even that can't hit the budget, skip the strategy)
    #      Final per-strategy chip allocation is filtered downstream by
    #      run_sweep_get_hardware_mappings against [min_num_chips, max_num_chips].
    #      CPU retrieval is normalized into its own resource group after the
    #      candidate is assembled. The pipeline order is tracked separately so
    #      non-adjacent GPU stages can still share a GPU resource group without
    #      implying that retrieval runs on GPU.
    #
    #   3. disaggregated:
    #      Each stage on its own GPU (requires len(stages) chips).
    if placement_policy == "collocated":
        if max_num_chips == 1:
            strategies = [[list(rago_sweep.stages)]]
        else:
            # retrieval is CPU vectordb work, not a GPU-resident stage. The
            # public df is normalized later so retrieval always appears as a
            # separate CPU resource group.
            # `disaggregate_decode = not allow_decode_collocation`: when
            # collocation is allowed (default), P/D disaggregation is a
            # CONSEQUENCE of the partition, PER-LLM — every LLM's prefill+decode
            # may share a group (unified engine) or be split (1P1D), freely. This
            # matches the measured runtime, which realizes BOTH for every LLM:
            # generator via VllmPdPair, query_expansion via an aux PD pair, etc.
            # No per-LLM keep-together constraint — measured supports what the CM
            # explores. Setting allow_decode_collocation=False reverts to RAGO's
            # heuristic of forcing decode into its own group.
            candidates = rago_sweep.get_collocation_combinations(
                rago_sweep.stages,
                filter_retrieval_at_beginning_or_end=True,
                disaggregate_decode=not allow_decode_collocation,
            )
            strategies = [
                s for s in candidates
                if len(s) <= max_num_chips
                and len(s) * max_num_chips_per_stage >= min_num_chips
            ]
    else:
        strategies = [[[stage] for stage in rago_sweep.stages]]

    strategy_workers = _cm_process_worker_count(len(strategies))

    # For one strategy, a one-CPU host, or a nested CM caller, stay in-process.
    if len(strategies) <= 1 or strategy_workers <= 1:
        rago_sweep.all_placement_strategies = strategies
        return rago_sweep.run_sweep_across_systems(**sweep_kwargs)

    logger.info(
        f"Parallelizing {len(strategies)} collocation strategies "
        f"across {strategy_workers} processes"
    )

    tasks = [
        (rago_sweep, strategy, sweep_kwargs)
        for strategy in strategies
    ]

    result_dfs: dict[int, pd.DataFrame] = {}
    # Spawn (not default fork): this pool also runs CONCURRENTLY with the
    # LLM-judge's async HTTP machinery (07-04 judge/CM overlap) — forking a
    # threaded, CUDA-initialized parent copies held locks into children and
    # can deadlock them. PERSISTENT (07-05): spawn workers re-import GenZ on
    # every call (~10-20 s of pure interpreter+import tax per CM invocation,
    # felt hardest by the perf-oracle's per-candidate pricing) — so the pool
    # is created once and reused; a broken pool is rebuilt on demand.
    from concurrent.futures.process import BrokenProcessPool
    for attempt in range(2):
        executor = _collocation_pool(strategy_workers)
        try:
            futures = {
                executor.submit(_sweep_one_strategy, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                df = future.result()
                if df is not None and len(df) > 0:
                    result_dfs[futures[future]] = df
            break
        except BrokenProcessPool:
            logger.warning(
                "collocation pool broke; rebuilding once before exact serial "
                "fallback"
            )
            _collocation_pool_reset()
            result_dfs = {}
    else:
        # Never return a partial placement space: it can change the selected
        # config and therefore optimizer behavior.  The serial RAGO path is
        # the exact same computation in deterministic strategy order.
        logger.warning(
            "collocation pool broke twice; falling back to exact serial sweep"
        )
        rago_sweep.all_placement_strategies = strategies
        return rago_sweep.run_sweep_across_systems(**sweep_kwargs)

    if not result_dfs:
        return pd.DataFrame()
    return pd.concat(
        [result_dfs[index] for index in sorted(result_dfs)],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Iterative-RAG cost path (RAGO IterativeRAGSweep)
# ---------------------------------------------------------------------------

def _run_iterative_sweep(
    processed_dfs: dict,
    placement_policies: list,
    seq_len_inference_prefill: int,
    dec_steps: int,
    gpus_per_server: int,
    max_batch_size_request: int,
    no_microbatching: bool,
    scheduling_policy: str,
    min_num_gpus: int,
    max_num_gpus: int,
    retrieval_interval: int,
) -> pd.DataFrame:
    """Sweep IterativeRAGSweep across placement policies and chip allocations.

    RAGO's IterativeRAGSweep.assemble_cost_iterative_retrieval works on a
    SINGLE PhysicalMapping at a time (it's a discrete-event simulation per
    config). We loop chip allocations ourselves to mirror what RAGSweep does
    internally.

    The simulation prints a deprecation warning per call which we suppress —
    over a 200-iteration sweep that's a lot of stderr.
    """
    import contextlib
    import io
    import itertools

    from rago.config import PhysicalMapping, RetrievalPolicy, EncodeDBPolicy
    from rago.rago_sweep_iterative import IterativeRAGSweep
    from rago.utils import get_power_of_two_list

    # IterativeRAGSweep asserts stages == ["retrieval","prefill","decode"].
    iterative_stages = ["retrieval", "prefill", "decode"]

    rago_sweep = IterativeRAGSweep(
        retrieval_policy=RetrievalPolicy(
            run_retrieval=True, retrieval_pattern="iterative",
        ),
        encode_db_policy=EncodeDBPolicy(run_encode_db=False),
        stages=iterative_stages,
        sweep_df={
            "query_expansion_prefill": None,
            "query_expansion_decode": None,
            "encode": None,
            "retrieval": processed_dfs.get("retrieval"),
            "passage_reranker": None,
            "passage_filter": None,
            "passage_compressor_prefill": None,
            "passage_compressor_decode": None,
            "prefill": processed_dfs.get("prefill"),
            "decode": processed_dfs.get("decode"),
        },
        seq_len_inference_prefill=seq_len_inference_prefill,
        dec_steps=dec_steps,
        num_chips_per_server=gpus_per_server,
    )

    # Enumerate (num_retrieval_servers, num_chips_prefill, num_chips_decode)
    # power-of-two combinations subject to total chip budget.
    chip_choices = get_power_of_two_list(max(1, max_num_gpus // 2))  # cap individual stage
    # Keep search space sane: filter combinations summing within bounds.
    candidates = []
    for n_retr_srv in chip_choices:
        for n_pref in chip_choices:
            for n_dec in chip_choices:
                # retrieval is on CPU servers (counted separately as
                # num_retrieval_servers in RAGO), so chip budget = pref + dec.
                if n_pref + n_dec < min_num_gpus or n_pref + n_dec > max_num_gpus:
                    continue
                candidates.append((n_retr_srv, n_pref, n_dec))
    if not candidates:
        # Fallback to a single sane allocation if filtering yielded nothing.
        candidates = [(1, max(1, min_num_gpus // 2), max(1, max_num_gpus // 2))]

    all_rows = []
    for placement_policy in placement_policies:
        if placement_policy != "disaggregated":
            # IterativeRAGSweep only supports disaggregated.
            logger.warning(
                f"IterativeRAGSweep skips placement_policy={placement_policy!r} "
                f"(only 'disaggregated' supported in iterative mode)"
            )
            continue
        for n_retr_srv, n_pref, n_dec in candidates:
            mapping = PhysicalMapping(
                num_retrieval_servers=n_retr_srv,
                num_chips={"prefill": n_pref, "decode": n_dec},
                placement_policy=placement_policy,
            )
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    df = rago_sweep.assemble_cost_iterative_retrieval(
                        physical_mapping=mapping,
                        scheduling_policy=scheduling_policy,
                        max_batch_size_request=max_batch_size_request,
                        no_microbatching=no_microbatching,
                        retrieval_interval=retrieval_interval,
                    )
            except AssertionError as e:
                logger.warning(
                    f"Iterative sweep skipped (n_retr={n_retr_srv}, n_pref={n_pref}, "
                    f"n_dec={n_dec}, retrieval_interval={retrieval_interval}): {e}"
                )
                continue
            except Exception as e:
                logger.warning(
                    f"Iterative sweep failed (n_retr={n_retr_srv}, n_pref={n_pref}, "
                    f"n_dec={n_dec}): {e}"
                )
                continue
            if df is None or len(df) == 0:
                continue
            all_rows.append(df)

    if not all_rows:
        raise RuntimeError(
            f"IterativeRAGSweep produced no valid configurations across "
            f"{len(candidates)} chip allocations × {len(placement_policies)} placements."
        )
    return pd.concat(all_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# RAGCMAssembly
# ---------------------------------------------------------------------------


class RAGCMAssembly:
    """The unified cost-model entry: owns candidate generation + engine dispatch.

    Exactly three public entries (the frozen input contract):

    * :meth:`evaluate_dynamic` — trace-driven CM (sweep or fixed placement).
    * :meth:`evaluate_static` — config-driven CM without a per-query trace;
      the static evaluators' aggregate rides on ``schema.runtime_stats``.
    * :meth:`evaluate_fixed` — fixed-point prediction of ONE resolved
      deployment (``pipeline_config`` + resolved ``system_config`` [+ trace]),
      no sweep.

    Normal optimization and config-driven/static costing use RAGO to generate
    candidate rows. Trace-driven fixed-config costing constructs a single row
    directly. Both trace paths use the same explicit dynamic pipeline for
    analysis, shortlisting and simulation, followed by final selection."""

    _SWEEP_DF_CACHE: dict[str, dict[str, pd.DataFrame]] = {}

    def __init__(self, hardware: SystemConfig):
        self.hardware = hardware

    def _run_fixed_dynamic_candidate(
        self,
        pipeline_config: dict,
        stages: list[str],
        dag: list[dict] | dict | None,
        fixed_placement: "PhysicalMapping",
        fixed_batch_size_request: int,
        fixed_batch_size_decode: int,
        stage_cost_models: dict[str, CostModelModule] | None = None,
    ) -> AssemblyResult:
        """Create the one fixed-config candidate row for DynamicEngine.

        This path intentionally does not call RAGO: trace-driven fixed costing is
        not a placement/resource/batch sweep. RAGO remains the swept candidate
        generator for normal optimization and the config-driven StaticEngine path.
        """
        search_space = self.hardware.search_space or {}
        scheduling_policy = search_space.get(
            "scheduling_policy", "continuous-batching",
        )
        df = _build_fixed_candidate_df(
            stages=stages,
            fixed_placement=fixed_placement,
            batch_size_request=fixed_batch_size_request,
            batch_size_decode=fixed_batch_size_decode,
            scheduling_policy=scheduling_policy,
        )
        runtime = pipeline_config.get("pipeline_runtime") or {}
        df["pipeline_mode"] = runtime.get("mode", "sequential")
        df["num_retrievals"] = _get_num_retrievals(dag)

        seq_len_prefill = _resolve_main_prefill_tokens(search_space, dag)
        if seq_len_prefill is None:
            from rag_stack.cost_model.token_stats import TokenStats
            ts = TokenStats.resolve(pipeline_config)
            seq_len_prefill = getattr(ts, "avg_llm_input_tokens", None)
            if seq_len_prefill is None:
                seq_len_prefill = search_space.get("seq_len_inference_prefill")
            if seq_len_prefill is None:
                raise ValueError(
                    "Fixed dynamic CM requires generator input tokens from the "
                    "trace aggregate, _token_stats.avg_llm_input_tokens, or "
                    "search_space.seq_len_inference_prefill."
                )

        qe_seq_len = None
        if (
            "query_expansion_prefill" in stages
            or "query_expansion_decode" in stages
        ):
            from rag_stack.cost_model.token_stats import TokenStats
            qe_seq_len = TokenStats.resolve(
                pipeline_config
            ).get_query_expansion_input_tokens()

        from rag_stack.cost_model.communication import apply_communication_costs
        from rag_stack.cost_model.candidate_layout import (
            materialize_candidate_system_layouts,
        )
        df = materialize_candidate_system_layouts(
            df,
            pipeline_config=pipeline_config,
            prefer_system_layout=True,
        )
        if stage_cost_models is None:
            stage_cost_models = build_performance_stage_cost_models(
                pipeline_config,
            )
        from rag_stack.cost_model.candidate_reprice import (
            require_candidate_genz_parallelism,
        )
        require_candidate_genz_parallelism(
            df,
            pipeline_config=pipeline_config,
            stage_cost_models=stage_cost_models,
        )
        df = apply_communication_costs(
            df,
            hardware=self.hardware,
            pipeline_config=pipeline_config,
            seq_len_prefill=int(seq_len_prefill),
            qe_seq_len=qe_seq_len,
            reranker_input_fanout=_get_reranker_input_fanout(
                pipeline_config, dag,
            ),
            query_expansion_fanout=_get_retrieval_backend_fanout(
                pipeline_config, dag,
            ),
        )
        return AssemblyResult(
            performance_score=0.0,
            best_hw_params={},
            rago_result_df=df,
            sweep_dfs=None,
        )

    @_cm_serialized
    def evaluate_static(
        self,
        pipeline_config: dict,
        *,
        workflow_schema: WorkflowSchema | dict | None = None,
        fixed_placement: "PhysicalMapping | None" = None,
        fixed_batch_size_request: int | None = None,
        fixed_batch_size_decode: int | None = None,
        stage_cost_models: dict[str, CostModelModule] | None = None,
    ) -> AssemblyResult:
        """Config-driven CM path (no per-query trace).

        ``workflow_schema`` carries structure/stage order; its
        ``runtime_stats`` field is the optional aggregate profile produced by
        static evaluators — it is not a trace and never routes to
        :class:`DynamicEngine`.
        """
        from rag_stack.cost_model.engines import StaticEngine
        from rag_stack.cost_model.support import require_cm_pipeline_support
        require_cm_pipeline_support(pipeline_config)

        schema = _resolve_workflow_schema(
            pipeline_config,
            workflow_schema,
            list(stage_cost_models.keys()) if stage_cost_models else None,
        )
        if getattr(schema, "dataflow_type", "sequential") != "sequential":
            raise ValueError(
                "evaluate_static requires a sequential workflow_schema. "
                "Pass trace=... to evaluate_dynamic for agentic/iterative workflows."
            )
        two_role_engines = _two_role_llm_engines_in_schema(schema)
        if two_role_engines:
            raise RuntimeError(
                "evaluate_static cannot publish serving QPS for two-role LLM "
                f"engine(s) {list(two_role_engines)!r}: the no-trace path has "
                "no scheduler-cycle composition, so its phase sweep is "
                "component-only and cannot consume the calibrated engine-cycle "
                "curve. Pass a quality trace to evaluate_dynamic/evaluate_fixed "
                "so authoritative DES can price each real engine cycle once."
            )

        pipeline_config = _materialize_corpus_stats(pipeline_config, schema)
        runtime_stats = schema.runtime_stats
        pipeline_config = _with_rag_ir_token_overrides(
            pipeline_config, runtime_stats,
        )
        ra = self._run_sweep(
            pipeline_config,
            stage_cost_models,
            dag=runtime_stats,
            fixed_placement=fixed_placement,
            fixed_batch_size_request=fixed_batch_size_request,
            fixed_batch_size_decode=fixed_batch_size_decode,
        )
        sweep_dfs = getattr(ra, "sweep_dfs", None) or {}
        if workflow_schema is None:
            schema = _resolve_workflow_schema(
                pipeline_config,
                None,
                list(sweep_dfs.keys()) or None,
            )

        df = StaticEngine(schema).cost(
            ra.rago_result_df,
            sweep_dfs,
            pipeline_config=pipeline_config,
            hardware=self.hardware,
        )
        if df is None or len(df) == 0:
            raise RuntimeError(
                "Static cost path produced no feasible configurations."
            )
        best_hw_params, score = apply_selection(df, self.hardware.performance)
        return AssemblyResult(
            performance_score=score,
            best_hw_params=best_hw_params,
            rago_result_df=df,
            sweep_dfs=None if fixed_placement is not None else sweep_dfs,
        )

    @_cm_serialized
    def evaluate_dynamic(
        self,
        pipeline_config: dict,
        *,
        workflow_schema: WorkflowSchema | dict | None = None,
        trace: QualityTraceInput | None = None,
        fixed_placement: "PhysicalMapping | None" = None,
        fixed_batch_size_request: int | None = None,
        fixed_batch_size_decode: int | None = None,
        stage_cost_models: dict[str, CostModelModule] | None = None,
    ) -> AssemblyResult:
        """Trace-driven CM path.

        ``trace`` is the only dynamic data input. Its aggregate profile supplies
        representative token lengths and invocation multiplicities to component
        sweeps and candidate construction. The explicit dynamic pipeline then
        analyzes the same retained per-call traces, shortlists candidates using
        the configured objective, and simulates final throughput and latency.
        """
        from rag_stack.cost_model.engines import DynamicEngine
        from rag_stack.cost_model.dynamic_pipeline import evaluate_candidates
        from rag_stack.cost_model.workload import prepare_trace_workload
        from rag_stack.rag_ir.normalize import with_observed_trace_components
        from rag_stack.cost_model.support import require_cm_pipeline_support
        require_cm_pipeline_support(pipeline_config)

        if trace is None:
            raise RuntimeError("RAGCMAssembly.evaluate_dynamic requires trace")
        schema = _coerce_workflow_schema(workflow_schema)
        pipeline_config = _materialize_corpus_stats(pipeline_config, schema)
        workload = prepare_trace_workload(
            trace,
            query_expansion_fanout=_get_query_expansion_fanout(pipeline_config),
        )
        traces = workload.execution_paths
        pipeline_config = with_observed_trace_components(pipeline_config, traces)
        if pipeline_config["_evaluator_fusion_calls"]:
            if schema is not None and "retrieval_fusion" not in schema.stages:
                raise ValueError(
                    "Evaluator trace records retrieval_fusion but the workflow schema "
                    "omits it; build the schema with build_workflow_schema(config, trace=trace)"
                )
            if stage_cost_models is not None and "retrieval_fusion" not in stage_cost_models:
                raise ValueError("Observed retrieval_fusion has no supplied component model")
        trace_profile = workload.summary
        normalized = workload.normalization
        logger.info(
            "CM quality trace: scope=%s normalization=%s input=%d complete=%d "
            "kept=%d priced=%d lossy=%s",
            normalized.scope,
            normalized.normalization,
            normalized.input_queries,
            normalized.complete_queries,
            normalized.kept_queries,
            len(traces),
            normalized.lossy,
        )
        pipeline_config = _with_rag_ir_token_overrides(
            pipeline_config, trace_profile,
        )

        if fixed_placement is not None:
            fixed_request_batch = _fixed_batch_from_config(
                pipeline_config, fixed_batch_size_request,
            )
            fixed_decode_batch = _fixed_decode_batch_from_config(
                pipeline_config, fixed_batch_size_decode, fixed_request_batch,
            )
            if schema is None:
                schema = _resolve_workflow_schema(
                    pipeline_config,
                    None,
                    list(stage_cost_models.keys()) if stage_cost_models else None,
                )
            stages = list(schema.stages)
            if not stages:
                if stage_cost_models is None:
                    stage_cost_models = build_performance_stage_cost_models(pipeline_config)
                stages = list(stage_cost_models.keys())
                schema = build_workflow_schema(pipeline_config, stages)
            ra = self._run_fixed_dynamic_candidate(
                pipeline_config,
                stages,
                trace_profile,
                fixed_placement,
                fixed_request_batch,
                fixed_decode_batch,
                stage_cost_models,
            )
            sweep_dfs = {}
        else:
            ra = self._run_sweep(
                pipeline_config,
                stage_cost_models,
                dag=trace_profile,
                fixed_batch_size_request=fixed_batch_size_request,
                fixed_batch_size_decode=fixed_batch_size_decode,
            )
            sweep_dfs = getattr(ra, "sweep_dfs", None) or {}
            if schema is None:
                schema = _resolve_workflow_schema(
                    pipeline_config,
                    None,
                    list(sweep_dfs.keys()) or None,
                )

        df = ra.rago_result_df
        if df is None or len(df) == 0:
            raise RuntimeError(
                f"Trace cost path produced no feasible configurations "
                f"(n_traces={len(traces)})."
            )
        df = evaluate_candidates(
            engine=DynamicEngine(schema),
            df=df,
            sweep_dfs=sweep_dfs,
            pipeline_config=pipeline_config,
            hardware=self.hardware,
            traces=traces,
            trace_source_question_ids=workload.source_question_ids,
            fixed_placement=fixed_placement,
            prepared_paths=workload.prepared_paths,
        )
        if df is None or len(df) == 0:
            raise RuntimeError(
                "Dynamic cost path produced no feasible configurations after analysis and simulation."
            )
        df = _attach_quality_trace_diagnostics(
            df,
            normalized,
            priced_queries=len(traces),
        )
        best_hw_params, score = apply_selection(df, self.hardware.performance)
        return AssemblyResult(
            performance_score=score,
            best_hw_params=best_hw_params,
            rago_result_df=df,
            sweep_dfs=None if fixed_placement is not None else sweep_dfs,
        )

    def _run_sweep(
        self,
        pipeline_config: dict,
        stage_cost_models: dict[str, CostModelModule] | None = None,
        *,
        dag: dict | None = None,
        fixed_placement: "PhysicalMapping | None" = None,
        fixed_batch_size_request: int | None = None,
        fixed_batch_size_decode: int | None = None,
    ) -> AssemblyResult:
        """Build component tables and enumerate feasible system candidates.

        ``dag`` is the legacy keyword for a derived workload summary dict;
        it is not a dependency graph or the ordered trace. Its representative
        lengths and invocation counts feed component models, memory checks and
        communication estimates. With no summary, configuration defaults apply.

        RAGO also computes coarse performance columns. The static path uses
        them; the dynamic path replaces them in ``DynamicEngine.analyze``.
        This method neither shortlists dynamic candidates nor selects a winner.
        """
        pipeline_config = _with_rag_ir_token_overrides(pipeline_config, dag)
        if stage_cost_models is None:
            stage_cost_models = build_performance_stage_cost_models(pipeline_config)
        search_space = self.hardware.search_space or {}
        default_max_batch_size = int(search_space["max_batch_size_request"])
        assert (
            default_max_batch_size >= 1
            and (default_max_batch_size & (default_max_batch_size - 1)) == 0
        ), (
            f"search_space.max_batch_size_request must be a positive power of 2 "
            f"(got {default_max_batch_size})"
        )

        sweep_dfs: dict[str, pd.DataFrame | None] = {
            "semantic_retrieval_vectorsearch": None, "generator_prefill": None, "generator_decode": None,
        }
        fixed_request_batch: int | None = None
        fixed_decode_batch: int | None = None
        if fixed_placement is not None:
            fixed_request_batch = _fixed_batch_from_config(
                pipeline_config, fixed_batch_size_request,
            )
            fixed_decode_batch = _fixed_decode_batch_from_config(
                pipeline_config, fixed_batch_size_decode, fixed_request_batch,
            )

        print("Start to generate sweep DataFrames for stages:", list(stage_cost_models.keys()))
        import time as _time
        _t0 = _time.perf_counter()

        # Generate sweep DataFrames in parallel across stages.
        # Use ProcessPoolExecutor (not Thread) because the per-stage work is
        # CPU-bound Python (GenZ / llm_sim / pandas ops) that holds the GIL;
        # threads would serialize. Spawn context avoids forking the parent's
        # address space (CUDA contexts, loaded datasets, tokenizers) into
        # every worker.
        import os as _os
        from concurrent.futures import as_completed
        from concurrent.futures.process import BrokenProcessPool

        n_workers = _cm_process_worker_count(len(stage_cost_models))
        stages = list(stage_cost_models.keys())

        # Per-stage batch ceilings — computed ONCE and reused across pool retries
        # and the serial fallback. ``stage_user_max`` is in user-request units
        # (matches max_batch_size_request); call-unit stages (retrieval, encode,
        # passage_reranker — see _get_stage_fanout) are multiplied by the per-user
        # fanout so the sweep reaches the same user-unit ceiling AFTER
        # RAGSweep._apply_fanout converts call-units back to user-units downstream.
        stage_specs = []  # list[(stage_name, cost_model, stage_call_max)]
        for stage_name, cost_model in stage_cost_models.items():
            if fixed_request_batch is not None and fixed_decode_batch is not None:
                stage_user_max = _fixed_stage_user_batch(
                    stage_name, fixed_request_batch, fixed_decode_batch,
                )
            else:
                stage_user_max = _resolve_max_batch_size_for_stage(
                    stage_name, pipeline_config, default_max=default_max_batch_size,
                )
            stage_fanout = _get_stage_fanout(stage_name, pipeline_config, dag)
            stage_specs.append((stage_name, cost_model, stage_user_max * stage_fanout))

        # Pinned-placement optimization: when ``fixed_placement`` is set we price
        # exactly ONE stage→group/chip partition, so the per-stage GenZ sweep only
        # needs the device counts that placement actually uses — a single-GPU
        # placement needs only num_devices=1, not the full 1,2,4,… grid up to
        # ``max_num_gpus_per_stage``. Cap the SWEEP-GENERATION hardware's per-stage
        # GPU ceiling to the placement's max chips. Strictly scoped to sweep df
        # generation: ``self.hardware`` and the ``search_space`` handed to
        # ``_run_rago_sweep`` / assemble are untouched, so the normal
        # placement-sweep path and the assemble math stay byte-for-byte identical.
        # (assemble at a group of N chips reads num_devices=N rows; capping the
        # grid to max(N) keeps every row assemble can reference.)
        sweep_hardware = self.hardware
        if fixed_placement is not None and _os.environ.get(
            "CM_PIN_SWEEP_GRID", "1"
        ).strip().lower() not in ("0", "false", "no", "off"):
            _nc = fixed_placement.num_chips
            _cap = max(_nc.values()) if isinstance(_nc, dict) else max(_nc)
            _ss = dict(self.hardware.search_space or {})
            _cur = _ss.get("max_num_gpus_per_stage", _ss.get("max_num_gpus", _cap))
            if isinstance(_cur, list):
                _cur = max(_cur)
            _ss["max_num_gpus_per_stage"] = min(int(_cur), int(_cap))
            from dataclasses import replace as _dc_replace
            sweep_hardware = _dc_replace(self.hardware, search_space=_ss)

        # Sweep-df cache token bucketing (07-06, opt-in, DEFAULT OFF — user
        # ruling: it introduces a perf approximation, so campaigns that must be
        # CM-comparable to exact-keyed baselines keep it off). Measured: 0/70
        # cache hits across two full optimization runs, because the key embeds
        # the dag's per-eval MEASURED token stats — no two evals ever share a
        # key, so every pricing pays the full sweep regeneration. With a bucket
        # B set (e.g. 128), token stats are rounded to B-multiples IN THE KEY
        # ONLY: same-structure evals with workloads within a bucket reuse the
        # first eval's sweep dfs (a miss still computes from the EXACT dag).
        # Enable per run via config: system.cm_sweep_cache_token_bucket: 128
        # (env CM_SWEEP_CACHE_TOKEN_BUCKET overrides for one-off experiments).
        _tok_bucket = 0
        try:
            _tok_bucket = int(
                (pipeline_config.get("system") or {}).get("cm_sweep_cache_token_bucket") or 0
            )
        except (ValueError, TypeError, AttributeError):
            _tok_bucket = 0
        _env_bucket = _os.environ.get("CM_SWEEP_CACHE_TOKEN_BUCKET", "").strip()
        if _env_bucket:
            try:
                _tok_bucket = int(_env_bucket)
            except ValueError:
                pass
        _cache_payload = {
            "pipeline_config": pipeline_config,
            "dag": _bucketed_dag_for_key(dag, _tok_bucket) if _tok_bucket > 0 else dag,
            "hardware": _cacheable(sweep_hardware),
            "stage_specs": [
                (stage_name, int(stage_call_max))
                for stage_name, _cost_model, stage_call_max in stage_specs
            ],
            "fixed_placement": (
                {
                    "num_chips": getattr(fixed_placement, "num_chips", None),
                    "num_retrieval_servers": getattr(
                        fixed_placement, "num_retrieval_servers", None,
                    ),
                    "batch_size_request": fixed_request_batch,
                    "batch_size_decode": fixed_decode_batch,
                }
                if fixed_placement is not None else None
            ),
        }
        _cache_key = hashlib.sha256(
            json.dumps(_cacheable(_cache_payload), sort_keys=True, default=str).encode()
        ).hexdigest()

        def _sweep_via_pool() -> dict[str, "pd.DataFrame"]:
            """Run every stage sweep on the PERSISTENT spawn pool (07-06, perf):
            the old code built a fresh ProcessPoolExecutor per pricing, paying
            worker spawn + full GenZ/pandas import (~10-15s of every ~47s sweep)
            on EVERY evaluate_* call. The shared warm pool (same one the
            collocation sweep uses) pays that once per process. A real
            per-stage error propagates (a genuinely-bad config is still marked
            INVALID); a BrokenProcessPool RESETS the shared pool (a broken
            executor rejects all future work) and propagates for the caller to
            retry — the retry then builds a fresh warm pool."""
            out: dict[str, "pd.DataFrame"] = {}
            executor = _collocation_pool(n_workers)
            try:
                futures = {
                    executor.submit(
                        _generate_stage_sweep_df,
                        cost_model, pipeline_config, sweep_hardware,
                        stage_name, stage_call_max,
                    ): stage_name
                    for stage_name, cost_model, stage_call_max in stage_specs
                }
                for future in as_completed(futures):
                    stage_name = futures[future]
                    try:
                        out[stage_name] = future.result()
                    except BrokenProcessPool:
                        raise  # handled below: reset + let the caller retry
                    except Exception as e:
                        logger.warning(
                            f"Stage '{stage_name}' sweep failed in subprocess: {e}"
                        )
                        raise
            except BrokenProcessPool:
                _collocation_pool_reset()
                raise
            return {
                stage_name: out[stage_name]
                for stage_name, _cost_model, _stage_call_max in stage_specs
            }

        def _sweep_serial() -> dict[str, "pd.DataFrame"]:
            """Last-resort in-process sweep — immune to the spawn-pool flakiness
            (slower, no parallelism). A per-stage error still propagates."""
            return {
                stage_name: _generate_stage_sweep_df(
                    cost_model, pipeline_config, sweep_hardware,
                    stage_name, stage_call_max,
                )
                for stage_name, cost_model, stage_call_max in stage_specs
            }

        # CM_SWEEP_SERIAL=1: skip the spawn pool entirely and build sweep_dfs in-process.
        # The spawn pool re-imports the whole (heavy: faiss/CUDA/transformers/ax) rag_stack
        # in every worker, EVERY call — ~tens of seconds of pure overhead per evaluate().
        # For a serial driver that evaluates many configs back-to-back (the benchmark, or
        # any single-config probe), that overhead dwarfs the few-second GenZ modeling, so
        # in-process is much faster. Default OFF (the optimizer wants the parallelism).
        _cached_sweeps = self._SWEEP_DF_CACHE.get(_cache_key)
        if _cached_sweeps is not None:
            sweep_dfs.update({k: v.copy(deep=True) for k, v in _cached_sweeps.items()})
            print(
                f"Successfully generated sweep DataFrames for stages: {stages}  "
                f"(cache hit, elapsed {_time.perf_counter() - _t0:.2f}s)"
            )
        else:
            if (
                n_workers <= 1
                or _os.environ.get("CM_SWEEP_SERIAL", "").strip().lower()
                in ("1", "true", "yes")
            ):
                sweep_dfs.update(_sweep_serial())
            else:
                # The spawn pool occasionally dies abruptly (``BrokenProcessPool``) in the
                # long-run / vLLM-churned context. This was empirically shown to be
                # TRANSIENT (55+ isolated reproductions never triggered it), so a fresh
                # pool very likely succeeds; serial is the final safety net. We retry /
                # fall back ONLY on BrokenProcessPool — a real per-stage error still
                # propagates so the controller marks the config INVALID (no silent mask).
                try:
                    sweep_dfs.update(_sweep_via_pool())
                except BrokenProcessPool as e:
                    logger.warning(
                        f"CM stage-sweep pool died abruptly ({e}); retrying once with a fresh pool."
                    )
                    try:
                        sweep_dfs.update(_sweep_via_pool())
                    except BrokenProcessPool as e2:
                        logger.warning(
                            f"CM stage-sweep pool died again ({e2}); falling back to SERIAL execution."
                        )
                        sweep_dfs.update(_sweep_serial())
            self._SWEEP_DF_CACHE[_cache_key] = {
                k: v.copy(deep=True)
                for k, v in sweep_dfs.items()
                if isinstance(v, pd.DataFrame)
            }
            print(f"Successfully generated sweep DataFrames for stages: {stages}  "
                  f"(elapsed {_time.perf_counter() - _t0:.2f}s)")

        if (
            fixed_placement is not None
            and fixed_request_batch is not None
            and fixed_decode_batch is not None
        ):
            filtered: dict[str, pd.DataFrame | None] = {}
            for stage_name, df in sweep_dfs.items():
                stage_batch = _fixed_stage_user_batch(
                    stage_name, fixed_request_batch, fixed_decode_batch,
                )
                filtered[stage_name] = _filter_stage_sweep_df_for_fixed(
                    df,
                    stage=stage_name,
                    fixed_placement=fixed_placement,
                    batch_size=stage_batch * _get_stage_fanout(
                        stage_name, pipeline_config, dag,
                    ),
                )
            sweep_dfs = filtered

        if not stages:
            return AssemblyResult(performance_score=0.0, best_hw_params={})

        

        result = self._run_rago_sweep(
            stages, sweep_dfs, search_space, pipeline_config, dag=dag,
            stage_cost_models=stage_cost_models,
            fixed_placement=fixed_placement,
            fixed_batch_size_request=fixed_request_batch,
            fixed_batch_size_decode=fixed_decode_batch,
        )
        # Attach per-stage sweep DataFrames for diagnostics
        result.sweep_dfs = {
            k: v for k, v in sweep_dfs.items()
            if v is not None and len(v) > 0
        }
        return result

    @_cm_serialized
    def compile_fixed(self, pipeline_config: dict, system_config: dict, *, trace):
        """Compile one concrete deployment without running analysis or DES.

        The returned plan contains the production simulator's actual physical
        graph and bound service functions. ``compiled.describe`` is its public
        preview; ``analyze_plan`` and ``simulate_plan`` consume that same object.
        """
        from dataclasses import replace
        from rago.config import PhysicalMapping
        from rag_stack.system_layout import cm_fixed_pipeline_config, cm_fixed_placement_inputs
        from rag_stack.cost_model.compiled import compile_workload, bind_plan, CompilationBlocked, _blocked

        try:
            config = cm_fixed_pipeline_config(pipeline_config, system_config)
            workload = compile_workload(config, trace)
            config = workload.pipeline_config
            coll, chips, servers, stages = cm_fixed_placement_inputs(system_config, config)
            schema = build_workflow_schema(config, stages)
            config = _materialize_corpus_stats(config, schema)
            config = _with_rag_ir_token_overrides(config, workload.summary)
            workload = replace(workload, pipeline_config=config)
            request_batch = _fixed_batch_from_config(config, system_config.get("batch_size_request"))
            decode_batch = _fixed_decode_batch_from_config(
                config, system_config.get("batch_size_decode"), request_batch,
            )
            placement = PhysicalMapping(
                num_chips=dict(chips), num_retrieval_servers=int(servers),
                placement_policy="collocated",
                collocation_strategy=_rago_strategy_from_resource_groups(coll, list(schema.stages)),
            )
            candidate = self._run_fixed_dynamic_candidate(
                config, list(schema.stages), workload.summary, placement,
                request_batch, decode_batch,
            )
            if candidate.rago_result_df is None or len(candidate.rago_result_df) != 1:
                raise CompilationBlocked("bind_plan", "invalid_fixed_candidate",
                                         "A fixed deployment must produce exactly one candidate")
            return bind_plan(workload, hardware=self.hardware,
                             candidate_row=candidate.rago_result_df.iloc[0])
        except CompilationBlocked:
            raise
        except Exception as exc:
            raise _blocked("bind_plan", exc) from exc

    def analyze_plan(self, plan):
        """Analyze a compiled fixed plan without recompiling its execution path."""
        from rag_stack.cost_model.engines import DynamicEngine
        schema = build_workflow_schema(plan.pipeline_config)
        return DynamicEngine(schema).analyze(
            pd.DataFrame([plan.candidate_row]), {}, pipeline_config=plan.pipeline_config,
            hardware=plan.hardware, traces=plan.workload.execution_paths,
            trace_source_question_ids=plan.workload.source_question_ids,
            compiled_plan=plan,
        )

    def simulate_plan(self, plan):
        from rag_stack.cost_model.compiled import simulate_plan
        return simulate_plan(plan)

    def evaluate_fixed(
        self,
        pipeline_config: dict,
        system_config: dict,
        *,
        trace: QualityTraceInput | None = None,
        runtime_stats: dict | None = None,
    ) -> AssemblyResult:
        """THE fixed entry: price ONE resolved deployment — no sweep.

        ``system_config`` is the resolved deployment layout JSON
        (``system_config_resolved.json`` shape: ``layout.resource_groups`` +
        per-stage devices, ``batch_size_request``/``batch_size_decode``,
        runtime blocks). It is folded into the pipeline config
        (:func:`rag_stack.system_layout.cm_fixed_pipeline_config`) and
        reconstructed into the CM placement inputs
        (:func:`rag_stack.system_layout.cm_fixed_placement_inputs`) — exactly
        the battle-tested rag_cm_accuracy reconstruction, now in one place.

        ``trace`` → :class:`DynamicEngine` replay of the quality run's
        recorded execution at this placement; ``trace=None`` → the static
        engine prices the fixed placement, reading the optional aggregate
        from ``runtime_stats`` (stamped onto the internally-built schema's
        ``runtime_stats`` field — the sanctioned no-trace vehicle).

        Corpus statistics are enforced at the delegated entries
        (:meth:`evaluate_static` / :meth:`evaluate_dynamic` materialize
        ``schema.corpus_stats`` back into the priced config), so this entry
        inherits the frozen input contract.
        """
        from rag_stack.system_layout import (
            cm_fixed_pipeline_config,
            cm_fixed_placement_inputs,
        )

        from rag_stack.cost_model.support import require_cm_pipeline_support
        require_cm_pipeline_support(pipeline_config)
        pcfg_cm = cm_fixed_pipeline_config(pipeline_config, system_config)
        if trace is not None:
            from rag_stack.rag_ir.normalize import with_observed_trace_components
            from rag_stack.rag_ir.trace import normalize_quality_trace

            pcfg_cm = with_observed_trace_components(
                pcfg_cm, normalize_quality_trace(trace).traces,
            )
        coll, num_chips, retrieval_servers, stages = cm_fixed_placement_inputs(
            system_config, pcfg_cm,
        )
        schema = build_workflow_schema(pcfg_cm, stages)
        if runtime_stats is not None:
            schema.runtime_stats = runtime_stats
        batch_size_request = system_config.get("batch_size_request")
        batch_size_decode = system_config.get(
            "batch_size_decode", batch_size_request,
        )
        logger.info(
            "[cm-fixed] pricing deployment preset=%s groups=%s chips=%s "
            "batch=%s/%s retrieval_servers=%s",
            system_config.get("deployment_preset", "<raw>"),
            coll, num_chips, batch_size_request, batch_size_decode,
            retrieval_servers,
        )
        return self._evaluate_fixed_placement(
            pcfg_cm,
            schema,
            trace=trace,
            collocation_strategy=coll,
            num_chips=num_chips,
            num_retrieval_servers=retrieval_servers,
            batch_size_request=batch_size_request,
            batch_size_decode=batch_size_decode,
        )

    def _evaluate_fixed_placement(
        self,
        pipeline_config: dict,
        workflow_schema: WorkflowSchema | dict | None = None,
        *,
        trace: QualityTraceInput | None = None,
        collocation_strategy: list[list[str]],
        num_chips: dict[str, int],
        num_retrieval_servers: int = 1,
        placement_policy: str = "collocated",
        batch_size_request: int | None = None,
        batch_size_decode: int | None = None,
        stage_cost_models: dict[str, CostModelModule] | None = None,
    ) -> AssemblyResult:
        """Cost a SINGLE fixed deployment — no placement/batch/resource sweep.

        Internal: public callers go through :meth:`evaluate_fixed`, which
        reconstructs these placement inputs from a resolved ``system_config``.

        The deployment is fully described by ONE partition:

        * ``collocation_strategy`` — resource groups. CPU ``retrieval`` must be
          its own group; it is represented by ``num_retrieval_servers`` and never
          by GPU chips. Pass ``workflow_schema`` to define execution order.
          Generator 1P1D, collocation, and unified serving all FALL OUT of this
          partition — prefill & decode in different groups ⇒ 1P1D (a priced KV
          hop); same group ⇒ unified. A GPU group allotted >1 chip is
          parallelised within it.
        * ``num_chips`` — chips per PerformanceStage, e.g.
          ``{"generator_prefill":1,"generator_decode":1,"passage_reranker":1,...}``;
          a group's chip count is the max over its stages.
        * ``num_retrieval_servers`` — CPU retrieval servers for the FAISS stage.
        * ``batch_size_request`` — request batch for the fixed deployment. If not
          passed, read ``pipeline_config["system"]["batch_size_request"]``.
        * ``batch_size_decode`` — decode engine concurrency. If not passed, read
          ``pipeline_config["system"]["batch_size_decode"]`` or default to request
          batch.

        Returns one row in ``rago_result_df`` for the fixed config. Temporary
        per-stage tables are used internally by RAGO / DynamicEngine but are not
        returned as diagnostic design-space ``sweep_dfs``.
        """
        from rago.config import PhysicalMapping

        schema = _resolve_workflow_schema(pipeline_config, workflow_schema)
        rago_collocation_strategy = _rago_strategy_from_resource_groups(
            collocation_strategy,
            list(schema.stages),
        )
        fixed_request_batch = _fixed_batch_from_config(
            pipeline_config, batch_size_request,
        )
        fixed_decode_batch = _fixed_decode_batch_from_config(
            pipeline_config, batch_size_decode, fixed_request_batch,
        )
        pm = PhysicalMapping(
            num_chips=dict(num_chips),
            num_retrieval_servers=int(num_retrieval_servers),
            placement_policy=placement_policy,
            collocation_strategy=rago_collocation_strategy,
        )
        if trace is not None:
            return self.evaluate_dynamic(
                pipeline_config,
                workflow_schema=schema,
                trace=trace,
                fixed_placement=pm,
                fixed_batch_size_request=fixed_request_batch,
                fixed_batch_size_decode=fixed_decode_batch,
                stage_cost_models=stage_cost_models,
            )
        # fixed + no trace = the static engine priced at this one placement;
        # the no-trace aggregate (if any) rides on schema.runtime_stats.
        return self.evaluate_static(
            pipeline_config,
            workflow_schema=schema,
            fixed_placement=pm,
            fixed_batch_size_request=fixed_request_batch,
            fixed_batch_size_decode=fixed_decode_batch,
            stage_cost_models=stage_cost_models,
        )

    def _run_rago_sweep(
        self,
        stages: list[str],
        sweep_dfs: dict[str, pd.DataFrame | None],
        search_space: dict,
        pipeline_config: dict,
        dag: dict | None = None,
        stage_cost_models: dict[str, CostModelModule] | None = None,
        fixed_placement: "PhysicalMapping | None" = None,
        fixed_batch_size_request: int | None = None,
        fixed_batch_size_decode: int | None = None,
    ) -> AssemblyResult:
        """Run RAGO's RAGSweep for full assembly with task placement.

        Follows the original RAGO example pattern (rago/examples/example_case1.py):
        1. Preprocess DataFrames into RAGO format
        2. Init RAGSweep with policies and stage data
        3. Sweep across placement policies × resource allocations
        4. Combine all results and apply selection policy
        """
        # Preprocess DataFrames into RAGO's expected format
        # ── RAGO boundary (inbound) ─────────────────────────────────────────
        # Everything above this point speaks PerformanceStage names; RAGO ids
        # exist only between here and the qualify step on RAGO's return value.
        from rag_stack.cost_model.rago_adapter import (
            RagoNameMap, to_rago_stage, to_rago_sweep_dfs,
        )
        from rag_stack.taxonomy import TAXONOMY

        name_map = RagoNameMap(list(stages))
        rago_stages = [s for s in stages if s != "retrieval_fusion"]
        lowered_sweep_dfs = to_rago_sweep_dfs(
            sweep_dfs,
            retrieval_fanout=_get_query_expansion_fanout(pipeline_config),
            max_retrieval_servers=max(
                1, int(search_space["max_num_gpus"]),
                int(getattr(fixed_placement, "num_retrieval_servers", 1) or 1),
            ),
        )

        # Preprocess DataFrames into RAGO's expected format (RAGO-id keyed —
        # RAGSweep's sweep_df dict must carry every slot it knows about).
        processed_dfs: dict[str, Any] = {
            "query_expansion_prefill": None,
            "query_expansion_decode": None,
            "encode": None,
            "retrieval": None,
            "passage_reranker": None,
            "passage_filter": None,
            "passage_compressor_prefill": None,
            "passage_compressor_decode": None,
            "prefill": None,
            "decode": None,
        }
        for rago_id, df in lowered_sweep_dfs.items():
            if df is None or len(df) == 0:
                continue
            stage_name = name_map.from_rago(rago_id)
            if rago_id == "retrieval":
                processed_dfs[rago_id] = preprocess_data(
                    df, stage="retrieval", return_pareto=False
                )
            else:
                # generate_sweep_df already produces RAGO-compatible format.
                # Forward/single-role stages may already be Pareto-filtered;
                # two-role LLM phases deliberately retain every batch until
                # authoritative DES has the engine-cycle curve.  Therefore
                # this boundary must keep return_pareto=False.  preprocess_data
                # only needs the GenZ sim kind ("prefill"/"decode") except for
                # the main generator pair, which keeps its own RAGO id.
                if rago_id in ("prefill", "decode"):
                    preprocess_stage = rago_id
                else:
                    preprocess_stage = TAXONOMY.performance_stage(stage_name).kind
                processed_dfs[rago_id] = preprocess_data(
                    df, stage=preprocess_stage, return_pareto=False,
                )

        # RAGSweep requires the generator prefill and decode data to function.
        # If either is missing (e.g. cost model failed), fall back early.
        for required_stage in ("generator_prefill", "generator_decode"):
            if (required_stage in stages
                    and processed_dfs.get(to_rago_stage(required_stage)) is None):
                raise ValueError(
                    f"No sweep data for required stage '{required_stage}'. "
                    f"Cost model may have failed to generate predictions."
                )

        # RAGO config from YAML search_space. Static no-CM-init runs may pass
        # a rag_ir aggregate DAG with runtime token stats; those win over the
        # offline YAML-injected defaults for this evaluation only.
        seq_len_inference_prefill = _resolve_main_prefill_tokens(search_space, dag)
        for sn in ("prefill", "decode"):
            df = processed_dfs.get(sn)
            if df is not None and len(df) > 0:
                if "seq_len_inference_prefill" in df.columns:
                    seq_len_inference_prefill = int(df["seq_len_inference_prefill"].iloc[0])
                break
        dec_steps = _resolve_main_dec_steps(search_space, dag)
        gpus_per_server = search_space["gpus_per_server"]
        max_batch_size_request = search_space["max_batch_size_request"]
        scheduling_policy = search_space.get("scheduling_policy", "continuous-batching")
        no_microbatching = bool(search_space.get("no_microbatching", True))
        if not no_microbatching:
            raise ValueError(
                "RAG-CM requires cm_search_space.no_microbatching=true: "
                "pre-decode stages share batch_size_request while generator "
                "decode is the single independent batch_size_decode axis"
            )
        possible_batch_sizes_decode = _requested_decode_batch_grid(
            pipeline_config,
            int(max_batch_size_request),
        )
        # Single source of truth for whether the decode stage may share a GPU
        # group with other stages (prefill, compressor, ...) under a collocated
        # multi-GPU placement. Default True: a collocated group is replicated
        # data-parallel across its chips (a unified prefill+decode engine), which
        # matches measured-mode serving. When False, decode is always forced into
        # its own group (RAGO's disaggregate_decode=True heuristic).
        allow_decode_collocation = bool(
            search_space.get("allow_decode_collocation", True)
        )

        # Resource allocation ranges
        min_num_gpus = search_space["min_num_gpus"]
        max_num_gpus = search_space["max_num_gpus"]
        max_num_gpus_per_stage = search_space["max_num_gpus_per_stage"]

        # Placement policies to sweep (can be a list)
        placement_policies = search_space.get("placement_policy", ["disaggregated"])
        if isinstance(placement_policies, str):
            placement_policies = [placement_policies]

        # Build RAGO config objects
        has_passage_reranker = "passage_reranker" in stages
        has_query_expansion = "query_expansion_prefill" in stages
        from rag_stack.cost_model.token_stats import TokenStats
        ts = TokenStats.resolve(pipeline_config)
        passage_reranker_seq_len = ts.get_passage_reranker_input_tokens()
        # See ``_get_reranker_input_fanout`` for the why-not-passage_reranker.top_k.
        # Note: this already includes the × query_expansion_fanout factor
        # when query_expansion is enabled (rerank sees N × top_k pairs per
        # user request under the no-dedup assumption).
        passage_reranker_topk = _get_reranker_input_fanout(pipeline_config, dag=dag)
        query_expansion_fanout = _get_retrieval_backend_fanout(
            pipeline_config, dag=dag,
        )
        if has_passage_reranker:
            passage_reranker_df = processed_dfs.get("passage_reranker")
            if passage_reranker_df is not None and len(passage_reranker_df) > 0:
                if "seq_len_inference_prefill" in passage_reranker_df.columns:
                    passage_reranker_seq_len = int(passage_reranker_df["seq_len_inference_prefill"].iloc[0])
        # Query expansion parameters
        query_expansion_seq_len = None
        query_expansion_seq_len_with_template = None
        query_expansion_dec_steps = None
        if has_query_expansion:
            query_expansion_seq_len = ts.get_query_expansion_input_tokens()
            # Prefill with template: use seq_len from the sweep_df if available
            query_expansion_prefill_df = processed_dfs.get("query_expansion_prefill")
            if query_expansion_prefill_df is not None and len(query_expansion_prefill_df) > 0:
                if "seq_len_inference_prefill" in query_expansion_prefill_df.columns:
                    query_expansion_seq_len = int(query_expansion_prefill_df["seq_len_inference_prefill"].iloc[0])
            query_expansion_seq_len_with_template = query_expansion_seq_len
            query_expansion_decode_df = processed_dfs.get("query_expansion_decode")
            if query_expansion_decode_df is not None and len(query_expansion_decode_df) > 0:
                if "dec_steps" in query_expansion_decode_df.columns:
                    query_expansion_dec_steps = int(query_expansion_decode_df["dec_steps"].iloc[0])
                if "seq_len_inference_prefill" in query_expansion_decode_df.columns:
                    query_expansion_seq_len_with_template = int(
                        query_expansion_decode_df["seq_len_inference_prefill"].iloc[0]
                    )
            if query_expansion_dec_steps is None:
                # Fallback: use rewriter output tokens as dec_steps
                query_expansion_dec_steps = ts.get_query_expansion_output_tokens("query_decompose")
        retrieval_policy = RetrievalPolicy(
            run_retrieval="semantic_retrieval_vectorsearch" in stages,
            retrieval_pattern="once",
            run_query_expansion=has_query_expansion,
            query_expansion_seq_len_inference_prefill=query_expansion_seq_len,
            query_expansion_seq_len_prefill_with_template=query_expansion_seq_len_with_template,
            query_expansion_dec_steps=query_expansion_dec_steps,
            # ``query_expansion_fanout`` propagates into RAGO's retrieval +
            # encode stage Pareto math (each user → N FAISS calls + N
            # encodes). Passed even when has_query_expansion=False — in
            # that case the helper returns 1 and RAGO's _apply_fanout is
            # a no-op.
            query_expansion_fanout=query_expansion_fanout,
            run_passage_reranker=has_passage_reranker,
            passage_reranker_seq_len_inference_prefill=passage_reranker_seq_len if has_passage_reranker else None,
            passage_reranker_topk=passage_reranker_topk if has_passage_reranker else None,
        )
        # Query-embedding ("encode") stage: ON when its sweep df exists
        # (semantic_retrieval present + embedding model modeled). nvec=1
        # degrades RAGO's encode-database math to per-(sub-)query embedding;
        # the query-expansion fanout multiplies it exactly like retrieval.
        # seq_len MUST equal the sweep df's value — RAGO filters
        # sweep_df["encode"] by exact equality on seq_len_inference_prefill.
        _encode_df = processed_dfs.get("encode")
        has_encode = (
            "semantic_retrieval_encode" in stages and _encode_df is not None and len(_encode_df) > 0
        )
        encode_seq_len = None
        if has_encode:
            encode_seq_len = int(ts.avg_query_tokens)
            if "seq_len_inference_prefill" in _encode_df.columns:
                encode_seq_len = int(_encode_df["seq_len_inference_prefill"].iloc[0])
        encode_db_policy = EncodeDBPolicy(
            run_encode_db=has_encode,
            encode_seq_len_inference_prefill=encode_seq_len,
            encode_db_nvec=1 if has_encode else None,
        )

        # Iterative-RAG branch: when the resolved pipeline_runtime says we're
        # in an iterative mode (flare / ircot), route to RAGO's
        # IterativeRAGSweep instead of the default RAGSweep. RAGO does not
        # distinguish between the iterative quality-eval methods — they all
        # share the same cost profile (every retrieval_interval tokens, do
        # another retrieve+prefill mid-decode).
        runtime = pipeline_config.get("pipeline_runtime") or {}
        mode = runtime.get("mode", "sequential")
        from rag_stack.search_space import ITERATIVE_PIPELINE_MODES
        if mode in ITERATIVE_PIPELINE_MODES:
            # IterativeRAGSweep models exactly retrieval/prefill/decode; the
            # encode (query-embedding) stage is instead folded into the trace
            # sim's per-call retrieve cost (cost_model/engines.py), so
            # drop it from the RPD guard here.
            _rpd_stages = [s for s in stages if s != "semantic_retrieval_encode"]
            assert sorted(_rpd_stages) == sorted(
                ["semantic_retrieval_vectorsearch", "generator_prefill", "generator_decode"]
            ), f"iterative pipeline requires bare RPD stages, got {stages}"
            num_retrievals = int(runtime.get("num_retrievals", 4))
            if num_retrievals < 1:
                raise ValueError(
                    f"pipeline.num_retrievals must be >= 1, got {num_retrievals}"
                )
            retrieval_interval = max(1, dec_steps // num_retrievals)
            print(
                f"Iterative cost path: mode={mode}, num_retrievals={num_retrievals}, "
                f"retrieval_interval={retrieval_interval}"
            )
            combined_df = _run_iterative_sweep(
                processed_dfs=processed_dfs,
                placement_policies=placement_policies,
                seq_len_inference_prefill=seq_len_inference_prefill,
                dec_steps=dec_steps,
                gpus_per_server=gpus_per_server,
                max_batch_size_request=max_batch_size_request,
                no_microbatching=no_microbatching,
                scheduling_policy=scheduling_policy,
                min_num_gpus=min_num_gpus,
                max_num_gpus=max_num_gpus,
                retrieval_interval=retrieval_interval,
            )
            # ── RAGO boundary (outbound, iterative) ─────────────────────
            combined_df = name_map.qualify_result_df(combined_df)
            combined_df = _restore_fusion_stage(combined_df, stages)
            combined_df["pipeline_mode"] = mode
            combined_df["num_retrievals"] = num_retrievals
            combined_df = _normalize_retrieval_resource_groups(combined_df)
            # Inter-group communication (iterative: the PD KV handoff repeats
            # once per retrieval round — num_retrievals scales that boundary).
            from rag_stack.cost_model.communication import apply_communication_costs
            combined_df = apply_communication_costs(
                combined_df,
                hardware=self.hardware,
                pipeline_config=pipeline_config,
                seq_len_prefill=int(seq_len_inference_prefill),
                qe_seq_len=query_expansion_seq_len_with_template,
                reranker_input_fanout=passage_reranker_topk,
                query_expansion_fanout=query_expansion_fanout,
            )
            return AssemblyResult(
                performance_score=0.0,
                best_hw_params={},
                rago_result_df=combined_df,
            )

        # Sequential cost path. Trace-driven agentic/iterative workflows enter
        # via evaluate_dynamic(..., trace=...), which reuses this RAGO candidate
        # generation before the DynamicEngine trace walk.
        print("Initializing RAGSweep...")
        from rag_stack.cost_model.interconnect_topology import rago_device_topology
        _topology_source = (pipeline_config.get("system") or {})
        if not _topology_source.get("interconnect"):
            _topology_source = self.hardware
        available_devices, device_links = rago_device_topology(
            _topology_source,
            required_count=int(max_num_gpus),
        )
        # Init RAGSweep
        rago_sweep = RAGSweep(
            retrieval_policy=retrieval_policy,
            encode_db_policy=encode_db_policy,
            stages=[to_rago_stage(s) for s in rago_stages],
            sweep_df=processed_dfs,
            seq_len_inference_prefill=seq_len_inference_prefill,
            dec_steps=dec_steps,
            num_chips_per_server=gpus_per_server,
            # Agentic prefill fanout (M generate rounds/query) — FLOAT, from the
            # trace aggregate; 1.0 for sequential RAG so _apply_fanout is a
            # no-op (StaticAssembly's normal path is byte-unchanged).
            prefill_fanout=_get_prefill_fanout(pipeline_config, dag),
            available_devices=available_devices,
            device_links=device_links,
        )

        if fixed_placement is not None:
            # SINGLE fixed placement (no sweep, no placement enumeration): price
            # exactly the given stage→group partition + chips + batch sizes.
            # This is the evaluate_fixed → _evaluate_fixed_placement path.
            print("Running RAGO single fixed placement (no sweep):",
                  fixed_placement.collocation_strategy)
            from rag_stack.cost_model.rago_adapter import to_rago_physical_mapping
            combined_df = rago_sweep.assemble_cost(
                physical_mapping=to_rago_physical_mapping(
                    fixed_placement,
                    system_config=(pipeline_config.get("system") or {}),
                ),
                scheduling_policy=scheduling_policy,
                max_batch_size_request=(
                    fixed_batch_size_request
                    if fixed_batch_size_request is not None
                    else max_batch_size_request
                ),
                no_microbatching=no_microbatching,
                fixed_batch_size_request=fixed_batch_size_request,
                fixed_batch_size_decode=fixed_batch_size_decode,
                possible_batch_sizes_decode=possible_batch_sizes_decode,
            )
            if combined_df is None or len(combined_df) == 0:
                raise RuntimeError(
                    f"Fixed-placement assemble_cost produced no rows. "
                    f"collocation_strategy={fixed_placement.collocation_strategy}, "
                    f"num_chips={fixed_placement.num_chips}, stages={list(stages)}."
                )
        else:
            # Sweep across all placement policies and combine results
            all_result_dfs = []
            for placement_policy in placement_policies:
                print("Running RAGO sweep with placement policy:", placement_policy)
                result_df = _run_sweep_parallel(
                    rago_sweep,
                    placement_policy=placement_policy,
                    min_num_chips=min_num_gpus,
                    max_num_chips=max_num_gpus,
                    max_num_chips_per_stage=max_num_gpus_per_stage,
                    scheduling_policy=scheduling_policy,
                    max_batch_size_request=max_batch_size_request,
                    no_microbatching=no_microbatching,
                    possible_batch_sizes_decode=possible_batch_sizes_decode,
                    allow_decode_collocation=allow_decode_collocation,
                )
                if result_df is not None and len(result_df) > 0:
                    all_result_dfs.append(result_df)

            print("Completed RAGO sweeps. Number of result DataFrames:", len(all_result_dfs))

            if not all_result_dfs:
                raise RuntimeError(
                    f"RAGO sweep produced no valid placement strategies. "
                    f"Stages: {list(stages)}, policies: {placement_policies}, "
                    f"GPUs: {min_num_gpus}-{max_num_gpus}. "
                    f"The model may be too large for the configured GPU count."
                )

            combined_df = pd.concat(all_result_dfs, ignore_index=True)

        if len(combined_df) == 0:
            raise RuntimeError(
                "RAGO sweep produced results but the combined DataFrame is empty."
            )

        # ── RAGO boundary (outbound) ────────────────────────────────────────
        # Translate every RAGO-id-keyed column/value back to PerformanceStage
        # names before ANY rag_stack code consumes the result.
        combined_df = name_map.qualify_result_df(combined_df)
        combined_df = _restore_fusion_stage(combined_df, stages)
        # RAGO already owns the independent generator-decode sweep. Expose it
        # as the single deployed decode axis at this common boundary so static
        # and dynamic pricing serialize the same machine that RAGO scored,
        # without DynamicEngine creating a second decode grid.
        combined_df = _materialize_rago_decode_axis(
            combined_df,
            pipeline_config,
        )

        # Tag mode for downstream (frontend) so iterative vs sequential rows
        # are distinguishable in rago_sweep.csv even when intermixed across
        # evaluations.
        combined_df["pipeline_mode"] = "sequential"
        # Per-request prefill→decode handoff count (one per generate round).
        # Sequential RAG → 1; agentic traces carry M in the aggregate so the
        # PD link's overhead/KV payload scale ×M (DynamicAssembly path).
        combined_df["num_retrievals"] = _get_num_retrievals(dag)
        combined_df = _normalize_retrieval_resource_groups(combined_df)

        # RAGO owns the physical mapping and emitted exact stage device ranks.
        # Qualify that mapping into the stable resolved-layout row contract
        # before GenZ/communication/engines consume the candidate.
        from rag_stack.cost_model.candidate_layout import (
            materialize_candidate_system_layouts,
        )
        combined_df = materialize_candidate_system_layouts(
            combined_df,
            pipeline_config=pipeline_config,
            sweep_dfs=sweep_dfs,
            prefer_system_layout=fixed_placement is not None,
        )
        from rag_stack.cost_model.candidate_reprice import (
            reprice_candidate_genz_services,
        )
        combined_df = reprice_candidate_genz_services(
            combined_df,
            pipeline_config=pipeline_config,
            hardware=self.hardware,
            stage_cost_models=stage_cost_models,
        )
        if combined_df is None or len(combined_df) == 0:
            raise RuntimeError(
                "RAGO produced placements, but none has a GenZ-feasible "
                "model-parallel deployment. Check the model's attention/KV "
                "heads and the requested TP/PP or P/D topology."
            )

        # Inter-group communication costs (RAGO models compute only): price
        # every adjacent-stage boundary that crosses a chip group — dominated
        # by the 1P1D prefill→decode KV transfer — into latency_s/qps.
        from rag_stack.cost_model.communication import apply_communication_costs
        combined_df = apply_communication_costs(
            combined_df,
            hardware=self.hardware,
            pipeline_config=pipeline_config,
            seq_len_prefill=int(seq_len_inference_prefill),
            qe_seq_len=query_expansion_seq_len_with_template,
            reranker_input_fanout=passage_reranker_topk,
            query_expansion_fanout=query_expansion_fanout,
        )
        if (
            fixed_placement is not None
            and fixed_batch_size_request is not None
            and fixed_batch_size_decode is not None
        ):
            combined_df = _filter_fixed_result_df(
                combined_df,
                batch_size_request=fixed_batch_size_request,
                batch_size_decode=fixed_batch_size_decode,
            )
        # NOTE: no PD-split serving penalty — the PD pair runs CUDA graphs
        # (eager removed 2026-06-10; A/B: TPOT 5.8ms == unified). The KV-hop
        # cost of the split is already priced by apply_communication_costs.

        return AssemblyResult(
            performance_score=0.0,
            best_hw_params={},
            rago_result_df=combined_df,
        )


# Back-compat functional entry — the trace-driven public wrapper (was
# dynamic_assembly.compute_trace_performance). Thin shim over RAGCMAssembly.
def compute_trace_performance(
    pipeline_config: dict,
    hardware: SystemConfig,
    traces,
) -> AssemblyResult:
    """Cost a measured ``trace_v1`` payload through the unified assembly."""
    schema = build_workflow_schema(pipeline_config)
    return RAGCMAssembly(hardware).evaluate_dynamic(
        pipeline_config,
        workflow_schema=schema,
        trace=traces,
    )
