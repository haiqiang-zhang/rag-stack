"""Final exact-device GenZ repricing for RAGO candidates."""
from __future__ import annotations

import copy
import itertools
import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from rag_stack.cost_model.candidate_layout import (
    CANDIDATE_LAYOUT_COLUMN,
    candidate_system_layout_from_row,
    update_candidate_parallelism,
)
from rag_stack.types import CMDeviceCapacityError, CMOutOfMemoryError


logger = logging.getLogger("RAG-Stack")


class CandidateParallelismInfeasible(ValueError):
    """One concrete candidate has no deployable GenZ parallelism plan.

    Sweep callers may discard this candidate while fixed-point callers must
    surface it: silently replacing a pinned TP/PP would price a different
    deployment than the one requested by the user.
    """

    def __init__(self, message: str, *, code: str = "parallelism_infeasible"):
        super().__init__(message)
        self.code = str(code)


class CandidateMemoryInfeasible(CMDeviceCapacityError):
    """All structurally valid exact-device plans for one row exceed memory."""


def _parallelism_model(model: Any, pipeline_config: dict) -> Any | None:
    """Return the GenZ model identity owned by a stage pricer, when present."""
    resolver = getattr(model, "parallelism_model", None)
    if not callable(resolver):
        # Compatibility for third-party GenZ adapters that have not yet added
        # the public hook.  Non-GenZ test/extension pricers have neither hook
        # and retain their historical behavior.
        resolver = getattr(model, "_get_model", None)
    return resolver(pipeline_config) if callable(resolver) else None


def _parallelism_tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        tp = value.get("tensor_parallel", value.get("tp"))
        pp = value.get("pipeline_parallel", value.get("pp"))
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        tp, pp = value
    else:
        tp = getattr(value, "tensor_parallel", None)
        pp = getattr(value, "pipeline_parallel", None)
    if tp is None or pp is None:
        raise TypeError(f"GenZ returned an invalid parallelism record: {value!r}")
    return int(tp), int(pp)


def _valid_parallelism_choices(
    model: Any,
    pipeline_config: dict,
    *,
    num_devices: int,
    batch_size: int,
    fixed: tuple[int, int] | None = None,
) -> list[tuple[int, int]]:
    """Ask GenZ for model-aware TP/PP choices on one concrete role."""
    if num_devices < 1:
        raise CandidateParallelismInfeasible(
            f"exact-device GenZ pricing requires at least one device, got {num_devices}",
            code="parallelism_no_devices",
        )

    genz_model = _parallelism_model(model, pipeline_config)
    if genz_model is None:
        # Exact-device stand-ins and legacy extensions do not claim GenZ model
        # semantics.  Production GenZ adapters always expose the hook above.
        if fixed is not None:
            choices = [fixed]
        else:
            from rag_stack.cost_model.utils import _get_parallelism
            choices = _get_parallelism(num_devices)
    else:
        from GenZ.parallelism import (
            check_model_parallelism,
            valid_parallelism_configs,
        )

        if fixed is not None:
            verdict = check_model_parallelism(
                genz_model,
                tensor_parallel=int(fixed[0]),
                pipeline_parallel=int(fixed[1]),
                batch_size=int(batch_size),
            )
            if not verdict.feasible:
                raise CandidateParallelismInfeasible(
                    verdict.reason,
                    code=verdict.code,
                )
            choices = [fixed]
        else:
            choices = [
                _parallelism_tuple(choice)
                for choice in valid_parallelism_configs(
                    genz_model,
                    int(num_devices),
                    batch_size=int(batch_size),
                )
            ]

    choices = sorted({(int(tp), int(pp)) for tp, pp in choices})
    choices = [
        (tp, pp) for tp, pp in choices
        if tp > 0 and pp > 0 and tp * pp == int(num_devices)
    ]
    if not choices:
        name = getattr(genz_model, "model", genz_model)
        raise CandidateParallelismInfeasible(
            f"model={name!r} has no valid TP/PP plan for "
            f"devices={num_devices}, batch_size={batch_size}",
            code="parallelism_no_valid_config",
        )
    return choices


def _check_disaggregated_pair(
    model: Any,
    pipeline_config: dict,
    *,
    prefill: tuple[int, int],
    decode: tuple[int, int],
) -> tuple[bool, str, str]:
    """Delegate P/D KV-transfer feasibility to GenZ model structure."""
    genz_model = _parallelism_model(model, pipeline_config)
    if genz_model is None:
        return True, "", ""
    from GenZ.parallelism import (
        NIXL_KV_TRANSFER_CAPABILITIES,
        check_disaggregated_parallelism,
    )

    verdict = check_disaggregated_parallelism(
        genz_model,
        prefill_tp=int(prefill[0]),
        prefill_pp=int(prefill[1]),
        decode_tp=int(decode[0]),
        decode_pp=int(decode[1]),
        capabilities=NIXL_KV_TRANSFER_CAPABILITIES,
    )
    return bool(verdict.feasible), str(verdict.code), str(verdict.reason)


def _stage_price(
    model: Any,
    pipeline_config: dict,
    hardware: Any,
    *,
    stage: str,
    batch: int,
    devices: list[str],
    parallelism: tuple[int, int] | None,
    memo: dict,
) -> dict:
    key = (
        id(model), stage, int(batch), tuple(devices),
        tuple(parallelism) if parallelism is not None else None,
    )
    if key not in memo:
        memo[key] = model.estimate_stage_for_devices(
            pipeline_config,
            hardware,
            stage=stage,
            batch_size=int(batch),
            devices=list(devices),
            parallelism=parallelism,
        )
    return copy.deepcopy(memo[key])


def _fixed_parallelism(layout: dict, stage: str) -> tuple[int, int] | None:
    info = (layout.get("performance_stages") or {}).get(stage) or {}
    if info.get("tp") is None or info.get("pp") is None:
        return None
    return int(info["tp"]), int(info["pp"])


def _two_role_llm_engine(stage: str) -> str | None:
    """Return the engine only for a concrete LLM prefill/decode stage.

    User-stage parents such as ``passage_reranker`` are heterogeneous: they
    contain both encoder-only forward components and LLM rerankers.  Parent
    membership therefore cannot make the bare forward PerformanceStage a
    TP/PP engine.  The taxonomy's concrete LLM set is the authoritative shape
    boundary shared with the analytical sweep.
    """
    from rag_stack.taxonomy import TAXONOMY

    if stage not in TAXONOMY.llm_performance_stages:
        return None
    try:
        engine = TAXONOMY.user_stage_of(stage).value
        llm_roles = {
            performance_stage.kind
            for performance_stage in TAXONOMY.performance_stages_of(engine)
            if performance_stage.name in TAXONOMY.llm_performance_stages
        }
    except (KeyError, ValueError):
        return None
    if not {"prefill", "decode"}.issubset(llm_roles):
        return None
    return engine


def require_candidate_genz_parallelism(
    df: pd.DataFrame,
    *,
    pipeline_config: dict,
    stage_cost_models: Mapping[str, Any] | None,
) -> None:
    """Validate pinned candidate layouts without performing latency pricing.

    The trace-driven fixed path intentionally bypasses RAGO/analytical
    repricing and goes straight to DES.  This lightweight gate preserves that
    fast path while ensuring its resolved TP/PP and P/D pair obey the same GenZ
    model constraints as swept candidates.
    """
    if df is None or not len(df) or not stage_cost_models:
        return

    for _, row in df.iterrows():
        layout = candidate_system_layout_from_row(row)
        perf = layout.get("performance_stages") or {}
        for engine, entry in (layout.get("engines") or {}).items():
            stages = []
            for stage, info in perf.items():
                if info.get("engine") != engine or stage not in stage_cost_models:
                    continue
                if info.get("role") not in ("prefill", "decode"):
                    continue
                if _two_role_llm_engine(stage) == str(engine):
                    stages.append(stage)
            if len(stages) < 2:
                continue
            stages.sort(
                key=lambda stage: 0
                if perf[stage].get("role") == "prefill" else 1
            )
            prefill_stage, decode_stage = stages[0], stages[1]
            prefill_choice = _fixed_parallelism(layout, prefill_stage)
            decode_choice = _fixed_parallelism(layout, decode_stage)
            if prefill_choice is None or decode_choice is None:
                raise CandidateParallelismInfeasible(
                    f"Fixed engine {engine!r} lacks explicit role TP/PP",
                    code="parallelism_missing_fixed",
                )
            prefill_batch = max(1, int(row[f"batch_size_{prefill_stage}"]))
            decode_batch = max(1, int(row[f"batch_size_{decode_stage}"]))
            _valid_parallelism_choices(
                stage_cost_models[prefill_stage],
                pipeline_config,
                num_devices=len(perf[prefill_stage].get("devices") or []),
                batch_size=prefill_batch,
                fixed=prefill_choice,
            )
            _valid_parallelism_choices(
                stage_cost_models[decode_stage],
                pipeline_config,
                num_devices=len(perf[decode_stage].get("devices") or []),
                batch_size=decode_batch,
                fixed=decode_choice,
            )
            if entry.get("pd_serving") == "disagg_pd":
                feasible, code, reason = _check_disaggregated_pair(
                    stage_cost_models[prefill_stage],
                    pipeline_config,
                    prefill=prefill_choice,
                    decode=decode_choice,
                )
                if not feasible:
                    raise CandidateParallelismInfeasible(reason, code=code)
            elif prefill_choice != decode_choice:
                raise CandidateParallelismInfeasible(
                    f"Unified engine {engine!r} has role-mismatched TP/PP: "
                    f"prefill={prefill_choice}, decode={decode_choice}",
                    code="parallelism_unified_role_mismatch",
                )


def _reprice_engine(
    row: pd.Series,
    layout: dict,
    engine: str,
    stages: list[str],
    *,
    stage_cost_models: Mapping[str, Any],
    pipeline_config: dict,
    hardware: Any,
    memo: dict,
) -> dict[str, dict]:
    perf = layout["performance_stages"]
    entry = (layout.get("engines") or {}).get(engine) or {}
    fixed = (
        layout.get("source") == "resolved_system_layout"
        or layout.get("parallelism_source") == "genz_exhaustive"
    )
    collocated = entry.get("pd_serving") != "disagg_pd"

    # A unified engine has one executable TP/PP plan for both roles.  Choose it
    # jointly against the exact ordered devices.
    if collocated and len(stages) > 1:
        devices = list(perf[stages[0]].get("devices") or [])
        if any(list(perf[stage].get("devices") or []) != devices for stage in stages):
            raise ValueError(
                f"Unified engine {engine!r} has inconsistent role devices: "
                f"{[(stage, perf[stage].get('devices')) for stage in stages]}"
            )
        fixed_choice = _fixed_parallelism(layout, stages[0]) if fixed else None
        if fixed and fixed_choice is None:
            raise CandidateParallelismInfeasible(
                f"Fixed unified engine {engine!r} lacks TP/PP",
                code="parallelism_missing_fixed",
            )
        stage_choices = [
            set(_valid_parallelism_choices(
                stage_cost_models[stage],
                pipeline_config,
                num_devices=len(devices),
                batch_size=max(1, int(row[f"batch_size_{stage}"])),
                fixed=fixed_choice,
            ))
            for stage in stages
        ]
        choices = sorted(set.intersection(*stage_choices))
        if not choices:
            raise CandidateParallelismInfeasible(
                f"Unified engine {engine!r} has no TP/PP plan valid for both roles",
                code="parallelism_no_common_config",
            )

        candidates: list[tuple[float, dict[str, dict]]] = []
        capacity_failures: list[CMDeviceCapacityError] = []
        for choice in choices:
            priced: dict[str, dict] = {}
            try:
                score = 0.0
                for stage in stages:
                    batch = max(1, int(row[f"batch_size_{stage}"]))
                    result = _stage_price(
                        stage_cost_models[stage], pipeline_config, hardware,
                        stage=stage, batch=batch, devices=devices,
                        parallelism=choice, memo=memo,
                    )
                    priced[stage] = result
                    # RAGO collocation capacity is harmonic across stage qps:
                    # sum(latency / batch) is exactly the group demand/query.
                    score += float(result["latency_s"]) / float(batch)
                candidates.append((score, priced))
            except CMDeviceCapacityError as exc:
                capacity_failures.append(exc)
        if not candidates:
            failure = CandidateMemoryInfeasible(
                f"Every structurally valid exact-device plan for unified "
                f"{engine!r} exceeds memory: "
                + "; ".join(str(exc) for exc in capacity_failures)
            )
            if capacity_failures:
                raise failure from capacity_failures[0]
            raise RuntimeError(
                f"Exact-device repricing produced no plan for unified {engine!r}"
            )
        return min(candidates, key=lambda item: item[0])[1]

    # A disaggregated engine may use different TP/PP on prefill and decode,
    # but those choices are coupled by the KV-transfer layout.  Enumerating
    # each role independently can select a pair which GenZ/vLLM cannot deploy
    # (notably a replicated producer with P-TP > D-TP).  Jointly enumerate the
    # concrete role widths, reject structurally invalid pairs *before* costly
    # pricing, then retain the legal pair with maximum assembled throughput.
    if not collocated and len(stages) > 1:
        prefill_stage = next(
            (stage for stage in stages if perf[stage].get("role") == "prefill"),
            None,
        )
        decode_stage = next(
            (stage for stage in stages if perf[stage].get("role") == "decode"),
            None,
        )
        if prefill_stage is None or decode_stage is None:
            raise CandidateParallelismInfeasible(
                f"Disaggregated engine {engine!r} lacks a prefill/decode role",
                code="parallelism_missing_pd_role",
            )

        prefill_devices = list(perf[prefill_stage].get("devices") or [])
        decode_devices = list(perf[decode_stage].get("devices") or [])
        prefill_fixed = _fixed_parallelism(layout, prefill_stage) if fixed else None
        decode_fixed = _fixed_parallelism(layout, decode_stage) if fixed else None
        if fixed and (prefill_fixed is None or decode_fixed is None):
            raise CandidateParallelismInfeasible(
                f"Fixed disaggregated engine {engine!r} lacks role-local TP/PP",
                code="parallelism_missing_fixed",
            )

        prefill_batch = max(1, int(row[f"batch_size_{prefill_stage}"]))
        decode_batch = max(1, int(row[f"batch_size_{decode_stage}"]))
        prefill_choices = _valid_parallelism_choices(
            stage_cost_models[prefill_stage],
            pipeline_config,
            num_devices=len(prefill_devices),
            batch_size=prefill_batch,
            fixed=prefill_fixed,
        )
        decode_choices = _valid_parallelism_choices(
            stage_cost_models[decode_stage],
            pipeline_config,
            num_devices=len(decode_devices),
            batch_size=decode_batch,
            fixed=decode_fixed,
        )

        groups = [
            list(group.get("performance_stages") or [])
            for group in layout.get("resource_groups", [])
        ]
        candidates: list[
            tuple[float, float, tuple[int, int], tuple[int, int], dict[str, dict]]
        ] = []
        invalid_pairs: list[str] = []
        capacity_failures: list[CMDeviceCapacityError] = []
        for prefill_choice in prefill_choices:
            for decode_choice in decode_choices:
                feasible, code, reason = _check_disaggregated_pair(
                    stage_cost_models[prefill_stage],
                    pipeline_config,
                    prefill=prefill_choice,
                    decode=decode_choice,
                )
                if not feasible:
                    invalid_pairs.append(
                        f"P={prefill_choice},D={decode_choice}: {code}: {reason}"
                    )
                    continue
                try:
                    priced = {
                        prefill_stage: _stage_price(
                            stage_cost_models[prefill_stage],
                            pipeline_config,
                            hardware,
                            stage=prefill_stage,
                            batch=prefill_batch,
                            devices=prefill_devices,
                            parallelism=prefill_choice,
                            memo=memo,
                        ),
                        decode_stage: _stage_price(
                            stage_cost_models[decode_stage],
                            pipeline_config,
                            hardware,
                            stage=decode_stage,
                            batch=decode_batch,
                            devices=decode_devices,
                            parallelism=decode_choice,
                            memo=memo,
                        ),
                    }
                except CMDeviceCapacityError as exc:
                    capacity_failures.append(exc)
                    continue

                trial = row.copy()
                trial[f"qps_{prefill_stage}"] = float(
                    priced[prefill_stage]["throughput_qps"]
                )
                trial[f"qps_{decode_stage}"] = float(
                    priced[decode_stage]["throughput_qps"]
                )
                qps, _ = _recompute_rago_capacity(trial, groups)
                demand = (
                    float(priced[prefill_stage]["latency_s"]) / prefill_batch
                    + float(priced[decode_stage]["latency_s"]) / decode_batch
                )
                # min() below: negate throughput, then prefer lower total
                # per-query demand and finally deterministic TP/PP order.
                candidates.append((
                    -float(qps),
                    float(demand),
                    prefill_choice,
                    decode_choice,
                    priced,
                ))

        if not candidates:
            if capacity_failures:
                failure = CandidateMemoryInfeasible(
                    f"Every structurally valid exact-device pair for "
                    f"disaggregated {engine!r} exceeds memory: "
                    + "; ".join(str(exc) for exc in capacity_failures)
                )
                raise failure from capacity_failures[0]
            detail = "; ".join(invalid_pairs) or "no role-local choices"
            raise CandidateParallelismInfeasible(
                f"Disaggregated engine {engine!r} has no valid P/D "
                f"parallelism pair: {detail}",
                code="parallelism_no_valid_pd_pair",
            )
        return min(candidates, key=lambda item: item[:4])[4]

    # A lone role still needs model-aware validation.  Enumerate explicitly
    # rather than passing parallelism=None to an adapter-specific search, so
    # every production path consumes the same GenZ feasibility contract.
    out: dict[str, dict] = {}
    for stage in stages:
        devices = list(perf[stage].get("devices") or [])
        batch = max(1, int(row[f"batch_size_{stage}"]))
        fixed_choice = _fixed_parallelism(layout, stage) if fixed else None
        if fixed and fixed_choice is None:
            raise CandidateParallelismInfeasible(
                f"Fixed stage {stage!r} lacks TP/PP",
                code="parallelism_missing_fixed",
            )
        choices = _valid_parallelism_choices(
            stage_cost_models[stage],
            pipeline_config,
            num_devices=len(devices),
            batch_size=batch,
            fixed=fixed_choice,
        )
        candidates: list[dict] = []
        capacity_failures: list[CMDeviceCapacityError] = []
        for choice in choices:
            try:
                candidates.append(_stage_price(
                    stage_cost_models[stage], pipeline_config, hardware,
                    stage=stage, batch=batch, devices=devices,
                    parallelism=choice, memo=memo,
                ))
            except CMDeviceCapacityError as exc:
                capacity_failures.append(exc)
        if not candidates:
            failure = CandidateMemoryInfeasible(
                f"Every structurally valid exact-device plan for stage "
                f"{stage!r} exceeds memory: "
                + "; ".join(str(exc) for exc in capacity_failures)
            )
            if capacity_failures:
                raise failure from capacity_failures[0]
            raise RuntimeError(
                f"Exact-device repricing produced no plan for stage {stage!r}"
            )
        out[stage] = min(candidates, key=lambda result: float(result["latency_s"]))
    return out


def _recompute_rago_capacity(row: pd.Series, groups: list[list[str]]) -> tuple[float, list[str]]:
    group_qps: list[tuple[float, list[str]]] = []
    for group in groups:
        stage_qps: list[float] = []
        for stage in group:
            col = f"qps_{stage}"
            if col not in row.index or pd.isna(row[col]):
                continue
            qps = float(row[col])
            if qps > 0 and np.isfinite(qps):
                stage_qps.append(qps)
        if stage_qps:
            group_qps.append((1.0 / sum(1.0 / qps for qps in stage_qps), group))
    if not group_qps:
        return float(row.get("qps", 0.0) or 0.0), []
    return min(group_qps, key=lambda item: item[0])


def _repricable_stages_by_engine(
    row: Mapping[str, Any],
    layout: Mapping[str, Any],
    repricable: Mapping[str, Any],
) -> dict[str, list[str]]:
    perf = layout.get("performance_stages") or {}
    by_engine: dict[str, list[str]] = {}
    for stage in repricable:
        if stage not in perf or f"batch_size_{stage}" not in row:
            continue
        role = perf[stage].get("role")
        if role not in ("prefill", "decode"):
            continue
        engine = _two_role_llm_engine(stage)
        if engine is not None:
            by_engine.setdefault(engine, []).append(stage)
    for stages in by_engine.values():
        stages.sort(
            key=lambda stage: 0
            if perf[stage].get("role") == "prefill" else 1
        )
    return by_engine


def _engine_parallelism_variants(
    row: Mapping[str, Any],
    layout: Mapping[str, Any],
    engine: str,
    stages: list[str],
    *,
    stage_cost_models: Mapping[str, Any],
    pipeline_config: dict,
) -> list[dict[str, tuple[int, int]]]:
    """Enumerate deployable role TP/PP plans without analytically picking one."""
    perf = layout.get("performance_stages") or {}
    entry = (layout.get("engines") or {}).get(engine) or {}
    collocated = entry.get("pd_serving") != "disagg_pd"

    if collocated and len(stages) > 1:
        devices = list(perf[stages[0]].get("devices") or [])
        if any(list(perf[stage].get("devices") or []) != devices for stage in stages):
            raise CandidateParallelismInfeasible(
                f"Unified engine {engine!r} has inconsistent role devices",
                code="parallelism_unified_device_mismatch",
            )
        choices_by_stage = [
            set(_valid_parallelism_choices(
                stage_cost_models[stage],
                pipeline_config,
                num_devices=len(devices),
                batch_size=max(1, int(row[f"batch_size_{stage}"])),
            ))
            for stage in stages
        ]
        common = sorted(set.intersection(*choices_by_stage))
        if not common:
            raise CandidateParallelismInfeasible(
                f"Unified engine {engine!r} has no TP/PP plan valid for both roles",
                code="parallelism_no_common_config",
            )
        return [{stage: choice for stage in stages} for choice in common]

    if not collocated and len(stages) > 1:
        prefill_stage = next(
            (stage for stage in stages if perf[stage].get("role") == "prefill"),
            None,
        )
        decode_stage = next(
            (stage for stage in stages if perf[stage].get("role") == "decode"),
            None,
        )
        if prefill_stage is None or decode_stage is None:
            raise CandidateParallelismInfeasible(
                f"Disaggregated engine {engine!r} lacks a prefill/decode role",
                code="parallelism_missing_pd_role",
            )
        prefill_choices = _valid_parallelism_choices(
            stage_cost_models[prefill_stage],
            pipeline_config,
            num_devices=len(perf[prefill_stage].get("devices") or []),
            batch_size=max(1, int(row[f"batch_size_{prefill_stage}"])),
        )
        decode_choices = _valid_parallelism_choices(
            stage_cost_models[decode_stage],
            pipeline_config,
            num_devices=len(perf[decode_stage].get("devices") or []),
            batch_size=max(1, int(row[f"batch_size_{decode_stage}"])),
        )
        variants: list[dict[str, tuple[int, int]]] = []
        invalid: list[str] = []
        for prefill_choice, decode_choice in itertools.product(
            prefill_choices, decode_choices,
        ):
            feasible, code, reason = _check_disaggregated_pair(
                stage_cost_models[prefill_stage],
                pipeline_config,
                prefill=prefill_choice,
                decode=decode_choice,
            )
            if feasible:
                variants.append({
                    prefill_stage: prefill_choice,
                    decode_stage: decode_choice,
                })
            else:
                invalid.append(
                    f"P={prefill_choice},D={decode_choice}: {code}: {reason}"
                )
        if not variants:
            raise CandidateParallelismInfeasible(
                f"Disaggregated engine {engine!r} has no valid P/D "
                f"parallelism pair: {'; '.join(invalid) or 'no role-local choices'}",
                code="parallelism_no_valid_pd_pair",
            )
        return variants

    variants = []
    for stage in stages:
        devices = list(perf[stage].get("devices") or [])
        for choice in _valid_parallelism_choices(
            stage_cost_models[stage],
            pipeline_config,
            num_devices=len(devices),
            batch_size=max(1, int(row[f"batch_size_{stage}"])),
        ):
            variants.append({stage: choice})
    return variants


def _expand_sweep_parallelism_variants(
    df: pd.DataFrame,
    *,
    pipeline_config: dict,
    repricable: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Expand each RAGO placement into every executable exact-width TP/PP.

    RAGO owns stage placement and device allocation.  Parallelism is a second,
    concrete system-design axis: collapsing it to the fastest component price
    before DynamicEngine/DES lets an analytical phase choose the deployment it
    is supposed only to rank.  The measured LLM runtime currently has one
    engine per role, so every variant consumes all allocated role devices and
    has exactly one replica. Fixed layouts remain exactly fixed.
    """
    rows: list[pd.Series] = []
    failures: dict[str, int] = {}
    expanded_any = False
    for _, original in df.iterrows():
        layout = candidate_system_layout_from_row(original)
        if layout.get("source") == "resolved_system_layout":
            rows.append(original.copy())
            continue
        by_engine = _repricable_stages_by_engine(original, layout, repricable)
        if not by_engine:
            rows.append(original.copy())
            continue
        try:
            engine_variants = [
                _engine_parallelism_variants(
                    original,
                    layout,
                    engine,
                    stages,
                    stage_cost_models=repricable,
                    pipeline_config=pipeline_config,
                )
                for engine, stages in sorted(by_engine.items())
            ]
        except CandidateParallelismInfeasible as exc:
            failures[exc.code] = failures.get(exc.code, 0) + 1
            continue

        combinations = list(itertools.product(*engine_variants))
        expanded_any = expanded_any or len(combinations) > 1
        for combination in combinations:
            row = original.copy()
            variant_layout = copy.deepcopy(layout)
            variant_layout["parallelism_source"] = "genz_exhaustive"
            for role_choices in combination:
                for stage, (tp, pp) in role_choices.items():
                    update_candidate_parallelism(
                        variant_layout, stage, tp=int(tp), pp=int(pp)
                    )
            row[CANDIDATE_LAYOUT_COLUMN] = variant_layout
            rows.append(row)

    if not rows:
        return df.iloc[0:0].copy(), failures
    expanded = pd.DataFrame(rows, columns=df.columns)
    # Duplicate source indices are unsafe for the row-local .at updates below.
    # Normal RAGO sweeps already use a RangeIndex, so reset only when a source
    # row genuinely expanded; one-variant/fixed callers retain their index.
    if expanded_any:
        expanded = expanded.reset_index(drop=True)
    return expanded, failures


def reprice_candidate_genz_services(
    df: pd.DataFrame,
    *,
    pipeline_config: dict,
    hardware: Any,
    stage_cost_models: Mapping[str, Any] | None,
) -> pd.DataFrame:
    """Replace homogeneous stage prices with exact candidate-device GenZ prices.

    This runs after RAGO has materialized/qualified physical mappings and before
    communication or selection.  The correction updates per-stage qps, RAGO
    resource-group capacity, and latency/TTFT/TPOT by the exact stage deltas.
    """
    if df is None or not len(df) or not stage_cost_models:
        return df
    repricable = {
        stage: model for stage, model in stage_cost_models.items()
        if callable(getattr(model, "estimate_stage_for_devices", None))
    }
    if not repricable:
        return df

    input_rows = len(df)
    df, expansion_failures = _expand_sweep_parallelism_variants(
        df,
        pipeline_config=pipeline_config,
        repricable=repricable,
    )
    if expansion_failures:
        logger.info(
            "GenZ parallelism feasibility removed %d/%d RAGO placements "
            "before concrete TP/PP expansion: %s",
            sum(expansion_failures.values()),
            input_rows,
            ", ".join(
                f"{code}={count}"
                for code, count in sorted(expansion_failures.items())
            ),
        )
    if len(df) == 0:
        raise CandidateParallelismInfeasible(
            "No RAGO placement has a deployable concrete TP/PP variant",
            code="parallelism_no_surviving_candidate",
        )
    if len(df) != input_rows:
        logger.info(
            "Expanded %d RAGO placements into %d concrete TP/PP candidates",
            input_rows,
            len(df),
        )

    out = df.copy()
    memo: dict = {}
    layouts: list[dict] = []
    all_prices: list[dict[str, dict]] = []
    baseline_qps: list[float] = []
    deltas: list[float] = []
    infeasible_indices: list[Any] = []
    infeasible_reasons: dict[str, int] = {}
    memory_infeasible_indices: list[Any] = []
    memory_infeasible_errors: list[CandidateMemoryInfeasible] = []

    for _, original in out.iterrows():
        row = original.copy()
        layout = copy.deepcopy(candidate_system_layout_from_row(row))
        perf = layout.get("performance_stages") or {}
        by_engine: dict[str, list[str]] = {}
        for stage, model in repricable.items():
            if stage not in perf or f"batch_size_{stage}" not in row.index:
                continue
            role = perf[stage].get("role")
            if role not in ("prefill", "decode"):
                continue
            engine = _two_role_llm_engine(stage)
            if engine is None:
                continue
            by_engine.setdefault(engine, []).append(stage)

        priced: dict[str, dict] = {}
        try:
            for engine, stages in by_engine.items():
                stages.sort(
                    key=lambda stage: 0
                    if perf[stage]["role"] == "prefill" else 1
                )
                priced.update(_reprice_engine(
                    row, layout, engine, stages,
                    stage_cost_models=repricable,
                    pipeline_config=pipeline_config,
                    hardware=hardware,
                    memo=memo,
                ))
        except CandidateMemoryInfeasible as exc:
            # Capacity is local to this exact device mapping.  Do not abort a
            # sweep while another concrete candidate may fit; conversely, do
            # not turn it into a generic "infeasible" topology reason.
            infeasible_indices.append(original.name)
            memory_infeasible_indices.append(original.name)
            memory_infeasible_errors.append(exc)
            infeasible_reasons["exact_device_memory"] = (
                infeasible_reasons.get("exact_device_memory", 0) + 1
            )
            continue
        except CandidateParallelismInfeasible as exc:
            # A fixed-point call represents one user-pinned deployment.  It is
            # an error to silently discard or replace it.  Sweeps, in contrast,
            # own a candidate set and must remove invalid rows before DES and
            # selection can observe them.
            if layout.get("source") == "resolved_system_layout":
                raise
            infeasible_indices.append(original.name)
            infeasible_reasons[exc.code] = infeasible_reasons.get(exc.code, 0) + 1
            continue

        baseline_qps.append(float(row.get("qps", 0.0) or 0.0))
        total_delta = 0.0
        ttft_delta = 0.0
        generator_decode_delta = 0.0
        for stage, price in priced.items():
            batch = max(1, int(row[f"batch_size_{stage}"]))
            old_qps = float(row.get(f"qps_{stage}", 0.0) or 0.0)
            old_latency = float(batch) / old_qps if old_qps > 0 else 0.0
            new_latency = float(price["latency_s"])
            delta = new_latency - old_latency
            total_delta += delta
            row[f"qps_{stage}"] = float(price["throughput_qps"])
            update_candidate_parallelism(
                layout, stage, tp=int(price["tp"]), pp=int(price["pp"])
            )
            if stage == "generator_decode":
                generator_decode_delta += delta
            else:
                ttft_delta += delta

        groups = [
            list(group.get("performance_stages") or [])
            for group in layout.get("resource_groups", [])
        ]
        qps, bound = _recompute_rago_capacity(row, groups)
        row["qps"] = qps
        row["qps_bound"] = bound
        chips = max(1, int(row.get("num_chips_e2e", 1) or 1))
        row["qps_per_chip"] = qps / chips
        row["latency_s"] = max(0.0, float(row.get("latency_s", 0.0)) + total_delta)
        if "latency_s_ttft" in row.index:
            row["latency_s_ttft"] = max(
                0.0, float(row["latency_s_ttft"]) + ttft_delta
            )
        if "latency_s_tpot" in row.index and generator_decode_delta:
            dec_steps = 1
            sdf = (stage_cost_models and getattr(stage_cost_models.get("generator_decode"), "_get_params", None))
            if callable(sdf):
                try:
                    dec_steps = max(1, int(sdf(pipeline_config)["output_tokens"]))
                except Exception:
                    dec_steps = 1
            row["latency_s_tpot"] = max(
                0.0, float(row["latency_s_tpot"]) + generator_decode_delta / dec_steps
            )

        for column, value in row.items():
            out.at[original.name, column] = value
        layouts.append(layout)
        all_prices.append(priced)
        deltas.append(total_delta)

    if infeasible_indices:
        out = out.drop(index=infeasible_indices)
        logger.info(
            "GenZ parallelism feasibility removed %d/%d concrete sweep "
            "candidates before DES/selection: %s",
            len(infeasible_indices),
            len(df),
            ", ".join(
                f"{code}={count}"
                for code, count in sorted(infeasible_reasons.items())
            ),
        )

    if len(memory_infeasible_indices) == len(df):
        error = CMOutOfMemoryError(
            "CM exact-device OOM: every concrete system candidate exceeds "
            f"target device memory ({len(df)}/{len(df)} candidates)"
        )
        raise error from memory_infeasible_errors[0]

    if len(out) == 0 and memory_infeasible_indices:
        structural_count = len(infeasible_indices) - len(memory_infeasible_indices)
        raise CandidateParallelismInfeasible(
            "Exact-device repricing has no surviving concrete system "
            f"candidate: structural={structural_count}, "
            f"memory={len(memory_infeasible_indices)}, total={len(df)}",
            code="parallelism_no_surviving_candidate",
        )

    out[CANDIDATE_LAYOUT_COLUMN] = pd.Series(layouts, index=out.index, dtype=object)
    out["candidate_genz_repriced"] = [bool(prices) for prices in all_prices]
    out["candidate_genz_baseline_qps"] = baseline_qps
    out["candidate_genz_latency_delta_s"] = deltas
    stages = sorted({stage for prices in all_prices for stage in prices})
    for stage in stages:
        out[f"candidate_genz_latency_s_{stage}"] = [
            prices.get(stage, {}).get("latency_s") for prices in all_prices
        ]
        out[f"candidate_effective_icn_gbps_{stage}"] = [
            prices.get(stage, {}).get("effective_icn_gbps") for prices in all_prices
        ]
        out[f"candidate_devices_{stage}"] = pd.Series(
            [
                list((layout.get("performance_stages") or {}).get(stage, {}).get("devices") or [])
                for layout in layouts
            ],
            index=out.index,
            dtype=object,
        )
        out[f"candidate_tp_{stage}"] = [
            (layout.get("performance_stages") or {}).get(stage, {}).get("tp")
            for layout in layouts
        ]
        out[f"candidate_pp_{stage}"] = [
            (layout.get("performance_stages") or {}).get(stage, {}).get("pp")
            for layout in layouts
        ]
        out[f"candidate_replicas_{stage}"] = [
            (layout.get("performance_stages") or {}).get(stage, {}).get("replicas")
            for layout in layouts
        ]
        out[f"candidate_replica_devices_{stage}"] = pd.Series(
            [
                list(
                    (layout.get("performance_stages") or {})
                    .get(stage, {})
                    .get("replica_devices") or []
                )
                for layout in layouts
            ],
            index=out.index,
            dtype=object,
        )
    # The optimizer may only publish deployments that the measured replay can
    # actually launch.  Keep this executable-space contract at the CM output
    # boundary so a future parallelism enumerator cannot silently reintroduce
    # hypothetical LLM replicas or another runtime-unsupported layout.
    from rag_stack.runtime_parallelism import (
        validate_candidate_frame_runtime_parallelism,
    )
    validate_candidate_frame_runtime_parallelism(
        out,
        pipeline_config,
        source="RAG-CM concrete system sweep",
    )
    return out
