"""Zero-dependency runtime and RAG-CM parallelism declarations.

Measured-layout search consumes the runtime allowlist.  RAG-CM separately
consumes the composability allowlist: executable dispatch is necessary but is
not evidence that a whole-stage observation can be split across replicas.
Keeping both declarations in this leaf module avoids importing GenZ and other
analytical dependencies from configuration code while making their difference
explicit.
"""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


# These rerankers load one model replica per declared device and dispatch
# balanced input shards concurrently in the measured evaluator.
RERANKER_MEASURED_DP_COMPONENTS: frozenset[str] = frozenset({
    "colbert_reranker",
    "flag_embedding_reranker",
    "monot5",
    "sentence_transformer_reranker",
    "tart",
})

# No reranker currently has operation-scoped calibration evidence that
# separates global IPC/pre/post work, replica-local model work, and dispatcher
# work.  StageServingCurve therefore exposes only an indivisible P1 active
# service, even for components whose measured runtime has a DP dispatcher.
RERANKER_CM_COMPOSABLE_DP_COMPONENTS: frozenset[str] = frozenset()

# Kept as a compatibility export for callers that imported the old hard cap.
# ``None`` now means the measured encoder has no software width cap; the
# physical/search-space GPU budget remains authoritative.
SEMANTIC_RETRIEVAL_ENCODE_MAX_GPUS: None = None


def validate_measured_parallelism_width(
    *,
    stage: str,
    component: str | None,
    width: int,
    source: str,
) -> None:
    """Reject a stage width the measured runtime cannot execute.

    This leaf helper is shared by search/config replay, CM validation, and the
    measured evaluator's direct resolved-config entry.  ``width`` is a
    deployment fact; it is never a workload shape or StageServingCurve key.
    """

    if type(width) is bool or type(width) is not int or width < 0:
        raise ValueError("measured runtime parallelism width must be a non-negative integer")
    stage_name = str(stage)
    component_name = str(component) if component else None
    if (
        stage_name == "passage_reranker"
        and width > 1
        and component_name not in RERANKER_MEASURED_DP_COMPONENTS
    ):
        raise NotImplementedError(
            f"{source}: passage_reranker component {component_name!r} "
            f"declares {width} GPUs, but measured multi-GPU dispatch exists "
            "only for "
            f"{sorted(RERANKER_MEASURED_DP_COMPONENTS)!r}; data-parallel "
            "capacity is not implemented for this component"
        )


def _parse_sequence(value: Any, *, field: str) -> Sequence[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"{field} must be a sequence") from exc
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return value


def _positive_int(value: Any, *, field: str) -> int:
    if type(value) is bool:
        raise TypeError(f"{field} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be a positive integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise ValueError(f"{field} must be a positive integer")
    parsed = int(numeric)
    if parsed < 1:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _record_declared_width(record: Any, *, field: str) -> int:
    """Return the largest concrete width claimed by one layout record."""

    if not isinstance(record, Mapping):
        return 0
    widths: list[int] = []
    if record.get("devices") is not None:
        devices = _parse_sequence(record.get("devices"), field=f"{field}.devices")
        assert devices is not None
        widths.append(len([device for device in devices if device]))
    if record.get("num_chips") is not None:
        widths.append(
            _positive_int(record.get("num_chips"), field=f"{field}.num_chips")
        )
    if record.get("replicas") is not None:
        widths.append(
            _positive_int(record.get("replicas"), field=f"{field}.replicas")
        )
    if record.get("replica_devices") is not None:
        replicas = _parse_sequence(
            record.get("replica_devices"), field=f"{field}.replica_devices"
        )
        assert replicas is not None
        flattened = 0
        for index, replica in enumerate(replicas):
            devices = _parse_sequence(
                replica, field=f"{field}.replica_devices[{index}]"
            )
            assert devices is not None
            flattened += len([device for device in devices if device])
        widths.append(flattened)
    return max(widths, default=0)


def _mapping_value(value: Any, *, field: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"{field} must be a mapping") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _selected_components(
    pipeline_config: Mapping[str, Any], stage: str
) -> tuple[str, ...]:
    """Return every active component occurrence declared for a user stage."""

    selected: list[str] = []
    for node_line in pipeline_config.get("node_lines", ()) or ():
        if not isinstance(node_line, Mapping):
            continue
        for node in node_line.get("nodes", ()) or ():
            if not isinstance(node, Mapping) or node.get("stage") != stage:
                continue
            modules = node.get("modules", ()) or ()
            if not isinstance(modules, Sequence) or isinstance(
                modules, (str, bytes)
            ):
                raise TypeError(f"{stage}.modules must be a sequence")
            for index, module in enumerate(modules):
                if not isinstance(module, Mapping):
                    raise TypeError(
                        f"{stage}.modules[{index}] must be a mapping"
                    )
                component = module.get("component")
                if component:
                    selected.append(str(component))
    return tuple(selected)


def _components_for_performance_stage(
    pipeline_config: Mapping[str, Any],
    performance_stage: str,
) -> tuple[str, ...]:
    from rag_stack.taxonomy import TAXONOMY

    stage = TAXONOMY.performance_stage(str(performance_stage))
    aliases = tuple(dict.fromkeys((
        TAXONOMY.user_stage_of(stage.name).value,
        stage.stage.value,
    )))
    components: list[str] = []
    for alias in aliases:
        components.extend(_selected_components(pipeline_config, alias))
    return tuple(components)


def selected_component_for_performance_stage(
    pipeline_config: Mapping[str, Any],
    performance_stage: str,
    *,
    require_exact: bool = False,
) -> str | None:
    """Resolve one PerformanceStage without guessing across alternatives."""

    components = _components_for_performance_stage(
        pipeline_config,
        performance_stage,
    )
    if len(components) == 1:
        return components[0]
    if require_exact:
        raise NotImplementedError(
            f"PerformanceStage {str(performance_stage)!r} requires exactly one "
            f"resolved active component; found {components!r}"
        )
    return None


def _row_get(row: Mapping[str, Any] | None, key: str) -> Any:
    if row is None:
        return None
    try:
        value = row.get(key)
    except AttributeError:
        return None
    try:
        if value != value:
            return None
    except (TypeError, ValueError):
        pass
    return value


def _candidate_layout(
    pipeline_config: Mapping[str, Any],
    *,
    candidate_row: Mapping[str, Any] | None,
    layout: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if layout is not None:
        return _mapping_value(layout, field="layout")
    raw = _row_get(candidate_row, "candidate_system_layout")
    if raw is not None:
        return _mapping_value(raw, field="candidate_system_layout")
    system = pipeline_config.get("system") or {}
    if isinstance(system, Mapping):
        return _mapping_value(system.get("layout"), field="system.layout")
    return None


def _declared_stage_width(
    stage: str,
    *,
    layout: Mapping[str, Any] | None,
    candidate_row: Mapping[str, Any] | None,
) -> int:
    widths: list[int] = []
    from rag_stack.taxonomy import TAXONOMY

    try:
        expected_engine = TAXONOMY.user_stage_of(stage).value
    except (KeyError, ValueError):
        expected_engine = None
    if layout is not None:
        stages = layout.get("performance_stages") or {}
        if not isinstance(stages, Mapping):
            raise TypeError("layout.performance_stages must be a mapping")
        widths.append(
            _record_declared_width(
                stages.get(stage), field=f"layout.performance_stages.{stage}"
            )
        )

        info = stages.get(stage)
        declared_engine = info.get("engine") if isinstance(info, Mapping) else None
        if (
            declared_engine
            and expected_engine
            and str(declared_engine) != expected_engine
        ):
            raise ValueError(
                f"layout.performance_stages.{stage}.engine claims "
                f"{declared_engine!r}; expected {expected_engine!r}"
            )
        engines = layout.get("engines") or {}
        if not isinstance(engines, Mapping):
            raise TypeError("layout.engines must be a mapping")
        if expected_engine:
            widths.append(
                _record_declared_width(
                    engines.get(expected_engine),
                    field=f"layout.engines.{expected_engine}",
                )
            )

    if stage == "semantic_retrieval_encode":
        stage_aliases = (
            "semantic_retrieval_encode", "encode", "semantic_retrieval",
        )
        numeric_columns = (
            "num_chips_semantic_retrieval_encode",
            "num_chips_encode",
            "num_gpus_semantic_retrieval_encode",
            "num_gpus_semantic_retrieval",
            "num_chips_semantic_retrieval",
            "candidate_replicas_semantic_retrieval_encode",
        )
    else:
        stage_aliases = (stage,)
        numeric_columns = (
            f"num_chips_{stage}",
            f"num_gpus_{stage}",
            f"candidate_replicas_{stage}",
        )

    for alias in stage_aliases:
        column = f"candidate_devices_{alias}"
        flat_devices = _row_get(candidate_row, column)
        if flat_devices is not None:
            devices = _parse_sequence(flat_devices, field=column)
            assert devices is not None
            widths.append(len([device for device in devices if device]))

    legacy_stage_devices = _row_get(candidate_row, "stage_devices")
    if legacy_stage_devices is not None:
        stage_device_map = _mapping_value(
            legacy_stage_devices, field="stage_devices"
        )
        assert stage_device_map is not None
        for alias in stage_aliases:
            if stage_device_map.get(alias) is None:
                continue
            devices = _parse_sequence(
                stage_device_map[alias], field=f"stage_devices.{alias}"
            )
            assert devices is not None
            widths.append(len([device for device in devices if device]))

    for column in numeric_columns:
        value = _row_get(candidate_row, column)
        if value is not None:
            widths.append(_positive_int(value, field=column))
    return max(widths, default=0)


def _validate_measured_aux_forward_parallelism(
    pipeline_config: Mapping[str, Any],
    *,
    layout: Mapping[str, Any] | None,
    candidate_row: Mapping[str, Any] | None,
    source: str,
) -> None:
    measured_aux_dp = {
        "passage_filter": frozenset(),
        "passage_compressor": frozenset({"llmlingua2"}),
    }
    from rag_stack.taxonomy import SHAPE_FORWARD, TAXONOMY

    for performance_stage, allowed in measured_aux_dp.items():
        width = _declared_stage_width(
            performance_stage,
            layout=layout,
            candidate_row=candidate_row,
        )
        if width <= 1:
            continue
        component = selected_component_for_performance_stage(
            pipeline_config,
            performance_stage,
            require_exact=True,
        )
        spec = TAXONOMY.components.get(component)
        if spec is not None and spec.shape != SHAPE_FORWARD:
            continue
        if component not in allowed:
            raise NotImplementedError(
                f"{source}: PerformanceStage {performance_stage!r} component "
                f"{component!r} declares {width} synchronized replicas, but "
                "the measured runtime implements forward-DP only for "
                f"{sorted(allowed)!r}"
            )


def _validate_llm_role_parallelism(
    *,
    layout: Mapping[str, Any] | None,
    candidate_row: Mapping[str, Any] | None,
    source: str,
) -> None:
    from rag_stack.taxonomy import TAXONOMY

    stages = (layout.get("performance_stages") or {}) if layout else {}
    if stages and not isinstance(stages, Mapping):
        raise TypeError("layout.performance_stages must be a mapping")
    for stage in sorted(TAXONOMY.llm_performance_stages):
        info = stages.get(stage) if isinstance(stages, Mapping) else None
        if isinstance(info, Mapping):
            devices = _parse_sequence(
                info.get("devices"),
                field=f"layout.performance_stages.{stage}.devices",
            )
            if devices is None:
                continue
            device_count = len([device for device in devices if device])
            if device_count < 1:
                continue
            replicas = _positive_int(
                info.get("replicas", 1),
                field=f"layout.performance_stages.{stage}.replicas",
            )
            tp = _positive_int(
                info.get("tp"),
                field=f"layout.performance_stages.{stage}.tp",
            )
            pp = _positive_int(
                info.get("pp"),
                field=f"layout.performance_stages.{stage}.pp",
            )
            if replicas != 1 or tp * pp != device_count:
                raise NotImplementedError(
                    f"{source}: LLM PerformanceStage {stage!r} requires one "
                    "measured TP×PP engine; got "
                    f"tp={tp}, pp={pp}, replicas={replicas}, "
                    f"devices={device_count}"
                )
            continue

        replicas_value = _row_get(candidate_row, f"candidate_replicas_{stage}")
        if replicas_value is not None and _positive_int(
            replicas_value, field=f"candidate_replicas_{stage}"
        ) != 1:
            raise NotImplementedError(
                f"{source}: LLM PerformanceStage {stage!r} does not "
                "implement data-parallel replicas"
            )


def validate_measured_runtime_parallelism(
    pipeline_config: Mapping[str, Any],
    *,
    candidate_row: Mapping[str, Any] | None = None,
    layout: Mapping[str, Any] | None = None,
    source: str = "candidate deployment",
) -> None:
    """Reject a concrete layout absent from the measured runtime."""

    resolved_layout = _candidate_layout(
        pipeline_config, candidate_row=candidate_row, layout=layout
    )
    encode_width = _declared_stage_width(
        "semantic_retrieval_encode",
        layout=resolved_layout,
        candidate_row=candidate_row,
    )
    validate_measured_parallelism_width(
        stage="semantic_retrieval_encode",
        component="vectordb",
        width=encode_width,
        source=source,
    )

    reranker_components = _selected_components(
        pipeline_config, "passage_reranker"
    )
    reranker_width = _declared_stage_width(
        "passage_reranker",
        layout=resolved_layout,
        candidate_row=candidate_row,
    )
    if reranker_width > 1 and len(reranker_components) != 1:
        raise NotImplementedError(
            f"{source}: multi-GPU passage_reranker requires exactly one "
            "resolved active component; found "
            f"{list(reranker_components)!r}"
        )
    reranker_component = (
        reranker_components[0] if len(reranker_components) == 1 else None
    )
    from rag_stack.taxonomy import SHAPE_LLM, TAXONOMY

    reranker_spec = TAXONOMY.components.get(reranker_component)
    if reranker_spec is None or reranker_spec.shape != SHAPE_LLM:
        validate_measured_parallelism_width(
            stage="passage_reranker",
            component=reranker_component,
            width=reranker_width,
            source=source,
        )
    _validate_llm_role_parallelism(
        layout=resolved_layout,
        candidate_row=candidate_row,
        source=source,
    )
    _validate_measured_aux_forward_parallelism(
        pipeline_config,
        layout=resolved_layout,
        candidate_row=candidate_row,
        source=source,
    )


def validate_candidate_frame_runtime_parallelism(
    frame: Any,
    pipeline_config: Mapping[str, Any],
    *,
    source: str,
) -> None:
    """Validate every concrete row entering measured-runtime replay."""

    if frame is None or len(frame) == 0:
        return
    for position in range(len(frame)):
        validate_measured_runtime_parallelism(
            pipeline_config,
            candidate_row=frame.iloc[position],
            source=f"{source} row {position}",
        )


__all__ = [
    "RERANKER_CM_COMPOSABLE_DP_COMPONENTS",
    "RERANKER_MEASURED_DP_COMPONENTS",
    "SEMANTIC_RETRIEVAL_ENCODE_MAX_GPUS",
    "selected_component_for_performance_stage",
    "validate_candidate_frame_runtime_parallelism",
    "validate_measured_parallelism_width",
    "validate_measured_runtime_parallelism",
]
