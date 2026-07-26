"""Public evaluator result contract, without eager model/runtime imports.

All backends implement :class:`Evaluator` structurally and return
:class:`EvaluationResult`; inheritance from ``BaseEvaluator`` is optional.
RAG-Stack owns validation and the conversion to its internal cache format.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
import logging
from numbers import Real
from typing import Any, Callable, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from rag_stack.rag_ir import QualityTraceEnvelope, WorkflowSchema


@dataclass(frozen=True)
class EvaluationResult:
    """Quality scores and the measured input/output shape of one quality run.

    ``trace`` contains serving component calls only, never judge calls. A
    backend using the host's resolved pipeline structure can omit ``workflow``.
    Otherwise supply a ``WorkflowSchema`` or its JSON representation. Workflows
    are also persisted in trace provenance so cached traces remain self-contained.
    ``runtime_stats`` is optional aggregate data for explicit static CM mode;
    it never substitutes for a canonical trace in dynamic mode. ``metadata``
    carries reserved host metadata such as ``{"__fp_hit__": "donor-id"}``.
    """

    metrics: dict[str, float]
    trace: QualityTraceEnvelope | None = None
    workflow: WorkflowSchema | dict[str, Any] | None = None
    runtime_stats: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Evaluator(Protocol):
    """Structural interface implemented by built-in and external backends."""

    def evaluate(
        self,
        config: dict,
        run_dir: str | None = None,
        metrics_override: Sequence[str | dict] | None = None,
        on_trace_ready: Callable[[dict], None] | None = None,
    ) -> EvaluationResult:
        """Evaluate one resolved configuration.

        A backend may publish its complete serving trace before scoring through
        ``on_trace_ready``. The exact same trace object must be returned; callback
        failures are advisory and must not discard a valid quality result.
        """
        ...


class BaseEvaluator(ABC):
    """Optional base class for the same structural evaluator interface."""

    @abstractmethod
    def evaluate(
        self,
        config: dict,
        run_dir: str | None = None,
        metrics_override: Sequence[str | dict] | None = None,
        on_trace_ready: Callable[[dict], None] | None = None,
    ) -> EvaluationResult:
        """Return scores and serving execution data for one resolved pipeline."""
        raise NotImplementedError


def _workflow_dict(workflow: Any) -> dict:
    # Shared host types are imported only when a result is validated, never at
    # plugin discovery or evaluator-package import time.
    from rag_stack.rag_ir import WorkflowSchema

    if isinstance(workflow, WorkflowSchema):
        schema = workflow
    elif isinstance(workflow, dict):
        schema = WorkflowSchema.from_dict(workflow)
    else:
        raise TypeError("workflow must be a WorkflowSchema or its dictionary representation")
    if schema.dataflow_type not in {"sequential", "agentic"}:
        raise ValueError("workflow.dataflow_type must be 'sequential' or 'agentic'")
    if not isinstance(schema.stages, list) or not all(
        isinstance(stage, str) and stage for stage in schema.stages
    ):
        raise ValueError("workflow.stages must be a list of non-empty stage names")
    for name in ("has_query_expansion", "has_reranker", "has_encode", "has_compressor"):
        if not isinstance(getattr(schema, name), bool):
            raise ValueError(f"workflow.{name} must be a boolean")
    return schema.to_dict()


def normalize_evaluation_result(
    result: EvaluationResult | Mapping[str, Any],
    *,
    require_trace: bool = False,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Validate a result and convert it to the host's internal flat cache format.

    Scores must be finite numbers. ``trace``, when supplied, is always canonical;
    aggregate-only data belongs in ``runtime_stats`` and can be consumed only
    when ``require_trace=False``. This distinction applies to every backend.
    ``allow_legacy`` accepts old flat cache/engine dictionaries; it never relaxes
    validation of typed results and is not the plugin interface.

    The trace object is preserved so early callback/CM overlap can use identity.
    Explicit workflows are persisted in trace provenance before that callback.
    In static mode they are persisted in a copied aggregate payload's provenance;
    a workflow-only result uses an otherwise empty aggregate carrier.
    """
    typed = isinstance(result, EvaluationResult)
    legacy = allow_legacy and not typed
    runtime_stats = None
    if typed:
        if not isinstance(result.metrics, Mapping):
            raise TypeError("EvaluationResult.metrics must be a mapping")
        if not isinstance(result.metadata, Mapping):
            raise TypeError("EvaluationResult.metadata must be a mapping")
        metrics = dict(result.metrics)
        trace = result.trace
        workflow = result.workflow
        runtime_stats = result.runtime_stats
        output: dict[str, Any] = dict(result.metadata)
        for name in output:
            if (not isinstance(name, str) or not name.startswith("__")
                    or name in {"__execution_dag__", "__workflow_schema__"}):
                raise ValueError(f"Invalid evaluator metadata key: {name!r}")
    elif isinstance(result, Mapping):
        output = dict(result)
        trace = output.pop("__execution_dag__", None)
        workflow = output.pop("__workflow_schema__", None)
        if legacy and isinstance(trace, dict) and "queries" not in trace:
            runtime_stats, trace = trace, None
        metrics = {}
        for name, value in list(output.items()):
            if isinstance(name, str) and name.startswith("__"):
                if not legacy:
                    raise ValueError(f"Unsupported evaluator result metadata: {name!r}")
            else:
                metrics[name] = value
                del output[name]
    else:
        raise TypeError("Evaluator.evaluate must return EvaluationResult")

    for name, value in metrics.items():
        if not isinstance(name, str) or not name.strip() or name.startswith("__"):
            raise ValueError(f"Invalid quality metric name: {name!r}")
        if legacy and value is None:
            output[name] = None
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f"Quality metric {name!r} must be a finite number, got {value!r}")
        output[name] = float(value)

    if runtime_stats is not None and not isinstance(runtime_stats, dict):
        raise TypeError("EvaluationResult.runtime_stats must be a dictionary")
    if trace is None:
        if require_trace:
            raise ValueError("This evaluation requires a canonical quality execution trace")
        if runtime_stats is not None:
            output["__execution_dag__"] = runtime_stats
    else:
        from rag_stack.rag_ir import validate_quality_trace_envelope

        validate_quality_trace_envelope(trace)
        output["__execution_dag__"] = trace

    carrier = trace if trace is not None else runtime_stats
    provenance = carrier.get("provenance") if isinstance(carrier, dict) else None
    if provenance is not None and not isinstance(provenance, dict):
        raise TypeError("Execution provenance must be a dictionary or None")
    embedded = provenance.get("workflow_schema") if provenance is not None else None
    if workflow is not None or embedded is not None:
        schema = _workflow_dict(workflow if workflow is not None else embedded)
        if embedded is not None and _workflow_dict(embedded) != schema:
            raise ValueError("Result workflow conflicts with trace provenance workflow_schema")
        if trace is not None:
            from rag_stack.rag_ir.workflow_schema import WorkflowSchema, validate_workflow_trace
            validate_workflow_trace(WorkflowSchema.from_dict(schema), trace)
        if trace is None:
            # Aggregate data has no callback identity contract. Copy before
            # adding metadata, so an adapter's reusable statistics stay intact.
            carrier = deepcopy(runtime_stats) if runtime_stats is not None else {}
        if carrier.get("provenance") is None:
            carrier["provenance"] = {}
        carrier["provenance"]["workflow_schema"] = schema
        output["__execution_dag__"] = carrier
        output["__workflow_schema__"] = schema
    return output


def publish_trace(
    trace: QualityTraceEnvelope,
    *,
    workflow: WorkflowSchema | dict[str, Any] | None = None,
    on_trace_ready: Callable[[dict], None] | None = None,
) -> QualityTraceEnvelope:
    """Validate and publish a complete serving trace before running judges.

    An explicit workflow is attached before the callback runs. Callback errors
    are logged and ignored: the host can retry cost modeling after scoring.
    The return value and callback argument are the original ``trace`` object.
    """
    normalize_evaluation_result(
        EvaluationResult(metrics={}, trace=trace, workflow=workflow), require_trace=True,
    )
    if on_trace_ready is not None:
        try:
            on_trace_ready(trace)
        except Exception:
            logging.getLogger(__name__).warning(
                "on_trace_ready callback failed; continuing quality evaluation",
                exc_info=True,
            )
    return trace


__all__ = ["EvaluationResult", "Evaluator", "BaseEvaluator", "normalize_evaluation_result", "publish_trace"]
