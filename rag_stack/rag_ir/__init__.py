"""RAG IR — the data layer the cost model consumes.

rag_ir owns the **data-source axis** and emits ``(workflow_schema, trace)`` to
rag_cm:

  * :mod:`rag_stack.rag_ir.trace` — the generic,
    pipeline-shape-agnostic per-query execution trace (component calls +
    quality-invocation token I/O). The DATA (real I/O by default; token_stats
    degraded).
  * :mod:`rag_stack.rag_ir.workflow_schema` — :class:`WorkflowSchema`: RAGO's
    ``RAGSchema`` equivalent. The STRUCTURE rag_cm dispatches on
    (``dataflow_type`` sequential/agentic + which stages are present).

The trace-driven cost ENGINE lives in
:class:`rag_stack.cost_model.assembly.RAGCMAssembly` (the sibling of
``StaticAssembly``). Dependency direction: ``cost_model → rag_ir`` (types +
schema only), never the reverse.
"""

from __future__ import annotations

from rag_stack.rag_ir.execution_path import compile_execution_paths

from rag_stack.rag_ir.trace import (  # noqa: F401
    TraceCall,
    Trace,
    TracePayload,
    QualityTraceEnvelope,
    QualityTraceInput,
    QualityTraceNormalization,
    TRACE_CALL_OPTIONAL_FIELDS,
    TRACE_CALL_REQUIRED_FIELDS,
    make_quality_trace_envelope,
    is_quality_trace_candidate,
    normalize_quality_trace,
    is_trace_payload,
    validate_quality_trace_envelope,
)
from rag_stack.rag_ir.workflow_schema import (  # noqa: F401
    CorpusStats,
    WorkflowSchema,
    corpus_stats_from_pipeline_config,
    dataflow_type_of,
    build_workflow_schema,
    validate_workflow_trace,
)
from rag_stack.rag_ir.plan import describe_pipeline

__all__ = [
    "compile_execution_paths",
    "TraceCall",
    "Trace",
    "TracePayload",
    "QualityTraceEnvelope",
    "QualityTraceInput",
    "QualityTraceNormalization",
    "TRACE_CALL_OPTIONAL_FIELDS",
    "TRACE_CALL_REQUIRED_FIELDS",
    "make_quality_trace_envelope",
    "is_quality_trace_candidate",
    "normalize_quality_trace",
    "is_trace_payload",
    "validate_quality_trace_envelope",
    "CorpusStats",
    "WorkflowSchema",
    "corpus_stats_from_pipeline_config",
    "dataflow_type_of",
    "build_workflow_schema",
    "validate_workflow_trace",
    "describe_pipeline",
]
