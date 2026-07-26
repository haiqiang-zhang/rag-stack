"""RAG-Stack-owned evaluator contract, discovery, and execution tracing.

Backend packages depend on this interface; the interface never imports an
unselected evaluator implementation.
"""

from rag_stack.evaluation.api import (
    EvaluationResult, Evaluator, BaseEvaluator, normalize_evaluation_result, publish_trace,
)
from rag_stack.evaluation.registry import (
    BackendSpec, EvaluatorContext, available_backends, create_quality_evaluator,
    get_backend_spec, register_backend, supported_metrics, supported_pipeline_modes,
)

from rag_stack.evaluation.tracing import (
    TraceSession, StageCall, SUPPORTED_STAGES, capture_trace, get_trace_session,
    trace_query, trace_batch, trace_stage,
)
from rag_stack.rag_ir.io_profile import summarize_items, summarize_lists

__all__ = [
    "EvaluationResult", "Evaluator", "BaseEvaluator", "normalize_evaluation_result", "publish_trace",
    "BackendSpec", "EvaluatorContext", "available_backends", "create_quality_evaluator",
    "get_backend_spec", "register_backend", "supported_metrics", "supported_pipeline_modes",
    "TraceSession", "StageCall", "SUPPORTED_STAGES", "capture_trace",
    "get_trace_session", "trace_query", "trace_batch", "trace_stage",
    "summarize_items", "summarize_lists",
]
