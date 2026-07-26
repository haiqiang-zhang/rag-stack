"""Host boundary for evaluator scores and persistent RAG-IR metadata.

RAG-Stack owns the evaluator API, registration, and instrumentation. This module
validates result cohorts and binds metadata for performance modeling/cache replay.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Iterable

from rag_stack.evaluation.api import normalize_evaluation_result


def dataset_owner_config_from_resolved(pipeline_config: dict) -> dict:
    """Adapt concrete static pipeline choices to the host dataset-owner schema.

    DatasetManager derives chunking/token statistics from the host configuration
    shape. Evaluator.evaluate receives the original resolved shape instead. This
    adapter carries only selected choices, without dataset paths, archived token
    statistics or optimizer candidates. Supply QA/corpus paths to the manager.
    """
    if not isinstance(pipeline_config.get("node_lines"), list):
        raise ValueError("A resolved static pipeline requires node_lines")
    return {
        "algo_search_space": {
            "corpus": deepcopy(pipeline_config.get("corpus_runtime") or {"chunker": {}}),
            "vectordb": deepcopy(pipeline_config.get("vectordb") or []),
            "pipeline": {**deepcopy(pipeline_config.get("pipeline_runtime") or {}),
                         "node_lines": deepcopy(pipeline_config["node_lines"])},
        },
        "system": deepcopy(pipeline_config.get("system") or {}),
    }


def configured_backend_options(config: dict) -> dict:
    """Read uniform backend options, translating the old FlashRAG YAML key."""
    settings = config.get("eval_backend_setting") or {}
    options = settings.get("backend_options")
    if options is not None:
        return options
    backend = str((config.get("global") or {}).get("eval_backend", "static_gt")).lower()
    return (settings.get("flashrag") or {}) if backend == "flashrag" else {}


def bind_trace_workflow(pipeline_config: dict, trace: dict | None) -> None:
    """Bind explicit workflow metadata for execution, measurement, and modeling.

    Provenance survives trace-cache persistence. Remove stale overrides when a
    different trial has no explicit workflow. No trace or caller schema is copied
    back into the evaluator's result, preserving early-callback object identity.
    """
    provenance = trace.get("provenance") if isinstance(trace, dict) else None
    workflow = provenance.get("workflow_schema") if isinstance(provenance, dict) else None
    if workflow is None:
        pipeline_config.pop("_evaluator_workflow_schema", None)
    else:
        from rag_stack.rag_ir.workflow_schema import WorkflowSchema, validate_workflow_trace
        schema = WorkflowSchema.from_dict(workflow)
        if isinstance(trace, dict) and "queries" in trace:
            validate_workflow_trace(schema, trace)
        pipeline_config["_evaluator_workflow_schema"] = deepcopy(workflow)
    if (isinstance(trace, dict) and "queries" in trace
            and isinstance(provenance, dict) and provenance.get("evaluator_backend")):
        from rag_stack.rag_ir.normalize import with_observed_trace_components

        observed = with_observed_trace_components(
            pipeline_config, [query["calls"] for query in trace["queries"]],
        )
        for key in ("_evaluator_trace_stages", "_evaluator_fusion_calls",
                    "_evaluator_fusion_calls_per_retrieval"):
            pipeline_config[key] = observed[key]
    else:
        pipeline_config.pop("_evaluator_trace_stages", None)
        pipeline_config.pop("_evaluator_fusion_calls", None)
        pipeline_config.pop("_evaluator_fusion_calls_per_retrieval", None)
    from rag_stack.pipeline_contract import has_hybrid_retrieval
    if has_hybrid_retrieval(pipeline_config) and trace is not None:
        if not isinstance(trace, dict) or "queries" not in trace:
            raise ValueError("Hybrid retrieval requires a canonical quality trace; recollect the trace")
        from rag_stack.rag_ir.workflow_schema import build_workflow_schema
        build_workflow_schema(pipeline_config, trace=trace)


def normalize_backend_result(
    result,
    *,
    backend: str,
    pipeline_config: dict,
    question_ids: Iterable | None = None,
) -> dict:
    """Normalize evaluator output and enforce the common cohort contract.

    All backends follow the same contract. Dynamic evaluation requires a canonical
    trace covering the evaluated questions; explicit static mode accepts aggregate
    runtime statistics.
    Metric names/values and workflow declarations are validated by the public API.
    """
    require_trace = (pipeline_config.get("_runtime") or {}).get("rag_ir_mode", "dynamic") != "static"
    payload = normalize_evaluation_result(
        result, require_trace=require_trace,
    )
    trace = payload.get("__execution_dag__")
    if isinstance(trace, dict) and "queries" in trace:
        if trace.get("provenance") is None:
            trace["provenance"] = {}
        provenance = trace["provenance"]
        provenance["evaluator_backend"] = str(backend).strip().lower()
    if isinstance(trace, dict) and "queries" in trace and question_ids is not None:
        expected = [str(qid) for qid in question_ids]
        actual = [str(query["question_id"]) for query in trace["queries"]]
        if len(set(expected)) != len(expected):
            raise ValueError("Evaluator dataset contains duplicate question IDs")
        if set(actual) != set(expected) or len(actual) != len(expected):
            raise ValueError(
                "Evaluator trace must cover exactly the evaluated question IDs; "
                f"missing={sorted(set(expected) - set(actual))}, "
                f"unexpected={sorted(set(actual) - set(expected))}"
            )
    # The serialized schema belongs to RAG-IR, never to optimizer objectives.
    payload.pop("__workflow_schema__", None)
    bind_trace_workflow(pipeline_config, trace)
    return payload
