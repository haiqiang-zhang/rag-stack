"""Validate and copy evaluator-declared calls independently of cost models.

The evaluator trace is the authority for executed operations. Configuration
fanout does not prove that fusion ran, so this compiler never synthesizes RRF
or any other call. Scheduling and component pricing consume these copied paths.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_stack.rag_ir.trace import TracePayload


_QUERY_EXPANSION_DEFAULT_FANOUT = {
    "hyde": 1,
    "query_decompose": 2,
    "multi_query_expansion": 4,
}


def query_expansion_fanout(pipeline_config: dict, summary: object = None) -> int:
    """Resolve subqueries per retrieval visit, independently of visit count.

    An aggregate's runtime value wins and is rounded up.  Otherwise preserve
    the established component defaults: HyDE wraps one passage; decomposition
    uses a heuristic two; multi-query requests three rewrites plus the original.
    These defaults are structural/heuristic inputs, not measured fanout.
    """
    if isinstance(summary, dict) and summary.get("shape") == "aggregate":
        runtime = summary.get("query_expansion_fanout")
        if runtime is not None:
            return max(1, int(math.ceil(float(runtime))))
    for line in pipeline_config.get("node_lines", []) or []:
        for node in line.get("nodes", []) or []:
            if node.get("stage") != "query_expansion":
                continue
            modules = node.get("modules") or []
            if modules:
                default = _QUERY_EXPANSION_DEFAULT_FANOUT.get(
                    modules[0].get("component", ""),
                )
                if default is not None:
                    return default
    return 1


def compile_execution_paths(
    traces: TracePayload, *, retrieval_fanout: int = 1,
) -> TracePayload:
    """Copy declared calls, preserving per-call logical I/O profiles.

    ``retrieval_fanout`` remains a validated compatibility argument for existing
    callers; it never inserts operations. Missing fusion observations stay
    missing. A production service model must reject insufficient workload
    information when a declared operation requires it.
    """
    from rag_stack.rag_ir.io_profile import validate_io_profile
    from rag_stack.rag_ir.call_identity import validate_call_identity

    if (
        isinstance(retrieval_fanout, bool)
        or not isinstance(retrieval_fanout, int)
        or retrieval_fanout < 1
    ):
        raise ValueError("retrieval_fanout must be a positive integer")
    paths = []
    for query in traces:
        calls = (query.get("calls") if isinstance(query, dict) else query) or []
        path = []
        prior_calls = {}
        for index, original in enumerate(calls):
            if not isinstance(original, dict) or not original.get("stage"):
                raise ValueError("execution-path call requires a stage")
            call = deepcopy(original)
            validate_call_identity(call, prior_calls, where=f"execution path call {index}")
            if "call_id" in call:
                prior_calls[call["call_id"]] = call
            for field in ("input_profile", "output_profile"):
                if field in call:
                    validate_io_profile(call[field], where=f"call {index} {field}")
            call["step_idx"] = index
            path.append(call)
        paths.append(path)
    return paths
