"""Performance-model admissibility, separate from executable RAG-IR validity.

A valid evaluator workflow may contain operations this cost engine cannot
price yet. Reject them here instead of deleting operations or assigning zero
cost. Adding a model also requires an explicit physical execution policy.
"""

from __future__ import annotations

from rag_stack.pipeline_contract import BM25_CM_REASON, has_hybrid_retrieval


def require_cm_pipeline_support(config: dict) -> None:
    if has_hybrid_retrieval(config):
        raise NotImplementedError(BM25_CM_REASON)


def require_cm_trace_support(traces: list) -> None:
    for calls in traces:
        if any(call["stage"] == "lexical_retrieval" for call in calls):
            raise NotImplementedError(BM25_CM_REASON)
        for index, call in enumerate(calls):
            if call["stage"] != "retrieval_fusion":
                continue
            if (index < 2 or calls[index - 1]["stage"] != "semantic_retrieval_vectorsearch"
                    or calls[index - 2]["stage"] != "semantic_retrieval_encode"):
                raise ValueError(
                    "CM retrieval_fusion must follow its observed local encode/search pair; "
                    "nonlocal fusion needs an explicit physical-worker policy"
                )
