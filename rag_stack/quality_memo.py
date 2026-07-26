"""Quality memoization — one expensive GT eval per unique ALGO sub-config.

Quality (the GPU pipeline run + LLM-judge scores) depends only on the algo
knobs (corpus/chunker, vectordb, retrieval, reranker, generator, prompts) —
never on the deployment (``system`` dims: preset/layout/TP·PP/batch), which
only changes how fast the same answers are produced. In the joint-space
baseline the optimizer routinely proposes the SAME algo sub-config under a
different ``deployment_preset``/``batch_size_request``; without memoization
each such trial re-burns the full GPU + judge run to reproduce a known
number.

Soundness scope: the memo is consulted only on the controller's standalone
GT-eval path (performance from the cost model). The measured path co-produces
quality and performance in one real run (``MeasuredProvider``) and keeps its
own ``(pipeline_config, system_config)``-keyed memo — untouched.

The recorded ``__execution_dag__`` trace is cached WITH the metrics: dynamic
CM pricing replays the quality run's trace, and the cached trace IS that
algo config's recorded execution — reusing (metrics, trace) as a unit is
precisely "this algo config was already measured once."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


def quality_memo_key(decoded_config: dict) -> str:
    """Hash of the decoded config MINUS the top-level ``system`` key.

    ``system`` holds every optimizer-facing deployment dim (the ``system__*``
    namespace: deployment_preset, gpu_layout, batch, faiss_num_threads — all
    timing-only). Everything else is the quality-relevant algo sub-vector.
    """
    from rag_stack.optimizer.utils import config_hash

    algo = {k: v for k, v in (decoded_config or {}).items() if k != "system"}
    return config_hash(algo)


@dataclass
class QualityMemoEntry:
    """One measured algo sub-config: pure metrics + its recorded trace.

    ``full`` distinguishes a full-metrics eval from a minimal-metrics one
    (ax-gen steps evaluate only the objective metric): a full entry serves
    both kinds of requests, a minimal entry never serves a full request.

    Resume-seeded entries carry ``execution_dag_path`` instead of the loaded
    trace (traces can be MBs each — hundreds of eagerly-loaded ones would
    bloat resume); :meth:`load_execution_dag` reads it on first use.
    """

    quality_metrics: dict
    execution_dag: Optional[Any]
    full: bool
    execution_dag_path: Optional[str] = None

    def serves(self, want_full: bool) -> bool:
        return self.full or not want_full

    def load_execution_dag(self) -> Optional[Any]:
        if self.execution_dag is None and self.execution_dag_path:
            import json

            try:
                with open(self.execution_dag_path) as f:
                    self.execution_dag = json.load(f)
            except Exception:  # noqa: BLE001 — a lost trace degrades, below
                self.execution_dag_path = None
        return self.execution_dag


MEMO_MARKER_FILENAME = "quality_memo.json"
