"""RAG-Stack's project-level dataset owner.

The host owns the full RAG-Stack configuration, search-space resolution, and
cost-model metadata.  Evaluator code uses the more specific
``DatasetEvalManager`` name for the underlying evaluation data lifecycle.

Keeping this host-facing class in ``rag_stack`` gives callers one stable place
to construct the shared manager while preserving the existing data, chunk
cache, and resume semantics.
"""

from __future__ import annotations

from typing import Any, Optional

from rag_stack_evaluator.static_rag_evaluator.dataset import DatasetEvalManager


class DatasetManager(DatasetEvalManager):
    """Own a RAG-Stack project's dataset and evaluator data lifecycle.

    ``config`` is the host's full configuration document.  The same manager is
    passed to quality and measured evaluators so their active corpus remains
    identical to the host's cost-model view.
    """

    def __init__(
        self,
        *,
        project_dir: str,
        config: Optional[dict] = None,
        qa_data_path: Optional[str] = None,
        corpus_data_path: Optional[str] = None,
        dataset: Optional[Any] = None,
    ) -> None:
        super().__init__(
            project_dir=project_dir,
            config=config,
            qa_data_path=qa_data_path,
            corpus_data_path=corpus_data_path,
            dataset=dataset,
        )
