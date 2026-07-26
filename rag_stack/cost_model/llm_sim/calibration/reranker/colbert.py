"""ColBERT reranker benchmark runner.

Default model ``colbert-ir/colbertv2.0`` is a BERT-base late-interaction
model. It encodes query and each passage into per-token embeddings via
``AutoModel(**inputs).last_hidden_state``, then computes a MaxSim score
in numpy. Cost model treats it as prefill-only (the numpy post-proc is
CPU-bound and small for realistic batch/seq sizes).
"""

from __future__ import annotations

from rag_stack.cost_model.llm_sim.calibration.base import (
    StaticRAGEvaluatorBenchmarkRunner,
)


class ColbertBenchmarkRunner(StaticRAGEvaluatorBenchmarkRunner):
    """Benchmark ColbertReranker (`colbert-ir/colbertv2.0`)."""

    def module_class(self) -> type:
        from rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.colbert import (
            ColbertReranker,
        )
        return ColbertReranker

    def build_input(self, batch_size: int, seq_len: int):
        """One query with ``batch_size`` passages of ~``seq_len`` tokens."""
        import pandas as pd

        query = "What is the theoretical basis"
        passage_text = self._synthesize_passage(seq_len)
        passages = [passage_text for _ in range(batch_size)]
        ids = [f"doc_{i}" for i in range(batch_size)]
        scores = [1.0 for _ in range(batch_size)]

        return pd.DataFrame({
            "qid": ["q0"],
            "query": [query],
            "generation_gt": [["synthetic"]],
            "retrieved_contents": [passages],
            "retrieved_ids": [ids],
            "retrieve_scores": [scores],
        })

    def pure_kwargs(self, batch_size: int, seq_len: int) -> dict:
        # ColbertReranker.pure() reads top_k + batch from kwargs.
        return {"top_k": batch_size, "batch": batch_size}
