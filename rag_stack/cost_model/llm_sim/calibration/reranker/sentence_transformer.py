"""Sentence-Transformers cross-encoder benchmark runner.

Default model ``cross-encoder/ms-marco-MiniLM-L-2-v2`` is a tiny 2-layer
BERT scoring (query, passage) pairs via a single forward through a
classification head. Served through ``sentence_transformers.CrossEncoder``.
Cost model treats it as prefill-only.
"""

from __future__ import annotations

from rag_stack.cost_model.llm_sim.calibration.base import (
    StaticRAGEvaluatorBenchmarkRunner,
)


class SentenceTransformerBenchmarkRunner(StaticRAGEvaluatorBenchmarkRunner):
    """Benchmark SentenceTransformerReranker (`cross-encoder/ms-marco-MiniLM-L-2-v2`)."""

    def module_class(self) -> type:
        from rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.sentence_transformer import (
            SentenceTransformerReranker,
        )
        return SentenceTransformerReranker

    # SentenceTransformerReranker.model is a sentence_transformers.CrossEncoder;
    # the HF tokenizer and the underlying torch nn.Module live one level deeper.
    def tokenizer(self):
        return self._get_module().model.tokenizer

    def torch_module(self):
        return self._get_module().model.model

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
        # SentenceTransformerReranker.pure() reads top_k + batch from kwargs.
        return {"top_k": batch_size, "batch": batch_size}
