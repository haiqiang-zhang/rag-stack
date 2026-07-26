"""FlagEmbedding reranker benchmark runner.

Follows the framework convention: instantiates the production
``FlagEmbeddingReranker`` class from
``rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.flag_embedding``
and calls its real ``pure()`` method. Timing / warmup / OOM handling
comes from the shared ``StaticRAGEvaluatorBenchmarkRunner`` base.

Default model ``BAAI/bge-reranker-large`` is an XLM-RoBERTa-based
cross-encoder that scores each (query, passage) pair via a single
encoder forward → classification head. Cost model treats it as
prefill-only.
"""

from __future__ import annotations

from rag_stack.cost_model.llm_sim.calibration.base import (
    StaticRAGEvaluatorBenchmarkRunner,
)


class FlagEmbeddingBenchmarkRunner(StaticRAGEvaluatorBenchmarkRunner):
    """Benchmark FlagEmbedding (`BAAI/bge-reranker-large`) end-to-end."""

    def module_class(self) -> type:
        from rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.flag_embedding import (
            FlagEmbeddingReranker,
        )
        return FlagEmbeddingReranker

    # FlagEmbeddingReranker.model is a CrossEncoder / FlagReranker wrapper;
    # the HF tokenizer and torch nn.Module live one level deeper.
    def tokenizer(self):
        return self._get_module().model.tokenizer

    def torch_module(self):
        return self._get_module().model.model

    def build_input(self, batch_size: int, seq_len: int):
        """One query with ``batch_size`` passages of ~``seq_len`` tokens.

        bge-reranker uses XLM-R sentencepiece; ``"token " * seq_len``
        lands close to ``seq_len`` subword pieces per passage.
        """
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
        # FlagEmbeddingReranker.pure() reads top_k + batch from kwargs.
        return {"top_k": batch_size, "batch": batch_size}
