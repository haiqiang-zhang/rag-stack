"""MonoT5 benchmark runner.

Follows the framework convention: instantiates the production ``MonoT5``
class from
``rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.monot5`` and
calls its real ``pure()`` method. Timing / warmup / OOM handling comes
from the shared ``StaticRAGEvaluatorBenchmarkRunner`` base.

MonoT5 is a T5 encoder-decoder. Per passage it runs a full encoder
forward + 1 decoder step to read logits over ``{"▁false", "▁true"}`` —
so the cost model maps it to ``prefill`` (plus a negligible 1-token
decode which GenZ's prefill roofline already captures for this shape
regime).
"""

from __future__ import annotations

from rag_stack.cost_model.llm_sim.calibration.base import (
    StaticRAGEvaluatorBenchmarkRunner,
)


class MonoT5BenchmarkRunner(StaticRAGEvaluatorBenchmarkRunner):
    """Benchmark MonoT5 (`castorini/monot5-3b-msmarco-10k` by default)."""

    # ---- production class to instantiate -------------------------------

    def module_class(self) -> type:
        # Imported lazily so the calibration package can be inspected
        # without pulling torch / transformers at import time.
        from rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.monot5 import (
            MonoT5,
        )
        return MonoT5

    # ---- synthetic input shaped like a real retrieval result -----------

    def build_input(self, batch_size: int, seq_len: int):
        """One query with ``batch_size`` passages of ~``seq_len`` tokens.

        MonoT5 wraps each (query, passage) as
        ``f"Query: {query} Document: {passage}"`` before tokenizing with
        T5 sentencepiece. ``"token "`` repetition lands ~1 BPE piece per
        word, so ``"token " * seq_len`` approximates ``seq_len`` tokens
        for the passage portion.
        """
        import pandas as pd

        query = "What is the theoretical basis"
        passage_text = self._synthesize_passage(seq_len)
        passages = [passage_text for _ in range(batch_size)]
        ids = [f"doc_{i}" for i in range(batch_size)]
        scores = [1.0 for _ in range(batch_size)]

        return pd.DataFrame({
            # Required by validate_qa_dataset
            "qid": ["q0"],
            "query": [query],
            "generation_gt": [["synthetic"]],
            # Consumed by BasePassageReranker.cast_to_run via cast_retrieve_infos
            "retrieved_contents": [passages],
            "retrieved_ids": [ids],
            "retrieve_scores": [scores],
        })

    # ---- kwargs forwarded to module.pure() -----------------------------

    def pure_kwargs(self, batch_size: int, seq_len: int) -> dict:
        # Match MonoT5's pure() signature: pure(previous_result, *, top_k, batch)
        return {
            "top_k": batch_size,
            "batch": batch_size,
        }
