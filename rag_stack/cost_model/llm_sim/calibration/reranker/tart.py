"""TART benchmark runner.

Reference implementation of the framework convention: the runner does
NOT re-implement model loading or inference — it instantiates the
production ``Tart`` class from
``rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.tart.tart`` and
calls its real ``pure()`` method. All timing / warmup / OOM handling
comes from the shared ``StaticRAGEvaluatorBenchmarkRunner`` base.

When adding a new reranker calibration runner, copy this file's structure:
declare ``module_class()``, ``build_input()``, ``pure_kwargs()`` — the
rest is inherited.
"""

from __future__ import annotations

from rag_stack.cost_model.llm_sim.calibration.base import (
    StaticRAGEvaluatorBenchmarkRunner,
)


class TartBenchmarkRunner(StaticRAGEvaluatorBenchmarkRunner):
    """Benchmark TART (`facebook/tart-full-flan-t5-xl`) end-to-end."""

    INSTRUCTION = "Find passage to answer given question"

    # ---- production class to instantiate -------------------------------

    def module_class(self) -> type:
        # Imported lazily so the calibration package can be inspected
        # without pulling torch / transformers at import time.
        from rag_stack_evaluator.static_rag_evaluator.nodes.passagereranker.tart.tart import (
            Tart,
        )
        return Tart

    # ---- synthetic input shaped like a real retrieval result -----------

    def build_input(self, batch_size: int, seq_len: int):
        """One query whose ``retrieved_contents`` list has ``batch_size``
        passages, each padded to roughly ``seq_len`` tokens. Mirrors the
        DataFrame the production retrieval node produces just before TART
        is invoked.
        """
        import pandas as pd

        query = "What is the theoretical basis"
        # Passage length is calibrated via the production tokenizer
        # (see StaticRAGEvaluatorBenchmarkRunner._synthesize_passage).
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
        # Match the signature TART's pure() expects:
        #   pure(previous_result, *, top_k, instruction, batch)
        return {
            "top_k": batch_size,
            "batch": batch_size,
            "instruction": self.INSTRUCTION,
        }
