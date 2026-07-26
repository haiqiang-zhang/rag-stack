"""Benchmark runner for the query-embedding ("encode") stage.

``RetrievalEncodeCostModel`` (cost_model/retrieval_encode.py) has no
``BaseModule`` wrapper — the embedding model is loaded straight from the
rag-stack registry. So this is a path-B runner (see calibration/base.py):
it drives the PRODUCTION embedding loader
(``EmbeddingModel.load(name)()`` → the same llama_index embedding object the
semantic-retrieval node uses) and times a real batched embed call, rather
than re-implementing model loading / tokenization.

The grid is over BATCH (how many queries are embedded per forward — the
``batch_size_request`` knob the cost model multiplies by); ``seq_len`` is the
per-query token length (queries are short, ~avg_query_tokens). One embed of
``batch_size`` queries == one cost-model forward, so measured and predicted
latencies line up per (batch, seq) point.
"""
from __future__ import annotations

import logging
from statistics import median

from rag_stack.cost_model.llm_sim.calibration.base import BaseBenchmarkRunner

logger = logging.getLogger("RAG-Stack")


class EncodeBenchmarkRunner(BaseBenchmarkRunner):
    """Time the production embedding model on a (batch, query_len) grid."""

    def __init__(self, embedding_model: str):
        self._name = embedding_model
        self._emb = None

    # -- production embedding object (registry loader, same as retrieval) --

    def _get_emb(self):
        if self._emb is None:
            from rag_stack_evaluator.static_rag_evaluator.embedding.base import EmbeddingModel
            self._emb = EmbeddingModel.load(self._name)()
        return self._emb

    def _tokenizer(self):
        """HF tokenizer behind the llama_index embedding (best-effort)."""
        emb = self._get_emb()
        for attr in ("_model", "_embed_model", "model"):
            m = getattr(emb, attr, None)
            tok = getattr(m, "tokenizer", None)
            if tok is not None:
                return tok
        return None

    def _synth_query(self, target_tokens: int) -> str:
        """A query string that tokenizes to ~``target_tokens`` tokens."""
        tok = self._tokenizer()
        if tok is None:
            # ~1.3 tokens/word fallback
            return "token " * max(int(target_tokens / 1.3), 1)
        probe_words = 100
        n = len(tok("token " * probe_words, add_special_tokens=False)["input_ids"])
        tokens_per_word = max(n / probe_words, 0.5)
        words = max(int(max(target_tokens - 2, 1) / tokens_per_word), 1)
        return "token " * words

    def detect_bits(self) -> str:
        # llama_index HuggingFace embeddings load in fp32 by default — matches
        # the f32 encode profiles (and how the production retrieval path runs).
        return "f32"

    def run(self, batch_sizes, seq_lens, warmup, measurements):
        try:
            import torch  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "EncodeBenchmarkRunner requires PyTorch with CUDA for timing."
            ) from e
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available — encode calibration must run on the "
                "target GPU."
            )
        emb = self._get_emb()
        results: dict[tuple[int, int], float] = {}
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)

        for bs in batch_sizes:
            for seq in seq_lens:
                if not self.supports_shape(bs, seq):
                    continue
                queries = [self._synth_query(seq) for _ in range(bs)]
                # One forward over the whole batch (don't let llama_index
                # micro-batch by its default embed_batch_size=10).
                try:
                    emb.embed_batch_size = bs
                except Exception:
                    pass

                for _ in range(warmup):
                    emb.get_text_embedding_batch(queries, show_progress=False)

                torch.cuda.synchronize()
                latencies_ms: list[float] = []
                for _ in range(measurements):
                    start_evt.record()
                    emb.get_text_embedding_batch(queries, show_progress=False)
                    end_evt.record()
                    torch.cuda.synchronize()
                    latencies_ms.append(start_evt.elapsed_time(end_evt))

                med = median(latencies_ms)
                results[(bs, seq)] = med / 1000.0
                logger.info(
                    f"EncodeBenchmarkRunner[{self._name}] (bs={bs}, seq={seq}): "
                    f"median={med:.3f} ms over {len(latencies_ms)} runs"
                )
                torch.cuda.empty_cache()

        return results
