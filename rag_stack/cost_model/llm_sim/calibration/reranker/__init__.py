"""Per-module benchmark runner factory for the passage_reranker node.

Each concrete reranker component (`tart`, `monot5`, `flag_embedding_reranker`,
etc.) registers its own ``BaseBenchmarkRunner`` subclass here. Both
``PassageRerankerCostModel.benchmark()`` and ``PassageRerankerCostModel.calibrate()``
dispatch through ``get_runner(component)``.
"""

from __future__ import annotations

from rag_stack.cost_model.llm_sim.calibration.base import BaseBenchmarkRunner

# ---------------------------------------------------------------------------
# Registry — lazy imports (don't pull torch/transformers unless needed)
# ---------------------------------------------------------------------------

def _build_tart() -> BaseBenchmarkRunner:
    from rag_stack.cost_model.llm_sim.calibration.reranker.tart import (
        TartBenchmarkRunner,
    )
    return TartBenchmarkRunner()


def _build_monot5() -> BaseBenchmarkRunner:
    from rag_stack.cost_model.llm_sim.calibration.reranker.monot5 import (
        MonoT5BenchmarkRunner,
    )
    return MonoT5BenchmarkRunner()


def _build_flag_embedding() -> BaseBenchmarkRunner:
    from rag_stack.cost_model.llm_sim.calibration.reranker.flag_embedding import (
        FlagEmbeddingBenchmarkRunner,
    )
    return FlagEmbeddingBenchmarkRunner()


def _build_sentence_transformer() -> BaseBenchmarkRunner:
    from rag_stack.cost_model.llm_sim.calibration.reranker.sentence_transformer import (
        SentenceTransformerBenchmarkRunner,
    )
    return SentenceTransformerBenchmarkRunner()


def _build_colbert() -> BaseBenchmarkRunner:
    from rag_stack.cost_model.llm_sim.calibration.reranker.colbert import (
        ColbertBenchmarkRunner,
    )
    return ColbertBenchmarkRunner()


# Each value is a thunk that returns a fresh runner instance when called.
# Using thunks keeps heavy imports (torch, transformers) lazy — modules are
# only loaded when the user actually calibrates that reranker.
RERANKER_RUNNERS: dict[str, callable] = {
    "tart": _build_tart,
    "monot5": _build_monot5,
    "flag_embedding_reranker": _build_flag_embedding,
    "sentence_transformer_reranker": _build_sentence_transformer,
    "colbert_reranker": _build_colbert,
    # "flag_embedding_llm_reranker": …,                   # TODO (uses PassageRerankerLLMCostModel)
    # "upr": …,                                            # TODO (uses PassageRerankerLLMCostModel)
    # "flashrank_reranker": …,                            # TODO
    # "openvino_reranker": …,                             # TODO
    # "cohere_reranker" / "jina_reranker" / "voyageai_reranker" /
    # "mixedbreadai_reranker": API-based — calibration only meaningful
    # if you want to benchmark network latency, needs different design.
}


def get_runner(component: str) -> BaseBenchmarkRunner:
    """Instantiate the benchmark runner for a reranker component.

    Raises ``NotImplementedError`` with a descriptive message if no runner
    is registered.
    """
    factory = RERANKER_RUNNERS.get(component)
    if factory is None:
        raise NotImplementedError(
            f"No reranker calibration runner for component={component!r}. "
            f"Implemented: {sorted(RERANKER_RUNNERS.keys())}. "
            f"Add one under rag_stack/cost_model/llm_sim/calibration/reranker/."
        )
    return factory()


__all__ = ["get_runner", "RERANKER_RUNNERS"]
