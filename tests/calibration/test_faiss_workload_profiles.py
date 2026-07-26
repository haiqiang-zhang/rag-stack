"""CPU contracts for shared FAISS profile preparation and corpus-scoped caches."""

from types import SimpleNamespace

import pytest

from rag_stack.cost_model import workload_profile
from rag_stack.cost_model.faiss_hnsw_sim.adapter import FaissHNSWCostModel
from rag_stack.cost_model.faiss_ivf_sim.adapter import FaissIVFCostModel


def context(*, corpus="corpus", embeddings=("embedding",), provider=None):
    return SimpleNamespace(
        corpus_key=corpus,
        search_space={"embedding_models": list(embeddings), "nprobe": [1, 4]},
        vector_provider=provider,
    )


@pytest.fixture(autouse=True)
def isolated_profile_caches(monkeypatch):
    monkeypatch.setattr(workload_profile, "_PROCESSED", set())
    monkeypatch.setattr(FaissIVFCostModel, "_corpus_id", "")
    monkeypatch.setattr(FaissIVFCostModel, "_profile_cache", {})
    monkeypatch.setattr(FaissHNSWCostModel, "_predictor_cache", {})


@pytest.mark.parametrize("adapter", [FaissIVFCostModel, FaissHNSWCostModel])
def test_adapter_without_vector_provider_does_not_build_profiles(monkeypatch, adapter):
    monkeypatch.setattr(workload_profile, "ensure_workload_profiles",
                        lambda ctx: pytest.fail("No vectors were supplied"))
    adapter().init(context())


@pytest.mark.parametrize("adapter", [FaissIVFCostModel, FaissHNSWCostModel])
def test_adapter_passes_the_run_context_to_shared_profile_preparation(monkeypatch, adapter):
    calls = []
    ctx = context(provider=lambda embedding: pytest.fail("No encoding in this test"))
    monkeypatch.setattr(workload_profile, "ensure_workload_profiles", calls.append)
    adapter().init(ctx)
    assert calls == [ctx]


def test_shared_preparation_builds_once_per_corpus_and_embedding(monkeypatch):
    calls = []
    provider = object()
    monkeypatch.setattr(workload_profile, "_ensure_for_embedding",
                        lambda *args: calls.append(args))
    ctx = context(embeddings=("embed-a", "embed-b", "embed-a"), provider=provider)
    FaissIVFCostModel().init(ctx)
    FaissHNSWCostModel().init(ctx)
    workload_profile.ensure_workload_profiles(context(corpus="other", embeddings=("embed-a",), provider=provider))
    assert calls == [
        ("corpus", "embed-a", provider, [1, 4]),
        ("corpus", "embed-b", provider, [1, 4]),
        ("other", "embed-a", provider, [1, 4]),
    ]


def test_ivf_run_initialization_discards_previous_corpus_cache(monkeypatch):
    from rag_stack.cost_model.faiss_ivf_sim import imbalance

    calls = []

    def load(key, directory):
        calls.append(key)
        return {"nlist": {"8": {"1": 1.25 if key.startswith("first") else 1.5}}}

    monkeypatch.setattr(imbalance, "load_imbalance_profile", load)
    adapter = FaissIVFCostModel()
    adapter.init(context(corpus="first"))
    assert adapter._cell_imbalance("embedding", 8, 1) == 1.25
    assert adapter._cell_imbalance("embedding", 8, 1) == 1.25
    adapter.init(context(corpus="second"))
    assert adapter._cell_imbalance("embedding", 8, 1) == 1.5
    assert calls == ["first__embedding", "second__embedding"]


def test_hnsw_predictor_cache_separates_corpus_and_embedding(monkeypatch):
    from rag_stack.cost_model.faiss_hnsw_sim.workload_predictor import WorkloadPredictor

    calls = []
    predictors = {}

    def load(dataset, embedding):
        calls.append((dataset, embedding))
        return predictors.setdefault((dataset, embedding), object())

    monkeypatch.setattr(WorkloadPredictor, "load_for_dataset", load)
    monkeypatch.setattr(WorkloadPredictor, "load_default", lambda: pytest.fail("Exact profile exists"))
    adapter = FaissHNSWCostModel()
    first = adapter._resolve_predictor({"dataset": {"dataset_name": "one"}}, {"embedding_model": "a"})
    assert adapter._resolve_predictor({"dataset": {"dataset_name": "one"}}, {"embedding_model": "a"}) is first
    assert adapter._resolve_predictor({"dataset": {"dataset_name": "one"}}, {"embedding_model": "b"}) is not first
    assert adapter._resolve_predictor({"dataset": {"dataset_name": "two"}}, {"embedding_model": "a"}) is not first
    assert calls == [("one", "a"), ("one", "b"), ("two", "a")]
