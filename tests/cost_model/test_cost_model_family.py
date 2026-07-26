from rag_stack.cost_model.faiss_hnsw_sim.adapter import FaissHNSWCostModel
from rag_stack.cost_model.faiss_ivf_sim.adapter import FaissIVFCostModel


def test_faiss_cost_models_are_retrieval_family():
    assert FaissIVFCostModel.family() == "retrieval"
    assert FaissHNSWCostModel.family() == "retrieval"
