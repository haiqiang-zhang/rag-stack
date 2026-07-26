from .base import SyntheticFunction
from .rag_stack_historical_surrogate import RagStackHistoricalSurrogateFunction

# Only the real-data historical surrogate is kept. The analytical/stylized testbeds
# (synthetic_rag, yahpo, zdt*, dtlz2, branin_currin, penicillin, nasbench201, mobons,
# rag_hpo_bench) were removed: synthetic_rag was shown not to transfer to real e2e, and
# the rest were generic MOBO benchmarks unused by the RAG optimizer work.
SYNTHETIC_FUNCTION_REGISTRY = {
    "rag_stack_historical_surrogate": RagStackHistoricalSurrogateFunction,
}
