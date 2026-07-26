"""Search-space construction and encoding utilities.

Exports are loaded lazily so lightweight modules such as
``rag_stack.search_space.placement`` can be imported without importing Ax.
"""

__all__ = [
    "AxSearchSpaceOps",
    "INACTIVE_DEFAULT",
    "SEP",
    "ITERATIVE_INCOMPATIBLE_NODES",
    "ITERATIVE_PIPELINE_MODES",
    "RagSearchSpace",
    "algo_vectordb_blocks",
    "is_sweepable",
    "scalarify_value",
]

SEP = "__"
INACTIVE_DEFAULT = 0.5
# Workflow validation needs these names without loading Ax or model runtimes.
# FlashRAG serves the iterative modes; sequential and react are separate paths.
ITERATIVE_PIPELINE_MODES = ("flare", "ircot")


def __getattr__(name: str):
    if name == "AxSearchSpaceOps":
        from rag_stack.search_space import ax_search_space_ops as _ax

        return getattr(_ax, name)
    if name in {
        "ITERATIVE_INCOMPATIBLE_NODES",
        "RagSearchSpace",
        "algo_vectordb_blocks",
        "is_sweepable",
        "scalarify_value",
    }:
        from rag_stack.search_space import rag_search_space as _rag

        return getattr(_rag, name)
    raise AttributeError(f"module 'rag_stack.search_space' has no attribute {name!r}")
