"""Query-embedding ("encode") stage cost model — GenZ-backed, inherits Base.

The retrieval stage of a real RAG pipeline is **encode + vectordb search**:
every (sub-)query is first embedded by the vectordb's bi-encoder on GPU
(~the dominant retrieval cost in measured runs), then searched in FAISS on
CPU. The FAISS side has its own cost models (``faiss_ivf_sim`` /
``faiss_hnsw_sim`` → RAGO stage ``retrieval``); this module supplies the
missing GPU half as RAGO's ``encode`` stage so the CM reports the SAME
granularity the measured path does (``semantic_retrieval_encode`` vs
``semantic_retrieval_search`` in measured ``per_stage_latency_s``).

Semantics: query-time encoding — ``input_tokens = TokenStats.avg_query_tokens``
(the raw user query, no context), ``output_tokens = 1`` (pooled vector, no LM
head — the GenZ model entries use ``vocab_size=1``), one embedding per
(sub-)query. RAGO multiplies the per-query workload by the query-expansion
fanout exactly like the retrieval stage (``rago_sweep._apply_fanout``), and
``EncodeDBPolicy(encode_db_nvec=1)`` degrades RAGO's encode-database math to
per-query embedding (1 vector per request).

Like rerankers, bi-encoders have no tensor-parallel runtime — multi-GPU is
data-parallel replicas — so the sweep emits single-chip rows only and RAGO's
``scale_up_performance_to_more_chips`` extrapolates qps×N.
"""
from __future__ import annotations

from rag_stack.cost_model import register_cost_model
from rag_stack.cost_model.llm_sim.genz_adapter import GenZLLMSimBase
from rag_stack.cost_model.llm_sim.genz_model_map import (
    GENZ_MODEL_MAP,
    resolve_genz_model,
)

def find_retrieval_vectordb_config(pipeline_config: dict) -> dict:
    """The vectordb config dict the semantic_retrieval node points at.

    Mirrors ``build_performance_stage_cost_models``'s db_type lookup: node →
    ``modules[0].vectordb`` name → matching entry in
    ``pipeline_config["vectordb"]``. Empty dict when absent.
    """
    for node_line in pipeline_config.get("node_lines", []):
        for node in node_line.get("nodes", []):
            if node.get("stage") not in ("semantic_retrieval", "hybrid_retrieval"):
                continue
            modules = node.get("modules", [])
            if not modules:
                continue
            vectordb_name = modules[0].get("vectordb", "")
            for vdb in pipeline_config.get("vectordb", []):
                if vdb.get("name") == vectordb_name:
                    return vdb
    return {}


@register_cost_model("retrieval_encode_sim")
class RetrievalEncodeCostModel(GenZLLMSimBase):
    """Query-embedding bi-encoder: prefill only, encoder-only model."""

    def _sweep_max_devices(self, search_space: dict) -> int:
        # Bi-encoders have no TP implementation in the real runtime; multi-GPU
        # is DATA-PARALLEL. Emit single-chip rows; RAGO extrapolates qps×N.
        return 1

    def _get_model(self, pipeline_config: dict):
        vdb = self._find_module(pipeline_config)
        name = vdb.get("embedding_model")
        if isinstance(name, list):
            if len(name) != 1:
                raise ValueError(
                    "retrieval encode pricing requires one materialized "
                    "embedding_model"
                )
            name = name[0]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("vectordb.embedding_model is required for encode pricing")
        if name.lower() not in GENZ_MODEL_MAP:
            raise ValueError(
                f"retrieval embedding model {name!r} has no exact "
                "GENZ_MODEL_MAP physical model; generic encoder fallback is retired"
            )
        return resolve_genz_model(name)

    def _get_params(self, pipeline_config: dict) -> dict:
        from rag_stack.cost_model.token_stats import TokenStats
        ts = TokenStats.resolve(pipeline_config)
        return {
            # Query-time encode: the raw user query length (no context).
            # MUST stay in lock-step with the EncodeDBPolicy
            # ``encode_seq_len_inference_prefill`` built in sweep_inputs —
            # RAGO filters sweep_df["encode"] by exact equality on it.
            "input_tokens": int(ts.avg_query_tokens),
            "output_tokens": 1,
            "bits": "bf16",
        }

    def _find_module(self, pipeline_config: dict) -> dict:
        return find_retrieval_vectordb_config(pipeline_config)

    def _get_model_name_for_df(self, pipeline_config: dict) -> str:
        vdb = self._find_module(pipeline_config)
        name = vdb.get("embedding_model")
        if isinstance(name, list):
            if len(name) != 1:
                raise ValueError(
                    "retrieval encode calibration identity requires one "
                    "materialized embedding_model"
                )
            name = name[0]
        if not isinstance(name, str) or name.lower() not in GENZ_MODEL_MAP:
            raise ValueError(
                f"retrieval embedding model {name!r} has no exact GenZ identity"
            )
        return name

    def _calibration_component(self, pipeline_config: dict) -> str:
        """Embedding records are shared by retrieval backends on one GPU."""
        # The selected vectordb owns ``embedding_model`` but intentionally has
        # no runtime component field.  Production lookup already uses the
        # component-agnostic fallback, so publication must use the same key.
        self._get_model_name_for_df(pipeline_config)
        return "agnostic"

    def _inject_calibration_shape(
        self, pipeline_config: dict, batch_size: int, seq_len: int
    ) -> dict:
        """Encode reads its input length from ``avg_query_tokens`` (the raw
        query — see ``_get_params``) and its batch from the VECTORDB entry
        ``_find_module`` returns, not from a node module. The default
        injection (``avg_chunk_tokens`` + node-module ``batch_size``) is
        therefore shape-blind for this model — the original all-mpnet
        profile had to be fit by hand because of exactly that."""
        import copy
        cfg = copy.deepcopy(pipeline_config)
        ts = cfg.setdefault("_token_stats", {})
        ts["avg_query_tokens"] = int(seq_len)
        ts.setdefault("prompt_template_tokens", 0)
        ts.setdefault("avg_output_tokens", 1)
        for vdb in cfg.get("vectordb", []):
            vdb["batch_size"] = int(batch_size)
        for nl in cfg.get("node_lines", []):
            for node in nl.get("nodes", []):
                for mod in node.get("modules", []):
                    mod["batch_size"] = int(batch_size)
        return cfg

    def _get_benchmark_runner(self, pipeline_config: dict):
        """Path-B runner driving the production embedding loader. Enables
        ``benchmark()`` / ``calibrate()`` for the encode stage so each
        embedding model gets its own (ceff, meff) profile."""
        from rag_stack.cost_model.llm_sim.calibration.encode import (
            EncodeBenchmarkRunner,
        )
        name = self._find_module(pipeline_config).get("embedding_model")
        if not name:
            raise NotImplementedError(
                "encode benchmark needs a vectordb with embedding_model set."
            )
        return EncodeBenchmarkRunner(str(name))
