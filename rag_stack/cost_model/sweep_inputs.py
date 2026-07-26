"""Pure RAGO sweep-INPUT construction — per-stage cost-table sweep dfs, fanout/
token-stat helpers, and stage cost-model assembly. Split out of static_assembly.py
(now assembly.py) so the system-space sweep orchestrator and both engines share
ONE copy of the input builders. No sweep/placement logic here — that lives in
RAGCMAssembly; these only produce what RAGO consumes."""

from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Any

import pandas as pd

from rag_stack.types import SystemConfig
from rag_stack.cost_model.base import CostModelModule
from rag_stack.taxonomy import TAXONOMY, Stage

logger = logging.getLogger("RAG-Stack")

def _retrieval_parallel_modes(pipeline_config: dict) -> list[int]:
    """Candidate ``faiss_ivf_parallel_mode`` values for the vector-search sweep.

    The latest standalone IVF evidence contains mode 0 only. Mode 1 is not a
    candidate design-space value until a separate mode-1 grid exists; accepting
    it would expose hardcoded parallel fractions and uncalibrated cache/sync
    behavior as if they were a ready profile.

    Resolution:
      * scalar pin ``0`` or explicit grid ``[0]`` is accepted;
      * any occurrence of mode ``1`` fails closed;
      * absent resolves to ``[0]``.
      * no faiss_ivf vectordb in the resolved config → single mode (HNSW
        ignores the knob; sweeping it would price identical duplicates).
    """
    system = pipeline_config.get("system") or {}
    pin = (system.get("retrieval") or {}).get("faiss_ivf_parallel_mode")
    if pin is not None:
        if isinstance(pin, (list, tuple)):
            raise ValueError(
                "system.retrieval.faiss_ivf_parallel_mode must be a SCALAR pin — "
                "the CM sweeps {0,1} internally by default; an explicit grid "
                "belongs in system.cm_search_space.faiss_ivf_parallel_mode."
            )
        mode = int(pin)
        if mode != 0:
            raise ValueError(
                "faiss_ivf_parallel_mode=1 is unsupported by the latest "
                "mode-0-only standalone IVF calibration"
            )
        return [0]
    grid = (system.get("cm_search_space") or {}).get("faiss_ivf_parallel_mode")
    modes = (sorted({int(v) for v in grid})
             if isinstance(grid, (list, tuple)) and grid else [0])
    if modes != [0]:
        raise ValueError(
            "system.cm_search_space.faiss_ivf_parallel_mode may contain only "
            "0; mode 1 has no standalone calibration evidence"
        )
    vdbs = pipeline_config.get("vectordb") or []
    ivf_present = any(
        isinstance(v, dict) and str(v.get("db_type", "")) == "faiss_ivf"
        for v in vdbs
    )
    return modes if ivf_present else [0]


def _generate_stage_sweep_df(
    cost_model: CostModelModule,
    pipeline_config: dict,
    hardware: SystemConfig,
    stage: str,
    max_batch_size: int,
) -> pd.DataFrame:
    """Generate a sweep DataFrame for one PerformanceStage.

    For LLM/reranker stages: delegates to cost_model.generate_sweep_df()
    which does RAGO-style multi-device × (TP,PP) × batch_size sweep.

    For the CPU vector search: simple batch_size sweep (no parallelism).
    """
    from rago.utils import get_power_of_two_list

    # Routing is NOT a hardcoded stage list: the FAISS vector-search stage uses
    # a simple per-batch estimate sweep (CPU roofline, no TP/PP); EVERY other
    # PerformanceStage (LLM / reranker / encoder / compressor / filter) uses the
    # RAGO-style generate_sweep_df. Which stages exist is the taxonomy's job.
    if stage == Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH.value:
        modes = _retrieval_parallel_modes(pipeline_config)
        results = []
        for bs in get_power_of_two_list(max_batch_size):
            best: tuple[float, int] | None = None
            for mode in modes:
                config = deepcopy(pipeline_config)
                config.setdefault("system", {}).setdefault("retrieval", {})[
                    "faiss_ivf_parallel_mode"] = mode
                for nl in config.get("node_lines", []):
                    for node in nl.get("nodes", []):
                        if node.get("stage") in ("semantic_retrieval", "hybrid_retrieval"):
                            mods = node.get("modules", [])
                            if mods:
                                mods[0]["batch_size"] = bs
                try:
                    result = cost_model.estimate(config, hardware)
                except Exception as e:
                    logger.warning(
                        f"Retrieval cost model failed for bs={bs}, "
                        f"parallel_mode={mode}: {e}")
                    continue
                if best is None or result.latency_s < best[0]:
                    best = (result.latency_s, mode)
            if best is not None:
                results.append({
                    # "retrieval" here is RAGO's SCHEMA constant for this df
                    # format (like "hardware.name": "system"), consumed by
                    # rago.data.preprocess_data — not a rag_stack stage name.
                    "db.name": "retrieval",
                    "hardware.name": "system",
                    "hardware.num_servers": 1,
                    "batch_size": bs,
                    "time_ms": best[0] * 1000,
                    # diagnostics column: the winning mode per batch (rago
                    # filters by the schema columns; extras pass through).
                    "faiss_ivf_parallel_mode": best[1],
                })
        return pd.DataFrame(results)

    try:
        df = cost_model.generate_sweep_df(
            pipeline_config, hardware, stage, max_batch_size
        )
        logger.info(
            f"generate_sweep_df for {stage}: {len(df)} rows "
            f"(max_num_gpus={hardware.search_space.get('max_num_gpus', 'N/A') if hardware.search_space else 'N/A'})"
        )
        return df
    except NotImplementedError:
        logger.warning(
            f"{cost_model.__class__.__name__} does not implement "
            f"generate_sweep_df, falling back to empty sweep"
        )
        return pd.DataFrame()
    except Exception:
        raise  # Let controller catch and mark INVALID


# ---------------------------------------------------------------------------
# Per-stage max_batch_size resolver (global default + per-node override)
# ---------------------------------------------------------------------------

# Stage ↔ stage mapping is derived from the single stage taxonomy in
# rag_stack/taxonomy.py (served to the frontend via
# GET /api/rag-project/taxonomy/stages).
# NOTE: only `semantic_retrieval` is supported today; hybrid_retrieval has a
# scaffolded node implementation but no working cost-model path yet.

_ALLOWED_NODE_SYSTEM_KEYS = {"max_batch_size"}  # the ONLY key allowed under node.system


# Per-user-request fanout introduced by each query-expansion component.
# Used as a fallback when no quality-eval runtime reading (avg N from the
# DAG payload) is available — typically during the optimizer's CM-only
# init trials, before any quality run has produced a real reading. The
# runtime avg from a quality run always wins.
#
# Provenance (each value is a heuristic, NOT an empirical measurement):
#   * hyde — code-enforced: ``hyde.py`` parser wraps each answer in
#     ``[passage]`` exactly once, so N is structurally 1.
#   * query_decompose — derived from the prompt's 5 few-shot examples:
#     3 are 2-step decompositions, 2 are "no decomposition" (1-step),
#     average ~1.6 → conservative round-up to 2. StrategyQA ground-truth
#     averages ~3 steps (Geva et al. 2021), but the few-shots in this
#     prompt deliberately bias the LLM toward fewer/no decomposition
#     when not strictly needed, so 2 better matches the LLM's actual
#     output for a typical mixed-difficulty RAG benchmark.
#   * multi_query_expansion — prompt asks for "3 different versions"
#     and the parser inserts the original query at position 0, so a
#     compliant LLM produces 3 rewrites + 1 original = 4.
def _get_query_expansion_fanout(
    pipeline_config: dict, dag: object | None = None,
) -> int:
    """Resolve K using the same normalization rule as path compilation."""
    from rag_stack.rag_ir.execution_path import query_expansion_fanout
    return query_expansion_fanout(pipeline_config, dag)


def _get_retrieval_invocations_per_query(
    dag: object | None = None,
) -> float:
    """Mean fused semantic-retrieval worker visits per user query.

    The canonical quality trace emits one encode/vectorsearch pair for each fused worker
    invocation.  This axis is independent from the query-expansion subquery
    fanout resolved by :func:`_get_query_expansion_fanout`.
    """
    if isinstance(dag, dict) and dag.get("shape") == "aggregate":
        value = dag.get("retrieval_invocations_per_query")
        if value is not None:
            return max(1.0, float(value))
    return 1.0


def _get_retrieval_backend_fanout(
    pipeline_config: dict,
    dag: object | None = None,
) -> float:
    """Total encode/search calls per user query: fused visits ``R`` × ``K``.

    RAGO and aggregate communication operate in per-user units and therefore
    need the product.  Per-visit services such as RRF must continue to consume
    only ``K`` through :func:`_get_query_expansion_fanout`.
    """
    if isinstance(dag, dict) and dag.get("shape") == "aggregate":
        total = dag.get("retrieval_backend_fanout")
        if total is not None:
            return max(1.0, float(total))
    total = (
        _get_retrieval_invocations_per_query(dag)
        * _get_query_expansion_fanout(pipeline_config, dag)
    )
    return max(1.0, float(total))


def _get_prefill_fanout(pipeline_config: dict, dag: object | None = None) -> float:
    """Per-user PREFILL fanout = avg # of generator (prefill) calls per user
    request. Agentic traces (react/corag/searchr1) carry it in the aggregate
    payload as ``{"shape": "aggregate", "prefill_fanout": <float>}`` (computed
    by ``DynamicAssembly`` from the measured trace). Kept a FLOAT (no ceil) so
    the mean round count is preserved — ``RAGSweep._apply_fanout`` accepts any
    positive value. Sequential RAG / no-trace inputs → ``1.0`` (RAGO's
    ``_apply_fanout`` is then a no-op, so StaticAssembly stays byte-unchanged)."""
    if isinstance(dag, dict) and dag.get("shape") == "aggregate":
        v = dag.get("prefill_fanout")
        if v is not None:
            return max(1.0, float(v))
    return 1.0


def _get_num_retrievals(dag: object | None) -> float:
    """Per-user prefill→decode handoff count (one per generate round) for the
    communication model. Agentic traces carry a fractional mean
    ``num_retrievals`` in the aggregate payload (kept FLOAT so the PD
    overhead/KV payload scale by the precise mean); sequential RAG / no-trace
    inputs → ``1.0`` (one handoff)."""
    if isinstance(dag, dict) and dag.get("shape") == "aggregate":
        v = dag.get("num_retrievals")
        if v is not None:
            return max(1.0, float(v))
    return 1.0


def _rag_ir_token_stats(dag: object | None) -> dict[str, Any]:
    """Extract runtime token stats from a rag_ir aggregate DAG payload."""
    if not isinstance(dag, dict):
        return {}
    if dag.get("shape") != "aggregate":
        return {}
    if dag.get("rag_ir_mode") != "dynamic":
        return {}
    stats = dag.get("token_stats")
    return stats if isinstance(stats, dict) else {}


def _with_rag_ir_token_overrides(
    pipeline_config: dict,
    dag: object | None,
) -> dict:
    """Return a config copy with rag_ir runtime token stats injected.

    CM-init and CM-only runs pass ``dag=None`` or a ``rag_ir_mode=static``
    aggregate, so this is a no-op and the offline TokenStats remain
    authoritative.
    """
    stats = _rag_ir_token_stats(dag)
    if not stats:
        return pipeline_config

    cfg = deepcopy(pipeline_config)
    token_stats = dict(cfg.get("_token_stats") or {})
    mapping = {
        "avg_query_tokens": "avg_query_tokens",
        "avg_generator_output_tokens": "avg_output_tokens",
        "avg_generator_input_tokens": "avg_llm_input_tokens",
        "avg_query_expansion_input_tokens": "avg_query_expansion_input_tokens",
        "avg_query_expansion_output_tokens": "avg_query_expansion_output_tokens",
        "avg_passage_reranker_input_tokens": "avg_passage_reranker_input_tokens",
    }
    for src, dst in mapping.items():
        value = stats.get(src)
        if value is not None:
            token_stats[dst] = int(value)
    cfg["_token_stats"] = token_stats
    return cfg


def _resolve_main_prefill_tokens(search_space: dict, dag: object | None) -> int | None:
    stats = _rag_ir_token_stats(dag)
    runtime_value = stats.get("avg_generator_input_tokens")
    if runtime_value is not None:
        return int(runtime_value)
    value = search_space.get("seq_len_inference_prefill")
    return int(value) if value is not None else None


def _resolve_main_dec_steps(search_space: dict, dag: object | None) -> int:
    stats = _rag_ir_token_stats(dag)
    runtime_value = stats.get("avg_generator_output_tokens")
    if runtime_value is not None:
        return max(1, int(runtime_value))
    return int(search_space["dec_steps"])


def _get_reranker_input_fanout(
    pipeline_config: dict, dag: object | None = None, default: int = 16,
) -> int:
    """Return the per-user work multiplier at the reranker stage.

    A reranker is a cross-encoder that scores every retrieved passage one
    (query, passage) pair at a time before keeping the top-k survivors, so
    its compute scales with the **input** count: how many passages flow
    INTO the reranker.

    With ``query_expansion`` enabled, one user request produces ``N``
    sub-queries, each retrieving ``semantic_retrieval.top_k`` passages — BUT
    the retrieval node FUSES those N lists back down to ``top_k`` (RRF /
    even-distribute, inside the retrieval node) BEFORE the reranker. So a
    separate downstream ``passage_reranker`` scores exactly ``top_k`` (query,
    passage) pairs per user request, NOT ``top_k × N``. The expansion fanout
    ``N`` is already charged at the retrieval/encode stages and must not be
    double-counted here. (Only a "reranker-as-merge" design — the reranker
    consuming the un-merged N×K union directly, as in FlashRAG's ``rerank``
    merge_method — would see ``top_k × N``; this pipeline does not do that:
    retrieval always pre-merges to ``top_k``.)

    Returns ``default`` when no ``semantic_retrieval`` node carries a
    resolved ``top_k``. The YAML's ``passage_reranker.top_k`` is **not**
    used here — it controls how many passages are KEPT after rerank, not
    how many enter.
    """
    retrieval_top_k = None
    for nl in pipeline_config.get("node_lines", []) or []:
        for node in nl.get("nodes", []) or []:
            if node.get("stage") == "semantic_retrieval":
                tk = node.get("top_k")
                if isinstance(tk, list):
                    tk = max(tk) if tk else None
                if tk is not None:
                    retrieval_top_k = int(tk)
    if retrieval_top_k is None:
        return default
    # Retrieval already fused the N sub-queries down to top_k, so the reranker
    # sees top_k inputs — NOT top_k × N (that would double-count the expansion
    # fanout already charged at retrieval/encode). See the docstring.
    return retrieval_top_k


def rrf_fusion_per_user_row_s(
    pipeline_config: dict, hardware, dag: object | None = None,
) -> float:
    """Mean observed fusion demand; fanout alone never supplies fusion work."""
    from rag_stack.cost_model.rrf import fusion_service_per_visit
    if isinstance(dag, dict) and dag.get("retrieval_fusion_calls"):
        pipeline_config = dict(pipeline_config)
        pipeline_config["_evaluator_fusion_calls"] = dag["retrieval_fusion_calls"]
        pipeline_config["_evaluator_fusion_calls_per_retrieval"] = (
            dag.get("fusion_invocations_per_query", 0.0)
            / max(1.0, dag.get("retrieval_invocations_per_query", 1.0))
        )
    if not pipeline_config.get("_evaluator_fusion_calls"):
        return 0.0
    return fusion_service_per_visit(pipeline_config, hardware)


def _get_stage_fanout(
    stage: str, pipeline_config: dict, dag: object | None = None,
) -> int:
    """Per-user work multiplier at the given stage.

    Sweep generation iterates ``batch_size`` in **call** units (one row =
    one back-end call's batch), but the system-level cap
    ``max_batch_size_request`` is in **user-request** units. The two
    coincide only when one user request maps to exactly one call at that
    stage. Dense retrieval uses fused worker visits R times expansion fanout
    K; downstream stages see the already-fused result of each visit.

      ===============================  ==================================
      PerformanceStage                 call-batch per user-request
      ===============================  ==================================
      semantic_retrieval_vectorsearch  R*K
      semantic_retrieval_encode        R*K
      passage_reranker                 T   (retrieval pre-merges the R*K
                                            sub-query lists back to top_k
                                            before the reranker; R*K is NOT
                                            double-counted here)
      generator_prefill / _decode      1
      query_expansion_prefill/_decode  1   (one LLM call emits all N
                                            sub-queries at once)
      passage_filter / compressor      1   (treated as single calls; the
                                            compressor's per-chunk fanout is
                                            priced inside its cost model, not
                                            at the sweep ceiling)
      ===============================  ==================================

    Returning this multiplier lets the sweep generator extend
    ``max_batch_size`` to ``max_batch_size_request × fanout`` so the
    user-unit Pareto (after ``RAGSweep._apply_fanout``) reaches the same
    upper bound the rest of the system enforces.
    """
    if stage in (Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH.value,
                 Stage.SEMANTIC_RETRIEVAL_ENCODE.value):
        # The performance sweep needs an integer ceiling so it contains the
        # largest possible backend batch.  Capacity/demand consumers use the
        # precise float mean from ``_get_retrieval_backend_fanout`` instead of
        # charging every request as the next whole invocation.
        return max(
            1, math.ceil(_get_retrieval_backend_fanout(pipeline_config, dag)),
        )
    if stage == "passage_reranker":
        return _get_reranker_input_fanout(pipeline_config, dag)
    if stage == "generator_prefill":
        # Batch-ceiling sizing uses the integer round-up of the (float)
        # prefill fanout so the swept Pareto reaches the user-unit ceiling;
        # the precise FLOAT fanout is applied separately in RAGSweep
        # (prefill_fanout=) by ``_apply_fanout``. Decode never fans out — its
        # M rounds are aggregated into ``dec_steps`` (Σ output tokens).
        return max(1, math.ceil(_get_prefill_fanout(pipeline_config, dag)))
    return 1


def _get_compressor_input_fanout(
    pipeline_config: dict, dag: object | None = None,
) -> int:
    """Per-user work multiplier at the passage_compressor stage.

    Only the per-chunk BERT compressor (``llmlingua2``) fans out: it processes
    each retrieved passage as an independent ≤512-token chunk, so its compute
    scales with the NUMBER of passages reaching it = effective top_k (the
    running-min of retrieval / reranker / filter top_k along the chain). The
    whole-context compressors — longllmlingua's ~7B causal LM and the
    tree_summarize / refine generators — prefill the concatenated context in
    ONE call per query → fanout 1. Mirrors
    ``llm_sim.genz_adapter._effective_context_top_k(before_node_type=...)``.
    """
    def _scalar(tk):
        return max(tk) if isinstance(tk, list) else int(tk)

    k: int | None = None
    for nl in pipeline_config.get("node_lines", []) or []:
        for node in nl.get("nodes", []) or []:
            nt = node.get("stage")
            if nt in ("semantic_retrieval", "hybrid_retrieval"):
                if node.get("top_k") is not None:
                    k = _scalar(node["top_k"])
            elif nt in ("passage_reranker", "passage_filter") and k is not None:
                if node.get("top_k") is not None:
                    k = min(k, _scalar(node["top_k"]))
            elif nt == "passage_compressor":
                mods = node.get("modules", []) or []
                is_per_chunk = bool(mods) and mods[0].get("component") == "llmlingua2"
                return k if (is_per_chunk and k) else 1
    return 1


def _validate_pow2(bs: int, where: str) -> int:
    bs = int(bs)
    assert bs >= 1 and (bs & (bs - 1)) == 0, (
        f"{where} must be a positive power of 2 (got {bs})"
    )
    return bs


def _short_unit_name(perf_stage_name: str, stage_value: str) -> str:
    """Node-scoped short key for a PerformanceStage (``generator_prefill``
    under the generator node → ``prefill``; single-unit bare names stay)."""
    prefix = f"{stage_value}_"
    if perf_stage_name.startswith(prefix):
        return perf_stage_name[len(prefix):]
    return perf_stage_name


def _resolve_max_batch_size_for_stage(
    perf_stage: str, pipeline_config: dict, default_max: int,
) -> int:
    """Resolve the per-PerformanceStage sweep cap.

    `node.system.max_batch_size` overrides the global cap. The override
    accepts two forms:
      - **scalar** `max_batch_size: 256` — applies to every PerformanceStage
        this node owns (convenient for single-unit nodes like passage_reranker).
      - **dict**   `max_batch_size: {prefill: 16, decode: 256}` — per-unit
        override with node-scoped SHORT keys (the node context supplies the
        owning stage). Missing units fall back to `default_max`.
    """
    owner = TAXONOMY.user_stage_of(perf_stage)
    short = _short_unit_name(perf_stage, owner.value)
    for nl in pipeline_config.get("node_lines", []):
        for node in nl.get("nodes", []):
            stage = node.get("stage")
            if stage != owner.value:
                continue
            sys_field = node.get("system", {})
            unknown = set(sys_field.keys()) - _ALLOWED_NODE_SYSTEM_KEYS
            assert not unknown, (
                f"node {stage}.system: unknown key(s) {sorted(unknown)}. "
                f"Only {sorted(_ALLOWED_NODE_SYSTEM_KEYS)} is allowed."
            )
            override = sys_field.get("max_batch_size")
            if override is None:
                continue

            if isinstance(override, dict):
                valid_shorts = {
                    _short_unit_name(ps.name, owner.value)
                    for comp in TAXONOMY.components_of(owner)
                    for ps in TAXONOMY.performance_stages_of_component(comp.name)
                }
                bad = set(override.keys()) - valid_shorts
                assert not bad, (
                    f"node {stage}.system.max_batch_size: unit key(s) "
                    f"{sorted(bad)} don't belong to {stage}. "
                    f"Valid: {sorted(valid_shorts)}"
                )
                if short in override:
                    return _validate_pow2(
                        override[short],
                        f"node {stage}.system.max_batch_size.{short}",
                    )
                # Unit not listed → fall through to default_max
            else:
                return _validate_pow2(
                    override, f"node {stage}.system.max_batch_size",
                )
    return default_max


def build_performance_stage_cost_models(
    pipeline_config: dict,
) -> dict[str, CostModelModule]:
    """Build ``{performance_stage_name: sub_cost_model_instance}`` — the cost
    model's ONE dispatch table.

    Consumes the rag_ir-normalized dispatch manifest
    (:func:`rag_stack.rag_ir.normalize.dispatch_manifest` — ``semantic_retrieval``
    arrives already split into its encode / vector-search stages; the cost model
    never expands). Each ``(stage, component)`` entry resolves its sub cost
    model through :func:`rag_stack.cost_model.dispatch.cost_model_for` and
    produces its PerformanceStages per the taxonomy naming law
    (:func:`rag_stack.cost_model.dispatch.performance_stages_for`) — an
    llm-shaped entry yields a prefill+decode pair sharing ONE sub-CM instance.
    """
    from rag_stack.cost_model import COST_MODEL_REGISTRY
    from rag_stack.cost_model.dispatch import cost_model_for, performance_stages_for
    from rag_stack.rag_ir.normalize import dispatch_manifest
    from rag_stack.cost_model.support import require_cm_pipeline_support

    require_cm_pipeline_support(pipeline_config)

    perf_stage_models: dict[str, CostModelModule] = {}

    for entry in dispatch_manifest(pipeline_config):
        cm_name = cost_model_for(entry.stage, entry.component, db_type=entry.db_type)
        cm_cls = COST_MODEL_REGISTRY.get(cm_name)
        if cm_cls is None:
            # Registered-name misses are a wiring error for priced components.
            raise KeyError(
                f"dispatch resolved ({entry.stage.value}, {entry.component}) to "
                f"unregistered cost model {cm_name!r}")
        instance = cm_cls()
        produced = performance_stages_for(entry.stage, entry.component)
        assert produced, (
            f"({entry.stage.value}, {entry.component}) is priced but produces "
            f"no PerformanceStage")
        for ps in produced:
            perf_stage_models[ps.name] = instance

    return perf_stage_models


# Transitional alias (scheduled for deletion with the registry package): the
# old name now returns PerformanceStage-keyed models — callers were migrated in
# the same change; the alias only keeps external notebooks/scripts importable.
