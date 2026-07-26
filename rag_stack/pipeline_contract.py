"""Lightweight configuration contracts shared by planning and optimization.

This module intentionally imports no evaluator, optimizer, or cost model. A
pipeline's executable structure and its available performance source are
separate checks, so a valid measured workflow remains inspectable without an
analytical model for every operation.
"""

from __future__ import annotations

from copy import deepcopy


STAGE_ALIASES = {
    "dense_retrieval": "semantic_retrieval",
    "sparse_retrieval": "lexical_retrieval",
}
COMPONENT_ALIASES = {
    "VectorDB": "vectordb", "BM25": "bm25",
    "HybridRRF": "hybrid_rrf", "HybridCC": "hybrid_cc",
}
HYBRID_STAGES = ("semantic_retrieval", "lexical_retrieval", "hybrid_retrieval")
BM25_CM_REASON = (
    "BM25 analytical cost modeling is unavailable; hybrid retrieval requires "
    "system.performance_source=measured. Missing lexical cost is not zero cost."
)
HYBRID_FUSION_CM_REASON = (
    "Hybrid fusion has no validated analytical scheduling/cost binding. The "
    "existing RRF model covers local dense-retrieval fusion, not a join of "
    "dense and lexical services; use system.performance_source=measured."
)


def canonical_retrieval_stage(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def normalize_retrieval_aliases(config: dict) -> dict:
    """Return a detached config with canonical retrieval stage/component names.

    Accept full configs, pipeline blocks, and resolved evaluator configs. No
    new input wiring is inferred here; node_lines remain the execution order.
    """
    result = deepcopy(config)
    pipeline = (result.get("algo_search_space") or {}).get("pipeline", result)
    if not isinstance(pipeline, dict):
        return result
    blocks = pipeline.get("methods", [pipeline])
    if not isinstance(blocks, list):
        return result
    for block in blocks:
        if not isinstance(block, dict):
            continue
        for line in block.get("node_lines", []) or []:
            if not isinstance(line, dict):
                continue
            for node in line.get("nodes", []) or []:
                if not isinstance(node, dict):
                    continue
                if "modules" not in node and isinstance(node.get("module"), dict):
                    node["modules"] = [node.pop("module")]
                stage = node.get("stage")
                if isinstance(stage, str):
                    node["stage"] = canonical_retrieval_stage(stage)
                for module in node.get("modules", []) or []:
                    if isinstance(module, dict):
                        name = module.get("component")
                        if isinstance(name, str):
                            module["component"] = COMPONENT_ALIASES.get(name, name)
    return result


def pipeline_nodes(block: dict) -> list[dict]:
    return [node for line in block.get("node_lines", []) or []
            if isinstance(line, dict)
            for node in line.get("nodes", []) or [] if isinstance(node, dict)]


def has_hybrid_retrieval(block: dict) -> bool:
    return any(
        canonical_retrieval_stage(str(node.get("stage", "")))
        in {"lexical_retrieval", "hybrid_retrieval"}
        or any(COMPONENT_ALIASES.get(str(m.get("component", "")), m.get("component"))
               in {"bm25", "hybrid_rrf", "hybrid_cc"}
               for m in (node.get("modules") or ([node["module"]] if isinstance(node.get("module"), dict) else []))
               if isinstance(m, dict))
        for node in pipeline_nodes(block)
    )


def _values(value) -> list:
    if isinstance(value, dict):
        value = value.get("ordered", value.get("range", value))
    return value if isinstance(value, list) else [value]


def _positive_integers(value) -> bool:
    values = _values(value)
    return bool(values) and all(type(v) is int and v > 0 for v in values)


def hybrid_structure_errors(block: dict, *, backend: str = "static_gt") -> list[str]:
    """Validate the supported one-dense, one-BM25, one-RRF workflow.

    Independent of performance_source: the same structural contract is used
    by configuration validation, the optimizer, and the workflow preview.
    """
    block = normalize_retrieval_aliases(block)
    if not has_hybrid_retrieval(block):
        return []
    errors: list[str] = []
    if backend != "static_gt":
        errors.append("Hybrid retrieval currently requires the static_gt evaluator.")
    dataflows = _values(block.get("rag_dataflow", block.get("mode", "sequential")))
    branches = [next(iter(v), "") if isinstance(v, dict) else v for v in dataflows]
    if block.get("mode", "sequential") != "sequential" or branches != ["sequential"]:
        errors.append("Hybrid retrieval supports sequential execution only; ReAct/iterative hybrid is unsupported.")
    nodes = pipeline_nodes(block)
    by_stage = {stage: [n for n in nodes if n.get("stage") == stage]
                for stage in HYBRID_STAGES}
    for stage, selected in by_stage.items():
        if len(selected) != 1:
            errors.append(f"Hybrid retrieval requires exactly one {stage} node; found {len(selected)}.")
    if all(len(selected) == 1 for selected in by_stage.values()):
        positions = [nodes.index(by_stage[stage][0]) for stage in HYBRID_STAGES]
        if positions != sorted(positions):
            errors.append("Hybrid retrieval node order must be semantic_retrieval, lexical_retrieval, hybrid_retrieval.")
        fusion_position = positions[-1]
        if any(n.get("stage") in {"passage_reranker", "passage_filter", "passage_compressor", "prompt_maker", "generator"}
               for n in nodes[:fusion_position]):
            errors.append("Hybrid retrieval must finish fusion before passage processing, prompting, or generation.")
    expected = dict(zip(HYBRID_STAGES, ("vectordb", "bm25", "hybrid_rrf")))
    for stage, selected in by_stage.items():
        for node in selected:
            if node.get("optional") or node.get("enabled") is False:
                errors.append(f"Hybrid input dependency requires {stage} to be enabled and non-optional.")
            if not _positive_integers(node.get("top_k")):
                errors.append(f"{stage}.top_k must contain positive integers.")
            modules = node.get("modules", []) or []
            if len(modules) != 1 or not isinstance(modules[0], dict) or modules[0].get("component") != expected[stage]:
                errors.append(f"{stage} requires one {expected[stage]} module in the supported hybrid workflow; hybrid_cc is not supported.")
                continue
            if stage == "hybrid_retrieval":
                module = modules[0]
                if not _positive_integers(module.get("weight", 60)):
                    errors.append("hybrid_rrf.weight must contain positive integers (RRF rank constant).")
                if "rrf_k" in module:
                    if not all(v == -1 or type(v) is int and v > 0 for v in _values(module["rrf_k"])):
                        errors.append("hybrid_rrf.rrf_k must be -1 or a positive integer; prefer weight.")
                    if "weight" in module and any(v != -1 for v in _values(module["rrf_k"])):
                        errors.append("Set hybrid_rrf.weight or its deprecated rrf_k alias, not both.")
    return errors


def hybrid_capability_errors(block: dict, *, performance_source: str) -> list[str]:
    """Reject unavailable performance modeling without changing the workflow."""
    if has_hybrid_retrieval(block) and performance_source == "cost_model":
        errors = [BM25_CM_REASON]
        if any(canonical_retrieval_stage(str(node.get("stage", ""))) == "hybrid_retrieval"
               for node in pipeline_nodes(block)):
            errors.append(HYBRID_FUSION_CM_REASON)
        return errors
    return []


def retrieval_component_capabilities(component: str) -> dict | None:
    """Known retrieval capability metadata; None leaves other providers alone."""
    common = {"supported_dataflows": ["sequential"]}
    if component == "bm25":
        return {**common, "execution_supported": True, "measured_supported": True,
                "cost_model_supported": False, "reason": BM25_CM_REASON,
                "operations": ["lexical_retrieval"]}
    if component == "hybrid_rrf":
        return {**common, "execution_supported": True, "measured_supported": True,
                "cost_model_supported": False, "reason": HYBRID_FUSION_CM_REASON,
                "operations": ["retrieval_fusion"]}
    if component == "hybrid_cc":
        return {"supported_dataflows": [], "execution_supported": False,
                "measured_supported": False, "cost_model_supported": False,
                "reason": "Convex-combination fusion is not supported by the current hybrid workflow contract.",
                "operations": []}
    return None


RETRIEVAL_COMPONENT_PARAMETERS = {
    "vectordb": {
        "vectordb": {"type": "string", "label": "Vector index", "default": "",
                     "description": "Reference an existing algo_search_space.vectordb index, or retain a family selector mapping."},
    },
    "bm25": {
        "bm25_tokenizer": {"type": "string", "label": "BM25 tokenizer", "default": "porter_stemmer"},
    },
    "hybrid_rrf": {
        "weight": {"type": "integer", "label": "RRF rank constant", "default": 60,
                   "description": "Positive rank smoothing constant; this is not a dense/lexical mixture weight."},
    },
}
