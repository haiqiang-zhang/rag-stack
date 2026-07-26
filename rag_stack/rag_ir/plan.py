"""Structural projection of host node-lines, independent of performance models.

The plan describes configured nodes and named data dependencies. Execution is
still serial. Internal operations omitted by a provider are learned only from
its actual trace; this module never infers an RRF call from query fanout.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy

from rag_stack.pipeline_contract import canonical_retrieval_stage
from rag_stack.taxonomy import TAXONOMY, SHAPE_LLM, expand_stage


def _pipeline_block(config: dict) -> dict:
    if "algo_search_space" not in config:
        return config
    pipeline = (config.get("algo_search_space") or {}).get("pipeline") or {}
    if "methods" in pipeline:
        methods = pipeline["methods"]
        if len(methods) != 1:
            raise ValueError("Select one pipeline method before describing its workflow")
        return methods[0]
    return pipeline


def structural_nodes(config: dict) -> list[dict]:
    """Return logical nodes, with stable identities within this configured plan."""
    block = _pipeline_block(config)
    source = [node for line in block.get("node_lines", []) or []
              for node in line.get("nodes", []) or []]
    counts = Counter(canonical_retrieval_stage(node.get("stage", "")) for node in source)
    ordinals = Counter()
    result = []
    query_node = dense_node = lexical_node = candidates = prompt_node = None
    for raw in source:
        stage = canonical_retrieval_stage(raw.get("stage", ""))
        if not stage:
            raise ValueError("Workflow nodes require a stage")
        ordinal = ordinals[stage]
        ordinals[stage] += 1
        node_id = stage if counts[stage] == 1 else f"{stage}:{ordinal}"
        modules = raw.get("modules") or ([raw["module"]] if raw.get("module") else [])
        components = list(dict.fromkeys(str(m["component"]) for m in modules if m.get("component")))
        inputs = {}
        operations = [stage]
        if stage in {"semantic_retrieval", "lexical_retrieval"}:
            if query_node:
                inputs["queries"] = [query_node]
            if stage == "semantic_retrieval":
                operations = [s.value for s in expand_stage(stage)]
                dense_node = node_id
            else:
                lexical_node = node_id
            candidates = node_id
        elif stage == "hybrid_retrieval":
            if dense_node is None or lexical_node is None:
                raise ValueError("Hybrid retrieval requires preceding dense and lexical producers")
            inputs = {"dense": [dense_node], "lexical": [lexical_node]}
            operations = ["retrieval_fusion"]
            candidates = node_id
        elif stage == "query_expansion":
            query_node = node_id
        elif stage in {"passage_reranker", "passage_compressor", "passage_filter", "passage_augmenter"}:
            if candidates:
                inputs["candidates"] = [candidates]
            candidates = node_id
        elif stage == "prompt_maker":
            if candidates:
                inputs["candidates"] = [candidates]
            prompt_node = node_id
            operations = []
        elif stage == "generator":
            if prompt_node:
                inputs["prompt"] = [prompt_node]
            elif candidates:
                inputs["candidates"] = [candidates]
        result.append({"node_id": node_id, "stage": stage, "components": components,
                       "inputs": inputs, "operations": operations})
    return result


def structural_stages(nodes: list[dict], observed: list[dict] = ()) -> list[str]:
    """Project the legacy stage summary using vocabulary, never CM availability."""
    result = []
    for node in nodes:
        operations = list(node["operations"])
        for call in observed:
            if call.get("node_id") == node["node_id"] and call["stage"] not in operations:
                operations.append(call["stage"])
        node["operations"] = operations
        components = [TAXONOMY.components.get(name) for name in node["components"]]
        if (components and all(component and component.shape is None for component in components)
                and node["stage"] not in {"lexical_retrieval", "hybrid_retrieval", "retrieval_fusion"}):
            continue
        for operation in operations:
            if operation in {"generator", "query_expansion", "passage_compressor", "passage_reranker"}:
                if any(component and component.shape == SHAPE_LLM for component in components):
                    names = [f"{operation}_prefill", f"{operation}_decode"]
                elif operation in {"generator", "query_expansion"}:
                    names = [f"{operation}_prefill", f"{operation}_decode"]
                else:
                    names = [operation]
            else:
                names = [operation]
            for name in names:
                if name not in result:
                    result.append(name)
    # Old dense traces carry no logical node IDs. Preserve their actually
    # observed local fusion in the legacy summary without inventing calls.
    if any(call["stage"] == "retrieval_fusion" for call in observed) and "retrieval_fusion" not in result:
        if "semantic_retrieval_vectorsearch" in result:
            result.insert(result.index("semantic_retrieval_vectorsearch") + 1, "retrieval_fusion")
        else:
            result.append("retrieval_fusion")
    return result


def describe_pipeline(config: dict) -> dict:
    """Describe one configured pipeline for API/editor use, without execution."""
    from rag_stack.rag_ir.workflow_schema import WorkflowSchema

    nodes = structural_nodes(config)
    stages = structural_stages(nodes)
    block = _pipeline_block(config)
    runtime = config.get("pipeline_runtime") or {}
    mode = block.get("mode", runtime.get("mode", "sequential"))
    flow = block.get("rag_dataflow", runtime.get("rag_dataflow", mode))
    if isinstance(flow, list):
        if len(flow) != 1:
            raise ValueError("Select one dataflow before describing its workflow")
        flow = flow[0]
    if isinstance(flow, dict):
        if len(flow) != 1:
            raise ValueError("Select one dataflow before describing its workflow")
        flow = next(iter(flow))
    result = WorkflowSchema(
        dataflow_type="agentic" if flow not in {"sequential", "naive", "vanilla"} else "sequential",
        stages=stages, has_query_expansion="query_expansion_prefill" in stages,
        has_reranker="passage_reranker" in stages,
        has_encode="semantic_retrieval_encode" in stages,
        has_compressor=any(stage.startswith("passage_compressor") for stage in stages),
        nodes=deepcopy(nodes), execution_order=[node["node_id"] for node in nodes],
    ).to_dict()
    # Preview has a fixed response shape even before the first node is added.
    result.update(nodes=deepcopy(nodes), execution_order=[node["node_id"] for node in nodes],
                  execution_policy="sequential")
    return result
