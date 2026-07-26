"""Stage normalization — the ONE place the declared stage expansion is applied.

The taxonomy DECLARES that ``semantic_retrieval`` is really two stages
(``semantic_retrieval_encode`` + ``semantic_retrieval_vectorsearch``,
:data:`rag_stack.taxonomy.STAGE_EXPANSION`); this module — the algorithm
layer — APPLIES it. The cost model never expands: it consumes the manifest
built here and hard-errors if it ever sees the un-split user stage
(:func:`rag_stack.cost_model.dispatch.cost_model_for`).

:func:`dispatch_manifest` turns a resolved pipeline_config into the ordered
list of :class:`DispatchEntry` — one entry per (stage, component) the cost
model must price. Downstream, ``cost_model.dispatch`` binds each entry to a
sub cost model, and each sub cost model produces its PerformanceStages.
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

from rag_stack.taxonomy import TAXONOMY, Stage, expand_stage


def with_observed_trace_components(pipeline_config: dict, traces: list) -> dict:
    """Bind component demand from validated, ordered evaluator calls.

    These private fields are transient derived inputs for model selection and
    coarse candidate tables. They never rewrite the producer's trace or infer
    an algorithm from query-expansion settings.
    """
    out = dict(pipeline_config)
    calls = [call for query in traces for call in query]
    out["_evaluator_trace_stages"] = sorted({call["stage"] for call in calls})
    fusion = [call for call in calls if call["stage"] == "retrieval_fusion"]
    out["_evaluator_fusion_calls"] = deepcopy(fusion)
    retrievals = sum(call["stage"] == "semantic_retrieval_vectorsearch" for call in calls)
    out["_evaluator_fusion_calls_per_retrieval"] = len(fusion) / retrievals if retrievals else 0.0
    return out


@dataclass(frozen=True)
class DispatchEntry:
    """One normalized (stage, component) unit of the cost-model manifest.

    ``node`` is the resolved pipeline node dict the component came from
    (module params, top_k, …); ``db_type`` is resolved from the node's
    vectordb reference and is only set for the vector-search split stage.
    """

    stage: Stage
    component: str
    node: dict
    db_type: str | None = None


def _resolve_db_type(pipeline_config: dict, module: dict) -> str | None:
    """The db_type of the vectordb instance a retrieval module points at."""
    vectordb_name = module.get("vectordb", "")
    for vdb in pipeline_config.get("vectordb", []) or []:
        if vdb.get("name") == vectordb_name:
            return vdb.get("db_type")
    return None


def _validate_fusion_downstream_shapes(nodes: list[dict], calls: list[dict]) -> None:
    """Reject observed shapes the current downstream CM adapters cannot price."""
    retrievals = [(index, node) for index, node in enumerate(nodes)
                  if node.get("stage") == "semantic_retrieval"]
    if len(retrievals) != 1:
        return  # Component binding separately rejects unsupported joins.
    index, retrieval = retrievals[0]
    downstream = {
        node.get("stage") for node in nodes[index + 1:]
        if node.get("stage") in {"passage_reranker", "passage_compressor"}
    }
    if not downstream:
        return
    top_k = retrieval.get("top_k")
    if isinstance(top_k, list):
        top_k = max(top_k) if top_k else None
    top_k = int(top_k) if top_k is not None else 16
    observed = {
        call["output_profile"]["items"] for call in calls
        if isinstance(call.get("output_profile"), dict)
        and "items" in call["output_profile"]
    }
    if any(count != top_k for count in observed):
        raise ValueError(
            "Observed retrieval_fusion output counts "
            f"{sorted(observed)} differ from semantic_retrieval.top_k={top_k}; "
            "the current fixed-passage-count cost-model adapter for downstream "
            f"{sorted(downstream)} does not support these observed shapes"
        )


def dispatch_manifest(pipeline_config: dict) -> list[DispatchEntry]:
    """Build the cost-model dispatch manifest from a RESOLVED pipeline config.

    Applies :func:`rag_stack.taxonomy.expand_stage` — ``semantic_retrieval``
    nodes yield one entry per split stage (sharing the node's vectordb
    config). Every other declared stage passes through identically. Fusion
    bindings must come from evaluator observations; query-expansion fanout
    never implicitly registers a fusion operation. Components the
    taxonomy marks as not CM-priced (``shape is None`` — prompt makers,
    ``time_reranker``) are skipped: they have no PerformanceStage.

    Unknown stages or components raise (taxonomy hard-KeyError semantics);
    nodes without a resolved module are skipped like the old dispatch did.
    """
    manifest: list[DispatchEntry] = []
    nodes = [
        node
        for line in pipeline_config.get("node_lines", []) or []
        for node in line.get("nodes", []) or []
        if node.get("modules")
    ]
    for node in nodes:
        module = node["modules"][0]
        stage = Stage(node.get("stage", ""))
        component_name = str(module.get("component", "") or "")
        component = TAXONOMY.components[component_name]
        if component.stage is not stage:
            raise KeyError(
                f"pipeline node declares stage {stage.value!r} but component "
                f"{component_name!r} belongs to {component.stage.value!r}")
        if component.shape is None:
            continue  # not CM-priced (no PerformanceStage)
        db_type = (_resolve_db_type(pipeline_config, module)
                   if component_name == "vectordb" else None)
        for ds in expand_stage(stage):
            manifest.append(DispatchEntry(
                stage=ds,
                component=component_name,
                node=node,
                db_type=(db_type if ds is Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH
                         else None),
            ))
    fusion_calls = pipeline_config.get("_evaluator_fusion_calls") or []
    if fusion_calls:
        implementations = {call.get("model_id") for call in fusion_calls}
        if implementations != {"rrf_pure"}:
            raise ValueError(
                "No retrieval_fusion cost-model binding for evaluator implementations "
                f"{sorted(map(str, implementations))}; supported: rrf_pure"
            )
        _validate_fusion_downstream_shapes(nodes, fusion_calls)
        if not any(entry.stage is Stage.RETRIEVAL_FUSION for entry in manifest):
            searches = [i for i, entry in enumerate(manifest)
                        if entry.stage is Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH]
            if len(searches) != 1:
                raise ValueError(
                    "Observed local rrf_pure requires exactly one resolved dense "
                    "retrieval component; general retrieval joins need a model adapter"
                )
            manifest.insert(searches[0] + 1, DispatchEntry(
                stage=Stage.RETRIEVAL_FUSION,
                component="rrf",
                node={"stage": "retrieval_fusion", "modules": [{"component": "rrf"}]},
            ))
    return manifest
