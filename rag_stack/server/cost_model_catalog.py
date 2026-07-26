"""Read-only API projection of RAG-IR names and production CM bindings.

This is a catalog, not a second compiler. Exact workers, engine instances,
devices and pricing profiles are described only by a bound CompiledCMPlan.
"""

from __future__ import annotations

from rag_stack.cost_model.compiled import SCHEMA_VERSION as PLAN_SCHEMA_VERSION
from rag_stack.cost_model.dispatch import dispatch_binding_table, performance_stages_for
from rag_stack.pipeline_contract import retrieval_component_capabilities
from rag_stack.taxonomy import TAXONOMY, Stage


CATALOG_SCHEMA_VERSION = "rag-stack.cost-model-catalog.v1"


def mapped_components(model_name: str) -> list[str]:
    """Actual component names handled by a registered cost model.

    Index families are selectors on ``vectordb``, not component names. The
    embedding binding is included here even though the legacy flat map could
    not represent it.
    """
    return sorted({binding.component for binding in dispatch_binding_table()
                   if binding.cost_model == model_name})


def build_cost_model_catalog() -> dict:
    bindings_by_component: dict[str, list] = {}
    for binding in dispatch_binding_table():
        bindings_by_component.setdefault(binding.component, []).append(binding)

    stages = []
    for stage in TAXONOMY.stages:
        components = TAXONOMY.components_of(stage)
        if not components:
            continue
        rows = []
        for component in components:
            bindings = bindings_by_component.get(component.name, [])
            if not bindings:
                capability = retrieval_component_capabilities(component.name)
                # Some names are not executable even in measured mode. Keep
                # that distinction from the shared pipeline contract.
                reason = (capability["reason"]
                          if capability and not capability["cost_model_supported"] else None)
                rows.append({
                    "component": component.name,
                    "dispatch_stage": stage.value,
                    "db_type": None,
                    "cost_model": None,
                    "status": "unsupported" if reason else "unpriced",
                    "reason": reason or (
                        "No standalone CM pricing binding is declared for this "
                        "component. This does not establish pipeline readiness or zero runtime cost."
                    ),
                    "execution_kind": component.shape,
                    "performance_stages": [],
                    "resources": [],
                })
                continue
            for binding in bindings:
                phases = performance_stages_for(binding.stage, component.name)
                rows.append({
                    "component": component.name,
                    "dispatch_stage": binding.stage.value,
                    "db_type": binding.db_type,
                    "cost_model": binding.cost_model,
                    "status": "priced",
                    "reason": (
                        "Only observed local rrf_pure after its dense encode/search "
                        "pair is bound here. Hybrid joins require a separate physical policy."
                        if binding.stage is Stage.RETRIEVAL_FUSION else None
                    ),
                    "execution_kind": component.shape,
                    "performance_stages": [
                        {"name": phase.name, "kind": phase.kind, "resource": phase.resource}
                        for phase in phases
                    ],
                    "resources": sorted({phase.resource for phase in phases}),
                })
        stages.append({"stage": stage.value,
                       "user_selectable": stage in TAXONOMY.user_stages,
                       "bindings": rows})
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "read_only": True,
        "scope": "dispatch_catalog",
        "stages": stages,
        "contract": {
            "rag_ir": "Evaluator observations provide canonical calls, I/O and dependencies; dense retrieval has separate encode and vector-search calls.",
            "compiler": "CM compile_workload validates observed paths; bind_plan lowers them through the production worker/engine compiler and binds deployment and pricing.",
            "compiled_plan_schema": PLAN_SCHEMA_VERSION,
            "readiness": "A registered binding or an available profile is not a readiness verdict. Use project CM Preview for a concrete pipeline, query cohort, deployment and calibration.",
        },
    }
