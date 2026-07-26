"""System-level search-space construction for RAG optimization.

Who owns the system/deployment design space is ``system.system_space_owner``
(:func:`rag_stack.performance_context.resolve_system_space_owner`):

* ``cost_model``: RAGO sweeps hardware deployment internally from
  ``system.cm_search_space``; no optimizer-facing deployment parameters are
  emitted (the decoupled search).
* ``optimizer``: the optimizer searches the ``system.system_design_space``
  block — orthogonal runtime knobs plus EITHER a ``deployment_preset``
  categorical (named practitioner templates) OR the auto-synthesized
  ``gpu_layout`` categorical that expands to a budget-valid deployment at
  resolve time. Exactly one deployment is then priced (CM fixed-placement)
  or launched (measured) per trial, per ``system.performance_source``.
"""

from __future__ import annotations

from typing import Callable, List

from ax.core.parameter import Parameter

from rag_stack.performance_context import (
    SYSTEM_SPACE_OWNER_OPTIMIZER,
    resolve_system_space_owner,
    system_design_space,
)
from rag_stack.search_space.ax_search_space_ops import SEP


MakeParam = Callable[[str, object], Parameter | None]


def extract_system_params(config: dict, make_param: MakeParam) -> List[Parameter]:
    """Dispatch system search-space construction by system-space owner."""
    if resolve_system_space_owner(config) == SYSTEM_SPACE_OWNER_OPTIMIZER:
        return extract_system_design_params(config, make_param)
    return extract_cost_model_system_params(config, make_param)


def extract_cost_model_system_params(
    config: dict, make_param: MakeParam,
) -> List[Parameter]:
    """Optimizer-facing system params when the COST MODEL owns the space.

    The cost model consumes ``system.cm_search_space`` directly and performs
    its own RAGO deployment sweep, so the optimizer only sees pipeline/corpus/
    vector DB parameters — plus the ``system.retrieval`` runtime knobs, which
    are optimizer-facing under BOTH owners (they parameterize the retrieval
    stage's execution, not the hardware deployment).
    """
    system = config.get("system") or {}
    return _extract_retrieval_params(system, make_param)


def extract_system_design_params(
    config: dict, make_param: MakeParam,
) -> List[Parameter]:
    """Optimizer-facing system params when the OPTIMIZER owns the space.

    Emits every multi-valued ``system.system_design_space`` knob (including a
    ``deployment_preset`` name list, which becomes an unordered categorical)
    except the derived/raw-layout keys. The ``gpu_layout`` categorical is
    auto-synthesized ONLY when no ``deployment_preset`` key is declared — the
    two are mutually exclusive ways to drive the layout (ConfigValidator
    enforces the exclusivity; the suppression here keeps the search space
    consistent with it).
    """
    from rag_stack.search_space.placement import (
        passage_reranker_components,
        possible_measured_gpu_stage_lists,
        valid_layout_options,
    )

    system = config.get("system") or {}
    params: List[Parameter] = _extract_retrieval_params(system, make_param)
    dss = system_design_space(system)
    prefix = f"system{SEP}"
    for key, value in dss.items():
        if (
            key in _DERIVED_KNOBS
            or key == "available_gpus"
            or key.startswith("cut_after_")
            or key.startswith("num_gpus_")
            or key.startswith("pipeline_parallel_")
        ):
            continue
        param = make_param(f"{prefix}{key}", value)
        if param is not None:
            params.append(param)

    if "deployment_preset" in dss:
        # Presets are complete deployments; the gpu_layout enumeration below
        # would be a second, conflicting driver of the same layout.
        return params

    gpus = dss.get("available_gpus") or []
    if not (isinstance(gpus, list) and len(gpus) >= 2):
        return params

    ss = system.get("cm_search_space") or {}
    cap = ss.get("max_num_gpus_per_stage", 1)
    if isinstance(cap, list):
        cap = max(cap)
    min_slots = int(ss.get("min_num_gpus", 1) or 1)
    max_slots = int(ss.get("max_num_gpus", len(gpus)) or len(gpus))

    layout_count = 0
    for stages in possible_measured_gpu_stage_lists(config):
        layout_count = max(
            layout_count,
            len(
                valid_layout_options(
                    stages,
                    gpus,
                    min_slots=min_slots,
                    max_slots=max_slots,
                    max_num_gpus_per_stage=int(cap),
                    reranker_components=passage_reranker_components(config),
                )
            ),
        )

    if layout_count > 1:
        param = make_param(
            f"{prefix}gpu_layout",
            [f"layout_{i:04d}" for i in range(layout_count)],
        )
        if param is not None:
            params.append(param)
    return params


def _extract_retrieval_params(
    system: dict, make_param: MakeParam,
) -> List[Parameter]:
    """Optimizer-facing dims from the ``system.retrieval`` block (both owners).

    Only ``faiss_num_threads`` (OMP threads for the FAISS search) is
    optimizer-facing here. ``faiss_ivf_parallel_mode`` is SYSTEM design space and
    is deliberately NOT an optimizer dim: the cost model manages it as a
    default-ON deployment axis inside its own sweep (per-batch argmin over
    {0,1} in ``sweep_inputs._retrieval_parallel_modes``; pin with a scalar
    ``system.retrieval.faiss_ivf_parallel_mode``). Neither knob ever enters the
    index-build signature (search-time only — the same on-disk index serves
    every value).
    """
    retrieval = system.get("retrieval") or {}
    params: List[Parameter] = []
    if isinstance(retrieval, dict):
        allowed = {"faiss_num_threads"}
        for key, value in retrieval.items():
            if key not in allowed:
                continue
            param = make_param(f"system{SEP}{key}", value)
            if param is not None:
                params.append(param)
    return params


_DERIVED_KNOBS = frozenset({
    "vllm_gpu_memory_utilization",
    "vllm_max_num_batched_tokens",
    "tensor_parallel_vllm",
    "pipeline_parallel_vllm",
    "serving_mode",
})
