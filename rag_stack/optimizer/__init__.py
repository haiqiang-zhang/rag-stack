"""Optimizer registry.

Four baselines plus the project's main optimizer:
  - ``sobol``            — pure Sobol quasi-random baseline (no surrogate; hierarchy-aware)
  - ``ax_qnehvi``        — qLogNEHVI via Ax AxClient (single-task GP per outcome)
  - ``ax_qnehvi_mixedgp``— qLogNEHVI via Ax with a mixed (categorical+continuous) kernel
  - ``smac_mobo``        — SMAC3 ParEGO baseline (ConfigSpace; non-decoupled)
  - ``agent_qnehvi``     — LLM-agent-guided qLogNEHVI (the main optimizer)
  - ``agent_qnehvi_slo`` — agent_qnehvi + suggest-time SLO gating: candidates
    are pre-screened by the static cost model against the run's
    ``system.performance_objectives`` constrained SLO (static quality-only
    flow only; the controller injects the CM probe)
  - ``greedy_then_hv_anchor_agent_qnehvi`` — greedy warm start followed by
    HV-band Agent/DOE/DLS candidate generation and qLogNEHVI arbitration

All consume an Ax ``SearchSpace`` built by the single search-space builder
``RagSearchSpace.build_ax_searchspace`` (hierarchy expressed with Ax's native
``ChoiceParameter.dependents``). The BoTorch/Ax optimizers live in the
``botorch`` subpackage; ``sobol`` lives at the package root; ``smac_mobo``
lives in ``smac`` and additionally translates that SearchSpace to a
ConfigSpace via ``smac.ax_to_configspace``.
"""

from .sobol import Sobol
from .greedy import GreedyParetoOptimizer
from .botorch import (
    AxQNEHVI,
    AxQNEHVIMixedGP,
    AgentGuidedQNEHVI,
    GreedyThenHVAnchorAgentQNEHVI,
    SLOAgentGuidedQNEHVI,
)

# SMAC is an optional dependency and is fragile against sklearn version skew
# (its random-forest backend imports private symbols that newer sklearn drops).
# Importing it must not break the other optimizers, so defer the failure until
# ``smac_mobo`` is actually selected.
try:
    from .smac import SMACMOBOOptimizer
except Exception as _smac_import_error:  # noqa: BLE001 — surface any import failure lazily
    _SMAC_IMPORT_ERROR = _smac_import_error

    class SMACMOBOOptimizer:  # type: ignore[no-redef]
        """SMAC3 ParEGO baseline (unavailable — SMAC failed to import)."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The 'smac_mobo' optimizer is unavailable because SMAC failed to "
                f"import: {_SMAC_IMPORT_ERROR!r}. This is usually a sklearn version "
                "mismatch. Use a different optimizer, or fix the SMAC/sklearn install."
            )


OPTIMIZER_REGISTRY = {
    "sobol": Sobol,
    "greedy": GreedyParetoOptimizer,
    "ax_qnehvi": AxQNEHVI,
    "ax_qnehvi_mixedgp": AxQNEHVIMixedGP,
    "smac_mobo": SMACMOBOOptimizer,
    "agent_qnehvi": AgentGuidedQNEHVI,
    "agent_qnehvi_slo": SLOAgentGuidedQNEHVI,
    "greedy_then_hv_anchor_agent_qnehvi": GreedyThenHVAnchorAgentQNEHVI,
}
