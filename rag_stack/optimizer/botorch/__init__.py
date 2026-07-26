"""BoTorch / Ax qLogNEHVI optimizers.

Three qLogNoisyExpectedHypervolumeImprovement (qLogNEHVI) optimizers built on
Ax + BoTorch over an Ax ``SearchSpace`` (hierarchy via
``ChoiceParameter.dependents``):

  - ``AxQNEHVI``          — Ax ``AxClient``, single-task GP per outcome
  - ``AxQNEHVIMixedGP``   — Ax with a mixed categorical+continuous kernel
  - ``AgentGuidedQNEHVI`` — LLM-agent-guided qLogNEHVI (the project's main optimizer)
  - ``SLOAgentGuidedQNEHVI`` — AgentGuidedQNEHVI + suggest-time SLO gating via
    a static cost-model probe (``system.performance_objectives`` constrained SLO)
  - ``GreedyThenHVAnchorAgentQNEHVI`` — greedy forward warm start followed by
    the E[ΔHV]-ranked anchor agent (v3; the v1/v2 wrapper is an internal base
    class in the same module since 07-07)
"""

from .ax_qnehvi import AxQNEHVI
from .ax_qnehvi_mixedgp import AxQNEHVIMixedGP
from rag_stack.optimizer.agent import AgentGuidedQNEHVI
from .ax_agent_qnehvi_slo import SLOAgentGuidedQNEHVI
from .greedy_then_hv_anchor_agent_qnehvi import GreedyThenHVAnchorAgentQNEHVI

__all__ = [
    "AxQNEHVI",
    "AxQNEHVIMixedGP",
    "AgentGuidedQNEHVI",
    "SLOAgentGuidedQNEHVI",
    "GreedyThenHVAnchorAgentQNEHVI",
]
