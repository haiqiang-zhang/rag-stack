"""The LLM-agent-guided qLogNEHVI optimizer package.

Modules:
  util        — small numeric/tensor helpers
  llm_agent   — AgentAdvice / LLMAgent (prompts)
  init_probes — sobol-first structured init mixin
  dls         — DLS operator mixin (tests monkeypatch THIS module's
                optimize_acqf_discrete_local_search)
  forced      — realized-HV stagnation and forced-delivery state
  prompts     — optimizer-side prompt feeds + advise orchestration mixin
  core        — AgentGuidedQNEHVI (the composed optimizer class)
"""

from .core import AgentGuidedQNEHVI
from .llm_agent import AgentAdvice, AgentAdviceError, LLMAgent

__all__ = [
    "AgentGuidedQNEHVI",
    "AgentAdvice",
    "AgentAdviceError",
    "LLMAgent",
]
