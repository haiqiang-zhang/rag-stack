"""LLM Sim cost model extension endpoints.

All calibration endpoints are now generic at
``/api/cost-model/{model_name}/(profiles|run|jobs|modules)`` so every
LLM-sim-family cost model (passage_reranker_sim, query_expansion_sim, llm_sim, …) picks
them up uniformly. This file just registers an empty extension router
so the cost model shows up in the registry; no per-model endpoints.
"""

from __future__ import annotations

from fastapi import APIRouter

from . import register_router

router = APIRouter(tags=["llm-sim"])


register_router("llm_sim", router)
