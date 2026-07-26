"""Shared FastAPI dependencies for the Cost Model API."""

from __future__ import annotations

from fastapi import HTTPException, Path

from rag_stack.cost_model import COST_MODEL_REGISTRY
from rag_stack.cost_model.base import CostModelModule
from rag_stack.types import SystemConfig

# Singleton instances — cost models are stateless, no need to recreate.
_instances: dict[str, CostModelModule] = {}


def get_cost_model(
    model_name: str = Path(..., description="Registered cost model name"),
) -> CostModelModule:
    """Resolve a cost model by registry name. Returns a singleton instance."""
    if model_name not in COST_MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown cost model: {model_name!r}. "
            f"Available: {list(COST_MODEL_REGISTRY.keys())}",
        )
    if model_name not in _instances:
        _instances[model_name] = COST_MODEL_REGISTRY[model_name]()
    return _instances[model_name]


def parse_hardware(hardware_dict: dict) -> SystemConfig:
    """Build a SystemConfig from a raw dict."""
    return SystemConfig.from_dict(hardware_dict)
