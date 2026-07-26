"""Extension router registry for cost model-specific endpoints.

Each cost model that needs custom endpoints creates a router in this package
and calls ``register_router()`` at import time. The app factory in
``cost_model_api.py`` iterates ``EXTENSION_ROUTERS`` to mount them.

Capabilities are declared on the CostModelModule class via
``cls.capabilities()`` — not at router registration time.
"""

from __future__ import annotations

from fastapi import APIRouter

EXTENSION_ROUTERS: dict[str, APIRouter] = {}


def register_router(cost_model_name: str, router: APIRouter) -> None:
    """Register an extension router for a specific cost model."""
    EXTENSION_ROUTERS[cost_model_name] = router


# Import router modules to trigger self-registration.
from . import faiss_ivf, llm_sim  # noqa: F401, E402
