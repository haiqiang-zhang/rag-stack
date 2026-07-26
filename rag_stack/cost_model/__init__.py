"""Cost model module registry.

Each cost model module registers itself via ``@register_cost_model(name)``.
The controller looks up modules by name to dispatch estimation.

All Python modules under this package (and sub-packages) are auto-imported
at startup so that ``@register_cost_model`` decorators execute and populate
``COST_MODEL_REGISTRY``.  To add a new cost model, just create a new file
with a decorated class — no manual import list to maintain.
"""

import importlib
import pkgutil
from pathlib import Path

from rag_stack.cost_model.types import (  # noqa: F401
    OperatorBreakdown,
    CostModelResult,
    SweepPoint,
    AssemblyResult,
)
from rag_stack.cost_model.base import (  # noqa: F401
    CostModelModule,
)
from rag_stack.cost_model.algorithm import (  # noqa: F401
    AlgorithmModel,
    AlgorithmWorkload,
    DataProfile,
    PhaseWork,
    OperatorWorkProfile,
    register_algorithm_model,
)

# RAGCMAssembly owns candidate preparation and final selection. Its trace path
# uses workload.py for normalization/summary and dynamic_pipeline.py for the
# explicit analytical -> shortlist -> simulation phases. The static path keeps
# config-driven RAGO estimates. Public assembly/trace helpers are exported below
# after component registration completes.

COST_MODEL_REGISTRY: dict[str, type[CostModelModule]] = {}

# Import failures used to be discarded by ``_auto_discover``.  That turns a
# dependency/provenance error (for example, importing an older GenZ checkout)
# into a much later and misleading ``unregistered cost model`` error for the
# first component encountered by a sweep.  Keep the original failures so the
# required-registry check below can fail at CM startup with the real cause.
COST_MODEL_DISCOVERY_ERRORS: dict[str, Exception] = {}


def register_cost_model(name: str):
    """Decorator to register a cost model class."""
    def decorator(cls: type[CostModelModule]) -> type[CostModelModule]:
        COST_MODEL_REGISTRY[name] = cls
        return cls
    return decorator


# Auto-import all modules in this package to trigger @register_cost_model
def _auto_discover():
    pkg_dir = Path(__file__).parent
    for info in pkgutil.walk_packages([str(pkg_dir)], prefix=__name__ + "."):
        # Skip this module and non-adapter modules that don't need registration
        if info.name == __name__:
            continue
        # Skip __main__ modules — importing them during package init causes
        # a RuntimeWarning when the user runs `python -m <subpkg>` because
        # Python will load __main__ a second time.
        if info.name.endswith(".__main__"):
            continue
        try:
            importlib.import_module(info.name)
        except Exception as exc:
            # Some discovery modules are optional tooling, so an unrelated
            # optional-module failure is not by itself fatal.  Missing a
            # registry binding required by dispatch *is* fatal and is checked
            # immediately after discovery by ``_validate_required_registry``.
            COST_MODEL_DISCOVERY_ERRORS[info.name] = exc


def _validate_required_registry() -> None:
    """Fail fast when dispatch targets were not registered during discovery.

    Registry completeness is a startup invariant, independent of the pipeline
    selected by an optimizer.  Reporting discovery exceptions here preserves
    the dependency error that caused the incomplete registry instead of
    blaming whichever component happened to be dispatched first.
    """
    from rag_stack.cost_model.dispatch import validate_against_registry

    try:
        validate_against_registry()
    except AssertionError as exc:
        details = "; ".join(
            f"{module}: {type(error).__name__}: {error}"
            for module, error in sorted(COST_MODEL_DISCOVERY_ERRORS.items())
        )
        message = f"required cost-model registry discovery failed: {exc}"
        if details:
            message += f". Import failures: {details}"
            first_error = next(iter(COST_MODEL_DISCOVERY_ERRORS.values()))
            raise RuntimeError(message) from first_error
        raise RuntimeError(message) from exc


_auto_discover()
_validate_required_registry()

# Unified assembly export — last, so the registry (_auto_discover) is ready.
from rag_stack.cost_model.assembly import (  # noqa: E402,F401
    RAGCMAssembly,
    compute_trace_performance,
)
