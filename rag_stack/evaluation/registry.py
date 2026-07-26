"""Lazy evaluator construction and capability registration.

External distributions declare a ``rag_stack.evaluators`` entry point whose
name is the configured backend and whose value resolves to a ``BackendSpec``.
Listing backends reads package metadata only; selecting one loads its descriptor.
Model libraries should be imported inside the descriptor's factory callable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import metadata
import re
from typing import Any, Callable

from rag_stack.evaluation.api import Evaluator

ENTRY_POINT_GROUP = "rag_stack.evaluators"
_MetadataValues = tuple[str, ...] | Callable[[], tuple[str, ...]]


@dataclass(frozen=True)
class EvaluatorContext:
    """Construction inputs supplied by the host, independent of a backend API."""

    dataset_manager: Any = None
    project_dir: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    dataset: Any = None


@dataclass(frozen=True)
class BackendSpec:
    """A backend factory and the capabilities used by configuration validation.

    ``metrics`` and ``pipeline_modes`` may be tuples or lazy zero-argument
    providers. ``node_lines`` uses the host's component search-space projection;
    ``methods`` uses method-based pipeline selection. Registering a backend does
    not add new cost-model component types or a measured-performance runner.
    """

    name: str
    factory: Callable[[EvaluatorContext], Evaluator]
    metrics: _MetadataValues
    pipeline_modes: _MetadataValues = ("sequential",)
    pipeline_style: str = "node_lines"
    supports_measured: bool = False
    supports_static_ir: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError("Backend names must match [a-z][a-z0-9_]*")
        if self.name == "static":
            raise ValueError("Backend name 'static' is reserved; use 'static_gt'")
        if not callable(self.factory):
            raise TypeError("BackendSpec.factory must be callable")
        if self.pipeline_style not in {"node_lines", "methods"}:
            raise ValueError("BackendSpec.pipeline_style must be 'node_lines' or 'methods'")
        if not isinstance(self.supports_measured, bool):
            raise TypeError("BackendSpec.supports_measured must be a boolean")
        if not isinstance(self.supports_static_ir, bool):
            raise TypeError("BackendSpec.supports_static_ir must be a boolean")
        for name in ("metrics", "pipeline_modes"):
            values = getattr(self, name)
            if not callable(values):
                _metadata_values(values, label=f"{self.name}.{name}")


def _metadata_values(values: _MetadataValues, *, label: str) -> tuple[str, ...]:
    resolved = values() if callable(values) else values
    if not isinstance(resolved, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in resolved
    ):
        raise TypeError(f"{label} must be a tuple of non-empty names or a callable returning one")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{label} contains duplicate names")
    return resolved


_BACKENDS: dict[str, BackendSpec] = {}


def _entry_points() -> tuple[Any, ...]:
    return tuple(metadata.entry_points(group=ENTRY_POINT_GROUP))


def available_backends() -> tuple[str, ...]:
    """List explicitly registered and installed names without loading plugins."""
    return tuple(sorted(set(_BACKENDS) | {entry.name for entry in _entry_points()}))


def register_backend(spec: BackendSpec) -> None:
    """Register a descriptor for this process; duplicate names are errors.

    Use entry points for CLI/subprocess use so every worker discovers the same
    backend. Explicit registration is useful for embedded Python applications.
    """
    if not isinstance(spec, BackendSpec):
        raise TypeError("register_backend expects a BackendSpec")
    if spec.name in _BACKENDS or any(entry.name == spec.name for entry in _entry_points()):
        raise ValueError(f"Evaluator backend {spec.name!r} is already registered or installed")
    _BACKENDS[spec.name] = spec


def get_backend_spec(backend: str = "static_gt") -> BackendSpec:
    """Resolve the selected descriptor, loading only its entry point when needed."""
    name = str(backend).strip().lower()
    if name == "static":
        raise ValueError(
            "global.eval_backend 'static' was renamed to 'static_gt'. Please update your YAML."
        )
    entries = [entry for entry in _entry_points() if entry.name == name]
    if len(entries) > 1:
        raise ValueError(f"Multiple installed evaluator entry points use the name {name!r}")
    if name in _BACKENDS:
        return _BACKENDS[name]
    if not entries:
        raise ValueError(
            f"Unknown global.eval_backend: {name!r}. Supported: {', '.join(available_backends())}."
        )
    spec = entries[0].load()
    if not isinstance(spec, BackendSpec):
        raise TypeError(f"Evaluator entry point {name!r} must resolve to a BackendSpec")
    if spec.name != name:
        raise ValueError(f"Evaluator entry point {name!r} declares backend name {spec.name!r}")
    _BACKENDS[name] = spec
    return spec


def _normalize_backend(backend: str) -> str:
    """Compatibility helper for callers of the previous private dispatcher."""
    return get_backend_spec(backend).name


def create_quality_evaluator(
    dataset: Any = None,
    project_dir: str | None = None,
    *,
    dataset_manager: Any = None,
    backend: str = "static_gt",
    backend_options: dict[str, Any] | None = None,
) -> Evaluator:
    """Construct the selected backend using the same context for every plugin."""
    spec = get_backend_spec(backend)
    if backend_options is not None and not isinstance(backend_options, dict):
        raise TypeError("backend_options must be a dictionary")
    options = backend_options
    evaluator = spec.factory(EvaluatorContext(
        dataset_manager=dataset_manager, project_dir=project_dir,
        options=dict(options or {}), dataset=dataset,
    ))
    if not callable(getattr(evaluator, "evaluate", None)):
        raise TypeError(f"Factory for {spec.name!r} must return an object with an evaluate method")
    return evaluator


def supported_pipeline_modes(backend: str = "static_gt") -> tuple[str, ...]:
    """Return selected backend modes without constructing its evaluator."""
    spec = get_backend_spec(backend)
    return _metadata_values(spec.pipeline_modes, label=f"{spec.name}.pipeline_modes")


def supported_metrics(backend: str = "static_gt") -> tuple[str, ...]:
    """Return the selected backend's quality metric vocabulary."""
    spec = get_backend_spec(backend)
    return _metadata_values(spec.metrics, label=f"{spec.name}.metrics")


__all__ = [
    "BackendSpec", "EvaluatorContext", "ENTRY_POINT_GROUP", "available_backends",
    "register_backend", "get_backend_spec", "create_quality_evaluator",
    "supported_metrics", "supported_pipeline_modes",
]
