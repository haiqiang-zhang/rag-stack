"""CM fixed-placement performance provider — price ONE deployment per trial.

The ``(system_space_owner: optimizer, performance_source: cost_model)`` cell:
the optimizer samples a complete deployment (``deployment_preset`` expanded
to a resolved ``system_config``), and the analytical cost model prices
EXACTLY that deployment — no internal sweep, no best-pick — simulating what
a measured run on the target hardware would report.

Positionally this mirrors ``MeasuredProvider`` (same ``(pipeline_config,
system_config)`` contract, same selection-policy scalar axis); mechanically
it is a thin shell over :meth:`RAGCMAssembly.evaluate_fixed`, which owns the
``cm_fixed_pipeline_config`` fold → ``cm_fixed_placement_inputs``
reconstruction. Returns the normal ``AssemblyResult``, so persistence
(assembly.json, rago_sweep.csv) and the optimizer-facing
``performance_score`` need no special-casing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("RAG-Stack")


class CmFixedPlacementProvider:
    """Prices a fixed deployment through the cost model.

    ``provides_quality = False``: unlike ``MeasuredProvider`` (whose real run
    co-produces quality), quality still comes from the normal GT evaluator —
    this provider replaces only the CM's deployment sweep."""

    provides_quality = False

    def __init__(self, assembly: Any):
        self.assembly = assembly

    def evaluate(
        self,
        pipeline_config: dict,
        system_config: dict,
        *,
        trace: Any | None = None,
        runtime_stats: Optional[dict] = None,
    ) -> Any:
        """AssemblyResult for exactly this deployment.

        ``trace`` → dynamic engine (replay of the quality run's recorded
        execution); ``runtime_stats`` (the static evaluators' aggregate,
        stamped onto the schema) or neither → static engine.
        ``batch_size_request``/``batch_size_decode`` come from the resolved
        ``system_config`` (already floored/derived by the resolve layer), so
        the priced batch is the one the deployment declares.
        """
        logger.info(
            "[cm-fixed] pricing deployment preset=%s batch=%s/%s",
            system_config.get("deployment_preset", "<raw>"),
            system_config.get("batch_size_request"),
            system_config.get(
                "batch_size_decode", system_config.get("batch_size_request"),
            ),
        )
        return self.assembly.evaluate_fixed(
            pipeline_config,
            system_config,
            trace=trace,
            runtime_stats=runtime_stats,
        )
