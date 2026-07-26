"""Structural work model for the evaluator's rank/align/sort RRF implementation.

This describes logical work, not calibrated Pandas/Python service time.  The
system adapter separately consumes the existing standalone CPU timing profile.
It deliberately models full rank sorts and final sort-before-truncation, rather
than replacing the implementation with a pre-ranked hash merge or top-k heap.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .algorithm import (
    AlgorithmWorkload, DataProfile, OperatorWorkProfile, PhaseWork,
    register_algorithm_model,
)


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@register_algorithm_model("rrf")
class RRFAlgorithmModel:
    """One RRF invocation over any number of named candidate-list inputs."""

    def build(
        self, config: Mapping[str, Any], workload: AlgorithmWorkload,
    ) -> OperatorWorkProfile:
        inputs = tuple(workload.inputs.values())
        if not inputs:
            raise ValueError("RRF requires at least one input list")
        observed_output = config.get("output_items")
        if observed_output is None:
            top_k = _count(config.get("top_k"), "top_k")
            if top_k == 0:
                raise ValueError("top_k must be positive")
        else:
            observed_output = _count(observed_output, "output_items")
        scalar_bytes = _count(config.get("scalar_bytes", 8), "scalar_bytes")
        id_reference_bytes = _count(
            config.get("id_reference_bytes", 8), "id_reference_bytes",
        )
        if scalar_bytes == 0 or id_reference_bytes == 0:
            raise ValueError("representation byte widths must be positive")
        unique_per_list = []
        assumptions = []
        observed_union = workload.statistics.get("union_items")
        if observed_union is not None:
            observed_union = _count(observed_union, "union_items")
        for index, data in enumerate(inputs):
            value = data.properties.get("unique_items")
            if value is None:
                if observed_union is None:
                    value = data.items
                    assumptions.append(f"input {index}: no within-list duplicates")
                else:
                    value = min(data.items, observed_union)
                    assumptions.append(
                        f"input {index}: unique_items uses the list-size/observed-union upper bound"
                    )
            unique = _count(value, "unique_items")
            if unique > data.items:
                raise ValueError("unique_items exceeds input items")
            unique_per_list.append(unique)
        present = sum(unique_per_list)
        union = observed_union
        if union is None:
            union = present
            assumptions.append("union_items uses the disjoint-list upper bound")
        union = _count(union, "union_items")
        if not max(unique_per_list) <= union <= present:
            raise ValueError("union_items is inconsistent with input cardinalities")

        lists = len(inputs)
        items = sum(data.items for data in inputs)
        cells = lists * union
        table_bytes = cells * scalar_bytes
        index_bytes = union * id_reference_bytes
        pair_bytes = scalar_bytes + id_reference_bytes
        if all(data.bytes is not None for data in inputs):
            input_bytes = sum(data.bytes for data in inputs)
        else:
            input_bytes = items * pair_bytes
            assumptions.append("input bytes use packed ID-reference/score records")
        levels = math.ceil(math.log2(union)) if union > 1 else 0
        output_items = min(top_k, union) if observed_output is None else observed_output
        if output_items > union:
            raise ValueError("output_items exceeds input union cardinality")
        mode = "python" if lists == 1 else "pandas"
        materialized_items = union + output_items if lists > 1 else output_items
        phases = (
            PhaseWork("align", {"hash_insert": items},
                      input_bytes + table_bytes + index_bytes,
                      working_set_bytes=table_bytes + index_bytes, compute_mode=mode),
            PhaseWork("rank", {"comparison": lists * union * levels},
                      2 * table_bytes, working_set_bytes=table_bytes,
                      compute_mode=mode),
            # Each present rank adds rrf_k before division. Multi-list
            # rrf_calculate also accumulates that reciprocal into the score.
            PhaseWork("sum", {"rank_check": cells,
                              "fp_add": present * (2 if lists > 1 else 1),
                              "fp_divide": present},
                      table_bytes + union * scalar_bytes,
                      working_set_bytes=table_bytes + union * scalar_bytes,
                      compute_mode="python"),
            PhaseWork("sort", {"comparison": union * levels},
                      2 * union * pair_bytes,
                      working_set_bytes=union * pair_bytes, compute_mode=mode),
            # The pandas branch calls tolist() before [:top_k]; the Python
            # single-list branch slices first and then builds the output lists.
            PhaseWork("output", {"copy_item": 2 * materialized_items},
                      materialized_items * pair_bytes, compute_mode="python"),
        )
        return OperatorWorkProfile(
            phases=phases,
            dependencies=tuple((a.name, b.name) for a, b in zip(phases, phases[1:])),
            outputs={"candidates": DataProfile(
                output_items, output_items * pair_bytes,
                properties={"unique_items": output_items, "ranked": True},
            )},
            metadata={
                "component": "rrf",
                "implementation": "rrf_pure",
                "implementation_path": "single_list_python" if lists == 1 else "multi_list_pandas",
                "work_basis": "structural_estimate",
                "input_items": items,
                "input_lists": lists,
                "union_items": union,
                "assumptions": tuple(assumptions),
                "memory_basis": "logical scalar/reference storage; excludes Python object overhead",
            },
        )
