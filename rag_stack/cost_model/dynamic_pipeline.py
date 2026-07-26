"""Explicit analytical scoring → shortlisting → closed-loop simulation.

Candidate enumeration and final objective selection belong to ``assembly``.
This module owns the intervening phase boundaries: RAGO's coarse scores are
replaced by trace-aware analysis, only that analysis ranks the DES shortlist,
and DES publishes the final candidate throughput and latency.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from rag_stack.cost_model import closed_loop_sim, engines, hybrid_screening
from rag_stack.cost_model.workload import PreparedExecutionPaths, resolve_execution_paths


def evaluate_candidates(
    engine: engines.DynamicEngine,
    df: pd.DataFrame,
    sweep_dfs: dict,
    *,
    pipeline_config: dict,
    hardware,
    traces: list,
    trace_source_question_ids: list[Any] | None = None,
    fixed_placement=None,
    prepared_paths: PreparedExecutionPaths | None = None,
) -> pd.DataFrame:
    """Price exact candidates and execute one objective-aware screening plan.

    All phases share the same trace cohort, concrete candidate layouts and
    frozen calibration snapshot. Rows rejected by screening remain in the
    output as audit records; final selection must honor simulator status.
    """
    if df is None or len(df) == 0:
        return df
    prepared_paths = resolve_execution_paths(traces, prepared_paths)
    traces = prepared_paths.paths
    pipeline_config = closed_loop_sim._pipeline_with_frozen_stage_batch_selection(
        pipeline_config, hardware,
    )
    # Fixed replay and Apps preview bind exactly the same physical plan. The
    # analytical projection is read from that graph, and DES receives the very
    # same retained inputs rather than compiling the route a second time.
    compiled_plan = None
    analysis_options = {}
    if fixed_placement is not None and len(df) == 1:
        from rag_stack.cost_model.compiled import workload_from_paths, bind_plan
        workload = workload_from_paths(
            pipeline_config, traces, trace_source_question_ids, prepared_paths=prepared_paths,
        )
        compiled_plan = bind_plan(
            workload, hardware=hardware, candidate_row=df.iloc[0],
            prefix_caching=engines._PREFIX_CACHING,
        )
        analysis_options["compiled_plan"] = compiled_plan
    analytical = engine.analyze(
        df,
        sweep_dfs,
        pipeline_config=pipeline_config,
        hardware=hardware,
        traces=traces,
        trace_source_question_ids=trace_source_question_ids,
        fixed_placement=fixed_placement,
        prepared_paths=prepared_paths,
        **analysis_options,
    )
    if analytical is None or len(analytical) == 0:
        return analytical
    screening = hybrid_screening.screen_des_candidates(
        analytical,
        selection=getattr(hardware, "performance", None),
        production=True,
    )
    simulation_options = {}
    if compiled_plan is not None:
        def bound_inputs(*args, **kwargs):
            compiled_plan.validate_candidate(kwargs["candidate_row"])
            return compiled_plan.sim_inputs, None
        simulation_options["builder_fn"] = bound_inputs
    return closed_loop_sim.apply_closed_loop_simulation(
        analytical,
        pipeline_config=pipeline_config,
        hardware=hardware,
        traces=traces,
        trace_source_question_ids=trace_source_question_ids,
        prefix_caching=engines._PREFIX_CACHING,
        screening=screening,
        prepared_paths=prepared_paths,
        **simulation_options,
    )
