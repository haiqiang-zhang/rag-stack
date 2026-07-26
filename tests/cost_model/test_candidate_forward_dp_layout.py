import pandas as pd

from rag_stack.cost_model.candidate_layout import (
    candidate_system_layout_from_row,
    materialize_candidate_system_layouts,
)


def _raw_rago_row(stages: list[str]) -> pd.DataFrame:
    devices = [f"cuda:{index}" for index in range(4)]
    return pd.DataFrame([{
        "collocation_strategy": [stages],
        "stage_devices": {stage: devices for stage in stages},
        "available_devices": devices,
        "pipeline_stage_order": stages,
        "num_chips_e2e": len(devices),
    }])


def test_raw_rago_forward_reranker_materializes_dp4_not_tp4() -> None:
    out = materialize_candidate_system_layouts(
        _raw_rago_row(["passage_reranker"]),
        pipeline_config={"system": {}},
        # A parent-shaped sweep must not turn the concrete encoder forward
        # into an LLM TP engine.
        sweep_dfs={
            "passage_reranker": pd.DataFrame([{
                "num_chips": 4,
                "batch_size": 8,
                "winning_tp": 4,
                "winning_pp": 1,
            }]),
        },
    )

    layout = candidate_system_layout_from_row(out.iloc[0])
    stage = layout["performance_stages"]["passage_reranker"]
    devices = [f"cuda:{index}" for index in range(4)]
    assert (stage["tp"], stage["pp"], stage["replicas"]) == (1, 1, 4)
    assert stage["replica_devices"] == [[device] for device in devices]
    engine = layout["engines"]["passage_reranker"]
    assert (engine["tp"], engine["pp"], engine["replicas"]) == (1, 1, 4)
    assert engine["replica_devices"] == [[device] for device in devices]
    assert out.loc[0, "candidate_tp_passage_reranker"] == 1
    assert out.loc[0, "candidate_pp_passage_reranker"] == 1
    assert out.loc[0, "candidate_replicas_passage_reranker"] == 4


def test_raw_rago_llm_reranker_roles_still_materialize_tp4() -> None:
    stages = ["passage_reranker_prefill", "passage_reranker_decode"]
    sweep = pd.DataFrame([{
        "num_chips": 4,
        "batch_size": 8,
        "winning_tp": 4,
        "winning_pp": 1,
    }])
    out = materialize_candidate_system_layouts(
        _raw_rago_row(stages),
        pipeline_config={"system": {}},
        sweep_dfs={stage: sweep for stage in stages},
    )

    layout = candidate_system_layout_from_row(out.iloc[0])
    devices = [f"cuda:{index}" for index in range(4)]
    for stage_name in stages:
        stage = layout["performance_stages"][stage_name]
        assert (stage["tp"], stage["pp"], stage["replicas"]) == (4, 1, 1)
        assert stage["replica_devices"] == [devices]
