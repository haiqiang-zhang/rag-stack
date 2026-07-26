"""SMAC ConfigSpace conversion: ordered numeric choices must be Ordinal.

The paper-bench fairness contract is that every optimizer family sees the
same search-space structure. ax/agent encode ordered numeric choices with
real value spacing (ssops.choice_grid); the SMAC side must expose the same
order via OrdinalHyperparameter — an unordered Categorical turns max_iter /
top_k / batch into one-hot classes and blinds the RF to monotone structure
(the 07-03 smac_s43 collapse). Hierarchy must survive as EqualsCondition.
"""
import yaml
import pytest
from copy import deepcopy
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    OrdinalHyperparameter,
)

from rag_stack.optimizer.smac.configspace import (
    ax_to_configspace,
    cs_config_to_dict,
    dict_to_cs_config,
)
from rag_stack.search_space.rag_search_space import RagSearchSpace

CFG = "configs/rag_stack/paper_bench_msmarco_smac_s43.yaml"


@pytest.fixture(scope="module")
def space():
    return RagSearchSpace.from_yaml(yaml.safe_load(open(CFG)))


@pytest.fixture(scope="module")
def cs(space):
    return ax_to_configspace(space.ss, seed=43)


def test_ordered_numeric_choices_become_ordinal(cs):
    mi = cs["pipeline__sequential__react__max_iter"]
    assert isinstance(mi, OrdinalHyperparameter)
    assert list(mi.sequence) == [3, 5, 10, 15]      # native ints, ascending
    tk = cs["pipeline__sequential__semantic_retrieval__top_k"]
    assert isinstance(tk, OrdinalHyperparameter)
    assert list(tk.sequence) == [1, 2, 4, 8, 16, 32, 64]


def test_string_choices_stay_categorical(cs):
    # binary gaters stay Categorical even though Ax forces is_ordered=True
    # on 2-value parameters (ordinal ≡ categorical at 2 levels; conditions
    # machinery untouched)
    assert isinstance(cs["pipeline__sequential__rag_dataflow"], CategoricalHyperparameter)
    # the model ladder is DECLARED ordered (algo_search_space.ordered_knobs)
    # → OrdinalHyperparameter in declaration (size) order
    from ConfigSpace.hyperparameters import OrdinalHyperparameter
    hp = cs["pipeline__sequential__generator__vllm_api__model"]
    assert isinstance(hp, OrdinalHyperparameter)
    assert [s.split("-")[-2] for s in hp.sequence] == ["1.5B", "3B", "7B", "14B"]


def test_hierarchy_conditions_survive(space, cs):
    n_dependents = sum(
        len(deps)
        for p in space.ss.parameters.values()
        if getattr(p, "is_hierarchical", False)
        for deps in p.dependents.values()
    )
    assert len(cs.conditions) == n_dependents > 0


def test_sampled_configs_decode(space, cs):
    """The PRODUCTION path: SMAC Configuration → cs_config_to_dict → controller.
    (The reverse encoders are unused legacy API and are not exercised here.)"""
    n_react = n_seq = 0
    for cfg in cs.sample_configuration(100):
        decoded = cs_config_to_dict(cfg, space.ss, ssops=space)
        p = decoded["pipeline"]["sequential"]
        if p["rag_dataflow"] == "react":
            n_react += 1
            assert isinstance(p["react"]["max_iter"], int)
            assert "query_expansion" not in p and "passage_compressor" not in p
        else:
            n_seq += 1
            assert "react" not in p
    assert n_react and n_seq


def test_checkpoint_conversion_preserves_active_gated_branches(space, cs):
    """Nested checkpoint configs must map back to the exact active CS branch."""
    from rag_stack.optimizer import OPTIMIZER_REGISTRY

    objectives = [
        {"objective_name": "performance", "cost": 1, "ref_point": 0},
        {"objective_name": "quality", "cost": 10, "ref_point": 0},
    ]
    opt = OPTIMIZER_REGISTRY["smac_mobo"](
        space.ss, objectives=objectives, seed=43, n_gt_init=4, ssops=space,
    )
    seen_vdb = set()
    seen_flow = set()
    seen_rerankers = set()
    seen_query_expanders = set()
    for original in cs.sample_configuration(300):
        active = dict(original)
        nested = cs_config_to_dict(original, space.ss, ssops=space)
        generator = nested["pipeline"]["sequential"]["generator"]
        generator["component"] = "vllm_api"  # fixed checkpoint-only fields
        generator["uri"] = "http://judge.invalid:8000"
        restored = dict_to_cs_config(nested, opt.cs, space.ss, ssops=space)
        assert dict(restored) == dict(original)
        seen_vdb.add(
            nested["pipeline"]["sequential"]["semantic_retrieval"]["vectordb"]
        )
        seen_flow.add(active["pipeline__sequential__rag_dataflow"])
        reranker = active.get("pipeline__sequential__passage_reranker__component")
        if reranker is not None:
            seen_rerankers.add(reranker)
        query_expander = active.get(
            "pipeline__sequential__query_expansion__component"
        )
        if query_expander is not None:
            seen_query_expanders.add(query_expander)
    assert seen_vdb == {"faiss_ivf_index", "faiss_hnsw_index"}
    assert seen_flow == {"sequential", "react"}
    assert "pipeline__sequential__query_expansion__component" in cs
    assert seen_query_expanders == {
        "query_decompose", "hyde", "multi_query_expansion",
    }
    assert seen_rerankers == {
        "tart",
        "monot5",
        "colbert_reranker",
        "sentence_transformer_reranker",
        "flag_embedding_reranker",
    }


def test_smac_trains_on_ordinal_space(space):
    """End-to-end micro-run: SMAC must build its RF over the Ordinal space and
    keep suggesting valid configs past the initial design (synthetic tells)."""
    from rag_stack.optimizer import OPTIMIZER_REGISTRY

    objectives = [
        {"objective_name": "performance", "cost": 1, "ref_point": 0},
        {"objective_name": "quality", "cost": 10, "ref_point": 0},
    ]
    opt = OPTIMIZER_REGISTRY["smac_mobo"](
        space.ss, objectives=objectives, seed=43, n_gt_init=4, ssops=space,
    )
    for i in range(8):                       # 4 init + 4 model-based
        xs, names = opt.suggest(1)
        assert xs and names == ["performance", "quality"]
        p = xs[0]["pipeline"]["sequential"]
        assert p["rag_dataflow"] in ("sequential", "react")
        opt.add_objectives(
            xs,
            {"performance": [10.0 + i], "quality": [0.2 + 0.01 * i]},
            eval_ids=[i],
        )
    configs, objs, _ = opt.get_pareto()
    assert configs and objs


def test_smac_warm_start_restores_history_without_remeasurement(space):
    """Resume restores observations and the next ask is a new config."""
    from rag_stack.optimizer import OPTIMIZER_REGISTRY

    objectives = [
        {"objective_name": "performance", "cost": 1, "ref_point": 0},
        {"objective_name": "quality", "cost": 10, "ref_point": 0},
    ]

    def make_optimizer():
        return OPTIMIZER_REGISTRY["smac_mobo"](
            space.ss,
            objectives=objectives,
            seed=43,
            n_gt_init=4,
            ssops=space,
        )

    original = make_optimizer()
    history = []
    for i in range(6):  # cross the four-point initial design into model-based BO
        configs, names = original.suggest()
        values = {"performance": [10.0 + i], "quality": [0.20 + 0.01 * i]}
        original.add_objectives(configs, values, eval_ids=[i])
        history.append({
            "eval_id": i,
            "config": configs[0],
            "objectives": {name: values[name][0] for name in names},
            "eval_type": "GT",
        })

    # Controller checkpoints contain the sampled config plus fixed fields
    # restored from CSV/base YAML. Those must not look like replay divergence.
    checkpoint_history = deepcopy(history)
    for ev in checkpoint_history:
        generator = ev["config"]["pipeline"]["sequential"]["generator"]
        generator["component"] = "vllm_api"
        generator["uri"] = "http://judge.invalid:8000"
        generator["max_tokens"] = 512

    resumed = make_optimizer()
    resumed.warm_start(checkpoint_history)
    source_seeds = {key.seed for key in original._smac.runhistory._data}
    restored_seeds = {key.seed for key in resumed._smac.runhistory._data}
    assert len(source_seeds) == 1
    assert restored_seeds == source_seeds
    assert resumed._smac.runhistory.submitted == len(history)
    assert resumed._smac.runhistory.finished == len(history)
    assert resumed._smac.runhistory.running == 0
    assert resumed._eval_ids == list(range(len(history)))

    resumed_next, _ = resumed.suggest()
    from rag_stack.optimizer.utils import config_signature
    historical = {config_signature(ev["config"]) for ev in history}
    assert config_signature(resumed_next[0]) not in historical
    assert resumed._smac.runhistory.submitted == len(history) + 1
    assert resumed._smac.runhistory.running == 1
    resumed.add_objectives(
        resumed_next,
        {"performance": [16.0], "quality": [0.26]},
        eval_ids=[len(history)],
    )
    assert resumed._eval_ids[-1] == len(history)
    assert resumed._smac.runhistory.finished == len(history) + 1
    assert resumed._smac.runhistory.running == 0


def test_smac_warm_start_skips_duplicate_checkpoint_rows(space):
    from rag_stack.optimizer import OPTIMIZER_REGISTRY

    objectives = [
        {"objective_name": "performance", "cost": 1, "ref_point": 0},
        {"objective_name": "quality", "cost": 10, "ref_point": 0},
    ]
    source = OPTIMIZER_REGISTRY["smac_mobo"](
        space.ss, objectives=objectives, seed=43, n_gt_init=4, ssops=space,
    )
    configs = []
    for i in range(2):
        xs, _ = source.suggest()
        source.add_objectives(
            xs,
            {"performance": [10.0 + i], "quality": [0.20 + 0.01 * i]},
            eval_ids=[i],
        )
        configs.append(deepcopy(xs[0]))

    duplicate = [
        {
            "eval_id": i,
            "config": configs[0],
            "objectives": {"performance": 10.0 + i, "quality": 0.20 + 0.01 * i},
            "eval_type": "GT",
        }
        for i in range(2)
    ]
    resumed = OPTIMIZER_REGISTRY["smac_mobo"](
        space.ss, objectives=objectives, seed=43, n_gt_init=4, ssops=space,
    )
    # Merged (tolerant) semantics: campaign checkpoints legitimately carry
    # duplicate configs (a re-proposed arm, a menu-era re-eval), so resume
    # SKIPS the duplicate row instead of aborting the whole restore — the
    # runhistory ends with ONE entry for the config and the resume proceeds.
    resumed.warm_start(duplicate)
    assert len(resumed._smac.runhistory) == 1
    assert len(resumed._warm_costs) == 1


def test_smac_warm_start_skips_corrupted_rows_and_fails_loud_when_all_bad(space, cs):
    from rag_stack.optimizer import OPTIMIZER_REGISTRY

    objectives = [
        {"objective_name": "performance", "cost": 1, "ref_point": 0},
        {"objective_name": "quality", "cost": 10, "ref_point": 0},
    ]
    corrupted = cs_config_to_dict(cs.sample_configuration(), space.ss, ssops=space)
    del corrupted["pipeline"]["sequential"]["generator"]["model"]
    good = cs_config_to_dict(cs.sample_configuration(), space.ss, ssops=space)
    resumed = OPTIMIZER_REGISTRY["smac_mobo"](
        space.ss, objectives=objectives, seed=43, n_gt_init=4, ssops=space,
    )
    # A corrupted row (missing ACTIVE parameter) must never be laundered into
    # the runhistory with fabricated encoding defaults — dict_to_cs_config
    # raises, warm_start skips the row, and the healthy rows still restore.
    resumed.warm_start([
        {"eval_id": 0, "config": corrupted,
         "objectives": {"performance": 10.0, "quality": 0.2}, "eval_type": "GT"},
        {"eval_id": 1, "config": good,
         "objectives": {"performance": 11.0, "quality": 0.3}, "eval_type": "GT"},
    ])
    assert len(resumed._smac.runhistory) == 1
    assert resumed._eval_ids == [1]

    # ... and when EVERY row is corrupted the restore fails loudly instead of
    # silently resuming with an empty model.
    fresh = OPTIMIZER_REGISTRY["smac_mobo"](
        space.ss, objectives=objectives, seed=43, n_gt_init=4, ssops=space,
    )
    with pytest.raises(RuntimeError, match="none of the 1"):
        fresh.warm_start([
            {"eval_id": 0, "config": corrupted,
             "objectives": {"performance": 10.0, "quality": 0.2}, "eval_type": "GT"},
        ])
    assert len(fresh._smac.runhistory) == 0
