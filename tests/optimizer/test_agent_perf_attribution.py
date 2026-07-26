"""Agent advise-context enrichment: per-stage CM attribution + global coverage.

Round-1 SOTA-loop change (07-04): the LLM's structural perf knowledge kept
losing the acqf arbitration because it could not SEE where time goes
(per-stage bottlenecks), that perf is deployment-fair (each config auto-tuned),
or the global tried/untried state of every knob value. These tests pin the
three information channels end to end at the unit level.
"""
import types

import pytest

from rag_stack.controller import _perf_attr_from_best
from rag_stack.search_space.rag_search_space import RagSearchSpace


BEST = {
    "dynamic_performance_model": "closed_loop_simulator",
    "simulator_status": "ok",
    "simulator_qps": 258.1639,
    "simulator_latency_mean_s": 15.8503,
    "simulator_engine_fill_mean": {
        "generator.prefill.r0": 0.4916,
        "generator.decode.r0": 0.9990,
    },
    "simulator_stage_batch_mean": {"semantic_retrieval": 3.9632},
    "simulator_transfer_latency_mean": {"generator_kv": 0.0008016},
    "simulator_transfer_queue_wait_mean": {"generator_kv": 0.0004355},
    "simulator_saturation_plateau": True,
    "simulator_saturation_stop_reason": "qps_plateau",
    "simulator_populations_tried": [273, 546, 1092],
    "load_concurrency": 1092,
    "candidate_system_layout": {
        "engines": {
            "generator": {"pd_serving": "disagg_pd"},
            "semantic_retrieval": {"pd_serving": "collocated_pd"},
        },
    },
    "placement_policy": "disaggregated",
    "num_chips_e2e": 4,
    "batch_size_request": 256,
    "batch_size_decode": 128,
    # Retained analytical fields conflict deliberately with DES and must not
    # leak into the Agent card.
    "qps": 258.1639,
    "latency_s": 99.0,
    "qps_bound": ["generator_prefill"],
    "qps_generator_prefill": 11.4,
    "qps_generator_decode": 7.29,
}


def _self_contained_space() -> RagSearchSpace:
    """Small in-memory analogue of the paper's hierarchical RAG space.

    Keep optimizer unit tests independent of mutable benchmark/config files
    while retaining the contracts they exercise: co-mode dataflow gating,
    optional stages, unordered component choices, and ordered model ladders.
    """
    nodes = [
        {
            "stage": "query_expansion",
            "optional": True,
            "modules": [{
                "component": "query_decompose",
                "model": {"ordered": ["model-1", "model-2", "model-3", "model-4"]},
            }],
        },
        {
            "stage": "semantic_retrieval",
            "top_k": [2, 4, 8],
            "modules": [{"component": "vectordb"}],
        },
        {
            "stage": "passage_reranker",
            "optional": True,
            "modules": [
                {"component": "reranker-a"},
                {"component": "reranker-b"},
                {"component": "reranker-c"},
            ],
        },
        {
            "stage": "passage_compressor",
            "optional": True,
            "modules": [{"component": "compressor", "rate": [0.2, 0.5]}],
        },
        {
            "stage": "prompt_maker",
            "modules": [
                {"component": "fstring"},
                {"component": "long_context_reorder"},
            ],
        },
        {
            "stage": "generator",
            "modules": [{
                "component": "vllm",
                "model": {"ordered": ["model-1", "model-2", "model-3", "model-4"]},
                "temperature": [0.0, 0.5],
            }],
        },
    ]
    return RagSearchSpace.from_yaml({
        "algo_search_space": {
            "pipeline": {
                "rag_dataflow": [
                    "sequential",
                    {"react": {"max_iter": [3, 5]}},
                ],
                "node_lines": [{"node_line_name": "main", "nodes": nodes}],
            },
        },
    })


def test_attr_extraction_uses_only_authoritative_des_evidence():
    res = types.SimpleNamespace(best_hw_params=BEST)
    attr = _perf_attr_from_best(res)
    assert attr["source"] == "authoritative_des"
    assert attr["qps"] == pytest.approx(258.1639)
    assert attr["latency_s"] == pytest.approx(15.8503)
    assert attr["engines"]["generator"]["pd_serving"] == "disagg_pd"
    assert attr["engines"]["generator"]["identifiability"] == "per_role"
    replicas = attr["engines"]["generator"]["replicas"]
    assert replicas["generator.decode.r0"]["fill_fraction"] == pytest.approx(0.9990)
    assert replicas["generator.prefill.r0"]["fill_fraction"] == pytest.approx(0.4916)
    assert attr["workers"]["semantic_retrieval"]["batch_mean"] == pytest.approx(3.9632)
    assert attr["transfers"]["generator_kv"]["queue_wait_mean_s"] == pytest.approx(0.0004355)
    assert attr["saturation"]["plateau"] is True
    assert attr["deployment"] == {
        "policy": "disaggregated",
        "chips": 4,
        "batch_request": 256,
        "batch_decode": 128,
    }
    assert "bound" not in attr and "stage_qps" not in attr

    # Checkpoint resume reads the persisted mapping directly.  It must render
    # exactly the same card and ignore every retained analytical diagnostic.
    restored = _perf_attr_from_best({
        **BEST,
        "qps_prefill_capacity": 6.2,
        "qps_gpu_resource_capacity": 5.9,
    })
    assert restored == attr


def test_attr_extraction_keeps_collocated_generator_unified():
    row = {
        **BEST,
        "simulator_engine_fill_mean": {"generator.unified.r0": 1.0},
        "simulator_transfer_latency_mean": {},
        "simulator_transfer_queue_wait_mean": {},
        "candidate_system_layout": {
            "engines": {"generator": {"pd_serving": "collocated_pd"}},
        },
    }
    attr = _perf_attr_from_best(row)
    generator = attr["engines"]["generator"]
    assert generator["pd_serving"] == "collocated_pd"
    assert generator["identifiability"] == "unified_only"
    assert set(generator["replicas"]) == {"generator.unified.r0"}
    rendered = _renderer()._fmt_attr(attr)
    assert "P/D not separable" in rendered
    assert "prefill" not in rendered and "decode" not in rendered


def test_attr_extraction_never_raises():
    assert _perf_attr_from_best(types.SimpleNamespace(best_hw_params=None)) is None
    assert _perf_attr_from_best(types.SimpleNamespace()) is None
    assert _perf_attr_from_best({
        "qps": 12.0,
        "qps_bound": ["generator_decode"],
        "qps_generator_decode": 12.0,
    }) is None


@pytest.fixture(scope="module")
def agent():
    from rag_stack.optimizer import OPTIMIZER_REGISTRY

    space = _self_contained_space()
    objectives = [
        {"objective_name": "performance", "cost": 1, "ref_point": 0},
        {"objective_name": "quality", "cost": 10, "ref_point": 0},
    ]
    opt = OPTIMIZER_REGISTRY["agent_qnehvi"](
        space.ss, objectives=objectives, seed=46, n_both_init=4,
        ssops=space,
    )
    xs, _ = opt.suggest(4)
    opt.add_objectives(
        xs,
        {"performance": [10.0, 40.0, 5.0, 90.0],
         "quality": [{"combined": 0.2}, {"combined": 0.3},
                     {"combined": 0.5}, {"combined": 0.15}]},
        eval_ids=[0, 1, 2, 3],
    )
    opt.register_perf_attribution(3, _perf_attr_from_best(
        types.SimpleNamespace(best_hw_params=BEST)))
    return opt


def _renderer():
    from rag_stack.optimizer.agent import LLMAgent
    return LLMAgent(model="unused")   # client is lazy — render only


def test_attr_reaches_obs_card(agent):
    hist = agent._history()
    tagged = [h for h in hist if h.get("perf_attr")]
    assert len(tagged) == 1
    card = _renderer()._obs_card(hist, set(hist[0]["flat"]), agent.ssops)
    assert "DES qps=" in card
    assert "generator.decode.r0 fill=0.999" in card
    assert "generator.prefill.r0 fill=0.492" in card
    assert "generator_prefill-bound" not in card


def test_coverage_card_shows_untried_and_counts(agent):
    hist = agent._history()
    vary = set(hist[0]["flat"])
    card = _renderer()._coverage_card(hist, vary, agent.ssops)
    assert "UNTRIED" in card          # unexplored values are surfaced
    assert "(1x," in card             # tried values carry counts + bests


def test_model_size_choices_ordinalized_in_agent_gp(agent):
    """SPACE-level ordered declaration ({ordered: [...]} in YAML, 07-04) keeps
    model-size ladders out of the categorical set for every optimizer; genuinely
    unordered string choices (component names) stay categorical."""
    names = agent.ssops.param_names
    cat_names = {names[i] for i in agent.cat_dims_set}
    model_dims = {n for n in names if n.endswith("__model")}
    assert model_dims, "space should have model knobs"
    assert not (model_dims & cat_names), "sized model choices must be ordinal"
    multi_unordered = {
        n for n in names
        if n.endswith("__component")
        and len(getattr(agent.ssops.ss.parameters[n], "values", [])) >= 3
    }
    assert multi_unordered & cat_names, \
        "multi-value component choices must remain categorical"


def test_cvd_pin_maps_through_visible_set(monkeypatch):
    """cuda:N in YAML is a LOGICAL index into the process's visible set:
    under CVD=2,3 pinning cuda:1 must select physical GPU 3 (07-04 escape
    bug: it pinned physical 1 and collided outside the pair)."""
    from rag_stack_evaluator.static_rag_evaluator.nodes.generator.vllm import Vllm

    class Bare(Vllm):
        def __init__(self):  # bypass engine construction — test the pin only
            pass

    b = Bare()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    b._maybe_pin_cvd("cuda:1")
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    b._restore_cvd()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    b2 = Bare()
    b2._maybe_pin_cvd("cuda:1")           # unrestricted → absolute unchanged
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    b2._restore_cvd()


def test_dls_seed_groups_partition(agent):
    """Provenance groups: llm (advice given) / incumbent / random partition
    the restart seeds; every group non-empty in a warm state with advice."""
    import torch
    from rag_stack.optimizer.agent import AgentAdvice

    centers = torch.stack(agent._x_rows[:2], dim=0)
    groups = agent._restart_seed_groups(AgentAdvice(centers=centers))
    assert set(groups) == {"llm", "incumbent", "random"}
    assert all(g.shape[0] >= 1 for g in groups.values())
    no_llm = agent._restart_seed_groups(None)
    assert "llm" not in no_llm and no_llm["random"].shape[0] >= 2


def test_grouped_dls_provenance_labels(agent, monkeypatch):
    """Each source returns one endpoint and one failing source stays isolated."""
    import torch
    from rag_stack.optimizer.agent import dls as mod
    from rag_stack.optimizer.agent import AgentAdvice

    calls = []
    def fake_dls(acq_function, discrete_choices, q, num_restarts,
                 batch_initial_conditions, X_avoid, max_batch_size):
        calls.append(num_restarts)
        # Exercise production novelization: evaluated seeds must be moved to
        # legal, unseen starts before BoTorch applies its X_avoid filtering.
        assert all(
            not agent.ssops.is_duplicate_decoded(row.reshape(1, -1), X_avoid)
            for row in batch_initial_conditions[:, 0, :]
        )
        if len(calls) == 2:                      # fail the middle group
            raise RuntimeError("boom")
        return batch_initial_conditions[:1, 0, :], None
    monkeypatch.setattr(mod, "optimize_acqf_discrete_local_search", fake_dls)

    centers = torch.stack(agent._x_rows[:2], dim=0)
    random = torch.stack(agent._x_rows[:3], dim=0)
    monkeypatch.setattr(
        agent,
        "_restart_seed_groups",
        lambda _advice: {
            "llm": centers,
            "incumbent": agent._x_rows[2].reshape(1, -1),
            "random": random,
        },
    )
    out = agent._dls_candidates_grouped(object(), AgentAdvice(centers=centers))
    assert out is not None
    endpoints, prov = out
    # The incumbent source fails; the remaining source-level multi-start calls
    # still return in deterministic task order.
    assert endpoints.shape[0] == len(prov)
    starts = agent._dls_channel_diagnostics["channels"]["dls-llm"]
    assert starts["evaluated_starts"] == centers.shape[0]
    assert starts["novelized_starts"] == centers.shape[0]
    assert prov == [
        "dls-llm",
        "dls-random",
        "dls-perf",
        "dls-quality",
    ]

def test_evaluated_pool_rows_are_dropped_with_provenance():
    """An evaluated duplicate must not survive into subclass pool ranking."""
    import torch
    from rag_stack.optimizer.agent import AgentGuidedQNEHVI

    class ExactDecodedOps:
        @staticmethod
        def is_duplicate_decoded(new_x, existing_x):
            return any(torch.equal(new_x[0], row) for row in existing_x)

    opt = AgentGuidedQNEHVI.__new__(AgentGuidedQNEHVI)
    opt._x_rows = [torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])]
    opt.tkwargs = {"dtype": torch.float32, "device": torch.device("cpu")}
    opt.ssops = ExactDecodedOps()
    opt._t = 20
    cand = torch.tensor([
        [1.0, 1.0],  # evaluated: this was the sole guard survivor in production
        [0.2, 0.8],  # novel
        [0.8, 0.2],  # novel
        [0.0, 0.0],  # evaluated
    ])
    kept, prov = opt._drop_evaluated_pool_rows(
        cand, ["duplicate", "novel-a", "novel-b", "duplicate"],
    )
    assert torch.equal(kept, cand[[1, 2]])
    assert prov == ["novel-a", "novel-b"]


def test_unique_rows_with_index_keeps_prov_alignment():
    import torch
    from rag_stack.optimizer.agent.util import _unique_rows_with_index

    x = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.1, 0.2], [0.5, 0.6]])
    deduped, keep = _unique_rows_with_index(x)
    assert deduped.shape[0] == 3 and keep == [0, 1, 3]


def test_global_vision_cards(agent):
    """07-05 three-card battery: ledger (mode-baseline diff), region rollup
    (joint-signature groups), and the memo round-trip."""
    from rag_stack.optimizer.agent import LLMAgent

    la = LLMAgent(ai_client=object())
    hist = agent._history()
    flats = [h["flat"] for h in hist]
    vary = la._varying_knobs(flats) or set(agent.ssops.param_names)
    f2s, _, _ = la._aliases(agent.ssops)

    ledger = la._ledger_card(hist, vary, agent.ssops, f2s)
    assert "LEDGER-BASE:" in ledger
    assert ledger.count("\n") >= len(hist)          # one row per eval + headers
    assert "e00" in ledger and f"e{len(hist)-1:02d}" in ledger

    roll = la._signature_card(hist, vary, agent.ssops, f2s)
    if roll:                                        # needs >=4 evals and a varying knob
        assert "REGION ROLLUP" in roll and "n=" in roll

    # memo round-trip: parsed search_notes → advice.notes → agent._notes → prompt
    candidate = {
        name: value
        for name, value in hist[0]["flat"].items()
        if name in agent.ssops.active_params(hist[0]["flat"])
    }
    parsed = {"reasoning": "r", "search_notes": "react fast end validated; stop seq bets",
              "candidates": [candidate]}
    adv = la._build_advice_from_parsed(parsed, hist, agent.ssops)
    assert adv.notes.startswith("react fast end")
    assert la._notes == adv.notes
    msgs = la._build_messages(hist, agent.ssops, mo={"pareto": []})
    joined = " ".join(m["content"] for m in msgs)
    assert "YOUR SEARCH NOTES" in joined and "GLOBAL LEDGER" in joined
    assert "search_notes" in joined                 # schema asks for upkeep


def test_gap_target_and_infeasible_context_reach_prompt(agent):
    """Ordinary Agent advice sees the HV gap and known-invalid arms."""
    from rag_stack.optimizer.agent import LLMAgent

    agent.register_infeasible({"pipeline__sequential__semantic_retrieval__top_k": 64},
                              "maximum context length is 32768 tokens")
    assert agent._infeasible

    la = LLMAgent(ai_client=object())
    hist = agent._history()
    mo = {"pareto": [], "hv_rectangle": agent._max_hv_rectangle(),
          "infeasible": agent._infeasible}
    msgs = la._build_messages(hist, agent.ssops, mo=mo)
    joined = " ".join(m["content"] for m in msgs)
    assert "MULTI-OBJECTIVE Bayesian optimizer" in joined
    assert "true target is the HYPERVOLUME" in joined
    assert "MAX-HV EMPTY RECTANGLE" in joined
    assert "INFEASIBLE ARMS" in joined and "32768" in joined
    assert "FILL THE MAX-HV RECTANGLE" in joined


def test_max_hv_rectangle_targets_ordinary_prompt(agent):
    """The literal max-area rectangle, not Euclidean distance, guides advice."""
    from rag_stack.optimizer.agent import LLMAgent
    la = LLMAgent(ai_client=object())
    rect = {"perf_lo": 60.0, "perf_hi": 250.0, "q_lo": 0.15, "q_hi": 0.17,
            "area_norm": 0.31,
            "target": {"perf": 155.0, "quality": 0.16}}
    ordinary = {"pareto": [], "hv_rectangle": rect}
    joined = " ".join(
        m["content"] for m in la._build_messages(agent._history(), agent.ssops, mo=ordinary)
    )
    assert "MAX-HV EMPTY RECTANGLE" in joined
    assert "155" in joined
    assert "FILL THE MAX-HV RECTANGLE" in joined
    assert "WIDEST FRONTIER GAP" not in joined


def test_max_hv_rectangle_picks_largest_area(agent):
    """_max_hv_rectangle must pick the adjacent frontier pair spanning the
    largest normalized AREA — not the longest Euclidean gap."""
    # Frontier: (10,0.9) (20,0.85) (100,0.2): the (20→100)x(0.85→0.2) block is
    # by far the largest area.
    agent._perf[:] = [10.0, 20.0, 100.0, 5.0][: len(agent._perf)]
    agent._corr[:] = [0.9, 0.85, 0.2, 0.1][: len(agent._corr)]
    rect = agent._max_hv_rectangle()
    assert rect and rect["perf_lo"] == 20.0 and rect["perf_hi"] == 100.0
    assert rect["q_lo"] == 0.2 and rect["q_hi"] == 0.85
    t = rect["target"]
    assert 20.0 < t["perf"] < 100.0 and 0.2 < t["quality"] < 0.85


def test_max_hv_rectangle_disagrees_with_long_thin_euclidean_gap(agent):
    """Regression: a long, thin Euclidean gap must not outrank a smaller-distance
    rectangle that contains much more unclaimed dominated area."""
    agent._perf[:] = [0.0, 40.0, 100.0, 0.0][: len(agent._perf)]
    agent._corr[:] = [1.0, 0.6, 0.55, 0.0][: len(agent._corr)]
    rect = agent._max_hv_rectangle()
    # The 40->100 segment has the greater normalized Euclidean length, while
    # the 0->40 segment encloses the larger hypervolume rectangle.
    assert rect["perf_lo"] == 0.0 and rect["perf_hi"] == 40.0
    assert rect["area_norm"] > 0.15


def test_agent_prompt_explains_react_runtime_and_stripped_stages(agent):
    """The Agent must see the executable ReAct loop, not a flat bag of knobs."""
    from rag_stack.optimizer.agent import LLMAgent

    la = LLMAgent(ai_client=object())
    card = la._dataflow_semantics_card(agent.ssops)
    assert "rag_dataflow=react" in card
    assert "semantic_retrieval, passage_reranker (if enabled), generator" in card
    assert "INACTIVE AND STRIPPED: query_expansion, passage_compressor, prompt_maker" in card
    assert "Branch-only controls: max_iter" in card
    assert "Thought -> Action -> Observation" in card
    assert "Search[query] or Finish[answer]" in card
    assert "actual count is data-dependent" in card
    assert "no-ops" in card

    mo = {"pareto": [], "hv_rectangle": agent._max_hv_rectangle()}
    joined = " ".join(
        m["content"] for m in la._build_messages(agent._history(), agent.ssops, mo=mo)
    )
    assert card in joined


def test_dual_anchor_perf_base(agent):
    """A candidate declaring base:'perf' merges its deltas onto the best-PERF
    evaluated config, not the best-quality one."""
    from rag_stack.optimizer.agent import LLMAgent

    la = LLMAgent(ai_client=object())
    hist = agent._history()
    # make perf/quality champions distinct rows
    agent._perf[:] = [1.0] * len(agent._perf); agent._perf[0] = 99.0
    agent._corr[:] = [0.1] * len(agent._corr); agent._corr[-1] = 0.9
    hist = agent._history()
    parsed = {"reasoning": "r", "candidates": [{"base": "perf"}]}   # empty delta
    adv = la._build_advice_from_parsed(parsed, hist, agent.ssops)
    assert adv.centers is not None
    got = agent.ssops.decode_flat(adv.centers[0].reshape(-1))
    perf_flat = hist[0]["flat"]
    same = sum(1 for k in perf_flat if got.get(k) == perf_flat[k])
    assert same >= len(perf_flat) - 2      # ≈ the perf champion (snap tolerance)
    joined = " ".join(m["content"] for m in la._build_messages(hist, agent.ssops, mo={"pareto": []}))
    assert "BASE_PERF" in joined


def test_ledger_query_engine_and_retrieval_round(agent):
    """07-05 agentic memory: filter/detail queries over full history; the
    advise flow honors a {"queries": ...} first reply with a second call."""
    from rag_stack.optimizer.agent import LLMAgent
    import json as _json

    la = LLMAgent(ai_client=object())
    hist = agent._history()
    # detail query
    ev = la._run_ledger_queries(["detail e00 e01"], hist, agent.ssops)
    assert "e00" in ev and "perf=" in ev
    # filter query on a real knob leaf name
    leaf = agent.ssops.param_names[0].split("__")[-1]
    v = hist[0]["flat"].get(agent.ssops.param_names[0])
    ev2 = la._run_ledger_queries([f"{leaf}={v}"], hist, agent.ssops)
    assert "QUERY:" in ev2
    # two-turn advise via stub client
    candidate = {
        name: value
        for name, value in hist[0]["flat"].items()
        if name in agent.ssops.active_params(hist[0]["flat"])
    }
    class StubClient:
        def __init__(self):
            self.calls = 0
        async def chat(self, messages, temperature=0.7):
            self.calls += 1
            if self.calls == 1:
                return _json.dumps({"queries": ["detail e00"]})
            return _json.dumps({"reasoning": "r", "search_notes": "n",
                                "candidates": [candidate]})
    stub = StubClient()
    la._client = stub
    adv = la._advise_oneshot(hist, agent.ssops, 5)
    assert stub.calls == 2                      # retrieval round happened
    assert adv.centers is not None


def test_structured_init_probes(agent):
    """Sobol-first init (07-08, user directive): the opening batch serves
    hierarchy-aware Sobol probes and DEFERS the LLM probe plan until those
    results are in history — the designer works from data, not blind."""
    agent._configs = []          # pretend fresh run
    agent._x_rows = []
    if hasattr(agent, "_init_plan"):
        del agent._init_plan
    if hasattr(agent, "_init_cursor"):
        del agent._init_cursor
    n = min(4, agent.init_sobol_k) or 4
    batch = agent._structured_init_batch(n)
    assert len(batch) == n and all(isinstance(c, dict) for c in batch)
    assert not hasattr(agent, "_init_plan"), "phase A must not build the LLM plan"
