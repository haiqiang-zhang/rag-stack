"""Tests for the SearchSpace adapter classes in
:mod:`rag_stack.search_space.ax_search_space_ops` (AxSearchSpaceOps) and
:mod:`rag_stack.search_space.rag_search_space` (RagSearchSpace).

Covers the surviving public surface used by the optimizers:

  * ``param_node_label`` / ``param_leaf_name`` (SEP-segment helpers)
  * ``params_under`` (node-label prefix expansion)
  * ``decode_to_config`` (tensor row → active-only nested config)
  * ``build_name_index`` (nested-key path → flat name reverse map)
  * ``clamp`` (inactive-dim canonicalization — the hierarchy invariant)

The invariant is that Ax's native ``ChoiceParameter.dependents`` is the single
source of truth for hierarchy; these tests pin the adapter behavior against a
real RAG search space built by the single builder
``RagSearchSpace.build_ax_searchspace`` (no synthetic-function fixtures — those
were removed in the optimizer cleanup).
"""
from __future__ import annotations

from tests import REPO_ROOT

from pathlib import Path

import torch
import yaml


from rag_stack.search_space import SEP, INACTIVE_DEFAULT, RagSearchSpace

# A real, hierarchical RAG search space (enabled/component/rag_dataflow
# gating → ChoiceParameter.dependents). Stable surviving config: the paper
# benchmark space (sequential+react co-mode, IVF+HNSW).
_CONFIG_PATH = (
    REPO_ROOT
    / "configs" / "rag_stack" / "paper_bench_agent_s43.yaml"
)


def _rag_ssops() -> RagSearchSpace:
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return RagSearchSpace.from_yaml(cfg)


def _top_labels(ssops: RagSearchSpace) -> set:
    """Distinct first SEP-segments present in the search space."""
    return {RagSearchSpace.param_node_label(n) for n in ssops.param_names}


# ---------------------------------------------------------------------------
# Static SEP-segment helpers (pure string ops; no search space needed).
# ---------------------------------------------------------------------------

def test_param_node_label():
    assert RagSearchSpace.param_node_label("retriever__module_type") == "retriever"
    assert RagSearchSpace.param_node_label("retriever__bm25__top_k") == "retriever"
    assert RagSearchSpace.param_node_label("pipeline__quant_bits") == "pipeline"
    assert RagSearchSpace.param_node_label("embedding_model") == "embedding_model"


def test_param_leaf_name():
    assert RagSearchSpace.param_leaf_name("retriever__bm25__top_k") == "top_k"
    assert RagSearchSpace.param_leaf_name("retriever__component") == "component"
    assert RagSearchSpace.param_leaf_name("embedding_model") == "embedding_model"


# ---------------------------------------------------------------------------
# Hierarchy comes from Ax (ss.is_hierarchical via ChoiceParameter.dependents).
# ---------------------------------------------------------------------------

def test_real_rag_space_is_hierarchical():
    a = _rag_ssops()
    assert a.ss.is_hierarchical
    # at least one ChoiceParameter declares dependents (the gating source) —
    # read from Ax directly, not from a rag-stack cache.
    gated = [n for n, p in a.ss.parameters.items() if getattr(p, "is_hierarchical", False)]
    assert gated, "expected gated ChoiceParameters in the RAG space"


# ---------------------------------------------------------------------------
# params_under — node-label / prefix expansion.
# ---------------------------------------------------------------------------

def test_params_under_prefix_semantics():
    a = _rag_ssops()
    for label in _top_labels(a):
        under = a.params_under(label)
        assert under, f"label {label!r} should match at least itself/children"
        assert all(n == label or n.startswith(label + SEP) for n in under)
    # non-existent label → empty set.
    assert a.params_under("definitely_not_a_label") == set()


# ---------------------------------------------------------------------------
# decode_to_config — tensor row → active-only nested config.
# ---------------------------------------------------------------------------

def test_decode_to_config_active_only_nested():
    a = _rag_ssops()
    torch.manual_seed(42)
    x = a.snap_to_grid(torch.rand(a.dim).double())
    cfg = a.decode_to_config(x)
    assert isinstance(cfg, dict) and cfg, "decode_to_config must return a non-empty dict"

    # Every leaf in the nested config must correspond to an ACTIVE param —
    # inactive (gated-off) params must not leak into the decoded config.
    flat = a.decode_flat(x)
    active_leaves = {
        RagSearchSpace.param_leaf_name(n) for n in a.active_params(flat)
    }

    def _leaves(d):
        for v in d.values():
            if isinstance(v, dict):
                yield from _leaves(v)
            else:
                yield True

    n_decoded_leaves = sum(1 for _ in _leaves(cfg))
    assert n_decoded_leaves <= len(a.active_params(flat))
    assert active_leaves, "expected some active params"


# ---------------------------------------------------------------------------
# clamp — inactive dims canonicalize to INACTIVE_DEFAULT (hierarchy invariant).
# ---------------------------------------------------------------------------

def test_clamp_canonicalizes_inactive_dims():
    a = _rag_ssops()
    torch.manual_seed(7)
    x = a.snap_to_grid(torch.rand(1, a.dim).double())
    decoded = a.decode_flat(x[0])
    active = a.active_params(decoded)
    inactive = [i for i, n in enumerate(a.param_names) if n not in active]
    assert inactive, "RAG space should have gated-off (inactive) dims for a random draw"

    cx = a.clamp(x)
    for i in inactive:
        assert cx[0, i].item() == INACTIVE_DEFAULT, (
            f"inactive dim {a.param_names[i]!r} not canonicalized by clamp"
        )

    # Two tensors differing ONLY in an inactive dim must clamp to the SAME row
    # (functionally-identical configs → one tensor; no phantom GP novelty). With
    # the Ax-native active set this holds for EVERY inactive dim — including an
    # inactive *parent* choice: its gated-off subtree stays inactive whatever its
    # value, so flipping it cannot shift the active set (the nested-hierarchy case
    # the old single-pass active_params got wrong, now fixed).
    x2 = x.clone()
    j = inactive[0]
    x2[0, j] = 1.0 - x2[0, j]
    assert torch.allclose(a.clamp(x), a.clamp(x2))


# ---------------------------------------------------------------------------
# build_name_index — reverse lookup (nested-key path → flat name).
# ---------------------------------------------------------------------------

def test_build_name_index_roundtrip():
    a = _rag_ssops()
    idx = a.build_name_index()
    # Every flat name is reachable from its index key.
    for name in a.param_names:
        key = a._index_key(name.split(SEP))
        assert key in idx, f"{name!r} (key {key}) missing from name index"
    # Depth-2 names map back to themselves.
    for name in a.param_names:
        parts = name.split(SEP)
        if len(parts) == 2:
            assert idx[(parts[0], parts[1])] == name


# ---------------------------------------------------------------------------
# Co-mode (rag_dataflow: sequential+react) gating — the dataflow knob is a
# FLAT knob whose dependents gate ONLY the mode-specific params; shared stages
# (retrieval / reranker / generator) keep one set of always-active dims.
# ---------------------------------------------------------------------------

def test_comode_rag_dataflow_gates_mode_specific_params():
    a = _rag_ssops()
    em = next(n for n in a.param_names if n.endswith(SEP + "rag_dataflow"))
    p = a.ss.parameters[em]
    assert getattr(p, "is_hierarchical", False), "rag_dataflow must gate"
    dep = p.dependents
    assert set(dep) == {"sequential", "react"}
    # react branch: only the react-only pipeline knobs (max_iter).
    assert dep["react"], "react branch should carry max_iter"
    assert all(
        RagSearchSpace.param_leaf_name(n) == "max_iter" for n in dep["react"]
    )
    # sequential branch: exactly the sequential-only stage heads.
    seq_stages = {n.split(SEP)[2] for n in dep["sequential"]}
    assert seq_stages == {"query_expansion", "passage_compressor", "prompt_maker"}
    # Shared stages must NOT hang off the mode knob (react supports an
    # OPTIONAL reranker — its head stays active in both modes).
    for n in dep["sequential"] + dep["react"]:
        for shared in ("passage_reranker", "semantic_retrieval", "generator"):
            assert shared not in n


def test_comode_react_resolution_strips_sequential_only_stages():
    a = _rag_ssops()
    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    from rag_stack.rag_ir.workflow_schema import dataflow_type_of

    react_over = {
        "rag_dataflow": "react",
        "react": {"max_iter": 5},
        "passage_reranker": {"enabled": "true", "component": "monot5", "top_k": 4},
        "semantic_retrieval": {"vectordb": "faiss_hnsw_index", "nprobe": 8,
                               "ef_search": 64, "top_k": 8},
        "generator": {"model": "Qwen/Qwen2.5-7B-Instruct", "temperature": 0.0},
    }
    rt, nl = a.resolve_pipeline_block(cfg, {"pipeline": {"sequential": react_over}})
    stages = [n["stage"] for line in nl for n in line["nodes"]]
    # retrieval + OPTIONAL reranker + generator survive; QE / compressor /
    # prompt_maker are stripped (the cost model must not provision them).
    assert stages == ["semantic_retrieval", "passage_reranker", "generator"], stages
    assert rt["rag_dataflow"] == "react" and rt["max_iter"] == 5
    assert dataflow_type_of({"pipeline_runtime": rt}) == "agentic"

    seq_over = {
        "rag_dataflow": "sequential",
        "query_expansion": {"enabled": "false"},
        "passage_reranker": {"enabled": "false"},
        "passage_compressor": {"enabled": "false"},
        "prompt_maker": {"component": "fstring"},
        "semantic_retrieval": {"vectordb": "faiss_ivf_index", "nprobe": 8,
                               "ef_search": 32, "top_k": 8},
        "generator": {"model": "Qwen/Qwen2.5-1.5B-Instruct", "temperature": 0.0},
    }
    rt2, nl2 = a.resolve_pipeline_block(cfg, {"pipeline": {"sequential": seq_over}})
    stages2 = [n["stage"] for line in nl2 for n in line["nodes"]]
    assert stages2 == ["semantic_retrieval", "prompt_maker", "generator"], stages2
    # react-only knob must NOT leak into a sequential trial's runtime — and a
    # sequential co-mode trial must dispatch to the SEQUENTIAL engine (the
    # historical presence test misrouted it to agentic).
    assert "max_iter" not in rt2
    assert dataflow_type_of({"pipeline_runtime": rt2}) == "sequential"


# ---------------------------------------------------------------------------
# Choice-level encoding geometry — ordered numeric choices are VALUE-spaced,
# matching what Ax's BO_MIXED chain hands the GP (raw values → digest bounds →
# Normalize), so agent and ax_qnehvi_mixedgp share kernel geometry on ordinal
# knobs. Unordered choices stay index-spaced (Ax integer codes; spacing is
# inert under the Hamming kernel). encode / snap / decode share one grid.
# ---------------------------------------------------------------------------

def test_ordered_numeric_choice_value_spacing_matches_ax_digest():
    from ax.core.parameter import ChoiceParameter, ParameterType
    from ax.core.search_space import SearchSpace
    from rag_stack.search_space.ax_search_space_ops import AxSearchSpaceOps

    p = ChoiceParameter(
        "system__batch_size_request", ParameterType.INT,
        [1, 8, 32, 64], is_ordered=True, sort_values=True,
    )
    u = ChoiceParameter(
        "retrieval__component", ParameterType.STRING,
        ["bm25", "vector", "hybrid"], is_ordered=False, sort_values=False,
    )
    a = AxSearchSpaceOps(SearchSpace(parameters=[p, u]))

    # Ordered numeric → (v - lo)/(hi - lo): the digest-normalized geometry.
    assert a.choice_grid("system__batch_size_request") == [0.0, 7 / 63, 31 / 63, 1.0]
    # Unordered → index spacing j/(n-1).
    assert a.choice_grid("retrieval__component") == [0.0, 0.5, 1.0]

    # encode lands exactly on the grid coordinate of each level.
    for v, coord in zip([1, 8, 32, 64], [0.0, 7 / 63, 31 / 63, 1.0]):
        x = a.encode_arm_params(
            {"system__batch_size_request": v, "retrieval__component": "bm25"}
        )
        assert abs(x[0].item() - coord) < 1e-12

    # snap projects by VALUE distance: 0.4 is nearer 32 (31/63≈0.49) than
    # 8 (7/63≈0.11) — the old index grid snapped 0.4 to level 1 (=8).
    snapped = a.snap_to_grid(torch.tensor([[0.4, 0.6]]).double())
    assert abs(snapped[0, 0].item() - 31 / 63) < 1e-12
    assert a.decode_flat(snapped[0])["system__batch_size_request"] == 32


def test_choice_encode_snap_decode_roundtrip_real_space():
    from ax.core.parameter import ChoiceParameter

    a = _rag_ssops()
    checked = 0
    for name in a.param_names:
        p = a.ss.parameters[name]
        if not isinstance(p, ChoiceParameter):
            continue
        for v in p.values:
            x = a.encode_arm_params({name: v})
            xs = a.snap_to_grid(x.reshape(1, -1))
            assert a.decode_flat(xs[0])[name] == v, (name, v)
            checked += 1
    assert checked, "real space should contain ChoiceParameters"


# ---------------------------------------------------------------------------
# vectordb family gating — the dict-form selector compiled into REAL Ax
# dependents (algo_search_space.vectordb schema). Pins the fix for the
# phantom-dim problem: before, both families' knobs were always-active, so
# clamp/is_duplicate_decoded/EqualsCondition were blind to them and every
# optimizer wasted GP capacity / eval budget on inert dims.
# ---------------------------------------------------------------------------

_SEL = f"pipeline{SEP}sequential{SEP}semantic_retrieval{SEP}vectordb{SEP}vectordb"
_NPROBE = f"pipeline{SEP}sequential{SEP}semantic_retrieval{SEP}vectordb{SEP}faiss_ivf_index{SEP}nprobe"
_EFS = f"pipeline{SEP}sequential{SEP}semantic_retrieval{SEP}vectordb{SEP}faiss_hnsw_index{SEP}ef_search"
_PAR = f"system{SEP}faiss_ivf_parallel_mode"


def _full_flat(a: RagSearchSpace, **over) -> dict:
    """A full (active+inactive) flat assignment: first choice value / range
    lower everywhere, overridden by ``over``."""
    flat = {}
    for n, p in a.ss.parameters.items():
        flat[n] = p.values[0] if hasattr(p, "values") else p.lower
    flat.update(over)
    return flat


def test_vdb_selector_dependents_wiring():
    a = _rag_ssops()
    params = a.ss.parameters
    assert _SEL in params
    sel = params[_SEL]
    assert sel.is_hierarchical
    ivf = set(sel.dependents["faiss_ivf_index"])
    hnsw = set(sel.dependents["faiss_hnsw_index"])
    # nested query knobs land in their OWN family only
    assert _NPROBE in ivf and _NPROBE not in hnsw
    assert _EFS in hnsw and _EFS not in ivf
    # each family's block params gate under it
    assert any(n.startswith(f"vectordb{SEP}faiss_ivf_index{SEP}") for n in ivf)
    assert any(n.startswith(f"vectordb{SEP}faiss_hnsw_index{SEP}") for n in hnsw)
    assert not any("faiss_hnsw" in n for n in ivf)
    # pq children keep index_type as their single parent (no double-gating)
    assert not any(f"{SEP}pq{SEP}" in n for n in ivf)
    # system-section knobs stay UNGATED even when family-specific in effect
    # (faiss_ivf_parallel_mode is IVF-only at runtime): algo selectors must not
    # gate system dims (layering), and _resolve_system_retrieval scalarify-
    # raises on a swept knob absent from the decoded config — gating it would
    # invalidate every HNSW trial.
    if _PAR in params:
        assert _PAR not in ivf and _PAR not in hnsw


def test_vdb_selector_active_params_truthful():
    a = _rag_ssops()
    active_ivf = a.active_params(_full_flat(a, **{_SEL: "faiss_ivf_index"}))
    assert _NPROBE in active_ivf and _EFS not in active_ivf
    assert not any(n.startswith(f"vectordb{SEP}faiss_hnsw_index{SEP}") for n in active_ivf)
    active_hnsw = a.active_params(_full_flat(a, **{_SEL: "faiss_hnsw_index"}))
    assert _EFS in active_hnsw and _NPROBE not in active_hnsw
    assert not any(n.startswith(f"vectordb{SEP}faiss_ivf_index{SEP}") for n in active_hnsw)
    # the system knob is active on BOTH branches (never family-gated)
    if _PAR in a.ss.parameters:
        assert _PAR in active_ivf and _PAR in active_hnsw


def test_vdb_phantom_pair_clamps_and_dedups():
    """Two tensors differing ONLY in the dead family's knob → one canonical
    tensor (GP de-noising for agent/mixedgp) and a detected duplicate
    (eval-budget protection for every optimizer)."""
    a = _rag_ssops()
    x = a.snap_to_grid(torch.rand(1, a.dim).double())
    i_sel = a.param_names.index(_SEL)
    grid = a.choice_grid(_SEL)
    values = list(a.ss.parameters[_SEL].values)
    x[0, i_sel] = grid[values.index("faiss_ivf_index")]
    j = a.param_names.index(f"vectordb{SEP}faiss_hnsw_index{SEP}M")
    x2 = x.clone()
    x2[0, j] = 1.0 - x2[0, j]
    assert torch.allclose(a.clamp(x), a.clamp(x2))
    assert a.is_duplicate_decoded(x2, x)


def test_vdb_selector_resolve_prunes_and_flattens():
    """resolve_pipeline_block + resolve_vectordbs: the runtime config carries
    ONLY the referenced family's block, and the module gets the chosen
    family's query knob flat (the other family's knob absent)."""
    import copy

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    a = RagSearchSpace.from_yaml(cfg)
    for fam, live, dead in (
        ("faiss_ivf_index", "nprobe", "ef_search"),
        ("faiss_hnsw_index", "ef_search", "nprobe"),
    ):
        flat = _full_flat(a, **{_SEL: fam})
        decoded = a.decode_to_nested({n: flat[n] for n in a.active_params(flat)})
        config = copy.deepcopy(cfg)
        runtime, node_lines = a.resolve_pipeline_block(cfg, decoded)
        config["pipeline_runtime"] = runtime
        config["node_lines"] = node_lines
        a.resolve_vectordbs(config, decoded)
        assert [b["name"] for b in config["vectordb"]] == [fam]
        for b in config["vectordb"]:
            assert not any(isinstance(v, list) for v in b.values()), b
        mod = next(
            m for nl in node_lines for n in nl["nodes"]
            if n["stage"] == "semantic_retrieval" for m in n["modules"]
        )
        assert mod["vectordb"] == fam
        assert live in mod and not isinstance(mod[live], (list, dict))
        assert dead not in mod


def test_vdb_smac_configspace_gets_family_conditions():
    """The smac ConfigSpace conversion turns the selector dependents into
    EqualsCondition edges — the RF never sees the dead family's dims."""
    try:
        from rag_stack.optimizer.smac.configspace import ax_to_configspace
    except Exception:  # smac extras not installed in every env
        import pytest
        pytest.skip("smac configspace unavailable")
    a = _rag_ssops()
    cs = ax_to_configspace(a.ss)
    conditioned = {c.child.name for c in cs.conditions}
    assert _NPROBE in conditioned and _EFS in conditioned
    assert any(n.startswith(f"vectordb{SEP}faiss_hnsw_index{SEP}") for n in conditioned)


def test_validator_rejects_old_vdb_shapes():
    from rag_stack.config_validator import ConfigValidator

    with open(_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # (a) top-level vectordb: → targeted error
    import copy
    bad = copy.deepcopy(cfg)
    bad["vectordb"] = bad["algo_search_space"].pop("vectordb")
    joined = "\n".join(ConfigValidator(bad).validate())
    assert "algo_search_space.vectordb" in joined

    # (b) multi-family LIST selector → dict-form error
    bad = copy.deepcopy(cfg)
    for nl in bad["algo_search_space"]["pipeline"]["node_lines"]:
        for node in nl["nodes"]:
            if node["stage"] != "semantic_retrieval":
                continue
            for mod in node["modules"]:
                sel = mod["vectordb"]
                mod["vectordb"] = list(sel)
    joined = "\n".join(ConfigValidator(bad).validate())
    assert "dict form" in joined

    # (c) flat query knob alongside a dict selector → nesting error
    bad = copy.deepcopy(cfg)
    for nl in bad["algo_search_space"]["pipeline"]["node_lines"]:
        for node in nl["nodes"]:
            if node["stage"] != "semantic_retrieval":
                continue
            for mod in node["modules"]:
                mod["nprobe"] = [8, 16]
    joined = "\n".join(ConfigValidator(bad).validate())
    assert "family-owned" in joined


def test_vdb_smac_sampling_respects_gating():
    """ConfigSpace SAMPLING (what SMAC's init + ParEGO actually draw from)
    must respect the compiled family gating — no dead-family dims, no
    cross-family query knob, chained index_type→pq intact, react strips the
    sequential-only heads. This is the behavioral half of
    ``test_vdb_smac_configspace_gets_family_conditions`` (structure)."""
    try:
        from rag_stack.optimizer.smac.configspace import ax_to_configspace
    except Exception:
        import pytest
        pytest.skip("smac configspace unavailable")

    a = _rag_ssops()
    cs = ax_to_configspace(a.ss, seed=43)
    sel = _SEL
    ivf_pre = f"vectordb{SEP}faiss_ivf_index{SEP}"
    hnsw_pre = f"vectordb{SEP}faiss_hnsw_index{SEP}"
    itype = f"vectordb{SEP}faiss_ivf_index{SEP}index_type"
    pq_pre = f"vectordb{SEP}faiss_ivf_index{SEP}pq{SEP}"
    dataflow = f"pipeline{SEP}sequential{SEP}rag_dataflow"
    seen = set()
    for c in cs.sample_configuration(200):
        active = {k for k in dict(c) if c.get(k) is not None}
        fam = str(c[sel])
        seen.add(fam)
        if fam == "faiss_hnsw_index":
            assert not any(k.startswith(ivf_pre) for k in active)
            assert _NPROBE not in active and _EFS in active
        else:
            assert not any(k.startswith(hnsw_pre) for k in active)
            assert _NPROBE in active and _EFS not in active
            if str(c[itype]) == "flat":
                assert not any(k.startswith(pq_pre) for k in active)
        if str(c[dataflow]) == "react":
            assert not any(
                "query_expansion" in k or "passage_compressor" in k
                or "prompt_maker" in k for k in active
            )
    assert seen == {"faiss_ivf_index", "faiss_hnsw_index"}
