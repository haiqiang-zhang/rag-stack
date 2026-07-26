"""Canonical GPU-placement taxonomy for measured mode.

Single source of truth shared by:
  * ``search_space.system.extract_system_design_params`` — emit
    ``system__placement_<stage>`` optimizer dims;
  * ``PerformanceContext.resolve_system_config`` — read them back into the
    deployment ``system_config``;
  * ``MeasuredEvaluator._inject_cache_and_devices`` / ``_collect_performance_metrics``
    — assign each node's device + report collocation;
  * ``VllmDeploymentManager`` — place the generator (and aux QE) vLLM.

Placement keys are named **by pipeline ``stage``** (``placement_generator``,
``placement_passage_reranker`` …) — NOT by backend/role (the old
``placement_vllm`` / ``placement_reranker`` mix). One taxonomy, no
stage→component indirection.

Stage vocabulary comes from the central taxonomy (:mod:`rag_stack.taxonomy` —
a leaf module, so no import cycles): placement partitions operate on
PerformanceStage names; engines are keyed by their owning user stage.
"""
from __future__ import annotations

import itertools
import math
import re
from typing import Dict, Iterable, List, Optional, Tuple

from rag_stack.runtime_parallelism import RERANKER_MEASURED_DP_COMPONENTS

# GPU-resident pipeline stages, in canonical (display) order. Node_types
# NOT listed here run on CPU and get no placement dim — including
# ``prompt_maker``, the faiss vectordb search itself, ``passage_filter``
# (default filters are score-threshold/recency = CPU; the similarity filter
# defaults to the OpenAI embedding API), and ``hybrid_retrieval`` /
# ``lexical_retrieval`` (we only use vectordb/dense ``semantic_retrieval`` as
# the retriever — hybrid_rrf is pure CPU rank-fusion of upstream results, no
# model). This set matches the chip-allocating stages of the cost model's
# stage taxonomy (rag_stack.taxonomy; semantic_retrieval_vectorsearch→semantic_retrieval,
# prefill/decode→generator, …; passage_filter there gets 0 chips). Multi-GPU
# cardinality is constrained by the measured runtime contract below: retrieval
# encode is always one card; reranker DP is available only to its explicit
# replica-capable component allowlist.
GPU_STAGES: List[str] = [
    "generator",            # main vLLM
    "semantic_retrieval",   # query-embedding model (the only retriever we use)
    "query_expansion",      # optional: its own aux vLLM
    "passage_reranker",     # optional: cross-encoder / LLM reranker
    "passage_compressor",   # optional: ~7B compressor LLM (llmlingua)
]

def passage_reranker_components(config: dict) -> tuple[str, ...]:
    """Return declared passage-reranker components in stable order.

    Works for both a resolved config (top-level ``node_lines``; normally one
    selected module) and the raw algorithm search space (possibly several
    alternative modules).  Keeping this extraction beside placement prevents
    the measured layout builder and the CM-owned RAGO sweep from maintaining
    separate component-capability tables.
    """
    node_line_sets: List[List[dict]] = []
    if config.get("node_lines"):
        node_line_sets.append(config.get("node_lines") or [])
    else:
        from rag_stack.search_space.rag_search_space import pipeline_method_blocks

        for block in pipeline_method_blocks(config):
            if isinstance(block, dict):
                node_line_sets.append(block.get("node_lines", []) or [])

    components: List[str] = []
    for node_lines in node_line_sets:
        for node_line in node_lines:
            for node in node_line.get("nodes", []) or []:
                if node.get("stage") != "passage_reranker":
                    continue
                for module in node.get("modules", []) or []:
                    component = str(module.get("component") or "").strip()
                    if component and component not in components:
                        components.append(component)
    return tuple(components)


def reranker_has_measured_data_parallelism(
    components: Iterable[str] | None,
) -> bool:
    """Whether at least one declared reranker has a measured DP runtime.

    Raw search-space construction passes all alternatives, so ``any`` keeps
    the multi-GPU layout categories needed by a supported branch.  Resolution
    passes the single selected component and therefore prunes unsupported
    branches to one card.  Missing/unknown component context fails closed.
    """
    return any(
        str(component) in RERANKER_MEASURED_DP_COMPONENTS
        for component in (components or ())
    )

# When an (optional) node has no independently-sampled placement, inherit a
# neighbour's GPU before the default device — keeps a toggled-on optional node
# colocated with a sensible sibling instead of defaulting to cuda:0.
_PLACEMENT_FALLBACK: Dict[str, List[str]] = {
    "query_expansion": ["generator"],          # aux vLLM next to the main vLLM
    "passage_compressor": ["passage_reranker"],  # both are retrieval-side GPU models
}


def placement_key(stage: str) -> str:
    """The system_config / search-dim leaf name for a stage's placement."""
    return f"placement_{stage}"


def gpu_stages_in_pipeline(config: dict) -> List[str]:
    """GPU-resident stages actually present in the pipeline definition, in
    canonical order.

    Handles optional nodes correctly: an ``optional: true`` node is still
    *defined* in the pipeline (just toggled per trial), so it DOES get a
    placement dim; a stage entirely absent from the pipeline gets none (no
    dead optimizer dimensions).
    """
    from rag_stack.search_space.rag_search_space import pipeline_method_blocks
    present = set()
    for block in pipeline_method_blocks(config):
        if not isinstance(block, dict):
            continue
        for nl in block.get("node_lines", []) or []:
            for node in (nl.get("nodes", []) or []):
                nt = node.get("stage")
                if nt in GPU_STAGES:
                    present.add(nt)
    return [nt for nt in GPU_STAGES if nt in present]


def resolve_node_device(stage: str, placements: dict, default: str) -> str:
    """Device string for ``stage``: its own sampled ``placement_<stage>``,
    else an inherited neighbour's (fallback chain), else ``default``.

    ``placements`` is any dict carrying ``placement_<stage>`` keys — the
    decoded ``config["system"]`` sub-dict (resolver) or the flat
    ``system_config`` (evaluator).
    """
    val = placements.get(placement_key(stage))
    if val:
        return str(val)
    for parent in _PLACEMENT_FALLBACK.get(stage, []):
        val = placements.get(placement_key(parent))
        if val:
            return str(val)
    return default


# ===========================================================================
# Stage-level partition machinery (measured-mode multi-GPU, Phase 2)
#
# RAGO models each LLM as separate prefill/decode STAGES; the measured search
# space mirrors that. The GPU-resident stage list (in pipeline order) for a
# config is the unit the collocation partition operates on. P/D disaggregation
# is a CONSEQUENCE of the partition: an LLM's prefill+decode in the SAME group
# = unified engine; in DIFFERENT groups = 1P1D (PD pair). One mechanism
# (`cut_after_<stage>` boundary bits) covers both cross-component collocation
# and per-LLM P/D — exactly RAGO's adjacent-partition space.
# ===========================================================================

def stage_engine(stage: str) -> str:
    """Owning user stage (= deployment engine) of any global stage name — a
    taxonomy table lookup, no string surgery."""
    from rag_stack.taxonomy import TAXONOMY
    return TAXONOMY.user_stage_of(stage).value


def measured_gpu_stages(config: dict) -> List[str]:
    """Ordered GPU-resident PerformanceStages present in the pipeline, in
    pipeline order (so the generator's decode is always last).

    Fully derived from the taxonomy (:meth:`gpu_performance_stages_for_node` —
    the components' shapes decide the expansion, so decode-bearing-ness needs
    no special-case set): llm-shaped components expand to their
    prefill+decode pair; ``semantic_retrieval`` contributes only its GPU half
    (``semantic_retrieval_encode`` — the vector search is CPU); pruning
    compressors are the single bare unit. CPU work (prompt_maker, FAISS
    search/fetch) is excluded — it gets no chips/partition. Used at BUILD time
    over the full config (all optional nodes → maximal stage list) and at
    RESOLVE time over the resolved config (only enabled nodes → present
    subset).
    """
    from rag_stack.taxonomy import TAXONOMY

    # Accept either the build config (``pipeline[*].node_lines``) or a resolved
    # config (top-level ``node_lines`` with only enabled nodes).
    node_lines: List = []
    if config.get("node_lines"):
        node_lines = config["node_lines"]
    else:
        from rag_stack.search_space.rag_search_space import pipeline_method_blocks
        for block in pipeline_method_blocks(config):
            if isinstance(block, dict):
                node_lines += block.get("node_lines", []) or []

    stages: List[str] = []
    for nl in node_lines:
        for node in (nl.get("nodes", []) or []):
            nt = node.get("stage")
            if nt not in GPU_STAGES:
                continue  # prompt_maker / FAISS search/fetch → CPU, no GPU stage
            components = [
                str(m.get("component") or "")
                for m in (node.get("modules") or [])
                if m.get("component") in TAXONOMY.components
            ]
            for ps in TAXONOMY.gpu_performance_stages_for_node(nt, components):
                if ps.name not in stages:
                    stages.append(ps.name)
    return stages


def cut_after_key(stage: str) -> str:
    """system_config / search-dim leaf for the 'cut after <stage>' boundary bit
    (1 = start a new collocation group after this stage; 0 = keep next stage in
    the same group)."""
    return f"cut_after_{stage}"


def num_gpus_key(engine: str) -> str:
    """system_config / search-dim leaf for an engine's GPU count (TP for LLMs,
    DP-replica count for the reranker)."""
    return f"num_gpus_{engine}"


def _engine_num_gpus(engine: str, num_gpus: Dict[str, int]) -> int:
    """Concrete chip count for one placement engine.

    Forward-only engines use this value as their data-parallel replica count;
    LLM engines interpret it through their TP/PP layout records downstream.
    """
    return max(1, int(num_gpus.get(engine, 1)))


def _stage_num_gpus(stage: str, num_gpus: Dict[str, int]) -> int:
    """Concrete chip count for one PerformanceStage.

    A role-qualified key (``num_gpus_generator_prefill`` — the stage's full
    name) wins over the per-engine key (``num_gpus_generator``), which stays
    the fallback for every symmetric deployment. This is what makes
    asymmetric PD (e.g. 1 prefill chip, 2 decode chips) expressible: the
    layout enumeration (``valid_layout_options``) remains per-engine, so
    ``gpu_layout`` cardinality is unchanged — role-qualified counts enter
    only through raw dims (deployment presets, CM-selected deployments).
    """
    engine = stage_engine(stage)
    if stage != engine and stage in num_gpus:
        return max(1, int(num_gpus[stage]))
    return _engine_num_gpus(engine, num_gpus)


def derive_layout(
    stages: List[str],
    cut_bits: Dict[str, int],
    num_gpus: Dict[str, int],
    available_gpus: List[str],
) -> Dict:
    """Deterministically turn (present stage list + cut bits + per-engine
    num_gpus) into a concrete GPU layout — the measured analog of RAGO's
    physical mapping. Pure / GPU-free / unit-testable.

    Returns::

        {
          "total_slots": int,                 # Σ group sizes (TrialInvalid if > len(gpus))
          "groups": [[perf_stage, ...], ...], # the collocation partition
          "group_devices": [[gpu, ...], ...], # consecutive GPU slice per group
          "engines": {engine: {"pd_serving": "collocated_pd",
                                 "devices": [gpu, ...]}
                              | {"pd_serving": "disagg_pd",
                                 "prefill_devices": [...], "decode_devices": [...],
                                 "devices": [...]}},
          "gpu_occupants": {gpu: [engine, ...]},  # for the util rule
        }

    Group GPU-set size = max(member engines' num_gpus) (= RAGO np.max). PD
    disaggregation is DERIVED, never a knob: an LLM's prefill/decode landing in
    different groups ⇒ ``disagg_pd``; same group ⇒ ``collocated_pd``.
    """
    if not stages:
        return {"total_slots": 0, "groups": [], "group_devices": [],
                "engines": {}, "gpu_occupants": {}}

    # 1. partition via cut bits (walk in order)
    groups: List[List[str]] = [[stages[0]]]
    for prev, s in zip(stages, stages[1:]):
        if int(cut_bits.get(cut_after_key(prev), 0)):
            groups.append([s])
        else:
            groups[-1].append(s)

    # 2. group sizes (max member stage num_gpus; role-qualified keys win) +
    #    consecutive GPU layout
    cursor = 0
    group_devices: List[List[str]] = []
    for g in groups:
        size = max(_stage_num_gpus(s, num_gpus) for s in g)
        group_devices.append(list(available_gpus[cursor: cursor + size]))
        cursor += size
    total_slots = cursor

    # 3. per-engine deployment (unified if all its stages in one group; else 1P1D)
    eng_to_groupidx: Dict[str, Dict[str, int]] = {}
    for gi, g in enumerate(groups):
        for s in g:
            eng_to_groupidx.setdefault(stage_engine(s), {})[s] = gi

    # Each engine occupies its own ``num_gpus`` chips *within* its group.
    # Collocated engines OVERLAP on the group's chips (RAGO time-share), and
    # an engine smaller than its group's max does not claim the extra chips a
    # larger co-member needs. RIDERS ARE BALANCED: engines smaller than their
    # group rotate round-robin across the group's cards (in pipeline order)
    # instead of stacking on its first card — under a TP generator the group
    # runs in lockstep, so piling every aux onto card 0 makes it the straggler
    # that paces ALL cards (and the memory hotspot the vLLM KV sizing must
    # shrink around) while the other cards idle their spare time slices.
    from rag_stack.taxonomy import TAXONOMY

    engines: Dict[str, Dict] = {}
    _rider_offset: Dict[int, int] = {}

    def _take(gi: int, n: int) -> List[str]:
        devs = group_devices[gi]
        if n >= len(devs):
            return list(devs)
        start = _rider_offset.get(gi, 0)
        picked = [devs[(start + j) % len(devs)] for j in range(n)]
        _rider_offset[gi] = (start + n) % len(devs)
        return picked

    for eng, stage_gi in eng_to_groupidx.items():
        gidxs = set(stage_gi.values())
        if len(gidxs) == 1:
            # Unified engine: one chip count. With role-qualified keys the
            # stages of a same-group LLM must agree; max() tolerates a
            # config that sizes only one role.
            n = max(_stage_num_gpus(s, num_gpus) for s in stage_gi)
            gi = next(iter(gidxs))
            engines[eng] = {"pd_serving": "collocated_pd",
                            "devices": _take(gi, n)}
        else:
            pf_stage, pf_gi = next(
                (s, gi) for s, gi in stage_gi.items()
                if TAXONOMY.performance_stage(s).kind == "prefill")
            dec_stage, dec_gi = next(
                (s, gi) for s, gi in stage_gi.items()
                if TAXONOMY.performance_stage(s).kind == "decode")
            n_pf = _stage_num_gpus(pf_stage, num_gpus)
            n_dec = _stage_num_gpus(dec_stage, num_gpus)
            engines[eng] = {
                "pd_serving": "disagg_pd",
                "prefill_devices": _take(pf_gi, n_pf),
                "decode_devices": _take(dec_gi, n_dec),
            }
            engines[eng]["devices"] = (
                engines[eng]["prefill_devices"] + engines[eng]["decode_devices"]
            )

    # 4. per-GPU occupancy (for the co-residency util rule)
    gpu_occupants: Dict[str, List[str]] = {}
    for eng, info in engines.items():
        for d in info["devices"]:
            gpu_occupants.setdefault(d, []).append(eng)

    return {
        "total_slots": total_slots,
        "groups": groups,
        "group_devices": group_devices,
        "engines": engines,
        "gpu_occupants": gpu_occupants,
    }


def powers_of_two_up_to(n: int) -> List[int]:
    """Return ``[1, 2, 4, ...]`` up to ``n`` (inclusive)."""
    out: List[int] = []
    p = 1
    while p <= max(1, int(n)):
        out.append(p)
        p *= 2
    return out


def valid_layout_options(
    stages: List[str],
    available_gpus: List[str],
    min_slots: int,
    max_slots: int,
    max_num_gpus_per_stage: int,
    *,
    reranker_components: Iterable[str] | None = None,
) -> List[Dict]:
    """Enumerate concrete measured layouts that fit the configured GPU budget.

    The measured optimizer samples a compact categorical layout id; this helper
    is the single source of truth for expanding that id into the same
    ``num_gpus_*`` and ``cut_after_*`` knobs that older configs/checkpoints used
    directly. A layout is valid when its derived total slot count is within the
    configured min/max range and physically fits ``available_gpus``.
    """
    if not stages:
        return []

    hard_max = len(available_gpus)
    lo = max(0, int(min_slots))
    hi = min(int(max_slots), hard_max)
    if hi < lo:
        return []

    cap = max(1, min(int(max_num_gpus_per_stage), hi if hi > 0 else 1))
    chip_choices = powers_of_two_up_to(cap)

    engines: List[str] = []
    for s in stages:
        e = stage_engine(s)
        if e not in engines:
            engines.append(e)

    cut_keys = [cut_after_key(s) for s in stages[:-1]]
    options: List[Dict] = []
    # ``None`` keeps this low-level geometry helper backward-compatible for
    # callers that intentionally have no pipeline/component context.  Every
    # production search-space/resolution caller passes an explicit tuple;
    # an empty tuple there means unknown and is restricted to one GPU.
    reranker_dp = (
        reranker_components is None
        or reranker_has_measured_data_parallelism(reranker_components)
    )
    engine_choices = []
    for engine in engines:
        if engine == "passage_reranker" and not reranker_dp:
            # Unknown and non-DP components fail closed to the one replica the
            # measured path can actually execute.
            engine_choices.append([1])
        else:
            engine_choices.append(chip_choices)
    for nums in itertools.product(*engine_choices):
        num_gpus = dict(zip(engines, nums))
        for cuts in itertools.product([0, 1], repeat=len(cut_keys)):
            cut_bits = dict(zip(cut_keys, cuts))
            layout = derive_layout(stages, cut_bits, num_gpus, available_gpus)
            total = int(layout.get("total_slots", 0) or 0)
            if lo <= total <= hi:
                options.append(
                    {
                        "num_gpus": num_gpus,
                        "cut_bits": cut_bits,
                        "layout": layout,
                    }
                )
    return options


def project_layout_to_slot_bounds(
    stages: List[str],
    cut_bits: Dict[str, int],
    num_gpus: Dict[str, int],
    available_gpus: List[str],
    min_slots: int,
    max_slots: int,
    max_num_gpus_per_stage: int,
    layout_choice: Optional[object] = None,
    *,
    reranker_components: Iterable[str] | None = None,
) -> Tuple[Dict[str, int], Dict[str, int], Dict, bool]:
    """Return a budget-valid measured layout, preserving the raw choice if valid.

    ``layout_choice`` is a categorical value such as ``"layout_0007"``. When
    present, it selects the corresponding valid layout for the *actual* active
    stage list. When absent, the raw ``num_gpus``/``cut_bits`` tuple is used if
    already valid; otherwise the nearest valid tuple is selected.
    """
    # Normal optimizer arms are projected onto capabilities the measured
    # runtime implements.  This is candidate pruning, not a capacity model:
    # saturated backlog cannot turn a single model into replicas.  Explicit
    # fixed/manual/replay layouts are rejected earlier by the compatibility
    # validator. Semantic retrieval is no longer projected: its child owns one
    # embedding replica per selected device. Unsupported reranker components
    # remain projected to the one replica they can execute.
    effective_num_gpus = dict(num_gpus)
    if (
        "passage_reranker" in stages
        and not reranker_has_measured_data_parallelism(reranker_components)
    ):
        effective_num_gpus["passage_reranker"] = 1
    capability_projected = effective_num_gpus != num_gpus

    raw_layout = derive_layout(
        stages, cut_bits, effective_num_gpus, available_gpus,
    )
    raw_total = int(raw_layout.get("total_slots", 0) or 0)
    hard_max = len(available_gpus)
    lo = max(0, int(min_slots))
    hi = min(int(max_slots), hard_max)

    if layout_choice is None and lo <= raw_total <= hi:
        return (
            dict(cut_bits), effective_num_gpus, raw_layout,
            capability_projected,
        )

    options = valid_layout_options(
        stages,
        available_gpus,
        min_slots=lo,
        max_slots=hi,
        max_num_gpus_per_stage=max_num_gpus_per_stage,
        reranker_components=reranker_components,
    )
    if not options:
        return (
            dict(cut_bits), effective_num_gpus, raw_layout,
            capability_projected,
        )

    choice_idx = _layout_choice_index(layout_choice)
    if choice_idx is not None:
        selected = options[choice_idx % len(options)]
        changed = (
            capability_projected
            or selected["cut_bits"] != cut_bits
            or selected["num_gpus"] != effective_num_gpus
            or selected["layout"] != raw_layout
        )
        return (
            dict(selected["cut_bits"]),
            dict(selected["num_gpus"]),
            selected["layout"],
            changed,
        )

    desired_total = min(max(raw_total, lo), hi)

    def _score(option: Dict) -> tuple:
        opt_nums = option["num_gpus"]
        opt_cuts = option["cut_bits"]
        opt_total = int(option["layout"].get("total_slots", 0) or 0)
        num_dist = 0.0
        for e in set(opt_nums) | set(effective_num_gpus):
            a = max(1, int(opt_nums.get(e, 1)))
            b = max(1, int(effective_num_gpus.get(e, 1)))
            num_dist += abs(math.log2(a) - math.log2(b))
        cut_dist = sum(
            int(opt_cuts.get(k, 0) != int(cut_bits.get(k, 0)))
            for k in set(opt_cuts) | set(cut_bits)
        )
        return (abs(opt_total - desired_total), num_dist, cut_dist)

    selected = min(options, key=_score)
    return (
        dict(selected["cut_bits"]),
        dict(selected["num_gpus"]),
        selected["layout"],
        True,
    )


def possible_measured_gpu_stage_lists(config: dict) -> List[List[str]]:
    """Enumerate stage lists reachable by optional-node enable/disable choices."""
    node_line_sets: List[List[dict]] = []
    if config.get("node_lines"):
        # Resolved config: node_lines lifted to top level by resolve_config.
        node_line_sets = [config.get("node_lines") or []]
    else:
        # Raw config: algo_search_space.pipeline (new-schema parsing).
        from rag_stack.search_space.rag_search_space import pipeline_method_blocks
        for block in pipeline_method_blocks(config):
            if isinstance(block, dict):
                node_line_sets.append(block.get("node_lines", []) or [])

    seen = set()
    out: List[List[str]] = []
    for node_lines in node_line_sets:
        nodes: List[dict] = []
        optional_idxs: List[int] = []
        for nl in node_lines:
            for node in (nl.get("nodes", []) or []):
                nodes.append(node)
                if node.get("optional", False):
                    optional_idxs.append(len(nodes) - 1)

        if not optional_idxs:
            stages = measured_gpu_stages({"node_lines": node_lines})
            key = tuple(stages)
            if key not in seen:
                seen.add(key)
                out.append(stages)
            continue

        max_masks = 1 << len(optional_idxs)
        for mask in range(max_masks):
            selected = set()
            for bit, node_idx in enumerate(optional_idxs):
                if mask & (1 << bit):
                    selected.add(node_idx)
            filtered = []
            cursor = 0
            for nl in node_lines:
                kept = []
                for node in (nl.get("nodes", []) or []):
                    include = (not node.get("optional", False)) or cursor in selected
                    if include:
                        kept.append(node)
                    cursor += 1
                filtered.append({"nodes": kept})
            stages = measured_gpu_stages({"node_lines": filtered})
            key = tuple(stages)
            if key not in seen:
                seen.add(key)
                out.append(stages)
    return out


def _layout_choice_index(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    m = re.search(r"(\d+)$", str(value))
    if m:
        return int(m.group(1))
    return None
