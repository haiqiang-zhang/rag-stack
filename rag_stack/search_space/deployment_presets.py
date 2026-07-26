"""Named practitioner deployment templates (``system.deployment_presets``).

A preset is a POLICY-level, pipeline-shape-agnostic deployment description::

    system:
      deployment_presets:
        collocated_tp4:  {placement: collocated, num_gpus: {generator: 4}}
        disagg_1p2d:     {placement: pd_disagg,
                          num_gpus: {generator_prefill: 1, generator_decode: 2}}
        collocated_pp2:  {placement: collocated, num_gpus: {generator: 4},
                          pipeline_parallel: {generator: 2}}
      system_design_space:
        deployment_preset: [collocated_tp4, disagg_1p2d]   # → categorical dim

At resolve time :func:`expand_preset` compiles the chosen template — for the
ACTUAL resolved pipeline shape (react strips the reranker; optional nodes may
be off) — into the raw measured dims vocabulary (``cut_after_<stage>`` /
``num_gpus_<engine|stage>`` / ``pipeline_parallel_*`` + ``available_gpus``),
which then feeds ``PerformanceContext.resolve_system_config(
system_dims_override=...)`` so the ONE existing layout builder emits the
deployment schema (no parallel serializer to drift).

Semantics:

* ``placement: collocated`` — every GPU stage in one resource group (all cut
  bits 0). ``pd_disagg`` — a single cut after ``generator_prefill``: the
  generator becomes a 1P1D pair and every auxiliary stage (encoder, reranker,
  query-expansion, compressor) rides the prefill-side group, 1 slot each
  unless sized explicitly. ``disaggregated`` — every GPU stage its own
  resource group (all cut bits 1): each aux stage gets its own GPU(s); this
  is the shape RAGO's ``disaggregated`` placement policy sweeps natively.
* ``num_gpus`` keys are engine names (``generator``) or role-qualified
  PerformanceStage names (``generator_prefill``) for asymmetric PD. Values
  MUST be powers of two — the CM sweep vocabulary / GenZ calibration only
  covers power-of-two chip counts, so anything else would be priced in
  never-validated territory.
* TP is DERIVED (chips / pp), never declared. ``pipeline_parallel`` values
  must divide the corresponding chip count.
* Stages absent from the current shape are silently dropped; a preset that
  cannot fit the current shape within the ``cm_search_space`` GPU bounds
  raises :class:`DeploymentPresetError` — the trial is penalized like a
  measured ``TrialInvalid``, never silently projected onto a different
  deployment (projection would alias two categorical values onto one
  deployment and poison the surrogate).

This module must stay import-light (pure geometry over
:mod:`rag_stack.search_space.placement`): it is imported by the search-space
builder and the validator — never import the measured runtime here.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Mapping, Tuple

from rag_stack.runtime_parallelism import RERANKER_MEASURED_DP_COMPONENTS
from rag_stack.search_space.placement import (
    cut_after_key,
    derive_layout,
    measured_gpu_stages,
    num_gpus_key,
    possible_measured_gpu_stage_lists,
    stage_engine,
)


logger = logging.getLogger("RAG-Stack")


class DeploymentPresetError(ValueError):
    """A preset cannot be realized for the current pipeline shape/budget.

    The cost-model analog of the measured path's ``TrialInvalid``: raised at
    trial time, caught by the controller, penalized — never retried. (A
    deliberate separate class: importing ``TrialInvalid`` would drag the
    measured vLLM package into cost-model runs.)
    """


VALID_PLACEMENTS = ("collocated", "pd_disagg", "disaggregated")
_PRESET_KEYS = frozenset({"placement", "num_gpus", "pipeline_parallel", "aux"})
VALID_AUX_MODES = ("ride", "left")


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _as_int_map(name: str, key: str, value: object) -> Dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            f"[system.deployment_presets.{name}.{key}] must be a mapping "
            f"(engine or role-qualified stage → int), got {type(value).__name__}."
        )
    out: Dict[str, int] = {}
    for k, v in value.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            raise ValueError(
                f"[system.deployment_presets.{name}.{key}.{k}] must be an "
                f"integer, got {v!r}."
            ) from None
    return out


def parse_deployment_presets(system: dict) -> Dict[str, dict]:
    """Normalize + structurally validate ``system.deployment_presets``.

    Raises ``ValueError`` with a config-path-prefixed message on any
    structural problem (the validator surfaces these verbatim). Shape/budget
    fit is NOT checked here — that needs a pipeline; see
    :func:`preset_fits_stages` / :func:`preset_fits_some_shape`.
    """
    block = (system or {}).get("deployment_presets")
    if block is None:
        return {}
    if not isinstance(block, Mapping) or not block:
        raise ValueError(
            "[system.deployment_presets] must be a non-empty mapping of "
            "preset name → {placement, num_gpus, pipeline_parallel}."
        )
    presets: Dict[str, dict] = {}
    for name, spec in block.items():
        name = str(name)
        if not isinstance(spec, Mapping):
            raise ValueError(
                f"[system.deployment_presets.{name}] must be a mapping, "
                f"got {type(spec).__name__}."
            )
        unknown = sorted(set(spec) - _PRESET_KEYS)
        if unknown:
            raise ValueError(
                f"[system.deployment_presets.{name}] unknown keys {unknown} "
                f"(allowed: {sorted(_PRESET_KEYS)}). TP is derived "
                "(chips / pipeline_parallel), never declared."
            )
        placement = str(spec.get("placement", "")).lower()
        if placement not in VALID_PLACEMENTS:
            raise ValueError(
                f"[system.deployment_presets.{name}.placement] "
                f"{spec.get('placement')!r} is not one of {VALID_PLACEMENTS}."
            )
        num_gpus = _as_int_map(name, "num_gpus", spec.get("num_gpus"))
        for key, n in num_gpus.items():
            if not _is_power_of_two(n):
                raise ValueError(
                    f"[system.deployment_presets.{name}.num_gpus.{key}] "
                    f"{n} is not a power of two — the CM sweep vocabulary / "
                    "GenZ calibration only covers power-of-two chip counts."
                )
            if key == "semantic_retrieval" and n != 1:
                raise ValueError(
                    f"[system.deployment_presets.{name}.num_gpus."
                    "semantic_retrieval] must be 1: the GPU-resident work is "
                    "the single-device query encoder (FAISS search is CPU)."
                )
        if placement == "collocated":
            roles = {
                k: v for k, v in num_gpus.items()
                if k.endswith("_prefill") or k.endswith("_decode")
            }
            if len(set(roles.values())) > 1:
                raise ValueError(
                    f"[system.deployment_presets.{name}.num_gpus] collocated "
                    f"placement puts prefill+decode in ONE group — asymmetric "
                    f"role counts {roles} are contradictory; use pd_disagg."
                )
        pp = _as_int_map(name, "pipeline_parallel", spec.get("pipeline_parallel"))
        for key, p in pp.items():
            if p < 1:
                raise ValueError(
                    f"[system.deployment_presets.{name}.pipeline_parallel."
                    f"{key}] must be >= 1, got {p}."
                )
            if placement == "collocated" and (
                key.endswith("_prefill") or key.endswith("_decode")
            ):
                raise ValueError(
                    f"[system.deployment_presets.{name}.pipeline_parallel."
                    f"{key}] role-qualified PP needs pd_disagg — a collocated "
                    "(unified) engine has ONE pipeline split; key it by the "
                    "engine name."
                )
            chips = num_gpus.get(key)
            if chips is None and (
                key.endswith("_prefill") or key.endswith("_decode")
            ):
                # role pp may size against the engine chips
                chips = num_gpus.get(key.rsplit("_", 1)[0])
            if chips is None:
                chips = 1
            if chips % p:
                raise ValueError(
                    f"[system.deployment_presets.{name}.pipeline_parallel."
                    f"{key}] pp={p} does not divide the chip count {chips} "
                    "(TP = chips / pp must be an integer)."
                )
        aux = str(spec.get("aux", "ride")).lower()
        if aux not in VALID_AUX_MODES:
            raise ValueError(
                f"[system.deployment_presets.{name}.aux] {spec.get('aux')!r} "
                f"is not one of {VALID_AUX_MODES}."
            )
        if aux == "left" and placement == "disaggregated":
            raise ValueError(
                f"[system.deployment_presets.{name}] aux: left is redundant "
                "with placement: disaggregated (every stage already gets its "
                "own group) — drop one of the two."
            )
        presets[name] = {
            "placement": placement,
            "num_gpus": num_gpus,
            "pipeline_parallel": pp,
            "aux": aux,
        }
    return presets


def _aux_engine_blocks(stages: List[str]) -> List[Tuple[str, List[str]]]:
    """Ordered non-generator engines with their (contiguous) stage lists.

    Raises if an aux stage appears AFTER a generator stage — the left-family
    compile cuts the aux block off the generator's cards, which requires the
    pipeline-order contiguity that every current taxonomy shape satisfies."""
    blocks: List[Tuple[str, List[str]]] = []
    seen_generator = False
    for s in stages:
        e = stage_engine(s)
        if e == "generator":
            seen_generator = True
            continue
        if seen_generator:
            raise DeploymentPresetError(
                f"aux stage {s!r} follows the generator in pipeline order — "
                "the left-family compile assumes a contiguous aux block."
            )
        if blocks and blocks[-1][0] == e:
            blocks[-1][1].append(s)
        else:
            blocks.append((e, [s]))
    return blocks


def _compile_left(
    preset: dict,
    stages: List[str],
    present_engines: List[str],
    budget: int,
    per_stage_cap: int,
    name: str,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Adaptive left-family compile → (cut_bits, num_gpus lookup).

    Rule (documented in docs/config_schema.md): the generator takes its declared
    power-of-two skeleton; the PRESENT aux engines are cut into
    ``min(#aux, leftover)`` contiguous single-card groups on the remaining
    cards (time-shared within a group); any spare card then grows the
    generator in power-of-two steps (pd_disagg: double decode, then prefill;
    collocated: double the unified engine) while it fits ``budget`` and
    ``max_num_gpus_per_stage``. A spare that nothing can absorb idles and is
    logged loudly.
    """
    placement = preset["placement"]
    declared = preset["num_gpus"]
    if placement == "pd_disagg":
        if "generator_prefill" not in stages or "generator_decode" not in stages:
            raise DeploymentPresetError(
                f"deployment_preset {name!r}: pd_disagg needs the generator's "
                f"prefill+decode pair; stages={stages}."
            )
        n_pf = int(declared.get("generator_prefill", declared.get("generator", 1)))
        n_dec = int(declared.get("generator_decode", declared.get("generator", 1)))
    else:  # collocated skeleton
        n_gen = int(declared.get("generator", 1))
        n_pf = n_dec = 0

    blocks = _aux_engine_blocks(stages)
    gen_total = (n_pf + n_dec) if placement == "pd_disagg" else n_gen
    leftover = budget - gen_total
    if leftover <= 0:
        raise DeploymentPresetError(
            f"deployment_preset {name!r}: aux=left needs leftover cards, but "
            f"the generator skeleton already takes {gen_total} of {budget}."
        )

    n_groups = min(len(blocks), leftover)
    spare = leftover - n_groups

    # spare growth: power-of-two doubling into the unused cards.
    if placement == "pd_disagg":
        while spare >= n_dec and 2 * n_dec <= per_stage_cap:
            spare -= n_dec
            n_dec *= 2
        while spare >= n_pf and 2 * n_pf <= per_stage_cap:
            spare -= n_pf
            n_pf *= 2
    else:
        while spare >= n_gen and 2 * n_gen <= per_stage_cap:
            spare -= n_gen
            n_gen *= 2
    if spare > 0:
        logger.info(
            f"deployment_preset {name!r}: {spare} card(s) idle for this shape "
            f"(aux engines: {[e for e, _ in blocks]}; the generator cannot "
            "grow further within power-of-two / per-stage bounds)."
        )

    # cut bits: boundaries between aux groups, after the aux block, and the
    # generator's own PD cut. Aux engines distribute contiguously and as
    # evenly as possible into the groups.
    cut_bits = {cut_after_key(s): 0 for s in stages[:-1]}
    if n_groups:
        base, extra = divmod(len(blocks), n_groups)
        idx = 0
        for gi in range(n_groups):
            take = base + (1 if gi < extra else 0)
            idx += take
            last_stage = blocks[idx - 1][1][-1]
            cut_bits[cut_after_key(last_stage)] = 1   # group/block boundary
    elif blocks:
        # more spare-consuming growth than groups can never strand aux:
        # n_groups == 0 means there were no leftover cards at all, which the
        # leftover<=0 gate above already rejected.
        raise DeploymentPresetError(
            f"deployment_preset {name!r}: internal — aux present but no aux "
            "groups allocated."
        )
    if placement == "pd_disagg":
        cut_bits[cut_after_key("generator_prefill")] = 1

    # num_gpus lookup (derive_layout vocabulary).
    lookup: Dict[str, int] = {}
    for e, _block_stages in blocks:
        lookup[e] = 1 if e == "semantic_retrieval" else int(declared.get(e, 1))
    if placement == "pd_disagg":
        lookup["generator_prefill"] = n_pf
        lookup["generator_decode"] = n_dec
        lookup["generator"] = max(n_pf, n_dec)
    else:
        lookup["generator"] = n_gen
    return cut_bits, lookup


def expand_preset(
    preset: dict,
    stages: List[str],
    available_gpus: List[str],
    cm_bounds: dict,
    *,
    name: str = "?",
    reranker_components: Iterable[str] | None = None,
) -> dict:
    """Compile one parsed preset into the raw measured dims vocabulary for
    the given (resolved) stage list, with a STRICT budget check.

    Returns the ``system_dims_override`` dict: ``cut_after_<stage>`` bits,
    ``num_gpus_<engine>`` (+ role-qualified ``num_gpus_<stage>`` when
    asymmetric), ``pipeline_parallel_*`` and ``available_gpus``. Raises
    :class:`DeploymentPresetError` when the preset cannot be realized —
    validity here is the ONLY budget gate on the cost-model path (there is
    no ``prepare_trial``), and it guarantees the downstream
    ``project_layout_to_slot_bounds`` takes its no-op early return: what the
    categorical names is exactly what gets priced.
    """
    if not stages:
        raise DeploymentPresetError(
            f"deployment_preset {name!r}: pipeline has no GPU stages."
        )

    # A named preset is an exact deployment contract: never silently rewrite
    # ``passage_reranker: 2`` to one card after the component branch resolves.
    # Raw/config-time geometry callers omit component context; the production
    # trial resolver always passes the resolved modules.  Exactly one selected
    # component must own measured DP support, otherwise the sampled
    # preset/component pair is an invalid trial.
    if "passage_reranker" in stages and reranker_components is not None:
        components = tuple(str(value) for value in reranker_components if value)
        reranker_gpus = int(preset["num_gpus"].get("passage_reranker", 1))
        if reranker_gpus > 1 and (
            len(components) != 1
            or components[0] not in RERANKER_MEASURED_DP_COMPONENTS
        ):
            component_label = (
                components[0] if len(components) == 1 else list(components)
            )
            raise DeploymentPresetError(
                f"deployment_preset {name!r}: passage_reranker declares "
                f"{reranker_gpus} GPUs, but resolved component "
                f"{component_label!r} has no measured data-parallel runtime; "
                "the preset is exact and will not be rewritten to one GPU"
            )
    placement = preset["placement"]
    present_engines = []
    for s in stages:
        e = stage_engine(s)
        if e not in present_engines:
            present_engines.append(e)

    lo = max(1, int(cm_bounds.get("min_num_gpus", 1) or 1))
    hi = min(
        int(cm_bounds.get("max_num_gpus", len(available_gpus))
            or len(available_gpus)),
        len(available_gpus),
    )
    per_stage_cap = int(cm_bounds.get("max_num_gpus_per_stage", hi) or hi)

    if preset.get("aux", "ride") == "left":
        # ADAPTIVE left-family compile: aux engines get the generator's
        # leftover cards (contiguous single-card groups, time-shared within
        # a group); unused spare grows the generator in power-of-two steps.
        cut_bits, lookup = _compile_left(
            preset, stages, present_engines, hi, per_stage_cap, name,
        )
    else:
        # 1. cut bits — collocated: all zero; pd_disagg: split decode off after
        #    the generator's prefill (aux stages ride the prefill-side group);
        #    disaggregated: every stage its own group.
        if placement == "disaggregated":
            cut_bits = {cut_after_key(s): 1 for s in stages[:-1]}
        else:
            cut_bits = {cut_after_key(s): 0 for s in stages[:-1]}
            if placement == "pd_disagg":
                if "generator_prefill" not in stages or "generator_decode" not in stages:
                    raise DeploymentPresetError(
                        f"deployment_preset {name!r}: pd_disagg needs the generator's "
                        f"prefill+decode pair; stages={stages}."
                    )
                cut_bits[cut_after_key("generator_prefill")] = 1

        # 2. num_gpus — engine keys pass through; role-qualified keys land under
        #    the stage's full name (derive_layout prefers them); absent
        #    stages/engines are silently dropped.
        lookup = {}      # derive_layout vocabulary (bare names)
        for key, n in preset["num_gpus"].items():
            if key in present_engines or key in stages:
                lookup[key] = int(n)
        for e in present_engines:
            role_vals = [
                lookup[s] for s in stages
                if s in lookup and stage_engine(s) == e
            ]
            if e not in lookup and role_vals:
                lookup[e] = max(role_vals)   # compat fallback for engine readers
            lookup.setdefault(e, 1)

    # 3. strict budget check — THE gate.
    layout = derive_layout(stages, cut_bits, lookup, available_gpus)
    total = int(layout.get("total_slots", 0) or 0)
    if not lo <= total <= hi:
        raise DeploymentPresetError(
            f"deployment_preset {name!r}: needs {total} GPU slot(s) for "
            f"stages={stages}, outside budget [{lo}, {hi}] "
            f"(available={len(available_gpus)})."
        )
    from rag_stack.search_space.placement import _stage_num_gpus
    for s in stages:
        n = _stage_num_gpus(s, lookup)
        if n > per_stage_cap:
            raise DeploymentPresetError(
                f"deployment_preset {name!r}: stage {s} sized {n} exceeds "
                f"max_num_gpus_per_stage={per_stage_cap}."
            )

    # 4. emit the dims-override vocabulary.
    dims: Dict[str, object] = {"available_gpus": list(available_gpus)}
    dims.update(cut_bits)
    for key, n in lookup.items():
        dims[num_gpus_key(key)] = int(n)
    for key, p in preset["pipeline_parallel"].items():
        if key.endswith("_prefill") or key.endswith("_decode"):
            base, role = key.rsplit("_", 1)
            if base not in present_engines:
                continue
            dims[f"pipeline_parallel_{base}_{role}"] = int(p)
            # Pin the sibling role explicitly (declared value or 1) — the
            # resolver's role→engine-key fallback must never leak one
            # role's PP onto the other.
            other = "decode" if role == "prefill" else "prefill"
            dims.setdefault(
                f"pipeline_parallel_{base}_{other}",
                int(preset["pipeline_parallel"].get(f"{base}_{other}", 1)),
            )
        elif key in present_engines:
            dims[f"pipeline_parallel_{key}"] = int(p)
    return dims


def preset_fits_stages(
    preset: dict,
    stages: List[str],
    available_gpus: List[str],
    cm_bounds: dict,
) -> bool:
    """True when :func:`expand_preset` succeeds for this stage list."""
    try:
        expand_preset(preset, stages, available_gpus, cm_bounds)
    except DeploymentPresetError:
        return False
    return True


def preset_shape_fit_report(
    preset: dict,
    config: dict,
    available_gpus: List[str],
    cm_bounds: dict,
) -> tuple[int, int]:
    """(#shapes the preset fits, #reachable shapes) over every optional-node
    on/off pipeline shape — the validator's config-time coverage check."""
    shapes = possible_measured_gpu_stage_lists(config)
    fits = sum(
        1 for stages in shapes
        if preset_fits_stages(preset, stages, available_gpus, cm_bounds)
    )
    return fits, len(shapes)


def resolved_stages(resolved_config: dict) -> List[str]:
    """The resolve-time stage list (only enabled nodes) — thin re-export so
    callers need one import."""
    return measured_gpu_stages(resolved_config)
