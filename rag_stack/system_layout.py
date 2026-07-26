"""Strict helpers for the resolved deployment schema.

Placement vocabulary is PerformanceStage (``layout.performance_stages`` /
``performance_stage_order`` / ``resource_groups[].performance_stages``);
engines are keyed by their owning user stage. PD disaggregation is DERIVED
from resource-group membership — the persisted ``pd_serving`` field is a
snapshot that loaders re-derive and assert against, never independent truth.
"""

from __future__ import annotations

from typing import Any


def require_layout(system_config: dict[str, Any]) -> dict[str, Any]:
    layout = system_config.get("layout")
    if not isinstance(layout, dict):
        raise KeyError("system_config['layout'] is required")
    return layout


def request_batch(system_config: dict[str, Any]) -> int:
    if "batch_size_request" not in system_config:
        raise KeyError("system_config['batch_size_request'] is required")
    return max(1, int(system_config["batch_size_request"]))


def decode_batch(system_config: dict[str, Any]) -> int:
    """Decode-engine sequence concurrency.

    The request batch remains the cross-stage cohort size. Decode may admit a
    larger continuous-batching population; values below request are floored to
    request so measured and CM never throttle decode below the request cohort.
    """
    request = request_batch(system_config)
    raw = system_config.get("batch_size_decode")
    if raw is None:
        batching = system_config.get("batching")
        if isinstance(batching, dict):
            raw = batching.get("decode")
    if raw is None:
        return request
    return max(request, int(raw))


def dynamic_batch_timeout_s(system_config: dict[str, Any]) -> float:
    batching = system_config.get("batching")
    if not isinstance(batching, dict):
        raise KeyError("system_config['batching'] is required")
    if "dynamic_timeout_s" not in batching:
        raise KeyError("system_config['batching']['dynamic_timeout_s'] is required")
    raw = batching["dynamic_timeout_s"]
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "system_config['batching']['dynamic_timeout_s'] must be numeric"
        ) from exc


def retrieval_config(system_config: dict[str, Any]) -> dict[str, Any]:
    cfg = system_config.get("retrieval")
    if not isinstance(cfg, dict):
        raise KeyError("system_config['retrieval'] is required")
    return cfg


def vllm_engine_config(system_config: dict[str, Any], engine: str) -> dict[str, Any]:
    vllm = system_config.get("vllm")
    if not isinstance(vllm, dict):
        raise KeyError("system_config['vllm'] is required")
    engines = vllm.get("engines")
    if not isinstance(engines, dict):
        raise KeyError("system_config['vllm']['engines'] is required")
    cfg = engines.get(engine)
    if not isinstance(cfg, dict):
        raise KeyError(f"system_config['vllm']['engines'][{engine!r}] is required")
    return cfg


def vllm_max_num_seqs(system_config: dict[str, Any], engine: str) -> int:
    cfg = vllm_engine_config(system_config, engine)
    if "max_num_seqs" not in cfg:
        raise KeyError(
            f"system_config['vllm']['engines'][{engine!r}]['max_num_seqs'] is required"
        )
    return max(1, int(cfg["max_num_seqs"]))


def vllm_prefill_max_num_seqs(system_config: dict[str, Any], engine: str) -> int:
    cfg = vllm_engine_config(system_config, engine)
    if "prefill_max_num_seqs" in cfg:
        return max(1, int(cfg["prefill_max_num_seqs"]))
    if engine == "generator":
        return request_batch(system_config)
    return vllm_max_num_seqs(system_config, engine)


def vllm_decode_max_num_seqs(system_config: dict[str, Any], engine: str) -> int:
    cfg = vllm_engine_config(system_config, engine)
    if "decode_max_num_seqs" in cfg:
        return max(1, int(cfg["decode_max_num_seqs"]))
    if engine == "generator":
        return decode_batch(system_config)
    return vllm_max_num_seqs(system_config, engine)


def vllm_kv_cache_dtype(system_config: dict[str, Any]) -> str:
    vllm = system_config.get("vllm")
    if not isinstance(vllm, dict):
        raise KeyError("system_config['vllm'] is required")
    if "kv_cache_dtype" not in vllm:
        raise KeyError("system_config['vllm']['kv_cache_dtype'] is required")
    return str(vllm["kv_cache_dtype"])


def vllm_max_num_batched_tokens(system_config: dict[str, Any]) -> int | None:
    vllm = system_config.get("vllm")
    if not isinstance(vllm, dict):
        raise KeyError("system_config['vllm'] is required")
    value = vllm.get("max_num_batched_tokens")
    if value is None:
        return None
    return int(value)


def performance_stage_info(system_config: dict[str, Any], stage: str) -> dict[str, Any]:
    stages = require_layout(system_config).get("performance_stages") or {}
    info = stages.get(stage)
    if not isinstance(info, dict):
        raise KeyError(
            f"system_config['layout']['performance_stages'][{stage!r}] is required"
        )
    return info


def performance_stage_devices(system_config: dict[str, Any], stage: str) -> list[str]:
    info = performance_stage_info(system_config, stage)
    if info.get("kind") == "cpu_retrieval":
        return []
    devices = info.get("devices")
    if not isinstance(devices, list) or not devices:
        raise KeyError(
            f"system_config['layout']['performance_stages'][{stage!r}]['devices']"
            f" is required"
        )
    return [str(d) for d in devices if d]


def performance_stage_num_chips(system_config: dict[str, Any], stage: str) -> int:
    info = performance_stage_info(system_config, stage)
    if info.get("kind") == "cpu_retrieval":
        return 0
    if "num_chips" not in info:
        raise KeyError(
            f"system_config['layout']['performance_stages'][{stage!r}]['num_chips']"
            f" is required"
        )
    return max(1, int(info["num_chips"]))


def engine_info(system_config: dict[str, Any], engine: str) -> dict[str, Any]:
    engines = require_layout(system_config).get("engines") or {}
    info = engines.get(engine)
    if not isinstance(info, dict):
        raise KeyError(f"system_config['layout']['engines'][{engine!r}] is required")
    return info


def engine_pd_serving(system_config: dict[str, Any], engine: str) -> str:
    """The engine's PD classification: ``collocated_pd`` | ``disagg_pd``.

    DERIVED from resource-group membership of the engine's prefill/decode
    PerformanceStages; the persisted ``pd_serving`` field, when present, is
    asserted to agree (a stale snapshot must fail loudly, not win)."""
    from rag_stack.taxonomy import TAXONOMY, Stage

    info = engine_info(system_config, engine)
    stored = info.get("pd_serving")
    derived: str | None = None
    try:
        units = TAXONOMY.performance_stages_of(Stage(engine))
    except ValueError:
        units = ()
    pd_names = [ps.name for ps in units if ps.kind in ("prefill", "decode")]
    groups = require_layout(system_config).get("resource_groups")
    if len(pd_names) == 2 and isinstance(groups, list):
        group_of: dict[str, int] = {}
        for gi, g in enumerate(groups):
            if not isinstance(g, dict):
                continue
            for st in g.get("performance_stages", []) or []:
                group_of[str(st)] = gi
        if all(n in group_of for n in pd_names):
            derived = ("disagg_pd"
                       if group_of[pd_names[0]] != group_of[pd_names[1]]
                       else "collocated_pd")
    if derived is None:
        if stored is None:
            raise KeyError(
                f"system_config['layout']['engines'][{engine!r}]['pd_serving'] "
                f"is required (not derivable from resource_groups)"
            )
        return str(stored)
    if stored is not None and str(stored) != derived:
        raise ValueError(
            f"layout.engines[{engine!r}].pd_serving={stored!r} contradicts the "
            f"resource-group derivation {derived!r} — the groups are the truth"
        )
    return derived


def engine_devices(system_config: dict[str, Any], engine: str) -> list[str]:
    info = engine_info(system_config, engine)
    devices = info.get("devices")
    if not isinstance(devices, list) or not devices:
        raise KeyError(
            f"system_config['layout']['engines'][{engine!r}]['devices'] is required"
        )
    return [str(d) for d in devices if d]


def _parallelism_from_record(record: dict[str, Any], *, label: str) -> tuple[int, int]:
    devices = [d for d in (record.get("devices") or []) if d]
    if not devices:
        raise ValueError(f"{label} requires a non-empty devices list")
    n = len(devices)
    if "tp" not in record or "pp" not in record:
        raise KeyError(f"{label} requires explicit tp and pp")
    tp = max(1, int(record["tp"]))
    pp = max(1, int(record["pp"]))
    if tp * pp != n:
        raise ValueError(f"{label} TP*PP={tp}*{pp} does not match devices={n}")
    return tp, pp


def engine_parallelism(system_config: dict[str, Any], engine: str) -> tuple[int, int]:
    info = engine_info(system_config, engine)
    if engine_pd_serving(system_config, engine) == "disagg_pd":
        raise ValueError(
            f"{engine} is PD-disaggregated; use engine_role_parallelism(..., role)"
        )
    return _parallelism_from_record(info, label=engine)


def _role_record_key(engine: str, role: str) -> str:
    """Disagg role records are keyed by the FULL PerformanceStage name
    (``generator_prefill`` / ``generator_decode`` …) — resolved via taxonomy."""
    from rag_stack.taxonomy import TAXONOMY

    for ps in TAXONOMY.performance_stages_of(engine):
        if ps.kind == role:
            return ps.name
    raise KeyError(f"{engine!r} has no {role!r} PerformanceStage")


def engine_role_info(
    system_config: dict[str, Any],
    engine: str,
    role: str,
) -> dict[str, Any]:
    info = engine_info(system_config, engine)
    if engine_pd_serving(system_config, engine) != "disagg_pd":
        raise ValueError(f"{engine} is not PD-disaggregated")
    key = _role_record_key(engine, role)
    role_info = info.get(key)
    if not isinstance(role_info, dict):
        raise KeyError(
            f"system_config['layout']['engines'][{engine!r}][{key!r}] is required"
        )
    return role_info


def engine_role_devices(
    system_config: dict[str, Any],
    engine: str,
    role: str,
) -> list[str]:
    info = engine_role_info(system_config, engine, role)
    devs = info.get("devices") or []
    return [str(d) for d in devs if d]


def engine_role_parallelism(
    system_config: dict[str, Any],
    engine: str,
    role: str,
) -> tuple[int, int]:
    info = engine_role_info(system_config, engine, role)
    return _parallelism_from_record(info, label=f"{engine}.{role}")


def stage_devices(system_config: dict[str, Any], stage: str) -> list[str]:
    if stage in ("generator", "query_expansion"):
        return engine_devices(system_config, stage)
    defined = require_layout(system_config).get("performance_stages") or {}
    # The stage's GPU unit(s): first defined GPU PerformanceStage owned by it.
    from rag_stack.taxonomy import TAXONOMY, Stage

    try:
        units = TAXONOMY.performance_stages_of(Stage(stage))
    except ValueError:
        units = ()
    candidates = [ps.name for ps in units if ps.resource == "gpu"] or [stage]
    if stage == "semantic_retrieval":
        candidates = ["semantic_retrieval_encode"]
    for stage in candidates:
        if stage in defined:
            return performance_stage_devices(system_config, stage)
    return []


def stage_device(
    system_config: dict[str, Any],
    stage: str,
    default: str = "cuda:0",
) -> str:
    devs = stage_devices(system_config, stage)
    return devs[0] if devs else default


def generator_chip_count(system_config: dict[str, Any]) -> int:
    info = engine_info(system_config, "generator")
    if engine_pd_serving(system_config, "generator") == "disagg_pd":
        pre = engine_role_devices(system_config, "generator", "prefill")
        dec = engine_role_devices(system_config, "generator", "decode")
        if not pre or not dec:
            raise ValueError("disaggregated generator requires prefill/decode devices")
        return len(pre) + len(dec)
    return max(1, len(info.get("devices") or []))


def llm_stage_pd_info(system_config: dict[str, Any], engine: str) -> dict[str, Any]:
    """PD classification of ANY llm-shaped stage's engine — every vLLM-backed
    stage (generator, query_expansion, generative compressor) supports PD
    disaggregation; the classification is derived, never a knob."""
    info = engine_info(system_config, engine)
    if engine_pd_serving(system_config, engine) == "disagg_pd":
        pre = engine_role_devices(system_config, engine, "prefill")
        dec = engine_role_devices(system_config, engine, "decode")
        pre_tp, pre_pp = engine_role_parallelism(system_config, engine, "prefill")
        dec_tp, dec_pp = engine_role_parallelism(system_config, engine, "decode")
        return {
            "pd_split": True,
            "pd_layout": (
                f"1p1d_prefill_tp{pre_tp}pp{pre_pp}"
                f"_decode_tp{dec_tp}pp{dec_pp}"
            ),
            "prefill_gpus": len(pre),
            "decode_gpus": len(dec),
            "prefill_tp": pre_tp,
            "prefill_pp": pre_pp,
            "decode_tp": dec_tp,
            "decode_pp": dec_pp,
        }
    devs = [d for d in (info.get("devices") or []) if d]
    return {
        "pd_split": False,
        "pd_layout": "collocated_pd",
        "prefill_gpus": len(devs) or None,
        "decode_gpus": len(devs) or None,
    }


def generator_pd_info(system_config: dict[str, Any]) -> dict[str, Any]:
    return llm_stage_pd_info(system_config, "generator")


def resource_groups(system_config: dict[str, Any]) -> list[dict[str, Any]]:
    groups = require_layout(system_config).get("resource_groups")
    if not isinstance(groups, list):
        raise KeyError("system_config['layout']['resource_groups'] is required")
    return [g for g in groups if isinstance(g, dict)]


def collocation_and_chips(
    system_config: dict[str, Any],
    stages: list[str],
) -> tuple[list[list[str]], dict[str, int], int]:
    """Return CM fixed-placement inputs from the current deployment schema."""
    wanted = list(stages)
    wanted_set = set(wanted)
    groups: list[list[str]] = []
    seen: set[str] = set()
    for group in resource_groups(system_config):
        present = [s for s in group.get("performance_stages", []) if s in wanted_set]
        if present:
            groups.append(present)
            seen.update(present)
    for stage in wanted:
        if stage not in seen:
            groups.append([stage])

    chips: dict[str, int] = {}
    for stage in wanted:
        if stage in ("semantic_retrieval_vectorsearch", "lexical_retrieval", "retrieval_fusion"):
            continue  # Host search/fusion never consumes GPU chips.
        chips[stage] = performance_stage_num_chips(system_config, stage)

    num_retrieval_servers = max(
        1,
        int(retrieval_config(system_config).get("num_servers", 1) or 1),
    )
    return groups, chips, num_retrieval_servers


# ---------------------------------------------------------------------------
# CM fixed-placement bridge — measured schema → RAGCMAssembly.evaluate_fixed
# ---------------------------------------------------------------------------

# Canonical pipeline order for collocation groups (decode LAST — the order
# RAGO's assemble_cost expects). Port of the rag_cm_accuracy benchmark's
# _STAGE_ORDER; that benchmark now consumes this module.
CM_STAGE_ORDER = [
    "query_expansion_prefill", "query_expansion_decode",
    "semantic_retrieval_encode", "semantic_retrieval_vectorsearch",
    "retrieval_fusion",
    "passage_reranker", "passage_filter", "passage_compressor",
    "passage_compressor_prefill", "passage_compressor_decode",
    "generator_prefill", "generator_decode",
]


def cm_stage_names_from_pipeline(pipeline_config: dict[str, Any]) -> list[str]:
    """Active PerformanceStage names of a RESOLVED pipeline config, without
    constructing cost-model instances.

    Use the shared structural lowering so fixed deployment and CM dispatch
    agree on evaluator-observed fusion as well as explicitly declared stages.
    """
    from rag_stack.rag_ir.normalize import dispatch_manifest
    from rag_stack.taxonomy import TAXONOMY

    stages: list[str] = []
    for entry in dispatch_manifest(pipeline_config):
        for ps in TAXONOMY.performance_stages_of_component(entry.component):
            if ps.stage is entry.stage and ps.name not in stages:
                stages.append(ps.name)
    return stages


def cm_fixed_pipeline_config(
    pipeline_config: dict[str, Any], system_config: dict[str, Any],
) -> dict[str, Any]:
    """Fold a concrete deployment (measured schema) into a CM pipeline config.

    The CM reads batches / layout TP-PP / retrieval runtime from
    ``pipeline_config["system"]`` — updating it with the resolved
    system_config pins the CM to exactly that deployment (the
    rag_cm_accuracy fold, verbatim)."""
    import copy

    out = copy.deepcopy(pipeline_config)
    out.setdefault("system", {}).update(system_config)
    return out


def cm_fixed_placement_inputs(
    system_config: dict[str, Any],
    pipeline_config: dict[str, Any],
) -> tuple[list[list[str]], dict[str, int], int, list[str]]:
    """(collocation_strategy, num_chips, num_retrieval_servers, stages) for
    ``RAGCMAssembly.evaluate_fixed`` from the resolved deployment
    schema — groups in canonical pipeline order (decode last), exactly the
    battle-tested rag_cm_accuracy reconstruction."""
    stages = cm_stage_names_from_pipeline(pipeline_config)
    groups, chips, retrieval_servers = collocation_and_chips(
        system_config, stages,
    )
    order = {s: i for i, s in enumerate(CM_STAGE_ORDER)}
    coll = [sorted(g, key=lambda s: order.get(s, 99)) for g in groups]
    return coll, chips, retrieval_servers, stages
