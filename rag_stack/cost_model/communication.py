"""Inter-stage communication costs for the RAGO sweep (post-processing).

RAGO's composition is pure compute: stages chained with ZERO data-movement
cost. Real deployments pay for every pipeline boundary whose two adjacent
stages live on different chips — dominated by the 1P1D KV-cache transfer
(prefill → decode over NIXL/PCIe on the 3090 box), with smaller text/vector
payloads at the other boundaries (encode→retrieval embedding vectors,
retrieval→reranker passage text, ...).

This module post-processes the sweep DataFrame instead of patching the
vendored RAGO: each row carries its resource ``collocation_strategy`` plus,
when resource groups are non-contiguous in the pipeline, ``pipeline_stage_order``.
Per row we

  1. find every adjacent-stage pair that crosses a device boundary
     (same GPU group = free HBM handoff, except any edge into/out of the CPU
     retrieval stage is always a host transfer),
  2. price each crossing as ``ICN_LL + payload_bytes / ICN_BW``,
  3. add the sum to ``latency_s`` and cap ``qps`` at the bandwidth bound
     ``min(ICN_BW / payload_bytes)`` (transfers of different requests overlap
     on the link, so only bandwidth — not per-request latency — limits
     throughput).

Payload model (bytes per user request):
  * ``*_prefill → *_decode`` of the same LLM engine (1P1D): the prompt's
    KV cache — ``input_tokens × 2(K,V) × layers × kv_heads × head_dim ×
    dtype_bytes`` with model shapes from GenZ's registry. Multiplied by
    ``num_retrievals`` for iterative rows (one handoff per retrieval round).
  * ``encode → retrieval``: query embedding vectors (fanout × dim × fp32).
  * ``retrieval → passage_reranker``: candidate passage text
    (reranker_input_fanout × avg_chunk_tokens × ~4 B/token).
  * boundary into ``prefill``: the assembled prompt text
    (seq_len_prefill × ~4 B/token).
  * anything else: sub-query text (fanout × avg_query_tokens × ~4 B/token).

Link parameters come from the hardware spec's GPU dict (``ICN`` GB/s,
``ICN_LL`` µs) — the same fields GenZ uses for intra-stage TP/PP collectives —
with the calibration layer's defaults when absent (string GPU presets).
"""
from __future__ import annotations

import ast
import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("RAG-Stack")
_CPU_RETRIEVAL_STAGES = frozenset({
    "semantic_retrieval_vectorsearch", "lexical_retrieval", "retrieval_fusion",
})

# Fallback UTF-8 bytes/token for text payloads, used ONLY when TokenStats
# carries no dataset-derived value (pre-migration serialized stats). The
# real ratio is computed per dataset+tokenizer in TokenStats.bytes_per_token
# (English ~4, CJK ~1.5-3, code ~3) — never assume a language here.
_TEXT_BYTES_PER_TOKEN_FALLBACK = 4.0
# KV cache dtype: generator engines run bf16 in every measured config.
_KV_DTYPE_BYTES = 2
# Calibration-layer defaults (mirrors llm_sim/calibration spec.get fallbacks)
_DEFAULT_ICN_GBS = 150.0
_DEFAULT_ICN_LL_US = 1.0


# This module prices the 1P1D PD handoff from physical data movement only:
# t_comm = link latency + KV bytes / link bandwidth, with the independent
# throughput bound link bandwidth / KV bytes. Runtime/e2e correction factors
# are intentionally not part of the portable communication model.


def pd_split(strategy_repr) -> bool:
    """True when the generator's prefill and decode sit in DIFFERENT groups
    (the deployment realizes a 1P1D PD pair). Strategy-parsing helper shared
    with analysis tooling. PD-split rows pay the prefill→decode KV-transfer
    boundary priced below (gpu_gpu link: latency + KV_bytes/bandwidth, from the
    declarative interconnect spec) — purely data-driven, no overhead/channel fudge.
    """
    try:
        groups = strategy_repr if isinstance(strategy_repr, list) else ast.literal_eval(
            str(strategy_repr)
        )
    except Exception:  # noqa: BLE001
        return False
    gi = {}
    for i, g in enumerate(groups):
        for st in g:
            gi[st] = i
    return (
        "generator_prefill" in gi
        and "generator_decode" in gi
        and gi["generator_prefill"] != gi["generator_decode"]
    )


def _is_pd_edge(src: str, dst: str) -> bool:
    """True for a same-LLM-engine prefill→decode boundary (the KV handoff)."""
    from rag_stack.taxonomy import TAXONOMY

    ps_src = TAXONOMY.performance_stages.get(src)
    ps_dst = TAXONOMY.performance_stages.get(dst)
    if ps_src is None or ps_dst is None:
        return False
    return (
        ps_src.kind == "prefill"
        and ps_dst.kind == "decode"
        and TAXONOMY.user_stage_of(src) is TAXONOMY.user_stage_of(dst)
    )


_warned_link_fallback: set = set()


def _parse_list_repr(value: Any):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:  # noqa: BLE001
            return None
    return value


def _bool_opt(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "on", "enabled"}:
        return True
    if s in {"0", "false", "no", "off", "disabled"}:
        return False
    return None


def include_non_pd_communication(pipeline_config: dict) -> bool:
    """Whether to model non-PD inter-stage communication.

    Only the canonical, explicit ``system.communication.include_non_pd_comm``
    switch is accepted.  Environment variables and historical aliases were
    removed because they silently changed production pricing outside the
    frozen replay/config provenance.  PD communication is always priced
    independently of this test/isolation switch.
    """
    syscfg = (pipeline_config or {}).get("system") or {}
    communication = syscfg.get("communication")
    if communication is None:
        return True
    if not isinstance(communication, dict):
        raise ValueError("system.communication must be a mapping")
    retired = set(communication) & {
        "include_non_pd_communication", "model_non_pd_comm",
        "model_non_pd_communication", "disable_non_pd_comm",
        "disable_non_pd_communication", "no_non_pd_comm",
        "no_non_pd_communication", "no_comm",
    }
    if retired:
        raise ValueError(
            "retired communication aliases are not accepted: "
            + ", ".join(sorted(retired))
        )
    raw = communication.get("include_non_pd_comm")
    if raw is None:
        return True
    enabled = _bool_opt(raw)
    if enabled is None:
        raise ValueError(
            "system.communication.include_non_pd_comm must be boolean"
        )
    return enabled


# ── Declarative interconnect link spec ─────────────────────────────────────
# A device's communication fabric is DECLARED IN THE YAML as raw numbers — there
# is NO preset/lookup table. Each device-pair (gpu_gpu / gpu_cpu) is a dict
# ``{bandwidth_gbps, latency_us?, type?}`` written directly in ``system.
# interconnect``. ``bandwidth_gbps`` is the link bandwidth (GB/s); ``latency_us``
# the per-message latency (µs, optional → ICN latency). ``type`` is a FREE-TEXT
# LABEL only (e.g. "nvlink4") — purely documentary, it is NOT resolved to any
# number. The whole spec flows verbatim into the benchmark/case JSON, so the
# numbers are visible there too. (Per-machine reference values for convenience:
# nvlink3 300/0.5µs, nvlink4 450/0.4µs, pcie4_p2p 50/1.0µs, pcie5_p2p 100/0.8µs,
# pcie4_host 27/1.0µs, pcie5_host 54/0.8µs — but these are NOT looked up; write
# the actual measured value for your box in the YAML.)


def _resolve_link(
    hardware: Any, pair: str, icn_gbs: float, icn_us: float, icn_declared: bool,
) -> tuple[float, float]:
    """Resolve ONE device-pair's link to (bandwidth bytes/s, latency s).

    Reads ``system.interconnect[pair]`` (``hardware.interconnect``) — a dict
    ``{bandwidth_gbps, latency_us?, type?}`` with the numbers written directly
    (NO preset table; ``type`` is a label only). Priority: explicit
    ``bandwidth_gbps``/``latency_us`` > ``ICN`` (the base single-link spec). A
    bare string entry is tolerated as a ``type`` label that carries no numbers →
    ICN fallback. Warns once if a link is undeclared AND the gpu has no ICN.
    """
    spec = getattr(hardware, "interconnect", None) or {}
    entry = spec.get(pair, {})
    if isinstance(entry, str):           # legacy bare string → label only, no numbers
        entry = {"type": entry}
    bw_gbps = entry.get("bandwidth_gbps")
    lat_us = entry.get("latency_us")
    if bw_gbps is None:                   # no explicit bandwidth → fall back to ICN
        bw_gbps = icn_gbs
        if not icn_declared:
            key = (str(getattr(hardware, "gpu", None)), pair)
            if key not in _warned_link_fallback:
                _warned_link_fallback.add(key)
                logger.warning(
                    f"communication: no system.interconnect.{pair} and no gpu "
                    f"ICN — using {_DEFAULT_ICN_GBS} GB/s default; declare "
                    f"system.interconnect.{pair} for accurate cross-group cost."
                )
    if lat_us is None:
        lat_us = icn_us
    return float(bw_gbps) * 1e9, float(lat_us) * 1e-6


def _comm_bandwidths(hardware: Any) -> tuple[float, float, float, float]:
    """Per-edge-kind link (bandwidth bytes/s, latency s), from the DECLARATIVE
    ``system.interconnect`` spec. Two physical transfer classes:

    * **gpu_gpu** — the prefill→decode KV handoff (KV cache lives in GPU memory,
      moved GPU↔GPU directly: NVLink / PCIe-P2P / host-staged — declared).
    * **gpu_cpu** — every other cross-stage edge (encode→retrieval query vectors,
      →prefill prompt text), which are host transfers GPU↔CPU.

    Each is resolved by :func:`_resolve_link` (preset type → explicit override →
    ICN). Returns ``(pd_bw, host_bw, pd_lat, host_lat)``.
    Measured rtx_3090: GPU-GPU P2P ~52 vs GPU↔CPU ~27 GB/s — they differ ~2×.
    """
    gpu = getattr(hardware, "gpu", None)
    if isinstance(gpu, dict) and "ICN" in gpu:   # quiet base read (no warning here)
        icn_gbs = float(gpu["ICN"])
        icn_us = float(gpu.get("ICN_LL", _DEFAULT_ICN_LL_US))
        icn_declared = True
    else:
        icn_gbs, icn_us, icn_declared = _DEFAULT_ICN_GBS, _DEFAULT_ICN_LL_US, False
    pd_bw, pd_lat = _resolve_link(hardware, "gpu_gpu", icn_gbs, icn_us, icn_declared)
    host_bw, host_lat = _resolve_link(hardware, "gpu_cpu", icn_gbs, icn_us, icn_declared)
    return pd_bw, host_bw, pd_lat, host_lat


def kv_bytes_per_token(model_name_or_cfg: Any) -> float:
    """Per-token KV-cache bytes for an LLM (both K and V, all layers, bf16)."""
    cfg = model_name_or_cfg
    if isinstance(cfg, str):
        from GenZ.Models import get_configs
        cfg = get_configs(cfg)
    kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (
        cfg.hidden_size // cfg.num_attention_heads
    )
    return 2.0 * cfg.num_decoder_layers * kv_heads * head_dim * _KV_DTYPE_BYTES


def _find_module(pipeline_config: dict, stage: str) -> dict:
    for nl in pipeline_config.get("node_lines", []):
        for node in nl.get("nodes", []):
            if node.get("stage") == stage:
                mods = node.get("modules", [])
                if mods:
                    return mods[0]
    return {}


def _genz_model_for(pipeline_config: dict, stage: str):
    """GenZ model cfg for a node's LLM, or None when absent/unresolvable."""
    from rag_stack.cost_model.llm_sim.genz_model_map import resolve_genz_model
    name = _find_module(pipeline_config, stage).get("model")
    if isinstance(name, list):
        name = name[0] if name else None
    if not name:
        return None
    try:
        resolved = resolve_genz_model(str(name))
        return kv_bytes_per_token(resolved)
    except Exception as e:  # noqa: BLE001 — comm modeling must never kill a sweep
        logger.warning(f"communication: KV size for {name!r} unresolved ({e}); "
                       f"PD transfer for {stage} not modeled")
        return None


def _embedding_dim(pipeline_config: dict) -> int:
    vdb_name = _find_module(pipeline_config, "semantic_retrieval").get("vectordb", "")
    for vdb in pipeline_config.get("vectordb", []):
        if vdb.get("name") == vdb_name:
            return int(vdb.get("embedding_dim", 768))
    return 768


class CommunicationPayloadModel:
    """One reusable payload contract for analytical and DES communication.

    Keeping payload construction here prevents the static post-processor and
    the closed-loop event simulator from silently pricing different byte
    volumes for the same pipeline boundary.
    """

    def __init__(
        self,
        pipeline_config: dict,
        *,
        seq_len_prefill: int,
        qe_seq_len: Optional[int],
        reranker_input_fanout: int,
        query_expansion_fanout: int,
    ) -> None:
        from rag_stack.cost_model.token_stats import TokenStats

        self.ts = TokenStats.resolve(pipeline_config)
        self.text_bpt = float(
            getattr(self.ts, "bytes_per_token", _TEXT_BYTES_PER_TOKEN_FALLBACK)
        )
        self.gen_kv = _genz_model_for(pipeline_config, "generator")
        self.qe_kv = _genz_model_for(pipeline_config, "query_expansion")
        self.embedding_dim = _embedding_dim(pipeline_config)
        self.fanout = max(int(query_expansion_fanout or 1), 1)
        self.seq_len_prefill = max(1, int(seq_len_prefill))
        self.qe_seq_len = int(qe_seq_len) if qe_seq_len else None
        self.reranker_input_fanout = max(int(reranker_input_fanout or 1), 1)

    def payload_bytes(
        self,
        src: str,
        dst: str,
        *,
        num_retrievals: float = 1.0,
        destination_input_tokens: Optional[int] = None,
        destination_input_bytes: Optional[float] = None,
    ) -> float:
        """Bytes moved by one user request across ``src -> dst``.

        Trace-v1 records the destination call's serialized ``input_bytes``;
        the DES uses that exact value for text/prompt boundaries.  Old traces
        may omit it, in which case ``destination_input_tokens`` (then aggregate
        TokenStats) is the compatibility fallback.  Tensor-shaped embedding
        and KV edges remain derived from their concrete shapes and deliberately
        ignore serialized text byte fields.
        """
        if src == "generator_prefill" and dst == "generator_decode":
            if self.gen_kv is not None:
                return (
                    float(self.seq_len_prefill)
                    * self.gen_kv
                    * max(float(num_retrievals), 1.0)
                )
            return 0.0
        if src == "query_expansion_prefill" and dst == "query_expansion_decode":
            if self.qe_kv is not None and self.qe_seq_len:
                return float(self.qe_seq_len) * self.qe_kv
            return 0.0
        if src == "semantic_retrieval_encode" and dst == "semantic_retrieval_vectorsearch":
            return float(self.fanout * self.embedding_dim * 4)
        if destination_input_bytes is not None:
            return max(0.0, float(destination_input_bytes))
        if src in _CPU_RETRIEVAL_STAGES and dst == "passage_reranker":
            return float(
                self.reranker_input_fanout
                * self.ts.avg_chunk_tokens
                * self.text_bpt
            )
        if dst == "generator_prefill":
            tokens = (
                max(1, int(destination_input_tokens))
                if destination_input_tokens is not None
                else self.seq_len_prefill
            )
            return float(tokens) * self.text_bpt
        if destination_input_tokens is not None:
            return float(max(1, int(destination_input_tokens))) * self.text_bpt
        return float(self.fanout * self.ts.avg_query_tokens * self.text_bpt)


def gpu_cpu_link_bytes_s(hardware: Any) -> tuple[float, float]:
    """Declared GPU/CPU host link as ``(bytes/s, seconds)``."""
    _, host_bw, _, host_lat = _comm_bandwidths(hardware)
    return host_bw, host_lat


def apply_communication_costs(
    df: pd.DataFrame,
    *,
    hardware: Any,
    pipeline_config: dict,
    seq_len_prefill: int,
    qe_seq_len: Optional[int],
    reranker_input_fanout: int,
    query_expansion_fanout: int,
    include_non_pd_edges: Optional[bool] = None,
) -> pd.DataFrame:
    """Add inter-group communication costs to a RAGO sweep DataFrame in place.

    New columns: ``comm_latency_s`` (per-request transfer time added to
    ``latency_s``), ``comm_qps_bound`` (link-bandwidth throughput ceiling;
    caps ``qps`` / ``qps_per_chip`` where binding), ``comm_boundaries``
    (human-readable per-boundary payloads). Rows whose partition keeps adjacent
    GPU stages co-resident pay nothing for those GPU→GPU handoffs; retrieval is
    CPU work, so encode→retrieval and retrieval/fusion→GPU stages are host
    transfers. CPU search→CPU fusion stays on the host and adds no link event.
    Disabling non-PD communication keeps the measured-visible
    PD prefill→decode handoff enabled.
    """
    if df is None or len(df) == 0 or "collocation_strategy" not in df.columns:
        logger.info("communication: no collocation_strategy column — skipped")
        return df
    if include_non_pd_edges is None:
        include_non_pd_edges = include_non_pd_communication(pipeline_config)

    pd_bw, host_bw, pd_lat, host_lat = _comm_bandwidths(hardware)
    system_cfg = (pipeline_config or {}).get("system") or {}
    topology_source = system_cfg if system_cfg.get("interconnect") else hardware
    payload_model = CommunicationPayloadModel(
        pipeline_config,
        seq_len_prefill=seq_len_prefill,
        qe_seq_len=qe_seq_len,
        reranker_input_fanout=reranker_input_fanout,
        query_expansion_fanout=query_expansion_fanout,
    )

    def _payload_bytes(src: str, dst: str, num_retrievals: float) -> float:
        return payload_model.payload_bytes(
            src, dst, num_retrievals=num_retrievals,
        )

    _memo: dict = {}

    def _stage_devices(stage: str, candidate_layout: Any = None) -> list[str]:
        layout = candidate_layout
        if not isinstance(layout, dict):
            layout = system_cfg.get("layout") if isinstance(system_cfg, dict) else None
        if not isinstance(layout, dict):
            return []
        stages = layout.get("performance_stages")
        if not isinstance(stages, dict):
            return []
        info = stages.get(stage)
        if not isinstance(info, dict):
            return []
        devices = info.get("devices")
        if not isinstance(devices, list):
            return []
        return [str(d) for d in devices if d]

    def _row_comm(
        strategy_repr: Any,
        num_retrievals: float,
        order_repr: Any = None,
        candidate_layout_repr: Any = None,
    ):
        candidate_layout = candidate_layout_repr
        if candidate_layout_repr is not None and not isinstance(candidate_layout_repr, dict):
            try:
                from rag_stack.cost_model.candidate_layout import (
                    candidate_system_layout_from_row,
                    CANDIDATE_LAYOUT_COLUMN,
                )
                candidate_layout = candidate_system_layout_from_row(
                    {CANDIDATE_LAYOUT_COLUMN: candidate_layout_repr}
                )
            except Exception as exc:
                raise ValueError("Invalid candidate_system_layout for communication") from exc
        key = (
            str(strategy_repr), num_retrievals, str(order_repr),
            str(candidate_layout.get("device_layout_id") if isinstance(candidate_layout, dict) else None),
            str((candidate_layout or {}).get("performance_stages") if isinstance(candidate_layout, dict) else None),
        )
        if key in _memo:
            return _memo[key]
        try:
            groups = _parse_list_repr(strategy_repr)
            if groups is None:
                raise ValueError(f"unparsable strategy {strategy_repr!r}")
            stage_group = {}
            for gi, g in enumerate(groups):
                for st in g:
                    stage_group[st] = gi
            order = _parse_list_repr(order_repr)
            if order is None:
                order = [st for g in groups for st in g]
            elif order and isinstance(order[0], list):
                order = [st for g in order for st in g]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"communication: unparsable strategy {strategy_repr!r}: {e}")
            _memo[key] = (0.0, float("inf"), "")
            return _memo[key]
        total_s, qps_bound, parts = 0.0, float("inf"), []
        for src, dst in zip(order, order[1:]):
            if src in _CPU_RETRIEVAL_STAGES and dst in _CPU_RETRIEVAL_STAGES:
                continue
            retrieval_edge = src in _CPU_RETRIEVAL_STAGES or dst in _CPU_RETRIEVAL_STAGES
            if (
                stage_group.get(src) is not None
                and stage_group.get(src) == stage_group.get(dst)
                and not retrieval_edge
            ):
                continue  # co-resident GPU stages — HBM handoff, free
            nbytes = _payload_bytes(src, dst, num_retrievals)
            if nbytes <= 0:
                continue
            # Every cross-group edge is priced purely by the DATA moved:
            # t_comm = link_latency + bytes/link_BW, throughput bound =
            # link_BW/bytes. No hardcoded per-handoff overhead/channels — the
            # payload (nbytes) already scales with the workload.
            #
            # LINK CLASS — gpu_gpu ONLY for the prefill→decode KV handoff; every
            # other edge is gpu_cpu (host). This MATCHES what measured actually
            # does, and that consistency is the point (the measured⟺CM rule):
            # static_rag_evaluator runs each stage as a SEPARATE model call
            # passing pandas DataFrames of TEXT (reranker → reranked passages →
            # prompt string), and the generator is a SEPARATE vLLM process that
            # only accepts a text prompt over its API. So a reranker→prefill (or
            # compress→prefill) hop is reranker(GPU compute) → text on the HOST →
            # vLLM(GPU): a genuine GPU↔CPU transfer, NOT GPU↔GPU. There is no GPU
            # tensor to keep on-device — the payload is the passage SELECTION
            # (text), not the reranker's activations. The ONLY true GPU↔GPU edge
            # is the 1P1D KV-cache handoff (a binary tensor over NixlConnector/
            # NVLink). Do NOT "optimize" the text edges to gpu_gpu: it is neither
            # what measured does nor numerically relevant (text bytes are ~KB →
            # sub-µs on either link, dwarfed by stage compute). To genuinely make
            # them GPU↔GPU you'd need a fused single-process zero-copy serving
            # engine — a different architecture, and measured must change FIRST.
            # Measured rtx_3090 link BWs: 52 (gpu_gpu) vs 27 (gpu_cpu) GB/s.
            is_pd_edge = _is_pd_edge(src, dst)
            if not is_pd_edge and not include_non_pd_edges:
                continue
            if is_pd_edge:
                src_devs = _stage_devices(src, candidate_layout)
                dst_devs = _stage_devices(dst, candidate_layout)
                if src_devs and dst_devs:
                    from rag_stack.cost_model.interconnect_topology import (
                        effective_sharded_gpu_transfer_bytes_s,
                    )
                    edge_bw, edge_lat = effective_sharded_gpu_transfer_bytes_s(
                        topology_source,
                        src_devs,
                        dst_devs,
                        default_gbps=pd_bw / 1e9,
                        default_us=pd_lat * 1e6,
                    )
                else:
                    from rag_stack.cost_model.interconnect_topology import has_gpu_topology
                    if has_gpu_topology(topology_source):
                        raise ValueError(
                            f"Topology-aware PD communication requires concrete devices for "
                            f"{src}->{dst}; got {src_devs}->{dst_devs}"
                        )
                    edge_bw, edge_lat = pd_bw, pd_lat
            else:
                edge_bw, edge_lat = host_bw, host_lat
            # Physical data movement only: link latency + bytes/bandwidth.
            t_comm = edge_lat + nbytes / edge_bw
            total_s += t_comm
            qps_bound = min(qps_bound, edge_bw / nbytes)
            parts.append(f"{src}→{dst}:{nbytes / 1e6:.2f}MB")
        _memo[key] = (total_s, qps_bound, "; ".join(parts))
        return _memo[key]

    n_ret_col = df["num_retrievals"] if "num_retrievals" in df.columns else None
    order_col = df["pipeline_stage_order"] if "pipeline_stage_order" in df.columns else None
    layout_col = df["candidate_system_layout"] if "candidate_system_layout" in df.columns else None
    comm = [
        _row_comm(
            df["collocation_strategy"].iloc[i],
            # FLOAT handoff count: agentic traces have a fractional mean
            # generate-rounds/query (e.g. 2.3); the PD overhead/KV payload
            # scale by the precise mean, not a rounded count. Sequential RAG /
            # iterative both pass whole numbers → unchanged.
            float(n_ret_col.iloc[i]) if n_ret_col is not None else 1.0,
            order_col.iloc[i] if order_col is not None else None,
            layout_col.iloc[i] if layout_col is not None else None,
        )
        for i in range(len(df))
    ]
    df["comm_latency_s"] = [c[0] for c in comm]
    df["comm_qps_bound"] = [c[1] for c in comm]
    df["comm_boundaries"] = [c[2] for c in comm]
    df["comm_scope"] = "all" if include_non_pd_edges else "pd_only"
    df["comm_non_pd_enabled"] = bool(include_non_pd_edges)

    # NOTE: communication is priced PURELY by data volume (bytes/BW per edge,
    # incl. the 1P1D PD KV handoff) — no hardcoded per-handoff overhead and no
    # channel-count fudge. Both the old 2.34 multiplicative factor and its
    # successor pd_link(overhead, channels) are gone: they were fitted to
    # under-saturated measurements (low-concurrency proxy/scheduling stalls)
    # and over-counted the real per-request transfer ~10×. comm_latency_s now
    # carries only ICN_LL + bytes/BW; comm_qps_bound only the BW/bytes bound.

    df["latency_s"] = df["latency_s"] + df["comm_latency_s"]
    # The e2e tail (only present on the trace-sim df) carries the same per-request
    # comm latency, so keep it consistent with the mean.
    if "latency_s_p90" in df.columns:
        df["latency_s_p90"] = df["latency_s_p90"] + df["comm_latency_s"]
    capped = df["comm_qps_bound"] < df["qps"]
    if capped.any():
        ratio = (df.loc[capped, "comm_qps_bound"] / df.loc[capped, "qps"]).clip(upper=1.0)
        df.loc[capped, "qps"] = df.loc[capped, "comm_qps_bound"]
        if "qps_per_chip" in df.columns:
            df.loc[capped, "qps_per_chip"] = df.loc[capped, "qps_per_chip"] * ratio
        if "qps_bound" in df.columns:
            df.loc[capped, "qps_bound"] = df.loc[capped, "qps_bound"].apply(
                lambda b: (list(b) if isinstance(b, (list, tuple)) else [str(b)]) + ["comm"]
            )
        logger.info(
            f"communication: link bandwidth caps qps on {int(capped.sum())}/{len(df)} rows"
        )

    return df
