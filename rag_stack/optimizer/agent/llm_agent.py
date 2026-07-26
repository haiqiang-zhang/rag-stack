"""LLM advisor, prompt cards, ledger queries, and proposal parsing."""

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
from torch import Tensor

from rag_stack.search_space.ax_search_space_ops import AxSearchSpaceOps

from .util import _round

logger = logging.getLogger("RAG-Stack")

# ======================================================================
# Agent interface
# ======================================================================

@dataclass
class AgentAdvice:
    """What an agent hands back to the optimizer each iteration.

    ``centers``: (m, dim) tensor of agent-proposed configs, already encoded to the
    normalized [0,1]^dim space and snapped to the grid (valid, decodable). Injected
    into the acquisition candidate pool and used as native-local-search restart seeds.
    ``rationale``: short free-text diagnosis (logged, not used numerically).
    """
    centers: Optional[Tensor] = None
    # Auditable, non-numerical proposal identity.  Kept alongside the encoded
    # rows so logs/tests can distinguish the chosen anchor from explicit
    # deltas without reverse-engineering a full merged config.
    proposal_meta: List[Dict[str, Any]] = field(default_factory=list)
    rationale: str = ""
    notes: str = ""      # the LLM's self-maintained SEARCH NOTES (07-05) — its running world-model, fed back next round
    raw: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        n_centers = 0 if self.centers is None else int(self.centers.shape[0])
        if self.proposal_meta and len(self.proposal_meta) != n_centers:
            raise ValueError(
                "AgentAdvice.proposal_meta must align one-to-one with centers"
            )

    def n(self) -> int:
        return 0 if self.centers is None else int(self.centers.shape[0])


class AgentAdviceError(RuntimeError):
    """The advisory LLM failed to produce benchmark-usable candidates.

    ``AgentGuidedQNEHVI`` treats this exception specially in strict paper-run
    mode: it is allowed to escape ``suggest()`` so the controller cannot
    commit an unrelated qNEHVI/random evaluation under the agent benchmark's
    name.
    """


@dataclass(frozen=True)
class ParetoGapAnchor:
    """One realized endpoint of a normalized Pareto-frontier gap.

    ``flat`` is the exact decoded Ax parameterization that produced the
    observation.  Both objectives must already be normalized over the current
    realized history; the Agent is deliberately not allowed to invent or
    recompute that geometry.
    """

    flat: Mapping[str, Any]
    normalized_performance: float
    normalized_quality: float
    sub_metrics: Mapping[str, Any] = field(default_factory=dict)
    perf_attr: Any = None

    def __post_init__(self) -> None:
        if not isinstance(self.flat, Mapping) or not self.flat:
            raise ValueError("ParetoGapAnchor.flat must be a non-empty mapping")
        for name, value in (
            ("normalized_performance", self.normalized_performance),
            ("normalized_quality", self.normalized_quality),
        ):
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be a finite value in [0, 1]") from exc
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{name} must be a finite value in [0, 1]")
            object.__setattr__(self, name, numeric)
        if not isinstance(self.sub_metrics, Mapping):
            raise ValueError("ParetoGapAnchor.sub_metrics must be a mapping")
        # Own shallow copies so a caller cannot mutate a persisted forced
        # request while an Agent call is in flight.
        object.__setattr__(self, "flat", dict(self.flat))
        object.__setattr__(self, "sub_metrics", dict(self.sub_metrics))


@dataclass(frozen=True)
class GapExplorationRequest:
    """Evidence packet for one maximum-gap forced exploration call.

    The endpoint orientation is explicit: ``quality_anchor`` is the
    lower-performance / higher-quality endpoint and ``performance_anchor`` is
    the higher-performance / lower-quality endpoint.  This fail-closed
    contract prevents a stale or incorrectly oriented gap from silently
    steering the Agent toward a frontier extreme.
    """

    gap_id: str
    quality_anchor: ParetoGapAnchor
    performance_anchor: ParetoGapAnchor
    failed_nearby: Tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        gap_id = str(self.gap_id).strip()
        if not gap_id:
            raise ValueError("GapExplorationRequest.gap_id must be non-empty")
        object.__setattr__(self, "gap_id", gap_id)
        q_anchor = self.quality_anchor
        p_anchor = self.performance_anchor
        if not isinstance(q_anchor, ParetoGapAnchor) or not isinstance(
            p_anchor, ParetoGapAnchor
        ):
            raise ValueError("gap anchors must be ParetoGapAnchor instances")
        if not (
            p_anchor.normalized_performance > q_anchor.normalized_performance
            and q_anchor.normalized_quality > p_anchor.normalized_quality
        ):
            raise ValueError(
                "gap anchors must form an internal trade-off rectangle: "
                "quality_anchor has lower performance/higher quality"
            )
        failed = []
        for item in self.failed_nearby:
            if not isinstance(item, Mapping):
                raise ValueError("failed_nearby entries must be mappings")
            failed.append(dict(item))
        object.__setattr__(self, "failed_nearby", tuple(failed))


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _agent_strict_env_override() -> Optional[bool]:
    """Return the process-level strict-mode override, if one was supplied."""
    raw = os.environ.get("RAG_STACK_AGENT_STRICT")
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in _TRUE_ENV_VALUES:
        return True
    if value in _FALSE_ENV_VALUES:
        return False
    raise ValueError(
        "RAG_STACK_AGENT_STRICT must be one of "
        f"{sorted(_TRUE_ENV_VALUES | _FALSE_ENV_VALUES)}, got {raw!r}"
    )


def _resolve_agent_strict(default: bool) -> bool:
    override = _agent_strict_env_override()
    return bool(default) if override is None else override


def _resolve_agent_wall_timeout(
    value: Optional[float], *, fallback: Optional[float] = None,
) -> Optional[float]:
    """Resolve and validate the per-chat hard wall-clock deadline.

    The environment override is intentionally process-scoped so a resumed
    project can change failure policy without mutating its hash-locked YAML
    snapshot.
    """
    raw = os.environ.get("RAG_STACK_AGENT_WALL_TIMEOUT_S")
    effective: Any = raw if raw is not None and raw.strip() else value
    if effective is None:
        effective = fallback
    if effective is None:
        return None
    try:
        timeout = float(effective)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "agent wall timeout must be a positive finite number, "
            f"got {effective!r}"
        ) from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(
            "agent wall timeout must be a positive finite number, "
            f"got {effective!r}"
        )
    return timeout


# ----------------------------------------------------------------------
# LLM agent (DeepSeek)
# ----------------------------------------------------------------------

_SUBMETRIC_SHORT = {
    "deepeval_answer_correctness": "correctness",
    "deepeval_answer_relevancy": "ans_rel",
    "deepeval_context_recall": "ctx_recall",
    "deepeval_context_precision": "ctx_prec",
    "deepeval_contextual_relevancy": "ctx_rel",
    "deepeval_faithfulness": "faith",
    "retrieval_recall": "ret_recall",
    "retrieval_ndcg": "ret_ndcg",
}


# Knob-name aliasing + value abbreviation — keep the agent prompt compact so DeepSeek
# emits short DELTAS quickly (its latency is output-token-bound, not concurrency).
_ALIAS_ABBR = {
    "query_expansion": "qexp",
    "multi_query_expansion": "mqe",
    "query_decompose": "decompose",
    "semantic_retrieval": "retrieval",
    "passage_reranker": "reranker",
    "passage_compressor": "compressor",
    "prompt_maker": "prompt",
    "component": "type",
    "temperature": "temp",
    "faiss_ivf_index": "faiss",
}
_ALIAS_DROP = {"pipeline", "sequential", "vllm"}


def _shorten_name(name: str) -> str:
    """pipeline__sequential__generator__vllm__model -> generator.model (compact, readable)."""
    out = [_ALIAS_ABBR.get(p, p) for p in name.split("__") if p not in _ALIAS_DROP]
    if out and out[0] == "vectordb":
        out = ["vdb"] + out[2:]        # drop the index-instance token
    elif out and out[0] == "corpus":
        out = out[1:]                  # corpus__chunker__x -> chunker.x
    return ".".join(out) if out else name


def _abbr_model(v):
    """Qwen/Qwen2.5-7B-Instruct -> '7B' (round-trips back via LLMAgent._coerce)."""
    if "Qwen" in str(v):
        m = re.match(r".*?(\d+\.?\d*)B", str(v))
        if m:
            return f"{m.group(1)}B"
    return v


def _abbr_value(name, v):
    s = _abbr_model(v)
    if s != v:
        return s
    return round(v, 3) if isinstance(v, float) else v


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort: pull the first top-level JSON object out of an LLM reply."""
    if not text:
        return None
    text = text.strip()
    # strip ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    # Let the JSON decoder, rather than a hand-written brace counter, find the
    # object boundary.  Braces inside a JSON string are data, not structure; the
    # old scanner truncated a perfectly valid reply such as
    # ``{"reasoning": "avoid } here", "candidates": [...]}`` at the brace in
    # ``reasoning``.  Scan later object starts as well so a stray prose/example
    # brace before the requested payload cannot hide an otherwise valid reply.
    decoder = json.JSONDecoder()
    first_object: Optional[dict] = None
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        if first_object is None:
            first_object = value
        # These are the two documented LLMAgent top-level response shapes.
        # Prefer them over an unrelated/example object in surrounding prose.
        if "candidates" in value or "queries" in value:
            return value
    return first_object


# Tool the LLM may call mid-reasoning to interrogate the surrogate before recommending.
# Role (evidence-driven): is_duplicate is RELIABLE; the GP mean/acqf are a FALLIBLE second
# opinion (the GP is often confidently WRONG in regions it hasn't sampled) — the prompt tells
# the LLM NOT to treat low variance as "explored". Configs are passed as knob DELTAS off BASE,
# exactly like the final candidates, so there is one mental + code model.
class LLMAgent:
    """DeepSeek agent: reads history + sub-metrics, proposes candidate configs.

    Returns flat ``{param: value}`` dicts. Each proposal is merged onto the
    best-observed config (so the agent may specify only the knobs it wants to change),
    type-coerced against the search space, encoded + snapped → a valid candidate.
    By default, any failure (API / JSON / encode) degrades to empty advice →
    plain qNEHVI.  ``strict=True`` instead raises :class:`AgentAdviceError`,
    which is the required behavior for attributable paper benchmarks.
    """

    def __init__(
        self,
        model: str = "deepseek-v4-flash",   # DeepSeek OpenAI-compatible endpoint
        n_propose: int = 4,
        recent_k: int = 8,
        temperature: float = 0.7,
        timeout: float = 150.0,            # HTTP timeout — the ~20KB prompt needs >30s default
        strict: bool = False,               # fail the iteration instead of silently removing the agent
        wall_timeout_s: Optional[float] = None,  # hard budget for one complete chat() call
        ai_client: Optional[Any] = None,
        enable_stage_signals: bool = True,
    ):
        if type(enable_stage_signals) is not bool:
            raise ValueError("enable_stage_signals must be a boolean")
        self.model_name = model
        self.n_propose = int(n_propose)
        self.recent_k = int(recent_k)
        self.temperature = float(temperature)
        self.enable_stage_signals = enable_stage_signals
        self.timeout = float(timeout)
        self.strict = _resolve_agent_strict(strict)
        self.wall_timeout_s = _resolve_agent_wall_timeout(
            wall_timeout_s,
            # Preserve ordinary-run timing exactly unless a deadline was
            # explicitly requested. Paper-strict runs need a finite default.
            fallback=self.timeout if self.strict else None,
        )
        self._client = ai_client  # lazily built if None
        # Self-managed memo (07-05): the LLM's own notes from its last advise,
        # rendered back into the next prompt (continuity across rounds). Purely
        # LLM-authored — the data cards remain the ground truth beside it.
        self._notes: str = ""

    # -- client --------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            from rag_stack.ai_clients import get_ai_client
            self._client = get_ai_client(
                f"openai/deepseek/{self.model_name}", timeout=self.timeout
            )
        return self._client

    def _failure(
        self, message: str, exc: Optional[BaseException] = None,
    ) -> AgentAdvice:
        """Raise in paper-strict mode, otherwise preserve legacy degradation."""
        if self.strict:
            logger.error("%s [strict agent mode: aborting iteration]", message)
            error = AgentAdviceError(message)
            if exc is not None:
                raise error from exc
            raise error
        logger.warning(message)
        return AgentAdvice()

    def _chat_with_wall_timeout(
        self, client, messages, *, temperature: Optional[float] = None,
    ):
        """Run one complete client chat under a cancellation deadline.

        The budget surrounds limiter waits, provider retries, and HTTP work;
        the client's internal request timeout alone cannot bound that aggregate.
        A retrieval follow-up is a separate chat call and receives its own
        deadline.
        """
        import anyio

        effective_temperature = (
            self.temperature if temperature is None else float(temperature)
        )

        async def _call():
            if self.wall_timeout_s is None:
                return await client.chat(
                    messages, temperature=effective_temperature,
                )
            with anyio.fail_after(self.wall_timeout_s):
                return await client.chat(
                    messages, temperature=effective_temperature,
                )

        return anyio.run(_call)

    # -- prompt --------------------------------------------------------
    # alias maps (full<->short), cached per ssops
    def _aliases(self, ssops):
        cache = getattr(self, "_alias_cache", None)
        if cache is None:
            cache = self._alias_cache = {}
        key = id(ssops)
        if key not in cache:
            full2short, short2full = {}, {}
            for name in ssops.param_names:
                base = _shorten_name(name)
                s, k = base, 2
                while s in short2full:           # uniqueness guard
                    s = f"{base}#{k}"; k += 1
                full2short[name], short2full[s] = s, name
            norm = {re.sub(r"[^a-z0-9]", "", s.lower()): n for s, n in short2full.items()}
            cache[key] = (full2short, short2full, norm)
        return cache[key]

    def _resolve_key(self, k, ssops):
        """Map an agent-emitted knob name (alias / full / fuzzy) -> real param name."""
        _, short2full, norm = self._aliases(ssops)
        if k in ssops.ss.parameters:
            return k
        if k in short2full:
            return short2full[k]
        return norm.get(re.sub(r"[^a-z0-9]", "", str(k).lower()))

    @staticmethod
    def _varying_knobs(flats):
        keys = set().union(*[set(f) for f in flats]) if flats else set()
        return {
            k for k in keys
            if len({json.dumps(f.get(k), sort_keys=True) for f in flats}) > 1
        }

    def _ordered(self, names, ssops):
        order = {n: i for i, n in enumerate(ssops.param_names)}
        return sorted(names, key=lambda n: order.get(n, 10**9))

    def _fmt_cfg(self, flat, knobs, ssops) -> str:
        f2s, _, _ = self._aliases(ssops)
        return " ".join(
            f"{f2s.get(n, n)}={_abbr_value(n, flat[n])}"
            for n in self._ordered(knobs, ssops) if n in flat
        )

    def _space_card(self, ssops) -> str:
        f2s, _, _ = self._aliases(ssops)
        lines = []
        for name in ssops.param_names:
            p = ssops.ss.parameters[name]
            vals = getattr(p, "values", None)
            short = f2s.get(name, name)
            if vals is not None:
                lines.append(f"  {short}: {[_abbr_value(name, v) for v in vals]}")
            else:
                lines.append(
                    f"  {short}: range[{getattr(p, 'lower', '?')},{getattr(p, 'upper', '?')}]"
                )
        return "\n".join(lines)

    def _dataflow_semantics_card(self, ssops) -> str:
        """Render branch execution semantics from the actual Ax hierarchy.

        The flat search-space card lists every legal knob, but that alone hides
        which stages disappear under a ``rag_dataflow`` choice. Derive those
        relationships from the ChoiceParameter's ``dependents`` and the source
        YAML node graph so the prompt cannot drift away from the executable
        search space.
        """
        try:
            from rag_stack.search_space.rag_search_space import (
                REACT_ACTIVE_STAGES,
                pipeline_method_blocks,
            )

            config = getattr(ssops, "_config", None) or {}
            blocks = {
                str(block.get("mode", "sequential")): block
                for block in pipeline_method_blocks(config)
                if isinstance(block, dict)
            }
            cards = []
            for name in ssops.param_names:
                if ssops.param_leaf_name(name) != "rag_dataflow":
                    continue
                param = ssops.ss.parameters[name]
                values = [str(v) for v in (getattr(param, "values", None) or [])]
                dependents = getattr(param, "dependents", None) or {}
                if len(values) < 2 or not dependents:
                    continue

                prefix, _, _ = name.rpartition("__")
                method = prefix.rsplit("__", 1)[-1]
                block = blocks.get(method)
                if not block:
                    continue

                stages = []
                optional = {}
                for node_line in block.get("node_lines", []) or []:
                    for node in node_line.get("nodes", []) or []:
                        stage = str(node.get("stage") or "")
                        if stage and stage not in stages:
                            stages.append(stage)
                            optional[stage] = bool(node.get("optional", False))
                if not stages:
                    continue

                exclusive = {branch: set() for branch in values}
                branch_controls = {branch: [] for branch in values}
                child_prefix = prefix + "__"
                for branch in values:
                    children = dependents.get(branch, dependents.get(str(branch), [])) or []
                    for child in children:
                        child = str(child)
                        if not child.startswith(child_prefix):
                            continue
                        tail = child[len(child_prefix):]
                        root, _, rest = tail.partition("__")
                        if root in stages:
                            exclusive[branch].add(root)
                        elif root == branch and rest:
                            control = rest.split("__", 1)[0]
                            if control not in branch_controls[branch]:
                                branch_controls[branch].append(control)

                if not any(exclusive.values()) and not any(branch_controls.values()):
                    continue

                lines = [
                    "DATAFLOW EXECUTION SEMANTICS (AUTHORITATIVE; derived from the "
                    "active search-space hierarchy):"
                ]
                for branch in values:
                    inactive = set().union(*(
                        exclusive[other] for other in values if other != branch
                    ))
                    if branch == "react":
                        # Share the evaluator/search-space source of truth so
                        # fixed (non-swept) sequential-only stages are still
                        # reported even though they have no Ax child parameter.
                        inactive.update(
                            stage for stage in stages
                            if stage not in REACT_ACTIVE_STAGES
                        )
                    active = [stage for stage in stages if stage not in inactive]
                    active_text = ", ".join(
                        f"{stage} (if enabled)" if optional.get(stage) else stage
                        for stage in active
                    )
                    lines.append(f"  rag_dataflow={branch}: active stages: {active_text}.")
                    if inactive:
                        ordered_inactive = [stage for stage in stages if stage in inactive]
                        lines.append(
                            "    INACTIVE AND STRIPPED: " + ", ".join(ordered_inactive) + "."
                        )
                    if branch_controls[branch]:
                        lines.append(
                            "    Branch-only controls: "
                            + ", ".join(branch_controls[branch]) + "."
                        )

                    if branch == "react":
                        lines.append(
                            "    Runtime meaning: Thought -> Action -> Observation loop. "
                            "The generator runs once for each unfinished query in a round "
                            "and chooses Search[query] or Finish[answer]. Search invokes "
                            "retrieval and the optional reranker; that Observation is consumed "
                            "by the next generator round. max_iter caps generator rounds (the "
                            "actual count is data-dependent), and a Search on the final allowed "
                            "round cannot add a usable Observation."
                        )
                lines.append(
                    "Knobs below an inactive/stripped stage are no-ops for that branch: "
                    "do not use them as quality or performance levers."
                )
                cards.append("\n".join(lines))
            return "\n\n".join(cards)
        except Exception as exc:  # prompt enrichment must never block optimization
            logger.debug("dataflow semantics card unavailable: %s", exc)
            return ""

    @staticmethod
    def _fmt_attr(attr) -> str:
        """Render raw authoritative-DES diagnostics without inventing a bound.

        In particular, a collocated engine exposes one ``unified`` queue, so
        DES cannot attribute its pressure separately to prefill and decode.
        Fill is resident occupancy/admission pressure, not GPU utilization or
        an independent per-stage QPS ceiling.
        """
        if not attr or attr.get("source") != "authoritative_des":
            return ""

        cells = [
            f"DES qps={float(attr.get('qps') or 0.0):.3g}",
            f"lat={float(attr.get('latency_s') or 0.0):.3g}s",
        ]
        for engine_name, engine in sorted((attr.get("engines") or {}).items()):
            replicas = engine.get("replicas") or {}
            replica_cells = []
            for replica, evidence in sorted(replicas.items()):
                if "fill_fraction" in evidence:
                    replica_cells.append(
                        f"{replica} fill={float(evidence['fill_fraction']):.3f}"
                    )
            identity = str(engine.get("identifiability") or "engine_only")
            if identity == "unified_only":
                replica_cells.append("P/D not separable")
            mode = str(engine.get("pd_serving") or "unknown")
            cells.append(
                f"{engine_name}({mode}):" + ",".join(replica_cells)
            )

        worker_cells = [
            f"{name} batch={float(evidence['batch_mean']):.3g}"
            for name, evidence in sorted((attr.get("workers") or {}).items())
            if "batch_mean" in evidence
        ]
        if worker_cells:
            cells.append("workers:" + ",".join(worker_cells))

        transfer_cells = []
        for name, evidence in sorted((attr.get("transfers") or {}).items()):
            details = []
            if "latency_mean_s" in evidence:
                details.append(
                    f"lat={1000.0 * float(evidence['latency_mean_s']):.3g}ms"
                )
            if "queue_wait_mean_s" in evidence:
                details.append(
                    f"wait={1000.0 * float(evidence['queue_wait_mean_s']):.3g}ms"
                )
            if details:
                transfer_cells.append(f"{name} " + ",".join(details))
        if transfer_cells:
            cells.append("transfer:" + ";".join(transfer_cells))

        saturation = attr.get("saturation") or {}
        cells.append(
            "saturation="
            + ("plateau" if saturation.get("plateau") else "not-proven")
            + f"/{saturation.get('stop_reason', 'unknown')}"
        )
        deployment = attr.get("deployment") or {}
        cells.append(
            f"layout={deployment.get('policy', 'unknown')},"
            f"chips={deployment.get('chips', 0)},"
            f"rb={deployment.get('batch_request', 0)},"
            f"db={deployment.get('batch_decode', 0)}"
        )
        return " [" + " | ".join(cells) + "]"

    def _visible_evidence(self, value):
        """Remove stage diagnostics without changing stored observations.

        Render-time filtering also protects direct callers and ledger follow-up
        requests that bypass the optimizer's already-filtered prompt feed.
        """
        if getattr(self, "enable_stage_signals", True):
            return value
        if isinstance(value, Mapping):
            return {
                key: self._visible_evidence(item)
                for key, item in value.items()
                if key not in {"sub_metrics", "perf_attr"}
            }
        if isinstance(value, (list, tuple)):
            return [self._visible_evidence(item) for item in value]
        return value

    def _obs_card(self, rows, vary, ssops) -> str:
        lines = []
        for h in self._visible_evidence(rows):
            sm = h.get("sub_metrics", {}) or {}
            sm_str = " ".join(
                f"{_SUBMETRIC_SHORT.get(k, k)}={v:.2f}"
                for k, v in sm.items()
                if isinstance(v, (int, float)) and v == v and k != "combined"
            )
            lines.append(
                f"  q={h['correctness']:.3f} perf={h['performance']:.0f}"
                f"{self._fmt_attr(h.get('perf_attr'))} | "
                f"{sm_str} | {self._fmt_cfg(h['flat'], vary, ssops)}"
            )
        return "\n".join(lines)

    def _pareto_card(self, pareto, vary, ssops) -> str:
        """Render the current (performance, quality) Pareto frontier — WITH the
        sub-metrics of each frontier point, so the agent can see WHY each
        trade-off point scores as it does (same columns as RECENT EVALUATIONS)."""
        if not pareto:
            return "  (no Pareto points yet)"
        lines = []
        for p in self._visible_evidence(pareto):
            sm = p.get("sub_metrics", {}) or {}
            sm_str = " ".join(
                f"{_SUBMETRIC_SHORT.get(k, k)}={v:.2f}"
                for k, v in sm.items()
                if isinstance(v, (int, float)) and v == v and k != "combined"
            )
            lines.append(
                f"  perf={p['perf']:.0f} q={p['corr']:.3f}"
                f"{self._fmt_attr(p.get('perf_attr'))} | {sm_str} | "
                f"{self._fmt_cfg(p['flat'], vary, ssops)}"
            )
        return "\n".join(lines)

    def _objectives_card(self, ssops, mo, submetric_keys) -> str:
        """Config-driven objective definitions: WHAT ``quality`` and ``performance``
        actually are for THIS run, read live from the RAG YAML stashed on
        ``ssops._config`` (``gt_evaluation.combined_quality`` + per-metric
        ``description`` + ``system.performance_objectives``). Nothing about the
        scoring is hardcoded — a config that judges quality differently, or
        optimizes latency instead of throughput, re-renders this card. Falls back
        to the bare objective names when the config is unavailable."""
        qn = (mo or {}).get("quality_name", "quality")
        pn = (mo or {}).get("perf_name", "performance")
        cfg = getattr(ssops, "_config", None) or {}
        gt = cfg.get("eval_backend_setting") or {}
        metric_desc = {
            m.get("metric_name"): " ".join(str(m.get("description", "")).split())
            for m in (gt.get("metrics") or [])
        }
        cq = gt.get("combined_quality") or {}
        qmetrics = cq.get("metrics") or []
        qdef = (f"{cq.get('mode', 'mean')} of [{', '.join(qmetrics)}]"
                if qmetrics else f"the run's {qn} score")
        po = (cfg.get("system") or {}).get("performance_objectives") or {}
        perf_sel = po.get("selection")
        if perf_sel == "constrained" and po.get("constraint"):
            # hard SLO: thresholds in natural sense (throughput ≥, latency ≤)
            slo = ", ".join(
                f"{m} {'≥' if m.startswith('throughput') else '≤'} {t}"
                for m, t in po["constraint"].items()
            )
            pdef = (
                f"{po.get('optimize', 'throughput_qps')} SUBJECT TO a hard SLO "
                f"({slo}); a config with NO deployment meeting the SLO scores "
                "worst-possible performance and is wasted — stay inside the SLO"
            )
        else:
            pdef = {
                "max_throughput": "system throughput (queries/sec)",
                "min_latency": "speed = inverse end-to-end latency",
            }.get(perf_sel, "system throughput/speed (cost-model performance score)")
        lines = [
            "OBJECTIVES — the optimizer maximizes the HYPERVOLUME of these two "
            "(both higher = better):",
            f"  - {qn} = {qdef}",
            f"  - {pn} = {pdef}",
        ]
        if not getattr(self, "enable_stage_signals", True):
            return "\n".join(lines)
        # Render every target-declared diagnostic, including metrics that are
        # not present in a historical warm start. System transfer may preserve
        # frozen quality from an older metric set; hiding a newly declared
        # diagnostic until after the first new GT point would make the first
        # agent decision use the wrong experiment semantics. Evaluation cards
        # still show values only where they genuinely exist.
        shown = [k for k in metric_desc if k != "combined"]
        if shown:
            lines.append(
                "INTERMEDIATE SUB-METRICS (diagnostic signals shown per eval when available; NOT optimized "
                "directly). Each description says what the metric captures and which stages TEND "
                "to drive it — but the components COUPLE (retrieval shapes what the generator "
                "sees; context precision trades off against recall; query expansion changes "
                "everything downstream), so use these as hints and let the ACTUAL evals tell you "
                "which knobs move them — do NOT assume a fixed metric→knob mapping:"
            )
            subset_n = int(gt.get("submetric_query_subset") or 0)
            if subset_n > 0:
                lines.append(
                    f"  COHORT WARNING: expensive non-objective diagnostics use the "
                    f"deterministic first {subset_n} queries, while the quality objective "
                    "uses the full evaluation cohort. Their absolute levels are not "
                    "cohort-aligned; use paired changes across configs cautiously."
                )
            for k in shown:
                lines.append(f"  - {_SUBMETRIC_SHORT.get(k, k)} ({k}): {metric_desc[k]}")
        lines.append(
            "PERFORMANCE DIAGNOSTICS: cards contain raw authoritative closed-loop "
            "DES observations only. Engine fill is resident occupancy/admission "
            "pressure, not a standalone QPS bound. A collocated unified engine "
            "cannot identify prefill versus decode; absent worker evidence means "
            "the exact stage cause is unknown."
        )
        return "\n".join(lines)

    def _confidence_line(self, mo, n_history) -> str:
        """Tell the agent HOW RELIABLE the model evidence is, from the optimizer's
        ``model_confidence``. Compares the quality GP's total predictive σ against
        ``quality_std`` (the spread of observed quality): ratio = σ/σ_data is the
        fraction of quality variation the GP CANNOT predict — →0 means it explains
        the data (trust its contrasts), →1 means it predicts no better than the mean
        (noisy/sparse, explore). HIGH/MODERATE/LOW sets exploit-vs-explore."""
        mc = (mo or {}).get("model_confidence") or {}
        n = mc.get("n_evals", n_history)
        sig = mc.get("quality_sigma")
        std = mc.get("quality_std") or 0.0
        lo, hi = mc.get("quality_range", (0.0, 0.0))
        if sig is None or std <= 1e-9:
            return (f"MODEL CONFIDENCE: {n} evals so far (observed quality "
                    f"{lo:.2f}–{hi:.2f}); the surrogate is still warming up — explore broadly.")
        ratio = sig / std
        if ratio < 0.5:
            verdict = ("HIGH — the surrogate predicts quality well (σ well below the data's "
                       "spread); its knob-contrast evidence is reliable, lean on it.")
        elif ratio < 0.85:
            verdict = ("MODERATE — the surrogate predicts quality only roughly; treat the "
                       "knob contrasts as directional and keep probing untried regions.")
        else:
            verdict = ("LOW — the surrogate predicts quality barely better than its average "
                       "(σ ≈ the data's spread); data is noisy/sparse, treat contrasts as "
                       "weak hints and prioritize EXPLORING untried knob values.")
        return (f"MODEL CONFIDENCE: {n} evals so far; the quality surrogate predicts a new "
                f"config's quality to ±σ≈{sig:.3f}, vs σ_data≈{std:.3f} of spread across "
                f"observed quality ({lo:.2f}–{hi:.2f}) → confidence {verdict}")

    def _evidence_branch_text(self, branch: Any, ssops) -> str:
        if not isinstance(branch, Mapping) or not branch:
            return "root" if not branch else str(branch)
        f2s, _, _ = self._aliases(ssops)
        return ",".join(
            f"{f2s.get(name, name)}={_abbr_value(name, value)!r}"
            for name, value in sorted(branch.items())
        )

    def _independent_evidence_card(self, evidence: Any, ssops) -> str:
        """Keep marginal GP screening distinct from landed DOE ablations."""
        payload = evidence if isinstance(evidence, Mapping) else {}
        gp = payload.get("gp_knob_tension")
        doe = payload.get("doe_landed_ablation")
        f2s, _, _ = self._aliases(ssops)

        gp_lines = [
            "GP PER-KNOB MULTIOBJECTIVE TENSION "
            "(MARGINAL SURROGATE SCREENING; NOT A LANDED ABLATION):"
        ]
        gp_rows = gp.get("rows", ()) if isinstance(gp, Mapping) else ()
        if not gp_rows:
            reason = (
                gp.get("reason")
                if isinstance(gp, Mapping)
                else "independent GP evidence was not supplied"
            )
            gp_lines.append(f"  unavailable: {reason or 'no tension contrast'}")
        else:
            gp_lines.append(
                "  Deltas and uncertainty are normalized by realized objective "
                "spans; u is a conservative posterior contrast bound."
            )
            rendered = 0
            for row in gp_rows:
                if not isinstance(row, Mapping):
                    continue
                name = str(row.get("knob", "?"))
                short = f2s.get(name, name)
                anchor_level = _abbr_value(name, row.get("anchor_level"))
                level = _abbr_value(name, row.get("level"))
                anchor_branch = self._evidence_branch_text(
                    row.get("anchor_branch"), ssops,
                )
                candidate_branch = self._evidence_branch_text(
                    row.get("candidate_branch"), ssops,
                )
                branch = (
                    anchor_branch
                    if anchor_branch == candidate_branch
                    else f"{anchor_branch}->{candidate_branch}"
                )
                gp_lines.append(
                    "  "
                    f"{short} {anchor_level!r}->{level!r} "
                    f"anchor={row.get('anchor', '?')} branch={branch} "
                    f"Δperf={float(row.get('delta_performance_norm', 0.0)):+.4f} "
                    f"Δquality={float(row.get('delta_quality_norm', 0.0)):+.4f} "
                    "u=("
                    f"{float(row.get('contrast_uncertainty_performance_norm', 0.0)):.4f},"
                    f"{float(row.get('contrast_uncertainty_quality_norm', 0.0)):.4f}"
                    ") "
                    f"support={int(row.get('support', 0))} "
                    f"levels={int(row.get('levels_considered', 1))} "
                    f"tension={float(row.get('tension', 0.0)):.6f}"
                )
                rendered += 1
            if rendered == 0:
                gp_lines.append("  unavailable: no valid tension rows")

        doe_lines = [
            "DOE LANDED EXACT CHAMPION ABLATIONS "
            "(REALIZED OAT/PAIR EVIDENCE; DISTINCT FROM GP SCREENING):"
        ]
        doe_rows = doe.get("rows", ()) if isinstance(doe, Mapping) else ()
        if not doe_rows:
            reason = (
                doe.get("reason")
                if isinstance(doe, Mapping)
                else "DOE landed-ablation evidence was not supplied"
            )
            doe_lines.append(
                f"  unavailable: {reason or 'no landed champion DOE ablation'}"
            )
        else:
            rendered = 0
            for row in doe_rows:
                if not isinstance(row, Mapping):
                    continue
                changes = row.get("changed_knobs", ())
                rendered_changes = []
                if isinstance(changes, (list, tuple)):
                    for change in changes:
                        if not isinstance(change, Mapping):
                            continue
                        name = str(change.get("parameter", "?"))
                        before = _abbr_value(
                            name, change.get("anchor_level"),
                        )
                        after = _abbr_value(
                            name, change.get("observed_level"),
                        )
                        rendered_changes.append(
                            f"{f2s.get(name, name)}:{before!r}->{after!r}"
                        )
                delta_performance = row.get("delta_performance", 0.0)
                delta_quality = row.get("delta_quality", 0.0)
                delivery = row.get("delivery", "arbitrated")
                interaction = row.get("interaction")
                interaction_text = ""
                if isinstance(interaction, Mapping):
                    if interaction.get("available"):
                        interaction_text = (
                            " interaction=("
                            f"{float(interaction.get('delta_performance', 0.0)):+.4f},"
                            f"{float(interaction.get('delta_quality', 0.0)):+.4f}"
                            ")"
                        )
                    else:
                        interaction_text = (
                            " interaction=unavailable"
                            f"(main_effect_evals="
                            f"{interaction.get('main_effect_eval_ids', [])})"
                        )
                doe_lines.append(
                    "  "
                    f"{str(row.get('kind', '?')).upper()} "
                    f"side={row.get('side', '?')} "
                    f"branch={row.get('branch_id', 'root')} "
                    f"changes={','.join(rendered_changes) or '?'} "
                    f"Δperf={float(delta_performance):+.4f} "
                    f"Δquality={float(delta_quality):+.4f} "
                    f"delivery={delivery} eval={row.get('eval_id', '?')}"
                    f"{interaction_text}"
                )
                rendered += 1
            if rendered == 0:
                doe_lines.append("  unavailable: no valid landed DOE rows")

        return "\n".join([*gp_lines, "", *doe_lines])

    def _coverage_card(self, history, vary, ssops, max_lines=28) -> str:
        """GLOBAL COVERAGE: per varying knob, every allowed value with
        (times tried, best q, best perf). The LLM's global view of the search:
        under-explored values are visible instead of falling out of the
        recency window, and a single bad reading is exposed as n=1 (possibly
        poisoned by OTHER knobs in that config) rather than a verdict."""
        from collections import defaultdict
        per_knob: dict = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0.0]))
        for h in history:
            for k, v in (h.get("flat") or {}).items():
                if k not in vary:
                    continue
                cell = per_knob[k][str(v)]
                cell[0] += 1
                cell[1] = max(cell[1], float(h.get("correctness") or 0.0))
                cell[2] = max(cell[2], float(h.get("performance") or 0.0))
        params = getattr(getattr(ssops, "ss", None), "parameters", {}) or {}
        lines = []
        for k in sorted(per_knob):
            vals = list(getattr(params.get(k), "values", []) or [])
            cells = []
            tried = per_knob[k]
            universe = [str(v) for v in vals] if vals else sorted(tried)
            for v in universe:
                if v in tried:
                    n, bq, bp = tried[v]
                    cells.append(f"{v}({n}x,q{bq:.2f},p{bp:.0f})")
                else:
                    cells.append(f"{v}(UNTRIED)")
            # keep enough context to disambiguate (module + knob tail)
            tail = "/".join(k.split("__")[-2:])
            lines.append(f"  {tail}: " + " ".join(cells))
            if len(lines) >= max_lines:
                break
        return "\n".join(lines) if lines else "  (no data yet)"

    @staticmethod
    def _ledger_card(history, vary, ssops, f2s) -> str:
        """GLOBAL LEDGER (07-05, user-approved design): EVERY evaluated config,
        one line each, in MODE-BASELINE DIFF encoding — a BASE profile (each
        varying knob's most common value across history) is printed once and
        every row lists only its deviations. Row-level global memory at
        ~20-25 tokens/row; the interaction structure (which flip moved what)
        is visible instead of compressed into per-knob marginals. All content
        computed from run data — nothing hardcoded."""
        from collections import Counter
        if not history:
            return ""
        vary = sorted(vary)
        mode = {k: Counter(
            str(_abbr_value(k, h["flat"].get(k))) for h in history if k in h["flat"]
        ).most_common(1)[0][0] for k in vary}
        base = "  ".join(f"{f2s.get(k, k)}={mode[k]}" for k in vary)
        lines = [
            "GLOBAL LEDGER (EVERY eval this run; rows show only knobs that DIFFER "
            "from LEDGER-BASE; e-id = evaluation order):",
            f"  LEDGER-BASE: {base}",
        ]
        for i, h in enumerate(history):
            diffs = []
            for k in vary:
                if k not in h["flat"]:
                    continue
                av = str(_abbr_value(k, h["flat"][k]))
                if av != mode[k]:
                    diffs.append(f"{f2s.get(k, k)}={av}")
            d = "{" + ",".join(diffs) + "}" if diffs else "{=base}"
            lines.append(
                f"  e{i:02d} {d} → perf={_round(h['performance'])} q={_round(h['correctness'])}"
            )
        return "\n".join(lines)

    @staticmethod
    def _signature_card(history, vary, ssops, f2s, top_knobs: int = 3, max_groups: int = 12) -> str:
        """REGION ROLLUP (07-05): group the full history by the JOINT values of
        the top-``top_knobs`` most decision-relevant knobs (ranked by how much
        their value choice moves perf+quality — computed from this run's data)
        and report each region's count/perf-range/quality-range. This is the
        scale-proof directory over the ledger: marginals hide interactions,
        joint signatures keep them."""
        if len(history) < 4 or not vary:
            return ""
        perfs = [h["performance"] for h in history]
        quals = [h["correctness"] for h in history]
        p_rng = (max(perfs) - min(perfs)) or 1.0
        q_rng = (max(quals) - min(quals)) or 1.0

        def _knob_score(k):
            groups = {}
            for h in history:
                if k in h["flat"]:
                    groups.setdefault(str(_abbr_value(k, h["flat"][k])), []).append(h)
            if len(groups) < 2:
                return 0.0
            pm = [sum(x["performance"] for x in g) / len(g) for g in groups.values()]
            qm = [sum(x["correctness"] for x in g) / len(g) for g in groups.values()]
            return (max(pm) - min(pm)) / p_rng + (max(qm) - min(qm)) / q_rng

        keys = sorted(vary, key=_knob_score, reverse=True)[:top_knobs]
        if not keys:
            return ""
        regions = {}
        for h in history:
            sig = "|".join(str(_abbr_value(k, h["flat"].get(k))) for k in keys)
            regions.setdefault(sig, []).append(h)
        hdr = "|".join(f2s.get(k, k) for k in keys)
        lines = [
            f"REGION ROLLUP (history grouped by [{hdr}] — the {len(keys)} knobs whose "
            "values move perf/quality most in THIS run's data; ranges are observed "
            "min→max per region):"
        ]
        ordered = sorted(regions.items(), key=lambda kv: -len(kv[1]))
        for sig, g in ordered[:max_groups]:
            gp = [x["performance"] for x in g]
            gq = [x["correctness"] for x in g]
            lines.append(
                f"  [{sig}] n={len(g)}  perf {_round(min(gp))}→{_round(max(gp))}  "
                f"q {_round(min(gq))}→{_round(max(gq))}"
            )
        rest = ordered[max_groups:]
        if rest:
            lines.append(f"  [+{len(rest)} smaller regions omitted]")
        return "\n".join(lines)

    def _infeasible_card(self, infeasible, ssops, f2s) -> str:
        """INFEASIBLE ARMS (07-05): configs that FAILED evaluation, with the raw
        error text. Pure data — no hand-written rules; the LLM induces the
        constraint (e.g. which knob combinations overflow the context window)
        and stops proposing into it."""
        if not infeasible:
            return ""
        lines = [
            "INFEASIBLE ARMS (these configs FAILED evaluation and were wasted "
            "budget — do not propose configs that repeat the same failing "
            "structure; induce the constraint from the error text):"
        ]
        for rec in infeasible:
            flat = rec.get("config") or {}
            try:
                shown = " ".join(
                    f"{f2s.get(k, k)}={_abbr_value(k, v)}"
                    for k, v in flat.items()
                ) or "(config unavailable)"
            except Exception:  # noqa: BLE001
                shown = str(flat)[:120]
            lines.append(f"  {shown[:150]}")
            lines.append(f"    → {rec.get('reason', '')[:150]}")
        return "\n".join(lines)

    @staticmethod
    def _submetric_corr_card(history) -> str:
        """SUB-METRIC RELIABILITY (07-05, user directive): sub-metrics are NOT
        assumed to track the final quality score — their Spearman correlation
        with it is COMPUTED from this run's data and shown, so the LLM leans
        only on the ones that actually predict quality here (a high
        ctx_recall with near-zero correlation is noise, not a lever)."""
        import numpy as _np
        rows = [h for h in history if h.get("sub_metrics")]
        if len(rows) < 6:
            return ""
        keys = sorted({k for r in rows for k in r["sub_metrics"]})
        q = _np.array([r["correctness"] for r in rows], dtype=float)
        if _np.std(q) < 1e-9:
            return ""
        def _spear(a, b):
            ra = _np.argsort(_np.argsort(a)).astype(float)
            rb = _np.argsort(_np.argsort(b)).astype(float)
            sa, sb = _np.std(ra), _np.std(rb)
            if sa < 1e-9 or sb < 1e-9:
                return 0.0
            return float(_np.corrcoef(ra, rb)[0, 1])
        lines = ["SUB-METRIC RELIABILITY (Spearman correlation of each sub-metric "
                 "with the FINAL quality score, computed on THIS run's evals — "
                 "trust a sub-metric as a lever ONLY if it correlates here; "
                 "low/negative = it does NOT drive quality on this data):"]
        cells = []
        for k in keys:
            vals, qs = [], []
            for r in rows:
                if k in r["sub_metrics"]:
                    try:
                        fv = float(r["sub_metrics"][k])
                    except (TypeError, ValueError):
                        continue
                    if fv == fv:  # NaN would poison the rank correlation
                        vals.append(fv); qs.append(float(r["correctness"]))
            if len(vals) >= 6:
                cells.append(
                    f"{_SUBMETRIC_SHORT.get(k, k)}="
                    f"{_spear(_np.array(vals), _np.array(qs)):+.2f}"
                    f"(evals={len(vals)})"
                )
        if not cells:
            return ""
        lines.append("  " + "  ".join(cells))
        return "\n".join(lines)

    @staticmethod
    def _toggle_effects_card(history, vary, f2s) -> str:
        """TOGGLE EFFECTS (07-05, user directive): enabling a module is NOT
        assumed to help any metric — the observed mean quality/perf with the
        toggle on vs off (this run's data) is shown instead."""
        import numpy as _np

        def _is_off(v) -> bool:
            return str(v).lower() in ("false", "0", "none", "off")

        lines = []
        for k in sorted(vary):
            if not k.endswith("__enabled"):
                continue
            on_q, off_q, on_p, off_p = [], [], [], []
            for h in history:
                v = h["flat"].get(k)
                if v is None:
                    continue
                (off_q if _is_off(v) else on_q).append(h["correctness"])
                (off_p if _is_off(v) else on_p).append(h["performance"])
            if len(on_q) >= 3 and len(off_q) >= 3:
                lines.append(
                    f"  {f2s.get(k, k)}: ON mean q={_np.mean(on_q):.3f} perf={_np.mean(on_p):.1f} (n={len(on_q)})"
                    f" | OFF mean q={_np.mean(off_q):.3f} perf={_np.mean(off_p):.1f} (n={len(off_q)})"
                )
        if not lines:
            return ""
        return ("TOGGLE EFFECTS (observed means THIS run — enabling a module is "
                "not assumed to help; read what it actually did):\n" + "\n".join(lines))

    def _run_ledger_queries(self, queries, history, ssops) -> str:
        """AGENTIC MEMORY RETRIEVAL (07-05, user directive): execute the LLM's
        ledger queries against the full history and return the evidence text.

        Grammar (forgiving):
          * ``detail e08 e22``      — full row dump (active knobs + sub-metrics)
          * ``<knob><op><value> [& ...]`` with op ∈ {=, !=, <=, >=} — matching
            rows in ledger format. Knob names may be leaf names (top_k) or
            full/partial paths; value comparison is numeric when both sides
            parse, else string equality.
        Results are capped; malformed queries return an error line (the LLM
        can still propose without them)."""
        history = self._visible_evidence(history)
        f2s, _, _ = self._aliases(ssops)
        out: List[str] = []

        def _fmt_row(i, h):
            active = ssops.active_params(h["flat"])
            knobs = " ".join(
                f"{f2s.get(k, k)}={_abbr_value(k, v)}"
                for k, v in sorted(h["flat"].items()) if k in active
            )
            sm = " ".join(f"{_SUBMETRIC_SHORT.get(k, k)}={_round(v)}"
                          for k, v in (h.get("sub_metrics") or {}).items())
            return (f"e{i:02d} perf={_round(h['performance'])} "
                    f"q={_round(h['correctness'])} | {knobs}"
                    + (f" | {sm}" if sm else ""))

        def _resolve_knob(token):
            token = token.strip()
            names = [n for n in ssops.param_names if n == token]
            if not names:
                names = [n for n in ssops.param_names if n.endswith("__" + token) or token in n]
            return names[0] if len(names) == 1 else (names or None)

        for q in list(queries)[:4]:
            q = str(q).strip()
            out.append(f"QUERY: {q}")
            try:
                if q.lower().startswith("detail"):
                    ids = [int(t.lstrip("eE")) for t in q.split()[1:] if t.lstrip("eE").isdigit()]
                    for i in ids[:6]:
                        if 0 <= i < len(history):
                            out.append("  " + _fmt_row(i, history[i]))
                    continue
                clauses = [c.strip() for c in q.split("&") if c.strip()]
                preds = []
                ok = True
                for c in clauses:
                    for op in ("!=", "<=", ">=", "="):
                        if op in c:
                            k, v = c.split(op, 1)
                            name = _resolve_knob(k)
                            if not isinstance(name, str):
                                out.append(f"  ! knob '{k.strip()}' ambiguous/unknown"
                                           + (f" (candidates: {name[:3]})" if name else ""))
                                ok = False
                            else:
                                preds.append((name, op, v.strip()))
                            break
                    else:
                        out.append(f"  ! cannot parse clause '{c}'")
                        ok = False
                if not ok or not preds:
                    continue
                def _match(h):
                    for name, op, v in preds:
                        hv = h["flat"].get(name)
                        try:
                            a, b = float(hv), float(v)
                            res = {"=": a == b, "!=": a != b, "<=": a <= b, ">=": a >= b}[op]
                        except (TypeError, ValueError):
                            sa, sb = str(hv), str(v)
                            res = {"=": sa == sb, "!=": sa != sb,
                                   "<=": sa <= sb, ">=": sa >= sb}[op]
                        if not res:
                            return False
                    return True
                hits = [(i, h) for i, h in enumerate(history) if _match(h)]
                for i, h in hits[:20]:
                    out.append("  " + _fmt_row(i, h))
                if len(hits) > 20:
                    out.append(f"  … {len(hits) - 20} more rows omitted")
                if not hits:
                    out.append("  (no matching evals)")
            except Exception as exc:  # noqa: BLE001
                out.append(f"  ! query failed: {exc}")
        return "\n".join(out)

    def _build_messages(self, history, ssops, mo=None) -> List[dict]:
        history = self._visible_evidence(history)
        mo = self._visible_evidence(mo)
        stage_signals = getattr(self, "enable_stage_signals", True)
        rows = sorted(history[-self.recent_k :], key=lambda h: h["correctness"], reverse=True)
        best_flat = max(history, key=lambda h: h["correctness"])["flat"]
        pareto = (mo or {}).get("pareto") or []
        # Hierarchical masking: show only ACTIVE knobs per config — a disabled or
        # non-selected module's knobs are dead (no functional effect), and showing
        # them feeds the LLM spurious correlations ("hyde.model=14B scored well"
        # while hyde was never enabled). NOTE: deltas still merge onto the FULL
        # best_flat, so the agent can re-enable modules and their knobs get values.
        def _af(f):
            act = ssops.active_params(f)
            return {k: v for k, v in f.items() if k in act}

        rows = [dict(r, flat=_af(r["flat"])) for r in rows]
        pareto = [dict(p, flat=_af(p["flat"])) for p in pareto]
        shown_best = _af(best_flat)
        # The GLOBAL cards (coverage/ledger/signature/toggles) must see the
        # same active-only masking — raw history leaks dead-knob phantom
        # values ("reranker.model=bge scored 0.52" from rows where the
        # reranker was disabled) into exactly the cards meant to steer
        # exploration.
        history_shown = [dict(h, flat=_af(h["flat"])) for h in history]
        # column set: only knobs that actually vary across what we show (the full BASE
        # config below carries every knob's value, so nothing is lost).
        flats = [r["flat"] for r in rows] + [p["flat"] for p in pareto] + [shown_best]
        vary = self._varying_knobs(flats) or set(ssops.param_names)
        sub_keys = sorted({k for r in rows for k in (r.get("sub_metrics") or {})})
        objectives = self._objectives_card(ssops, mo, sub_keys)
        space = self._space_card(ssops)
        dataflow_semantics = self._dataflow_semantics_card(ssops)
        dataflow_semantics_block = (
            f"{dataflow_semantics}\n\n" if dataflow_semantics else ""
        )
        base = self._fmt_cfg(shown_best, ssops.param_names, ssops)
        obs = self._obs_card(rows, vary, ssops)
        coverage = self._coverage_card(history_shown, vary, ssops)
        _f2s, _, _ = self._aliases(ssops)
        # global-vision cards (07-05): full-run row memory + region directory.
        # vary is computed over the SHOWN rows; the ledger spans ALL history, so
        # widen to knobs that vary anywhere in the run.
        _all_flats = [h["flat"] for h in history_shown]
        _vary_all = self._varying_knobs(_all_flats) or vary
        ledger = self._ledger_card(history_shown, _vary_all, ssops, _f2s)
        signature = self._signature_card(history_shown, _vary_all, ssops, _f2s)
        # perf-sorted twin of the (quality-sorted) recent card (07-05, user
        # directive: the row-level view was systematically quality-biased)
        perf_rows = sorted(history, key=lambda h: h["performance"], reverse=True)[: self.recent_k]
        perf_rows = [dict(r, flat=_af(r["flat"])) for r in perf_rows]
        perf_card = self._obs_card(perf_rows, vary, ssops)
        smcorr = self._submetric_corr_card(history)
        toggles = self._toggle_effects_card(history_shown, _vary_all, _f2s)
        infeasible_card = self._infeasible_card((mo or {}).get("infeasible") or [], ssops, _f2s)
        confidence = self._confidence_line(mo, len(history))
        intel = (f"{confidence}\n\n" if confidence else "")
        intel += self._independent_evidence_card(
            (mo or {}).get("independent_evidence"), ssops,
        ) + "\n\n"
        hier_note = (
            " Toggle knobs gate the rest: knobs of a module that is disabled "
            "(enabled=false) or not selected by its .type knob are INACTIVE — they are "
            "hidden from the cards and setting them has NO effect unless you also flip "
            "the toggle in the SAME candidate. "
        )
        final_schema = (
            '{"reasoning": "<1-2 sentences>", '
            '"search_notes": "<up to 120 words — YOUR updated running notes: validated '
            'hypotheses, dead ends to stop repeating, and the next bets. They are fed '
            'back to you next round>", '
            '"candidates": [{"<knob>": <value>}, ...]}'
        )
        cand_schema = (
            'EITHER (to inspect the full eval history first) '
            '{"queries": ["<knob><op><value> [& ...] with op in =,!=,<=,>= — e.g. '
            'rag_dataflow=react & top_k<=4", "detail e08 e22", ...]} — the matching '
            "rows (every eval, not just the windows above) come back and you answer "
            "again; use it whenever the cards are not enough evidence. OR the final "
            "answer: " + final_schema
        )
        delta_rules = (hier_note +
            "For EACH candidate output ONLY the knobs that DIFFER from its anchor; "
            "do NOT repeat unchanged knobs. Each candidate may declare its anchor with "
            '"base": "quality" (default — deltas apply on BASE) or "base": "perf" (deltas '
            "apply on BASE_PERF). PERFORMANCE-side proposals MUST use \"base\": \"perf\" — "
            "otherwise every knob you omit silently inherits the best-QUALITY config and your "
            "fast-side intent is lost (this exact failure was observed). Use knob names and "
            'values exactly as in the SEARCH SPACE (models by size, e.g. "7B"). Respond with '
            "ONLY a JSON object:\n"
            + cand_schema
        )
        perf_flat = max(history, key=lambda h: h["performance"])["flat"]
        shown_perf = _af(perf_flat)
        base_perf = self._fmt_cfg(shown_perf, ssops.param_names, ssops)
        base_block = (
            "BASE (best-QUALITY evaluated config; the default delta anchor):\n"
            f"  {base}\n"
            "BASE_PERF (best-PERFORMANCE evaluated config; anchor for perf-side deltas):\n"
            f"  {base_perf}\n\n"
        )
        tradeoff_evidence = (
            "intermediate sub-metrics" if stage_signals else "observed final objectives"
        )
        quality_evidence = (
            "evaluations and sub-metrics" if stage_signals else "evaluations"
        )
        system = (
            "You are an expert at tuning Retrieval-Augmented-Generation (RAG) pipelines "
            "for a MULTI-OBJECTIVE Bayesian optimizer. The optimizer jointly maximizes the "
            "two objectives defined in the OBJECTIVES section below — its true target is the "
            "HYPERVOLUME of the (performance, quality) Pareto front. A config that maximizes "
            "quality but is very slow does NOT improve the hypervolume. Read the past "
            "evaluations and the current Pareto frontier, diagnose trade-offs using the "
            f"{tradeoff_evidence}, then propose a DIVERSE set of NEW configs that EXTEND "
            "the frontier — some pushing quality, some trading a little quality for much "
            "higher performance, some filling gaps. Avoid configs the frontier already "
            "dominates. IMPORTANT: do NOT assume generic RAG heuristics hold (e.g. 'a "
            "reranker / query-expansion / compressor always helps quality') — let the "
            f"{quality_evidence} tell you what ACTUALLY drives quality on THIS data, "
            "even when that contradicts conventional RAG wisdom."
        )
        # The normalized empty rectangle is the actual missing-area geometry
        # for every ordinary Agent proposal.
        _rect = (mo or {}).get("hv_rectangle") or {}
        _rect_line = ""
        if _rect.get("target"):
            _target = _rect["target"]
            _rect_line = (
                "MAX-HV EMPTY RECTANGLE (largest normalized unclaimed dominated area "
                "between adjacent frontier points): "
                f"perf in [{_rect['perf_lo']:.1f}, {_rect['perf_hi']:.1f}], "
                f"quality in [{_rect['q_lo']:.3f}, {_rect['q_hi']:.3f}] "
                f"(normalized area {_rect['area_norm']:.3f}); aim near "
                f"perf≈{_target['perf']:.0f}, q≈{_target['quality']:.3f}.\n\n"
            )
        recent_heading = (
            "RECENT EVALUATIONS (best quality first; sub-metrics are intermediate signals; "
            "only varying knobs shown):\n"
            if stage_signals else
            "RECENT EVALUATIONS (best quality first; only varying knobs shown):\n"
        )
        performance_guidance = (
            "PERF ATTRIBUTION: every perf number is that config's PERFORMANCE CEILING — "
            "the cost model sweeps chips/TP/PP/batching per config and reports the "
            "OPTIMUM, deterministically (same config ⇒ same perf; rerunning cannot "
            "improve it — only a DIFFERENT config moves perf). The '[<stage>-bound ...]' tag lists each "
            "stage's standalone qps under that deployment; the smallest (the bound stage) "
            "caps the whole pipeline — to raise perf, attack the bound stage (smaller/"
            "faster generator, fewer react rounds, disable that stage, smaller top_k, "
            "cheaper index).\n\n"
            if stage_signals else
            "PERFORMANCE OBJECTIVE: every perf number is the cost-model optimum "
            "over deployments for that configuration. The score is deterministic; "
            "changing the configuration is required to improve it. Use observed "
            "objective changes to guide performance-side proposals.\n\n"
        )
        end_guidance = (
            "(flip the ONE knob that most relieves the bound stage on the proven-fastest "
            "structure — see its PERF ATTRIBUTION tag) or the quality end (beat the current "
            if stage_signals else
            "(change a knob on the proven-fastest structure using observed objective "
            "trade-offs) or the quality end (beat the current "
        )
        diagnosis_guidance = (
            "  (c) DIAGNOSE FROM THE DATA, not from RAG folklore: use a sub-metric as a "
            "lever ONLY where SUB-METRIC RELIABILITY shows it correlates with the final "
            "quality score on THIS run; do NOT assume enabling a module improves any "
            "metric — read TOGGLE EFFECTS / the LEDGER for what actually happened when "
            "it was toggled. Measured outcomes override every prior you have.\n"
            if stage_signals else
            "  (c) DIAGNOSE FROM THE DATA, not from RAG folklore: use observed final "
            "quality and performance. Do NOT assume enabling a module improves either "
            "objective — read TOGGLE EFFECTS / the LEDGER for what actually happened "
            "when it was toggled. Measured outcomes override every prior you have.\n"
        )
        user = (
            f"{objectives}\n\n"
            "SEARCH SPACE (knob: allowed values; models shown by size, 7B = Qwen2.5-7B-Instruct):\n"
            f"{space}\n\n"
            f"{dataflow_semantics_block}"
            f"{base_block}"
            f"{intel}"
            "CURRENT PARETO FRONTIER (perf=performance score, q=quality score; only varying knobs shown):\n"
            f"{self._pareto_card(pareto, vary, ssops)}\n\n"
            f"{recent_heading}"
            f"{obs}\n\n"
            f"{performance_guidance}"
            "GLOBAL COVERAGE (per knob: value(times tried, best q, best perf) — the "
            "GLOBAL state of the search. UNTRIED values are free information; an n=1 "
            "reading may be poisoned by OTHER knobs in that config, so a promising value "
            "with a single bad reading deserves a CLEAN retry):\n"
            f"{coverage}\n\n"
            + (f"{ledger}\n\n" if ledger else "")
            + (f"{signature}\n\n" if signature else "")
            + (f"{infeasible_card}\n\n" if infeasible_card else "")
            + ("TOP PERFORMANCE EVALS (best perf first — same columns; these are "
               "the proven fast structures to build perf-side proposals from):\n"
               f"{perf_card}\n\n" if perf_card else "")
            + (f"{smcorr}\n\n" if smcorr else "")
            + (f"{toggles}\n\n" if toggles else "")
            + (
                "YOUR SEARCH NOTES (written by YOU last round — your running "
                "world-model of this search. Trust the data cards over stale "
                "notes when they disagree):\n"
                f"{self._notes}\n\n"
                if self._notes else ""
            )
            + _rect_line
            + f"Propose {self.n_propose} NEW configs that maximize HYPERVOLUME by COVERING "
            "THE TRADE-OFF CURVE — a good Pareto set is well-SPREAD, not clustered. With "
            "both ENDS usually already anchored by the initial probes, the biggest gains "
            "are the SPARSE MIDDLE, so:\n"
            "  (a) FILL THE MAX-HV RECTANGLE (top priority): land at least one config in "
            "the MAX-HV EMPTY RECTANGLE shown above — a balanced trade-off between its "
            "two endpoints (e.g. a mid-size generator, a moderate top_k, one optional stage on). "
            "Read the two bracketing frontier points and the LEDGER for which knob settings "
            "sit between them, and interpolate. This is where the most hypervolume is "
            "unclaimed.\n"
            "  (b) EXTEND AN END only if it is NOT yet anchored: push the performance end "
            f"{end_guidance}"
            "best quality) — but ONLY if that end is still improving; do not re-propose a "
            "saturated corner the frontier already covers.\n"
            f"{diagnosis_guidance}"
            f"Avoid configs the frontier already dominates. {delta_rules}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _build_gap_exploration_messages(
        self,
        request: GapExplorationRequest,
        history: List[dict],
        ssops: AxSearchSpaceOps,
        independent_evidence: Optional[Mapping[str, Any]] = None,
    ) -> List[dict]:
        """Render a single-purpose maximum-gap exploration request.

        Unlike the ordinary advisory prompt, this call is made only by the
        realized-HV-stagnation forced policy. It receives the normalized gap
        geometry as immutable input and asks for exploration candidates; it
        does not grant a direct evaluation seat or make a scheduling decision.
        """

        q_anchor = request.quality_anchor
        p_anchor = request.performance_anchor
        stage_signals = getattr(self, "enable_stage_signals", True)
        f2s, _, _ = self._aliases(ssops)

        def _active(flat: Mapping[str, Any]) -> Tuple[Dict[str, Any], set]:
            decoded = dict(flat)
            active = ssops.active_params(decoded)
            return (
                {name: decoded[name] for name in ssops.param_names
                 if name in active and name in decoded},
                active,
            )

        q_flat, q_active = _active(q_anchor.flat)
        p_flat, p_active = _active(p_anchor.flat)
        differing = []
        for name in ssops.param_names:
            q_live = name in q_active
            p_live = name in p_active
            if not (q_live or p_live):
                continue
            q_value = q_anchor.flat.get(name) if q_live else "<inactive>"
            p_value = p_anchor.flat.get(name) if p_live else "<inactive>"
            if q_live != p_live or q_value != p_value:
                differing.append(
                    f"  {f2s.get(name, name)}: quality_anchor={q_value!r}; "
                    f"performance_anchor={p_value!r}"
                )
        diff_card = "\n".join(differing) if differing else "  (none)"

        def _submetric_card(anchor: ParetoGapAnchor) -> str:
            if not anchor.sub_metrics:
                return "{}"
            return json.dumps(
                dict(sorted(anchor.sub_metrics.items())),
                sort_keys=True,
                default=str,
            )

        def _cm_card(anchor: ParetoGapAnchor) -> str:
            if not anchor.perf_attr:
                return "(unavailable)"
            concise = ""
            try:
                concise = self._fmt_attr(anchor.perf_attr)
            except Exception:  # an audit payload need not be a pricing dict
                concise = ""
            raw = json.dumps(anchor.perf_attr, sort_keys=True, default=str)
            if len(raw) > 2000:
                raw = raw[:1997] + "..."
            return ((concise + " ") if concise else "") + raw

        failed_lines: List[str] = []
        # The controller orders this collection by relevance to the current
        # gap. Bound the prompt without introducing a search-policy knob.
        for idx, item in enumerate(self._visible_evidence(request.failed_nearby[:8])):
            flat = item.get("flat") or item.get("config") or {}
            if isinstance(flat, Mapping):
                try:
                    shown, _ = _active(flat)
                    cfg = self._fmt_cfg(shown, shown.keys(), ssops)
                except Exception:
                    cfg = json.dumps(flat, sort_keys=True, default=str)
            else:
                cfg = str(flat)
            context = {
                key: value for key, value in item.items()
                if key not in {"flat", "config"}
            }
            detail = json.dumps(context, sort_keys=True, default=str)
            failed_lines.append(f"  f{idx}: {cfg}" + (f" | {detail}" if context else ""))
        failed_card = "\n".join(failed_lines) if failed_lines else "  (none supplied)"

        hierarchy = self._dataflow_semantics_card(ssops)
        hierarchy_block = (
            "HIERARCHY / ACTIVE-BRANCH SEMANTICS:\n" + hierarchy + "\n\n"
            if hierarchy else
            "HIERARCHY / ACTIVE-BRANCH SEMANTICS:\n"
            "  Follow the Ax search-space dependencies. A child knob is usable only "
            "when its selector branch is active.\n\n"
        )
        independent_evidence_block = self._independent_evidence_card(
            self._visible_evidence(independent_evidence), ssops,
        )
        q_diagnostics = (
            f"  quality submetrics: {_submetric_card(q_anchor)}\n"
            f"  CM bottleneck/audit: {_cm_card(q_anchor)}\n"
            if stage_signals else ""
        )
        p_diagnostics = (
            f"  quality submetrics: {_submetric_card(p_anchor)}\n"
            f"  CM bottleneck/audit: {_cm_card(p_anchor)}\n"
            if stage_signals else ""
        )
        schema = (
            '{"gap_id": ' + json.dumps(request.gap_id) + ', '
            '"reasoning": "<brief evidence-grounded diagnosis>", '
            '"candidates": ['
            '{"base": "quality_anchor|performance_anchor", '
            '"deltas": {"<knob>": "<allowed value>"}, '
            '"hypothesis": "<why this explores the empty gap>"}'
            ']}'
        )
        system = (
            "You design a bounded FORCED EXPLORATION for a stalled, multi-objective "
            "optimizer. The two supplied points are the endpoints of its largest "
            "realized normalized Pareto gap. Explore the empty trade-off region "
            "between them; do not exploit a known endpoint and do not assume generic "
            "domain folklore. Every proposal must be justified by the supplied "
            "observations, valid in the declared hierarchy, and distinct from tested "
            "or failed configurations. You only design candidates; another policy "
            "validates, ranks, and may decline all of them."
        )
        user = (
            f"GAP ID: {request.gap_id}\n"
            "NORMALIZED REALIZED GAP (current realized-history scale):\n"
            f"  quality_anchor: performance={q_anchor.normalized_performance:.6f}, "
            f"quality={q_anchor.normalized_quality:.6f}\n"
            f"  performance_anchor: performance={p_anchor.normalized_performance:.6f}, "
            f"quality={p_anchor.normalized_quality:.6f}\n\n"
            "QUALITY ANCHOR — complete active configuration:\n"
            f"  {self._fmt_cfg(q_flat, q_flat.keys(), ssops)}\n"
            f"{q_diagnostics}\n"
            "PERFORMANCE ANCHOR — complete active configuration:\n"
            f"  {self._fmt_cfg(p_flat, p_flat.keys(), ssops)}\n"
            f"{p_diagnostics}\n"
            "ACTIVE KNOB DIFFERENCES BETWEEN THE ANCHORS:\n"
            f"{diff_card}\n\n"
            "FAILED OR NON-IMPROVING NEARBY CONFIGURATIONS (do not repeat):\n"
            f"{failed_card}\n\n"
            f"{independent_evidence_block}\n\n"
            "SEARCH SPACE (knob: allowed values):\n"
            f"{self._space_card(ssops)}\n\n"
            f"{hierarchy_block}"
            f"Return at most {self.n_propose} diverse candidates. For each candidate, "
            "choose exactly one supplied anchor and provide a non-empty set of "
            "explicit deltas containing only knobs that differ from that anchor. "
            "A branch change and the child value needed on that branch belong in the "
            "same candidate. Prefer an unobserved compatible interaction, an untried "
            "intermediate/third level, or a one-sided graft whose evidence suggests it "
            "could land inside the gap. Do not merely copy or make a no-op change to an "
            "anchor. Use knob names and values exactly as listed above.\n\n"
            "Respond with ONLY this exact JSON shape (no queries, markdown, notes, or "
            f"additional keys):\n{schema}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _gap_exploration_empty(
        self,
        request: GapExplorationRequest,
        message: str,
        exc: Optional[BaseException] = None,
    ) -> AgentAdvice:
        """Fail strict runs or return an auditable non-strict empty result."""

        if self.strict:
            detail = f"LLMAgent gap exploration unavailable: {message}"
            logger.error("%s [strict agent mode: aborting iteration]", detail)
            error = AgentAdviceError(detail)
            if exc is not None:
                raise error from exc
            raise error
        logger.warning("LLMAgent gap exploration unavailable: %s", message)
        return AgentAdvice(
            rationale=f"gap exploration unavailable: {message}"[:500],
            raw={
                "kind": "gap_exploration",
                "status": "unavailable",
                "gap_id": request.gap_id,
                "error": str(message)[:500],
            },
        )

    # -- coercion ------------------------------------------------------
    @staticmethod
    def _coerce(name: str, value: Any, ssops: AxSearchSpaceOps) -> Any:
        """Match an LLM-emitted value to a valid value of param ``name``."""
        p = ssops.ss.parameters[name]
        vals = getattr(p, "values", None)
        if vals is not None:
            allowed = list(vals)
            if value in allowed:
                return value
            sval = str(value).strip()
            for v in allowed:
                if str(v) == sval:
                    return v
            # model-size abbreviation: '7B' -> Qwen/Qwen2.5-7B-Instruct
            for v in allowed:
                av = _abbr_model(v)
                if av != v and str(av).lower() == sval.lower():
                    return v
            try:
                fv = float(value)
                # nearest allowed numeric
                num = [v for v in allowed if isinstance(v, (int, float))]
                if num:
                    return min(num, key=lambda v: abs(float(v) - fv))
            except (TypeError, ValueError):
                pass
            return None  # unmatchable → drop this key
        # RangeParameter
        try:
            fv = float(value)
            lo, hi = float(p.lower), float(p.upper)
            fv = max(lo, min(hi, fv))
            return int(round(fv)) if getattr(p, "parameter_type", None) and "INT" in str(p.parameter_type) else fv
        except (TypeError, ValueError):
            return None

    def advise(self, history, ssops, t, mo=None) -> AgentAdvice:
        """Build the prompt, call the LLM once, parse its candidates → AgentAdvice."""
        if not history:
            return AgentAdvice()
        return self._advise_oneshot(history, ssops, t, mo)

    def _advise_oneshot(self, history, ssops, t, mo=None) -> AgentAdvice:
        try:
            messages = self._build_messages(history, ssops, mo)
            client = self._get_client()
            reply = self._chat_with_wall_timeout(client, messages)
        except Exception as exc:  # API / network / auth
            return self._failure(
                "LLMAgent: initial chat call failed "
                f"({type(exc).__name__}: {exc}); skipping agent this iter",
                exc,
            )
        try:
            parsed = _extract_json(reply or "")
        except Exception as exc:  # compatible clients may return a non-string payload
            return self._failure(
                "LLMAgent: initial reply JSON parsing failed "
                f"({type(exc).__name__}: {exc}); skipping agent this iter",
                exc,
            )
        # AGENTIC MEMORY RETRIEVAL (07-05): one bounded evidence round — the
        # LLM may answer with {"queries": [...]} instead of candidates; we run
        # them against the FULL history and ask again with the results. This is
        # always available and deliberately not a configurable policy: the
        # second response is parsed directly, so a third round cannot occur.
        if (
            parsed
            and isinstance(parsed.get("queries"), list)
            and not parsed.get("candidates")
        ):
            try:
                evidence = self._run_ledger_queries(parsed["queries"], history, ssops)
                logger.info(
                    f"[agent t={t}] retrieval round: {len(parsed['queries'])} queries, "
                    f"{evidence.count(chr(10)) + 1} evidence lines"
                )
                messages = messages + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content":
                        "LEDGER QUERY RESULTS:\n" + evidence +
                        "\n\nNow respond with ONLY the final JSON object "
                        "(reasoning / search_notes / candidates) — no further queries."},
                ]
                reply = self._chat_with_wall_timeout(client, messages)
                parsed = _extract_json(reply or "")
            except Exception as exc:  # noqa: BLE001
                if self.strict:
                    return self._failure(
                        "LLMAgent: retrieval round failed "
                        f"({type(exc).__name__}: {exc})",
                        exc,
                    )
                logger.warning(
                    f"LLMAgent: retrieval round failed ({exc}); using first reply"
                )
            if not parsed or not isinstance(parsed.get("candidates"), list):
                return self._failure(
                    "LLMAgent: retrieval round produced no parseable candidate list; "
                    "skipping agent this iter"
                )
        if not parsed or not isinstance(parsed.get("candidates"), list):
            return self._failure(
                "LLMAgent: could not parse candidates from reply; skipping agent this iter"
            )
        try:
            return self._build_advice_from_parsed(parsed, history, ssops)
        except AgentAdviceError:
            raise
        except Exception as exc:  # candidate schema/coercion/encoding
            return self._failure(
                "LLMAgent: candidate parsing failed "
                f"({type(exc).__name__}: {exc}); skipping agent this iter",
                exc,
            )

    def explore_gap(
        self,
        request: GapExplorationRequest,
        history: List[dict],
        ssops: AxSearchSpaceOps,
        t: int,
        independent_evidence: Optional[Mapping[str, Any]] = None,
    ) -> AgentAdvice:
        """Ask the existing Agent for maximum-gap exploration candidates.

        This is intentionally independent of :meth:`advise` and does not
        mutate the Agent's ordinary search notes.
        An unavailable or malformed response aborts a strict paper iteration
        before a GT seat is committed; non-strict callers receive auditable
        empty advice and may continue with ordinary arbitration.
        """

        if not isinstance(request, GapExplorationRequest):
            raise TypeError("request must be a GapExplorationRequest")
        try:
            messages = self._build_gap_exploration_messages(
                request,
                history,
                ssops,
                independent_evidence=independent_evidence,
            )
            reply = self._chat_with_wall_timeout(
                self._get_client(), messages, temperature=0.0,
            )
        except Exception as exc:  # API / network / auth / prompt rendering
            return self._gap_exploration_empty(
                request, f"chat failed ({type(exc).__name__}: {exc})",
                exc,
            )
        try:
            parsed = _extract_json(reply or "")
            return self._build_gap_advice_from_parsed(
                parsed, request, history, ssops,
            )
        except AgentAdviceError:
            raise
        except Exception as exc:  # strict schema / encode / novelty rejection
            return self._gap_exploration_empty(
                request, f"response rejected ({type(exc).__name__}: {exc})",
                exc,
            )

    def _build_gap_advice_from_parsed(
        self,
        parsed: Any,
        request: GapExplorationRequest,
        history: List[dict],
        ssops: AxSearchSpaceOps,
    ) -> AgentAdvice:
        """Validate, encode, canonicalize, and deduplicate a gap response."""

        if not isinstance(parsed, dict):
            raise ValueError("reply is not a JSON object")
        expected_top = {"gap_id", "reasoning", "candidates"}
        if set(parsed) != expected_top:
            raise ValueError(
                f"top-level keys must be exactly {sorted(expected_top)}"
            )
        if str(parsed["gap_id"]) != request.gap_id:
            raise ValueError("reply gap_id does not match the active gap")
        if not isinstance(parsed["reasoning"], str):
            raise ValueError("reasoning must be a string")
        raw_candidates = parsed["candidates"]
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise ValueError("candidates must be a non-empty list")
        if len(raw_candidates) > max(1, self.n_propose * 2):
            raise ValueError("candidate list exceeds the bounded response contract")

        normalized: List[Tuple[str, Dict[str, Any], str]] = []
        expected_candidate = {"base", "deltas", "hypothesis"}
        for candidate in raw_candidates:
            if not isinstance(candidate, dict) or set(candidate) != expected_candidate:
                raise ValueError(
                    "each candidate must contain exactly base, deltas, hypothesis"
                )
            base = str(candidate["base"])
            if base not in {"quality_anchor", "performance_anchor"}:
                raise ValueError(
                    "candidate base must be quality_anchor or performance_anchor"
                )
            deltas = candidate["deltas"]
            if not isinstance(deltas, dict) or not deltas:
                raise ValueError("candidate deltas must be a non-empty object")
            hypothesis = candidate["hypothesis"]
            if not isinstance(hypothesis, str) or not hypothesis.strip():
                raise ValueError("candidate hypothesis must be a non-empty string")

            # Unlike the ordinary best-effort advisor, the forced interface
            # must never silently drop a misspelled knob or uncoercible value
            # and evaluate a different / partial experiment.
            for key, value in deltas.items():
                full = self._resolve_key(key, ssops)
                if full is None:
                    raise ValueError(f"unknown knob {key!r}")
                if self._coerce(full, value, ssops) is None:
                    raise ValueError(f"invalid value for knob {key!r}")
            normalized.append((base, dict(deltas), hypothesis.strip()[:500]))

        anchors = {
            "quality_anchor": request.quality_anchor.flat,
            "performance_anchor": request.performance_anchor.flat,
        }

        existing_rows: List[Tensor] = []
        existing_flats: List[Any] = [
            request.quality_anchor.flat,
            request.performance_anchor.flat,
            *(h.get("flat") for h in history if isinstance(h, dict)),
            *(
                item.get("flat") or item.get("config")
                for item in request.failed_nearby
            ),
        ]
        for flat in existing_flats:
            if not isinstance(flat, Mapping) or not flat:
                continue
            row = ssops.snap_to_grid(
                ssops.encode_arm_params(dict(flat)).reshape(1, -1)
            )
            existing_rows.append(ssops.clamp(row))
        existing = (
            torch.cat(existing_rows, dim=0)
            if existing_rows
            else torch.empty((0, ssops.dim), **ssops.tkwargs)
        )

        centers: List[Tensor] = []
        proposal_meta: List[Dict[str, Any]] = []
        for base, deltas, hypothesis in normalized:
            row, meta = self._encode_proposal_with_metadata(
                deltas, ssops, anchors[base],
            )
            if row is None:
                continue
            # Canonical active-branch identity is the novelty contract used by
            # the optimizer's final gate as well.
            row = ssops.clamp(ssops.snap_to_grid(row))
            if ssops.is_duplicate_decoded(row, existing):
                continue
            centers.append(row)
            proposal_meta.append({
                "kind": "gap_exploration",
                "gap_id": request.gap_id,
                "base": base,
                "hypothesis": hypothesis,
                **meta,
            })
            existing = torch.cat([existing, row], dim=0)
            if len(centers) >= self.n_propose:
                break

        if not centers:
            raise ValueError(
                "all candidates were invalid, duplicate, or no-op",
            )
        return AgentAdvice(
            centers=torch.cat(centers, dim=0),
            proposal_meta=proposal_meta,
            rationale=str(parsed["reasoning"])[:500],
            raw={
                "kind": "gap_exploration",
                "status": "ok",
                "gap_id": request.gap_id,
                "response": parsed,
            },
        )

    def _encode_proposal_with_metadata(
        self, prop, ssops, best_flat,
    ) -> "Tuple[Optional[Tensor], Dict[str, Any]]":
        """Encode a sparse delta and retain its auditable semantic intent."""
        if not isinstance(prop, dict):
            return None, {"deltas": {}, "inferred_ancestors": {}}
        merged = dict(best_flat)
        resolved: Dict[str, Any] = {}
        for k, v in prop.items():
            if k == "base":
                continue
            full = self._resolve_key(k, ssops)
            if full is None:
                continue
            cv = self._coerce(full, v, ssops)
            if cv is None:
                continue
            merged[full] = cv
            resolved[full] = cv
        try:
            # Sparse deltas may name a child of a currently inactive branch
            # without repeating every selector on the path (for example, set
            # a component-specific model while the anchor has that optional
            # stage disabled).  Use Ax's declared dependency tree to infer the
            # unique ancestor choices that make the explicit intent live.
            # This is schema-driven: no stage, component, or knob names are
            # special-cased.  Contradictory or ambiguous paths fail this one
            # proposal closed instead of evaluating a different config.
            inferred_ancestors: Dict[str, Any] = {}
            pending = list(resolved)
            visited: set[str] = set()
            while pending:
                child_name = pending.pop()
                if child_name in visited:
                    continue
                visited.add(child_name)

                parent_matches: Dict[str, List[Any]] = {}
                for parent_name, parameter in ssops.ss.parameters.items():
                    if not getattr(parameter, "is_hierarchical", False):
                        continue
                    for branch, children in (parameter.dependents or {}).items():
                        if child_name not in {str(name) for name in (children or [])}:
                            continue
                        branch_value = self._coerce(parent_name, branch, ssops)
                        if branch_value is not None:
                            parent_matches.setdefault(parent_name, []).append(branch_value)

                if not parent_matches:
                    continue
                if len(parent_matches) != 1:
                    raise ValueError(
                        f"hierarchical child {child_name!r} has ambiguous parents "
                        f"{sorted(parent_matches)}"
                    )

                parent_name, candidates = next(iter(parent_matches.items()))
                unique: List[Any] = []
                for value in candidates:
                    if value not in unique:
                        unique.append(value)
                current = merged.get(parent_name)
                if current in unique:
                    selected = current
                elif parent_name in resolved:
                    raise ValueError(
                        f"explicit selector {parent_name}={current!r} disables "
                        f"explicit child {child_name!r}"
                    )
                elif parent_name in inferred_ancestors:
                    raise ValueError(
                        f"explicit children require conflicting {parent_name!r} "
                        f"branches {inferred_ancestors[parent_name]!r} and {unique!r}"
                    )
                elif len(unique) == 1:
                    selected = unique[0]
                    merged[parent_name] = selected
                    inferred_ancestors[parent_name] = selected
                else:
                    raise ValueError(
                        f"hierarchical child {child_name!r} requires one of "
                        f"multiple {parent_name!r} branches {unique!r}"
                    )
                pending.append(parent_name)

            x = ssops.snap_to_grid(
                ssops.encode_arm_params(merged).reshape(1, -1)
            )
            active = ssops.active_params(merged)
            inactive_intent = sorted(
                name for name in {*resolved, *inferred_ancestors}
                if name not in active
            )
            if inactive_intent:
                raise ValueError(
                    "hierarchical activation left explicit intent inactive: "
                    f"{inactive_intent}"
                )
            return x, {
                "deltas": resolved,
                "inferred_ancestors": inferred_ancestors,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLMAgent: rejected contradictory delta proposal (%s)", exc)
            return None, {
                "deltas": resolved,
                "inferred_ancestors": {},
                "error": str(exc)[:220],
            }

    def _encode_proposal(self, prop, ssops, best_flat) -> Optional[Tensor]:
        """Encode one proposal row for initialization-probe callers."""
        row, _metadata = self._encode_proposal_with_metadata(
            prop, ssops, best_flat,
        )
        return row

    def _build_advice_from_parsed(self, parsed, history, ssops) -> AgentAdvice:
        """Parsed JSON candidates → AgentAdvice. Candidate ORDER is preserved (the
        LLM-driven pick takes the first non-duplicate). Each candidate may declare
        ``"base": "perf"`` to merge its deltas onto the best-PERFORMANCE evaluated
        config instead of the best-quality one (07-05: perf-side intent was being
        silently swallowed by the quality-anchored default)."""
        rationale = str(parsed.get("reasoning", ""))[:500]
        notes = str(parsed.get("search_notes", "") or "")[:1200]
        if notes:
            self._notes = notes
        best_flat = max(history, key=lambda h: h["correctness"])["flat"]
        perf_flat = max(history, key=lambda h: h["performance"])["flat"]
        centers = []
        proposal_meta: List[Dict[str, Any]] = []
        for raw_prop in parsed["candidates"][: self.n_propose * 2]:
            # Never mutate ``parsed``: it is retained as ``AgentAdvice.raw``
            # for audit and may be consumed by another policy layer.
            prop = dict(raw_prop) if isinstance(raw_prop, dict) else raw_prop
            anchor = best_flat
            anchor_name = "quality"
            if isinstance(prop, dict):
                _b = str(prop.pop("base", "") or "").lower()
                if _b.startswith("perf"):
                    anchor = perf_flat
                    anchor_name = "perf"
            x, meta = self._encode_proposal_with_metadata(
                prop, ssops, anchor,
            )
            if x is not None:
                centers.append(x)
                proposal_meta.append({"base": anchor_name, **meta})
        if not centers:
            advice = self._failure(
                "LLMAgent: all proposals failed to encode; skipping agent this iter"
            )
            advice.rationale = rationale
            return advice
        return AgentAdvice(
            centers=torch.cat(centers, dim=0),
            proposal_meta=proposal_meta,
            rationale=rationale,
            notes=notes,
            raw=parsed,
        )
