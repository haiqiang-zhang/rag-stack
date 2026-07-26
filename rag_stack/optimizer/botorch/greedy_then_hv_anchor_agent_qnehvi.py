"""Greedy warm start plus expected-HV-gain agent qNEHVI (v3).

Phase contract (the base wrapper layer lives in this same module — the
v1/v2 standalone registration was removed 07-07):

* first spend a successful-eval budget with greedy forward;
* then run agent-assisted qNEHVI, with sub-metrics used only in the prompt;

v3 replaces v2's hand-written veto stack (quantile routes, q_floor,
empty-fallback — measured on s44: pool collapsed 294→1 for 18 rounds and 14
picks produced one front hit) with a single principled selection layer:

* every pool candidate is scored by its EXPECTED HV GAIN — a
  per-front-segment EI sum over the quality posterior at the GP perf mean;
* the pool is RANKED, never vetoed: top-K plus all LLM-family candidates
  survive (floor 8) and qNEHVI arbitrates among them;
* the FINAL pick is qLogNEHVI's argmax over the survivors — there is no
  post-acqf rerank layer (the synthetic-trace oracle channel and its
  dormant CM backfill plumbing were deleted — 07-06 flip-pricing forensics:
  21-52% APE on trace-affecting transforms).

The one retained hard rule is the cold-start guard: for the first few
post-warm-start picks, the slow-corner width-expansion reject prevents v1's
quality-corner death spiral.
"""

from __future__ import annotations

from copy import deepcopy
import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from rag_stack.optimizer.agent import (
    AgentAdvice,
    AgentGuidedQNEHVI,
    LLMAgent,
)
from rag_stack.optimizer.agent.llm_agent import _SUBMETRIC_SHORT
from rag_stack.optimizer.agent.util import _pareto_mask
from rag_stack.search_space.ax_search_space_ops import AxSearchSpaceOps
from ..base import BaseOptimizer
from ..greedy import GreedyParetoOptimizer

logger = logging.getLogger("RAG-Stack")


# ═════════════════════════════════════════════════════════════════════════
# BASE LAYER — formerly greedy_then_anchor_agent_qnehvi.py (v1/v2).
# Merged 07-07 (user directive: one file, v3 only). The v1/v2 wrapper is NO
# LONGER a registered optimizer; these classes survive purely as the base
# layer of the v3 method below.
#
# Greedy-warm-started anchor-agent qNEHVI optimizer.
#
# This optimizer is intentionally separate from ``agent_qnehvi``:
#
# * phase 1 runs the existing greedy forward planner for a fixed GT budget;
# * phase 2 runs a new anchor-refinement LLM agent over the existing qNEHVI pool
#   machinery, warm-started from the greedy archive;
# * sub-metrics are prompt-only diagnostics and never enter the GP.
# ═════════════════════════════════════════════════════════════════════════


class GreedyAnchorLLMAgent(LLMAgent):
    """LLM proposal agent specialized for post-greedy Pareto refinement.

    The agent receives numbered real Pareto anchors (P0, P1, ...), emits local
    deltas from those anchors, and leaves final selection to qNEHVI.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_anchor_flats: Dict[str, dict] = {}

    @staticmethod
    def _configured_diagnostic_metric_names(ssops: AxSearchSpaceOps) -> set[str]:
        cfg = getattr(ssops, "_config", None) or {}
        gt = cfg.get("eval_backend_setting") or {}
        metrics = gt.get("metrics") or []
        combined = set(((gt.get("combined_quality") or {}).get("metrics")) or [])
        names = {m.get("metric_name") for m in metrics if m.get("metric_name")}
        return names - combined

    def _active_metric_names(self, ssops: AxSearchSpaceOps) -> set[str]:
        return self._configured_diagnostic_metric_names(ssops)

    def _filter_sm(self, sm: Optional[dict]) -> dict:
        return {
            k: v for k, v in (sm or {}).items()
            if isinstance(v, (int, float)) and v == v  # v == v: NaN never reaches the prompt
        }

    def _metric_descriptions(self, ssops: AxSearchSpaceOps) -> str:
        cfg = getattr(ssops, "_config", None) or {}
        metrics = ((cfg.get("eval_backend_setting") or {}).get("metrics")) or []
        desc = {
            m.get("metric_name"): " ".join(str(m.get("description", "")).split())
            for m in metrics
            if m.get("metric_name")
        }
        lines = [
            "PROMPT-ONLY DIAGNOSTIC SUB-METRICS (do not optimize directly; "
            "use only to explain or repair scalar quality failures):"
        ]
        for name in sorted(self._active_metric_names(ssops)):
            short = _SUBMETRIC_SHORT.get(name, name)
            text = desc.get(name, "Configured diagnostic metric.")
            lines.append(f"  - {short} ({name}): {text}")
        return "\n".join(lines)

    def _sm_str(self, sm: Optional[dict]) -> str:
        kept = self._filter_sm(sm)
        if not kept:
            return "diagnostics=n/a"
        return " ".join(
            f"{_SUBMETRIC_SHORT.get(k, k)}={float(v):.2f}"
            for k, v in sorted(kept.items())
        )

    def _anchor_card(self, pareto, vary, ssops) -> str:
        if not pareto:
            return "  (no Pareto anchors yet)"
        self._last_anchor_flats = {}
        lines = []
        for i, p in enumerate(pareto):
            key = f"p{i}"
            self._last_anchor_flats[key] = dict(p["flat"])
            lines.append(
                f"  P{i}: perf={p['perf']:.0f} q={p['corr']:.3f} "
                f"| {self._sm_str(p.get('sub_metrics'))} | "
                f"{self._fmt_cfg(p['flat'], vary, ssops)}"
            )
        return "\n".join(lines)

    def _obs_card_compact(self, rows, vary, ssops) -> str:
        if not rows:
            return "  (no evaluations yet)"
        lines = []
        for idx, h in rows:
            lines.append(
                f"  e{idx:02d}: perf={h['performance']:.0f} "
                f"q={h['correctness']:.3f} | {self._sm_str(h.get('sub_metrics'))} | "
                f"{self._fmt_cfg(h['flat'], vary, ssops)}"
            )
        return "\n".join(lines)

    def _move_ledger_card(self, history, ssops) -> str:
        """EXPLORE/EXPLOIT accounting (07-07): label every past eval
        MECHANICALLY — its minimum active-knob distance to the THEN-current
        Pareto front (≤2 = EXPLOIT/refinement, >2 = EXPLORE/jump) — and show
        each class's realized payoff (front entries, ΔHV under the final
        normalization, best hit). The LLM then sets the round's mix from
        measured evidence instead of hand-tuned special-case rules; the s45
        vein pattern (a paid-off explore whose neighborhood went unmined) is
        readable directly off this card."""
        if len(history) < 3:
            return ""
        p_rng = max(1e-9, max(h["performance"] for h in history))
        q_rng = max(1e-9, max(h["correctness"] for h in history))

        def _hv(rows) -> float:
            pts = sorted(
                ((h["performance"] / p_rng, h["correctness"] / q_rng) for h in rows),
                key=lambda t: -t[0],
            )
            s, prev_q = 0.0, 0.0
            for x, y in pts:
                if y > prev_q:
                    s += x * (y - prev_q)
                    prev_q = y
            return s

        def _dist(a: dict, b: dict) -> int:
            try:
                keys = set(ssops.active_params(a)) | set(ssops.active_params(b))
            except Exception:  # noqa: BLE001 — fall back to shared keys
                keys = set(a) | set(b)
            return sum(1 for k in keys if str(a.get(k)) != str(b.get(k)))

        stats = {
            "X": {"n": 0, "hits": 0, "dhv": 0.0, "best": (None, 0.0)},
            "E": {"n": 0, "hits": 0, "dhv": 0.0, "best": (None, 0.0)},
        }
        labels: List[str] = []
        hv_prev = _hv(history[:1])
        for i in range(1, len(history)):
            prior = history[:i]
            y = np.array([[h["performance"], h["correctness"]] for h in prior])
            front = [prior[j] for j in range(len(prior)) if _pareto_mask(y)[j]]
            d = min(_dist(history[i]["flat"], f["flat"]) for f in front)
            lab = "X" if d <= 2 else "E"
            hv_now = _hv(history[: i + 1])
            dhv = max(0.0, hv_now - hv_prev)
            hv_prev = hv_now
            st = stats[lab]
            st["n"] += 1
            st["dhv"] += dhv
            if dhv > 1e-9:
                st["hits"] += 1
                if dhv > st["best"][1]:
                    st["best"] = (i, dhv)
            labels.append(lab)

        def _line(name, lab):
            st = stats[lab]
            if st["n"] == 0:
                return f"  {name}: 0 evals"
            best = (f" (best e{st['best'][0]:02d} +{st['best'][1]:.3f})"
                    if st["best"][0] is not None else "")
            return (f"  {name}: {st['n']} evals, {st['hits']} advanced the front, "
                    f"ΔHV +{st['dhv']:.3f}{best}")

        tail = " ".join(labels[-10:])
        return (
            "MOVE LEDGER (mechanical: EXPLOIT = ≤2 active-knob diff from the "
            "then-current front, EXPLORE = >2; ΔHV in final-normalized units):\n"
            f"{_line('EXPLOIT', 'X')}\n"
            f"{_line('EXPLORE', 'E')}\n"
            f"  last moves (old→new): {tail}"
        )

    def _build_messages(self, history, ssops, mo=None) -> List[dict]:
        pareto = list((mo or {}).get("pareto") or [])
        if not pareto and history:
            y = np.array([[h["performance"], h["correctness"]] for h in history])
            idxs = [i for i in range(len(history)) if _pareto_mask(y)[i]]
            idxs.sort(key=lambda i: history[i]["performance"])
            pareto = [
                {
                    "flat": history[i]["flat"],
                    "perf": history[i]["performance"],
                    "corr": history[i]["correctness"],
                    "sub_metrics": history[i].get("sub_metrics", {}),
                }
                for i in idxs
            ]

        best_q = max(history, key=lambda h: h["correctness"])
        best_p = max(history, key=lambda h: h["performance"])
        self._last_anchor_flats = {
            "quality": dict(best_q["flat"]),
            "perf": dict(best_p["flat"]),
        }

        shown = [h["flat"] for h in history[-self.recent_k:]]
        shown += [p["flat"] for p in pareto]
        shown += [best_q["flat"], best_p["flat"]]
        vary = self._varying_knobs(shown) or set(ssops.param_names)
        recent_idx_rows = list(enumerate(history))[-self.recent_k:]
        recent_by_q = sorted(recent_idx_rows, key=lambda x: x[1]["correctness"], reverse=True)
        recent_by_p = sorted(recent_idx_rows, key=lambda x: x[1]["performance"], reverse=True)

        anchor_card = self._anchor_card(pareto, vary, ssops)
        # _anchor_card resets the dict; add named anchors back after numbering.
        self._last_anchor_flats["quality"] = dict(best_q["flat"])
        self._last_anchor_flats["perf"] = dict(best_p["flat"])
        # Champion-style diagnostic usage (v3): the live Spearman card of each
        # sub-metric vs final quality — the baseline agent that quoted these
        # numbers before naming a knob is the only one observed diagnosing
        # correctly; the anchor agents never cited a value without it.
        smcorr = self._submetric_corr_card(history)
        smcorr_block = f"{smcorr}\n\n" if smcorr else ""
        # Full-history channels (base-agent machinery, inherited): the GLOBAL
        # LEDGER shows EVERY eval in mode-baseline diff form (~20-25 tok/row —
        # the 8-row recent window otherwise hides ~60% of a 50-eval run and
        # exclusion evidence like "all character-chunk react rows are flat"
        # silently ages out); the INFEASIBLE card stops re-proposing structures
        # that already failed; ledger QUERIES (see prompt) give on-demand full
        # detail via the inherited one-round retrieval in _advise_oneshot.
        _f2s, _, _ = self._aliases(ssops)
        vary_all = self._varying_knobs([h["flat"] for h in history]) or vary
        ledger_card = (
            self._ledger_card(history, vary_all, ssops, _f2s)
            if len(history) <= 80 else ""   # token gate for far bigger budgets
        )
        infeasible_card = self._infeasible_card(
            (mo or {}).get("infeasible") or [], ssops, _f2s)
        move_ledger = self._move_ledger_card(history, ssops)

        system = (
            "You are the proposal agent for a post-greedy multi-objective RAG "
            "optimizer. Greedy has already spent the first budget building a real "
            "Pareto archive. Your job is only to propose local, anchor-based "
            "candidate deltas. qNEHVI will arbitrate every proposal together with "
            "Sobol/DLS/oracle candidates; you never decide what is evaluated."
        )
        user = (
            "TARGET: maximize hypervolume over scalar performance and scalar quality. "
            "The GP/acquisition uses only those two scalar objectives. The diagnostic "
            "sub-metrics below are prompt-only context and must not be treated as "
            "objectives.\n\n"
            f"{self._metric_descriptions(ssops)}\n\n"
            f"{smcorr_block}"
            "SEARCH SPACE (knob: allowed values; model sizes abbreviate Qwen2.5 variants):\n"
            f"{self._space_card(ssops)}\n\n"
            "NUMBERED REAL PARETO ANCHORS (choose base from P0/P1/... or perf/quality):\n"
            f"{anchor_card}\n\n"
            "BEST QUALITY ANCHOR:\n"
            f"  {self._fmt_cfg(best_q['flat'], ssops.param_names, ssops)}\n"
            "BEST PERFORMANCE ANCHOR:\n"
            f"  {self._fmt_cfg(best_p['flat'], ssops.param_names, ssops)}\n\n"
            + (f"{ledger_card}\n\n" if ledger_card else "")
            + (f"{infeasible_card}\n\n" if infeasible_card else "")
            + (f"{move_ledger}\n\n" if move_ledger else "")
            + "RECENT EVALS BY QUALITY:\n"
            f"{self._obs_card_compact(recent_by_q, vary, ssops)}\n\n"
            "RECENT EVALS BY PERFORMANCE:\n"
            f"{self._obs_card_compact(recent_by_p, vary, ssops)}\n\n"
            "RULES:\n"
            "  - Output only JSON.\n"
            "  - Each candidate must include a base: P0/P1/.../perf/quality.\n"
            "  - Output only active knobs that differ from that base; avoid "
            "repeating unchanged values.\n"
            "  - Propose a diverse set: fast-anchor refinement, quality-anchor "
            "refinement, knee refinement, and one local escape if justified.\n"
            "  - Do not chase a diagnostic metric unless it plausibly repairs scalar "
            "quality or extends the Pareto front.\n"
            "  - When you modify an anchor, cite its weakest diagnostic VALUE "
            "(e.g. 'P2 ctx_recall=0.31') and say how your delta improves it.\n"
            "  - Do not assume reranker/compressor/query expansion helps; only use "
            "them when the real archive supports that local move.\n"
            "  - EXPLORE/EXPLOIT: set this round's mix from the MOVE LEDGER's "
            "measured payoffs and state it in 'reasoning' (e.g. 'exploit 4 / "
            "explore 2'). A paid-off EXPLORE is the strongest exploit signal "
            "there is — mine the causal knob's remaining UNTRIED values (its "
            "whole ladder, not the adjacent step) before stepping further out; "
            "quality jumps within ~0.03 are judge noise, not veins. Several "
            "front-less EXPLOITs in a row = the neighborhood is mined out; "
            "shift budget to EXPLORE.\n\n"
            "Before proposing you may OPTIONALLY reply with ONLY "
            '{"queries": ["<knob><op><value> [& ...]" or "detail e08 e22", ...]} '
            "(ops =,!=,<=,>=; at most 4) — the matching FULL-HISTORY rows come "
            "back and you then answer with the final JSON. Use this to check a "
            "hypothesis against ALL evals (the ledger above shows every row in "
            "diff form; queries return full detail).\n\n"
            "Schema:\n"
            '{"reasoning": "<1-2 sentences>", '
            '"search_notes": "<brief persistent notes>", '
            '"candidates": [{"base": "P0", "<knob>": <value>}, ...]}\n'
            f"Use exact knob names or aliases from the cards. Propose {self.n_propose} candidates."
        )
        if self._notes:
            user += (
                "\n\nYOUR PRIOR SEARCH NOTES (discard if contradicted by the archive):\n"
                f"{self._notes}"
            )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _build_advice_from_parsed(self, parsed, history, ssops) -> AgentAdvice:
        rationale = str(parsed.get("reasoning", ""))[:500]
        notes = str(parsed.get("search_notes", "") or "")[:1200]
        if notes:
            self._notes = notes

        best_q = max(history, key=lambda h: h["correctness"])["flat"]
        best_p = max(history, key=lambda h: h["performance"])["flat"]
        anchors = dict(self._last_anchor_flats)
        anchors.setdefault("quality", best_q)
        anchors.setdefault("perf", best_p)

        centers = []
        proposal_meta = []
        for raw in parsed.get("candidates", [])[: self.n_propose * 2]:
            if not isinstance(raw, dict):
                continue
            prop = dict(raw)
            if "base" not in prop:
                continue
            base = str(prop.pop("base") or "").lower()
            if base.startswith("p") and base[1:].isdigit():
                base = f"p{int(base[1:])}"
            anchor = anchors.get(base)
            if anchor is None:
                continue
            for meta in ("reason", "rationale", "tag", "notes"):
                prop.pop(meta, None)
            x, meta = self._encode_proposal_with_metadata(
                prop, ssops, anchor,
            )
            if x is not None:
                centers.append(x)
                proposal_meta.append({"base": base, **meta})
        if not centers:
            logger.warning("GreedyAnchorLLMAgent: all proposals failed to encode; skipping")
            return AgentAdvice(rationale=rationale)
        return AgentAdvice(
            centers=torch.cat(centers, dim=0),
            proposal_meta=proposal_meta,
            rationale=rationale,
            notes=notes,
            raw=parsed,
        )


class AnchorAgentQNEHVI(AgentGuidedQNEHVI):
    """New phase-2 optimizer: old qNEHVI pool machinery, new agent policy."""

    def _create_agent(
        self,
        *,
        model: str,
        strict: bool,
        wall_timeout_s: Optional[float],
    ) -> LLMAgent:
        return GreedyAnchorLLMAgent(
            model=model,
            strict=strict,
            wall_timeout_s=wall_timeout_s,
        )

    def _allowed_submetrics(self) -> set[str]:
        cfg = getattr(self.ssops, "_config", None) or {}
        gt = cfg.get("eval_backend_setting") or {}
        metric_rows = gt.get("metrics") or []
        configured = {m.get("metric_name") for m in metric_rows if m.get("metric_name")}
        combined = set(((gt.get("combined_quality") or {}).get("metrics")) or [])
        return set(configured - combined)

class GreedyThenAnchorAgentQNEHVI(BaseOptimizer):
    """Run greedy forward first, then a new anchor-agent qNEHVI refinement."""

    def __init__(
        self,
        config_space,
        objectives: List[dict],
        seed: int = 42,
        greedy_budget: int = 30,
        n_iterations: Optional[int] = None,
        greedy_search_mode: str = "forward",
        greedy_beam_width: int = 3,
        greedy_stage_weight: str = "log",
        greedy_pass_budget_fraction: float = 0.6,
        greedy_max_passes: int = 8,
        greedy_plateau_window: int = 0,
        greedy_plateau_eps: float = 0.005,
        greedy_stage_stop_after: int = 2,
        # M4 (07-07, surrogate-validated on the forked s44 landscape: 4W/1L,
        # mean +12%, floor +32%, variance halved — a floor-lifting variance
        # reducer): within each stage quota, probe the SIBLING branch's proven
        # values first. Default OFF — flip per config.
        greedy_sibling_seeding: bool = False,
        ssops: Optional[AxSearchSpaceOps] = None,
        **kwargs,
    ):
        self.seed = int(seed)
        self.greedy_budget = max(1, int(greedy_budget))
        # Global HV-plateau handoff: DEFAULT OFF (window=0). A trailing-window
        # plateau cannot distinguish "this stage is a dud" from "everything
        # remaining is a dud", and FORWARD greedy visits stages once in order —
        # replayed on the s44 trajectory it fires at eval 11, before the
        # generator stage (quality 0.475/0.547) and the react refits ever run.
        # Budget adaptation for forward mode is stage_stop_after (in-stage
        # 2-miss stop; freed quota re-splits onto LATER stages) plus the natural
        # exhaustion handoff when the pass completes early. Keep window>0 only
        # for lookback-style multi-pass schedules.
        self.greedy_plateau_window = max(0, int(greedy_plateau_window))
        self.greedy_plateau_eps = float(greedy_plateau_eps)
        self.n_iterations = int(n_iterations) if n_iterations is not None else None
        self.total_success_budget = (
            max(self.greedy_budget, self.n_iterations)
            if self.n_iterations is not None else None
        )
        self.objective_names = [o["objective_name"] for o in objectives]
        self.objective_costs = {
            o["objective_name"]: float(o.get("cost", 1.0)) for o in objectives
        }
        # Keep controller Phase 1 one-at-a-time; the wrapper owns the 30-step split.
        self.n_both_init = 1

        self.greedy = GreedyParetoOptimizer(
            config_space=config_space,
            objectives=objectives,
            seed=seed,
            search_mode=greedy_search_mode,
            n_iterations=self.greedy_budget,
            beam_width=greedy_beam_width,
            stage_weight=greedy_stage_weight,
            pass_budget_fraction=greedy_pass_budget_fraction,
            max_passes=greedy_max_passes,
            stage_stop_after=greedy_stage_stop_after,
            sibling_seeding=greedy_sibling_seeding,
            ssops=ssops,
        )

        agent_kwargs = dict(kwargs)
        agent_kwargs.setdefault("n_both_init", min(10, self.greedy_budget))
        self.agent = self._build_phase2_optimizer(
            config_space=config_space,
            objectives=objectives,
            seed=seed,
            ssops=ssops,
            agent_kwargs=agent_kwargs,
        )

        self._evaluations: List[dict] = []
        self._perf_attrs: Dict[int, dict] = {}
        self._infeasible: List[dict] = []
        self._using_agent = False
        self._greedy_successes = 0
        self._greedy_exhausted = False
        self._agent_successes = 0
        self._agent_success_limit: Optional[int] = None
        self._pending_owner = "greedy"
        self._pending_full_metrics: List[bool] = []

        logger.info(
            "GreedyThenAnchorAgentQNEHVI: greedy_success_budget=%d total_success=%s "
            "total=%s seed=%d agent=%s",
            self.greedy_budget,
            self.total_success_budget,
            self.n_iterations,
            self.seed,
            type(self.agent.agent).__name__,
        )

    def _build_phase2_optimizer(
        self,
        *,
        config_space,
        objectives: List[dict],
        seed: int,
        ssops: Optional[AxSearchSpaceOps],
        agent_kwargs: dict,
    ) -> AgentGuidedQNEHVI:
        """Construct the wrapper's only phase-2 optimizer."""
        return AnchorAgentQNEHVI(
            config_space=config_space,
            objectives=objectives,
            seed=seed,
            ssops=ssops,
            **agent_kwargs,
        )

    def _sync_agent_hooks(self) -> None:
        for name in ("_advice_log_dir",):
            if hasattr(self, name):
                try:
                    setattr(self.agent, name, getattr(self, name))
                except Exception:  # noqa: BLE001
                    pass
        for eid, attr in self._perf_attrs.items():
            self.agent.register_perf_attribution(eid, attr)
        for rec in self._infeasible:
            self.agent.register_infeasible(
                rec.get("config") or {},
                rec.get("reason", ""),
                eval_id=rec.get("eval_id"),
                optimizer_metadata=rec.get("optimizer_metadata"),
                replay=True,
            )

    @staticmethod
    def _obj_scalar(v) -> Optional[float]:
        """Objective cell → scalar; quality may arrive as {'combined': v}
        (same unwrap the greedy phase itself applies)."""
        if isinstance(v, dict):
            v = v.get("combined")
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if math.isfinite(f) else None

    @staticmethod
    def _hv_unit(pts: List[Tuple[float, float]]) -> float:
        """2-D staircase HV of already-normalized (max-scaled) points, ref (0,0)."""
        hv, prev_q = 0.0, 0.0
        for p, q in sorted(pts, key=lambda t: -t[0]):
            if q > prev_q:
                hv += p * (q - prev_q)
                prev_q = q
        return hv

    def _greedy_hv_plateaued(self) -> bool:
        """True when the last ``greedy_plateau_window`` greedy successes added
        less than ``greedy_plateau_eps`` normalized HV in total. Wrapper-side
        only (reads its own eval log); both windows share one normalization so
        the delta is a true HV increment."""
        w = self.greedy_plateau_window
        if w <= 0 or self._greedy_successes < max(10, w + 2):
            return False
        pts: List[Tuple[float, float]] = []
        for ev in self._evaluations:
            objs = ev.get("objectives") or {}
            p = self._obj_scalar(objs.get("performance"))
            q = self._obj_scalar(objs.get("quality"))
            if p is None or q is None:
                continue
            pts.append((max(0.0, p), max(0.0, q)))
        if len(pts) < w + 2:
            return False
        p_rng = max(1e-9, max(p for p, _ in pts))
        q_rng = max(1e-9, max(q for _, q in pts))
        norm = [(p / p_rng, q / q_rng) for p, q in pts]
        return self._hv_unit(norm) - self._hv_unit(norm[:-w]) < self.greedy_plateau_eps

    def _ensure_agent_started(self) -> None:
        if self._using_agent:
            return
        if self._greedy_successes < self.greedy_budget and not self._greedy_exhausted:
            if not self._greedy_hv_plateaued():
                return
            logger.info(
                "GreedyThenAnchorAgentQNEHVI: greedy HV plateaued (last %d "
                "successes gained < %.4f) — handing over early at %d/%d successes",
                self.greedy_plateau_window,
                self.greedy_plateau_eps,
                self._greedy_successes,
                self.greedy_budget,
            )
            self._greedy_exhausted = True
        self._sync_agent_hooks()
        self.agent.warm_start(self._evaluations)
        self._using_agent = True
        if self.total_success_budget is not None:
            # PAID evaluations only: fingerprint-inherited rows (M3, free=True)
            # sit in _evaluations but returned their budget seat — counting
            # them here would shrink the agent's allowance and end the run
            # short of the controller's budget.
            paid = sum(1 for ev in self._evaluations if not ev.get("free"))
            self._agent_success_limit = max(
                0,
                self.total_success_budget - paid,
            )
        logger.info(
            "GreedyThenAnchorAgentQNEHVI: switched to anchor-agent qNEHVI "
            "after %d greedy successes%s; agent_success_limit=%s",
            self._greedy_successes,
            " (greedy exhausted)" if self._greedy_exhausted else "",
            self._agent_success_limit,
        )

    def suggest(
        self, n_suggestions: int = 1,
    ) -> Tuple[Optional[List[dict]], Optional[List[str]]]:
        self._ensure_agent_started()
        if not self._using_agent:
            remaining = self.greedy_budget - self._greedy_successes
            n = max(1, min(int(n_suggestions), remaining))
            configs, objs = self.greedy.suggest(n)
            if configs is None or objs is None:
                self._greedy_exhausted = True
                self._ensure_agent_started()
                if self._using_agent:
                    return self.suggest(n_suggestions)
                return None, None
            self._pending_owner = "greedy"
            self._pending_full_metrics = [True] * len(configs or [])
            return configs, objs

        if (
            self._agent_success_limit is not None
            and self._agent_successes >= self._agent_success_limit
        ):
            logger.info(
                "GreedyThenAnchorAgentQNEHVI: agent success budget exhausted "
                "(%d/%d valid evals)",
                self._agent_successes,
                self._agent_success_limit,
            )
            return None, None
        if self._agent_success_limit is None:
            n = max(1, int(n_suggestions))
        else:
            n = max(
                1,
                min(int(n_suggestions), self._agent_success_limit - self._agent_successes),
            )
        configs, objs = self.agent.suggest(n)
        self._pending_owner = "agent"
        self._pending_full_metrics = list(
            getattr(self.agent, "_pending_full_metrics", []) or [True] * len(configs or [])
        )
        return configs, objs

    def add_objectives(
        self,
        x: List[dict],
        obj_dict: Dict[str, List[Union[float, dict]]],
        eval_ids: Optional[List[int]] = None,
        free_mask: Optional[List[bool]] = None,
    ) -> None:
        # free rows (M3 fingerprint-inherited quality) are archive observations
        # that must not advance the phase success budgets.
        optimizer_metadata = self.pending_observation_metadata()
        n_paid = len(x) - (sum(1 for f in (free_mask or [])[: len(x)] if f))
        owner = self._pending_owner
        if owner == "agent":
            self.agent.add_objectives(x, obj_dict, eval_ids=eval_ids,
                                      free_mask=free_mask)
            self._agent_successes += n_paid
        else:
            self.greedy.add_objectives(x, obj_dict, eval_ids=eval_ids,
                                       free_mask=free_mask)
            self._greedy_successes = min(
                self.greedy_budget,
                self._greedy_successes + n_paid,
            )

        for i, cfg in enumerate(x):
            objs = {}
            complete = True
            for name in self.objective_names:
                vals = obj_dict.get(name) or []
                if i >= len(vals):
                    complete = False
                    break
                objs[name] = vals[i]
            if not complete:
                continue
            free = bool(free_mask[i]) if free_mask and i < len(free_mask) else False
            self._evaluations.append({
                "config": cfg,
                "objectives": objs,
                "eval_id": eval_ids[i] if eval_ids else -1,
                "eval_type": "GT_FP" if free else "GT",
                "free": free,
                "optimizer_metadata": dict(optimizer_metadata),
            })
        self._ensure_agent_started()

    def pending_observation_metadata(self) -> dict:
        if self._pending_owner == "agent":
            return self.agent.pending_observation_metadata()
        return {"forced": False, "agent_proposal_round": 0}

    def get_pareto(self) -> Tuple[List[dict], List[Dict[str, float]], List[int]]:
        if self._using_agent:
            return self.agent.get_pareto()
        return self.greedy.get_pareto()

    def warm_start(self, evaluations: List[Dict]) -> None:
        self._evaluations = []
        complete = []
        for ev in evaluations:
            objs = ev.get("objectives") or {}
            if all(name in objs for name in self.objective_names):
                complete.append(ev)
                self._evaluations.append(ev)
        # Phase split by PAID successes: free rows (M3, ev["free"]) sit in the
        # archive but never consumed a greedy/agent budget seat.
        paid_seen = 0
        split = len(complete)
        for idx, ev in enumerate(complete):
            if not ev.get("free"):
                paid_seen += 1
            if paid_seen >= self.greedy_budget:
                split = idx + 1
                break
        greedy_evals = complete[:split]
        self._greedy_successes = min(
            self.greedy_budget,
            sum(1 for ev in greedy_evals if not ev.get("free")),
        )
        self._agent_successes = sum(
            1 for ev in complete[split:] if not ev.get("free")
        )
        if greedy_evals:
            self.greedy.warm_start(greedy_evals)
        if self._greedy_successes >= self.greedy_budget:
            self._sync_agent_hooks()
            self.agent.warm_start(complete)
            self._using_agent = True
            if self.total_success_budget is not None:
                self._agent_success_limit = max(
                    0,
                    self.total_success_budget - self.greedy_budget,
                )
        logger.info(
            "GreedyThenAnchorAgentQNEHVI.warm_start: restored %d evals; phase=%s",
            len(complete),
            "agent" if self._using_agent else "greedy",
        )

    def register_perf_attribution(self, eval_id: int, attr: dict) -> None:
        self._perf_attrs[int(eval_id)] = dict(attr or {})
        if self._using_agent:
            self.agent.register_perf_attribution(eval_id, attr)

    def register_infeasible(
        self,
        config: dict,
        reason: str,
        *,
        eval_id: Optional[int] = None,
        optimizer_metadata: Optional[dict] = None,
        replay: bool = False,
    ) -> None:
        rec = {
            "config": dict(config or {}),
            "reason": str(reason),
            "eval_id": eval_id,
            "optimizer_metadata": deepcopy(optimizer_metadata),
        }
        self._infeasible.append(rec)
        was_using_agent = self._using_agent
        if was_using_agent:
            self.agent.register_infeasible(
                config,
                reason,
                eval_id=eval_id,
                optimizer_metadata=optimizer_metadata,
                replay=replay,
            )


# ═════════════════════════════════════════════════════════════════════════
# V3 METHOD LAYER — E[ΔHV] ranking (the registered optimizer:
# ``greedy_then_hv_anchor_agent_qnehvi``).
# ═════════════════════════════════════════════════════════════════════════


class HVAnchorLLMAgent(GreedyAnchorLLMAgent):
    """Post-greedy proposal agent with explicit HV coverage instructions."""

    def _hv_band_card(self, history: List[dict]) -> str:
        if len(history) < 2:
            return ""
        p = np.asarray([float(h["performance"]) for h in history], dtype=float)
        q = np.asarray([float(h["correctness"]) for h in history], dtype=float)
        y = np.column_stack([p, q])
        front = [
            (float(p[i]), float(q[i]))
            for i, keep in enumerate(_pareto_mask(y))
            if keep
        ]
        front.sort(reverse=True)
        p_max = max(1e-9, float(p.max()))
        q_max = float(q.max())

        bands = [
            ("FAST-SLOPE", 0.60, 1.01, "raise quality while staying near the fast end"),
            ("MID-HIGH", 0.25, 0.60, "fill the high-performance quality ramp"),
            ("KNEE", 0.06, 0.25, "trade performance for a substantial quality jump"),
            ("QUALITY-WIDTH", 0.0, 0.06, "increase performance at near-best quality"),
        ]
        lines = [
            "HV ANCHOR COVERAGE TASK:",
            "  Hypervolume needs rectangle area, not isolated max-quality points.",
            "  Current non-dominated frontier, high perf to high quality:",
        ]
        lines.extend(f"    perf={pp:.1f} q={qq:.3f}" for pp, qq in front)
        lines.append("  Target bands this round:")
        for name, lo, hi, why in bands:
            rows = [(pp, qq) for pp, qq in front if lo * p_max <= pp < hi * p_max]
            if rows:
                best = max(rows, key=lambda z: z[1])
                lines.append(
                    f"    {name}: current best q={best[1]:.3f} at perf={best[0]:.1f}; {why}."
                )
            else:
                lines.append(
                    f"    {name}: EMPTY; propose one candidate for this missing band."
                )
        if q_max >= 0.55:
            lines.append(
                "  Quality is already high (q>=0.55): do NOT spend proposals on "
                "slower quality-only refinements unless they also improve performance "
                "at q within 0.03 of the best observed quality."
            )
        lines.append(
            "  Candidate quota: one FAST-SLOPE, one MID-HIGH, one KNEE, and at most "
            "one QUALITY-WIDTH proposal. Prefer anchors on the relevant frontier "
            "segment, not the global best-quality anchor by default."
        )
        return "\n".join(lines)

    def _build_messages(self, history, ssops, mo=None) -> List[dict]:
        messages = super()._build_messages(history, ssops, mo)
        hv_card = self._hv_band_card(history)
        if not hv_card:
            return messages
        user = messages[1]["content"]
        insert = (
            hv_card
            + "\n\nSELECTION DISCIPLINE:\n"
            "  - A candidate that is dominated by the current frontier is wasted.\n"
            "  - The biggest missing area usually sits on the performance-quality ramp, "
            "not at the slowest quality endpoint.\n\n"
        )
        marker = "RULES:\n"
        if marker in user:
            user = user.replace(marker, insert + marker, 1)
        else:
            user = user + "\n\n" + insert
        messages[1]["content"] = user
        return messages


class HVAnchorQNEHVI(AnchorAgentQNEHVI):
    """Anchor qNEHVI whose pool selection is an E[ΔHV] ranking.

    No vetoes: the ranking keeps top-K plus every LLM-family candidate and
    qLogNEHVI arbitrates among the survivors — its argsort IS the decision;
    no post-acqf rerank exists.
    """

    _PRIORITY_PROV = ("llm", "dls-llm")

    def _create_agent(
        self,
        *,
        model: str,
        strict: bool,
        wall_timeout_s: Optional[float],
    ) -> LLMAgent:
        return HVAnchorLLMAgent(
            model=model,
            strict=strict,
            wall_timeout_s=wall_timeout_s,
        )

    def __init__(
        self,
        *args,
        rank_top_k: int = 16,
        rank_floor: int = 8,
        coldstart_guard_picks: int = 3,
        anchor_filter_low_perf_quantile: float = 0.5,
        anchor_filter_quality_margin_frac: float = 0.05,
        **kwargs,
    ):
        self.rank_top_k = max(1, int(rank_top_k))
        self.rank_floor = max(1, int(rank_floor))
        self.coldstart_guard_picks = max(0, int(coldstart_guard_picks))
        self.anchor_filter_low_perf_quantile = min(
            1.0, max(0.0, float(anchor_filter_low_perf_quantile))
        )
        self.anchor_filter_quality_margin_frac = max(
            0.0, float(anchor_filter_quality_margin_frac)
        )
        self._guard_until_t = self.coldstart_guard_picks
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # objective-space scoring
    # ------------------------------------------------------------------

    def _expected_hv_gain(
        self, perf: float, mu_q: float, sigma_q: float
    ) -> Optional[float]:
        """E over q ~ N(mu_q, sigma_q) of the 2-D HV gain of (perf, q) vs the
        archive front: a per-front-segment EI sum. Performance and quality use
        their historical maxima as scales, with reference point ``(0, 0)``."""
        if len(getattr(self, "_perf", [])) < 2 or len(getattr(self, "_corr", [])) < 2:
            return None
        y = np.asarray(
            [[self._perf[i], self._corr[i]] for i in range(len(self._corr))],
            dtype=float,
        )
        if y.size == 0:
            return None
        front = y[_pareto_mask(y)]
        p_rng = max(1e-9, float(y[:, 0].max()))
        q_rng = max(1e-9, float(y[:, 1].max()))
        p_star = max(0.0, float(perf) / p_rng)
        if p_star <= 0.0:
            return 0.0
        mu = float(mu_q) / q_rng
        sd = max(1e-12, float(sigma_q) / q_rng)

        rows = sorted(
            ((float(r[0]) / p_rng, float(r[1]) / q_rng) for r in front),
            key=lambda t: -t[0],
        )
        ps: List[float] = []
        hs: List[float] = []
        best_q = 0.0
        for p, q in rows:
            best_q = max(best_q, q)
            if ps and p == ps[-1]:
                hs[-1] = best_q
            else:
                ps.append(p)
                hs.append(best_q)

        def _ei(h: float) -> float:
            z = (mu - h) / sd
            cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
            return (mu - h) * cdf + sd * pdf

        gain = (p_star - ps[0]) * _ei(0.0) if p_star > ps[0] else 0.0
        for j in range(len(ps)):
            left = ps[j + 1] if j + 1 < len(ps) else 0.0
            w = min(p_star, ps[j]) - left
            if w > 0:
                gain += w * _ei(hs[j])
        return float(gain)

    def _low_perf_quality_width_reject(self, perf: float, qual: float) -> "tuple[bool, str]":
        """Cold-start guard: reject slow-end width expansions once the quality
        endpoint is strong. Objective-space only; active only while
        ``_t < _guard_until_t``."""
        if len(getattr(self, "_perf", [])) < 4 or len(getattr(self, "_corr", [])) < 4:
            return False, "insufficient-history"
        y = np.asarray(
            [[self._perf[i], self._corr[i]] for i in range(len(self._corr))],
            dtype=float,
        )
        front = y[_pareto_mask(y)]
        if front.shape[0] < 3:
            return False, "short-frontier"
        perf_cut = float(np.quantile(front[:, 0], self.anchor_filter_low_perf_quantile))
        best_q = float(front[:, 1].max())
        q_rng = max(1e-9, float(y[:, 1].max() - y[:, 1].min()))
        required_q = best_q + self.anchor_filter_quality_margin_frac * q_rng
        if float(perf) < perf_cut and float(qual) <= required_q:
            return True, f"low-perf-width perf<{perf_cut:.2f} q_hat<={required_q:.3f}"
        return False, "ramp-or-quality"

    # ------------------------------------------------------------------
    # pool selection: rank, don't veto
    # ------------------------------------------------------------------

    def _prefilter_pool_with_prov(self, cand: torch.Tensor, prov: list) -> "tuple[torch.Tensor, list]":
        cand, prov = super()._prefilter_pool_with_prov(cand, prov)
        n = cand.shape[0]
        if (
            n <= self.rank_floor
            or getattr(self, "model", None) is None
            or len(getattr(self, "_perf", [])) < 2
        ):
            return cand, prov
        try:
            with torch.no_grad():
                post = self.model.posterior(cand.unsqueeze(-2))
                mu = post.mean.reshape(n, -1)
                var = post.variance.reshape(n, -1)
        except Exception as exc:  # noqa: BLE001 — never let scoring kill a round
            logger.debug("[hv-rank] posterior unavailable (%s); keeping pool", exc)
            return cand, prov

        # perf̂ is the GP mean: the synthetic-trace oracle channel was DELETED
        # 07-06 (flip-pricing forensics: 21-52% APE on trace-affecting
        # transforms — synthetic prices are not fit to steer arbitration), and
        # the dormant CM backfill plumbing was removed with it (07-07).
        guard_active = self._t < self._guard_until_t
        scores = np.full(n, -np.inf, dtype=float)
        guard_drops = 0
        for i in range(n):
            perf_hat = float(mu[i, self._perf_idx].item())
            mu_q = float(mu[i, self._qual_idx].item())
            sd_q = math.sqrt(max(1e-18, float(var[i, self._qual_idx].item())))
            if guard_active:
                rej, _ = self._low_perf_quality_width_reject(perf_hat, mu_q)
                if rej:
                    guard_drops += 1
                    continue
            g = self._expected_hv_gain(perf_hat, mu_q, sd_q)
            scores[i] = g if g is not None else 0.0

        ranked = [int(i) for i in np.argsort(-scores) if math.isfinite(scores[int(i)])]
        if not ranked:
            # Guard dropped everything (tiny slow-corner pool) — fail open.
            logger.info(
                "[hv-rank t=%s] guard dropped all %d candidates; keeping pool",
                getattr(self, "_t", "?"), n,
            )
            return cand, prov
        # keep-set = top-K by score ∪ every (unguarded) LLM-family candidate,
        # floor-filled to rank_floor. Ranking replaces vetoing: qNEHVI still
        # arbitrates among the survivors.
        priority = {
            i for i, src in enumerate(prov)
            if src in self._PRIORITY_PROV and math.isfinite(scores[i])
        }
        keep_set = set(ranked[: self.rank_top_k]) | priority
        for i in ranked:  # floor fill
            if len(keep_set) >= self.rank_floor:
                break
            keep_set.add(i)
        keep = sorted(keep_set)

        logger.info(
            "[hv-rank t=%s] pool %d->%d (top=%.5f guard_drops=%d)",
            getattr(self, "_t", "?"), n, len(keep),
            float(scores[ranked[0]]), guard_drops,
        )
        keep_t = torch.tensor(keep, dtype=torch.long, device=cand.device)
        return cand[keep_t], [prov[i] for i in keep]

    def warm_start(self, evaluations: List[dict]) -> None:
        # Re-arm the cold-start guard RELATIVE to the resumed _t (an absolute
        # `_t < k` never fires after a greedy warm start, where _t is already
        # ~20).
        super().warm_start(evaluations)
        self._guard_until_t = self._t + self.coldstart_guard_picks

class GreedyThenHVAnchorAgentQNEHVI(GreedyThenAnchorAgentQNEHVI):
    """Greedy forward warm start, then E[ΔHV]-ranked agent qNEHVI."""

    def _build_phase2_optimizer(
        self,
        *,
        config_space,
        objectives: List[dict],
        seed: int,
        ssops: Optional[AxSearchSpaceOps],
        agent_kwargs: dict,
    ) -> AgentGuidedQNEHVI:
        return HVAnchorQNEHVI(
            config_space=config_space,
            objectives=objectives,
            seed=seed,
            ssops=ssops,
            **agent_kwargs,
        )

    def __init__(
        self,
        config_space,
        objectives: List[dict],
        seed: int = 42,
        greedy_budget: int = 30,
        n_iterations: Optional[int] = None,
        greedy_search_mode: str = "forward",
        greedy_beam_width: int = 3,
        greedy_stage_weight: str = "log",
        greedy_pass_budget_fraction: float = 0.6,
        greedy_max_passes: int = 8,
        greedy_plateau_window: int = 0,
        greedy_plateau_eps: float = 0.005,
        greedy_stage_stop_after: int = 2,
        ssops=None,
        agent_model: str = "deepseek-v4-flash",
        rank_top_k: int = 16,
        rank_floor: int = 8,
        coldstart_guard_picks: int = 3,
        anchor_filter_low_perf_quantile: float = 0.5,
        anchor_filter_quality_margin_frac: float = 0.05,
        **kwargs,
    ):
        super().__init__(
            config_space=config_space,
            objectives=objectives,
            seed=seed,
            greedy_budget=greedy_budget,
            n_iterations=n_iterations,
            greedy_search_mode=greedy_search_mode,
            greedy_beam_width=greedy_beam_width,
            greedy_stage_weight=greedy_stage_weight,
            greedy_pass_budget_fraction=greedy_pass_budget_fraction,
            greedy_max_passes=greedy_max_passes,
            greedy_plateau_window=greedy_plateau_window,
            greedy_plateau_eps=greedy_plateau_eps,
            greedy_stage_stop_after=greedy_stage_stop_after,
            ssops=ssops,
            agent_model=agent_model,
            rank_top_k=rank_top_k,
            rank_floor=rank_floor,
            coldstart_guard_picks=coldstart_guard_picks,
            anchor_filter_low_perf_quantile=anchor_filter_low_perf_quantile,
            anchor_filter_quality_margin_frac=anchor_filter_quality_margin_frac,
            **kwargs,
        )
        logger.info(
            "GreedyThenHVAnchorAgentQNEHVI: phase2=%s rank_top_k=%d floor=%d",
            type(self.agent).__name__,
            rank_top_k,
            rank_floor,
        )
