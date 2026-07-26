"""Optimizer-side history, evidence, geometry, and Agent orchestration."""

import json
import logging
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from ax.core.parameter import ChoiceParameter
from torch import Tensor

from .llm_agent import AgentAdvice, AgentAdviceError, LLMAgent
from .util import _pareto_mask

logger = logging.getLogger("RAG-Stack")


class PromptFeedMixin:
    def _allowed_submetrics(self) -> "set | None":
        """The sub-metric names configured in eval_backend_setting.metrics.

        Resumed projects carry evals recorded under an OLDER, larger metric set
        (07-06: retrieval_token_* and context_* were dropped as misleading /
        redundant). Filtering every card through this allow-list keeps the
        agent's view consistent with the CURRENT config — stale sub-metrics
        from pre-change evals never leak back in. None ⇒ no config ⇒ no filter.
        """
        cfg = getattr(self.ssops, "_config", None) or {}
        metrics = ((cfg.get("eval_backend_setting") or {}).get("metrics")) or []
        names = {m.get("metric_name") for m in metrics if m.get("metric_name")}
        return names or None

    def _history(self) -> List[dict]:
        allow = self._allowed_submetrics()
        stage_signals = getattr(self, "enable_stage_signals", True)
        def _sm(d):
            if not stage_signals:
                return {}
            if allow is None or not d:
                return d
            return {k: v for k, v in d.items() if k in allow or k == "combined"}
        hist = []
        for i in range(len(self._configs)):
            hist.append({
                "x": self._x_rows[i],
                "flat": self.ssops.decode_flat(self._x_rows[i]),
                "performance": self._perf[i],
                "correctness": self._corr[i],
                "sub_metrics": _sm(self._sub_metrics[i]),
                "perf_attr": self._perf_attr.get(int(self._eval_ids[i])) if stage_signals and i < len(self._eval_ids) else None,
            })
        return hist

    def _pareto_for_agent(self, cap: int = 8) -> List[dict]:
        """Current (perf, correctness) Pareto frontier entries, for the MO-aware agent."""
        if not self._configs:
            return []
        Y = np.array([[self._perf[i], self._corr[i]] for i in range(len(self._configs))])
        idxs = [i for i in range(len(self._configs)) if _pareto_mask(Y)[i]]
        idxs.sort(key=lambda i: self._perf[i])  # low-throughput/high-corr → high-throughput
        if len(idxs) > cap:
            step = len(idxs) / cap
            idxs = [idxs[int(k * step)] for k in range(cap)]
        allow = self._allowed_submetrics()
        stage_signals = getattr(self, "enable_stage_signals", True)
        def _sm(d):
            if not stage_signals:
                return {}
            if allow is None or not d:
                return d
            return {k: v for k, v in d.items() if k in allow or k == "combined"}
        return [
            {
                "flat": self.ssops.decode_flat(self._x_rows[i]),
                "perf": self._perf[i],
                "corr": self._corr[i],
                "sub_metrics": _sm(self._sub_metrics[i]),
                "perf_attr": self._perf_attr.get(int(self._eval_ids[i])) if stage_signals and i < len(self._eval_ids) else None,
            }
            for i in idxs
        ]

    def _max_hv_rectangle(self) -> dict:
        """Render the shared largest realized-HV gap for the ordinary Agent."""
        gap = self._largest_hv_gap()
        if gap is None:
            return {}
        quality_index = gap.quality_index
        performance_index = gap.performance_index
        perf_lo = float(self._perf[quality_index])
        perf_hi = float(self._perf[performance_index])
        quality_hi = float(self._corr[quality_index])
        quality_lo = float(self._corr[performance_index])
        return {
            "perf_lo": perf_lo,
            "perf_hi": perf_hi,
            "q_lo": quality_lo,
            "q_hi": quality_hi,
            "area_norm": gap.area,
            "target": {
                "perf": (perf_lo + perf_hi) / 2.0,
                "quality": (quality_lo + quality_hi) / 2.0,
            },
        }

    def _model_confidence(self) -> dict:
        """GP reliability signal for the agent — how much to TRUST the data-driven
        findings vs keep exploring. ``quality_sigma`` = the quality GP's **leave-one-out
        CV** RMSE (in quality units): for each eval, predict its quality from all the
        OTHERS and measure the error. This is the honest "how well does the GP
        generalize to a new config" number — it captures both noise and misfit, on the
        configs that actually exist.

        (NOT the posterior σ over a Sobol sweep: in a 14-dim space ~every sampled point
        is far from the handful of evals, so that σ ≈ the prior ≈ σ_data regardless of
        fit quality — it can't discriminate a good GP from a useless one. LOO can.)

        Computed in closed form from the fitted kernel (Rasmussen & Williams Eq. 5.12,
        no refit): with C = K + σ_n²I and α = C⁻¹(y−m), the LOO residual for point i is
        α_i / [C⁻¹]_ii. Compared against ``quality_std`` (spread of observed quality):
        LOO-RMSE ≈ std ⇒ GP predicts no better than guessing the mean ⇒ findings weak."""
        q_obs = self._corr
        std = float(np.std(q_obs)) if len(q_obs) > 1 else 0.0
        conf = {
            "n_evals": len(self._configs),
            "quality_range": (float(min(q_obs)), float(max(q_obs))) if q_obs else (0.0, 0.0),
            "quality_std": std,
            "quality_sigma": None,
        }
        if len(q_obs) < 4 or std <= 1e-9:
            return conf
        try:
            gp = self.model.models[self._qual_idx]
            with torch.no_grad():
                X = gp.train_inputs[0]
                y = gp.train_targets.reshape(-1)            # standardized by Standardize
                m = gp.mean_module(X).reshape(-1)
                Kf = gp.covar_module(X).to_dense()
                noise = gp.likelihood.noise.reshape(-1)
                n = X.shape[0]
                C = Kf + torch.diag(noise.expand(n) if noise.numel() == 1 else noise)
                Cinv = torch.linalg.inv(C)
                alpha = Cinv @ (y - m)
                e_std = alpha / torch.diag(Cinv)            # LOO residuals (standardized)
                loo_rmse_std = float(e_std.pow(2).mean().sqrt().item())
            conf["quality_sigma"] = loo_rmse_std * std       # back to quality units
        except Exception as exc:  # diagnostic only — never block the BO step
            logger.debug(f"_model_confidence: LOO estimate failed ({exc})")
        return conf

    def _evidence_branch(self, row: Tensor) -> Dict[str, Any]:
        """Exact active hierarchy selectors for one encoded row.

        The evidence card must not infer a branch from benchmark/stage names.
        Ax's declared hierarchical ``ChoiceParameter`` values are the complete
        and generic branch identity presented to the Agent.
        """
        decoded = self.ssops.decode_flat(row.reshape(-1))
        active = self.ssops.active_params(decoded)
        return {
            name: decoded.get(name)
            for name in self.ssops.param_names
            if (
                name in active
                and isinstance(
                    self.ssops.ss.parameters[name], ChoiceParameter,
                )
                and self.ssops.ss.parameters[name].is_hierarchical
            )
        }

    def _gp_knob_tension_evidence(self) -> dict:
        """Diagnostic GP contrasts for every history-supported tension knob.

        For each objective champion and active ChoiceParameter, evaluate every
        alternative legal level as a one-knob graft and report the strongest
        contrast for that champion/knob pair. Objective contrasts and posterior
        uncertainty are divided by the realized objective spans, so the payload
        is invariant to positive affine changes of either axis. The score is
        the normalized opposite-sign rectangle
        ``max(0, -delta_performance * delta_quality)``; zero-tension knobs stay
        visible. This is marginal surrogate screening evidence, not a landed
        or causal ablation.
        """
        if (
            getattr(self, "model", None) is None
            or len(getattr(self, "_x_rows", ())) < 2
        ):
            return {
                "status": "unavailable",
                "reason": "GP or realized history is not available",
                "rows": [],
            }

        performance = [float(value) for value in self._perf]
        quality = [float(value) for value in self._corr]
        p_span = max(performance) - min(performance)
        q_span = max(quality) - min(quality)
        if (
            not math.isfinite(p_span)
            or not math.isfinite(q_span)
            or p_span <= 0.0
            or q_span <= 0.0
        ):
            return {
                "status": "unavailable",
                "reason": "realized objective spans are degenerate",
                "rows": [],
            }

        def _champion(objective: int) -> int:
            values = (performance, quality)
            other = 1 - objective
            return max(
                range(len(self._x_rows)),
                key=lambda index: (
                    values[objective][index],
                    values[other][index],
                    -index,
                ),
            )

        champion_specs = [
            ("performance_champion", _champion(0)),
            ("quality_champion", _champion(1)),
        ]
        if champion_specs[0][1] == champion_specs[1][1]:
            champion_specs = [("shared_objective_champion", champion_specs[0][1])]

        descriptors = []
        query_rows: List[Tensor] = []
        for anchor_name, anchor_index in champion_specs:
            anchor = self._x_rows[anchor_index].reshape(-1)
            anchor_decoded = self.ssops.decode_flat(anchor)
            anchor_active = self.ssops.active_params(anchor_decoded)
            anchor_branch = self._evidence_branch(anchor)
            for dim, name in enumerate(self.ssops.param_names):
                parameter = self.ssops.ss.parameters[name]
                if (
                    not isinstance(parameter, ChoiceParameter)
                    or name not in anchor_active
                ):
                    continue
                for coordinate in self.ssops.choice_grid(name):
                    candidate = anchor.reshape(1, -1).clone()
                    candidate[0, dim] = float(coordinate)
                    candidate = self.ssops.clamp(
                        self.ssops.snap_to_grid(candidate),
                    ).reshape(-1)
                    if self.ssops.is_duplicate_decoded(
                        candidate.reshape(1, -1),
                        anchor.reshape(1, -1),
                    ):
                        continue
                    candidate_decoded = self.ssops.decode_flat(candidate)
                    candidate_branch = self._evidence_branch(candidate)
                    level = candidate_decoded.get(name)
                    support = 0
                    for observed in self._x_rows:
                        observed_decoded = self.ssops.decode_flat(
                            observed.reshape(-1),
                        )
                        if (
                            self._evidence_branch(observed) == candidate_branch
                            and name in self.ssops.active_params(observed_decoded)
                            and observed_decoded.get(name) == level
                        ):
                            support += 1
                    descriptors.append({
                        "knob": name,
                        "anchor": anchor_name,
                        "anchor_level": anchor_decoded.get(name),
                        "level": level,
                        "anchor_branch": anchor_branch,
                        "candidate_branch": candidate_branch,
                        "support": int(support),
                    })
                    query_rows.extend((anchor, candidate))

        if not descriptors:
            return {
                "status": "unavailable",
                "reason": "no legal one-knob champion contrasts",
                "rows": [],
            }

        try:
            query = torch.stack(query_rows, dim=0).to(**self.tkwargs)
            with torch.no_grad():
                posterior = self.model.posterior(query.unsqueeze(-2))
                means = posterior.mean.reshape(query.shape[0], -1)
                variances = posterior.variance.reshape(query.shape[0], -1)
        except Exception as exc:  # diagnostic enrichment must not block Agent
            logger.debug("GP knob-tension evidence unavailable: %s", exc)
            return {
                "status": "unavailable",
                "reason": f"GP posterior failed ({type(exc).__name__})",
                "rows": [],
            }

        contrast_rows = []
        for index, descriptor in enumerate(descriptors):
            anchor_position = 2 * index
            level_position = anchor_position + 1
            try:
                delta_performance = float(
                    (
                        means[level_position, self._perf_idx]
                        - means[anchor_position, self._perf_idx]
                    ).item()
                ) / p_span
                delta_quality = float(
                    (
                        means[level_position, self._qual_idx]
                        - means[anchor_position, self._qual_idx]
                    ).item()
                ) / q_span
                # The exact contrast covariance is not exposed uniformly by
                # every ModelList posterior.  The sum of endpoint standard
                # deviations is a conservative contrast-uncertainty bound.
                performance_uncertainty = (
                    math.sqrt(max(
                        0.0,
                        float(variances[anchor_position, self._perf_idx].item()),
                    ))
                    + math.sqrt(max(
                        0.0,
                        float(variances[level_position, self._perf_idx].item()),
                    ))
                ) / p_span
                quality_uncertainty = (
                    math.sqrt(max(
                        0.0,
                        float(variances[anchor_position, self._qual_idx].item()),
                    ))
                    + math.sqrt(max(
                        0.0,
                        float(variances[level_position, self._qual_idx].item()),
                    ))
                ) / q_span
            except (IndexError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (
                delta_performance,
                delta_quality,
                performance_uncertainty,
                quality_uncertainty,
            )):
                continue
            tension = max(0.0, -(delta_performance * delta_quality))
            contrast_rows.append({
                **descriptor,
                "delta_performance_norm": delta_performance,
                "delta_quality_norm": delta_quality,
                "contrast_uncertainty_performance_norm": (
                    performance_uncertainty
                ),
                "contrast_uncertainty_quality_norm": quality_uncertainty,
                "tension": tension,
            })

        # The card is explicitly per knob, not per knob level. For each
        # champion/knob pair, expose the legal level with the strongest
        # predicted trade-off; if every level has zero opposite-sign tension,
        # keep the strongest marginal contrast with tension=0. This makes
        # absence of predicted tension visible instead of silently deleting
        # that knob from the Agent's evidence.
        contrast_rows.sort(key=lambda row: (
            str(row["knob"]),
            str(row["anchor"]),
            -float(row["tension"]),
            -(
                abs(float(row["delta_performance_norm"]))
                + abs(float(row["delta_quality_norm"]))
            ),
            -int(row["support"]),
            repr(row["level"]),
        ))
        rows = []
        levels_considered: Dict[Tuple[str, str], int] = {}
        for row in contrast_rows:
            key = (str(row["knob"]), str(row["anchor"]))
            levels_considered[key] = levels_considered.get(key, 0) + 1
            if any(
                (str(existing["knob"]), str(existing["anchor"])) == key
                for existing in rows
            ):
                continue
            rows.append(row)
        for row in rows:
            row["levels_considered"] = levels_considered[
                (str(row["knob"]), str(row["anchor"]))
            ]
        rows.sort(key=lambda row: (
            -float(row["tension"]),
            str(row["knob"]),
            str(row["anchor"]),
        ))
        if not rows:
            return {
                "status": "unavailable",
                "reason": "GP produced no finite champion level contrast",
                "rows": [],
            }
        return {
            "status": "ok",
            "rows": rows,
            "normalization": {
                "performance_span": p_span,
                "quality_span": q_span,
            },
        }

    def _doe_landed_evidence(self) -> dict:
        """Read the controller's exact landed-ablation payload if available.

        DOE owns the ticket/anchor/era ledger. Prompt code deliberately does
        not reconstruct exact ablations from observational history.
        """
        controller = getattr(self, "doe", None)
        if controller is None:
            return {
                "status": "unavailable",
                "reason": "DOE controller is not available",
                "rows": [],
            }
        provider = getattr(controller, "ablation_card_payload", None)
        if not callable(provider):
            return {
                "status": "unavailable",
                "reason": "no DOE landed-ablation payload",
                "rows": [],
            }
        try:
            rows = provider()
        except Exception as exc:
            logger.debug("DOE landed evidence unavailable: %s", exc)
            return {
                "status": "unavailable",
                "reason": (
                    "DOE landed-evidence provider failed "
                    f"({type(exc).__name__})"
                ),
                "rows": [],
            }
        if not isinstance(rows, tuple):
            return {
                "status": "unavailable",
                "reason": "invalid DOE landed-ablation payload",
                "rows": [],
            }
        normalized_rows = [
            dict(row) for row in rows if isinstance(row, Mapping)
        ]
        return {
            "status": "ok" if normalized_rows else "unavailable",
            "reason": (
                "" if normalized_rows
                else "no landed champion DOE ablations yet"
            ),
            "rows": normalized_rows,
        }

    def _independent_evidence_payload(self) -> dict:
        """One diagnostic payload shared by ordinary and forced Agent calls."""
        return {
            "gp_knob_tension": self._gp_knob_tension_evidence(),
            "doe_landed_ablation": self._doe_landed_evidence(),
        }

    def _maybe_advise(self) -> Optional[AgentAdvice]:
        if not getattr(self, "enable_agent", True):
            return None
        mo = {
            "quality_name": self.objective_names[self._qual_idx],
            "perf_name": self.objective_names[self._perf_idx],
            "ref_point": self.ref_point.tolist(),
            "pareto": self._pareto_for_agent(),
            "model_confidence": self._model_confidence(),
            # The Agent's geometric target must match the acquisition
            # objective. Euclidean "widest gap" distance is not HV geometry;
            # the largest normalized empty rectangle is.
            "hv_rectangle": self._max_hv_rectangle(),
            "infeasible": list(self._infeasible[-12:]),
            # Independent evidence is diagnostic only: it never changes the GP,
            # DOE plan, candidate pool, or acquisition ranking.
            "independent_evidence": self._independent_evidence_payload(),
        }
        mc = mo["model_confidence"]
        if mc.get("quality_sigma") is not None and mc.get("quality_std"):
            r = mc["quality_sigma"] / mc["quality_std"]
            logger.info(
                f"[agent t={self._t}] GP-confidence: quality LOO σ={mc['quality_sigma']:.3f} / "
                f"σ_data={mc['quality_std']:.3f} = ratio {r:.2f} "
                f"({'HIGH' if r < 0.5 else 'MODERATE' if r < 0.85 else 'LOW'})"
            )
        # Resume continuity (07-05): rehydrate the agent's SEARCH NOTES from the
        # persisted advice log on the first advise after a checkpoint restore.
        if (
            getattr(self.agent, "_notes", None) == ""
            and getattr(self, "_advice_log_dir", None)
        ):
            try:
                path = os.path.join(self._advice_log_dir, "agent_advice.jsonl")
                if os.path.isfile(path):
                    with open(path) as stream:
                        for line in stream:
                            record = json.loads(line)
                            # Notes authored with diagnostics enabled are not
                            # valid evidence for a stage-signal ablation.
                            if record.get("enable_stage_signals", True) != getattr(self, "enable_stage_signals", True):
                                continue
                            notes = record.get("notes")
                            if notes:
                                self.agent._notes = str(notes)[:1200]
            except Exception:  # noqa: BLE001 — memo rehydration is best-effort
                pass
        try:
            advice = self.agent.advise(self._history(), self.ssops, self._t, mo)
        except Exception as exc:
            if self.agent_strict:
                if isinstance(exc, AgentAdviceError):
                    raise
                raise AgentAdviceError(
                    f"agent.advise failed in strict mode ({exc})"
                ) from exc
            logger.warning(f"agent.advise failed ({exc}); continuing without advice")
            return None
        if (
            self.agent_strict
            and isinstance(self.agent, LLMAgent)
            and (advice is None or advice.n() == 0)
        ):
            # Covers LLMAgent subclasses with a custom empty-advice fallback.
            raise AgentAdviceError(
                "strict LLMAgent returned no usable candidate centers"
            )
        if advice and advice.rationale:
            logger.info(f"[agent t={self._t}] {advice.n()} cands | {advice.rationale[:200]}")
            # Instrumentation (07-04, logging only): persist the decoded LLM
            # proposals so the indirect DLS-seed channel can be attributed
            # post-hoc (pool-win distance-to-nearest-LLM-center, see
            # [pool-pick] llm_dist). No behavioral effect.
            try:
                if advice.centers is not None and getattr(self, "_advice_log_dir", None):
                    rows = [self.ssops.decode_flat(x.reshape(-1)) for x in advice.centers]
                    with open(os.path.join(self._advice_log_dir, "agent_advice.jsonl"), "a") as stream:
                        stream.write(json.dumps(
                            {
                                "t": self._t,
                                "centers": rows,
                                "proposal_meta": advice.proposal_meta,
                                "notes": advice.notes,
                                "enable_stage_signals": getattr(self, "enable_stage_signals", True),
                            },
                            default=str,
                        ) + "\n")
            except Exception:  # noqa: BLE001 — instrumentation must never break advise
                pass
        return advice
