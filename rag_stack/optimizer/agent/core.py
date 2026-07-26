"""Agent-guided qLogNEHVI with DOE, DLS, and one forced channel."""

from dataclasses import dataclass
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    WeightedMCMultiOutputObjective,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

from rag_stack.search_space.ax_search_space_ops import AxSearchSpaceOps
from ..base import BaseOptimizer
from ..utils import default_tkwargs
from .dls import DLSMixin
from .doe import DOECandidate, DOEController
from .forced import ForcedChannelMixin, _FRONTIER_NEGLECT
from .init_probes import InitProbesMixin
from .llm_agent import (
    AgentAdvice,
    AgentAdviceError,
    GapExplorationRequest,
    LLMAgent,
    ParetoGapAnchor,
    _agent_strict_env_override,
    _resolve_agent_wall_timeout,
)
from .prompts import PromptFeedMixin
from .util import _pareto_mask, _unique_rows_with_index

logger = logging.getLogger("RAG-Stack")


@dataclass(frozen=True)
class _LargestHVGap:
    """One exact, indexed maximum-area realized-frontier gap."""

    request: GapExplorationRequest
    quality_index: int
    performance_index: int
    quality_x: Tensor
    performance_x: Tensor
    area: float
    # Raw-objective affine frame used for both stagnation detection and
    # posterior gap-fill scoring: (p_min, p_span, q_min, q_span).
    normalization: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)


class AgentGuidedQNEHVI(
    InitProbesMixin, DLSMixin, ForcedChannelMixin, PromptFeedMixin,
    BaseOptimizer,
):
    """qLogNEHVI MOBO with LLM-agent candidate-injection (mechanism ①).

    Args:
        config_space: Ax ``SearchSpace``.
        objectives: list of {objective_name, ref_point, cost} (perf + quality).
        n_candidates: size of the random candidate pool scored each iter.
        agent_strict: fail the whole BO iteration on any agent/BO error.
        agent_wall_timeout_s: optional hard aggregate deadline per LLM chat.
        enable_agent: enable post-initialization Agent proposals and DLS seeds.
            Structured initialization is unchanged when this is false.
        enable_stage_signals: expose stage diagnostics to the Agent; evaluation
            and the two scalar GP targets are unchanged.
        enable_dls: enable all discrete local-search candidate sources.
        enable_doe: enable Pareto-tension crossover (the DOE implementation).
        enable_forced: enable stagnation-triggered Agent+PTC exploration.
    """

    def _create_agent(
        self,
        *,
        model: str,
        strict: bool,
        wall_timeout_s: Optional[float],
    ) -> LLMAgent:
        """Construct the Agent shared by initialization and optional BO advice."""
        return LLMAgent(
            model=model,
            strict=strict,
            wall_timeout_s=wall_timeout_s,
        )

    def __init__(
        self,
        config_space,
        objectives: List[dict],
        seed: int = 42,
        n_both_init: int = 10,
        init_sobol_k: int = 5,
        ssops: Optional[AxSearchSpaceOps] = None,
        agent_model: str = "deepseek-v4-flash",
        agent_strict: bool = False,            # paper runs: fail instead of silently changing policy
        agent_wall_timeout_s: Optional[float] = None,  # hard total budget per agent chat call
        n_candidates: int = 256,
        dls_num_restarts: int = 8,
        use_gpu: bool = False,
        enable_doe: bool = True,
        enable_agent: bool = True,
        enable_stage_signals: bool = True,
        enable_dls: bool = True,
        enable_forced: bool = True,
    ):
        # Exact booleans catch quoted YAML values such as "false", which would
        # otherwise silently leave a mechanism enabled in an ablation run.
        for name, value in {
            "enable_agent": enable_agent,
            "enable_stage_signals": enable_stage_signals,
            "enable_dls": enable_dls,
            "enable_doe": enable_doe,
            "enable_forced": enable_forced,
        }.items():
            if type(value) is not bool:
                raise ValueError(f"{name} must be a boolean, got {type(value).__name__}")
            setattr(self, name, value)
        self.search_space = config_space
        self.seed = int(seed)
        self.objective_names: List[str] = [o["objective_name"] for o in objectives]
        self.objective_costs: Dict[str, float] = {
            o["objective_name"]: float(o.get("cost", 1.0)) for o in objectives
        }
        self.n_both_init = int(n_both_init)
        # SOBOL-FIRST INIT (07-08, user directive): the first init_sobol_k
        # seats are hierarchy-aware Sobol probes evaluated BEFORE the LLM
        # designs its probes, so the probe design happens WITH data instead of
        # blind (blind probe sets caused the s45-old quality-corner starvation,
        # and their nondeterministic count made same-seed runs unreproducible).
        self.init_sobol_k = max(0, min(int(init_sobol_k), self.n_both_init))

        self.ssops: AxSearchSpaceOps = (
            ssops if ssops is not None
            else AxSearchSpaceOps(
                config_space,
                default_tkwargs(prefer_gpu=True) if use_gpu else {
                    "device": torch.device("cpu"), "dtype": torch.double,
                },
            )
        )
        # Share the ssops device/dtype so encode() tensors and GP tensors match.
        self.tkwargs = dict(self.ssops.tkwargs)
        self.dim = self.ssops.dim
        # Ordered-choice treatment (e.g. model-size ladders) is a SPACE-level
        # property now: declare {ordered: [...]} in YAML and ssops keeps the
        # dim out of cat_dims_set for every optimizer uniformly.
        self.cat_dims_set = self.ssops.cat_dims_set

        # Guard: this optimizer's acquisition step relies on discrete local search
        # (_dls_candidates -> _grid_choices), which needs every dim to be a
        # ChoiceParameter (it enumerates per-dim .values). A RangeParameter has no
        # .values, so _grid_choices would raise and DLS would SILENTLY degrade to
        # pool-only (Sobol breadth + agent inject, no local-search depth) every BO
        # iteration — a quiet quality regression, not a crash. Fail loudly instead:
        # write continuous dims as discrete lists `[v1, v2, ...]` in the config.
        from ax.core.parameter import RangeParameter
        range_dims = [
            n for n, p in self.ssops.ss.parameters.items()
            if isinstance(p, RangeParameter)
        ]
        if range_dims:
            raise ValueError(
                "AgentGuidedQNEHVI requires a fully discrete (all-ChoiceParameter) "
                "search space — its discrete-local-search acquisition step cannot "
                f"refine continuous RangeParameter dims {range_dims}. Express these "
                "as discrete lists `[v1, v2, ...]` in the config, or use ax_qnehvi / "
                "ax_qnehvi_mixedgp (which handle mixed spaces via optimize_acqf_mixed)."
            )

        self.standard_bounds = torch.zeros(2, self.dim, **self.tkwargs)
        self.standard_bounds[1] = 1.0
        self.ref_point = torch.tensor(
            [float(o["ref_point"]) for o in objectives], **self.tkwargs
        )
        self._perf_idx = self.objective_names.index("performance") if "performance" in self.objective_names else 0
        self._qual_idx = self.objective_names.index("quality") if "quality" in self.objective_names else 1

        # Initialization always retains its Agent. The enable_agent switch
        # controls BO only, so a source ablation does not change initialization.
        # Environment overrides let a hash-locked resumed paper project opt
        # into strictness without mutating its config snapshot.
        strict_override = _agent_strict_env_override()
        strict_requested = (
            bool(agent_strict) if strict_override is None else strict_override
        )
        resolved_wall_timeout = _resolve_agent_wall_timeout(agent_wall_timeout_s)
        self.agent_strict = strict_requested
        agent = self._create_agent(
            model=agent_model,
            strict=self.agent_strict,
            wall_timeout_s=resolved_wall_timeout,
        )
        self.agent_wall_timeout_s = getattr(agent, "wall_timeout_s", None)
        self.agent = agent
        self.agent.enable_stage_signals = self.enable_stage_signals
        # Forced delivery has one meaning: after realized normalized-HV
        # stagnation, explore the largest realized Pareto gap from one
        # deduplicated Agent+DOE candidate set.
        self.n_candidates = int(n_candidates)
        self.dls_num_restarts = int(dls_num_restarts)
        # Hierarchical-space correctness: our space gates knobs behind enabled/type
        # toggles. Without canonicalization, dead-knob values leak random variation
        # into the GP's X (phantom dims), give pool/dls candidates fake σ-novelty
        # along inactive dims, and let functionally identical configs look distinct.
        # Hierarchical clamping is therefore ALWAYS ON (07-07, option removed):
        # GP inputs and every generated candidate route through ssops.clamp()
        # (inactive dims → Ax dummy midpoint), the same convention Ax itself
        # applies via its Cast transform.
        # EQUAL-AXIS HV: rescale each
        # objective by its RUNNING observed range before qLogNEHVI's dominated-
        # area math, so a full-range quality step is worth a full-range perf
        # step — the same geometry the final verdict scores ([0,1]^2 union
        # normalization). Motivation: the new CM stretched the perf axis ~1.5x
        # and every raw-unit HV optimizer rationally starved the quality
        # corner (s43 07-13: 5 quality seats vs 12, qmax 0.394 vs 0.418).
        # This is the optimizer's sole HV geometry; there is no raw-unit
        # compatibility path or configuration switch.
        # One controller owns the entire DOE policy and state machine.  DOE is
        # a candidate-policy owner, never a scheduler: champion anchors,
        # history-backed tension, effective factor count, OAT/pair construction,
        # and its shared candidate pool are derived in one place. Core owns the
        # realized-HV stagnation clock and the single Agent+DOE forced
        # exploration arbitration.
        self.doe = DOEController(
            self.ssops,
            self._perf_idx,
            self._qual_idx,
            enabled=bool(enable_doe),
        )

        # Forced state is checkpointed only after a completed evaluation or
        # deterministic invalid result, so resume replays the same proposal
        # round after a pre-evaluation crash.
        self._forced_fired = False
        self._forced_state_loaded = False
        # Unified forced exploration. Only ordinary post-init evaluations age
        # this clock; one forced exploration resets it and must be followed by
        # a fresh run of normal qLogNEHVI opportunities.
        self._normal_no_hv_gain = 0
        self._stagnation_normalization: Optional[Dict[str, float]] = None
        self._last_accounted_eval_id = -1
        self._restoring_history = False
        self._pool_sobol_ss = None   # lazy Ax search space handle (pool breadth)
        self._x_bad: List[Tensor] = []   # registered-infeasible canonical rows (hard pool filter)

        # history (all in normalized [0,1]^dim space, which is what encode() emits)
        self._x_rows: List[Tensor] = []
        self._configs: List[dict] = []
        self._perf: List[float] = []
        self._corr: List[float] = []
        self._sub_metrics: List[dict] = []
        # eval_id → compact per-stage CM attribution (controller pushes via
        # register_perf_attribution; rendered into the LLM's obs/pareto cards).
        self._perf_attr: Dict[int, dict] = {}
        self._eval_ids: List[int] = []
        self._pending_full_metrics: List[bool] = []  # per-pending: full sub-metrics vs objective-only

        self.model: Optional[ModelListGP] = None
        self._t = 0           # BO iteration counter
        self._infeasible: List[dict] = []   # failed arms (advisory, LLM prompt only)
        self._sobol_calls = 0
        self._init_sobol_client = None   # lazy Ax hierarchy-aware Sobol (init phase)

        torch.manual_seed(self.seed)
        logger.info(
            f"AgentGuidedQNEHVI: dim={self.dim}, seed={seed}, n_init={self.n_both_init}, "
            f"agent={type(agent).__name__}, "
            f"agent_strict={self.agent_strict}, "
            f"agent_wall_timeout_s={self.agent_wall_timeout_s}, "
            f"n_candidates={self.n_candidates}, "
            f"dls_restarts={self.dls_num_restarts}, "
            f"device={self.tkwargs.get('device')}"
        )
        logger.info(
            "[agent-mechanisms] agent=%s stage_signals=%s dls=%s ptc=%s "
            "forced=%s initialization=sobol_then_agent",
            self.enable_agent, self.enable_stage_signals, self.enable_dls,
            self.enable_doe, self.enable_forced,
        )

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    def suggest(self, n_suggestions: int = 1) -> Tuple[Optional[List[dict]], Optional[List[str]]]:
        # init phase — full metrics (the sub-metric foundation the agent later reasons from).
        # STRUCTURED INIT (07-05): part of the budget goes to single-knob
        # extreme probes instead of pure Sobol — greedy-style stage sweeps own
        # both frontier ends from birth (observed: greedy hits perf≥200 at
        # GT2; Sobol-init agents need GT10-34), so the agent now opens with
        # the same structured moves before BO takes over.
        if len(self._configs) < self.n_both_init:
            configs = self._structured_init_batch(max(1, int(n_suggestions)))
            self._pending_full_metrics = [True] * len(configs)
            return configs, list(self.objective_names)
        # BO phase: enabled candidate sources share the ordinary acquisition
        # pool; the explicit forced switch controls the stagnation branch.
        try:
            self.doe.load_file(self._doe_state_path())
            self._load_forced_state()
            self._fit_models()
            forced_due = (
                getattr(self, "enable_forced", True)
                and self._forced_due(_FRONTIER_NEGLECT)
            )
            forced_gap = (
                self._largest_hv_gap(
                    normalization=self._stagnation_normalization,
                )
                if forced_due else None
            )
            forced_agent_call = (
                forced_due
                and forced_gap is not None
            )
            if not getattr(self, "enable_agent", True):
                advice = None
            elif forced_agent_call:
                # A due forced round has one Agent call. It is the bounded
                # largest-gap design call, not the
                # normal advisory call plus a second hidden request.
                advice = self._maybe_advise_forced_gap(forced_gap)
            else:
                if forced_due:
                    # Without an internal gap, preserve the forced clock and
                    # run the ordinary Agent candidate arbitration.
                    logger.info(
                        "[forced t=%s] due but no actionable gap request; "
                        "preserving clock and continuing ordinary arbitration",
                        self._t,
                    )
                advice = self._maybe_advise()
            config = self._propose_pool(
                advice,
                forced_due=bool(forced_due and forced_gap is not None),
                forced_gap=forced_gap,
            )
        except Exception as exc:
            if self.agent_strict:
                logger.error(
                    "AgentGuidedQNEHVI: strict BO step failed (%s); "
                    "aborting before suggestion/evaluation commit",
                    exc,
                )
                raise
            logger.warning(f"AgentGuidedQNEHVI: BO step failed ({exc}); falling back to random")
            self._forced_fired = False
            config = self._sobol_batch(1)[0]
        if config is None:
            # Pool exhausted (every candidate duplicates an evaluated config):
            # propagate the (None, None) stop contract instead of wrapping None
            # in a batch the controller would try to evaluate.
            self._pending_full_metrics = []
            logger.info("AgentGuidedQNEHVI: candidate pool exhausted — suggesting stop")
            return None, None
        # Keep evaluation semantics identical across ablations. Signals may
        # be hidden from the Agent without changing which metrics are measured.
        self._pending_full_metrics = [True]
        self._t += 1
        return [config], list(self.objective_names)

    def _encode_nested_config(self, config: dict) -> Tensor:
        """Encode one nested controller config with its exact active branch.

        ``encode_flat_config`` cannot disambiguate sibling component leaves
        that collapse to the same runtime path (for example two components'
        ``temperature`` knobs).  Recover the active Ax parameterization first,
        then encode that unambiguous flat mapping.
        """
        active = self.ssops.active_flat_config(
            config, require_all_active=True,
        )
        return self.ssops.encode_arm_params(active)

    def register_perf_attribution(self, eval_id: int, attr: dict) -> None:
        """Per-stage CM attribution of one eval (advisory; LLM prompt only)."""
        self._perf_attr[int(eval_id)] = attr

    def pending_observation_metadata(self) -> dict:
        """Metadata needed to replay the pending proposal's optimizer state."""
        return {
            "forced": bool(self._forced_fired),
            "agent_proposal_round": int(self._t),
        }

    @staticmethod
    def _parse_observation_metadata(
        metadata,
        *,
        eval_id: int,
    ) -> Tuple[bool, int]:
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"eval_{int(eval_id):04d} is missing optimizer observation "
                "metadata"
            )
        forced = metadata.get("forced")
        proposal_round = metadata.get("agent_proposal_round")
        if type(forced) is not bool:
            raise RuntimeError(
                f"eval_{int(eval_id):04d} optimizer metadata has no exact "
                "boolean forced field"
            )
        if (
            isinstance(proposal_round, bool)
            or not isinstance(proposal_round, int)
            or proposal_round < 0
        ):
            raise RuntimeError(
                f"eval_{int(eval_id):04d} optimizer metadata has no "
                "non-negative integer agent_proposal_round"
            )
        return forced, proposal_round

    def register_infeasible(
        self,
        config: dict,
        reason: str,
        *,
        eval_id: Optional[int] = None,
        optimizer_metadata: Optional[dict] = None,
        replay: bool = False,
    ) -> None:
        """An arm that FAILED evaluation (context overflow etc.) — invisible in
        the objective history. Two consumers (07-08):
        * ADVISORY — the INFEASIBLE prompt card (raw configs + raw error text;
          the LLM induces the rule itself);
        * HARD FILTER — the canonical encoded row joins ``_x_bad`` and every
          pool candidate matching it by decoded-active identity is dropped
          BEFORE ranking/acqf (arbitration must never pick a config the
          controller's known-invalid seatbelt would refuse as DUPLICATE)."""
        self._infeasible.append({"config": dict(config or {}), "reason": str(reason)[:220]})
        xb = None
        try:
            xb = self.ssops.clamp(self.ssops.snap_to_grid(
                self._encode_nested_config(dict(config or {})).reshape(1, -1)))
            self._x_bad.append(xb.reshape(-1).to(**self.tkwargs))
        except Exception:  # noqa: BLE001 — advisory list still works
            pass
        if xb is not None:
            # A DOE row can reach evaluation through either normal arbitration
            # or the stagnation forced set. If it is invalid, retire its pending
            # ticket instead of leaving a phantom in-flight interaction.
            self.doe.mark_failed(xb)
        if eval_id is not None:
            forced, proposal_round = self._parse_observation_metadata(
                optimizer_metadata,
                eval_id=int(eval_id),
            )
            self._t = max(self._t, proposal_round)
            changed = self._record_unscored_attempt(
                eval_id=int(eval_id),
                forced=forced,
            )
            if changed:
                self._save_forced_state()
        if not replay and not self._restoring_history:
            self.doe.save_file(self._doe_state_path())

    def _filter_known_bad(self, cand: "Tensor", prov: "list[str]"):
        """Drop pool candidates that match a registered-infeasible config by
        decoded-active identity (the same identity the pick-time dup check
        uses — a strict superset of the controller's string-key seatbelt)."""
        if not self._x_bad or cand is None or cand.shape[0] == 0:
            return cand, prov
        X_bad = torch.stack(self._x_bad, dim=0).to(cand.device, cand.dtype)
        keep = [i for i in range(cand.shape[0])
                if not self.ssops.is_duplicate_decoded(cand[i].reshape(1, -1), X_bad)]
        if len(keep) != cand.shape[0]:
            logger.info(
                f"[pool t={self._t}] known-invalid filter dropped "
                f"{cand.shape[0] - len(keep)} candidate(s)")
            idx = torch.tensor(keep, dtype=torch.long, device=cand.device)
            return cand[idx], [prov[i] for i in keep]
        return cand, prov

    def add_objectives(
        self,
        x,
        obj_dict,
        eval_ids=None,
        free_mask=None,
        optimizer_metadata=None,
    ) -> None:
        # free_mask (M3 fingerprint-inherited rows) is accepted for interface
        # parity: free rows are full GP observations here — budget accounting
        # lives in the controller / the greedy wrapper, not in this class.
        # Forced state: restore BEFORE the first landed/replayed eval can
        # snapshot over the persisted file (warm_start replays through here).
        self._load_forced_state()
        self.doe.load_file(self._doe_state_path())
        for i, cfg in enumerate(x):
            eval_id = int(eval_ids[i]) if eval_ids else -1
            row_metadata = (
                optimizer_metadata[i]
                if optimizer_metadata is not None
                and i < len(optimizer_metadata)
                else self.pending_observation_metadata()
            )
            forced, proposal_round = self._parse_observation_metadata(
                row_metadata,
                eval_id=eval_id,
            )
            xt = self._encode_nested_config(cfg).reshape(-1).to(**self.tkwargs)
            # canonicalize dead dims → functional dups share one X
            xt = self.ssops.clamp(xt.reshape(1, -1)).reshape(-1)
            if self._x_rows:
                existing = torch.stack(self._x_rows, dim=0).to(**self.tkwargs)
                if self.ssops.is_duplicate_decoded(xt.reshape(1, -1), existing):
                    logger.warning(
                        "add_objectives: ignoring duplicate decoded observation "
                        "for eval_id=%s; a repeated row must not receive extra GP weight",
                        eval_id,
                    )
                    continue
            perf_v = obj_dict["performance"][i]
            qual_v = obj_dict["quality"][i]
            if isinstance(qual_v, dict):
                corr = float(qual_v.get("combined"))
                sub = {k: v for k, v in qual_v.items() if k != "combined"}
            else:
                corr = float(qual_v)
                sub = {}
            perf_f = float(perf_v)
            # non-finite objectives (upstream aggregation slip) would poison the
            # GP fit and silently push every later suggest into the random
            # fallback — clamp to the 0-penalty trough and log loudly instead.
            if not (math.isfinite(perf_f) and math.isfinite(corr)):
                logger.warning(
                    "add_objectives: non-finite objectives (perf=%r, corr=%r) "
                    "for eval_id=%s; clamped to 0.0 (penalty trough)",
                    perf_f, corr, eval_id,
                )
                perf_f = perf_f if math.isfinite(perf_f) else 0.0
                corr = corr if math.isfinite(corr) else 0.0
            should_account = (
                proposal_round > 0
                and eval_id > self._last_accounted_eval_id
            )
            if should_account:
                self._record_realized_outcome(
                    perf_f,
                    corr,
                    eval_id=eval_id,
                    forced=forced,
                )
            self._x_rows.append(xt)
            self._configs.append(cfg)
            self._perf.append(perf_f)
            self._corr.append(corr)
            self._sub_metrics.append(sub)
            self._eval_ids.append(eval_id)
            self.doe.observe(
                xt,
                {"performance": perf_f, "quality": corr},
                eval_id=eval_id,
                forced=forced,
            )
        # POST-EVAL SNAPSHOT (the only save site): persisted forced state
        # is always consistent with the history a resume will replay.
        if not self._restoring_history:
            self._save_forced_state()
            self.doe.save_file(self._doe_state_path())

    def get_pareto(self) -> Tuple[List[dict], List[Dict[str, float]], List[int]]:
        if not self._configs:
            return [], [], []
        Y = np.array([[self._perf[i], self._corr[i]] for i in range(len(self._configs))])
        mask = _pareto_mask(Y)
        n = len(self._configs)
        cfgs = [self._configs[i] for i in range(n) if mask[i]]
        objs = [
            {"performance": self._perf[i], "quality": self._corr[i]}
            for i in range(n) if mask[i]
        ]
        eids = [self._eval_ids[i] for i in range(n) if mask[i]]
        return cfgs, objs, eids

    # ------------------------------------------------------------------
    # Unified forced-channel state
    # ------------------------------------------------------------------
    def warm_start(self, evaluations: List[Dict]) -> None:
        """Resume from a checkpoint by replaying historical evals through the
        same accounting :meth:`add_objectives` uses — so the GP training data
        (``_x_rows``, clamped), configs, perf/quality, sub-metrics and
        eval_ids are restored identically to a live run. ``_t`` is advanced past
        the init phase so the next :meth:`suggest` resumes in the same BO
        proposal round.

        Each eval needs ``config`` (decoded) and ``objectives`` with both
        objective names; ``quality`` may be the full dict (combined + sub-metrics)
        or a bare scalar. A config that fails to encode is skipped (best-effort
        resume) rather than aborting the whole warm-start."""
        self._load_forced_state()
        self.doe.load_file(self._doe_state_path())
        if self._last_accounted_eval_id < 0:
            # No authoritative state snapshot exists. Rebuild the clock from
            # the replayable observation ledger instead of layering history
            # onto possibly stale counters.
            self._normal_no_hv_gain = 0
            self._stagnation_normalization = None
        restored = 0
        restored_proposal_round = 0
        self._restoring_history = True
        try:
            for ev in sorted(
                evaluations,
                key=lambda item: int(item.get("eval_id", -1)),
            ):
                cfg = ev.get("config")
                objs = ev.get("objectives") or {}
                if cfg is None or any(o not in objs for o in self.objective_names):
                    continue
                optimizer_metadata = ev.get("optimizer_metadata")
                _, proposal_round = self._parse_observation_metadata(
                    optimizer_metadata,
                    eval_id=int(ev.get("eval_id", -1)),
                )
                restored_proposal_round = max(
                    restored_proposal_round,
                    proposal_round,
                )
                try:
                    before = len(self._configs)
                    self.add_objectives(
                        [cfg],
                        {name: [objs[name]] for name in self.objective_names},
                        [ev.get("eval_id", -1)],
                        free_mask=[bool(ev.get("free", False))],
                        optimizer_metadata=[optimizer_metadata],
                    )
                    restored += int(len(self._configs) > before)
                except Exception as e:  # noqa: BLE001 — one bad config must not kill resume
                    logger.warning(
                        "AgentGuidedQNEHVI.warm_start: skipping eval %s (%s)",
                        ev.get("eval_id"), e,
                    )
        finally:
            self._restoring_history = False
        # Resume in the BO phase: suggest() gates on len(_configs) >= n_both_init.
        # Keep the proposal round monotonic for deterministic Agent notes, DOE,
        # forced state, and random streams.
        history_round = max(0, len(self._configs) - self.n_both_init)
        self._t = max(history_round, restored_proposal_round)
        self._save_forced_state()
        self.doe.save_file(self._doe_state_path())
        logger.info(
            "AgentGuidedQNEHVI.warm_start: restored %d/%d evals (%d total, t=%d, %s)",
            restored, len(evaluations), len(self._configs), self._t,
            "BO phase" if len(self._configs) >= self.n_both_init else "init phase",
        )

    # ------------------------------------------------------------------
    # candidate-feasibility hooks (no-ops here)
    # ------------------------------------------------------------------
    # Subclasses (e.g. SLOAgentGuidedQNEHVI) override these to keep
    # infeasible configs out of the acqf pool / final suggestion. The base
    # optimizer accepts everything, so behavior is unchanged.

    def _prefilter_pool(self, cand: Tensor) -> Tensor:
        """Filter the candidate pool BEFORE acqf scoring."""
        return cand

    def _prefilter_pool_with_prov(self, cand: Tensor, prov: list) -> "tuple[Tensor, list]":
        """Provenance-aligned wrapper of :meth:`_prefilter_pool`. The base
        filter is identity today; a subclass that actually drops rows must
        override THIS to keep provenance aligned (drop the same indices)."""
        filtered = self._prefilter_pool(cand)
        if filtered.shape[0] == cand.shape[0]:
            return filtered, prov
        # base-class fallback: filter changed row count without index info —
        # provenance can't be trusted anymore, degrade labels to 'pool'
        return filtered, ["pool"] * filtered.shape[0]

    def _prefilter_forced_pool_with_prov(
        self,
        cand: Tensor,
        prov: list,
    ) -> "tuple[Tensor, list]":
        """Apply legality filters without ordinary-pool shortlist ranking.

        Calling the base provenance wrapper directly still dispatches to a
        subclass's feasibility-only ``_prefilter_pool`` hook, while bypassing
        a subclass override that ranks or truncates the normal qLogNEHVI pool.
        Forced exploration must score the complete legal Agent+DOE set against
        its selected objective-space gap.
        """
        return AgentGuidedQNEHVI._prefilter_pool_with_prov(
            self,
            cand,
            prov,
        )

    def _drop_evaluated_pool_rows(
        self, cand: Tensor, prov: list,
    ) -> "tuple[Tensor, list]":
        """Remove rows that decode to an already-evaluated configuration.

        Such rows can never be returned by the final proposal scan, so keeping
        them through subclass prefilters is both wasted work and semantically
        dangerous: a feasibility/ranking guard can retain an evaluated row
        while dropping every novel row, making a non-empty search space look
        exhausted.  Preserve candidate order and provenance exactly.
        """
        if not self._x_rows or cand.shape[0] == 0:
            return cand, prov
        existing = torch.stack(self._x_rows, dim=0).to(**self.tkwargs)
        keep = [
            i for i in range(cand.shape[0])
            if not self.ssops.is_duplicate_decoded(
                cand[i].reshape(1, -1), existing,
            )
        ]
        if len(keep) == cand.shape[0]:
            return cand, prov
        logger.info(
            "[pool-novel t=%s] removed %d/%d evaluated decoded duplicates",
            getattr(self, "_t", "?"), cand.shape[0] - len(keep), cand.shape[0],
        )
        if not keep:
            return cand[:0], []
        keep_t = torch.tensor(keep, dtype=torch.long, device=cand.device)
        return cand[keep_t], [prov[i] for i in keep]

    def _accept_candidate(self, x: Tensor, cfg: dict) -> bool:
        """Final gate on a non-duplicate candidate about to be suggested.

        Pool candidates pass through ``_filter_known_bad`` earlier, while the
        unified forced channel calls this hook directly. Keep the known-
        invalid identity check here as the common final seatbelt so no proposal
        path can re-spend a failed configuration.
        """
        if self._x_bad:
            bad = torch.stack(self._x_bad, dim=0).to(x.device, x.dtype)
            if self.ssops.is_duplicate_decoded(x.reshape(1, -1), bad):
                logger.info(
                    "[candidate-reject t=%s] known-invalid decoded identity",
                    getattr(self, "_t", "?"),
                )
                return False
        return True

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _fit_single_gp(self, X: Tensor, Y: Tensor):
        cat_dims = sorted(self.cat_dims_set)
        if len(cat_dims) == 0 or len(cat_dims) == self.dim:
            gp = SingleTaskGP(train_X=X, train_Y=Y, outcome_transform=Standardize(m=1))
        else:
            gp = MixedSingleTaskGP(
                train_X=X, train_Y=Y, cat_dims=cat_dims, outcome_transform=Standardize(m=1)
            )
        gp = gp.to(**self.tkwargs)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        return gp

    def _fit_models(self) -> None:
        X = torch.stack(self._x_rows, dim=0).to(**self.tkwargs)  # (N, dim) in [0,1]
        perf = torch.tensor(self._perf, **self.tkwargs).reshape(-1, 1)
        corr = torch.tensor(self._corr, **self.tkwargs).reshape(-1, 1)
        perf_gp = self._fit_single_gp(X, perf)
        corr_gp = self._fit_single_gp(X, corr)
        models = [None, None]
        models[self._perf_idx] = perf_gp
        models[self._qual_idx] = corr_gp
        self.model = ModelListGP(*models)

    def _build_acqf(self):
        x_baseline = torch.stack(self._x_rows, dim=0).to(**self.tkwargs)
        # Keep acquisition sampling on its own round-derived stream.  DOE
        # rounds may skip pool construction entirely; relying on the global
        # torch RNG here would make that harmless optimization change every
        # later qLogNEHVI draw for the same seed/history.
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([128]),
            seed=int((self.seed + 300_000 + self._t) % (2**31 - 1)),
        )
        # EQUAL-AXIS HV (see __init__): qLogNEHVI maximizes dominated
        # area in whatever units Y arrives in.  Match the final union-min/max
        # geometry exactly: divide each objective by its running observed
        # range and place the reference point at the correspondingly scaled
        # observed minimum.  (Subtracting the minimum from every sample with a
        # zero reference is algebraically identical.)  Scaling while leaving
        # ref=0 is *not* min-max normalization and gives the two axes unequal
        # baseline rectangles whenever their raw minima differ.
        # Rebuilt every round, so the bounds track the run. General rule — no
        # dataset/config constants.
        if not self._perf or not self._corr:
            raise RuntimeError(
                "normalized-HV acquisition requires at least one observed "
                "performance and quality value"
            )

        weights = torch.ones(len(self.objective_names), **self.tkwargs)
        ref = self.ref_point.to(**self.tkwargs).clone()
        for idx, values in (
            (self._perf_idx, self._perf),
            (self._qual_idx, self._corr),
        ):
            observed_min = float(min(values))
            observed_span = float(max(values) - observed_min)
            # A constant axis has no empirical range yet.  Keep it in the same
            # normalized geometry using its distance from the configured
            # reference. If that is also zero, the axis has no scale
            # information at all: give it zero weight and a transformed
            # reference of -1. Its fixed width of one then cancels from every
            # HV comparison, so ranking depends only on informative normalized
            # axes. Crucially, no case falls back to raw multi-objective HV.
            reference_span = abs(
                observed_min - float(self.ref_point[idx].item())
            )
            if observed_span > 1e-12:
                scale = observed_span
            elif reference_span > 1e-12:
                scale = reference_span
            else:
                weights[idx] = 0.0
                ref[idx] = -1.0
                continue
            weights[idx] = 1.0 / scale
            ref[idx] = observed_min / scale

        objective = WeightedMCMultiOutputObjective(weights=weights)
        return qLogNoisyExpectedHypervolumeImprovement(
            model=self.model,
            ref_point=ref,
            X_baseline=x_baseline,
            prune_baseline=True,
            sampler=sampler,
            cache_root=None,
            objective=objective,
        )

    def _doe_state_path(self) -> Optional[str]:
        directory = getattr(self, "_advice_log_dir", None)
        return os.path.join(str(directory), "doe_state.json") if directory else None

    def _doe_round_plan(self):
        """Return and audit the controller's one immutable plan for this round."""
        plan = self.doe.plan_round(
            t=self._t,
            accept_candidate=self._accept_candidate,
        )
        logger.info(
            "[doe t=%s] phases=%s oat_cells=%d pair_cells=%d "
            "reason=%s tension_score=%.9f",
            plan.audit.round, ",".join(plan.audit.phases) or "-",
            plan.audit.oat_cells, plan.audit.pair_cells,
            plan.audit.reason,
            plan.audit.tension_score,
        )
        return plan

    def _largest_hv_gap(
        self,
        *,
        normalization: Optional[Dict[str, float]] = None,
    ) -> Optional[_LargestHVGap]:
        """Build the maximum adjacent realized-frontier gap in one fixed frame.

        A forced episode passes the exact affine frame frozen when stagnation
        began. Ordinary Agent cards omit it and use the current realized range.
        Clipping to that frame's unit box prevents a later dominated outlier
        from silently redefining which empty rectangle the safety channel
        attacks.
        """
        count = len(self._x_rows)
        if count < 2:
            return None
        if normalization is None:
            p_min, p_max = float(min(self._perf)), float(max(self._perf))
            q_min, q_max = float(min(self._corr)), float(max(self._corr))
            p_span, q_span = p_max - p_min, q_max - q_min
        else:
            try:
                p_min = float(normalization["p_min"])
                p_span = float(normalization["p_span"])
                q_min = float(normalization["q_min"])
                q_span = float(normalization["q_span"])
            except (KeyError, TypeError, ValueError):
                return None
        if (
            not all(math.isfinite(value) for value in (
                p_min, p_span, q_min, q_span,
            ))
            or p_span <= self._HV_GAIN_EPS
            or q_span <= self._HV_GAIN_EPS
        ):
            return None
        normalized_performance = [
            max(0.0, min(1.0, (float(value) - p_min) / p_span))
            for value in self._perf
        ]
        normalized_quality = [
            max(0.0, min(1.0, (float(value) - q_min) / q_span))
            for value in self._corr
        ]

        frontier = [
            index
            for index in range(count)
            if not any(
                other != index
                and self._perf[other] >= self._perf[index]
                and self._corr[other] >= self._corr[index]
                and (
                    self._perf[other] > self._perf[index]
                    or self._corr[other] > self._corr[index]
                )
                for other in range(count)
            )
        ]
        frontier.sort(key=lambda index: (
            self._perf[index],
            -self._corr[index],
            self._eval_ids[index] if self._eval_ids[index] >= 0 else index,
            index,
        ))
        gaps = []
        for quality_index, performance_index in zip(frontier, frontier[1:]):
            p_width = (
                normalized_performance[performance_index]
                - normalized_performance[quality_index]
            )
            q_width = (
                normalized_quality[quality_index]
                - normalized_quality[performance_index]
            )
            if p_width > 0.0 and q_width > 0.0:
                gaps.append((
                    p_width * q_width,
                    quality_index,
                    performance_index,
                ))
        if not gaps:
            return None
        area, quality_index, performance_index = max(
            gaps,
            key=lambda item: (item[0], -item[1], -item[2]),
        )

        quality_x = self._x_rows[quality_index].detach().clone()
        performance_x = self._x_rows[performance_index].detach().clone()
        quality_id = self.doe._row_signature(quality_x)
        performance_id = self.doe._row_signature(performance_x)

        def _anchor(index: int) -> ParetoGapAnchor:
            eval_id = int(self._eval_ids[index])
            return ParetoGapAnchor(
                flat=self.ssops.decode_flat(self._x_rows[index]),
                normalized_performance=normalized_performance[index],
                normalized_quality=normalized_quality[index],
                sub_metrics=(
                    dict(self._sub_metrics[index])
                    if getattr(self, "enable_stage_signals", True)
                    and index < len(self._sub_metrics)
                    else {}
                ),
                perf_attr=(
                    self._perf_attr.get(eval_id)
                    if getattr(self, "enable_stage_signals", True) else None
                ),
            )

        request = GapExplorationRequest(
            gap_id=f"{quality_id}:{performance_id}",
            quality_anchor=_anchor(quality_index),
            performance_anchor=_anchor(performance_index),
            failed_nearby=tuple(self._infeasible[-12:]),
        )
        return _LargestHVGap(
            request=request,
            quality_index=quality_index,
            performance_index=performance_index,
            quality_x=quality_x,
            performance_x=performance_x,
            area=float(area),
            normalization=(p_min, p_span, q_min, q_span),
        )

    def _maybe_advise_forced_gap(
        self, gap: _LargestHVGap,
    ) -> Optional[AgentAdvice]:
        """Run the one due-round Agent call and enforce paper strictness."""
        if not getattr(self, "enable_agent", True):
            return None
        explore = getattr(self.agent, "explore_gap", None)
        if not callable(explore):
            message = (
                "forced exploration requires an Agent with explore_gap()"
            )
            if self.agent_strict:
                raise AgentAdviceError(message)
            logger.warning(message)
            return None
        try:
            advice = explore(
                gap.request,
                self._history(),
                self.ssops,
                self._t,
                independent_evidence=self._independent_evidence_payload(),
            )
        except AgentAdviceError as exc:
            if self.agent_strict:
                raise
            logger.warning(
                "[forced t=%s] source=agent call=rejected error=%s",
                self._t, exc,
            )
            return None
        except Exception as exc:
            if self.agent_strict:
                raise AgentAdviceError(
                    f"forced gap Agent call failed: {exc}",
                ) from exc
            logger.warning(
                "[forced t=%s] source=agent call=failed error=%s",
                self._t, exc,
            )
            return None
        if not isinstance(advice, AgentAdvice) or advice.n() == 0:
            raw = advice.raw if isinstance(advice, AgentAdvice) else None
            detail = raw.get("error") if isinstance(raw, dict) else "empty result"
            message = (
                "forced gap Agent produced no benchmark-usable candidate: "
                f"{detail}"
            )
            if self.agent_strict:
                raise AgentAdviceError(message)
            logger.warning(
                "[forced t=%s] source=agent call=empty error=%s",
                self._t, message,
            )
            return None
        logger.info(
            "[forced t=%s] source=agent call=accepted gap=%s "
            "area=%.9f candidates=%d",
            self._t,
            gap.request.gap_id,
            gap.area,
            advice.n(),
        )
        return advice

    @staticmethod
    def _expected_bounded_extent(
        mean: Tensor,
        variance: Tensor,
        lower: float,
        upper: float,
    ) -> Tensor:
        """Exact ``E[clip(Y, lower, upper) - lower]`` for normal ``Y``."""
        if not upper > lower:
            raise ValueError("forced gap bounds must be strictly ordered")
        variance = variance.clamp_min(0.0)
        std = variance.sqrt()
        safe_std = std.clamp_min(torch.finfo(mean.dtype).tiny)

        def _positive_part(threshold: float) -> Tensor:
            offset = mean - float(threshold)
            z = offset / safe_std
            cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            pdf = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
            stochastic = offset * cdf + safe_std * pdf
            deterministic = offset.clamp_min(0.0)
            return torch.where(std > 0.0, stochastic, deterministic)

        return (
            _positive_part(lower) - _positive_part(upper)
        ).clamp(min=0.0, max=float(upper - lower))

    def _forced_gap_coverage_scores(
        self,
        rows: Tensor,
        gap: _LargestHVGap,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Expected coverage of the chosen empty objective-space rectangle.

        Performance and quality use independent GPs in ``ModelListGP``. Their
        expected bounded extents therefore factor into an exact expected area,
        integrating posterior uncertainty without a sampler, fitted
        coefficient, or new candidate generator.
        """
        if self.model is None:
            raise RuntimeError("forced gap scoring requires a fitted GP model")
        p_min, p_span, q_min, q_span = gap.normalization
        if (
            not all(math.isfinite(value) for value in gap.normalization)
            or p_span <= 0.0
            or q_span <= 0.0
        ):
            raise ValueError("forced gap normalization is invalid")
        q_anchor = gap.request.quality_anchor
        p_anchor = gap.request.performance_anchor
        p_lo = q_anchor.normalized_performance
        p_hi = p_anchor.normalized_performance
        q_lo = p_anchor.normalized_quality
        q_hi = q_anchor.normalized_quality

        with torch.no_grad():
            posterior = self.model.posterior(rows.unsqueeze(-2))
            means = posterior.mean.reshape(rows.shape[0], -1)
            variances = posterior.variance.reshape(rows.shape[0], -1)

        p_mean = (means[:, self._perf_idx] - p_min) / p_span
        q_mean = (means[:, self._qual_idx] - q_min) / q_span
        p_variance = variances[:, self._perf_idx] / (p_span * p_span)
        q_variance = variances[:, self._qual_idx] / (q_span * q_span)
        p_extent = self._expected_bounded_extent(
            p_mean, p_variance, p_lo, p_hi,
        )
        q_extent = self._expected_bounded_extent(
            q_mean, q_variance, q_lo, q_hi,
        )
        return (
            p_extent * q_extent,
            p_extent,
            q_extent,
            p_mean,
            q_mean,
        )

    def _try_forced_exploration(
        self,
        plan,
        advice: Optional[AgentAdvice],
        gap: _LargestHVGap,
    ) -> Optional[dict]:
        """Spend the one ``forced`` channel from a deduplicated Agent+DOE set.

        The existing posterior ranks every legal candidate by its exact
        expected coverage of the maximum empty realized-frontier rectangle.
        Candidate source remains audit metadata: Agent and DOE receive no
        fixed quota or source weight.
        """
        # Distinguish a deliberately removed/exhausted source from a broken
        # posterior. Strict ablations may fall back only for the former.
        self._forced_candidates_unavailable = False
        records: Dict[str, dict] = {}

        def _add(row: Tensor, family: str, label: str, ticket=None) -> None:
            canonical = self.ssops.clamp(
                row.reshape(1, -1).to(**self.tkwargs),
            ).reshape(-1)
            signature = self.doe._row_signature(canonical)
            record = records.setdefault(signature, {
                "row": canonical,
                "families": set(),
                "labels": set(),
                "doe_ticket": None,
            })
            record["families"].add(family)
            record["labels"].add(label)
            if ticket is not None:
                record["doe_ticket"] = ticket

        if plan is not None and getattr(self.doe, "enabled", True):
            for candidate in plan.pool:
                _add(
                    candidate.x,
                    "doe",
                    candidate.provenance,
                    ticket=candidate,
                )
        if (
            getattr(self, "enable_agent", True)
            and isinstance(advice, AgentAdvice) and advice.centers is not None
        ):
            for row in advice.centers:
                _add(row, "agent", "agent")
        if not records:
            self._forced_candidates_unavailable = True
            return None

        ordered_signatures = sorted(records)
        rows = torch.stack([
            records[signature]["row"] for signature in ordered_signatures
        ], dim=0)
        rows, _ = self._filter_known_bad(rows, ordered_signatures)
        rows, _ = self._drop_evaluated_pool_rows(rows, ordered_signatures)
        rows, _ = self._prefilter_forced_pool_with_prov(
            rows,
            ordered_signatures,
        )
        if rows.shape[0] == 0:
            self._forced_candidates_unavailable = True
            return None

        try:
            (
                gap_scores,
                p_extents,
                q_extents,
                p_means,
                q_means,
            ) = self._forced_gap_coverage_scores(rows, gap)
        except Exception as exc:  # noqa: BLE001 — caller owns strictness
            logger.warning(
                "[forced t=%s] gap posterior scoring failed: %s",
                self._t,
                exc,
            )
            return None

        prepared: Dict[str, tuple] = {}
        any_accepted = False
        for index, row in enumerate(rows):
            canonical = self.ssops.clamp(row.reshape(1, -1)).reshape(-1)
            signature = self.doe._row_signature(canonical)
            record = records.get(signature)
            if record is None or signature in prepared:
                continue
            config = self.ssops.decode_to_config(canonical)
            if not self._accept_candidate(canonical.reshape(1, -1), config):
                continue
            any_accepted = True
            score = float(gap_scores[index].item())
            if not math.isfinite(score):
                continue
            prepared[signature] = (
                score,
                float(p_extents[index].item()),
                float(q_extents[index].item()),
                float(p_means[index].item()),
                float(q_means[index].item()),
                config,
                record,
            )
        if not prepared:
            self._forced_candidates_unavailable = not any_accepted
            return None

        candidate_id, selected = min(
            prepared.items(),
            key=lambda item: (-item[1][0], item[0]),
        )
        (
            score,
            p_extent,
            q_extent,
            p_mean,
            q_mean,
            config,
            record,
        ) = selected
        families = sorted(record["families"])
        source_family = families[0] if len(families) == 1 else "joint"
        ticket: Optional[DOECandidate] = record["doe_ticket"]
        if ticket is not None:
            self.doe.mark_selected(ticket)
        self._mark_forced()
        logger.info(
            "[forced-pick t=%s] source_family=%s all_sources=%s "
            "source_labels=%s candidate=%s gap=%s area=%.9f "
            "expected_gap_fill=%.9f gap_fill_fraction=%.9f "
            "expected_perf_extent=%.9f expected_quality_extent=%.9f "
            "posterior_normalized_perf_mean=%.9f "
            "posterior_normalized_quality_mean=%.9f "
            "(global qLogNEHVI bypassed)",
            self._t,
            source_family,
            ",".join(families),
            ",".join(sorted(record["labels"])),
            candidate_id,
            gap.request.gap_id,
            gap.area,
            score,
            score / gap.area if gap.area > 0.0 else 0.0,
            p_extent,
            q_extent,
            p_mean,
            q_mean,
        )
        return config

    def _pool_dls_seeds(self, acqf, random_cand: Tensor) -> Optional[Tensor]:
        """Rank novel Sobol breadth rows for the additive dls-pool source."""
        try:
            seed_cand, _ = _unique_rows_with_index(random_cand)
            seed_prov = ["sobol"] * seed_cand.shape[0]
            seed_cand, seed_prov = self._filter_known_bad(seed_cand, seed_prov)
            seed_cand, seed_prov = self._drop_evaluated_pool_rows(
                seed_cand, seed_prov,
            )
            if seed_cand.shape[0] == 0:
                raise ValueError("no novel breadth candidates for DLS restarts")
            with torch.no_grad():
                values = acqf(seed_cand.unsqueeze(-2)).reshape(-1)
            count = min(self.dls_num_restarts, seed_cand.shape[0])
            return seed_cand[values.argsort(descending=True)[:count]]
        except Exception as exc:  # noqa: BLE001 — seeds are additive only
            logger.debug(f"pool-seed scoring failed ({exc}); dls-pool skipped")
            return None

    def _propose_pool(
        self,
        advice: Optional[AgentAdvice],
        *,
        forced_due: bool,
        forced_gap: Optional[_LargestHVGap] = None,
    ) -> dict:
        """qLogNEHVI over a candidate pool = Sobol (breadth) ∪ LLM-inject (①) ∪
        DLS endpoints (depth; each logical source uses official multi-start
        DLS and contributes its best provenance-labelled endpoint).
        LogNEHVI scores every
        candidate and the argmax is evaluated; the pick log reports the
        winner's provenance (sobol / llm / dls-llm / dls-incumbent /
        dls-random) so channel win-rates are directly measurable."""
        self._forced_fired = False
        forced_due = bool(forced_due and getattr(self, "enable_forced", True))
        if not getattr(self, "enable_agent", True):
            # Guard the pool boundary as well as suggest(), so callers cannot
            # re-inject cached advice into an explicitly disabled source.
            advice = None

        # One immutable DOE plan feeds both delivery paths. Normal rounds put
        # its cells into qLogNEHVI; a due stagnation episode combines its exact
        # cells with the specialized largest-gap Agent result under one
        # deterministic, deduplicated exploration rule.
        doe_plan = self._doe_round_plan()
        doe_provenance_by_signature = {
            self.doe._row_signature(candidate.x): candidate.provenance
            for candidate in doe_plan.pool
        }
        if forced_due:
            if forced_gap is None:
                raise RuntimeError(
                    "forced channel is due without a concrete gap"
                )
            forced_cfg = self._try_forced_exploration(
                doe_plan, advice, forced_gap,
            )
            if forced_cfg is not None:
                return forced_cfg
        if forced_due:
            source_ablation = (
                not getattr(self, "enable_agent", True)
                or not getattr(self.doe, "enabled", True)
            )
            ablated_set_empty = (
                source_ablation
                and getattr(self, "_forced_candidates_unavailable", False)
            )
            if self.agent_strict and not ablated_set_empty:
                raise AgentAdviceError(
                    "forced Agent+DOE candidate set was exhausted after "
                    "deduplication, legality filtering, and gap scoring"
                )
            if ablated_set_empty:
                logger.info(
                    "[forced t=%s] ablation left no eligible forced candidates; "
                    "preserving the clock and continuing normal arbitration "
                    "(agent=%s ptc=%s)",
                    self._t, getattr(self, "enable_agent", True),
                    getattr(self.doe, "enabled", True),
                )
            else:
                logger.info(
                    "[forced t=%s] due Agent+DOE set exhausted; preserving the "
                    "clock and continuing normal arbitration",
                    self._t,
                )

        acqf = self._build_acqf()
        # base random candidate pool — breadth. Hierarchy-aware, ALWAYS
        # (07-07, option removed): arms sample ONLY active params; raw
        # full-cube Sobol below is the exception fallback.
        cand = self._hier_pool_sobol(self.n_candidates)
        if cand is None:
            raw = draw_sobol_samples(
                bounds=self.standard_bounds, n=self.n_candidates, q=1,
                seed=self.seed + 100_000 + self._t,
            ).squeeze(-2)
            cand = self.ssops.snap_to_grid(raw)
        random_cand = self.ssops.clamp(cand).to(**self.tkwargs)
        prov: list = ["sobol"] * cand.shape[0]

        # mechanism ① — inject the LLM's candidates directly into the pool.
        llm_x = None
        if advice is not None and advice.centers is not None:
            llm_x = advice.centers.to(**self.tkwargs)
            cand = torch.cat([cand, llm_x], dim=0)
            prov += ["llm"] * llm_x.shape[0]

        # DOE is an independent pool generator, not a DLS side effect.  Its
        # OAT/pair cells must remain available even if one local-search channel
        # fails, and they never reserve an evaluation seat on normal rounds.
        if doe_plan.pool:
            doe_rows = torch.cat([
                cell.x.reshape(1, -1).to(**self.tkwargs)
                for cell in doe_plan.pool
            ], dim=0)
            cand = torch.cat([cand, doe_rows], dim=0)
            prov += [cell.provenance for cell in doe_plan.pool]
            logger.info(
                "[doe-pool t=%s] oat_cells=%d pair_cells=%d",
                self._t, doe_plan.audit.oat_cells,
                doe_plan.audit.pair_cells,
            )

        # dls-pool seeds (07-08): score the breadth pool once and hand its
        # top-K unevaluated rows to the local search as GLOBAL restarts (the
        # sample→climb composition the generator was missing).
        pool_seeds = None
        if getattr(self, "enable_dls", True):
            pool_seeds = self._pool_dls_seeds(acqf, random_cand)

        # Each source's official multi-start endpoint joins the SAME pool —
        # one acqf scoring arbitrates across all sources after canonical
        # decoded deduplication. Disabled DLS performs no restart scoring.
        if getattr(self, "enable_dls", True):
            grouped = self._dls_candidates_grouped(acqf, advice, pool_seeds=pool_seeds)
        else:
            grouped = None
            self._dls_channel_diagnostics = {"round": int(self._t), "channels": {}}
        if grouped is not None:
            D, d_prov = grouped
            cand = torch.cat([cand, D.to(**self.tkwargs)], dim=0)
            prov += list(d_prov)

        # canonicalize inactive dims → functional duplicates collapse and the
        # acqf can't chase fake σ-novelty along dead knobs (always on, 07-07)
        cand = self.ssops.clamp(cand)
        if llm_x is not None:
            llm_x = self.ssops.clamp(llm_x)  # same canonicalization → comparable to picks
        cand, _keep = _unique_rows_with_index(cand)
        prov = [prov[i] for i in _keep]
        cand, prov = self._filter_known_bad(cand, prov)
        # pool-composition observability (07-07): per-source counts were never
        # logged, which blinded the s45 postmortem to the breadth/far-field split.
        _pc: Dict[str, int] = {}
        for _s in prov:
            _pc[_s] = _pc.get(_s, 0) + 1
        logger.info(
            f"[pool t={self._t}] "
            + " ".join(f"{s}={c}" for s, c in sorted(_pc.items()))
            + f" total={cand.shape[0]}"
        )

        # Rows already in the evaluation ledger can never win any path below.
        # Drop them BEFORE subclass ranking/guards so an unselectable duplicate
        # cannot become the sole survivor and falsely signal pool exhaustion.
        cand, prov = self._drop_evaluated_pool_rows(cand, prov)
        if cand.shape[0] == 0:
            logger.info(
                f"[pool-pick t={self._t}] pool exhausted "
                "(all candidates already evaluated)"
            )
            return None

        def _pick_source(x: Tensor, idx: int) -> str:
            """Provenance of the picked pool row (verbatim-LLM identity wins
            over a dedup that may have folded an llm row into a sobol twin)."""
            if llm_x is not None and llm_x.shape[0] and self.ssops.is_duplicate_decoded(x, llm_x):
                return "llm"
            doe_source = doe_provenance_by_signature.get(
                self.doe._row_signature(x.reshape(-1)),
            )
            if doe_source is not None:
                return doe_source
            return prov[idx] if 0 <= idx < len(prov) else "pool"
        # feasibility hook: known-bad candidates never get acqf-scored
        cand, prov = self._prefilter_pool_with_prov(cand, prov)

        with torch.no_grad():
            # qLogNEHVI — the multi-objective hypervolume signal is the sole arbiter:
            # agent-injected points (①) compete in this same scoring, picked only if
            # their acquisition value wins.
            vals = acqf(cand.unsqueeze(-2)).detach().reshape(-1)

        order = vals.argsort(descending=True)

        # Audit the exact agent proposals before any winner is returned.  DLS
        # endpoints and raw centers share this same final acquisition vector;
        # recording the raw ranks makes proposal starvation distinguishable
        # from prompt failure without changing arbitration.
        if llm_x is not None and llm_x.shape[0]:
            ranks = {int(idx): rank + 1 for rank, idx in enumerate(order.tolist())}
            for proposal_idx, center in enumerate(llm_x):
                matched = None
                for pool_idx in range(cand.shape[0]):
                    if self.ssops.is_duplicate_decoded(
                        center.reshape(1, -1), cand[pool_idx:pool_idx + 1],
                    ):
                        matched = pool_idx
                        break
                if matched is None:
                    logger.info(
                        f"[llm-rank t={self._t}] proposal={proposal_idx} "
                        "absent_after_filters"
                    )
                else:
                    logger.info(
                        f"[llm-rank t={self._t}] proposal={proposal_idx} "
                        f"rank={ranks[matched]}/{cand.shape[0]} "
                        f"acqf={vals[matched].item():.3f}"
                    )

        existing = torch.stack(self._x_rows, dim=0).to(**self.tkwargs)
        for idx in order.tolist():
            new_x = cand[idx].unsqueeze(0)
            if self.ssops.is_duplicate_decoded(new_x, existing):
                continue
            cfg = self.ssops.decode_to_config(new_x.reshape(-1))
            if not self._accept_candidate(new_x, cfg):
                continue
            _llm_dist = self._min_knob_dist(new_x, llm_x)
            _src = _pick_source(new_x, int(idx))
            logger.info(f"[pool-pick t={self._t}] source={_src} "
                        f"llm_dist={_llm_dist} "
                        f"(acqf={vals[idx].item():.3f}, pool={cand.shape[0]})")
            self._mark_dls_selected(_src)
            self.doe.mark_selected(new_x)
            return cfg
        # Every pool row duplicates an evaluated config (or is rejected):
        # re-evaluating a known point burns a success slot for zero information
        # (observed: one config judged 3x in one run) — signal exhaustion.
        logger.info(f"[pool-pick t={self._t}] pool exhausted (all duplicates/rejected)")
        return None
