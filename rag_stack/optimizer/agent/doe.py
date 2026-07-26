"""Champion-anchored design-of-experiments for the agent optimizer.

DOE has one owner.  The optimizer supplies realized observations and consumes
one immutable round plan; core, DLS, and forced delivery do not independently
choose anchors, rank dimensions, create cells, or persist DOE state.

For each objective direction, DOE starts from that single-objective champion
and looks for same-branch ``ChoiceParameter`` dimensions whose observed level
means exhibit a normalized performance/quality trade-off.  It emits
one-factor-at-a-time (OAT) ladders and, when the evidence has at least two
effective factors, one bounded top-two interaction cell.  The number of
effective factors is derived from the inverse participation ratio of the
history-backed tension scores; it is not a configured top-k.
Marginal level means are only screening evidence.  Exact landed
champion-relative OAT contrasts are exposed separately for audit and prompting.

The normal pool and the realized-HV-stagnation ``forced`` channel consume the
same immutable candidate pool. Core alone decides when the forced seat is due;
it joins these DOE cells with the maximum-gap Agent candidates and applies the
one shared forced exploration rule.

No benchmark, stage, knob name, preferred value, fixed cadence, factor count,
cell cap, or fitted threshold is encoded here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
from ax.core.parameter import ChoiceParameter
from torch import Tensor


logger = logging.getLogger("RAG-Stack")
CandidateAcceptor = Callable[[Tensor, dict], bool]


@dataclass(frozen=True)
class DOECandidate:
    x: Tensor
    kind: str  # oat | pair
    era_id: str

    @property
    def provenance(self) -> str:
        return f"doe-{self.kind}"


@dataclass(frozen=True)
class DOEAudit:
    round: int
    history_size: int
    tension_score: float
    phases: Tuple[str, ...]
    oat_cells: int
    pair_cells: int
    reason: str


@dataclass(frozen=True)
class DOEPlan:
    pool: Tuple[DOECandidate, ...]
    audit: DOEAudit


@dataclass(frozen=True)
class DOEAblationEvidence:
    """One landed, exact one-factor change from an objective champion.

    This is deliberately distinct from :class:`_ScreeningEvidence`: screening
    uses marginal level means to decide which factors deserve an experiment,
    while an ablation row is direct same-anchor OAT evidence suitable for an
    Agent evidence card.
    """

    side: str
    anchor_id: str
    branch_id: str
    parameter: str
    dim: int
    anchor_level: object
    observed_level: object
    delta_performance: float
    delta_quality: float
    utility: float
    eval_id: int
    forced: bool

    def as_dict(self) -> dict:
        return {
            "evidence_type": "landed_champion_oat",
            "kind": "oat",
            "side": self.side,
            "anchor_id": self.anchor_id,
            "branch_id": self.branch_id,
            "parameter": self.parameter,
            "dim": self.dim,
            "anchor_level": self.anchor_level,
            "observed_level": self.observed_level,
            "changed_knobs": [{
                "parameter": self.parameter,
                "dim": self.dim,
                "anchor_level": self.anchor_level,
                "observed_level": self.observed_level,
            }],
            "delta_performance": self.delta_performance,
            "delta_quality": self.delta_quality,
            "utility": self.utility,
            "eval_id": self.eval_id,
            "forced": self.forced,
            "delivery": "forced" if self.forced else "arbitrated",
            "interaction": None,
        }


@dataclass(frozen=True)
class DOEPairEvidence:
    """One exact landed two-factor row for the current champion's top factors."""

    side: str
    anchor_id: str
    branch_id: str
    changed_knobs: Tuple[Tuple[str, int, object, object], ...]
    delta_performance: float
    delta_quality: float
    utility: float
    eval_id: int
    forced: bool
    interaction_available: bool
    interaction_delta_performance: Optional[float]
    interaction_delta_quality: Optional[float]
    main_effect_eval_ids: Tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "evidence_type": "landed_champion_pair",
            "kind": "pair",
            "side": self.side,
            "anchor_id": self.anchor_id,
            "branch_id": self.branch_id,
            "changed_knobs": [
                {
                    "parameter": parameter,
                    "dim": dim,
                    "anchor_level": anchor_level,
                    "observed_level": observed_level,
                }
                for parameter, dim, anchor_level, observed_level
                in self.changed_knobs
            ],
            "delta_performance": self.delta_performance,
            "delta_quality": self.delta_quality,
            "utility": self.utility,
            "eval_id": self.eval_id,
            "forced": self.forced,
            "delivery": "forced" if self.forced else "arbitrated",
            "interaction": {
                "available": self.interaction_available,
                "delta_performance": self.interaction_delta_performance,
                "delta_quality": self.interaction_delta_quality,
                "main_effect_eval_ids": list(self.main_effect_eval_ids),
            },
        }


@dataclass(frozen=True)
class _ScreeningEvidence:
    """Marginal level-mean evidence used only to screen DOE factors."""

    dim: int
    tension: float
    score: float
    best_observed_utility: float
    observed_level_utilities: Tuple[Tuple[str, float], ...]


@dataclass(frozen=True)
class _Observation:
    x: Tensor
    performance: float
    quality: float
    eval_id: int
    forced: bool


class DOEController:
    """Generate champion-anchored OAT/pair cells for one shared candidate pool."""

    # The policy models first-order discovery followed by one second-order
    # confirmation.  This bounds generated interactions; it does not reserve
    # or schedule an evaluation seat.
    INTERACTION_ORDER = 2

    def __init__(
        self,
        ssops,
        perf_idx: int,
        qual_idx: int,
        *,
        enabled: bool = True,
    ) -> None:
        self.ssops = ssops
        self.perf_idx = int(perf_idx)
        self.qual_idx = int(qual_idx)
        self.enabled = bool(enabled)

        self._observations: list[_Observation] = []
        self._observation_signatures: set[str] = set()
        self._pair_landed: set[str] = set()
        self._selected: Dict[str, DOECandidate] = {}
        self._known_invalid: set[str] = set()
        self._ticket_index: Dict[str, DOECandidate] = {}
        self._history_revision = 0
        self._plan_cache_key: Optional[Tuple[int, int]] = None
        self._plan_cache: Optional[DOEPlan] = None
        self._loaded_state_path: Optional[str] = None

    def observe(
        self,
        x: Tensor,
        objectives,
        eval_id: int = -1,
        *,
        forced: bool = False,
    ) -> None:
        row = self._canonical_row(x)
        signature = self._row_signature(row)
        if signature in self._observation_signatures:
            return
        performance, quality = self._objective_pair(objectives)
        ticket = self._selected.pop(signature, None)
        self._observations.append(_Observation(
            x=row.detach().clone(),
            performance=performance,
            quality=quality,
            eval_id=int(eval_id),
            forced=bool(forced),
        ))
        self._observation_signatures.add(signature)
        if ticket is not None and ticket.kind == "pair":
            self._pair_landed.add(ticket.era_id)
        self._invalidate_plan()

    def plan_round(
        self,
        *,
        t: int,
        accept_candidate: Optional[CandidateAcceptor] = None,
    ) -> DOEPlan:
        key = (int(t), self._history_revision)
        if key == self._plan_cache_key and self._plan_cache is not None:
            return self._plan_cache
        plan = self._build_plan(
            t=int(t), accept_candidate=accept_candidate,
        )
        self._plan_cache_key = key
        self._plan_cache = plan
        self._ticket_index = {
            self._row_signature(candidate.x): candidate for candidate in plan.pool
        }
        return plan

    def landed_oat_evidence(self) -> Tuple[DOEAblationEvidence, ...]:
        """Return exact champion-relative OAT observations in normalized units.

        Marginal level means are useful for screening a broad historical sample,
        but they are confounded by the other coordinates in those samples.  This
        payload includes only observations whose active decoded configuration
        differs from the current objective champion in exactly one
        ``ChoiceParameter``.  Consumers can therefore present it as landed
        ablation evidence without overstating the screening statistic.
        """
        points = self._normalized_y() if self.enabled else []
        champions = self._champion_indices() if points else None
        if champions is None or champions[0] == champions[1]:
            return ()

        performance_champion, quality_champion = champions
        directions = [
            (performance_champion, "p"),
            (quality_champion, "q"),
        ]

        result = []
        for anchor_idx, side in directions:
            anchor = self._observations[anchor_idx]
            anchor_id = self._row_signature(anchor.x)
            branch_id = self._branch_id(anchor.x)
            anchor_decoded = self.ssops.decode_flat(anchor.x)
            for observation_idx, observation in enumerate(self._observations):
                if (
                    observation_idx == anchor_idx
                    or self._branch_id(observation.x) != branch_id
                ):
                    continue
                dims = self._active_diff_dims(anchor.x, observation.x)
                if len(dims) != 1:
                    continue
                dim = dims[0]
                name = self.ssops.param_names[dim]
                if not isinstance(
                    self.ssops.ss.parameters[name], ChoiceParameter,
                ):
                    continue
                decoded = self.ssops.decode_flat(observation.x)
                if name not in self.ssops.active_params(decoded):
                    continue
                delta_performance = (
                    points[observation_idx][0] - points[anchor_idx][0]
                )
                delta_quality = (
                    points[observation_idx][1] - points[anchor_idx][1]
                )
                result.append(DOEAblationEvidence(
                    side=side,
                    anchor_id=anchor_id,
                    branch_id=branch_id,
                    parameter=name,
                    dim=dim,
                    anchor_level=self._json_value(anchor_decoded.get(name)),
                    observed_level=self._json_value(decoded.get(name)),
                    delta_performance=float(delta_performance),
                    delta_quality=float(delta_quality),
                    utility=float(self._directional_utility(
                        points[anchor_idx], points[observation_idx], side,
                    )),
                    eval_id=observation.eval_id,
                    forced=observation.forced,
                ))
        return tuple(sorted(
            result,
            key=lambda item: (
                item.side,
                item.dim,
                item.eval_id if item.eval_id >= 0 else math.inf,
                self._level_key(item.observed_level),
            ),
        ))

    def landed_pair_evidence(self) -> Tuple[DOEPairEvidence, ...]:
        """Return exact landed rows changing the current top two DOE factors.

        Interaction residuals are reported only when both matching one-factor
        rows exist at the exact pair levels.  A missing main effect produces an
        explicit unavailable interaction rather than a posterior or marginal
        level-mean substitute.
        """
        points = self._normalized_y() if self.enabled else []
        champions = self._champion_indices() if points else None
        if champions is None or champions[0] == champions[1]:
            return ()

        performance_champion, quality_champion = champions
        directions = [
            (performance_champion, "p"),
            (quality_champion, "q"),
        ]

        result = []
        for anchor_idx, side in directions:
            ranked = self._ranked_screening_evidence(
                anchor_idx=anchor_idx,
                side=side,
                points=points,
            )
            if len(ranked) < self.INTERACTION_ORDER:
                continue
            top_dims = tuple(sorted(
                item.dim for item in ranked[: self.INTERACTION_ORDER]
            ))
            anchor = self._observations[anchor_idx]
            anchor_id = self._row_signature(anchor.x)
            branch_id = self._branch_id(anchor.x)
            anchor_decoded = self.ssops.decode_flat(anchor.x)

            for pair_idx, pair_observation in enumerate(self._observations):
                if (
                    pair_idx == anchor_idx
                    or self._branch_id(pair_observation.x) != branch_id
                    or self._active_diff_dims(
                        anchor.x, pair_observation.x,
                    ) != top_dims
                ):
                    continue
                pair_decoded = self.ssops.decode_flat(pair_observation.x)
                changes = tuple(
                    (
                        self.ssops.param_names[dim],
                        dim,
                        self._json_value(
                            anchor_decoded.get(self.ssops.param_names[dim]),
                        ),
                        self._json_value(
                            pair_decoded.get(self.ssops.param_names[dim]),
                        ),
                    )
                    for dim in top_dims
                )

                main_effect_indices = []
                for dim in top_dims:
                    name = self.ssops.param_names[dim]
                    observed_level = pair_decoded.get(name)
                    match = next(
                        (
                            observation_idx
                            for observation_idx, observation
                            in enumerate(self._observations)
                            if observation_idx != anchor_idx
                            and self._branch_id(observation.x) == branch_id
                            and self._active_diff_dims(
                                anchor.x, observation.x,
                            ) == (dim,)
                            and self.ssops.decode_flat(
                                observation.x,
                            ).get(name) == observed_level
                        ),
                        None,
                    )
                    if match is not None:
                        main_effect_indices.append(match)

                pair_delta_performance = (
                    points[pair_idx][0] - points[anchor_idx][0]
                )
                pair_delta_quality = (
                    points[pair_idx][1] - points[anchor_idx][1]
                )
                interaction_available = (
                    len(main_effect_indices) == self.INTERACTION_ORDER
                )
                if interaction_available:
                    main_delta_performance = sum(
                        points[index][0] - points[anchor_idx][0]
                        for index in main_effect_indices
                    )
                    main_delta_quality = sum(
                        points[index][1] - points[anchor_idx][1]
                        for index in main_effect_indices
                    )
                    interaction_delta_performance = (
                        pair_delta_performance - main_delta_performance
                    )
                    interaction_delta_quality = (
                        pair_delta_quality - main_delta_quality
                    )
                else:
                    interaction_delta_performance = None
                    interaction_delta_quality = None

                result.append(DOEPairEvidence(
                    side=side,
                    anchor_id=anchor_id,
                    branch_id=branch_id,
                    changed_knobs=changes,
                    delta_performance=float(pair_delta_performance),
                    delta_quality=float(pair_delta_quality),
                    utility=float(self._directional_utility(
                        points[anchor_idx], points[pair_idx], side,
                    )),
                    eval_id=pair_observation.eval_id,
                    forced=pair_observation.forced,
                    interaction_available=interaction_available,
                    interaction_delta_performance=(
                        float(interaction_delta_performance)
                        if interaction_delta_performance is not None
                        else None
                    ),
                    interaction_delta_quality=(
                        float(interaction_delta_quality)
                        if interaction_delta_quality is not None
                        else None
                    ),
                    main_effect_eval_ids=tuple(
                        self._observations[index].eval_id
                        for index in main_effect_indices
                    ),
                ))
        return tuple(sorted(
            result,
            key=lambda item: (
                item.side,
                item.eval_id if item.eval_id >= 0 else math.inf,
                tuple(change[1] for change in item.changed_knobs),
            ),
        ))

    def ablation_card_payload(self) -> Tuple[dict, ...]:
        """JSON-ready exact OAT and top-two pair evidence for an Agent card."""
        rows = [
            *(item.as_dict() for item in self.landed_oat_evidence()),
            *(item.as_dict() for item in self.landed_pair_evidence()),
        ]
        return tuple(sorted(
            rows,
            key=lambda row: (
                row["side"],
                row["eval_id"] if row["eval_id"] >= 0 else math.inf,
                row["kind"],
            ),
        ))

    def mark_selected(self, candidate_or_x) -> Optional[DOECandidate]:
        if isinstance(candidate_or_x, DOECandidate):
            candidate = candidate_or_x
        else:
            signature = self._row_signature(self._canonical_row(candidate_or_x))
            candidate = self._ticket_index.get(signature)
            if candidate is None:
                return None
        signature = self._row_signature(candidate.x)
        if signature not in self._selected:
            self._selected[signature] = candidate
            # A pending selection is no longer novel for either the normal DOE
            # pool or the forced set. Invalidate the immutable round plan now,
            # rather than waiting for its observation to arrive.
            self._invalidate_plan()
        return candidate

    def mark_failed(self, candidate_or_x) -> None:
        row = candidate_or_x.x if isinstance(candidate_or_x, DOECandidate) else candidate_or_x
        signature = self._row_signature(self._canonical_row(row))
        self._selected.pop(signature, None)
        self._known_invalid.add(signature)
        self._invalidate_plan()

    # ------------------------------------------------------------------
    # Resume state belongs to DOE, not the generic forced-channel ledger.
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "pair_landed": sorted(self._pair_landed),
            "known_invalid": sorted(self._known_invalid),
            "observations": [
                {
                    "x": [float(value) for value in obs.x.tolist()],
                    "performance": obs.performance,
                    "quality": obs.quality,
                    "eval_id": obs.eval_id,
                    "forced": obs.forced,
                }
                for obs in self._observations
            ],
        }

    def load_state_dict(self, state: Optional[Mapping]) -> None:
        if not state:
            return
        self._pair_landed = {str(value) for value in state.get("pair_landed", ())}
        self._known_invalid = {str(value) for value in state.get("known_invalid", ())}
        self._observations = []
        self._observation_signatures = set()
        for raw in state.get("observations", ()):
            try:
                row = self._canonical_row(torch.tensor(raw["x"], **self.ssops.tkwargs))
                signature = self._row_signature(row)
                if signature in self._observation_signatures:
                    continue
                self._observations.append(_Observation(
                    x=row,
                    performance=float(raw["performance"]),
                    quality=float(raw["quality"]),
                    eval_id=int(raw.get("eval_id", -1)),
                    forced=bool(raw.get("forced", False)),
                ))
                self._observation_signatures.add(signature)
            except (KeyError, TypeError, ValueError):
                continue
        # Pending DOE tickets are deliberately runtime-only. Production writes
        # this state only after a result lands or a candidate is rejected, when
        # no in-flight selection remains.
        self._selected = {}
        self._invalidate_plan()

    def load_file(self, path: Optional[str]) -> None:
        if not path or self._loaded_state_path == path:
            return
        self._loaded_state_path = path
        if not os.path.exists(path):
            return
        try:
            with open(path) as stream:
                self.load_state_dict(json.load(stream))
            logger.info(
                "[doe-state] restored observations=%d pairs=%d",
                len(self._observations), len(self._pair_landed),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[doe-state] load failed (%s); replaying history", exc)

    def save_file(self, path: Optional[str]) -> None:
        if not path:
            return
        try:
            temporary = path + ".tmp"
            with open(temporary, "w") as stream:
                json.dump(self.state_dict(), stream, sort_keys=True)
            os.replace(temporary, path)
            self._loaded_state_path = path
        except Exception as exc:  # noqa: BLE001
            logger.warning("[doe-state] save failed (%s)", exc)

    # ------------------------------------------------------------------
    # Evidence state machine and normalized Pareto geometry
    # ------------------------------------------------------------------

    def _build_plan(
        self,
        *,
        t: int,
        accept_candidate: Optional[CandidateAcceptor],
    ) -> DOEPlan:
        points = self._normalized_y() if self.enabled else []
        champions = self._champion_indices() if points else None
        if champions is None:
            return self._empty_plan(
                t,
                "disabled" if not self.enabled else "insufficient_champion_tradeoff",
            )

        performance_champion, quality_champion = champions
        if performance_champion == quality_champion:
            return self._empty_plan(t, "shared_objective_champion")

        directions = [
            (performance_champion, quality_champion, "p"),
            (quality_champion, performance_champion, "q"),
        ]

        pair_cells: list[DOECandidate] = []
        oat_cells: list[DOECandidate] = []
        phases = []
        tension_score = 0.0

        for anchor_idx, target_idx, side in directions:
            anchor = self._observations[anchor_idx]
            anchor_id = self._row_signature(anchor.x)
            branch_id = self._branch_id(anchor.x)
            era_id = self._digest(("champion-doe-v1", side, anchor_id, branch_id))
            evidence = self._ranked_screening_evidence(
                anchor_idx=anchor_idx,
                side=side,
                points=points,
            )
            if not evidence:
                phases.append(f"{side}:no_history_backed_tension")
                continue

            factor_count = self._effective_factor_count(evidence)
            selected_evidence = evidence[:factor_count]
            tension_score += sum(item.score for item in selected_evidence)
            ladders: Dict[int, Tuple[DOECandidate, ...]] = {}
            for item in selected_evidence:
                ladders[item.dim] = self._champion_oat_ladder(
                    anchor_idx=anchor_idx,
                    target_idx=target_idx,
                    evidence=item,
                    branch_id=branch_id,
                    era_id=era_id,
                    accept_candidate=accept_candidate,
                )
            for cells in ladders.values():
                oat_cells.extend(cells)

            local_evidence = {
                item.dim: self._best_local_oat_observation(
                    anchor_idx, item.dim, side, points,
                )
                for item in selected_evidence
            }
            top_two_evidence = tuple(selected_evidence[: self.INTERACTION_ORDER])
            top_two = tuple(item.dim for item in top_two_evidence)
            pair_ready = bool(
                len(top_two_evidence) == self.INTERACTION_ORDER
                and all(local_evidence.get(dim) is not None for dim in top_two)
            )
            intended_pair_row = (
                self._champion_pair_row(
                    anchor_idx=anchor_idx,
                    evidence=top_two_evidence,
                    local_evidence=local_evidence,
                    branch_id=branch_id,
                )
                if pair_ready
                else None
            )
            pair_observed = self._has_pair_observation(intended_pair_row)
            pair = None
            if (
                intended_pair_row is not None
                and era_id not in self._pair_landed
                and not pair_observed
            ):
                pair = self._champion_pair_candidate(
                    row=intended_pair_row,
                    era_id=era_id,
                    accept_candidate=accept_candidate,
                )
            if pair is not None:
                pair_cells.append(pair)

            if pair_observed or era_id in self._pair_landed:
                phases.append(f"{side}:exploit(k={factor_count})")
            elif pair is not None:
                phases.append(
                    f"{side}:pair_confirmation(k={factor_count})"
                )
            elif any(ladders.values()):
                phases.append(f"{side}:oat_discovery(k={factor_count})")
            else:
                phases.append(f"{side}:exhausted(k={factor_count})")

        # Pair first keeps the single bounded interaction visible under any
        # downstream deterministic pool truncation. Every cell remains an
        # ordinary shared-pool candidate; DOE never schedules a forced seat.
        pool = self._unique_candidates([*pair_cells, *oat_cells])

        return DOEPlan(
            pool=pool,
            audit=DOEAudit(
                round=t,
                history_size=len(self._observations),
                tension_score=float(tension_score),
                phases=tuple(phases),
                oat_cells=sum(cell.kind == "oat" for cell in pool),
                pair_cells=sum(cell.kind == "pair" for cell in pool),
                reason="champion_tension",
            ),
        )

    def _normalized_y(self) -> list[Tuple[float, float]]:
        if len(self._observations) < 2:
            return []
        performances = [obs.performance for obs in self._observations]
        qualities = [obs.quality for obs in self._observations]
        p_low, p_high = min(performances), max(performances)
        q_low, q_high = min(qualities), max(qualities)
        if p_high <= p_low or q_high <= q_low:
            return []
        return [
            ((p - p_low) / (p_high - p_low), (q - q_low) / (q_high - q_low))
            for p, q in zip(performances, qualities)
        ]

    def _champion_indices(self) -> Optional[Tuple[int, int]]:
        if not self._observations:
            return None

        def _key(index: int, objective: int) -> tuple:
            observation = self._observations[index]
            values = (observation.performance, observation.quality)
            other = 1 - objective
            # Earliest landed observation wins a complete objective tie.  The
            # row index remains the deterministic fallback for legacy eval_id=-1.
            eval_order = observation.eval_id if observation.eval_id >= 0 else index
            return values[objective], values[other], -eval_order, -index

        performance = max(
            range(len(self._observations)), key=lambda index: _key(index, 0),
        )
        quality = max(
            range(len(self._observations)), key=lambda index: _key(index, 1),
        )
        return performance, quality

    @classmethod
    def _level_key(cls, value) -> str:
        return json.dumps(
            cls._json_value(value),
            sort_keys=True,
            separators=(",", ":"),
        )

    def _legal_graft_levels(
        self, anchor: Tensor, dim: int, branch_id: str,
    ) -> list[Tuple[int, float, Tensor, str]]:
        name = self.ssops.param_names[dim]
        result = []
        for order, level in enumerate(self.ssops.choice_grid(name)):
            row = anchor.reshape(1, -1).clone()
            row[0, dim] = float(level)
            row = self._canonical_row(row)
            decoded = self.ssops.decode_flat(row)
            if (
                self._branch_id(row) != branch_id
                or name not in self.ssops.active_params(decoded)
            ):
                continue
            result.append((
                order,
                float(row[dim]),
                row,
                self._level_key(decoded.get(name)),
            ))
        return result

    @staticmethod
    def _directional_utility(
        anchor: Tuple[float, float],
        candidate: Tuple[float, float],
        side: str,
    ) -> float:
        """Scale-free gain discounted by, but never erased by, endpoint cost."""
        preserve, gain = (0, 1) if side == "p" else (1, 0)
        directional_gain = max(0.0, candidate[gain] - anchor[gain])
        preservation_cost = max(0.0, anchor[preserve] - candidate[preserve])
        return directional_gain / (1.0 + preservation_cost)

    def _screening_dimension_evidence(
        self,
        *,
        anchor_idx: int,
        dim: int,
        side: str,
        points: Sequence[Tuple[float, float]],
    ) -> Optional[_ScreeningEvidence]:
        anchor = self._observations[anchor_idx].x
        anchor_decoded = self.ssops.decode_flat(anchor)
        name = self.ssops.param_names[dim]
        parameter = self.ssops.ss.parameters[name]
        if (
            not isinstance(parameter, ChoiceParameter)
            or name not in self.ssops.active_params(anchor_decoded)
        ):
            return None

        branch_id = self._branch_id(anchor)
        legal = self._legal_graft_levels(anchor, dim, branch_id)
        if len({item[3] for item in legal}) < 2:
            return None
        legal_by_key = {item[3]: item for item in legal}

        groups: Dict[str, list[int]] = {}
        for index, observation in enumerate(self._observations):
            if self._branch_id(observation.x) != branch_id:
                continue
            decoded = self.ssops.decode_flat(observation.x)
            if name not in self.ssops.active_params(decoded):
                continue
            key = self._level_key(decoded.get(name))
            if key in legal_by_key:
                groups.setdefault(key, []).append(index)
        if len(groups) < 2:
            return None

        means = {
            key: (
                sum(points[index][0] for index in indices) / len(indices),
                sum(points[index][1] for index in indices) / len(indices),
            )
            for key, indices in groups.items()
        }
        total = float(sum(len(indices) for indices in groups.values()))
        weights = {
            key: len(indices) / total for key, indices in groups.items()
        }

        # ANOVA-style level means expose a factor's measured trade-off without
        # naming a preferred value.  Pairwise rectangle energy is positive only
        # when performance and quality move in opposite directions.
        tension = 0.0
        keys = sorted(means)
        for left_index, left in enumerate(keys):
            for right in keys[left_index + 1:]:
                delta_p = means[left][0] - means[right][0]
                delta_q = means[left][1] - means[right][1]
                tension += (
                    2.0
                    * weights[left]
                    * weights[right]
                    * max(0.0, -(delta_p * delta_q))
                )
        if tension <= 0.0:
            return None

        anchor_key = self._level_key(anchor_decoded.get(name))
        if anchor_key not in means:
            return None
        anchor_mean = means[anchor_key]
        utilities: Dict[str, float] = {}
        for key, mean in means.items():
            if key == anchor_key:
                utilities[key] = 0.0
                continue
            utilities[key] = self._directional_utility(
                anchor_mean,
                mean,
                side,
            )
        positive = [
            (utility, key)
            for key, utility in utilities.items()
            if utility > 0.0 and key in legal_by_key
        ]
        if not positive:
            return None
        best_utility, best_key = max(
            positive,
            key=lambda item: (
                item[0],
                -legal_by_key[item[1]][0],
            ),
        )
        score = tension * best_utility
        if score <= 0.0:
            return None
        return _ScreeningEvidence(
            dim=dim,
            tension=float(tension),
            score=float(score),
            best_observed_utility=float(best_utility),
            observed_level_utilities=tuple(sorted(utilities.items())),
        )

    def _ranked_screening_evidence(
        self,
        *,
        anchor_idx: int,
        side: str,
        points: Sequence[Tuple[float, float]],
    ) -> list[_ScreeningEvidence]:
        result = []
        for dim in range(len(self.ssops.param_names)):
            evidence = self._screening_dimension_evidence(
                anchor_idx=anchor_idx,
                dim=dim,
                side=side,
                points=points,
            )
            if evidence is not None:
                result.append(evidence)
        return sorted(
            result,
            key=lambda item: (
                item.score,
                item.tension,
                item.best_observed_utility,
                -item.dim,
            ),
            reverse=True,
        )

    @staticmethod
    def _effective_factor_count(
        evidence: Sequence[_ScreeningEvidence],
    ) -> int:
        """Scale-free effective support of the tension-score distribution."""
        if not evidence:
            return 0
        total = sum(item.score for item in evidence)
        if total <= 0.0:
            return 0
        probabilities = [item.score / total for item in evidence]
        effective = 1.0 / sum(value * value for value in probabilities)
        # Half-up rounding is deterministic (Python's round-to-even is not the
        # intended cardinality rule at an exact half).
        count = int(math.floor(effective + 0.5))
        return min(len(evidence), max(1, count))

    def _champion_oat_ladder(
        self,
        *,
        anchor_idx: int,
        target_idx: int,
        evidence: _ScreeningEvidence,
        branch_id: str,
        era_id: str,
        accept_candidate: Optional[CandidateAcceptor],
    ) -> Tuple[DOECandidate, ...]:
        anchor = self._observations[anchor_idx].x
        target = self._observations[target_idx].x
        target_decoded = self.ssops.decode_flat(target)
        name = self.ssops.param_names[evidence.dim]
        target_key = self._level_key(target_decoded.get(name))
        utilities = dict(evidence.observed_level_utilities)
        levels = self._legal_graft_levels(anchor, evidence.dim, branch_id)
        levels.sort(
            key=lambda item: (
                -utilities.get(item[3], 0.0),
                item[3] != target_key,
                item[0],
            ),
        )
        result = []
        for _, _, row, level_key in levels:
            if torch.equal(row, anchor) or not self._novel_valid(
                row, accept_candidate,
            ):
                continue
            result.append(DOECandidate(
                x=row,
                kind="oat",
                era_id=era_id,
            ))
        return tuple(result)

    def _best_local_oat_observation(
        self,
        anchor_idx: int,
        dim: int,
        side: str,
        points: Sequence[Tuple[float, float]],
    ) -> Optional[_Observation]:
        anchor_observation = self._observations[anchor_idx]
        anchor = anchor_observation.x
        branch_id = self._branch_id(anchor)
        candidates = []
        for index, observation in enumerate(self._observations):
            if (
                observation is anchor_observation
                or self._branch_id(observation.x) != branch_id
                or self._active_diff_dims(anchor, observation.x) != (dim,)
            ):
                continue
            utility = self._directional_utility(
                points[anchor_idx],
                points[index],
                side,
            )
            if utility > 0.0:
                candidates.append((utility, -index, observation))
        if not candidates:
            return None
        return max(candidates, key=lambda item: item[:2])[2]

    def _champion_pair_row(
        self,
        *,
        anchor_idx: int,
        evidence: Tuple[_ScreeningEvidence, _ScreeningEvidence],
        local_evidence: Mapping[int, Optional[_Observation]],
        branch_id: str,
    ) -> Optional[Tensor]:
        row = self._observations[anchor_idx].x.reshape(1, -1).clone()
        for item in evidence:
            observation = local_evidence.get(item.dim)
            if observation is None:
                return None
            row[0, item.dim] = observation.x[item.dim]
        row = self._canonical_row(row)
        decoded = self.ssops.decode_flat(row)
        active = self.ssops.active_params(decoded)
        dims = tuple(sorted(item.dim for item in evidence))
        if (
            self._branch_id(row) != branch_id
            or any(self.ssops.param_names[dim] not in active for dim in dims)
        ):
            return None
        return row

    def _champion_pair_candidate(
        self,
        *,
        row: Tensor,
        era_id: str,
        accept_candidate: Optional[CandidateAcceptor],
    ) -> Optional[DOECandidate]:
        if not self._novel_valid(row, accept_candidate):
            return None
        return DOECandidate(
            x=row,
            kind="pair",
            era_id=era_id,
        )

    def _has_pair_observation(
        self, intended_row: Optional[Tensor],
    ) -> bool:
        return bool(
            intended_row is not None
            and self._row_signature(intended_row) in self._observation_signatures
        )

    # ------------------------------------------------------------------
    # Canonical identity and serialization helpers
    # ------------------------------------------------------------------

    def _canonical_row(self, x: Tensor) -> Tensor:
        row = x.detach().clone().reshape(1, -1).to(**self.ssops.tkwargs)
        return self.ssops.clamp(self.ssops.snap_to_grid(row)).reshape(-1)

    def _row_signature(self, x: Tensor) -> str:
        decoded = self.ssops.decode_flat(x.reshape(-1))
        active = self.ssops.active_params(decoded)
        identity = tuple(
            (name, self._json_value(decoded.get(name))) for name in sorted(active)
        )
        return self._digest(identity)

    def _branch_id(self, x: Tensor) -> str:
        decoded = self.ssops.decode_flat(x.reshape(-1))
        active = self.ssops.active_params(decoded)
        selectors = []
        for name, parameter in self.ssops.ss.parameters.items():
            if (
                name in active
                and isinstance(parameter, ChoiceParameter)
                and parameter.is_hierarchical
            ):
                selectors.append((name, self._json_value(decoded.get(name))))
        return self._digest(tuple(sorted(selectors)))

    def _active_diff_dims(self, left: Tensor, right: Tensor) -> Tuple[int, ...]:
        left_decoded = self.ssops.decode_flat(left.reshape(-1))
        right_decoded = self.ssops.decode_flat(right.reshape(-1))
        common = self.ssops.active_params(left_decoded) & self.ssops.active_params(right_decoded)
        return tuple(
            dim for dim, name in enumerate(self.ssops.param_names)
            if name in common and left_decoded.get(name) != right_decoded.get(name)
        )

    def _novel_valid(
        self, row: Tensor, accept_candidate: Optional[CandidateAcceptor],
    ) -> bool:
        signature = self._row_signature(row)
        if (
            signature in self._observation_signatures
            or signature in self._known_invalid
            or signature in self._selected
        ):
            return False
        if accept_candidate is None:
            return True
        config = self.ssops.decode_to_config(row.reshape(-1))
        return bool(accept_candidate(row.reshape(1, -1), config))

    def _objective_pair(self, objectives) -> Tuple[float, float]:
        if isinstance(objectives, Mapping):
            performance, quality = objectives["performance"], objectives["quality"]
        else:
            performance, quality = objectives[self.perf_idx], objectives[self.qual_idx]
        if isinstance(performance, Mapping):
            performance = performance.get("combined")
        if isinstance(quality, Mapping):
            quality = quality.get("combined")
        performance, quality = float(performance), float(quality)
        if not (math.isfinite(performance) and math.isfinite(quality)):
            raise ValueError("DOE observations require finite objectives")
        return performance, quality

    def _unique_candidates(
        self, candidates: Sequence[DOECandidate],
    ) -> Tuple[DOECandidate, ...]:
        seen, result = set(), []
        for candidate in candidates:
            signature = self._row_signature(candidate.x)
            if signature not in seen:
                seen.add(signature)
                result.append(candidate)
        return tuple(result)

    def _empty_plan(self, t: int, reason: str) -> DOEPlan:
        return DOEPlan(
            pool=(),
            audit=DOEAudit(
                round=t,
                history_size=len(self._observations),
                tension_score=0.0,
                phases=(),
                oat_cells=0,
                pair_cells=0,
                reason=reason,
            ),
        )

    def _invalidate_plan(self) -> None:
        self._history_revision += 1
        self._plan_cache_key = None
        self._plan_cache = None
        self._ticket_index = {}

    @staticmethod
    def _json_value(value):
        return value if isinstance(value, (str, int, float, bool)) or value is None else repr(value)

    @staticmethod
    def _digest(value) -> str:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

__all__ = [
    "DOEAblationEvidence",
    "DOEAudit",
    "DOECandidate",
    "DOEController",
    "DOEPairEvidence",
    "DOEPlan",
]
