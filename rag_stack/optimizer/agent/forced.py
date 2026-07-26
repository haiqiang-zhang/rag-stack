"""Realized-HV stagnation state for the one forced-delivery channel.

DOE owns champion anchors, tension evidence, and OAT/pair candidates. Core
owns the single ``forced`` arbitration: after realized normalized-HV
stagnation, one candidate from the deduplicated Agent+DOE exploration set
bypasses normal arbitration. Candidate source is metadata, never another
delivery channel.
"""

import json
import logging
import math
import os
from typing import Dict, Iterable, Optional, Tuple

logger = logging.getLogger("RAG-Stack")

# After three ordinary evaluations add no realized normalized hypervolume, the
# next eligible proposal may use the one forced Agent+DOE exploration seat.
_FRONTIER_NEGLECT = 3


class ForcedChannelMixin:
    _HV_GAIN_EPS = 1e-12

    def _forced_state_path(self) -> Optional[str]:
        d = getattr(self, "_advice_log_dir", None)
        return os.path.join(str(d), "forced_state.json") if d else None

    def _load_forced_state(self) -> None:
        """Restore forced-channel state from the project dir (once, best-effort).
        No-op until the controller injects ``_advice_log_dir`` (keeps retrying
        until then); a missing file = fresh run; a corrupt file must never
        poison a resume — defaults win."""
        if self._forced_state_loaded:
            return
        path = self._forced_state_path()
        if path is None:
            return
        self._forced_state_loaded = True
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                st = json.load(f)
            if not isinstance(st, dict):
                raise ValueError("state is not a mapping")
            last_accounted_eval_id = st.get("last_accounted_eval_id", -1)
            if (
                isinstance(last_accounted_eval_id, bool)
                or not isinstance(last_accounted_eval_id, int)
            ):
                raise ValueError("last_accounted_eval_id must be an integer")
            forced = st.get("forced") or {}
            if not isinstance(forced, dict):
                raise ValueError("forced must be a mapping")
            normal_no_hv_gain = forced.get("normal_no_hv_gain", 0)
            if (
                isinstance(normal_no_hv_gain, bool)
                or not isinstance(normal_no_hv_gain, int)
                or normal_no_hv_gain < 0
            ):
                raise ValueError(
                    "normal_no_hv_gain must be a non-negative integer"
                )
            raw_scale = forced.get("normalization")
            normalization = None
            if isinstance(raw_scale, dict):
                required = {"p_min", "p_span", "q_min", "q_span"}
                if set(raw_scale) != required:
                    raise ValueError("normalization has unexpected fields")
                parsed = {key: float(raw_scale[key]) for key in required}
                if (
                    not all(math.isfinite(value) for value in parsed.values())
                    or parsed["p_span"] <= 0.0
                    or parsed["q_span"] <= 0.0
                ):
                    raise ValueError("normalization is invalid")
                normalization = parsed
            elif raw_scale is not None:
                raise ValueError("normalization must be a mapping or null")

            # Commit only after the whole payload validates. A corrupt suffix
            # cannot leave an accepted last_accounted_eval_id behind.
            self._last_accounted_eval_id = last_accounted_eval_id
            self._normal_no_hv_gain = normal_no_hv_gain
            self._stagnation_normalization = normalization
            logger.info(
                "[forced-state] restored: "
                f"normal_no_hv_gain={self._normal_no_hv_gain}"
            )
        except Exception as e:  # noqa: BLE001 — state must never break a resume
            logger.warning(f"[forced-state] load failed ({e}); using fresh defaults")

    def _save_forced_state(self) -> None:
        """Atomically snapshot state after replayable objectives are committed."""
        path = self._forced_state_path()
        if path is None:
            return
        try:
            st = {
                "last_accounted_eval_id": self._last_accounted_eval_id,
                "forced": {
                    "normal_no_hv_gain": self._normal_no_hv_gain,
                    "normalization": self._stagnation_normalization,
                },
            }
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[forced-state] save failed ({e})")

    def _mark_forced(self) -> None:
        """Record the one forced delivery allowed in a proposal round."""
        if self._forced_fired:
            raise RuntimeError(
                "forced invariant violated: this round's only forced seat "
                "was already spent"
            )
        self._forced_fired = True

    # ------------------------------------------------------------------
    # Realized normalized-HV stagnation and unified forced state
    # ------------------------------------------------------------------

    @staticmethod
    def _hv2(points: Iterable[Tuple[float, float]]) -> float:
        """Exact 2-D maximization hypervolume above reference ``(0, 0)``.

        The caller supplies points in one frozen normalization frame.  Values
        above one are intentionally retained: a new record outside the scale
        frozen at the start of a stagnation episode is real progress, not a
        value to clamp away.
        """
        pts = [
            (float(p), float(q)) for p, q in points
            if math.isfinite(float(p)) and math.isfinite(float(q))
            and float(p) > 0.0 and float(q) > 0.0
        ]
        frontier = []
        for i, point in enumerate(pts):
            if any(
                j != i
                and other[0] >= point[0]
                and other[1] >= point[1]
                and (other[0] > point[0] or other[1] > point[1])
                for j, other in enumerate(pts)
            ):
                continue
            frontier.append(point)
        area = 0.0
        previous = 0.0
        for performance, quality in sorted(frontier):
            if performance <= previous:
                continue
            area += (performance - previous) * quality
            previous = performance
        return float(area)

    def _derive_stagnation_normalization(self) -> Optional[Dict[str, float]]:
        if len(self._perf) < 2 or len(self._corr) < 2:
            return None
        p_min, p_max = float(min(self._perf)), float(max(self._perf))
        q_min, q_max = float(min(self._corr)), float(max(self._corr))
        p_span, q_span = p_max - p_min, q_max - q_min
        if not all(math.isfinite(v) for v in (p_min, q_min, p_span, q_span)):
            return None
        if p_span <= self._HV_GAIN_EPS or q_span <= self._HV_GAIN_EPS:
            return None
        return {
            "p_min": p_min,
            "p_span": p_span,
            "q_min": q_min,
            "q_span": q_span,
        }

    @staticmethod
    def _normalize_objective_pair(
        performance: float, quality: float, scale: Dict[str, float],
    ) -> Tuple[float, float]:
        return (
            (float(performance) - scale["p_min"]) / scale["p_span"],
            (float(quality) - scale["q_min"]) / scale["q_span"],
        )

    def _realized_normalized_hv_gain(
        self,
        performance: float,
        quality: float,
        *,
        scale: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        """Marginal realized HV of one landed result in a frozen frame."""
        resolved = dict(scale) if scale is not None else \
            self._derive_stagnation_normalization()
        if resolved is None:
            return None, None
        existing = [
            self._normalize_objective_pair(p, q, resolved)
            for p, q in zip(self._perf, self._corr)
        ]
        candidate = self._normalize_objective_pair(
            float(performance), float(quality), resolved,
        )
        gain = self._hv2([*existing, candidate]) - self._hv2(existing)
        return max(0.0, float(gain)), resolved

    def _record_realized_outcome(
        self,
        performance: float,
        quality: float,
        *,
        eval_id: int,
        forced: bool,
    ) -> Optional[float]:
        """Advance the forced clock after a completed post-init evaluation.

        Only ordinary qLogNEHVI-arbitrated evaluations age the stagnation
        counter.  Any forced delivery closes the current episode and must be
        followed by a fresh run of normal evaluations before another forced
        seat can fire.
        """
        scale = self._stagnation_normalization
        gain, resolved = self._realized_normalized_hv_gain(
            performance, quality, scale=scale,
        )
        self._last_accounted_eval_id = max(
            self._last_accounted_eval_id,
            int(eval_id),
        )
        if forced:
            logger.info(
                "[forced-result t=%s] eval_id=%s "
                "realized_normalized_dhv=%s",
                getattr(self, "_t", "?"), eval_id,
                "unavailable" if gain is None else f"{gain:.9f}",
            )
            self._normal_no_hv_gain = 0
            self._stagnation_normalization = None
            return gain

        if gain is None:
            # A degenerate scale cannot support a meaningful normalized-HV
            # stagnation claim.  Keep normal optimization active until two
            # non-degenerate objective spans exist.
            self._normal_no_hv_gain = 0
            self._stagnation_normalization = None
            return None
        if gain > self._HV_GAIN_EPS:
            self._normal_no_hv_gain = 0
            self._stagnation_normalization = None
        else:
            if self._normal_no_hv_gain == 0:
                self._stagnation_normalization = resolved
            self._normal_no_hv_gain += 1
        logger.info(
            "[realized-hv t=%s] eval_id=%s normalized_dhv=%.9f "
            "normal_no_gain=%d",
            getattr(self, "_t", "?"), eval_id, gain,
            self._normal_no_hv_gain,
        )
        return gain

    def _record_unscored_attempt(
        self,
        *,
        eval_id: int,
        forced: bool,
    ) -> bool:
        """Commit one deterministic attempt that produced no objectives."""
        eval_id = int(eval_id)
        if eval_id <= self._last_accounted_eval_id:
            return False
        self._last_accounted_eval_id = eval_id
        if forced:
            # An invalid/unscored candidate did not perform the safety
            # experiment. Retire that concrete candidate elsewhere, but keep
            # the due episode alive so resume/the next proposal can try a new
            # maximum-gap design without waiting for three unrelated rows.
            logger.info(
                "[forced-result t=%s] eval_id=%s unscored; "
                "preserving due episode",
                getattr(self, "_t", "?"), eval_id,
            )
        return True

    def _forced_due(self, patience: int) -> bool:
        return bool(
            getattr(self, "enable_forced", True)
            and int(patience) > 0
            and self._normal_no_hv_gain >= int(patience)
            and self._stagnation_normalization is not None
        )
