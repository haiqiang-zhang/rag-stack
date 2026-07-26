"""Budget-aware greedy baseline — per-stage coordinate search with a Pareto
beam, in two modes, over the hierarchy-true stage decomposition.

Why this exists
---------------
AutoRAG-style node-wise greedy is the canonical cheap baseline for RAG
pipeline search: walk the pipeline stage by stage, try each stage's options,
commit, move on. This implementation is its faithful adaptation to OUR
setting, with exactly three forced deviations (each dictated by the setting,
not by a wish to strengthen the baseline):

  1. **Budgeted subsampling instead of per-node exhaustion.** AutoRAG grids
     every node exhaustively; our GT budget (``n_iterations``, ~50) is far
     below a single stage's option count, so each stage receives a FAIR SHARE
     of the budget instead: proportional allocation with weight
     ``w_s = log2(1 + #options_s)`` (information-proportional — deciding a
     49-way stage is ~log2(49)/log2(4) ≈ 2.8× harder than a 4-way one, not
     12×), integerized by largest-remainder (no residue, no positional
     privilege). It is implemented incrementally — at each stage visit the
     share table is recomputed over the not-yet-visited stages and the
     remaining budget — which is mathematically identical to the up-front
     split when every stage absorbs its share, and additionally handles
     stages that only ACTIVATE mid-pass (a beam member switching index
     family) plus rollover of unspendable shares.
  2. **End-to-end bi-objective evaluation instead of intermediate metrics.**
     AutoRAG scores middle nodes with intermediate metrics (retrieval recall
     for rerankers, …); we have no per-stage ground truth — every probe is a
     full pipeline evaluated on (performance, quality).
  3. **A Pareto beam instead of scalarized argmax.** The final metric is the
     archive's Pareto front (hypervolume), so committing each stage to a
     single scalar winner would cripple the baseline by metric definition.
     The beam carries up to ``beam_width`` non-dominated configs, capped by
     ROLE: both chain ends (span anchors) plus the interior points with the
     largest exclusive hypervolume contribution — the knee, where refinement
     buys the most of the final metric (default width 3 = fast end + accurate
     end + knee); ``beam_width=1`` degrades to the classic single-incumbent
     chain.

Modes
-----
* ``search_mode="forward"`` — ONE pass per branch in causal/topological stage
  order. The AutoRAG-faithful rung of the baseline ladder.
* ``search_mode="lookback"`` — after the forward pass, alternate backward /
  forward refinement passes (re-optimising each stage against the *current*
  downstream instead of defaults) with a geometric budget split: each pass
  receives ``pass_budget_fraction`` of the branch's remaining budget. Stops
  on a stale pass (no new archive entries), branch-budget exhaustion, or
  ``max_passes``.

Fork semantics
--------------
A METHOD-level cross-stage gater (flat name ``pipeline__<method>__<knob>``
with dependents reaching other stages — today ``rag_dataflow``; a swept
``pipeline__mode`` matches the same rule) is a pipeline SHAPE switch: its
branches activate different stage sets, so beam members from different
branches are not comparable coordinate systems. Such a parameter is treated
as an outer FORK, the analog of AutoRAG's "one run per pipeline YAML": each
value gets an independent greedy chain with its own beam and stage order,
budgets split across branches proportionally to the branches' log-weight
sums (largest-remainder), both branch bases evaluated first (anytime
guarantee), unspent branch budget rolling to the next branch. NODE-level
cross-stage gaters (the vectordb family selector) are NOT forks — they are
component choices inside one shape, competing inside the beam exactly like a
reranker choice, with the gated family stages ordered after the selector by
the topological sort.

Hierarchy iron rules (all enforced here, each pinned by a test)
---------------------------------------------------------------
The single source of hierarchy truth is Ax ``ChoiceParameter.dependents``
(compiled by the search-space builder); activity questions go to
``ssops.active_params``, which matches what the evaluator actually runs.
Every emitted candidate passes through :meth:`_complete` (defaults + context
+ local assignment → active filter), so configs never carry inactive dims
and the archive key is the sorted active-item tuple. Stage-local enumeration
(:meth:`_local_assignments`) recurses the stage's dependents forest —
never cartesian-product-then-filter — and is computed PER CONTEXT at visit
time (the old init-time precompute froze non-default branches: react's
``max_iter`` was unreachable). Stage order is a topological sort of the
cross-stage gating edges, so a gater is always decided before what it gates.

Budget accounting
-----------------
``remaining = n_iterations - len(archive)``: every suggest evaluates both
objectives (one GT slot), and ``warm_start`` fills the archive, so resume
self-accounts. The controller additionally charges INVALID evaluations
against its own budget without them ever reaching ``add_objectives`` —
greedy's ``remaining`` is therefore an upper bound; the controller's hard
loop bound is the enforcer.

Interface
---------
Implements :class:`~rag_stack.optimizer.base.BaseOptimizer`. No surrogate,
no acquisition, no cost-model side information (the CM probe was removed
deliberately: the baseline is pure black-box, like AutoRAG). Driven one
config per ``suggest``; the planner generator emits batches internally and
resumes only after the whole batch is archived.
"""

import itertools
import logging
import math
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from ax.core.parameter import ChoiceParameter, ParameterType
from ax.core.search_space import SearchSpace

from rag_stack.search_space.ax_search_space_ops import AxSearchSpaceOps
from .base import BaseOptimizer

logger = logging.getLogger("RAG-Stack")

SEP = "__"

# Stage-local enumeration guard: hierarchy-valid assignments per stage are
# expected in the tens; anything past this is a mis-specified space.
_ENUM_CAP = 4096
# A shape switch with more values than this is not a plausible dataflow/mode
# knob — treat it as an ordinary stage instead of splitting the budget.
_FORK_MAX_VALUES = 8


def _largest_remainder(
    budget: int,
    weights: Dict[str, float],
    floors: Dict[str, int],
    caps: Dict[str, int],
) -> Dict[str, int]:
    """Fair integer split of ``budget`` proportional to ``weights``.

    Floors are granted first (weight-descending when they alone exceed the
    budget); the rest is split by largest-remainder rounding; shares are
    capped at ``caps`` with overflow redistributed among uncapped keys. The
    result sums to ``min(budget, sum(caps))`` — no residue, no positional
    privilege.
    """
    out = {k: 0 for k in caps}
    keys = [k for k in caps if caps[k] > 0 and weights.get(k, 0.0) > 0.0]
    if budget <= 0 or not keys:
        return out
    by_weight = sorted(keys, key=lambda k: (-weights[k], k))
    granted = {k: min(floors.get(k, 0), caps[k]) for k in keys}
    if sum(granted.values()) >= budget:
        left = budget
        for k in by_weight:
            take = min(granted[k], left)
            out[k] = take
            left -= take
            if left <= 0:
                break
        return out
    for k in keys:
        out[k] = granted[k]
    remaining = budget - sum(granted.values())
    while remaining > 0:
        pool = [k for k in keys if out[k] < caps[k]]
        if not pool:
            break
        total_w = sum(weights[k] for k in pool)
        shares = {k: remaining * weights[k] / total_w for k in pool}
        placed = 0
        for k in pool:
            base = min(int(math.floor(shares[k])), caps[k] - out[k])
            out[k] += base
            placed += base
        crumbs = remaining - placed
        if crumbs > 0:
            frac_order = sorted(
                (k for k in pool if out[k] < caps[k]),
                key=lambda k: (-(shares[k] - math.floor(shares[k])), k),
            )
            for k in frac_order:
                if crumbs <= 0:
                    break
                out[k] += 1
                crumbs -= 1
        if crumbs == remaining:  # nothing placeable (all capped)
            break
        remaining = crumbs
    return out


class GreedyParetoOptimizer(BaseOptimizer):
    """Two-mode, budget-aware Pareto-beam greedy over pipeline stages.

    Args:
        config_space: Ax ``SearchSpace`` (hierarchy via ``dependents``).
        objectives: list of ``{objective_name, cost, ref_point}`` dicts; both
            objectives are MAXIMISED.
        seed: governs ONLY the within-stratum neighbor shuffle (the
            MO-ParamILS random-order rule); quotas, strata, beam, fork
            budgets and stage order are deterministic. Report 2-3 seeds.
        search_mode: ``"forward"`` (single pass per branch) or ``"lookback"``
            (geometric refinement passes). See module docstring.
        n_iterations: TOTAL GT budget (base configs included). Injected by
            the controller / benchmark CLI; ``None`` falls back to the legacy
            per-visit ``per_stage_cap`` behavior (direct construction / tests
            only) with a warning.
        beam_width: BREEDING slots per branch, filled by role — both chain
            ends first (span anchors), then interior points by exclusive
            hypervolume contribution (the knee). Default 3 = fast end +
            accurate end + knee; 2 = ends only; 1 → plain chain.
        stage_weight: ``"log"`` (default, information-proportional) or
            ``"uniform"`` (ablation) — the per-stage allocation weight.
        pass_budget_fraction: γ — lookback only; each pass receives γ of the
            branch's remaining budget.
        max_passes: lookback pass cap.
        per_stage_cap: LEGACY per-visit quota, used only when
            ``n_iterations`` is None.
        ssops: RAG-aware search-space adapter; generic fallback when omitted.

    """

    def __init__(
        self,
        config_space: SearchSpace,
        objectives: List[dict],
        seed: int = 42,
        search_mode: str = "lookback",
        n_iterations: Optional[int] = None,
        beam_width: int = 3,
        stage_weight: str = "log",
        pass_budget_fraction: float = 0.6,
        max_passes: int = 8,
        per_stage_cap: int = 4,
        stage_stop_after: int = 0,
        sibling_seeding: bool = False,
        ssops: Optional[AxSearchSpaceOps] = None,
    ):
        if search_mode not in ("forward", "lookback"):
            raise ValueError(
                f"search_mode must be 'forward' or 'lookback', got {search_mode!r}"
            )
        if stage_weight not in ("log", "uniform"):
            raise ValueError(
                f"stage_weight must be 'log' or 'uniform', got {stage_weight!r}"
            )
        if not (0.0 < float(pass_budget_fraction) <= 1.0):
            raise ValueError(
                "pass_budget_fraction must be in (0, 1], got "
                f"{pass_budget_fraction!r}"
            )
        self.search_space = config_space
        self.seed = seed
        # Seeded RNG for the ONE stochastic element (MO-ParamILS neighbor
        # shuffle in _visit_stage); everything else stays deterministic.
        self._rng = random.Random(seed)
        self.search_mode = search_mode
        self.stage_weight = stage_weight
        self.pass_budget_fraction = float(pass_budget_fraction)
        self.max_passes = max(1, int(max_passes))
        self.beam_width = max(1, int(beam_width))
        self.per_stage_cap = max(1, int(per_stage_cap))
        # Stage early-stop: after this many consecutive landed evals that fail
        # to enter the branch beam, leave the stage (its unspent quota re-splits
        # over the later stages via the caller's incremental _visit_quota).
        # 0 = off — the AutoRAG-faithful default, so standalone greedy baselines
        # are untouched; the greedy_then_* wrappers turn it on.
        self.stage_stop_after = max(0, int(stage_stop_after))
        # M4 (07-07, surrogate-validated 4W/1L +12%; default OFF =
        # baseline-identical): probe the SIBLING branch's proven values first
        # within each stage quota. The greedy_then_* wrappers expose it as
        # greedy_sibling_seeding; standalone greedy baselines never set it.
        self.sibling_seeding = bool(sibling_seeding)
        # Checkpoint-resume replay state (see warm_start / suggest).
        self._replay_queue: Dict[tuple, Tuple[dict, int, bool]] = {}
        self._replay_fed = 0
        self._replay_miss_streak = 0
        # Budget seats returned by free rows (M3 fingerprint-inherited quality).
        self._free_evals = 0
        self._n_iterations: Optional[int] = (
            int(n_iterations) if n_iterations is not None else None
        )
        if self._n_iterations is None:
            logger.warning(
                "GreedyParetoOptimizer: no n_iterations supplied — falling back "
                "to the LEGACY per-visit per_stage_cap=%d behavior. Live runs "
                "get the budget injected by the controller/benchmark CLI.",
                self.per_stage_cap,
            )

        self.objective_names: List[str] = [o["objective_name"] for o in objectives]
        self.objective_costs: Dict[str, float] = {
            o["objective_name"]: float(o.get("cost", 1.0)) for o in objectives
        }
        # Greedy seeds itself from the branch bases, not a random init batch.
        self.n_both_init: int = 1

        self._ss: AxSearchSpaceOps = (
            ssops if ssops is not None else AxSearchSpaceOps(config_space)
        )
        self._params = config_space.parameters
        self._all_names: List[str] = list(self._params.keys())
        self._defaults: Dict[str, object] = {
            n: self._default(n) for n in self._all_names
        }

        # --- fork + stages (hierarchy-true decomposition) ---
        self._fork: Optional[str] = self._detect_fork()
        stage_names = [n for n in self._all_names if n != self._fork]
        self._stage_params: Dict[str, List[str]] = {}
        for n in stage_names:
            self._stage_params.setdefault(self._stage_key(n), []).append(n)
        self._stages: List[str] = self._build_stage_order()
        # In-stage gaters: hierarchical params with at least one child in the
        # SAME stage (drives stratum grouping; cross-stage children are
        # handled by _complete + the child's own stage).
        self._stage_gaters: Dict[str, Set[str]] = {}
        for stage, sp in self._stage_params.items():
            sp_set = set(sp)
            gaters = set()
            for p in sp:
                param = self._params[p]
                if isinstance(param, ChoiceParameter) and param.is_hierarchical:
                    if any(c in sp_set for kids in param.dependents.values()
                           for c in kids):
                        gaters.add(p)
            self._stage_gaters[stage] = gaters

        # --- state ---
        self._archive: Dict[tuple, dict] = {}
        self._attempted: set = set()
        self._beams: Dict[int, List[dict]] = {}
        # Per (branch, stage): stage-local combos evaluated on ANY base —
        # the cross-visit "value coverage before value repetition" ledger
        # (in-memory only; a resumed session re-covers at worst, never skips).
        self._visited_locals: Dict[int, Dict[str, Set[tuple]]] = {}
        self._branch_ctxs: List[dict] = (
            [{self._fork: v} for v in self._params[self._fork].values]
            if self._fork is not None else [{}]
        )
        self._gen = self._planner()
        self._gen_started: bool = False
        self._buf: List[dict] = []
        self._pending: List[Tuple[dict, dict]] = []

        logger.info(
            "GreedyParetoOptimizer: mode=%s budget=%s fork=%s branches=%d "
            "stages=%d %s | beam=%d weight=%s gamma=%.2f max_passes=%d",
            self.search_mode, self._n_iterations, self._fork,
            len(self._branch_ctxs), len(self._stages), self._stages,
            self.beam_width, self.stage_weight, self.pass_budget_fraction,
            self.max_passes,
        )

    # ==================================================================
    # Search-space primitives.
    # ==================================================================

    def _default(self, name: str):
        p = self._params[name]
        if isinstance(p, ChoiceParameter):
            # Optional nodes start OFF → a minimal, fast base pipeline; the
            # node is toggled on explicitly in its own stage.
            if name.endswith(f"{SEP}enabled") and "false" in {str(v) for v in p.values}:
                return "false"
            return p.values[0]
        return p.lower  # RangeParameter

    def _candidate_values(self, name: str) -> List:
        p = self._params[name]
        if isinstance(p, ChoiceParameter):
            return list(p.values)
        # RangeParameter → small evenly-spaced grid (this domain is all-choice;
        # handled generically for robustness).
        lo, hi = p.lower, p.upper
        n = 5
        if getattr(p, "parameter_type", None) == ParameterType.INT:
            return sorted({int(round(lo + (hi - lo) * i / (n - 1))) for i in range(n)})
        return [lo + (hi - lo) * i / (n - 1) for i in range(n)]

    def _complete(self, partial: Dict[str, object]) -> dict:
        """Re-resolve a partial assignment to a full, hierarchy-valid ACTIVE
        config: defaults everywhere, committed values on top, inactive params
        dropped, newly-active params defaulted (iron rules 2 & 6)."""
        full = dict(self._defaults)
        for k, v in partial.items():
            if k in full:
                full[k] = v
        active = self._ss.active_params(full)
        return {n: full[n] for n in active}

    @staticmethod
    def _key(flat: Dict[str, object]) -> tuple:
        return tuple(sorted(flat.items()))

    # ------------------------------------------------------------------
    # Fork + stage decomposition.
    # ------------------------------------------------------------------

    @staticmethod
    def _stage_key(name: str) -> str:
        parts = name.split(SEP)
        head = parts[0]
        if head == "pipeline":
            if len(parts) > 1 and parts[1] == "mode":
                return f"pipeline{SEP}mode"
            return SEP.join(parts[:3])      # pipeline__<method>__<node>
        if head == "system":
            # The whole deployment is ONE coordinate-descent unit: the
            # ``deployment_preset`` categorical and the ``batch_size_*`` knobs
            # are joint coordinates of a single deployment (a preset's
            # throughput is meaningless without its batch), NOT gated
            # alternatives. Grouping them into one ``system`` stage lets the
            # beam search the deployment combos jointly instead of doing blind
            # coordinate descent (best preset at a default batch, then best
            # batch at that preset — never the coupled optimum).
            return head
        if head in ("corpus", "vectordb"):
            # vectordb splits per index family (gated ALTERNATIVES → separate
            # stages); corpus has a single ``chunker`` sub-namespace.
            return SEP.join(parts[:2])
        return head

    def _detect_fork(self) -> Optional[str]:
        """METHOD-level cross-stage gater = pipeline SHAPE switch (fork).

        Flat name ``pipeline__mode`` or ``pipeline__<method>__<knob>`` (no
        node segment) whose dependents reach OTHER stages. Node-level
        cross-stage gaters (the vectordb family selector) do NOT match — they
        are component choices competing inside the beam.
        """
        candidates: List[str] = []
        for name, p in self._params.items():
            if not (isinstance(p, ChoiceParameter) and p.is_hierarchical):
                continue
            parts = name.split(SEP)
            if parts[0] != "pipeline" or len(parts) > 3:
                continue
            my_stage = self._stage_key(name)
            if any(self._stage_key(c) != my_stage
                   for kids in p.dependents.values() for c in kids):
                candidates.append(name)
        if not candidates:
            return None
        if len(candidates) > 1:
            raise ValueError(
                "multiple method-level shape switches are unsupported: "
                f"{candidates}"
            )
        fork = candidates[0]
        if len(self._params[fork].values) > _FORK_MAX_VALUES:
            logger.warning(
                "fork %s has %d values (> %d) — treating as an ordinary stage",
                fork, len(self._params[fork].values), _FORK_MAX_VALUES,
            )
            return None
        return fork

    def _build_stage_order(self) -> List[str]:
        """Kahn topological sort over cross-stage gating edges; ties broken by
        (causal priority class, -gating out-degree, first appearance). A cycle
        is a mis-built space → fail fast (iron rule 5)."""
        stages = list(self._stage_params.keys())
        rank = {s: i for i, s in enumerate(stages)}
        edges: Dict[str, Set[str]] = {s: set() for s in stages}
        indeg: Dict[str, int] = {s: 0 for s in stages}
        for name, p in self._params.items():
            if name == self._fork:
                continue
            if not (isinstance(p, ChoiceParameter) and p.is_hierarchical):
                continue
            src = self._stage_key(name)
            if src not in edges:
                continue
            for kids in p.dependents.values():
                for c in kids:
                    if c == self._fork:
                        continue
                    dst = self._stage_key(c)
                    if dst != src and dst in edges and dst not in edges[src]:
                        edges[src].add(dst)
                        indeg[dst] += 1

        def priority(s: str) -> Tuple:
            if s.startswith("corpus"):
                cls = 0
            elif s.startswith("vectordb"):
                cls = 1
            elif s == f"pipeline{SEP}mode":
                cls = 2
            elif s.startswith("pipeline"):
                cls = 3
            elif s.startswith("system"):
                cls = 5
            else:
                cls = 4
            return (cls, -len(edges[s]), rank[s])

        ready = sorted((s for s in stages if indeg[s] == 0), key=priority)
        order: List[str] = []
        while ready:
            s = ready.pop(0)
            order.append(s)
            for dst in edges[s]:
                indeg[dst] -= 1
                if indeg[dst] == 0:
                    ready.append(dst)
            ready.sort(key=priority)
        if len(order) != len(stages):
            raise ValueError(
                "cyclic cross-stage gating between stages: "
                f"{sorted(set(stages) - set(order))}"
            )
        return order

    # ------------------------------------------------------------------
    # Context-live stage enumeration (iron rules 3 & 4).
    # ------------------------------------------------------------------

    def _local_assignments(self, stage: str, ctx: Dict[str, object]) -> List[dict]:
        """Distinct hierarchy-valid assignments of ``stage``'s own params ON
        TOP OF ``ctx`` — recursing the stage's dependents forest (a gater's
        children enter the frontier only under the value that activates
        them), never cartesian-product-then-filter. Empty when the stage is
        inactive in this context (external gates off)."""
        sp = self._stage_params[stage]
        sp_set = set(sp)
        base = dict(self._defaults)
        base.update(ctx)
        active0 = self._ss.active_params(base)
        kids: Dict[str, Dict[object, List[str]]] = {}
        for p in self._stage_gaters[stage]:
            param = self._params[p]
            kids[p] = {
                v: [c for c in children if c in sp_set]
                for v, children in param.dependents.items()
            }
        gated_in_stage = {c for m in kids.values() for cs in m.values() for c in cs}
        roots = [p for p in sp if p not in gated_in_stage and p in active0]
        if not roots:
            return []
        out: List[dict] = []

        def rec(assign: dict, frontier: List[str]) -> None:
            if len(out) >= _ENUM_CAP:
                return
            gater = next((p for p in frontier if p in kids), None)
            if gater is None:
                if not frontier:
                    out.append(dict(assign))
                    return
                value_lists = [self._candidate_values(p) for p in frontier]
                for combo in itertools.product(*value_lists):
                    if len(out) >= _ENUM_CAP:
                        return
                    d = dict(assign)
                    d.update(zip(frontier, combo))
                    out.append(d)
                return
            rest = [p for p in frontier if p != gater]
            for v in self._candidate_values(gater):
                rec({**assign, gater: v}, rest + kids[gater].get(v, []))

        rec({}, roots)
        if len(out) >= _ENUM_CAP:
            logger.warning(
                "stage %s enumeration hit cap %d — space likely mis-specified",
                stage, _ENUM_CAP,
            )
        return out

    def _stage_candidates(self, stage: str, beam: List[dict]) -> Dict[tuple, dict]:
        """Fresh (not archived / not attempted) full configs from applying
        every local assignment of ``stage`` onto every beam member."""
        out: Dict[tuple, dict] = {}
        for member in beam:
            for local in self._local_assignments(stage, member):
                cand = self._complete({**member, **local})
                k = self._key(cand)
                if k in self._archive or k in self._attempted or k in out:
                    continue
                out[k] = cand
        return out

    # ------------------------------------------------------------------
    # Pareto helpers (both objectives MAXIMISED).
    # ------------------------------------------------------------------

    def _nondominated(self, items: List[dict]) -> List[dict]:
        keep: List[dict] = []
        for i, a in enumerate(items):
            dominated = False
            for j, b in enumerate(items):
                if j == i:
                    continue
                ge = all(b[o] >= a[o] for o in self.objective_names)
                gt = any(b[o] > a[o] for o in self.objective_names)
                if ge and gt:
                    dominated = True
                    break
                if all(b[o] == a[o] for o in self.objective_names) and j < i:
                    dominated = True
                    break
            if not dominated:
                keep.append(a)
        return keep

    def _beam_cap(self, items: List[dict], k: int) -> List[dict]:
        """Cap a non-dominated front to ``k`` BREEDING slots, by role.

        NSGA-II's crowding distance degenerates at small k (its ∞-boundary
        rule hands every slot to the two chain ends, so the KNEE — the
        region with the largest exclusive hypervolume contribution, i.e.
        where refinement buys the most of the final metric — never breeds).
        Role-based capping instead:

          * k >= 2: both chain ENDS are always kept (they anchor the front's
            span — losing an end kills that whole trade-off lineage);
          * remaining slots go to the interior points with the largest
            EXCLUSIVE HYPERVOLUME contribution (per-axis normalized staircase
            rectangles — the literal "how much front area dies if this point
            is dropped", the same geometry as the final score);
          * k == 1: the single largest contributor (no arbitrary ∞ ties).
        """
        if len(items) <= k:
            return list(items)
        o1, o2 = self.objective_names[0], self.objective_names[1]
        chain = sorted(items, key=lambda it: (-it[o1], it[o2]))
        n = len(chain)
        s1 = (chain[0][o1] - chain[-1][o1]) or 1.0
        s2 = (chain[-1][o2] - chain[0][o2]) or 1.0

        def contribution(i: int) -> float:
            right = chain[i + 1][o1] if i + 1 < n else chain[-1][o1]
            left = chain[i - 1][o2] if i - 1 >= 0 else chain[0][o2]
            return max(0.0, (chain[i][o1] - right) / s1) * \
                   max(0.0, (chain[i][o2] - left) / s2)

        if k == 1:
            best = max(range(n), key=lambda i: (contribution(i), -i))
            return [chain[best]]
        keep_idx = {0, n - 1}
        interior = sorted(
            range(1, n - 1), key=lambda i: (-contribution(i), i),
        )
        for i in interior[: max(0, k - 2)]:
            keep_idx.add(i)
        return [chain[i] for i in sorted(keep_idx)]

    def _branch_of(self, flat: dict) -> int:
        if self._fork is None:
            return 0
        v = flat.get(self._fork)
        for i, ctx in enumerate(self._branch_ctxs):
            if str(ctx[self._fork]) == str(v):
                return i
        return 0

    def _recompute_beam(self, branch: int, candidate_flats: List[dict]) -> None:
        keys = {self._key(c) for c in candidate_flats}
        keys.update(self._key(b) for b in self._beams.get(branch, []))
        # A beam's membership is a set, but its order becomes observable when
        # the next stage chooses the lowest-index member as the base for a
        # local assignment.  Iterating ``keys`` directly made that base depend
        # on the process-wide Python hash seed, so a fresh process could not
        # replay an otherwise identical checkpoint trajectory.  Canonical key
        # order removes that accidental entropy; the configured Greedy seed
        # remains the only source of proposal-order randomness.
        pool = [
            self._archive[k]
            for k in sorted(keys, key=repr)
            if k in self._archive
        ]
        if not pool:
            return
        front = self._beam_cap(self._nondominated(pool), self.beam_width)
        if front:
            self._beams[branch] = [it["flat"] for it in front]

    # ------------------------------------------------------------------
    # Budget allocation (proportional share, incremental form).
    # ------------------------------------------------------------------

    def _w_of(self, n_candidates: int) -> float:
        if n_candidates <= 0:
            return 0.0
        if self.stage_weight == "uniform":
            return 1.0
        return math.log2(1 + n_candidates)

    def _remaining(self) -> float:
        if self._n_iterations is None:
            return math.inf
        # free rows (M3 fingerprint-inherited) sit in the archive but returned
        # their budget seat — the planner may spend that seat elsewhere.
        return max(0, self._n_iterations - len(self._archive) + self._free_evals)

    def _visit_quota(
        self, stage: str, unvisited: Sequence[str], branch: int, budget_left: float,
    ) -> int:
        """This stage's fair share of the pass budget, recomputed over the
        not-yet-visited stages with LIVE pool counts (mid-pass activation +
        rollover; equals the up-front split when every stage absorbs its
        share — see module docstring). Every share is spendable by
        construction: a visit stratified-subsamples the stage's full pool,
        so the pool count is both the difficulty weight and the true cap."""
        if budget_left <= 0:
            return 0
        beam = self._beams[branch]
        pool = {s: len(self._stage_candidates(s, beam)) for s in unvisited}
        if pool.get(stage, 0) <= 0:
            return 0
        if self._n_iterations is None:  # legacy
            return min(self.per_stage_cap, pool[stage])
        budget = int(min(budget_left, self._remaining()))
        table = _largest_remainder(
            budget,
            weights={s: self._w_of(n) for s, n in pool.items()},
            floors={s: min(2, pool[s]) for s in unvisited},
            caps=pool,
        )
        return table.get(stage, 0)

    # ==================================================================
    # Planner (generator) — emits batches; resumes once they are archived.
    # ==================================================================

    def _emit(self, flats: List[dict]):
        fresh: List[dict] = []
        for f in flats:
            kk = self._key(f)
            if kk in self._archive or kk in self._attempted:
                continue
            self._attempted.add(kk)
            fresh.append(f)
        if fresh:
            yield fresh
        return len(fresh)

    def _visit_stage(self, stage: str, quota: int, branch: int):
        """Spend ``quota`` on one stage: a STRATIFIED, low-discrepancy
        subsample of the stage's hierarchy-valid pool — the budgeted form of
        AutoRAG's per-node grid, with no per-member scan state (a beam
        reshuffle can never reset value coverage).

        Rounds until the quota is spent or the pool runs dry:

          * enumerate the FRESH pool (local assignment × beam member) under
            the current beam;
          * stratify by in-stage gater signature (component/off branches) so
            a 1-combo branch ("off") is never drowned by a 50-combo sibling
            — floor 1 each, remainder ∝ log2(size), largest-remainder;
          * within a stratum, UNTRIED locals first (a local ever evaluated on
            any base in this branch — including the beam members' own
            incumbents — is refinement, not coverage; "value coverage before
            value repetition" ACCUMULATES across visits and is immune to beam
            churn), each group in SEEDED-RANDOM order — MO-ParamILS's
            neighbor rule (neighbors of each archive member are evaluated in
            random order; Blot et al., LION 2016), the standard for
            budget-limited configuration search. Any fixed order encodes a
            bias we paid for twice: even-stride re-anchored on the re-sorted
            pool's ends and deterministically shadowed the middle (14B/3B
            never sampled); endpoint-first front-loaded the most expensive
            extremes (6/50 evals on TP2 14B in a perf-heavy HV geometry).
            The seed governs ONLY this shuffle; quotas, strata, beam, fork
            budgets and stage order stay deterministic. Runs are seed-
            reproducible end-to-end; a RESUME redraws from the seed against
            the restored archive (fresh shuffle, same distribution);
          * recompute the beam after every landing — a beam change can
            activate new strata (a member switching index family), which the
            next round picks up.

        Returns evaluations spent — equal to ``quota`` while the pool
        suffices."""
        spent = 0
        misses = 0  # consecutive landed evals that failed to enter the branch beam
        visited = self._visited_locals.setdefault(branch, {}).setdefault(
            stage, set(),
        )
        while spent < quota and self._remaining() > 0:
            beam = self._beams[branch]
            gaters = self._stage_gaters[stage]
            # Beam members' own stage-locals are archived configs — coverage
            # they already provide, never a first-visit target.
            for member in beam:
                visited.add(tuple(sorted(
                    (p, str(member[p])) for p in self._stage_params[stage]
                    if p in member
                )))
            strata: Dict[tuple, List[Tuple[tuple, int, dict]]] = {}
            seen: Set[tuple] = set()
            for mi, member in enumerate(beam):
                for local in self._local_assignments(stage, member):
                    cand = self._complete({**member, **local})
                    k = self._key(cand)
                    if k in self._archive or k in self._attempted or k in seen:
                        continue
                    seen.add(k)
                    lk = tuple(sorted((p, str(v)) for p, v in local.items()))
                    sig = tuple(sorted(
                        (p, str(v)) for p, v in local.items() if p in gaters
                    ))
                    strata.setdefault(sig, []).append((lk, mi, cand))
            if not strata:
                return spent
            shares = _largest_remainder(
                quota - spent,
                weights={s: self._w_of(len(v)) for s, v in strata.items()},
                floors={s: 1 for s in strata},
                caps={s: len(v) for s, v in strata.items()},
            )
            batch: List[Tuple[tuple, dict]] = []
            for sig in sorted(strata):
                share = shares.get(sig, 0)
                if share <= 0:
                    continue
                entries = strata[sig]
                # one entry per local (lowest member index), locals in
                # seeded-random order, untried group before tried group;
                # repeats-on-other-bases only after every local has a probe
                by_local: Dict[tuple, Tuple[tuple, dict]] = {}
                rest: List[Tuple[tuple, dict]] = []
                for lk, _, c in sorted(entries, key=lambda t: (t[0], t[1])):
                    if lk in by_local:
                        rest.append((lk, c))
                    else:
                        by_local[lk] = (lk, c)
                untried = [lk for lk in by_local if lk not in visited]
                tried = [lk for lk in by_local if lk in visited]
                self._rng.shuffle(untried)
                self._rng.shuffle(tried)
                self._rng.shuffle(rest)
                if self.sibling_seeding and len(self._branch_ctxs) > 1:
                    # M4 (07-07, experimental; default OFF): within the quota,
                    # probe the SIBLING branch's proven values first — s45's
                    # react chain re-learned top_k/generator-size lessons the
                    # sequential chain had already paid for. Stable partition:
                    # sibling-endorsed untried locals ahead of the rest (the
                    # shuffle above still randomizes order WITHIN each group,
                    # so the RNG stream stays identical when the flag is off).
                    endorsed = self._sibling_endorsed_locals(untried, branch)
                    if endorsed:
                        untried = (
                            [lk for lk in untried if lk in endorsed]
                            + [lk for lk in untried if lk not in endorsed]
                        )
                first = [by_local[lk] for lk in untried + tried]
                batch.extend((first + rest)[:share])
            if not batch:
                return spent
            for lk, cand in batch[: quota - spent]:
                if self._remaining() <= 0:
                    break
                got = yield from self._emit([cand])
                spent += got
                if got:
                    visited.add(lk)
                    self._recompute_beam(branch, [cand])
                    # stage_stop counts LANDED evals only: an INVALID proposal
                    # (controller excluded it — never archived) says nothing
                    # about the stage's yield and must not burn a miss.
                    if self.stage_stop_after > 0 and self._key(cand) in self._archive:
                        in_beam = self._key(cand) in {
                            self._key(m) for m in self._beams[branch]
                        }
                        misses = 0 if in_beam else misses + 1
                        if misses >= self.stage_stop_after:
                            # Greedy's yield is front-loaded (s44: each stage's
                            # FIRST probes carry ~99% of its HV; repeats carry
                            # ~0): consecutive landed evals that leave the beam
                            # unchanged mean this stage's remaining quota pays
                            # better downstream.
                            return spent
        return spent

    def _planner(self):
        legacy = self._n_iterations is None
        ctxs = self._branch_ctxs

        # 0. Resume: seed each branch beam from its archive front.
        if self._archive:
            per_branch: Dict[int, List[dict]] = {}
            for it in self._archive.values():
                per_branch.setdefault(self._branch_of(it["flat"]), []).append(it)
            for bi, items in per_branch.items():
                front = self._beam_cap(self._nondominated(items), self.beam_width)
                if front:
                    self._beams[bi] = [it["flat"] for it in front]

        # 1. Branch bases first (anytime guarantee: every shape lands one
        #    archived point before any chain begins).
        for bi, ctx in enumerate(ctxs):
            base = self._complete(ctx)
            yield from self._emit([base])
            if not self._beams.get(bi):
                self._beams[bi] = [base]
            self._recompute_beam(bi, [base])

        # 2. Branch budgets ∝ branch log-weight sums (largest remainder).
        branch_w: Dict[int, float] = {}
        for bi in range(len(ctxs)):
            branch_w[bi] = sum(
                self._w_of(len(self._stage_candidates(s, self._beams[bi])))
                for s in self._stages
            )
        branch_order = sorted(range(len(ctxs)), key=lambda b: (-branch_w[b], b))
        if legacy:
            budgets: Dict[int, float] = {bi: math.inf for bi in branch_order}
        else:
            total = int(self._remaining())
            split = _largest_remainder(
                total,
                weights={str(b): branch_w[b] for b in branch_order},
                floors={str(b): 0 for b in branch_order},
                caps={str(b): total for b in branch_order},
            )
            budgets = {b: split.get(str(b), 0) for b in branch_order}

        # 3. Per-branch chains; unspent budget rolls to the next branch.
        carry = 0.0
        for bi in branch_order:
            branch_budget = budgets[bi] + carry
            spent_branch = 0
            passes, stale, direction = 0, 0, 1
            while True:
                if self._remaining() <= 0:
                    break
                if not legacy and spent_branch >= branch_budget:
                    break
                if self.search_mode == "forward" and passes >= 1:
                    break
                if self.search_mode == "lookback" and (
                    passes >= self.max_passes or stale >= 1
                ):
                    break
                if legacy:
                    pass_budget: float = math.inf
                elif self.search_mode == "forward":
                    pass_budget = branch_budget - spent_branch
                else:
                    left = branch_budget - spent_branch
                    pass_budget = max(1, int(round(self.pass_budget_fraction * left)))
                order = (
                    self._stages if direction > 0 else list(reversed(self._stages))
                )
                archive_before = len(self._archive)
                pass_spent = 0
                for i, stage in enumerate(order):
                    if pass_spent >= pass_budget or self._remaining() <= 0:
                        break
                    quota = self._visit_quota(
                        stage, order[i:], bi, pass_budget - pass_spent,
                    )
                    if quota <= 0:
                        continue
                    got = yield from self._visit_stage(stage, quota, bi)
                    pass_spent += got
                spent_branch += pass_spent
                passes += 1
                direction *= -1
                stale = 0 if len(self._archive) > archive_before else stale + 1
                if legacy and pass_spent == 0:
                    break  # legacy termination: nothing new anywhere
            if not legacy:
                carry = max(0.0, branch_budget - spent_branch)
                if carry:
                    logger.info(
                        "greedy branch %d left %s unspendable evals — rolling on",
                        bi, int(carry),
                    )
        logger.info(
            "Greedy planner finished: archive=%d remaining=%s",
            len(self._archive),
            self._remaining() if not legacy else "inf(legacy)",
        )

    def _pump(self) -> bool:
        try:
            if self._gen_started:
                batch = self._gen.send(None)
            else:
                batch = next(self._gen)
                self._gen_started = True
        except StopIteration:
            return False
        self._buf = list(batch)
        return True

    # ==================================================================
    # BaseOptimizer interface.
    # ==================================================================

    def suggest(
        self, n_suggestions: int = 1
    ) -> Tuple[Optional[List[dict]], Optional[List[str]]]:
        # RESUME REPLAY: the planner generator is deterministic given the seed
        # and the sequence of observed outcomes, so a checkpoint is "resumed" by
        # letting the planner re-propose its own history while we feed each
        # proposal its ARCHIVED outcome (no re-evaluation, no re-measurement
        # noise). When a proposal has no archived outcome we have reached the
        # frontier of history: that proposal is the genuine next candidate.
        while True:
            nested_out, names = self._suggest_impl(n_suggestions)
            if nested_out is None:
                if self._replay_queue:
                    self._flush_replay("planner exhausted during replay")
                return nested_out, names
            if not self._replay_queue:
                return nested_out, names
            live: List[dict] = []
            for nested, flat in list(self._pending):
                # Match on the NESTED-roundtrip key, not the raw proposal flat:
                # a proposal flat can carry inert rider values (e.g. the
                # unchosen vectordb family's knobs inherited from a beam
                # member) that don't survive nested serialization — the
                # archived config canonicalizes without them.
                canon = self._flatten_config(nested)
                rkey = self._key(canon) if canon is not None else self._key(flat)
                hit = self._replay_queue.pop(rkey, None)
                if hit is None:
                    live.append(nested)
                    continue
                objs, eid, free = hit
                self._replay_fed += 1
                self._replay_miss_streak = 0
                self.add_objectives(
                    [nested],
                    {n: [objs[n]] for n in self.objective_names},
                    [eid],
                    free_mask=[free],
                )
            if live:
                # Unarchived proposal mid-replay: either the frontier of
                # history, or an eval that originally went INVALID (never
                # landed) — both are correct to hand out live. A STREAK of
                # misses while outcomes remain queued means the planner has
                # diverged from the recorded run (e.g. the code changed
                # between runs) — stop replaying and fall back to direct
                # archive injection so no recorded data is lost.
                self._replay_miss_streak += 1
                if self._replay_queue and self._replay_miss_streak >= 3:
                    self._flush_replay(
                        f"{self._replay_miss_streak} consecutive proposals "
                        "not found in the archived history"
                    )
                return live, names
            if not self._replay_queue:
                logger.info(
                    "Greedy.warm_start replay complete: %d archived outcomes "
                    "re-fed through the planner", self._replay_fed,
                )

    def _suggest_impl(
        self, n_suggestions: int = 1
    ) -> Tuple[Optional[List[dict]], Optional[List[str]]]:
        n = max(1, int(n_suggestions))
        picked: List[dict] = []
        while len(picked) < n:
            if not self._buf and not self._pump():
                break
            if self._buf:
                picked.append(self._buf.pop(0))
            else:
                break
        if not picked:
            return None, None
        self._pending = []
        nested_out: List[dict] = []
        for flat in picked:
            nested = self._ss.decode_to_nested(flat)
            self._pending.append((nested, flat))
            nested_out.append(nested)
        return nested_out, list(self.objective_names)

    @staticmethod
    def _scalar(v: Union[float, dict]) -> float:
        return float(v["combined"] if isinstance(v, dict) else v)

    def add_objectives(
        self,
        x: List[dict],
        obj_dict: Dict[str, List[Union[float, dict]]],
        eval_ids: Optional[List[int]] = None,
        free_mask: Optional[List[bool]] = None,
    ) -> None:
        consumed: set = set()
        for i, nested in enumerate(x):
            flat = next(
                (f for j, (pn, f) in enumerate(self._pending)
                 if pn is nested and j not in consumed),
                None,
            )
            if flat is None:
                logger.warning(
                    "Greedy.add_objectives: result %d not in pending; skipping", i
                )
                continue
            consumed.add(next(
                j for j, (pn, _f) in enumerate(self._pending) if pn is nested
            ))
            # free rows (M3 fingerprint-inherited / penalty) enter the archive
            # as real observations but give their budget seat back via
            # _remaining. Counted per MATCHED row only — an unmatched free row
            # must not return a seat it never took.
            if free_mask and i < len(free_mask) and free_mask[i]:
                self._free_evals += 1
            key = self._key(flat)
            item = {
                "key": key, "flat": flat, "nested": nested,
                "eval_id": (eval_ids[i] if eval_ids else -1),
            }
            for name in self.objective_names:
                item[name] = self._scalar(obj_dict[name][i])
            self._archive[key] = item

    def get_pareto(self) -> Tuple[List[dict], List[Dict[str, float]], List[int]]:
        items = list(self._archive.values())
        if not items:
            return [], [], []
        front = self._nondominated(items)
        configs = [it["nested"] for it in front]
        objs = [{n: it[n] for n in self.objective_names} for it in front]
        ids = [it["eval_id"] for it in front]
        return configs, objs, ids

    def warm_start(self, evaluations: List[Dict]) -> None:
        """Queue archived outcomes for PLANNER REPLAY instead of injecting them
        into the archive directly.

        The old direct-injection resume seeded the beam but restarted the
        planner from scratch with only the LEFTOVER budget — a mid-run resume
        therefore ran a different (compressed) schedule, not a continuation.
        Replay reconstructs the planner's exact position: ``suggest`` walks the
        deterministic proposal sequence from the top, feeding each proposal its
        recorded outcome (identical inputs ⇒ identical stage/quota/RNG state),
        and hands out the first proposal history doesn't cover. Costs planner
        arithmetic only — no re-evaluation.

        Until the first post-resume ``suggest`` drains the queue, the archive
        (and ``get_pareto``) is intentionally empty. If the planner diverges
        from the recorded history (code changed between runs), the queue is
        flushed via direct injection — the old behavior — so no data is lost.
        """
        for ev in evaluations:
            objs = ev.get("objectives", {}) or {}
            if any(n not in objs for n in self.objective_names):
                continue  # need every objective to place the point on the front
            flat = self._flatten_config(ev.get("config", {}) or {})
            if flat is None:
                continue
            self._replay_queue[self._key(flat)] = (
                {n: objs[n] for n in self.objective_names},
                ev.get("eval_id", -1),
                bool(ev.get("free")),
            )
        logger.info(
            "Greedy.warm_start: %d archived outcomes queued for planner replay",
            len(self._replay_queue),
        )

    def _sibling_endorsed_locals(self, locals_keys, branch: int) -> set:
        """M4 helper: the subset of local-assignment keys (tuples of
        ``(param, str(value))`` pairs) every one of whose assignments matches
        some SIBLING branch's beam member — values the other chain already
        paid evals to validate."""
        siblings = [
            member
            for other, beam in self._beams.items() if other != branch
            for member in beam
        ]
        if not siblings:
            return set()
        endorsed: set = set()
        for lk in locals_keys:
            for member in siblings:
                if all(p in member and str(member[p]) == v for p, v in lk):
                    endorsed.add(lk)
                    break
        return endorsed

    def _flatten_config(self, nested: dict) -> Optional[Dict[str, object]]:
        """Nested config → active-params flat dict (the archive/key identity)."""
        try:
            return self._ss.active_flat_config(
                nested, require_all_active=True,
            )
        except Exception as e:  # noqa: BLE001 — resume best-effort
            logger.warning(
                "Greedy.warm_start: could not flatten config (%s); skipping", e
            )
            return None

    def _flush_replay(self, reason: str) -> None:
        """Divergence fallback: inject the un-replayed outcomes straight into
        the archive (beam-seeding resume, the old behavior) so their data still
        shapes the front and the budget accounting, then stop replaying."""
        if not self._replay_queue:
            return
        per_branch: Dict[int, List[Dict[str, object]]] = {}
        n_flushed = 0
        for key, (objs, eid, free) in self._replay_queue.items():
            flat = dict(key)  # _key = tuple(sorted(flat.items())) — invertible
            item = {
                "key": key, "flat": flat,
                "nested": self._ss.decode_to_nested(flat),
                "eval_id": eid,
            }
            for n in self.objective_names:
                item[n] = self._scalar(objs[n])
            self._archive[key] = item
            self._attempted.add(key)
            if free:
                self._free_evals += 1
            per_branch.setdefault(self._branch_of(flat), []).append(flat)
            n_flushed += 1
        self._replay_queue.clear()
        for branch, flats in per_branch.items():
            self._recompute_beam(branch, flats)
        logger.warning(
            "Greedy.warm_start replay flushed %d outcomes into the archive (%s)",
            n_flushed, reason,
        )
