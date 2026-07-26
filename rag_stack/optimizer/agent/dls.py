"""Grouped discrete local search with hierarchy-aware restart pools."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from botorch.optim.optimize import optimize_acqf_discrete_local_search
from botorch.utils.sampling import draw_sobol_samples

from .llm_agent import AgentAdvice
from .util import _pareto_mask, _unique_rows

logger = logging.getLogger("RAG-Stack")


class DLSMixin:
    def _hierarchy_safe_acqf(self, fn):
        """Evaluate an acquisition function on canonical active identities.

        BoTorch's lattice local search mutates every encoded coordinate,
        including children disabled by the current hierarchy selector.  The GP
        is trained on clamped rows, so scoring those raw neighbors creates
        artificial variance along dead knobs and can make a climb collapse to
        a duplicate when its endpoint is finally clamped.  Canonicalize every
        acquisition call, preserving all leading batch/q dimensions.

        All DLS tasks in this optimizer use ``q=1``, so a lightweight callable
        is sufficient.  Non-hierarchical spaces retain the original callable
        and their existing continuous-polish behavior unchanged.
        """
        if not self.ssops.ss.is_hierarchical:
            return fn

        def canonicalized(X: Tensor):
            shape = X.shape
            flat = X.reshape(-1, shape[-1])
            # A selector flip can activate children whose inactive canonical
            # value is 0.5, which is not necessarily a legal Choice level.
            # Snap first, then re-apply inactive canonicalization.
            safe = self.ssops.clamp(
                self.ssops.snap_to_grid(flat)
            ).reshape(shape)
            return fn(safe)

        return canonicalized

    def _hier_pool_sobol(self, n: int, seed_offset: int = 100_000) -> Optional[Tensor]:
        """Pool BREADTH candidates via the hierarchy-aware Ax Sobol (07-07).

        The 06-16 init fix (``_sobol_batch``) never reached this second call
        site: the per-round breadth pool still drew raw Sobol over the FULL
        encoded cube, spending ~1/3 of its quasi-random resolution on gated-off
        child dims and thereby thinning far-field coverage (s45 autopsy: the
        corner the acqf already valued 2.5-5x was not generated for 22 rounds).
        Same generator family as init, but batched through the adapter — ONE
        ``gen(n)`` call per round, no per-trial client overhead. Returns None
        on any failure so the caller falls back to the raw path."""
        try:
            from ax.adapter.registry import Generators
            if self._pool_sobol_ss is None:
                self._pool_sobol_ss = self._get_init_sobol_client().experiment
            m = Generators.SOBOL(
                experiment=self._pool_sobol_ss,
                search_space=self._pool_sobol_ss.search_space,
                seed=int(self.seed + seed_offset + self._t),
            )
            rows: List[Tensor] = []
            for arm in m.gen(n=int(n)).arms:
                cfg = self.ssops.decode_to_nested(dict(arm.parameters))
                if not cfg:
                    continue
                rows.append(
                    self.ssops.snap_to_grid(
                        self._encode_nested_config(cfg).reshape(1, -1)
                    )
                )
            if not rows:
                return None
            # canonicalize here, not only at pool assembly: decode_to_nested
            # reconstructs defaults for gated-off sections, so the encoded row
            # carries live-looking values on INACTIVE dims until clamped.
            return self.ssops.clamp(torch.cat(rows, dim=0)).to(**self.tkwargs)
        except Exception as exc:  # noqa: BLE001 — pool build must never fail
            logger.warning(
                f"[pool t={self._t}] hier sobol failed ({exc}); raw fallback"
            )
            return None

    def _grid_choices(self) -> List[Tensor]:
        """Per-dim normalized level grids (ssops.choice_grid, cf. snap_to_grid)."""
        choices = []
        for name in self.ssops.param_names:
            vals = getattr(self.ssops.ss.parameters[name], "values", None)
            if vals is None:
                raise ValueError("discrete local search requires an all-choice space")
            choices.append(torch.tensor(self.ssops.choice_grid(name), **self.tkwargs))
        return choices

    def _restart_seed_groups(self, advice: Optional[AgentAdvice]) -> "dict[str, Tensor]":
        """Local-search restart seeds organized by provenance:

        ``llm`` (agent centers, ①) / ``incumbent`` (best-q + Pareto spread) /
        ``random`` (Sobol fill). Each source is one official BoTorch
        multi-start DLS call that returns its best ``q=1`` endpoint. The
        configured restart budget belongs in full to the global/random source,
        while Agent and incumbent starts are additive and do not consume that
        budget."""
        X_eval = torch.stack(self._x_rows, dim=0).to(**self.tkwargs)
        groups: "dict[str, Tensor]" = {}
        if (
            getattr(self, "enable_agent", True)
            and advice is not None and advice.centers is not None
        ):
            groups["llm"] = advice.centers.to(**self.tkwargs)
        Y = np.array([[self._perf[i], self._corr[i]] for i in range(len(self._corr))])
        inc = {int(np.argmax(Y[:, 1]))}
        pidx = sorted([i for i in range(len(Y)) if _pareto_mask(Y)[i]],
                      key=lambda i: self._perf[i])
        for k in np.linspace(0, len(pidx) - 1, min(3, len(pidx))).astype(int):
            inc.add(pidx[int(k)])
        groups["incumbent"] = X_eval[sorted(inc)]
        # This is a per-source budget, not a shared total. Subtracting the
        # structured starts here reduced the usual 8 configured global
        # restarts to the two-restart floor, even though the sources run as
        # separate multi-start optimizations.
        n_rand = max(2, self.dls_num_restarts)
        # hierarchy-aware restarts, always (07-07): distinct seed stream so the
        # starts differ from the breadth pool drawn in the same round
        hier = self._hier_pool_sobol(n_rand, seed_offset=300_000)
        if hier is None:
            raw = draw_sobol_samples(bounds=self.standard_bounds, n=n_rand, q=1,
                                     seed=self.seed + 200_000 + self._t).squeeze(-2)
            hier = self.ssops.snap_to_grid(raw)
        groups["random"] = self.ssops.clamp(hier).to(**self.tkwargs)
        return groups

    def _gradient_polish(self, fn, x: Tensor, steps: int = 25, lr: float = 0.05) -> Optional[Tensor]:
        """Non-hierarchical joint refinement on the continuous relaxation.

        Coordinate-scan hill climbing (the discrete local search) can move one
        knob at a time and therefore cannot cross valleys that require several
        NUMERIC knobs to move together (s44 knee autopsy: the winning basin
        differs in ef_search+chunk_size+M+ef_construction+top_k jointly).
        The encoding is already a [0,1]^d relaxation with a snap grid, and the
        acqf is differentiable through the GP posterior — so run a short
        projected gradient ascent from the endpoint, snap every few steps, and
        return the best SNAPPED point that strictly improves on the input
        (None otherwise). Categorical index axes get meaningless gradients;
        the accept-only-if-snapped-improves gate makes that harmless.
        """
        # In a hierarchical all-choice space the relaxed path can cross a
        # selector boundary while newly-active children still carry inactive
        # midpoint values.  Snapping inside the acquisition would cut the
        # gradient; leaving it unsnapped scores nonexistent configurations.
        # The discrete scan is authoritative for this space, so skip only this
        # optional polish layer.
        if self.ssops.ss.is_hierarchical:
            return None
        try:
            base = self.ssops.snap_to_grid(
                self.ssops.clamp(x.detach().reshape(1, -1)))
            with torch.no_grad():
                best_val = float(fn(base.unsqueeze(1)).reshape(-1)[0])
            z = x.detach().reshape(1, -1).clone().requires_grad_(True)
            optz = torch.optim.Adam([z], lr=lr)
            best: Optional[Tensor] = None
            for step in range(steps):
                optz.zero_grad()
                val = fn(z.unsqueeze(1)).reshape(-1)[0]
                (-val).backward()
                optz.step()
                with torch.no_grad():
                    z.clamp_(0.0, 1.0)
                    if (step + 1) % 5 == 0 or step == steps - 1:
                        snapped = self.ssops.snap_to_grid(
                            self.ssops.clamp(z.detach().reshape(1, -1)))
                        sval = float(fn(snapped.unsqueeze(1)).reshape(-1)[0])
                        if sval > best_val + 1e-9:
                            best_val, best = sval, snapped.clone()
            return best
        except Exception as exc:  # noqa: BLE001 — polish is best-effort
            logger.debug(f"gradient polish failed ({exc}); keeping scan endpoint")
            return None

    def _mark_dls_selected(self, source: str) -> None:
        """Optional selection hook for callers that arbitrate a DLS pool.

        Candidate generation cannot know which endpoint the outer pool will
        select.  The existing optimizer log already records
        ``[pool-pick] source=...``; this hook additionally lets a caller attach
        that fact to the round-local structured diagnostics without coupling
        DLS to the arbitration policy.
        """
        diagnostics = getattr(self, "_dls_channel_diagnostics", None)
        if not isinstance(diagnostics, dict):
            return
        channels = diagnostics.get("channels", {})
        if source in channels:
            channels[source]["selected"] = int(
                channels[source].get("selected", 0)
            ) + 1
        # Generation-time diagnostics cannot know the outer-pool winner.  Emit
        # one final row for every DLS channel at the actual arbitration point,
        # including the all-zero case when Sobol/Agent/DOE wins.
        for label, stats in sorted(channels.items()):
            logger.info(
                "[dls-selection t=%s] channel=%s selected=%d winner=%s",
                getattr(self, "_t", "?"), label,
                int(stats.get("selected", 0)), source,
            )

    def _dls_candidates_grouped(
        self, acqf, advice: Optional[AgentAdvice], q: int = 1,
        pool_seeds: Optional[Tensor] = None,
    ) -> "Optional[tuple[Tensor, list[str]]]":
        """Return one provenance-labelled endpoint per DLS source.

        Seeds from the same source are passed together to official BoTorch
        multi-start coordinate DLS. For ``q=1``, BoTorch climbs from every
        start and returns only the highest-acquisition endpoint. Source
        endpoints later enter one common pool for canonical decoded
        deduplication and LogNEHVI arbitration. ``None`` degrades to the
        non-DLS pool.
        """
        self._dls_channel_diagnostics = {
            "round": int(self._t),
            "channels": {},
        }
        if not getattr(self, "enable_dls", True):
            return None

        def _empty_channel_stats() -> Dict[str, Any]:
            return {
                "attempted_starts": 0,
                "evaluated_starts": 0,
                "novelized_starts": 0,
                "filtered_no_novel_start": 0,
                "filtered_duplicate_start": 0,
                "returned_endpoints": 0,
                "filtered_returned_evaluated": 0,
                "filtered_returned_decoded_duplicate": 0,
                "returned_decoded_duplicate_other_channel": 0,
                "terminal_failures": 0,
                "unique_survivors": 0,
                "selected": 0,
                "failure_reasons": [],
            }

        def _merge_channel_stats(label: str, values: Dict[str, Any]) -> None:
            channels = self._dls_channel_diagnostics["channels"]
            target = channels.setdefault(label, _empty_channel_stats())
            for key, value in values.items():
                if key == "failure_reasons":
                    target[key].extend(value)
                elif isinstance(value, (int, float)):
                    target[key] = int(target.get(key, 0)) + int(value)

        try:
            from botorch.acquisition import PosteriorMean
            from botorch.acquisition.objective import ScalarizedPosteriorTransform

            X_eval = torch.stack(self._x_rows, dim=0).to(**self.tkwargs)
            grid = self._grid_choices()

            # ---- task prep (serial, cheap, deterministic) -----------------
            # Each task fully describes one full-grid local search.  Execute
            # tasks serially in this fixed order: GPyTorch acquisition objects
            # share mutable lazy state, so concurrent evaluation is unsafe.
            tasks: "list[tuple[str, Any, Tensor, int, List[Tensor]]]" = []
            canonical_acqf = self._hierarchy_safe_acqf(acqf)
            groups = self._restart_seed_groups(advice)
            for name, X in groups.items():
                if X is None or X.shape[0] == 0:
                    continue
                # Raw Agent proposals already remain in the common pool as
                # exact ``llm`` rows.  ``dls-llm`` is a separate hybrid channel:
                # Agent rows remain unrestricted full-grid starts: DLS may
                # correct a locally wrong proposed delta, while the verbatim
                # ``llm`` rows remain in the outer pool independently. All
                # starts from one source follow BoTorch's official multi-start
                # contract and therefore produce one best q=1 endpoint.
                X0 = self.ssops.clamp(_unique_rows(X))
                tasks.append((
                    f"dls-{name}", canonical_acqf, X0.unsqueeze(1), q,
                    grid,
                ))

            def _mean_acqf(obj_idx: int):
                w = torch.zeros(len(self.objective_names), **self.tkwargs)
                w[obj_idx] = 1.0
                return self._hierarchy_safe_acqf(PosteriorMean(
                    self.model,
                    posterior_transform=ScalarizedPosteriorTransform(weights=w),
                ))

            def _top_seeds(vals) -> Tensor:
                idx = np.argsort(vals)[-3:]
                X0 = _unique_rows(torch.stack(
                    [self._x_rows[int(i)] for i in idx], dim=0
                ).to(**self.tkwargs))
                return self.ssops.clamp(X0)

            # dls-perf (07-05): PERFORMANCE GP mean from the perf-top
            # incumbents — the pool always contains a genuinely perf-directed
            # candidate for the extreme arm / arbitration.
            try:
                tasks.append(("dls-perf", _mean_acqf(self._perf_idx),
                              _top_seeds(self._perf).unsqueeze(1), 1, grid))
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"dls: dls-perf prep failed ({exc}); skipped")
            # dls-quality (07-06): mirror on the QUALITY GP mean from the
            # quality-top incumbents — pushes the quality frontier end without
            # bypassing the HV arbiter.
            try:
                tasks.append(("dls-quality", _mean_acqf(self._qual_idx),
                              _top_seeds(self._corr).unsqueeze(1), 1, grid))
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"dls: dls-quality prep failed ({exc}); skipped")
            # dls-pool (07-08, user-approved reach fix): GLOBAL restarts — the
            # top-acqf UNEVALUATED rows of this round's breadth pool. The
            # missing composition was sample-without-climb (the Sobol pool is
            # scored but never refined) plus climb-without-global-starts
            # (every other seed anchors to evaluated history): a pool sample
            # landing a couple of flips from a high-acqf basin lost arbitration
            # instead of climbing into it (s44 knee autopsy: known-good configs
            # scored 2.56 while the raw pool capped at 2.50).
            if pool_seeds is not None and pool_seeds.shape[0] > 0:
                try:
                    X0p = self.ssops.clamp(_unique_rows(
                        pool_seeds.to(**self.tkwargs)
                    ))
                    tasks.append((
                        "dls-pool", canonical_acqf, X0p.unsqueeze(1), 1,
                        grid,
                    ))
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"dls: dls-pool prep failed ({exc}); skipped")
            # ---- DLS execution -------------------------------------------
            # Seeds that ARE evaluated rows get silently dropped inside
            # botorch (`_filter_invalid(starts, X_avoid)`), and with
            # num_restarts == n_seeds no random top-up is generated — the
            # search then crashes on an empty start set and the whole group
            # is skipped. That killed dls-{incumbent,perf,quality}
            # on EVERY round (s45 production log: 20/20 rounds,
            # zero picks from those groups). One-flip each evaluated seed to
            # its nearest novel grid neighbor: the climb starts one step out,
            # X_avoid keeps deduplicating the climb itself.
            def _novelize(
                X0: Tensor, task_grid: List[Tensor],
            ) -> Tuple[Tensor, Dict[str, Any]]:
                stats = _empty_channel_stats()
                flat_rows = X0.reshape(X0.shape[0], -1)
                stats["attempted_starts"] = int(flat_rows.shape[0])
                d = flat_rows.shape[-1]
                out: List[Tensor] = []
                for ri, row in enumerate(flat_rows):
                    r = self.ssops.clamp(self.ssops.snap_to_grid(
                        row.reshape(1, -1)
                    )).reshape(-1)
                    if self.ssops.is_duplicate_decoded(
                        r.reshape(1, -1), X_eval,
                    ):
                        stats["evaluated_starts"] += 1
                        novel = None
                        for off in range(d):
                            j = (ri + off) % d
                            vals = task_grid[j].to(r)
                            alt = vals[(vals - r[j]).abs() > 1e-12]
                            if alt.numel() == 0:
                                continue
                            orig = r[j].clone()
                            r[j] = alt[(alt - orig).abs().argmin()]
                            cand_r = self.ssops.clamp(self.ssops.snap_to_grid(
                                r.reshape(1, -1)
                            )).reshape(-1)
                            if not self.ssops.is_duplicate_decoded(
                                cand_r.reshape(1, -1), X_eval,
                            ):
                                novel = cand_r
                                break
                            r[j] = orig  # flip was inert (dead dim) — try next
                        if novel is None:
                            stats["filtered_no_novel_start"] += 1
                            continue
                        r = novel
                        stats["novelized_starts"] += 1
                    if out and self.ssops.is_duplicate_decoded(
                        r.reshape(1, -1), torch.stack(out, dim=0),
                    ):
                        stats["filtered_duplicate_start"] += 1
                        continue
                    out.append(r)
                if not out:
                    return X0.new_empty((0, *X0.shape[1:])), stats
                return torch.stack(out, dim=0).reshape(
                    len(out), *X0.shape[1:]
                ), stats

            def _run_task(
                fn, X0, q_task: int, task_grid: List[Tensor],
            ):
                X0, stats = _novelize(X0, task_grid)
                if X0.shape[0] == 0:
                    return [], stats, None
                try:
                    cand, _ = optimize_acqf_discrete_local_search(
                        acq_function=fn, discrete_choices=task_grid,
                        q=q_task, num_restarts=X0.shape[0],
                        batch_initial_conditions=X0,
                        X_avoid=X_eval, max_batch_size=512,
                    )
                except Exception as exc:  # noqa: BLE001 — isolate task failure
                    return [], stats, exc
                canonical = self.ssops.clamp(self.ssops.snap_to_grid(
                    cand.reshape(q_task, -1)
                ))
                stats["returned_endpoints"] = int(canonical.shape[0])
                rows: List[Tensor] = []
                for row in canonical:
                    row = row.reshape(1, -1)
                    if self.ssops.is_duplicate_decoded(row, X_eval):
                        stats["filtered_returned_evaluated"] += 1
                        continue
                    if rows and self.ssops.is_duplicate_decoded(
                        row, torch.cat(rows, dim=0),
                    ):
                        stats["filtered_returned_decoded_duplicate"] += 1
                        continue
                    rows.append(row)
                return rows, stats, None

            results: "dict[int, list[Tensor]]" = {}
            task_stats: "dict[int, Dict[str, Any]]" = {}
            for i, (lbl, fn, X0, qt, tg) in enumerate(tasks):
                rows, stats, exc = _run_task(fn, X0, qt, tg)
                if exc is None:
                    results[i], task_stats[i] = rows, stats
                else:
                    stats["terminal_failures"] = 1
                    stats["failure_reasons"].append(type(exc).__name__)
                    task_stats[i] = stats
                    logger.warning(
                        f"dls: task '{lbl}' failed ({exc}); skipping task"
                    )

            endpoints, prov, _src_task = [], [], []
            for i, (lbl, _fn, _X0, _qt, _tg) in enumerate(tasks):
                _merge_channel_stats(lbl, task_stats.get(i, _empty_channel_stats()))
                for row in results.get(i, []):
                    endpoints.append(row)
                    prov.append(lbl)
                    _src_task.append(i)

            # Official coordinate DLS is the complete search method for the
            # paper's hierarchical all-Choice space.  The historical Adam
            # relaxation remains available only for non-hierarchical spaces;
            # applying it here would cross selector branches, while adding a
            # custom two-coordinate enumerator would be a different method.
            _n_gradient_polished = 0
            for k in range(len(endpoints)):
                task = tasks[_src_task[k]]
                _lbl, fn, _X0, _qt, task_grid = task
                if self.ssops.ss.is_hierarchical:
                    continue
                better = self._gradient_polish(fn, endpoints[k])
                if better is None:
                    continue
                try:
                    rescanned, _rescan_stats, _rescan_exc = _run_task(
                        fn, better.unsqueeze(1), 1, task_grid,
                    )
                    if _rescan_exc is None and rescanned:
                        with torch.no_grad():
                            if float(fn(rescanned[0].unsqueeze(1)).reshape(-1)[0]) > \
                               float(fn(better.unsqueeze(1)).reshape(-1)[0]):
                                better = rescanned[0]
                except Exception:  # noqa: BLE001 — rescan is best-effort
                    pass
                endpoints[k] = better.reshape(1, -1)
                _n_gradient_polished += 1
            if self.ssops.ss.is_hierarchical and endpoints:
                logger.info(
                    f"[dls-polish t={self._t}] mode=hierarchical "
                    f"endpoints={len(endpoints)} official_coordinate_dls=complete "
                    "gradient=not-applicable"
                )
            elif endpoints:
                logger.info(
                    f"[dls-polish t={self._t}] mode=nonhierarchical-gradient "
                    f"endpoints={len(endpoints)} improved={_n_gradient_polished}"
                )

            # Preserve the established grouped-return contract: every channel
            # hands its endpoint + provenance to the outer pool, whose single
            # canonical dedup decides which source survives.  Audit decoded
            # convergence here without pre-empting that outer ordering.
            seen: List[Tensor] = []
            for row, lbl in zip(endpoints, prov):
                channel = self._dls_channel_diagnostics["channels"][lbl]
                if seen and self.ssops.is_duplicate_decoded(
                    row, torch.cat(seen, dim=0),
                ):
                    channel["returned_decoded_duplicate_other_channel"] += 1
                else:
                    channel["unique_survivors"] += 1
                    seen.append(row)

            for lbl, stats in sorted(
                self._dls_channel_diagnostics["channels"].items()
            ):
                reasons = ",".join(stats["failure_reasons"]) or "-"
                logger.info(
                    f"[dls-channel t={self._t}] channel={lbl} "
                    f"attempted_starts={stats['attempted_starts']} "
                    f"evaluated_starts={stats['evaluated_starts']} "
                    f"novelized_starts={stats['novelized_starts']} "
                    f"filtered_no_novel_start={stats['filtered_no_novel_start']} "
                    f"filtered_duplicate_start={stats['filtered_duplicate_start']} "
                    f"returned_endpoints={stats['returned_endpoints']} "
                    f"filtered_returned_evaluated="
                    f"{stats['filtered_returned_evaluated']} "
                    f"filtered_returned_decoded_duplicate="
                    f"{stats['filtered_returned_decoded_duplicate']} "
                    f"returned_decoded_duplicate_other_channel="
                    f"{stats['returned_decoded_duplicate_other_channel']} "
                    f"terminal_failures={stats['terminal_failures']} "
                    f"unique_survivors={stats['unique_survivors']} "
                    "selected=pending "
                    f"failure_reasons={reasons} selection=outer-pool"
                )
            if not endpoints:
                return None
            return torch.cat(endpoints, dim=0), prov
        except Exception as exc:
            logger.warning(f"dls: grouped search failed ({exc}); pool-only this iter")
            return None

    def _min_knob_dist(self, x, llm_x) -> str:
        """Decoded-knob hamming distance from ``x`` to the NEAREST LLM center
        this round ('-' when no centers). 0 = verbatim LLM proposal; 1-2 = a
        DLS-refined LLM seed's fingerprint; large = unrelated to the LLM.
        Instrumentation only (07-04): quantifies the indirect seed channel."""
        try:
            if llm_x is None or not llm_x.shape[0]:
                return "-"
            a = self.ssops.decode_flat(x.reshape(-1))
            best = None
            for row in llm_x:
                b = self.ssops.decode_flat(row.reshape(-1))
                d = sum(1 for k in set(a) | set(b) if a.get(k) != b.get(k))
                best = d if best is None else min(best, d)
            return str(best)
        except Exception:  # noqa: BLE001
            return "?"
