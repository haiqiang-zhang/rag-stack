"""Hierarchy-aware Sobol probes followed by data-informed Agent probes."""

import logging
from typing import List, Optional

import torch

from botorch.utils.sampling import draw_sobol_samples

from .llm_agent import _extract_json

logger = logging.getLogger("RAG-Stack")


class InitProbesMixin:
    def _get_init_sobol_client(self):
        """Lazy Ax client whose ONLY node is a hierarchy-aware ``Generators.SOBOL``
        — the SAME init generator ``ax_qnehvi_mixedgp`` uses. Per arm it samples
        ONLY the ACTIVE params (gated-off children get no value), so init coverage
        is spent entirely on the meaningful space."""
        if self._init_sobol_client is None:
            from ax.generation_strategy.generation_node import GenerationStep
            from ax.generation_strategy.generation_strategy import GenerationStrategy
            from ax.service.ax_client import AxClient, ObjectiveProperties
            from ax.adapter.registry import Generators
            from rag_stack.optimizer.botorch.ax_qnehvi import _search_space_to_ax_parameters
            gs = GenerationStrategy(
                name="agent_init_sobol",
                steps=[GenerationStep(generator=Generators.SOBOL, num_trials=-1)],
            )
            client = AxClient(random_seed=self.seed, generation_strategy=gs,
                              verbose_logging=False)
            client.create_experiment(
                name="agent_init_sobol",
                parameters=_search_space_to_ax_parameters(self.search_space),
                objectives={
                    name: ObjectiveProperties(
                        minimize=False, threshold=float(self.ref_point[i].item())
                    )
                    for i, name in enumerate(self.objective_names)
                },
            )
            self._init_sobol_client = client
        return self._init_sobol_client

    def _canonical_init_config(
        self, config: dict,
    ) -> "tuple[dict, torch.Tensor]":
        """Return one init config in the optimizer's canonical identity.

        Init candidates can arrive as Ax arms, decoded Sobol rows, or LLM
        proposals.  Route all three through the same snap + inactive-dimension
        clamp before comparing or serving them so dead hierarchical knobs can
        never make two executable-equivalent configs look distinct.
        """
        x = self.ssops.clamp(self.ssops.snap_to_grid(
            self._encode_nested_config(config).reshape(1, -1)
        ))
        return self.ssops.decode_to_config(x.reshape(-1)), x

    def _sobol_batch(self, n: int) -> List[dict]:
        """Draw ``n`` DISTINCT init configs via Ax's HIERARCHY-AWARE Sobol.

        The agent's old init used a raw ``draw_sobol_samples`` over the FULL
        encoded space, then decoded — but ~1/3 of the dims are conditional
        children (pq M/nbits, gated-off module knobs) that are INACTIVE for a
        given arm, so a third of the quasi-random resolution was spent on dims
        that get dropped at decode (diluting active-space coverage vs the
        Ax-native Sobol that ``ax_qnehvi_mixedgp`` inits with). Here we pull arms
        from the SAME Ax Sobol generator: each arm carries only its ACTIVE params,
        so coverage matches Ax exactly. Dedup against already-evaluated configs
        (resume safety); fall back to raw Sobol if Ax ever errors."""
        target = max(1, int(n))
        out: List[dict] = []
        chosen = (
            torch.stack(self._x_rows, dim=0).to(**self.tkwargs) if self._x_rows else None
        )
        def _accept(config: dict) -> bool:
            nonlocal chosen
            canonical, x = self._canonical_init_config(config)
            if chosen is not None and self.ssops.is_duplicate_decoded(x, chosen):
                return False
            out.append(canonical)
            chosen = x if chosen is None else torch.cat([chosen, x], dim=0)
            return True

        try:
            client = self._get_init_sobol_client()
            attempts = 0
            while len(out) < target and attempts < target * 20:
                params, ti = client.get_next_trial()
                client.abandon_trial(trial_index=ti)  # Sobol needs no completion
                attempts += 1
                cfg = self.ssops.decode_to_nested(params)
                _accept(cfg)
        except Exception as exc:  # noqa: BLE001 — never let init stall the run
            logger.warning(
                f"AgentGuidedQNEHVI: Ax hierarchy-aware Sobol init failed ({exc}); "
                f"falling back to raw Sobol for the remaining {target - len(out)}."
            )
        # raw-Sobol pad (Ax under-delivered or errored)
        raw_attempts = 0
        max_raw_attempts = max(100, target * 100)
        while len(out) < target and raw_attempts < max_raw_attempts:
            raw = draw_sobol_samples(
                bounds=self.standard_bounds, n=1, q=1, seed=self.seed + self._sobol_calls
            ).squeeze(-2)
            self._sobol_calls += 1
            raw_attempts += 1
            x = self.ssops.clamp(self.ssops.snap_to_grid(raw))
            _accept(self.ssops.decode_to_config(x.reshape(-1)))
        if len(out) < target:
            logger.warning(
                "AgentGuidedQNEHVI: init Sobol exhausted novel decoded configs "
                "(%d/%d delivered)",
                len(out), target,
            )
        return out[:target]

    @property
    def init_chunks(self) -> List[int]:
        """Remaining init batches for the controller's chunked Phase 1.

        Sobol-first init (07-08): the init budget is served as
        ``[hier-Sobol probes, LLM-designed probes]`` with an evaluate +
        feed-back barrier between the two, so the LLM designs its probes FROM
        the Sobol results. Computed from the current history so a mid-init
        resume re-enters exactly where it stopped (the previous behavior —
        skipping straight to the BO phase — silently truncated init).
        """
        done = len(self._configs)
        chunks: List[int] = []
        if done < self.init_sobol_k:
            chunks.append(self.init_sobol_k - done)
        done = max(done, self.init_sobol_k)
        if done < self.n_both_init:
            chunks.append(self.n_both_init - done)
        return chunks

    def _structured_init_batch(self, n: int) -> List[dict]:
        """Init proposals: hier-Sobol probes first, LLM-designed probes after.

        Phase A (first ``init_sobol_k`` seats): hierarchy-aware Sobol — the
        deterministic, structure-covering baseline no LLM mood can starve.
        Phase B (the rest): the LLM designs EXACTLY the remaining probes,
        with phase A's observed (config, performance, quality) results in its
        prompt (07-08, user directive — the informed LLM channel hits 46.7%
        frontier mid-run; blind init probes are where it failed).
        """
        out: List[dict] = []
        n = max(1, int(n))
        chosen = (
            torch.stack(self._x_rows, dim=0).to(**self.tkwargs)
            if self._x_rows else None
        )

        def _append_unique(config: dict) -> bool:
            nonlocal chosen
            try:
                canonical, x = self._canonical_init_config(config)
            except Exception as exc:  # noqa: BLE001 — malformed LLM row is skipped
                logger.warning(
                    "[structured-init] discarded unencodable probe (%s)", exc,
                )
                return False
            if chosen is not None and self.ssops.is_duplicate_decoded(x, chosen):
                return False
            out.append(canonical)
            chosen = x if chosen is None else torch.cat([chosen, x], dim=0)
            return True

        def _fill_sobol_unique(target_size: int) -> None:
            # Keep the existing (and subclass-overridable) hierarchy-aware
            # Sobol path. The serving-side identity check adds this pending
            # batch to its exclusions without changing the generator policy.
            rounds = 0
            max_rounds = max(20, target_size * 20)
            while len(out) < target_size and rounds < max_rounds:
                sampled = self._sobol_batch(target_size - len(out))
                rounds += 1
                if not sampled:
                    break
                for config in sampled:
                    _append_unique(config)
            if len(out) < target_size:
                logger.warning(
                    "[structured-init] Sobol could only fill %d/%d novel probes",
                    len(out), target_size,
                )

        # Phase A: serve hier-Sobol until init_sobol_k configs are in history.
        # Gated on len(_configs) (not a cursor): an INVALID Sobol probe is
        # replaced by the next fresh draw, never re-served.
        sobol_target = min(n, max(0, self.init_sobol_k - len(self._configs)))
        _fill_sobol_unique(sobol_target)
        if len(out) >= n:
            return out[:n]
        if not hasattr(self, "_init_plan"):
            # LLM-DESIGNED PROBES (07-05, user directive; 07-08: now built
            # AFTER phase A with its results). The plan is generated ONE-SHOT;
            # an empty/failed plan falls back to the Sobol fill below.
            _remaining = self.n_both_init - (len(self._configs) + len(out))
            llm_plan = self._llm_probe_plan(_remaining)
            if llm_plan:
                self._init_plan = llm_plan
                logger.info(
                    f"[structured-init] {len(llm_plan)} LLM-designed probes queued "
                    f"(informed by {len(self._configs)} evaluated init results)"
                )
        if not hasattr(self, "_init_plan"):
            self._init_plan = []
        # cursor advances per SERVING, not per successful eval (07-05: an
        # INVALID probe doesn't enter _configs, so indexing the plan by
        # len(_configs) re-served the same failed probe forever — 45 burned
        # iterations of "skipped: known invalid config").
        if not hasattr(self, "_init_cursor"):
            self._init_cursor = 0
        while len(out) < n and self._init_cursor < len(self._init_plan):
            idx = self._init_cursor
            self._init_cursor += 1
            _append_unique(self._init_plan[idx])
        if len(out) < n:
            # LLM under-delivery includes malformed rows and decoded duplicates.
            # Fill the remaining seats with the existing hierarchy-aware Sobol,
            # excluding both evaluated history and this pending init batch.
            _fill_sobol_unique(n)
        return out[:n]

    def _llm_probe_plan(self, cap: Optional[int] = None) -> List[dict]:
        """One-shot LLM probe design for the init phase (07-05, user
        directive). The LLM gets the SEARCH SPACE and the base config plus an
        explicit exploration brief; it returns the probe configs directly.
        Empty list on any failure → rule-based fallback."""
        try:
            # 07-08: the LLM owns the REMAINING init budget after the
            # hier-Sobol phase; the count is EXACT so same-seed runs stop
            # diverging on how many probes the LLM felt like returning.
            if cap is None:
                cap = self.n_both_init - min(len(self._configs), self.init_sobol_k)
            cap = int(cap)
            if cap < 2:
                return []
            x0 = self.ssops.snap_to_grid(
                draw_sobol_samples(bounds=self.standard_bounds, n=1, q=1,
                                   seed=self.seed + 777).squeeze(-2))
            flat0 = self.ssops.decode_flat(x0.reshape(-1))
            la = self.agent
            space = la._space_card(self.ssops)
            dataflow_semantics = la._dataflow_semantics_card(self.ssops)
            dataflow_semantics_block = (
                dataflow_semantics + "\n\n" if dataflow_semantics else ""
            )
            f2s, _, _ = la._aliases(self.ssops)
            base_line = la._fmt_cfg(flat0, self.ssops.param_names, self.ssops)
            # OBSERVED RESULTS (07-08, user directive): phase A's hier-Sobol
            # probes are evaluated before this call — hand the LLM their
            # configs + objectives so it designs from data, not from priors.
            observed = ""
            if self._configs:
                lines = []
                for i in range(len(self._configs)):
                    try:
                        cfg_line = la._fmt_cfg(
                            self.ssops.decode_flat(self._x_rows[i].reshape(-1)),
                            self.ssops.param_names, self.ssops,
                        )
                    except Exception:  # noqa: BLE001 — formatting must not kill init
                        cfg_line = str(self._configs[i])
                    lines.append(
                        f"  probe#{i}: performance={self._perf[i]:.4g} "
                        f"quality={self._corr[i]:.4g}\n    {cfg_line}"
                    )
                observed = (
                    "OBSERVED RESULTS — hierarchy-aware Sobol probes already "
                    "evaluated (performance = higher is faster, quality = higher "
                    "is better; a 0.0 on both marks an INFEASIBLE config):\n"
                    + "\n".join(lines) + "\n\n"
                )
            system = (
                "You are designing the STRUCTURED INITIALIZATION probes for a "
                "multi-objective (quality vs performance) RAG-pipeline optimizer. "
                + ("A first batch of space-filling probes has been evaluated; "
                   "their results are in the user message. Design the probes that "
                   "cover what those results leave dark."
                   if self._configs else
                   "No evaluations exist yet. Your probes are evaluated verbatim and "
                   "become the surrogate model's entire initial world-view.")
            )
            user = (
                "SEARCH SPACE (knob: allowed values):\n" + space + "\n\n"
                + dataflow_semantics_block
                + "BASE CONFIG (deltas apply on top of this; omit unchanged knobs):\n  "
                + base_line + "\n\n"
                + observed +
                f"Design EXACTLY {cap} probes that structurally cover the regions the "
                "optimizer must know about from birth:\n"
                "  1. the PERFORMANCE extreme — the cheapest end of every cost-driving "
                "knob (smallest generator, minimal retrieval context, optional stages "
                "off) for EACH structural pipeline alternative (e.g. each dataflow), "
                "since the fastest structure is not known a priori;\n"
                "  2. the QUALITY extreme — the richest configuration the space allows;\n"
                "  3. one probe per remaining untouched STRUCTURAL alternative (selector "
                "values not covered above), so no branch of the space is dark.\n"
                "PURITY IS MANDATORY, hedging wastes the probe: a performance-extreme "
                "probe must move EVERY cost-driving knob to its cheapest value and must "
                "NOT enable any enrichment module or enlarge any context knob — mixing "
                "(e.g. a minimal pipeline with a huge chunk size or a reranker enabled) "
                "certifies neither region and teaches the surrogate nothing crisp. "
                "Internally label each probe with the single region it certifies and "
                "verify the set covers: every structural alternative's performance "
                "extreme, the quality extreme, and each untouched structural branch. "
                "Respond with ONLY "
                '{"candidates": [{"<knob>": <value>, ...}, ...]} using knob names and '
                "values exactly as in the SEARCH SPACE."
            )
            client = la._get_client()
            import anyio
            import functools as _ft
            reply = anyio.run(_ft.partial(
                client.chat,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,
            ))
            parsed = _extract_json(reply or "")
            if not parsed or not isinstance(parsed.get("candidates"), list):
                return []
            out: List[dict] = []
            for prop in parsed["candidates"][:cap]:
                x = la._encode_proposal(prop, self.ssops, flat0)
                if x is not None:
                    x = self.ssops.clamp(self.ssops.snap_to_grid(x))
                    out.append(self.ssops.decode_to_config(x.reshape(-1)))
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[structured-init] LLM probe design failed ({exc}); rule-based fallback")
            return []
