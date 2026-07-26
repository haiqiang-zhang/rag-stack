"""Pre-flight validation for RAG optimization configs.

Validates config structure and constraints before optimization starts.
Collects all errors so the user sees everything at once.

Usage:
    from rag_stack.config_validator import ConfigValidator
    errors = ConfigValidator(config).validate()
    if errors:
        raise ValueError(f"Config validation failed:\\n" + "\\n".join(f"  - {e}" for e in errors))
"""

import logging
from typing import List

logger = logging.getLogger("RAG-Stack")

# system_design_space keys that are not optimizer-facing design dimensions.
# Some are derived from placement, while vLLM memory util is measured-runtime
# launch policy, so listing them is silently ignored. Mirrors
# rag_search_space._DERIVED_KNOBS / performance_context._DERIVED_SYSTEM_KNOBS.
_DERIVED_MEASURED_KNOBS = frozenset({
    "vllm_gpu_memory_utilization", "vllm_max_num_batched_tokens",
    "tensor_parallel_vllm",
    "pipeline_parallel_vllm", "serving_mode",
})


def _is_power_of_two(n) -> bool:
    try:
        n = int(n)
    except (TypeError, ValueError):
        return False
    return n > 0 and (n & (n - 1)) == 0


def _flatten_values(v) -> list:
    """Flatten a scalar / list / selector-dict to a list of values.

    ``{ordered: [...]}`` sweep specs flatten to their VALUES (declaration-
    ordered choice — see ``_make_param``). Any other non-range dict is the
    dict-form vectordb family selector (``vectordb: {<family>: {<query
    knobs>}}``) — the referenced values are its KEYS. ``{range: [...]}``
    sweep specs pass through unchanged.
    """
    if isinstance(v, dict) and "ordered" in v:
        ov = v["ordered"]
        return list(ov) if isinstance(ov, list) else [ov]
    if isinstance(v, dict) and "range" not in v:
        return list(v.keys())
    if isinstance(v, list):
        return v
    return [v]


def _pipeline_blocks(config: dict) -> list:
    """``algo_search_space.pipeline`` → internal method-block list (lazy import
    to avoid a heavy/circular import at module load)."""
    from rag_stack.search_space.rag_search_space import pipeline_method_blocks
    return pipeline_method_blocks(config)


def _vdb_blocks(config: dict) -> list:
    """``algo_search_space.vectordb`` block list, tolerantly (the validator
    REPORTS the old top-level location instead of raising like
    ``algo_vectordb_blocks`` does)."""
    return ((config.get("algo_search_space") or {}).get("vectordb")) or []


class ConfigValidator:
    """Validates a RAG-Stack optimization config dict.

    Independent of Controller — can be used from apps or CLI.
    """

    def __init__(self, config: dict, *, check_model_cache: bool = True):
        from rag_stack.pipeline_contract import normalize_retrieval_aliases
        self.config = normalize_retrieval_aliases(config)
        self.errors: List[str] = []
        # When False, skip _check_hf_model_cache (which may snapshot_download
        # missing models over the network). Cost-model-only callers that never
        # execute the models (e.g. the historical-surrogate benchmark) set this
        # False to avoid the network round-trip / hang; live runs keep it True.
        self._check_model_cache_enabled = check_model_cache
        # GT backend selector — defaults to the static evaluator. The
        # FlashRAG backend has its own metric vocabulary, no prompt_maker
        # requirement, and no RAGO power-of-two ``top_k`` constraint, so
        # we skip the static-specific checks under that backend.
        glob = self.config.get("global") or {}
        self.gt_backend: str = str(glob.get("eval_backend", "static_gt")).lower()
        # Performance source: "cost_model" (default — RAGO predicts performance)
        # or "measured" (real on-hardware timing via the static evaluator).
        # System-space owner: "cost_model" (CM sweeps deployments internally)
        # or "optimizer" (the optimizer searches system.system_design_space +
        # available_gpus and exactly one deployment is priced/launched per
        # trial). Derived from performance_source when the key is absent.
        from rag_stack.performance_context import (
            resolve_performance_source,
            resolve_system_space_owner,
        )
        self.performance_source: str = resolve_performance_source(self.config)
        self.system_space_owner: str = resolve_system_space_owner(self.config)

    def _backend_spec(self):
        """Resolve backend metadata without constructing its runtime."""
        from rag_stack.evaluation.registry import get_backend_spec
        return get_backend_spec(self.gt_backend)

    def validate(self) -> List[str]:
        """Run all validation checks. Returns list of error messages (empty = valid)."""
        self.errors = []
        self._check_dataset()
        self._check_performance_source()
        self._check_system_space_owner()
        self._check_system()
        # Schema check runs first — fatal-shape errors will short-circuit the
        # downstream per-block checks (no point in walking pipeline blocks
        # if the schema is wrong).
        self._check_pipeline_schema()
        if not self.errors:
            self._check_pipeline_methods()
            self._check_pipeline_blocks()
        self._check_vectordb()
        self._check_gt_evaluation()
        self._check_optimizer()
        # Invalid workflows (including unavailable performance modeling) must
        # fail before model-cache discovery can start a network download.
        if self._check_model_cache_enabled and not self.errors:
            self._check_hf_model_cache()
        return self.errors

    def validate_system_only(
        self,
        *,
        require_cpu: bool = False,
        require_gpu: bool = False,
        require_available_gpus: bool = False,
    ) -> List[str]:
        """Validate a calibration-only config that carries just ``system:``.

        ``configs/calibrate/*.yaml`` files are hardware-spec inputs, not full
        optimizer configs, so the full six-section validator would reject them
        for missing dataset/vectordb/pipeline sections. This subset still reuses
        the canonical system checks: CPU topology, preset-vs-inline exclusivity,
        and FAISS retrieval runtime knobs.
        """
        self.errors = []
        system = self.config.get("system")
        if not isinstance(system, dict):
            self.errors.append("[system] Missing 'system:' section.")
            return self.errors
        if "gpu" not in system and "cpu" not in system:
            self.errors.append("[system] Must specify at least 'cpu' or 'gpu'.")
        if require_cpu and "cpu" not in system:
            self.errors.append("[system.cpu] Required for this calibration command.")
        if require_gpu and "gpu" not in system:
            self.errors.append("[system.gpu] Required for this calibration command.")

        ps = str(system.get("performance_source", "cost_model")).lower()
        if ps not in ("cost_model", "measured"):
            self.errors.append(
                f"[system.performance_source] {ps!r} is not one of "
                "'cost_model' / 'measured'."
            )
        owner = system.get("system_space_owner")
        if owner is not None and str(owner).lower() not in (
            "cost_model", "optimizer",
        ):
            self.errors.append(
                f"[system.system_space_owner] {owner!r} is not one of "
                "'cost_model' / 'optimizer'."
            )

        cpu = system.get("cpu")
        if isinstance(cpu, dict):
            self._check_cpu_topology(cpu)
        self._check_hw_preset_exclusive(system)
        self._check_system_retrieval(system)
        self._check_rag_cm(system)

        if require_available_gpus:
            from rag_stack.performance_context import system_design_space
            gpus = system.get("available_gpus") or system_design_space(
                system
            ).get("available_gpus")
            if not isinstance(gpus, list) or not gpus:
                self.errors.append(
                    "[system.available_gpus] Required: non-empty GPU device list."
                )
        return self.errors

    # ------------------------------------------------------------------
    # Dataset section
    # ------------------------------------------------------------------

    def _check_dataset(self):
        """Require ``dataset.dataset_name`` — the explicit, human-readable id the
        IVF cell-imbalance profile is keyed by (``dataset_name__embedding``).

        Mandatory and NEVER derived from a file path: the in-project corpus copy
        has a fixed basename and absolute paths differ across machines, so a path
        cannot be a stable cross-host key. Must be a non-empty string.
        """
        dataset = self.config.get("dataset")
        if not isinstance(dataset, dict):
            self.errors.append("[dataset] Missing 'dataset:' section.")
            return
        name = dataset.get("dataset_name")
        if not isinstance(name, str) or not name.strip():
            self.errors.append(
                "[dataset.dataset_name] Required: a non-empty string naming the "
                "dataset (e.g. 'dragonball_en'). It keys the IVF cell-imbalance "
                "profile (dataset_name__embedding) and is never path-derived."
            )

    # ------------------------------------------------------------------
    # Performance source (cost_model vs measured)
    # ------------------------------------------------------------------

    def _check_performance_source(self):
        """Validate ``system.performance_source`` — WHERE performance numbers
        come from. The structural requirements of an optimizer-owned system
        space (GPU list, design-space dims) live in
        ``_check_system_space_owner``; the one measured-specific requirement
        left here is the evaluator backend."""
        ps = self.performance_source
        if ps not in ("cost_model", "measured"):
            self.errors.append(
                f"[system.performance_source] {ps!r} is not one of "
                "'cost_model' / 'measured'."
            )
            return
        if ps != "measured":
            return
        # performance_only is a standalone-replay/benchmark switch: it skips
        # ALL quality scoring, so an optimize run (which needs the quality
        # objective) must never set it.
        ebs = self.config.get("eval_backend_setting") or {}
        if ebs.get("performance_only") and self.config.get("optimizer"):
            self.errors.append(
                "[eval_backend_setting.performance_only] not allowed in an "
                "optimize config — the quality objective would be empty. It is "
                "meant for standalone measured perf replays/benchmarks only."
            )
        # The static evaluator owns the measured path (MeasuredEvaluator wraps
        # StaticRAGEvaluatorQualityOnly); flashrag has no measured evaluator.
        if self.gt_backend != "static_gt":
            self.errors.append(
                f"[gt_evaluation.backend] performance_source=measured requires "
                f"backend 'static_gt' (got {self.gt_backend!r})."
            )

    def _check_system_space_owner(self):
        """Validate ``system.system_space_owner`` — WHO searches the system
        design space — the (owner, performance_source) matrix, and the
        optimizer-owned design-space requirements (GPU list + dims block)."""
        from rag_stack.performance_context import (
            SYSTEM_SPACE_OWNER_VALID_VALUES,
            system_design_space,
        )

        owner = self.system_space_owner
        system = self.config.get("system") or {}
        if owner not in SYSTEM_SPACE_OWNER_VALID_VALUES:
            self.errors.append(
                f"[system.system_space_owner] {owner!r} is not one of "
                "'cost_model' / 'optimizer'."
            )
            return
        if (
            "system_design_space" in system
            and "measured_search_space" in system
        ):
            self.errors.append(
                "[system.system_design_space] both 'system_design_space' and "
                "its deprecated spelling 'measured_search_space' are set — "
                "keep only 'system_design_space'."
            )
        if owner != "optimizer":
            if self.performance_source == "measured":
                self.errors.append(
                    "[system.system_space_owner] system_space_owner=cost_model "
                    "is incompatible with performance_source=measured: a "
                    "measured run launches ONE deployment per trial, it cannot "
                    "realize the cost model's internal deployment sweep. Set "
                    "system_space_owner: optimizer (or drop the key)."
                )
            if system_design_space(system):
                logger.warning(
                    "[system.system_design_space] declared but "
                    "system_space_owner=cost_model — the block is ignored "
                    "because the cost model sweeps deployments internally. "
                    "Set system_space_owner: optimizer to search it."
                )
            return
        # owner == optimizer: the design space must be realizable.
        dss = system_design_space(system)
        gpus = dss.get("available_gpus")
        if not isinstance(gpus, list) or not gpus:
            self.errors.append(
                "[system.available_gpus] system_space_owner=optimizer (e.g. "
                "performance_source=measured) requires a non-empty "
                "'system_design_space.available_gpus' list (e.g. [cuda:0, cuda:1])."
            )
        else:
            ss = system.get("cm_search_space") or {}
            try:
                min_gpus = int(ss.get("min_num_gpus", 1) or 1)
                max_gpus = int(ss.get("max_num_gpus", len(gpus)) or len(gpus))
                max_per_stage = int(ss.get("max_num_gpus_per_stage", max_gpus) or max_gpus)
            except (TypeError, ValueError):
                self.errors.append(
                    "[system.search_space] optimizer-owned GPU bounds "
                    "min_num_gpus/max_num_gpus/max_num_gpus_per_stage must be integers."
                )
            else:
                if min_gpus < 1:
                    self.errors.append(
                        "[system.search_space.min_num_gpus] must be >= 1 when "
                        "the optimizer owns the system space."
                    )
                if max_gpus < min_gpus:
                    self.errors.append(
                        "[system.search_space.max_num_gpus] must be >= min_num_gpus "
                        f"(got min={min_gpus}, max={max_gpus})."
                    )
                if max_gpus > len(gpus):
                    self.errors.append(
                        "[system.search_space.max_num_gpus] cannot exceed "
                        f"len(system.available_gpus)={len(gpus)} when the "
                        f"optimizer owns the system space (got {max_gpus})."
                    )
                if max_per_stage < 1:
                    self.errors.append(
                        "[system.search_space.max_num_gpus_per_stage] must be >= 1 "
                        "when the optimizer owns the system space."
                    )
        if not dss:
            self.errors.append(
                "[system.system_design_space] system_space_owner=optimizer "
                "requires a 'system_design_space' dict of orthogonal runtime "
                "knobs (batch_size_request, batch_size_decode, "
                "dynamic_batch_timeout_s, vllm_kv_cache_dtype) plus either a "
                "'deployment_preset' name list or nothing layout-shaped (the "
                "gpu_layout categorical is then generated from "
                "system.search_space.max_num_gpus_per_stage + available_gpus)."
            )
        else:
            # Guard-rail: warn (don't fail) about stale dims that are now DERIVED
            # from the partition / fixed — listing them in the design space
            # is silently ignored, which misleads about what's actually searched.
            stale = sorted(set(dss) & _DERIVED_MEASURED_KNOBS)
            if stale:
                logger.warning(
                    "[system.system_design_space] these keys are now DERIVED "
                    "from the GPU partition (or fixed) and are IGNORED if listed: "
                    f"{stale}. Remove them — the layout comes from system__gpu_layout "
                    "(expanded using system.search_space.max_num_gpus_per_stage "
                    "+ system.available_gpus). Keep only batch_size_request / "
                    "batch_size_decode / dynamic_batch_timeout_s / "
                    "vllm_kv_cache_dtype."
                )
            # Per-stage batch dims are GONE: heterogeneous retrieval/rerank/etc.
            # batch caps are microbatching. The admitted global exceptions are
            # batch_size_decode (generator decode continuous-batching capacity)
            # and dynamic_batch_timeout_s (one shared count-or-timeout wait for
            # non-vLLM batched services).
            removed = sorted(set(dss) & {
                "batch_size_generator", "batch_size_query_expansion",
                "batch_size_retrieval", "batch_size_rerank",
            })
            if removed:
                self.errors.append(
                    "[system.system_design_space] per-stage batch dims were "
                    f"removed (found {removed}): they contradict the "
                    "no-microbatching world model the cost model prices "
                    "(one request batch flows through all stages). Use the "
                    "single global 'batch_size_request' instead."
                )
            if "batch_size_request" in dss and "vllm_max_num_seqs" in dss:
                self.errors.append(
                    "[system.system_design_space] specify only one of "
                    "'batch_size_request' or 'vllm_max_num_seqs'. The global "
                    "batch drives the vLLM request-concurrency knob."
                )
        self._check_deployment_presets(system, dss)

    def _check_deployment_presets(self, system: dict, dss: dict):
        """Validate ``system.deployment_presets`` + the design-space
        ``deployment_preset`` dim: structure, name references, exclusivity
        against raw layout dims, and shape/budget coverage."""
        from rag_stack.search_space.deployment_presets import (
            parse_deployment_presets,
            preset_shape_fit_report,
        )

        try:
            presets = parse_deployment_presets(system)
        except ValueError as exc:
            self.errors.append(str(exc))
            return
        chosen = dss.get("deployment_preset")
        if not presets:
            if chosen is not None:
                self.errors.append(
                    "[system.system_design_space.deployment_preset] declared "
                    "but 'system.deployment_presets' defines no templates."
                )
            return
        if chosen is None:
            logger.warning(
                "[system.deployment_presets] templates declared but "
                "'system_design_space.deployment_preset' does not reference "
                "any — the block is dead. List the preset names there to "
                "search them."
            )
            return
        names = chosen if isinstance(chosen, list) else [chosen]
        if not names:
            self.errors.append(
                "[system.system_design_space.deployment_preset] must be a "
                "non-empty list of preset names."
            )
            return
        unknown = sorted(set(map(str, names)) - set(presets))
        if unknown:
            self.errors.append(
                f"[system.system_design_space.deployment_preset] unknown "
                f"preset name(s) {unknown} — define them under "
                f"system.deployment_presets (have: {sorted(presets)})."
            )
        # A preset IS the layout: raw layout dims would be a second,
        # conflicting driver (same rule as batch_size_request vs
        # vllm_max_num_seqs).
        raw_layout_keys = sorted(
            k for k in dss
            if k.startswith(("cut_after_", "num_gpus_", "pipeline_parallel_"))
        )
        if raw_layout_keys:
            self.errors.append(
                "[system.system_design_space] 'deployment_preset' cannot "
                f"coexist with raw layout dims {raw_layout_keys} — the preset "
                "is the complete deployment."
            )
        gpus = dss.get("available_gpus")
        cm_bounds = system.get("cm_search_space") or {}
        if not isinstance(gpus, list) or not gpus:
            return  # available_gpus error already reported above
        for nm in names:
            preset = presets.get(str(nm))
            if preset is None:
                continue
            fits, total = preset_shape_fit_report(
                preset, self.config, gpus, cm_bounds,
            )
            if total and fits == 0:
                self.errors.append(
                    f"[system.deployment_presets.{nm}] fits NONE of the "
                    f"{total} reachable pipeline shape(s) within the "
                    "cm_search_space GPU bounds — it could never produce a "
                    "valid trial."
                )
            elif total and fits < total:
                logger.warning(
                    f"[system.deployment_presets.{nm}] fits {fits} of {total} "
                    "reachable pipeline shapes — trials on the other shapes "
                    "will be penalized as invalid."
                )

    # ------------------------------------------------------------------
    # System / hardware
    # ------------------------------------------------------------------

    def _check_system(self):
        system = self.config.get("system")
        if system is None:
            self.errors.append("[system] Missing 'system' section.")
            return
        if "gpu" not in system and "cpu" not in system:
            self.errors.append("[system] Must specify at least 'cpu' or 'gpu'.")
        search_space = system.get("cm_search_space")
        if search_space is None:
            self.errors.append("[system] Missing 'search_space' in system config.")
            return
        batch_sizes = search_space.get("batch_size", [])
        for bs in _flatten_values(batch_sizes):
            if not _is_power_of_two(bs):
                self.errors.append(f"[system.search_space] batch_size={bs} is not a power of 2.")
        # CPU topology — discriminated by type, cross-checks num_cores.
        cpu = system.get("cpu")
        if isinstance(cpu, dict):
            self._check_cpu_topology(cpu)
        # hardware_key is the SINGLE locator for ceff/meff + all calibration: a known
        # preset key must NOT carry extra specs (else it would silently shift the base
        # the calibration was fit against). New keys may set specs freely.
        self._check_hw_preset_exclusive(system)
        self._check_system_retrieval(system)
        self._check_rag_cm(system)

    def _check_rag_cm(self, system: dict) -> None:
        """Validate the default RAG-CM policy block.

        Runtime profile roots are deliberately not per-config knobs.  The CM uses
        its canonical root (or the process-wide environment
        override), so accepting store fields here would silently preserve a
        configuration that the active cost model does not consume.
        """

        rag_cm = system.get("rag_cm")
        if rag_cm is None:
            return
        if not isinstance(rag_cm, dict):
            self.errors.append("[system.rag_cm] must be a mapping.")
            return
        unsupported = sorted(set(rag_cm) - {"calibration_mode"})
        for field in unsupported:
            self.errors.append(
                f"[system.rag_cm.{field}] is not supported by the default "
                "cost_model configuration contract. Remove it; the CM reads "
                "the canonical calibration_profiles root (or the "
                "process-wide RAG_STACK_CALIBRATION_PROFILE_DIR override)."
            )
        calibration_mode = rag_cm.get("calibration_mode", "strict")
        if (
            not isinstance(calibration_mode, str)
            or calibration_mode not in {"strict", "uncalibrated_spec"}
        ):
            self.errors.append(
                "[system.rag_cm.calibration_mode] must be 'strict' or "
                "'uncalibrated_spec'."
            )

    def _check_hw_preset_exclusive(self, system: dict) -> None:
        """Enforce: a ``gpu``/``cpu`` whose key matches a known PRESET may not also
        carry explicit hardware specs (calibration is anchored on the preset base).
        A NEW key (not a preset) may set all specs. ``interconnect`` is exempt — it
        has no calibration profile, so its preset⊕override (a link type plus a
        measured ``bandwidth_gbps``) stays valid. See docs/config_schema.md."""
        try:
            from Systems.system_configs import system_configs as _gpu_presets
        except Exception:  # noqa: BLE001
            _gpu_presets = {}
        from rag_stack.cost_model.cpu_specs import CPU_SPECS as _cpu_presets

        for name, presets, key_fields in (
            ("gpu", set(_gpu_presets), ("hardware_key",)),
            ("cpu", set(_cpu_presets), ("hardware_key", "cpu_preset")),
        ):
            val = system.get(name)
            if isinstance(val, str) or not isinstance(val, dict):
                continue  # bare preset string (or absent) → nothing to conflict with
            key = next((val[k] for k in key_fields if val.get(k)), None)
            if key is None:
                self.errors.append(
                    f"[system.{name}] inline hardware specs require a non-empty "
                    "hardware_key; calibration identity cannot be inferred "
                    "from the controller host."
                )
                continue
            if key in presets:
                extra = sorted(k for k in val if k not in key_fields)
                if extra:
                    self.errors.append(
                        f"[system.{name}] '{key}' is a known preset — it locates all "
                        f"calibration by itself, so it must NOT also set hardware specs "
                        f"{extra}. Use the bare key (e.g. {name}: {key}) for the preset, "
                        f"or a NEW hardware_key to define custom specs."
                    )

    def _check_system_retrieval(self, system: dict) -> None:
        """Validate the optional ``system.retrieval`` runtime-knob block."""
        css = system.get("cm_search_space")
        if isinstance(css, dict) and "faiss_parallel_mode" in css:
            self.errors.append(
                "[system.cm_search_space.faiss_parallel_mode] renamed to "
                "'faiss_ivf_parallel_mode' — the knob is IVF-only (HNSW has "
                "no parallel_mode in FAISS)."
            )
        grid = css.get("faiss_ivf_parallel_mode") if isinstance(css, dict) else None
        if grid is not None:
            values = list(grid) if isinstance(grid, (list, tuple)) else [grid]
            if not values or any(value != 0 for value in values):
                self.errors.append(
                    "[system.cm_search_space.faiss_ivf_parallel_mode] may "
                    "contain only 0; mode 1 has no standalone calibration evidence."
                )
        retrieval = system.get("retrieval")
        if retrieval is None:
            return
        if not isinstance(retrieval, dict):
            self.errors.append("[system.retrieval] must be a mapping.")
            return
        known = {
            "faiss_num_threads",
            "faiss_ivf_parallel_mode",
            "faiss_indexing_thread",
        }
        if "faiss_parallel_mode" in retrieval:
            self.errors.append(
                "[system.retrieval.faiss_parallel_mode] renamed to "
                "'faiss_ivf_parallel_mode' — the knob is IVF-only (HNSW has no "
                "parallel_mode in FAISS). The calibrated IVF model currently "
                "supports mode 0 only."
            )
        for key in retrieval:
            if key not in known and key != "faiss_parallel_mode":
                self.errors.append(
                    f"[system.retrieval] unknown key {key!r} "
                    f"(known: {sorted(known)})."
                )
        nt = retrieval.get("faiss_num_threads")
        if nt is not None:  # explicit null = "derive min(batch, cores)" (same as omitting)
            for v in _flatten_values(nt):
                if not isinstance(v, int) or v < 1:
                    self.errors.append(
                        f"[system.retrieval.faiss_num_threads] {v!r} must be an "
                        "int >= 1, a list of them (search dim), or null "
                        "(= derive min(batch, cores))."
                    )
        it = retrieval.get("faiss_indexing_thread")
        if it is not None:  # explicit null = default cpu_count-2 (same as omitting)
            if not isinstance(it, int) or isinstance(it, bool) or it < 1:
                self.errors.append(
                    f"[system.retrieval.faiss_indexing_thread] {it!r} must be an "
                    "int >= 1 (build-time OMP threads) or null/omitted "
                    "(= default cpu_count-2). NOT a search dim — build cost is "
                    "outside the cost model."
                )
        pm = retrieval.get("faiss_ivf_parallel_mode")
        if isinstance(pm, list):
            self.errors.append(
                "[system.retrieval.faiss_ivf_parallel_mode] a LIST here is the old "
                "optimizer-dim declaration. The calibrated IVF model accepts "
                "only the scalar value 0."
            )
        elif pm is not None and pm != 0:
            self.errors.append(
                f"[system.retrieval.faiss_ivf_parallel_mode] {pm!r} is "
                "unsupported; the standalone calibration covers mode 0 only."
            )
    def _check_cpu_topology(self, cpu: dict) -> None:
        """Validate the optional ``cpu.topology:`` block.

        Three mutually exclusive types (no NUMA+heterogeneous fusion):
          - ``uniform`` (or topology block absent) — single socket, homogeneous
          - ``numa`` — multi-socket; requires ``sockets`` + ``cores_per_socket``,
            and ``sockets × cores_per_socket == num_cores``
          - ``heterogeneous`` — big.LITTLE; requires ``p_cores.count`` +
            ``e_cores.count``, and ``p+e == num_cores``
        """
        topology = cpu.get("topology")
        if topology is None:
            return
        if not isinstance(topology, dict):
            self.errors.append(
                f"[system.cpu.topology] must be a dict, got {type(topology).__name__}."
            )
            return
        t_type = str(topology.get("type", "uniform")).lower()
        if t_type not in ("uniform", "numa", "heterogeneous"):
            self.errors.append(
                f"[system.cpu.topology.type] {t_type!r} is not one of "
                "'uniform' / 'numa' / 'heterogeneous'."
            )
            return

        num_cores = int(cpu.get("num_cores", 0))

        if t_type == "numa":
            sockets = topology.get("sockets")
            cores_per_socket = topology.get("cores_per_socket")
            if sockets is None or cores_per_socket is None:
                self.errors.append(
                    "[system.cpu.topology] type=numa requires both 'sockets' "
                    "and 'cores_per_socket'."
                )
                return
            if int(sockets) < 2:
                self.errors.append(
                    f"[system.cpu.topology] type=numa needs sockets >= 2 "
                    f"(got {sockets}); use type=uniform for single-socket systems."
                )
            if int(sockets) * int(cores_per_socket) != num_cores:
                self.errors.append(
                    f"[system.cpu.topology] sockets ({sockets}) × "
                    f"cores_per_socket ({cores_per_socket}) = "
                    f"{int(sockets) * int(cores_per_socket)} != num_cores ({num_cores})."
                )

        elif t_type == "heterogeneous":
            p = topology.get("p_cores") or {}
            e = topology.get("e_cores") or {}
            if not p or not e:
                self.errors.append(
                    "[system.cpu.topology] type=heterogeneous requires both "
                    "'p_cores' and 'e_cores' sub-blocks."
                )
                return
            p_count = int(p.get("count", 0))
            e_count = int(e.get("count", 0))
            if p_count <= 0 or e_count <= 0:
                self.errors.append(
                    f"[system.cpu.topology] heterogeneous needs p_cores.count > 0 "
                    f"and e_cores.count > 0 (got p={p_count}, e={e_count})."
                )
            if p_count + e_count != num_cores:
                self.errors.append(
                    f"[system.cpu.topology] p_cores.count ({p_count}) + "
                    f"e_cores.count ({e_count}) = {p_count + e_count} != "
                    f"num_cores ({num_cores})."
                )
            # Reject illegal NUMA/sockets keys at heterogeneous systems.
            for stray in ("sockets", "cores_per_socket"):
                if stray in topology:
                    self.errors.append(
                        f"[system.cpu.topology] type=heterogeneous must not set "
                        f"{stray!r} (NUMA+heterogeneous coexistence is not "
                        "supported)."
                    )

    # ------------------------------------------------------------------
    # Pipeline schema (strict — rejects old shape with migration messages)
    # ------------------------------------------------------------------

    def _check_pipeline_schema(self):
        """Validate ``algo_search_space.pipeline`` shape per eval backend.

        static_gt → ``{node_lines: [...]}`` (sequential ONLY — no ``mode``);
        flashrag  → ``{methods: [{mode, node_lines, ...}, ...]}``.
        Legacy top-level keys (``pipeline:``, ``node_lines:``, ``corpus:``,
        ``data:``, ``gt_evaluation:``, ``n_iterations:``) are hard-rejected —
        the 6-section schema has no backward compatibility.
        """
        for legacy in (
            "pipeline", "node_lines", "corpus", "data", "gt_evaluation", "n_iterations",
        ):
            if legacy in self.config:
                self.errors.append(
                    f"[{legacy}] Top-level '{legacy}:' is no longer supported. The "
                    "6-section schema is: global / dataset / system / "
                    "optimizer / eval_backend_setting / algo_search_space."
                )
        if "vectordb" in self.config:
            self.errors.append(
                "[vectordb] Top-level 'vectordb:' has moved — the vector-index "
                "section is ALGO search space (index family/build knobs move "
                "quality). Put the block list under 'algo_search_space.vectordb:'; "
                "the resolved runtime config still exposes config['vectordb']."
            )

        try:
            backend_spec = self._backend_spec()
        except (ValueError, TypeError, ImportError) as exc:
            self.errors.append(f"[global.eval_backend] {exc}")
            return

        asp = self.config.get("algo_search_space")
        if not isinstance(asp, dict):
            self.errors.append("[algo_search_space] Missing 'algo_search_space:' section.")
            return
        pipeline = asp.get("pipeline")
        if not isinstance(pipeline, dict):
            self.errors.append(
                "[algo_search_space.pipeline] must be a mapping: {node_lines: [...]} "
                "(static_gt) or {methods: [...]} (flashrag)."
            )
            return
        if backend_spec.pipeline_style == "node_lines":
            if "methods" not in pipeline and "node_lines" not in pipeline:
                self.errors.append(
                    f"[algo_search_space.pipeline] {self.gt_backend} requires {{node_lines: [...]}} "
                    "(sequential) or {methods: [{mode, node_lines}, ...]} (e.g. react)."
                )
        elif "methods" not in pipeline:
            self.errors.append(
                f"[algo_search_space.pipeline] {self.gt_backend} requires "
                "{methods: [{mode, node_lines, ...}, ...]}."
            )

        # Collaborative co-mode: a flat `rag_dataflow` knob over a SHARED
        # node_lines. Each entry may carry that branch's SUB-CONFIG
        # ({react: {max_iter: [...]}}) — the search-space builder wires the
        # declared hierarchy into Ax dependents. Validate shape + value set.
        if isinstance(pipeline, dict) and "execution_mode" in pipeline:
            self.errors.append(
                "[algo_search_space.pipeline.execution_mode] renamed to "
                "`rag_dataflow` (same values: sequential / react). Update the config."
            )
        if isinstance(pipeline, dict) and "max_agent_steps_safety" in pipeline:
            self.errors.append(
                "[algo_search_space.pipeline.max_agent_steps_safety] removed — "
                "`max_iter` (react's sub-config) is the loop's only round "
                "cap now. Delete this key and nest max_iter under the react "
                "branch of rag_dataflow."
            )
        if isinstance(pipeline, dict) and "max_iter" in pipeline:
            self.errors.append(
                "[algo_search_space.pipeline.max_iter] max_iter is react's "
                "SUB-CONFIG — nest it under the react branch: "
                "rag_dataflow: [sequential, {react: {max_iter: [...]}}]."
            )
        if isinstance(pipeline, dict) and "rag_dataflow" in pipeline:
            from rag_stack.search_space.rag_search_space import parse_dataflow_spec
            from rag_stack.evaluation.registry import supported_pipeline_modes
            allowed = (
                set(supported_pipeline_modes(self.gt_backend)) & {"sequential", "react"}
                if backend_spec.pipeline_style == "node_lines" else set()
            )
            try:
                branch_names, branch_knobs = parse_dataflow_spec(
                    pipeline["rag_dataflow"]
                )
            except ValueError as e:
                self.errors.append(f"[algo_search_space.pipeline.rag_dataflow] {e}")
            else:
                bad = [v for v in branch_names if v not in allowed]
                if bad:
                    self.errors.append(
                        f"[algo_search_space.pipeline.rag_dataflow] {bad} not allowed; "
                        f"{self.gt_backend} co-mode supports {sorted(allowed)}."
                    )
                # react reachable → its branch MUST declare max_iter: the react
                # loop's ONLY round cap (engine contract).
                if "react" in branch_names and "max_iter" not in (
                    branch_knobs.get("react") or {}
                ):
                    self.errors.append(
                        "[algo_search_space.pipeline.rag_dataflow.react.max_iter] "
                        "required — max_iter is the react loop's only round cap "
                        "(max_agent_steps_safety was removed)."
                    )

        try:
            blocks = _pipeline_blocks(self.config)
        except ValueError as e:
            self.errors.append(f"[algo_search_space.pipeline] {e}")
            return
        if not blocks:
            self.errors.append("[algo_search_space.pipeline] no method blocks found.")
            return

        seen_modes: set[str] = set()
        for i, block in enumerate(blocks):
            if not isinstance(block, dict):
                self.errors.append(
                    f"[algo_search_space.pipeline[{i}]] block must be a dict, got "
                    f"{type(block).__name__}."
                )
                continue
            mode = block.get("mode")
            if not isinstance(mode, str):
                self.errors.append(
                    f"[algo_search_space.pipeline[{i}]] 'mode' must be a string."
                )
                continue
            if mode in seen_modes:
                self.errors.append(
                    f"[algo_search_space.pipeline[{i}]] Duplicate method '{mode}'."
                )
            seen_modes.add(mode)
            if "node_lines" not in block:
                self.errors.append(
                    f"[algo_search_space.pipeline[{i}].{mode}] Missing 'node_lines'."
                )

    # ------------------------------------------------------------------
    # Pipeline methods (backend-aware registry; lazy import)
    # ------------------------------------------------------------------

    def _check_pipeline_methods(self):
        """Cross-check every pipeline block's ``mode`` against the backend's
        supported-methods registry. Lazy-imports the registry so the
        static_gt path never pulls in the FlashRAG package."""
        pipeline_cfg = _pipeline_blocks(self.config)
        if not isinstance(pipeline_cfg, list):
            return

        from rag_stack.evaluation.registry import supported_pipeline_modes

        backend = self.gt_backend
        supported = frozenset(supported_pipeline_modes(backend))
        label = "PIPELINE_MAP" if backend == "flashrag" else "registered pipeline modes"

        for i, block in enumerate(pipeline_cfg):
            if not isinstance(block, dict):
                continue
            mode = block.get("mode")
            if not isinstance(mode, str):
                continue
            if mode not in supported:
                self.errors.append(
                    f"[pipeline[{i}].mode] Method {mode!r} is not in "
                    f"{label} for backend '{self.gt_backend}'. "
                    f"Supported: {sorted(supported)}."
                )

    # ------------------------------------------------------------------
    # Per-method-block node-line checks (replaces old top-level walk)
    # ------------------------------------------------------------------

    def _check_pipeline_blocks(self):
        """Walk every method block's ``node_lines`` and verify the same
        per-block constraints the old top-level validator enforced
        (generator present, prompt_maker present for static_gt, top_k
        power-of-two for static_gt's RAGO cost model, etc.)."""
        pipeline_cfg = _pipeline_blocks(self.config)
        if not isinstance(pipeline_cfg, list):
            return
        from rag_stack.pipeline_contract import (
            hybrid_capability_errors, hybrid_structure_errors,
        )

        for block in pipeline_cfg:
            if not isinstance(block, dict):
                continue
            mode = block.get("mode")
            node_lines = block.get("node_lines")
            if not isinstance(node_lines, list) or not node_lines:
                continue
            block_label = f"pipeline[{mode}]"
            self.errors.extend(
                f"[{block_label}] {error}"
                for error in hybrid_structure_errors(block, backend=self.gt_backend)
            )
            self.errors.extend(
                f"[{block_label}] {error}"
                for error in hybrid_capability_errors(
                    block, performance_source=self.performance_source,
                )
            )

            has_generator = False
            has_prompt_maker = False
            has_retrieval = False

            for nl in node_lines:
                nl_name = nl.get("node_line_name", "<unnamed>")
                full_loc = f"{block_label}.{nl_name}"
                nodes = nl.get("nodes", [])
                for node in nodes:
                    stage = node.get("stage")
                    if not stage:
                        self.errors.append(f"[{full_loc}] Node missing 'stage'.")
                        continue

                    if stage == "generator":
                        has_generator = True
                        self._check_generator_node(node, full_loc)
                    elif stage == "prompt_maker":
                        has_prompt_maker = True
                        self._check_prompt_maker_node(node, full_loc)
                    elif stage in ("semantic_retrieval", "lexical_retrieval", "hybrid_retrieval"):
                        has_retrieval = True
                        self._check_retrieval_node(node, full_loc)
                    elif stage == "passage_reranker":
                        self._check_top_k_power_of_two(node, full_loc, "passage_reranker")
                    elif stage == "passage_compressor":
                        self._check_compressor_node(node, full_loc)

                    modules = node.get("modules", [])
                    if not modules:
                        self.errors.append(f"[{full_loc}.{stage}] No modules specified.")
                    self._check_vllm_chat_contract(
                        modules,
                        f"{full_loc}.{stage}",
                    )
                    if stage != "prompt_maker" and self.performance_source == "cost_model":
                        self._check_cost_model_mapping(modules, full_loc, stage)

            if not has_generator:
                self.errors.append(f"[{block_label}] No 'generator' node found. At least one is required.")
            if self.gt_backend == "static_gt" and mode != "react" and not has_prompt_maker:
                # react builds its own agentic prompt internally (like the flashrag
                # agentic methods), so it needs no prompt_maker node.
                self.errors.append(f"[{block_label}] No 'prompt_maker' node found. At least one is required.")
            if not has_retrieval:
                self.errors.append(f"[{block_label}] No retrieval node found (semantic_retrieval, lexical_retrieval, or hybrid_retrieval).")

    def _check_generator_node(self, node: dict, nl_name: str):
        modules = node.get("modules", [])
        for i, mod in enumerate(modules):
            if "model" not in mod:
                self.errors.append(
                    f"[{nl_name}.generator.modules[{i}]] Missing 'model' field."
                )
            backend = mod.get("generator_backend")
            if backend:
                self._check_generator_backend(backend, f"{nl_name}.generator.modules[{i}]")

    def _check_generator_backend(self, backend, location: str):
        try:
            from rag_stack_evaluator.static_rag_evaluator.nodes.generator.registry import GENERATOR_BACKENDS
            if backend not in GENERATOR_BACKENDS:
                self.errors.append(
                    f"[{location}] Unknown generator_backend '{backend}'. "
                    f"Valid: {list(GENERATOR_BACKENDS.keys())}"
                )
        except ImportError:
            pass

    def _check_vllm_chat_contract(self, modules: list, location: str):
        """Keep historical native completion replay explicit and quality-only."""
        from rag_stack_evaluator.generation_protocol import resolve_generation_protocol

        for i, mod in enumerate(modules):
            if not isinstance(mod, dict):
                continue
            uses_local_vllm = (
                mod.get("component") == "vllm"
                or mod.get("generator_backend") == "vllm"
            )
            if not uses_local_vllm and "generation_protocol" not in mod:
                continue
            allow_legacy = (
                uses_local_vllm
                and getattr(self, "gt_backend", "static_gt") == "static_gt"
                and getattr(self, "performance_source", "cost_model") == "cost_model"
            )
            try:
                resolve_generation_protocol(mod, allow_legacy=allow_legacy)
            except ValueError as exc:
                self.errors.append(f"[{location}.modules[{i}]] {exc}")

    def _check_prompt_maker_node(self, node: dict, nl_name: str):
        modules = node.get("modules", [])
        has_prompt = any(mod.get("prompt") for mod in modules)
        if not has_prompt:
            self.errors.append(
                f"[{nl_name}.prompt_maker] No module has a 'prompt' field."
            )

    def _check_retrieval_node(self, node: dict, nl_name: str):
        if "top_k" not in node:
            self.errors.append(
                f"[{nl_name}.{node['stage']}] Missing 'top_k'."
            )
        elif node["stage"] == "semantic_retrieval":
            self._check_top_k_power_of_two(node, nl_name, node["stage"])

        # Check vectordb reference exists. A module may reference a single
        # index (scalar) or a swept list of indexes (the optimizer chooses
        # one per trial, e.g. [faiss_ivf_index, faiss_hnsw_index]); every
        # referenced name must resolve.
        modules = node.get("modules", [])
        vdb_names = {vdb["name"] for vdb in _vdb_blocks(self.config) if "name" in vdb}
        for mod in modules:
            sel = mod.get("vectordb")
            if isinstance(sel, list) and len(sel) > 1:
                self.errors.append(
                    f"[{nl_name}.{node['stage']}] vectordb: a multi-family "
                    "selector must use the dict form — vectordb: {<family>: "
                    "{<its query-time knobs>}} — so per-family knob ownership "
                    "(nprobe/ef_search/…) is declared by nesting and compiled "
                    "into Ax gating. The flat list form leaves the non-chosen "
                    "family's knobs as always-active phantom dims."
                )
            if isinstance(sel, dict):
                for query_knob in ("nprobe", "ef_search"):
                    if query_knob in mod:
                        self.errors.append(
                            f"[{nl_name}.{node['stage']}] '{query_knob}' is a "
                            "family-owned query-time knob — declare it INSIDE "
                            "its family in the vectordb selector dict, not "
                            "flat on the module (flat leaves it always-active "
                            "for the other family)."
                        )
            for vdb_ref in _flatten_values(sel):
                if vdb_ref and vdb_ref not in vdb_names:
                    self.errors.append(
                        f"[{nl_name}.{node['stage']}] vectordb '{vdb_ref}' "
                        f"not found in algo_search_space.vectordb. "
                        f"Available: {vdb_names or 'none'}"
                    )

    def _check_top_k_power_of_two(self, node: dict, nl_name: str, stage: str):
        # The power-of-2 constraint comes from RAGO's cost model. The FlashRAG
        # backend uses its own retriever path (no RAGO retrieval cost), so any
        # top_k is fine.
        if self.gt_backend == "flashrag" or getattr(self, "performance_source", "cost_model") != "cost_model":
            return
        top_k = node.get("top_k")
        if top_k is None:
            return
        for v in _flatten_values(top_k):
            if not _is_power_of_two(v):
                self.errors.append(
                    f"[{nl_name}.{stage}] top_k={v} is not a power of 2. "
                    f"RAGO cost model requires powers of 2 (1, 2, 4, 8, 16, ...)."
                )

    def _check_compressor_node(self, node: dict, nl_name: str):
        modules = node.get("modules", [])
        for mod in modules:
            mod_type = mod.get("component", "")
            if mod_type in ("tree_summarize", "refine") and "generator_backend" not in mod:
                self.errors.append(
                    f"[{nl_name}.passage_compressor.{mod_type}] "
                    f"Missing 'generator_backend' for LLM-based compressor."
                )

    # ------------------------------------------------------------------
    # Cost model mapping
    # ------------------------------------------------------------------

    # components that are pass-through (no computation, no cost model needed).
    # Skippable nodes are now expressed via `optional: true` + `enabled=false`
    # (see controller._resolve_config), so no pass_* components remain here.
    _PASS_THROUGH_MODULES = {
        # prompt_maker modules (no GPU cost)
        "fstring", "long_context_reorder", "window_replacement", "chat_fstring",
        # rerankers with no ML model (pure-Python sort) — intentionally absent
        # from cost_model_mapping.json so they contribute zero rerank cost.
        "time_reranker",
    }

    def _check_cost_model_mapping(self, modules: list, nl_name: str, stage: str):
        """Check that non-pass-through modules are priceable by the CM dispatch."""
        from rag_stack.cost_model.dispatch import cost_model_for
        from rag_stack.taxonomy import TAXONOMY, Stage, expand_stage

        vdb_configs = {vdb["name"]: vdb for vdb in _vdb_blocks(self.config) if "name" in vdb}

        for mod in modules:
            mod_type = mod.get("component", "")
            if not mod_type or mod_type in self._PASS_THROUGH_MODULES:
                continue

            component = TAXONOMY.components.get(mod_type)
            if component is None or component.shape is None:
                # Unknown components are caught by the module-name check;
                # unpriced shapes (prompt makers, time_reranker) never reach
                # the CM — nothing to validate here.
                continue

            # vectordb: the vector-search split stage prices per db_type. The
            # ref may be a swept list of indexes — every chosen index's
            # db_type must have a dispatch binding.
            if mod_type == "vectordb":
                for vdb_ref in _flatten_values(mod.get("vectordb", "")):
                    vdb_cfg = vdb_configs.get(vdb_ref)
                    if vdb_cfg is None:
                        continue  # missing ref is caught by _check_retrieval_node
                    db_type = vdb_cfg.get("db_type", "")
                    if not db_type:
                        continue
                    try:
                        cost_model_for(
                            Stage.SEMANTIC_RETRIEVAL_VECTORSEARCH, "vectordb",
                            db_type=db_type,
                        )
                    except KeyError:
                        self.errors.append(
                            f"[{nl_name}.{stage}] vectordb db_type '{db_type}' "
                            f"has no cost model binding (cost_model/dispatch.py)."
                        )
                continue

            for ds in expand_stage(component.stage):
                try:
                    cost_model_for(ds, mod_type)
                except NotImplementedError:
                    # Raw YAML may declare these alongside HyDE.  The normal
                    # paper search-space builder removes the multi-list arms;
                    # a resolved/manual config that still selects one reaches
                    # cost_model_for again and fails directly.
                    if mod_type in {
                        "query_decompose",
                        "multi_query_expansion",
                    }:
                        continue
                    raise
                except KeyError:
                    self.errors.append(
                        f"[{nl_name}.{stage}] component '{mod_type}' has no cost "
                        f"model binding (cost_model/dispatch.py). Performance "
                        f"estimation will skip this module."
                    )

    # ------------------------------------------------------------------
    # VectorDB
    # ------------------------------------------------------------------

    def _check_vectordb(self):
        vectordbs = _vdb_blocks(self.config)
        names = set()
        for i, vdb in enumerate(vectordbs):
            name = vdb.get("name")
            if not name:
                self.errors.append(f"[algo_search_space.vectordb[{i}]] Missing 'name'.")
                continue
            if name in names:
                self.errors.append(f"[algo_search_space.vectordb] Duplicate name '{name}'.")
            names.add(name)

            db_type = vdb.get("db_type")
            if not db_type:
                self.errors.append(f"[vectordb.{name}] Missing 'db_type'.")

            if not vdb.get("embedding_model"):
                self.errors.append(f"[vectordb.{name}] Missing 'embedding_model'.")

            if "parallel_mode" in vdb:
                self.errors.append(
                    f"[vectordb.{name}] 'parallel_mode' moved to "
                    "'system.retrieval.faiss_ivf_parallel_mode' — it is a runtime "
                    "threading knob, not an index-build parameter. (Saved "
                    "per-eval pipeline_configs from old runs still replay via "
                    "the legacy fallback; new run configs must use the new "
                    "location.)"
                )

            # FAISS-specific checks
            if db_type == "faiss_ivf":
                self._check_faiss_ivf(vdb, name)
            elif db_type == "faiss_hnsw":
                self._check_faiss_hnsw(vdb, name)

    def _check_faiss_ivf(self, vdb: dict, name: str):
        """Validate a unified faiss_ivf vectordb (index_type: pq | flat)."""
        # embedding_dim is normally DERIVED from embedding_model (a fixed-dim
        # local model) and embedding_model may be a sweep list → every candidate
        # model contributes a dim, and M must divide ALL of them.
        dims = self._resolve_embedding_dims(vdb, name)
        if not dims:
            return  # error already recorded
        # index_type forms: nested dict {pq:{M,nbits}, flat:{...}} (preferred),
        # or a scalar "pq"/"flat", or a swept list [pq, flat] (back-compat).
        it = vdb.get("index_type", "pq")
        # Tolerate a single-dict-wrapped-in-a-list form [{pq:..., flat:...}] that
        # a UI/YAML round-trip can produce — treat it as the nested dict.
        if isinstance(it, list) and len(it) == 1 and isinstance(it[0], dict):
            it = it[0]
        if isinstance(it, dict):
            index_types = list(it.keys())
            pq_block = it.get("pq") or {}
            pq_M = pq_block.get("M", [])
            pq_dsub = pq_block.get("dsub", [])
        else:
            index_types = _flatten_values(it)
            pq_M = vdb.get("M", [])
            pq_dsub = vdb.get("dsub", [])
        bad = [t for t in index_types if t not in ("pq", "flat")]
        if bad:
            self.errors.append(
                f"[vectordb.{name}] index_type must be 'pq' or 'flat', got {bad}."
            )
            return
        # M must divide embedding_dim — only matters when PQ is (possibly)
        # selected; IVF-Flat has no sub-quantizers so M/nbits are irrelevant.
        # With a swept embedding_model, M must divide EVERY candidate model's
        # dim so the M knob stays flat (no per-model M gating).
        if "pq" in index_types:
            # ``dsub`` (target sub-vector dim D/M) → M is DERIVED at resolve
            # time as a divisor of D capped at 64, so it's valid by construction
            # — only require dsub be a positive number. Raw ``M`` configs still
            # must divide every candidate embedding dim.
            for d in _flatten_values(pq_dsub):
                if not isinstance(d, (int, float)) or d < 1:
                    self.errors.append(
                        f"[vectordb.{name}] dsub={d} must be a positive number "
                        f"(target PQ sub-vector dim D/M)."
                    )
            for m in _flatten_values(pq_M):
                m = int(m)
                bad_dims = sorted(dd for dd in dims if dd % m != 0)
                if bad_dims:
                    self.errors.append(
                        f"[vectordb.{name}] M={m} does not divide embedding_dim "
                        f"{bad_dims} evenly (dim derived from embedding_model). "
                        f"Pick M that divides every swept model's dim."
                    )

    def _resolve_embedding_dims(self, vdb: dict, name: str) -> set:
        """Resolve the set of candidate embedding dims for a faiss vectordb.

        ``embedding_dim`` is DERIVED from ``embedding_model`` for fixed-dim
        local models (``EMBEDDING_MODEL_DIMS``), and ``embedding_model`` may be
        a sweep list — so this returns EVERY candidate dim; M must divide all of
        them. Configurable-dim models (openai*/mock) aren't in the map → fall
        back to an explicit ``embedding_dim``. Records an error and returns an
        empty set when the dim can't be determined or an explicit value
        conflicts with the derived one.
        """
        from rag_stack_evaluator.static_rag_evaluator.embedding.base import (
            EMBEDDING_MODEL_DIMS,
        )
        models = _flatten_values(vdb.get("embedding_model", []))
        explicit = vdb.get("embedding_dim")
        derived = {
            EMBEDDING_MODEL_DIMS[m] for m in models if m in EMBEDDING_MODEL_DIMS
        }
        unknown = [m for m in models if m not in EMBEDDING_MODEL_DIMS]
        if derived:
            if explicit is not None and int(explicit) not in derived:
                self.errors.append(
                    f"[vectordb.{name}] embedding_dim={explicit} conflicts with "
                    f"dim(s) {sorted(derived)} derived from embedding_model — "
                    f"embedding_dim is derived for fixed-dim models; remove it."
                )
            if unknown:
                self.errors.append(
                    f"[vectordb.{name}] embedding_model mixes fixed-dim models "
                    f"{sorted(derived)} with configurable-dim {unknown}; a single "
                    f"sweep can't mix the two (dims/divisibility are ambiguous)."
                )
            return derived
        # No fixed-dim model → an explicit embedding_dim is required.
        if explicit is None:
            self.errors.append(
                f"[vectordb.{name}] Missing 'embedding_dim' "
                f"(embedding_model {models or '∅'} is not a known fixed-dim "
                f"model to derive it from)."
            )
            return set()
        return {int(explicit)}

    def _check_faiss_hnsw(self, vdb: dict, name: str):
        # HNSW does not constrain M relative to embedding_dim (unlike PQ), but
        # the dimension must be RESOLVABLE — explicit, or derived from
        # embedding_model (possibly a sweep list) exactly like faiss_ivf — so
        # the cost model and the runtime IndexHNSWFlat agree.
        self._resolve_embedding_dims(vdb, name)
        for m in _flatten_values(vdb.get("M", [])):
            if int(m) < 2:
                self.errors.append(
                    f"[vectordb.{name}] M={m} must be >= 2 for faiss_hnsw."
                )
        for ef in _flatten_values(vdb.get("ef_construction", [])):
            if int(ef) < 1:
                self.errors.append(
                    f"[vectordb.{name}] ef_construction={ef} must be >= 1."
                )
        for ef in _flatten_values(vdb.get("ef_search", [])):
            if int(ef) < 1:
                self.errors.append(
                    f"[vectordb.{name}] ef_search={ef} must be >= 1."
                )

    # ------------------------------------------------------------------
    # GT Evaluation metrics
    # ------------------------------------------------------------------

    def _check_gt_evaluation(self):
        glob = self.config.get("global") or {}
        eval_setting = self.config.get("eval_backend_setting") or {}
        # Dynamic traces are the common contract. Aggregate/static mode requires
        # an explicit capability advertised by the selected backend.
        from rag_stack.performance_context import (
            RAG_IR_MODE_VALID_VALUES,
            resolve_rag_ir_mode,
        )
        rag_ir_mode = glob.get("rag_ir_mode")
        if (
            "cm_data_source" not in glob
            and rag_ir_mode is not None
            and str(rag_ir_mode).lower() not in RAG_IR_MODE_VALID_VALUES
        ):
            self.errors.append(
                f"[global.rag_ir_mode] {rag_ir_mode!r} is not one of "
                "'static' / 'dynamic' (default: dynamic)."
            )
        elif "cm_data_source" in glob or (
            self.performance_source == "cost_model"
        ):
            try:
                resolve_rag_ir_mode(self.config)
            except ValueError as e:
                self.errors.append(str(e))
        options = eval_setting.get("backend_options")
        if options is not None and not isinstance(options, dict):
            self.errors.append("[eval_backend_setting.backend_options] Must be a dictionary.")
        if options is not None and self.gt_backend == "flashrag" and eval_setting.get("flashrag") is not None:
            self.errors.append("Configure FlashRAG with backend_options or flashrag, not both.")
        metrics = eval_setting.get("metrics", [])
        # The host registry resolves each backend's declared capabilities.
        from rag_stack.evaluation.registry import supported_metrics

        backend = self.gt_backend
        try:
            backend_spec = self._backend_spec()
            valid_metrics = set(supported_metrics(backend))
        except (ValueError, TypeError, ImportError) as exc:
            message = f"[global.eval_backend] {exc}"
            if message not in self.errors:
                self.errors.append(message)
            return
        if (
            self.performance_source == "cost_model"
            and str(rag_ir_mode).lower() == "static"
            and not backend_spec.supports_static_ir
        ):
            self.errors.append(
                f"[global.rag_ir_mode] Backend {backend!r} does not support static RAG-IR."
            )
        for m in metrics:
            name = m.get("metric_name") if isinstance(m, dict) else m
            if name and valid_metrics and name not in valid_metrics:
                self.errors.append(
                    f"[gt_evaluation] Unknown metric '{name}' for backend "
                    f"'{self.gt_backend}'. Valid: {sorted(valid_metrics)}"
                )

    def _check_hf_model_cache(self):
        """Check that HuggingFace models used in config are cached locally."""
        import json
        import os
        from pathlib import Path

        # Load mappings from the model-identity JSON
        mapping_file = Path(__file__).parent / "hf_model_mapping.json"
        if not mapping_file.is_file():
            return
        mapping = json.loads(mapping_file.read_text())
        hf_module_models: dict[str, list[str]] = mapping.get("module_models", {})
        hf_embedding_models: dict[str, str] = mapping.get("embedding_models", {})

        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            cached_repos = {repo.repo_id for repo in cache_info.repos}
        except Exception:
            cached_repos = set()

        # Also check sentence-transformers cache (uses different path)
        st_cache = os.path.join(os.path.expanduser("~"), ".cache", "torch", "sentence_transformers")
        if os.path.isdir(st_cache):
            for name in os.listdir(st_cache):
                # folder name format: org_model -> org/model
                cached_repos.add(name.replace("_", "/", 1))

        needed_repos = set()

        # OpenRouter/API names map to real HF tokenizer repos via TOKENIZER_MAP.
        from rag_stack.model_map import TOKENIZER_MAP

        # Modules: user-specified model overrides defaults from JSON.
        # node_lines now live inside each pipeline[i] method block.
        for block in _pipeline_blocks(self.config) or []:
            if not isinstance(block, dict):
                continue
            for nl in block.get("node_lines") or []:
                for node in nl.get("nodes", []):
                    for mod in node.get("modules", []):
                        mod_type = mod.get("component", "")
                        explicit_model = mod.get("model")
                        if explicit_model:
                            for name in _flatten_values(explicit_model):
                                repo = TOKENIZER_MAP.get(name, name)
                                if "/" in repo:
                                    needed_repos.add(repo)
                        elif mod_type in hf_module_models:
                            needed_repos.update(hf_module_models[mod_type])

        # Embedding models from vectordb config (may be a swept list).
        for vdb in _vdb_blocks(self.config):
            emb = vdb.get("embedding_model", "")
            for emb_value in _flatten_values(emb):
                if isinstance(emb_value, str) and emb_value in hf_embedding_models:
                    needed_repos.add(hf_embedding_models[emb_value])

        for repo in needed_repos:
            if repo not in cached_repos:
                logger.info(f"[model_cache] Downloading HuggingFace model '{repo}'...")
                try:
                    from huggingface_hub import snapshot_download
                    snapshot_download(repo)
                    logger.info(f"[model_cache] Successfully cached '{repo}'")
                except Exception as e:
                    self.errors.append(
                        f"[model_cache] Failed to download HuggingFace model '{repo}': {e}. "
                        f"Check your internet connection or download manually."
                    )

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def _check_optimizer(self):
        optimizer = self.config.get("optimizer")
        if optimizer is None:
            self.errors.append("[optimizer] Missing 'optimizer' section.")
            return
        opt_type = optimizer.get("type")
        from rag_stack.optimizer import OPTIMIZER_REGISTRY
        valid_types = set(OPTIMIZER_REGISTRY.keys())
        if opt_type and opt_type not in valid_types:
            self.errors.append(
                f"[optimizer] Unknown type '{opt_type}'. Valid: {sorted(valid_types)}"
            )
        params = optimizer.get("params", {})
        objectives = params.get("objectives", [])
        for obj in objectives:
            if "objective_name" not in obj:
                self.errors.append("[optimizer.objectives] Missing 'objective_name'.")
