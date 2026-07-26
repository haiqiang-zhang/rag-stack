import traceback
import ast
import csv
import datetime
import hashlib
import inspect
import json
import logging
import math
import os
import shutil
import tempfile
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import pandas as pd

from rag_stack.optimizer import OPTIMIZER_REGISTRY
from rag_stack.utils.config import load_yaml_config
from rag_stack.utils.objectives import (
    aggregate_quality_metrics,
    format_objective_value,
)
from rag_stack.evaluation.registry import create_quality_evaluator
from rag_stack.evaluator_contract import (
    normalize_backend_result, bind_trace_workflow, configured_backend_options,
)
from rag_stack.types import SystemConfig, EvalType
from rag_stack.cost_model.types import AssemblyResult
from rag_stack.cost_model.assembly import RAGCMAssembly
from rag_stack.rag_ir.trace import is_quality_trace_candidate, is_trace_payload
from rag_stack.rag_ir.workflow_schema import build_workflow_schema
from rag_stack.performance_context import PerformanceContext
from rag_stack.quality_store import QualityStore
from rag_stack.system_transfer_backend import (
    config_uses_only_local_vllm,
    project_to_vllm_chat,
)
from rag_stack.system_transfer_contract import validate_system_transfer_artifact
from rag_stack.optimizer.utils import config_hash, is_non_dominated_np
from rag_stack.project_summary import (
    ProjectStatus, ProjectSummary, CONFIG_FILENAME,
    load_project_summary, save_project_summary,
    pid_alive,
)
from rag_stack.project_config import (
    ProjectConfigError,
    freeze_project_config,
    project_config_lock,
    verify_project_config,
    write_project_config,
)

if TYPE_CHECKING:
    from rag_stack.dataset_generator.base import GeneratedDataset

logger = logging.getLogger("RAG-Stack")
_OPTIMIZER_OBSERVATION_FILENAME = "optimizer_observation.json"

# gen_fingerprint_dedup (M3) is a METHOD feature of the greedy/agent MOBO
# optimizers only. Every baseline must run WITHOUT it so the sweep's baseline
# semantics stay exactly historical — a baseline config carrying the flag is
# an experimental-design error, not a preference.
_GEN_FP_ALLOWED_OPTIMIZERS = frozenset({
    "greedy_then_hv_anchor_agent_qnehvi",   # greedy MOBO (v3, greedy_agent_mobo)
    "agent_qnehvi",                         # agent MOBO
})


def _restored_gt_budget_count(evaluations: Sequence[dict]) -> int:
    """Reconstruct consumed valid-evaluation seats across a resume.

    A quality-cache hit avoids GPU/judge work, but it is still a candidate the
    optimizer selected and observed, and the live path counts it toward the GT
    evaluation budget.  Resume must use the same definition or cache hits make
    a restarted run evaluate extra candidates and change its trajectory.
    """
    return sum(
        1 for evaluation in evaluations
        if evaluation.get("eval_type") == EvalType.GT
        and not evaluation.get("free")
    )


def _gen_fp_dedup_allowed(config: dict) -> bool:
    """True iff the config requests gen_fingerprint_dedup AND the optimizer is
    one of the method families. Raises on a baseline requesting it."""
    if not ((config.get("global") or {}).get("gen_fingerprint_dedup")):
        return False
    opt_type = str(((config.get("optimizer") or {}).get("type")) or "")
    if opt_type not in _GEN_FP_ALLOWED_OPTIMIZERS:
        raise ValueError(
            "gen_fingerprint_dedup is a METHOD feature (allowed optimizers: "
            f"{sorted(_GEN_FP_ALLOWED_OPTIMIZERS)}); optimizer type "
            f"{opt_type!r} is a baseline and must not run with it. Remove the "
            "flag from this config."
        )
    return True


def _parse_checkpoint_csv_cell(val: str):
    """Cast an all_evaluations.csv cell back to its live type. Numeric
    strings → int/float. Section columns (``pipeline__sequential``,
    ``corpus__chunker``, …) hold a whole subtree as a python-repr dict/list
    string — parse it back, or decode_to_nested splices a STRING where a
    dict belongs and every dim under it silently falls back to defaults on
    re-encode (warm_start key collapse: 28 distinct evals → 5 keys, poisoning
    EVERY optimizer's resume). Unparseable blobs pass through unchanged."""
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    if val[:1] in "{[":
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
    return val


def _restore_checkpoint_selector_choices(
    decoded: dict,
    pipeline_config: dict,
    selector_paths: set[tuple[str, ...]],
) -> dict:
    """Repair selector choices lost by historical enriched CSV rows.

    ``all_evaluations.csv`` stores a two-deep, human-readable config.  Older
    rows could contain the *declaration* of a dict-form module selector (for
    example ``vectordb: {ivf: ..., hnsw: ...}``) where the live optimizer arm
    had contained the selected scalar family.  Re-encoding such a row leaves
    the selector at its dummy default, which then marks the selected family's
    parameters inactive and collapses distinct checkpoint trials.

    The per-eval ``pipeline_config.json`` is the authoritative resolved runtime
    record.  Recover any dict-form selector from the selected runtime module
    with the same stage/key.  Ambiguous active selectors fail closed rather
    than silently choosing the first family.  Disabled/absent optional stages
    are left alone because their parameters are inactive by construction.
    """
    restored = deepcopy(decoded)
    pipeline_tree = restored.get("pipeline")
    if not isinstance(pipeline_tree, dict):
        return restored

    runtime_mode = (pipeline_config.get("pipeline_runtime") or {}).get("mode")
    if not isinstance(runtime_mode, str):
        # Single-method historical artifacts may omit pipeline_runtime.  Only
        # infer when there is exactly one method; otherwise selection is
        # genuinely ambiguous and must remain fail-closed below.
        if len(pipeline_tree) == 1:
            runtime_mode = str(next(iter(pipeline_tree)))
        else:
            return restored
    method_config = pipeline_tree.get(runtime_mode)
    if not isinstance(method_config, dict):
        return restored

    runtime_by_stage: dict[str, list[tuple[dict, dict]]] = {}
    for node_line in pipeline_config.get("node_lines") or []:
        if not isinstance(node_line, dict):
            continue
        for node in node_line.get("nodes") or []:
            if not isinstance(node, dict) or not isinstance(node.get("stage"), str):
                continue
            modules = [m for m in (node.get("modules") or []) if isinstance(m, dict)]
            for module in modules:
                runtime_by_stage.setdefault(node["stage"], []).append((node, module))

    for stage, stage_config in method_config.items():
        if not isinstance(stage_config, dict):
            continue
        selector_specs = {
            key: value
            for key, value in stage_config.items()
            if isinstance(value, dict)
            and "range" not in value
            and "ordered" not in value
            and value
            and ("pipeline", runtime_mode, stage, key) in selector_paths
        }
        if not selector_specs:
            continue

        entries = runtime_by_stage.get(stage, [])
        selected_component = stage_config.get("component")
        if isinstance(selected_component, str):
            component_entries = [
                entry for entry in entries
                if entry[1].get("component") == selected_component
            ]
            if component_entries:
                entries = component_entries

        for key, declaration in selector_specs.items():
            choices = {str(choice) for choice in declaration}
            selected_values: set[str] = set()
            for runtime_node, runtime_module in entries:
                for owner in (runtime_module, runtime_node):
                    selected = owner.get(key)
                    if isinstance(selected, str) and selected in choices:
                        selected_values.add(selected)
            if len(selected_values) == 1:
                stage_config[key] = selected_values.pop()
                continue
            if (
                len(choices) > 1
                and str(stage_config.get("enabled", "true")).lower() != "false"
            ):
                raise ValueError(
                    "checkpoint selector choice cannot be recovered for "
                    f"pipeline.{runtime_mode}.{stage}.{key}: "
                    f"declared={sorted(choices)}, runtime={sorted(selected_values)}"
                )

    return restored


def _format_exception(exc: BaseException) -> str:
    msg = str(exc)
    if msg:
        return f"{type(exc).__name__}: {msg}"
    return f"{type(exc).__name__}: <no message>"


def _eval_retry_decision(exc: BaseException) -> "tuple[EvalType, bool]":
    """(failure EvalType, retryable?) — the UNIFIED retry policy for every eval
    arm (measured / GT static / CM all key off this one function):

      * ``TrialInvalid`` (budget/launch-infeasible deployment) — deterministic,
        never retried.
      * ``OOM`` / ``CONTEXT_OVERFLOW`` (classified from the message) —
        deterministic properties of the sampled config, never retried; retrying
        re-burns a full real eval to fail identically. (The GT-static call
        site overrides OOM to retryable-then-abort: in cost_model mode the
        quality eval's deployment is the harness's own, so its OOM is an
        infrastructure failure — never an INVALID verdict on the config.)
      * judge-unavailable (out-of-credits 402 / "judge failed on ALL test
        cases" raised by the deepeval wrapper) — non-retryable: a retry
        re-burns a full measured eval (vLLM relaunch + whole pipeline) for the
        same outcome; the NEXT eval probes the judge fresh anyway.
      * ``INVALID`` (everything else) — may be transient (vLLM spawn races,
        isolated LLM-judge timeouts, retrieval flakes) → retried with backoff.
        The CM arm is analytically deterministic so it runs once regardless.
    """
    et = EvalType.classify_failure(exc)
    from rag_stack.search_space.deployment_presets import DeploymentPresetError
    if isinstance(exc, DeploymentPresetError):
        # Unfittable preset for this pipeline shape — deterministic, the
        # cost-model analog of TrialInvalid.
        return et, False
    try:
        from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import TrialInvalid
        if isinstance(exc, TrialInvalid):
            return et, False
    except ImportError:  # measured deps absent — type check is moot
        pass
    if _is_judge_outage(exc):
        return et, False
    return et, et.is_retryable_failure


def _classify_eval_arm_failures(
    gt_error: object | None,
    cm_error: object | None,
) -> "tuple[EvalType, bool, bool, bool]":
    """Classify independent GT/CM failures without losing their provenance.

    Returns ``(outcome_type, gt_genuine, cm_genuine, excluded)``.  Only a GT
    context overflow and a CM memory infeasibility are genuine sampled-config
    properties.  Any other failure makes the row ``INVALID`` even if the
    other arm also had a genuine failure, so a bug can never be hidden behind
    an allowed penalty category.
    """
    gt_genuine = (
        gt_error is not None
        and EvalType.classify_failure(gt_error) is EvalType.CONTEXT_OVERFLOW
    )
    cm_genuine = (
        cm_error is not None
        and EvalType.classify_cm_failure(cm_error) is EvalType.OOM
    )
    excluded = (
        (gt_error is not None and not gt_genuine)
        or (cm_error is not None and not cm_genuine)
    )
    if excluded:
        outcome_type = EvalType.INVALID
    elif gt_genuine:
        outcome_type = EvalType.CONTEXT_OVERFLOW
    elif cm_genuine:
        outcome_type = EvalType.OOM
    else:
        outcome_type = EvalType.INVALID
    return outcome_type, gt_genuine, cm_genuine, excluded


def _invalid_record_is_config_verdict(record: object) -> bool:
    """Whether a persisted invalid record proves config infeasibility.

    ``invalid_configs.json`` predates the provenance-aware failure taxonomy, so
    resumed projects can contain old GT/judge or measured-harness failures in
    that file.  Those rows remain useful in ``all_evaluations.csv`` for audit,
    but must not rebuild either the proposal blacklist or the agent's
    infeasible-advisory card.  Keep this predicate shared by both consumers so
    the two cannot disagree again.

    Unprefixed records are resolver/chunk/preset failures produced before an
    eval arm starts and retain the historical deterministic-verdict semantics.
    """
    if not isinstance(record, dict):
        return False
    persisted = record.get("config_verdict")
    if isinstance(persisted, bool):
        return persisted

    # Legacy compatibility only.  New records persist the verdict explicitly:
    # a typed CMOutOfMemoryError cannot be reconstructed faithfully from its
    # formatted string on resume.
    reason = record.get("reason")
    gt_part = cm_part = measured_part = None
    for segment in str(reason or "").split(" | "):
        if segment.startswith("GT: "):
            gt_part = segment[4:]
        elif segment.startswith("CM: "):
            cm_part = segment[4:]
        elif segment.startswith("MEASURED: "):
            measured_part = segment[len("MEASURED: "):]

    if measured_part is not None:
        measured_type, _, _ = _measured_failure_penalty(
            measured_part, eval_gt=False,
        )
        return measured_type in (EvalType.OOM, EvalType.CONTEXT_OVERFLOW)
    if gt_part is not None or cm_part is not None:
        _, _, _, excluded = _classify_eval_arm_failures(gt_part, cm_part)
        return not excluded
    return True


def _checkpoint_next_eval_id(
    eval_dir_names: Sequence[object],
    rows: Sequence[dict],
) -> int:
    """Return the monotonic raw-evaluation high watermark for resume.

    Failure-only projects have no optimizer observations, but their audit IDs
    are still committed.  Derive the next ID from both ledgers before any
    no-valid-observation early return so a resumed init cannot overwrite an
    existing ``eval_NNNN`` directory or CSV row.
    """
    ids: list[int] = []
    for dirname in eval_dir_names:
        text = str(dirname)
        if not text.startswith("eval_"):
            continue
        try:
            ids.append(int(text[len("eval_"):]))
        except ValueError:
            pass
    for row in rows:
        try:
            ids.append(int(row.get("eval_id", -1)))
        except (TypeError, ValueError):
            pass
    return max(ids, default=-1) + 1


def _checkpoint_penalty_row_is_replayable(row: dict) -> bool:
    """Gate penalty rows by both category and failure provenance."""
    eval_type = str(row.get("eval_type") or "")
    error = str(row.get("eval_error") or "").lstrip()
    if eval_type == EvalType.INVALID.value:
        # An invalid sampled deployment is a deterministic search-space
        # verdict and is live-fed as a zero-objective observation. Other
        # INVALID rows are infrastructure/code failures and remain excluded.
        return error.startswith("deployment_preset:")
    if eval_type not in EvalType.reload_as_penalty():
        return False
    # Static-GT harness OOM was historically mislabeled OOM.  It is never a
    # sampled-config property, even if an old artifact happened to persist
    # zero objective cells.  Genuine GT context overflow remains replayable.
    if eval_type == EvalType.OOM.value and error.startswith("GT: "):
        return False
    return True


def _measured_failure_penalty(
    error: object,
    *,
    eval_gt: bool,
    metric_names: "Sequence[str]" = (),
) -> "tuple[EvalType, dict, dict | None]":
    """Classify a measured-evaluator failure and build its GP observation.

    A measured run executes the *sampled deployment* itself, so a CUDA/launch
    OOM there is a real property of that deployment (unlike an OOM in the
    static quality harness used by ``performance_source=cost_model``).  Context
    overflow is likewise a deterministic config property.  Those two outcomes
    receive exact zero objectives and remain budget-free; every other measured
    failure is an ``INVALID`` bug/infrastructure row and must not be fed to the
    optimizer.

    Returns ``(eval_type, persisted_objectives, quality_score_for_optimizer)``.
    ``quality_score_for_optimizer`` is ``None`` for perf-only evaluations and
    for INVALID failures.
    """
    failure_type = EvalType.classify_failure(error)
    if failure_type not in (EvalType.OOM, EvalType.CONTEXT_OVERFLOW):
        return EvalType.INVALID, {}, None

    objectives: dict = {"performance": 0.0}
    quality_score = None
    if eval_gt:
        quality_score = {
            "combined": 0.0,
            **{name: 0.0 for name in metric_names if name},
        }
        objectives["quality"] = quality_score
    return failure_type, objectives, quality_score


def _perf_attr_from_best(assembly_result) -> "Optional[dict]":
    """Return only authoritative DES observations for Agent diagnostics.

    The selected row retains analytical ``qps_bound``/``qps_<stage>`` columns
    after closed-loop DES overwrites the final QPS.  Combining those two
    sources produced impossible cards (DES total QPS above an alleged stage
    capacity) and, more seriously, false prefill/decode diagnoses.  Keep the
    advisory contract strict: expose raw DES observations, preserve concrete
    engine role identity, and omit attribution when DES evidence is absent.

    These observations are deliberately *not* converted into a categorical
    bottleneck.  Engine fill is resident occupancy/admission pressure, stage
    batch is an achieved batch size, and transfer wait is observed queueing.
    None is a fitted coefficient or an independent stage-capacity estimate.
    """
    try:
        # Live evaluation passes an AssemblyResult, while checkpoint restore
        # reads the already-persisted ``best_hw_params`` mapping.  Both paths
        # must produce the exact same Agent card; a second restore-only parser
        # previously renamed ``bound`` to ``qps_bound`` and exposed internal
        # capacity fields as fake stages.
        if isinstance(assembly_result, dict):
            b = assembly_result.get("best_hw_params", assembly_result)
        else:
            b = getattr(assembly_result, "best_hw_params", None)
        if b is None:
            return None
        if not isinstance(b, dict):
            b = dict(b)

        if (
            b.get("dynamic_performance_model") != "closed_loop_simulator"
            or b.get("simulator_status") != "ok"
        ):
            return None

        qps = float(b.get("simulator_qps"))
        latency_s = float(b.get("simulator_latency_mean_s"))
        if not math.isfinite(qps) or qps <= 0.0:
            return None
        if not math.isfinite(latency_s) or latency_s < 0.0:
            return None

        def _float_map(value: object) -> dict[str, float]:
            if not isinstance(value, dict):
                return {}
            result: dict[str, float] = {}
            for key, raw in value.items():
                try:
                    number = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(number):
                    result[str(key)] = number
            return result

        fills = _float_map(b.get("simulator_engine_fill_mean"))
        stage_batches = _float_map(b.get("simulator_stage_batch_mean"))
        transfer_latency = _float_map(
            b.get("simulator_transfer_latency_mean")
        )
        transfer_wait = _float_map(
            b.get("simulator_transfer_queue_wait_mean")
        )

        layout = b.get("candidate_system_layout")
        if isinstance(layout, str):
            try:
                layout = json.loads(layout)
            except (TypeError, ValueError, json.JSONDecodeError):
                layout = {}
        if not isinstance(layout, dict):
            layout = {}
        layout_engines = layout.get("engines")
        if not isinstance(layout_engines, dict):
            layout_engines = {}

        engine_names = set(layout_engines)
        engine_names.update(key.split(".", 1)[0] for key in fills if "." in key)
        engines: dict[str, dict] = {}
        for engine_name in sorted(engine_names):
            spec = layout_engines.get(engine_name)
            if not isinstance(spec, dict):
                spec = {}
            replicas = {
                key: {"fill_fraction": value}
                for key, value in sorted(fills.items())
                if key.startswith(f"{engine_name}.")
            }
            if not replicas:
                continue
            roles = {
                key.split(".")[1]
                for key in replicas
                if len(key.split(".")) >= 3
            }
            if roles == {"unified"}:
                identifiability = "unified_only"
            elif roles and roles.issubset({"prefill", "decode"}):
                identifiability = "per_role"
            else:
                identifiability = "engine_only"
            engines[str(engine_name)] = {
                "pd_serving": str(spec.get("pd_serving") or "unknown"),
                "identifiability": identifiability,
                "replicas": replicas,
            }

        workers = {
            name: {"batch_mean": value}
            for name, value in sorted(stage_batches.items())
        }
        transfers = {
            name: {
                **(
                    {"latency_mean_s": transfer_latency[name]}
                    if name in transfer_latency else {}
                ),
                **(
                    {"queue_wait_mean_s": transfer_wait[name]}
                    if name in transfer_wait else {}
                ),
            }
            for name in sorted(set(transfer_latency) | set(transfer_wait))
        }

        populations = b.get("simulator_populations_tried")
        if not isinstance(populations, (list, tuple)):
            populations = []
        return {
            "source": "authoritative_des",
            "qps": qps,
            "latency_s": latency_s,
            "deployment": {
                "policy": str(b.get("placement_policy") or "unknown"),
                "chips": int(b.get("num_chips_e2e") or 0),
                "batch_request": int(b.get("batch_size_request") or 0),
                "batch_decode": int(b.get("batch_size_decode") or 0),
            },
            "saturation": {
                "plateau": bool(b.get("simulator_saturation_plateau")),
                "stop_reason": str(
                    b.get("simulator_saturation_stop_reason") or "unknown"
                ),
                "population": int(b.get("load_concurrency") or 0),
                "populations_tried": [int(value) for value in populations],
            },
            "engines": engines,
            "workers": workers,
            "transfers": transfers,
        }
    except Exception:  # noqa: BLE001 — attribution is advisory, never blocks
        return None


def _portable_assembly_result(assembly_result: AssemblyResult) -> AssemblyResult:
    """Drop sweep-sized payloads after an evaluation has been persisted.

    System transfer can reprice dozens of source observations, and one CM
    result may own tens of thousands of rows across ``rago_result_df``,
    ``sweep_dfs`` and ``swept_configs``.  The durable evaluation directory is
    the owner of those audit payloads.  In memory we only need the selected
    deployment for performance attribution and the scalar objective.

    Returning the same public result type keeps downstream count/summary
    contracts intact while making the absence of the heavyweight fields
    explicit rather than relying on ad-hoc ``None`` sentinels.
    """
    return AssemblyResult(
        performance_score=float(assembly_result.performance_score),
        best_hw_params=deepcopy(assembly_result.best_hw_params or {}),
        rago_result_df=None,
        sweep_dfs=None,
        swept_configs=None,
    )


def _is_judge_outage(exc: BaseException) -> bool:
    """Systemic LLM-judge unavailability — NOT a property of the sampled config.

    Two signatures: an out-of-credits 402 (deepseek "insufficient balance" /
    openrouter "insufficient credits") and the deepeval wrapper's
    "judge failed on ALL test cases" guard. Unlike a per-config failure this
    recurs identically on every subsequent eval and silently zeros quality —
    the 06-10 disaster where 44/50 evals' quality was quietly nulled. So it is
    both non-retryable AND a hard stop (see ``_should_abort_eval_failure``).
    """
    msg = str(exc).lower()
    return (
        "insufficient balance" in msg       # deepseek 402
        or "insufficient credits" in msg    # openrouter 402
        or "judge failed on all" in msg     # deepeval wrapper all-errored guard
    )


from rag_stack.security import (
    strip_secrets as _strip_secrets,
    safe_json_dump,
    safe_yaml_dump,
    safe_csv_row,
)

_MAX_EVAL_RETRIES = 3
_CM_OVERLAP_TIMEOUT_S = 3600.0
_FAIL_FAST_EVAL_ERRORS_ENV = "RAG_STACK_FAIL_FAST_EVAL_ERRORS"
_SYSTEM_TRANSFER_STATE_FILENAME = "system_transfer_state.json"
_TRANSFER_REFINEMENT_MARKER_FILENAME = "transfer_refinement.json"
_TRANSFER_ACTION_RECOMPUTED = "cm_v1_dynamic_recomputed"
_TRANSFER_ACTION_REUSED = "validated_prior_target_reuse"


def _canonical_payload_sha256(payload: Any) -> str:
    encoded = json.dumps(
        _strip_secrets(payload), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class OverlappedCMTimeoutError(TimeoutError):
    """The judge-overlapped cost model outlived its hard wall-clock limit.

    A timed-out overlap worker cannot be safely reused or allowed to run beside
    the next evaluation: its daemon thread may still own the shared CM process
    pool and pricing lock.  Callers therefore persist the current attempt as an
    infrastructure INVALID and abort the optimization so process teardown can
    reap the worker tree.
    """


# Default 2-objective MOO spec for the rag-stack controller pathway. Every
# YAML under configs/rag_stack/ optimizes the same two objectives with the
# same ref_points (objectives are higher-is-better; minimized-quantity
# objectives are negated upstream so 0 is the universal lower bound). The
# cost ratio (1:10) approximates per-eval wall time: CM-only evals are
# ~1× and GT evals (LLM-judged metrics) are ~10× more expensive. Optimizers
# read both fields, so this is the single source of truth — YAMLs no
# longer need to repeat the block. To override, set
# optimizer.params.objectives in YAML.
DEFAULT_OBJECTIVES: List[dict] = [
    {"objective_name": "performance", "ref_point": 0, "cost": 1},
    {"objective_name": "quality", "ref_point": 0, "cost": 10},
]


def _fail_fast_eval_errors() -> bool:
    raw = os.environ.get(_FAIL_FAST_EVAL_ERRORS_ENV, "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _should_abort_eval_failure(exc: BaseException) -> bool:
    """Whether a (non-retryable) eval failure should HALT the entire run.

    ONLY a systemic judge outage qualifies (``_is_judge_outage``): it recurs on
    every subsequent eval and silently zeros quality, so continuing just burns
    the budget building a garbage Pareto. Every per-config deterministic
    failure — OOM, CONTEXT_OVERFLOW (e.g. high top_k × large chunk_size floods
    the generator's context window), or an infeasible sampled deployment
    layout — is the search space's own signal that a corner is bad: it is
    recorded as INVALID and penalized to 0 so the optimizer learns to avoid
    that region and keeps going (CLAUDE.md: these are "deterministic and never
    retried", not fatal). ``RAG_STACK_FAIL_FAST_EVAL_ERRORS=0`` disables even
    the judge-outage stop.
    """
    if not _fail_fast_eval_errors():
        return False
    return _is_judge_outage(exc)


class Controller:
    # ----------------------------------------------------------------------
    # Construction & setup
    # ----------------------------------------------------------------------

    def __init__(
        self,
        config_path: str,
        qa_data_path: str = None,
        corpus_data_path: str = None,
        project_dir: str = "",
        dataset: Optional["GeneratedDataset"] = None,
        quality_store_path: str | None = None,
        defer_checkpoint_warm_start: bool = False,
    ):
        # Set PROJECT_DIR before loading config so ${PROJECT_DIR} expands correctly
        self.project_dir = os.path.abspath(project_dir)
        os.environ["PROJECT_DIR"] = self.project_dir

        # Project lifecycle gate: forbid starting if another process is
        # actively running on this project_dir. The source of truth is
        # whether the recorded owner PID is alive — not the status string
        # alone, which can be stale (SIGKILL leaves status=RUNNING; a
        # concurrent crash can falsely flip status to FAILED).
        with project_config_lock(self.project_dir):
            summary = load_project_summary(self.project_dir)
            if (
                summary is not None
                and summary.pid is not None
                and summary.pid != os.getpid()
                and pid_alive(summary.pid)
            ):
                raise RuntimeError(
                    f"Project '{self.project_dir}' is already being optimized "
                    f"by PID {summary.pid} (status='{summary.status.value}'). "
                    f"Wait for it to finish, or kill it before restarting."
                )
            # Bind the original YAML before loading models, copying datasets,
            # or expanding runtime values. Direct Python callers obey the same
            # ownership rule as CLI/API launches, including alternate paths.
            with open(config_path, encoding="utf-8", newline="") as stream:
                write_project_config(self.project_dir, stream.read())
            self._project_config_raw = freeze_project_config(self.project_dir)
            config_path = os.path.join(self.project_dir, CONFIG_FILENAME)

        # Build the model-free performance context: config load + validation,
        # the RagSearchSpace adapter, resume-safe corpus/QA binding (+ corpus
        # size N and token-stats injection), the base decoded config, the
        # hardware spec, and the StaticAssembly cost model — plus the
        # config-resolution coordinator (``resolve_config``) and both
        # performance sources (cost_model + measured layout derivation). This
        # is extracted into PerformanceContext so lightweight callers (e.g.
        # the historical surrogate benchmark) can build the SAME performance
        # path WITHOUT the optimizer / quality evaluator (which load models).
        # Aliased back onto ``self`` below so every public attribute name is
        # unchanged and the live path runs byte-identical code.
        self._ctx = PerformanceContext(
            config_path=config_path,
            project_dir=self.project_dir,
            qa_data_path=qa_data_path,
            corpus_data_path=corpus_data_path,
            dataset=dataset,
            validate=True,
        )
        self.config = self._ctx.config
        # Generation-fingerprint dedup (M3, 07-07): config-gated, exported as
        # an env flag because the consumer (the generator NODE) has no config
        # access. Default OFF — baselines and existing runs are byte-identical.
        # HARD-GATED to the method optimizers: this is a METHOD feature
        # ("dedup-aware budget"); a baseline configured with it is an
        # experimental-design error and fails loudly. The pop() on the OFF
        # path prevents a stale shell export from a prior method run leaking
        # the feature into a baseline launched from the same environment.
        if _gen_fp_dedup_allowed(self.config):
            os.environ["RAG_STACK_GEN_FP_DEDUP"] = "1"
            logger.info("gen_fingerprint_dedup ON: identical generation inputs "
                        "inherit the donor's quality (eval_type=GT_FP, no budget seat)")
        else:
            os.environ.pop("RAG_STACK_GEN_FP_DEDUP", None)
        self._ssops = self._ctx._ssops
        self.search_space = self._ctx.search_space
        self._data = self._ctx._data
        self._base_decoded_config = self._ctx._base_decoded_config
        self.hardware = self._ctx.hardware
        # ONE unified cost-model assembly. Static and trace-driven paths are
        # called explicitly with a WorkflowSchema; the old ambiguous payload
        # dispatch remains only as an external compatibility shim.
        self.assembly = self._ctx.assembly

        # One-time CM init: let each cost model prepare its data-side artifacts
        # (e.g. the IVF cell-imbalance profile) once, instead of per-eval.
        self._init_cost_models()

        self.optimizer_type = self.config["optimizer"]["type"]
        # Optional spec for combined-quality aggregation (see
        # ``aggregate_quality_metrics`` docstring). ``None`` = legacy mean
        # over all non-None sub-metrics.
        self._combined_quality_spec = (
            (self.config.get("eval_backend_setting") or {}).get("combined_quality")
        )
        # Keep composite-objective metadata in the resolved project config for
        # expanded CSV columns. It is not an optimizer
        # constructor argument: current optimizers consume the objective values
        # passed to add_objectives, not this duplicate metadata block.
        optimizer_params = dict(self.config["optimizer"].get("params", {}))
        # Inject DEFAULT_OBJECTIVES when YAML omits the block (it never
        # varies across rag-stack runs — see comment on DEFAULT_OBJECTIVES).
        if "objectives" not in optimizer_params:
            optimizer_params["objectives"] = deepcopy(DEFAULT_OBJECTIVES)
            self.config.setdefault("optimizer", {}).setdefault("params", {})[
                "objectives"
            ] = deepcopy(DEFAULT_OBJECTIVES)
        if "composite_objectives" not in optimizer_params:
            derived = self._ssops.derive_composite_objectives(
                self.config.get("eval_backend_setting"),
            )
            if derived:
                self.config.setdefault("optimizer", {}).setdefault("params", {})[
                    "composite_objectives"
                ] = derived
        optimizer_params.pop("composite_objectives", None)
        optimizer_class = OPTIMIZER_REGISTRY[self.optimizer_type]
        constructor_parameters = inspect.signature(
            optimizer_class.__init__,
        ).parameters
        # Only budget-aware optimizers receive the run budget. Passing generic
        # kwargs to every implementation previously required silent **kwargs
        # sinks and let misspelled/dead optimizer controls appear to work.
        if "n_iterations" in constructor_parameters:
            optimizer_params.setdefault(
                "n_iterations",
                int((self.config.get("global") or {}).get("n_iterations", 100)),
            )
        # agent_qnehvi_slo: suggestions are pre-screened by the static cost
        # model against the run's constrained SLO. Validate the run is
        # actually configured for that (fail-fast) and inject the CM probe —
        # the optimizer itself stays cost-model-agnostic behind the callable.
        if self.optimizer_type == "agent_qnehvi_slo":
            self._validate_slo_optimizer_setup()
            optimizer_params.setdefault("cm_probe", self._ctx.cm_performance_probe)
        # Inject our RagSearchSpace so the optimizer's internal flat→nested
        # decode produces RAG-namespace-lifted output (vectordb__<vdb>__<k>
        # → config[<vdb>][<k>], not config["vectordb"][<k>]) — controller's
        # `_apply_overrides` looks up `decoded[vdb_name]` and expects that
        # exact shape. Optimizers fall back to generic AxSearchSpaceOps
        # when ssops is None (benchmark / synthetic-function path).
        self.optimizer = optimizer_class(
            self.search_space,
            **optimizer_params,
            ssops=self._ssops,
        )
        # Resume-sensitive DOE and forced-channel state lives beside the
        # project checkpoint. Install the
        # path before `_load_checkpoint()` replays observations; setting it only
        # in `get_pareto()` is too late to reconstruct per-row forced history.
        try:
            setattr(self.optimizer, "_advice_log_dir", self.project_dir)
        except Exception:  # noqa: BLE001 - optional optimizer instrumentation
            pass

        # Build GeneratedDataset for the quality evaluator (consumed by both
        # backends through the same constructor signature). ``self._data`` and
        # ``self.config`` are the cost-model context's objects, aliased above.
        from rag_stack.dataset_generator.schema import Raw, Corpus, QA
        from rag_stack.dataset_generator.base import GeneratedDataset
        _dataset = dataset if dataset is not None else GeneratedDataset(
            raw=Raw(), corpus=Corpus(self._data.corpus_data), qa=QA(self._data.qa_data),
            strategy_name="from_parquet",
        )
        # Backend selection: default is the static evaluator (byte-identical
        # to pre-FlashRAG behavior); ``global.eval_backend: flashrag``
        # opts into the FlashRAG-backed evaluator via lazy import so
        # static-only users never pull FlashRAG into their import graph.
        self.rag_quality_evaluator = self._make_quality_evaluator(_dataset)
        self._assembly_results: List[AssemblyResult] = []

        # Performance source — owned by the PerformanceContext. Default
        # "cost_model" → byte-identical legacy path (RAGO predicts
        # performance). "measured" → real on-hardware timing via the static
        # evaluator + a run-spanning MeasuredProvider (owns the ModelCache +
        # vLLM deployment). The cost-model code path is untouched in either
        # mode; measured adds a self-contained branch in _evaluate_objectives,
        # gated on self._measured_provider being set (None under cost_model).
        self.performance_source = self._ctx.performance_source
        self.system_space_owner = self._ctx.system_space_owner
        self._measured_provider = self._ctx.make_measured_provider(
            self.rag_quality_evaluator
        )
        # Joint-space cell (owner=optimizer + cost_model): CM prices exactly
        # ONE sampled deployment per trial. None everywhere else — all its
        # gates short-circuit, keeping cm-swept and measured paths untouched.
        self._cm_fixed_provider = self._ctx.make_cm_fixed_provider()

        # Incremental experiment tracking
        self._eval_counter: int = 0
        # per-run tally of non-success eval_types written to disk (INVALID /
        # DUPLICATE / OOM / CONTEXT_OVERFLOW / ...) — warned per occurrence
        # and repeated in the completion summary.
        self._failure_counts: Dict[str, int] = {}
        self._invalid_configs: List[dict] = []
        self._invalid_config_keys: set = set()

        # Cumulative counters across resumes. ``_total_cost`` is per-run
        # (zeroed in ``get_pareto`` regardless of checkpoint) — under the
        # topo-decoupled flow it tracks fake/synthetic budget anyway, so
        # carrying it across resumes would be misleading.
        # ``_cm_count`` / ``_gt_count`` are real evaluation counts and are
        # restored from ``summary.json`` on resume so progress dashboards
        # show "X cm + Y gt across the project's lifetime".
        self._cm_count: int = 0
        self._gt_count: int = 0

        # Quality memoization: {algo-hash → QualityMemoEntry}. Quality is
        # deployment-independent, so trials that differ only in system dims
        # (deployment_preset / batch / gpu_layout) reuse the first trial's
        # judge scores + recorded trace instead of re-burning the GPU+judge
        # run. Seeded from persisted eval records on resume. ``_gt_memo_hits``
        # counts reuses (unique judge runs = _gt_count − _gt_memo_hits).
        # EXPLICITLY DISABLED under performance_source=measured: there quality
        # and performance are co-produced by one real serving run (and the
        # measured branch never reaches the memo code anyway) — a reused
        # quality answer would decouple the two halves of the same run. This
        # flag makes that policy visible instead of relying on control flow.
        self._quality_memo_enabled: bool = self._measured_provider is None
        self._quality_memo: dict = {}
        self._gt_memo_hits: int = 0
        self._external_quality_store: QualityStore | None = None

        # Checkpoint resume: load existing evaluations if the project was
        # previously run. A SIGKILL/crash can leave summary.status="running";
        # after the owner PID is gone, treat that as a stale-running checkpoint
        # instead of restarting from eval_0000 and overwriting prior results.
        self._checkpoint_loaded = False
        stale_running = (
            summary is not None
            and summary.status == ProjectStatus.RUNNING
            and summary.pid is not None
            and not pid_alive(summary.pid)
        )
        if (
            summary is not None
            and (
                summary.status in (ProjectStatus.COMPLETED, ProjectStatus.FAILED)
                or stale_running
            )
        ):
            self._load_checkpoint(warm_start=not defer_checkpoint_warm_start)

        if quality_store_path is not None:
            self._attach_quality_store(quality_store_path)

        # Runtime enrichment (environment expansion, token statistics, and
        # optimizer defaults) is complete. Trial resolution works on copies;
        # the controller's effective input must remain stable across trials.
        self._project_runtime_config = deepcopy(self.config)

    def _quality_algo_config(self, decoded_config: dict) -> dict:
        """Canonical full algorithm config used by local and global caches."""
        enriched = self._ssops.enrich_configs(
            [decoded_config], self._base_decoded_config,
        )[0]
        return QualityStore.algorithm_config(enriched)

    def _quality_algo_key(self, decoded_config: dict) -> str:
        return QualityStore.key_for_config(
            self._quality_algo_config(decoded_config),
        )

    def _attach_quality_store(self, store_path: str) -> None:
        """Seed the project memo from a dataset-compatible canonical store."""
        if not self._quality_memo_enabled:
            raise ValueError(
                "A global QualityStore cannot be attached in measured mode; "
                "measured quality and performance are co-produced."
            )
        store = QualityStore.load(store_path)
        if not store.requires_quality_trace:
            raise ValueError(
                "Normal optimization requires a canonical QualityStore with quality traces"
            )
        dataset_cfg = self.config.get("dataset") or {}
        qa_path = dataset_cfg.get("qa")
        corpus_path = dataset_cfg.get("corpus")
        if not qa_path or not os.path.isfile(os.path.expanduser(str(qa_path))):
            qa_path = os.path.join(self.project_dir, "data", "qa.parquet")
        if not corpus_path or not os.path.isfile(os.path.expanduser(str(corpus_path))):
            raw_path = os.path.join(self.project_dir, "data", "raw_corpus.parquet")
            corpus_path = raw_path if os.path.isfile(raw_path) else os.path.join(
                self.project_dir, "data", "original_corpus.parquet",
            )
        store.assert_dataset_compatible(
            self.config,
            qa_data_path=qa_path,
            corpus_data_path=corpus_path,
        )

        configured_metrics = [
            metric.get("metric_name") or metric.get("metric")
            for metric in (self.config.get("eval_backend_setting") or {}).get(
                "metrics", []
            )
            if isinstance(metric, dict)
            and (metric.get("metric_name") or metric.get("metric"))
        ]
        objective_metrics = list(
            ((self.config.get("eval_backend_setting") or {}).get(
                "combined_quality"
            ) or {}).get("metrics") or []
        )
        active_quality_spec = QualityStore.quality_spec(self.config)
        from rag_stack.quality_memo import QualityMemoEntry

        seeded = 0
        incompatible_quality_specs = 0
        for record in store.get_quality_results():
            record_quality_spec = (
                (record.get("record") or {}).get("quality_spec")
            )
            if not QualityStore.quality_specs_match(
                record_quality_spec, active_quality_spec,
            ):
                # A metric name alone is not identity: judge model, prompt,
                # threshold and aggregation semantics can all change its value.
                # Keep such records in the dataset store, but never expose them
                # to this optimizer run.
                incompatible_quality_specs += 1
                continue
            metrics = record.get("quality_metrics") or {}
            if objective_metrics and not all(
                metrics.get(name) is not None for name in objective_metrics
            ):
                continue
            key = record["config_hash"]
            full = all(metrics.get(name) is not None for name in configured_metrics)
            previous = self._quality_memo.get(key)
            if previous is not None:
                # Current-project checkpoint data wins. The global cache may
                # only upgrade a local partial entry to a full one.
                if previous.full or not full:
                    continue
            trace_path = record.get("execution_dag_path") or os.path.join(
                store.store_path, "traces", f"{key}.json",
            )
            self._quality_memo[key] = QualityMemoEntry(
                quality_metrics=dict(metrics),
                execution_dag=None,
                full=full,
                execution_dag_path=trace_path,
            )
            seeded += 1
        self._external_quality_store = store
        logger.info(
            "Attached exact-config QualityStore %s: indexed %d compatible "
            "quality+trace record(s), quality_spec_mismatch=%d, "
            "cache_entries=%d; optimizer history and "
            "candidate generation are unchanged",
            store.store_path, seeded, incompatible_quality_specs,
            len(self._quality_memo),
        )

    def _validate_slo_optimizer_setup(self):
        """Fail fast on a mis-configured ``agent_qnehvi_slo`` run.

        The SLO-gated optimizer replays the run's OWN selection policy at
        suggest time, so it only makes sense when:

        * ``system.performance_objectives`` defines a constrained SLO
          (``selection: constrained`` + ``constraint`` + ``optimize``) —
          without it the CM never emits the INFEASIBLE_SCORE sentinel and
          the gate is vacuous (note ``select_constrained`` silently falls
          back to min_latency when ``constraint``/``optimize`` is missing);
        * ``performance_source: cost_model`` — the probe is a STATIC CM
          prediction; gating measured-performance runs on it would enforce
          a different metric than the one being optimized;
        * ``global.eval_backend: static_gt`` — iterative/flashrag runs
          cost performance via rag_ir trace replay, and no trace exists at
          suggest time, so the static probe would disagree with the run's
          actual performance path.
        """
        sysblk = self.config.get("system") or {}
        po = sysblk.get("performance_objectives") or {}
        if (
            po.get("selection") != "constrained"
            or not po.get("constraint")
            or not po.get("optimize")
        ):
            raise ValueError(
                "optimizer.type=agent_qnehvi_slo requires an SLO in "
                "system.performance_objectives: selection: constrained, "
                "constraint: {<metric>: <threshold>, ...} (natural sense — "
                "latency_s ≤ L*, throughput_qps ≥ Q*) and optimize: <metric>. "
                f"Got: {po!r}"
            )
        perf_src = str(sysblk.get("performance_source", "cost_model")).lower()
        if perf_src != "cost_model":
            raise ValueError(
                "agent_qnehvi_slo pre-screens candidates with the STATIC cost "
                "model and is only valid with performance_source: cost_model "
                f"(got {perf_src!r})."
            )
        from rag_stack.performance_context import (
            SYSTEM_SPACE_OWNER_COST_MODEL, resolve_system_space_owner,
        )
        owner = resolve_system_space_owner(self.config)
        if owner != SYSTEM_SPACE_OWNER_COST_MODEL:
            raise ValueError(
                "agent_qnehvi_slo's suggest-time probe is a FULL-SWEEP CM "
                "prediction (best deployment per candidate); under "
                "system_space_owner: optimizer each trial is priced at ONE "
                "sampled deployment, so the probe would gate on a different "
                f"metric. Got system_space_owner={owner!r}."
            )
        backend = (self.config.get("global") or {}).get("eval_backend") or "static_gt"
        if backend != "static_gt":
            raise ValueError(
                "agent_qnehvi_slo is only supported with the static_gt quality "
                f"backend (got {backend!r}): iterative/flashrag runs cost "
                "performance via rag_ir trace replay, which does not exist at "
                "suggest time."
            )

    def _run_phase_1(self, optimizer, obj_columns) -> None:
        """Both-objective init evals (Phase 1), chunk-aware (07-08).

        Optimizers exposing ``init_chunks`` — a list of REMAINING init batch
        sizes recomputed from their own history — get an evaluate+feed-back
        barrier between batches: the sobol-first agent init uses this so its
        LLM probe designer receives the first batch's observed results.
        Optimizers without the attribute run the classic single batch of
        ``n_both_init``, bit-identically to the pre-chunking controller.
        """
        _chunked = getattr(optimizer, "init_chunks", None) is not None
        _rounds = 0
        while True:
            _rounds += 1
            if _chunked:
                chunks = list(optimizer.init_chunks or [])
                if not chunks:
                    break  # init complete (immediately true when resumed past init)
                n_ask = int(chunks[0])
            else:
                n_ask = int(optimizer.n_both_init)
            eval_id_start = self._eval_counter
            configs, eval_obj_names = optimizer.suggest(n_suggestions=n_ask)
            if configs is None or eval_obj_names is None:
                raise ValueError("Optimizer failed to generate initial candidates.")
            logger.info(
                f"[progress] Phase 1 both-objective init start: "
                f"{len(configs)} candidate(s), objectives={eval_obj_names}, "
                f"eval_ids={eval_id_start}-{eval_id_start + len(configs) - 1}"
            )
            obj_dict, valid_configs, n_success, free_flags = self._evaluate_objectives(
                configs, eval_obj_names, obj_columns)
            if valid_configs:
                eids = list(self._last_valid_eval_ids)
                if len(eids) != len(valid_configs):
                    raise RuntimeError(
                        "Phase 1 evaluator returned misaligned valid eval ids: "
                        f"configs={len(valid_configs)}, ids={len(eids)}"
                    )
                try:
                    optimizer.add_objectives(
                        valid_configs, obj_dict, eval_ids=eids,
                        free_mask=free_flags if any(free_flags) else None,
                    )
                except TypeError:
                    optimizer.add_objectives(valid_configs, obj_dict, eval_ids=eids)
                self._push_perf_attribution(optimizer, eids)

            self._total_cost += sum(optimizer.objective_costs[n] for n in eval_obj_names) * len(valid_configs)
            # valid-eval counting (07-06): only SUCCESS evals consume the GT
            # budget; genuine-infeasible penalty rows fed the GP above but keep
            # their budget seat available.
            self._gt_count += n_success

            logger.info(
                f"Phase 1: {n_success} valid GT evals "
                f"({len(valid_configs) - n_success} penalized), cost: {self._total_cost:.2f}"
            )
            self._update_summary()
            if not _chunked:
                break  # classic optimizers: exactly one batch, as before
            if _rounds >= 8:
                logger.warning(
                    "Phase 1: chunk-round cap reached with init still "
                    "incomplete; proceeding to Phase 2 (BO tops init up)"
                )
                break

    def _load_checkpoint(self, *, warm_start: bool = True):
        """Load existing evaluations from disk and warm-start the optimizer.

        Called when resuming a completed project.  Reads per-eval directories,
        reconstructs (config, objectives) pairs, feeds them to the optimizer
        via ``warm_start()``, and sets ``_eval_counter`` so new evals get
        fresh IDs.
        """
        evals_dir = os.path.join(self.project_dir, "evaluations")
        if not os.path.isdir(evals_dir):
            logger.info("Checkpoint: no evaluations/ directory found, starting fresh")
            return

        # Reload the known-invalid registry (incrementally persisted by
        # _record_invalid) so resumed runs don't re-burn deterministic
        # INVALID arms (CONTEXT_OVERFLOW etc.).
        #
        # Re-arm the proposal blacklist ONLY for genuine config verdicts,
        # mirroring the penalize-vs-exclude taxonomy at outcome time: a
        # GT-side CONTEXT_OVERFLOW or a CM-side analytic OOM is a property of
        # the config and stays blocked forever. An INFRASTRUCTURE failure
        # (judge outage/timeouts, engine death) says nothing about the config;
        # it is blacklisted within the session that observed it, but a resume
        # happens AFTER the operator had a chance to fix the environment
        # (07-08: a frontier-grade config was permanently blocked because the
        # judge timed out on the night the fix for those timeouts shipped).
        # Records without the GT:/CM: prefixes (resolve/chunk/preset failures)
        # are deterministic and re-arm as before.
        invalid_path = os.path.join(self.project_dir, "invalid_configs.json")
        if os.path.isfile(invalid_path):
            try:
                n_skipped = 0
                with open(invalid_path) as f:
                    for rec in json.load(f):
                        cfg = rec.get("config")
                        if not isinstance(cfg, dict):
                            continue
                        if _invalid_record_is_config_verdict(rec):
                            self._invalid_configs.append(rec)
                            self._add_invalid_key(cfg)
                        else:
                            n_skipped += 1
                            logger.info(
                                f"Checkpoint: invalid record eval_{rec.get('eval_id')} "
                                "ignored by blacklist/advisory "
                                "(infrastructure audit row, not a config verdict)"
                            )
                logger.info(
                    f"Checkpoint: loaded {len(self._invalid_configs)} known-invalid "
                    f"configs; skipped {n_skipped} infrastructure audit record(s)"
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Checkpoint: invalid_configs.json load failed: {exc}")

        eval_dirs = sorted(
            d for d in os.listdir(evals_dir)
            if d.startswith("eval_") and os.path.isdir(os.path.join(evals_dir, d))
        )
        if not eval_dirs:
            logger.info("Checkpoint: evaluations/ is empty, starting fresh")
            return

        obj_names = list(self.optimizer.objective_costs.keys())
        evaluations: List[dict] = []

        for dirname in eval_dirs:
            eval_path = os.path.join(evals_dir, dirname)
            try:
                eval_id = int(dirname.replace("eval_", ""))
            except ValueError:
                continue

            # Read pipeline config to reconstruct decoded config
            pc_path = os.path.join(eval_path, "pipeline_config.json")
            if not os.path.isfile(pc_path):
                continue

            # Read assembly result for performance score
            assembly_path = os.path.join(eval_path, "assembly.json")
            performance_path = os.path.join(eval_path, "performance.json")
            quality_path = os.path.join(eval_path, "quality_metrics.json")

            objectives: Dict[str, float] = {}

            # Performance from assembly (cost_model) or performance.json (measured).
            if os.path.isfile(assembly_path):
                try:
                    with open(assembly_path) as f:
                        assembly_data = json.load(f)
                    perf = assembly_data.get("performance_score")
                    if perf is not None:
                        objectives["performance"] = perf
                except Exception:
                    pass
            elif os.path.isfile(performance_path):
                try:
                    with open(performance_path) as f:
                        perf_data = json.load(f)
                    if self._measured_provider is not None:
                        from rag_stack_evaluator.static_rag_evaluator.measured.artifact_contract import (
                            require_measured_gt_admissible,
                        )
                        require_measured_gt_admissible(
                            perf_data.get("performance")
                        )
                    perf = perf_data.get("performance_score")
                    if perf is not None:
                        objectives["performance"] = perf
                    elif self._measured_provider is not None:
                        raise ValueError(
                            "measured performance artifact has no performance_score"
                        )
                except Exception as exc:
                    if self._measured_provider is not None:
                        logger.warning(
                            "Checkpoint: excluding eval_%04d measured artifact "
                            "from optimizer warm-start: %s",
                            eval_id, _format_exception(exc),
                        )
                        continue

            # Quality from metrics — pass through as a dict so optimizers
            # with sub-metric topology (topo_gp_mobo) can rebuild per-sub_metric
            # tensors. Optimizers that only need a scalar pull `v["combined"]`.
            quality_metrics_raw = None
            if os.path.isfile(quality_path):
                try:
                    with open(quality_path) as f:
                        quality_metrics = json.load(f)
                    quality_metrics_raw = quality_metrics
                    # ``combined`` MUST be computed the SAME way as the live path
                    # (``aggregate_quality_metrics`` honouring ``combined_quality``),
                    # NOT a naive mean over all sub-metrics. With combined_quality =
                    # just answer_correctness, a mean over all 8 metrics inflates it
                    # (faithfulness/relevancy ~0.9) and the warm-started GP optimises
                    # a corrupted quality axis. Sub-metrics still pass through for
                    # topo optimizers; scalar consumers read ``combined``.
                    combined = aggregate_quality_metrics(
                        quality_metrics, self._combined_quality_spec
                    )
                    quality_value: Dict[str, float] = {"combined": combined}
                    for sm_name, sm_val in quality_metrics.items():
                        if sm_val is None:
                            continue
                        quality_value[sm_name] = float(sm_val)
                    objectives["quality"] = quality_value
                except Exception:
                    pass

            if not objectives:
                continue

            # Reconstruct decoded config from the CSV row or enriched config
            # We use all_evaluations.csv for this since it has the decoded config
            # But simpler: read from the eval dir's pipeline_config and reverse-map
            # Actually, let's use a simpler approach: store decoded configs in eval records
            # For now, read from all_evaluations.csv
            from rag_stack.quality_memo import MEMO_MARKER_FILENAME
            memo_marker = None
            try:
                with open(os.path.join(eval_path, MEMO_MARKER_FILENAME)) as _mf:
                    memo_marker = json.load(_mf)
            except Exception:  # noqa: BLE001 — records may predate the marker
                pass
            evaluations.append({
                "eval_id": eval_id,
                "objectives": objectives,
                "eval_type": EvalType.GT if "quality" in objectives else EvalType.CM,
                # Quality-memo seeding inputs (stripped before warm_start use):
                # raw metrics dict + the eval dir (its execution_dag.json is
                # lazily loaded on the first memo hit) + the marker carrying
                # the PERSISTED algo-hash (the only valid memo key on resume).
                "_eval_path": eval_path,
                "_quality_metrics_raw": quality_metrics_raw,
                "_memo_marker": memo_marker,
                "quality_memo_hit": bool((memo_marker or {}).get("hit")),
            })

        if not evaluations:
            # Do not return yet: a project may contain only budget-free OOM /
            # CONTEXT_OVERFLOW observations.  Their zero objectives live in
            # all_evaluations.csv (not performance.json/quality_metrics.json)
            # and must still be replayed into the optimizer on resume.
            logger.info(
                "Checkpoint: no successful evaluation artifacts found; "
                "checking the CSV for replayable penalty rows"
            )

        # Read decoded configs from all_evaluations.csv
        csv_path = os.path.join(self.project_dir, "all_evaluations.csv")
        if not os.path.isfile(csv_path):
            logger.info("Checkpoint: no all_evaluations.csv found, starting fresh")
            return

        import csv
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Establish the raw-ID high watermark before any early return below.
        # A project containing only infrastructure audit rows has no
        # warm-start observations, but those IDs are still immutable.
        self._eval_counter = max(
            self._eval_counter,
            _checkpoint_next_eval_id(eval_dirs, rows),
        )

        # Build eval_id → decoded config mapping from CSV rows. Column
        # decoding (``<top>__<leaf>`` 2-seg headers → nested dict) goes
        # through the shared RagSearchSpace adapter so the same naming
        # convention rules (vectordb/corpus deviations) live in one place.
        for ev in evaluations:
            eid = ev["eval_id"]
            # Find matching CSV row
            row = next((r for r in rows if r.get("eval_id") == str(eid)), None)
            if row is None:
                continue

            # Skip invalid/skipped (centralized in EvalType.excluded_from_data)
            eval_type_csv = row.get("eval_type", "")
            if eval_type_csv in EvalType.excluded_from_data():
                continue
            if eval_type_csv == EvalType.GT_FP:
                # Fingerprint-inherited row (M3): a real archive observation
                # that must NOT re-consume a budget seat after resume.
                ev["free"] = True

            flat: dict = {}
            for col, val in row.items():
                if col in ("eval_type", "eval_error", "eval_id") or not val:
                    continue
                if any(col.startswith(obj) for obj in obj_names):
                    continue
                flat[col] = _parse_checkpoint_csv_cell(val)

            decoded = self._ssops.decode_to_nested(flat)
            if decoded:
                try:
                    with open(
                        os.path.join(ev["_eval_path"], "pipeline_config.json")
                    ) as stream:
                        resolved_pipeline = json.load(stream)
                    ev["config"] = _restore_checkpoint_selector_choices(
                        decoded,
                        resolved_pipeline,
                        self._ssops.hierarchical_selector_paths(),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Checkpoint: excluding eval_%04d because its exact "
                        "selector choices could not be restored: %s",
                        eid, _format_exception(exc),
                    )

        # Replay genuine-infeasible PENALTY rows (07-06, valid-eval counting):
        # OOM / CONTEXT_OVERFLOW rows carry the exact 0-penalty objectives that
        # were fed to the GP live (persisted in the CSV objective cells since
        # 07-06; older projects saved objectives={} → cells empty → row skipped,
        # best-effort). This restores the GP's infeasibility trough across
        # resumes WITHOUT consuming valid-eval seats: loaded_gt/loaded_cm below
        # count only eval_type == GT/CM, and these evs keep their failure type.
        # Every other failure type stays fully excluded (user rule 07-06).
        _restored_ids = {ev["eval_id"] for ev in evaluations if "config" in ev}
        n_penalty_replayed = 0
        for row in rows:
            et = row.get("eval_type", "")
            if not _checkpoint_penalty_row_is_replayable(row):
                continue
            try:
                eid = int(row.get("eval_id", -1))
            except (TypeError, ValueError):
                continue
            if eid in _restored_ids:
                continue
            perf_cell = row.get("performance", "")
            qual_cell = row.get("quality__combined", "")
            if not perf_cell and not qual_cell:
                continue  # pre-07-06 penalty row: fed values were never persisted
            objectives: Dict[str, object] = {}
            if perf_cell:
                try:
                    objectives["performance"] = float(perf_cell)
                except (TypeError, ValueError):
                    objectives["performance"] = 0.0
            # Reconstruct exactly the quality objective that was live-fed.
            # A perf-only measured OOM has no quality cells and must remain
            # perf-only after resume; a multi-objective OOM keeps every
            # persisted zero sub-metric, not merely ``combined``.
            quality_objective: Dict[str, float] = {}
            for col, val in row.items():
                if not col.startswith("quality__") or not val:
                    continue
                try:
                    quality_objective[col[len("quality__"):]] = float(val)
                except (TypeError, ValueError):
                    quality_objective[col[len("quality__"):]] = 0.0
            if quality_objective:
                objectives["quality"] = quality_objective
            elif row.get("quality", ""):
                try:
                    objectives["quality"] = float(row["quality"])
                except (TypeError, ValueError):
                    objectives["quality"] = 0.0
            flat = {}
            for col, val in row.items():
                if col in ("eval_type", "eval_error", "eval_id") or not val:
                    continue
                if any(col.startswith(obj) for obj in obj_names):
                    continue
                flat[col] = _parse_checkpoint_csv_cell(val)
            decoded = self._ssops.decode_to_nested(flat)
            if not decoded:
                continue
            try:
                with open(os.path.join(
                    self.project_dir,
                    "evaluations",
                    f"eval_{eid:04d}",
                    "pipeline_config.json",
                )) as stream:
                    resolved_pipeline = json.load(stream)
                decoded = _restore_checkpoint_selector_choices(
                    decoded,
                    resolved_pipeline,
                    self._ssops.hierarchical_selector_paths(),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Checkpoint: excluding penalty eval_%04d because its "
                    "exact selector choices could not be restored: %s",
                    eid, _format_exception(exc),
                )
                continue
            evaluations.append({
                "eval_id": eid,
                "objectives": objectives,
                "eval_type": et,  # failure type → never counted as GT/CM below
                "config": decoded,
                "_eval_path": os.path.join(
                    self.project_dir,
                    "evaluations",
                    f"eval_{eid:04d}",
                ),
                # penalty rows never consumed a budget seat live — mark free so
                # post-resume phase/seat ledgers don't charge them either
                "free": True,
            })
            n_penalty_replayed += 1

        # Filter to evaluations with configs; replay in eval-id order
        valid_evals = sorted(
            (ev for ev in evaluations if "config" in ev),
            key=lambda ev: ev["eval_id"],
        )
        for ev in valid_evals:
            marker_path = os.path.join(
                ev["_eval_path"],
                _OPTIMIZER_OBSERVATION_FILENAME,
            )
            try:
                with open(marker_path) as stream:
                    metadata = json.load(stream)
                if isinstance(metadata, dict):
                    ev["optimizer_metadata"] = metadata
            except FileNotFoundError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Checkpoint: ignoring malformed optimizer metadata for "
                    "eval_%04d: %s",
                    ev["eval_id"],
                    _format_exception(exc),
                )

        if not valid_evals:
            logger.info("Checkpoint: could not reconstruct configs, starting fresh")
            return

        # Seed the quality memo from persisted records: {canonical full
        # algo-hash → metrics + lazy trace path}. CSV configs are enriched with
        # fixed algorithm semantics (chat template, prompts, fixed model knobs),
        # which is exactly the cross-project cache key. Old marker hashes were
        # computed from sparse optimizer overrides and are provenance only.
        # Later eval_ids win; a full entry is never replaced by a minimal one.
        # Measured mode: memoization is disabled outright (quality is
        # co-produced with performance by one real run) — skip seeding too.
        from rag_stack.quality_memo import QualityMemoEntry
        _configured_metrics = [
            m.get("metric_name")
            for m in (self.config.get("eval_backend_setting") or {}).get("metrics", [])
            if m.get("metric_name")
        ]
        _seeded = 0
        for ev in sorted(valid_evals, key=lambda e: e["eval_id"]):
            if not self._quality_memo_enabled:
                break
            metrics = ev.get("_quality_metrics_raw")
            if not isinstance(metrics, dict) or not metrics:
                continue
            key = self._quality_algo_key(ev["config"])
            full = (
                all(metrics.get(name) is not None for name in _configured_metrics)
                if _configured_metrics else True
            )
            prev = self._quality_memo.get(key)
            if prev is not None and prev.full and not full:
                continue
            dag_path = os.path.join(ev["_eval_path"], "execution_dag.json")
            self._quality_memo[key] = QualityMemoEntry(
                quality_metrics=dict(metrics),
                execution_dag=None,
                full=full,
                execution_dag_path=dag_path if os.path.isfile(dag_path) else None,
            )
            _seeded += 1
        if _seeded:
            logger.info(
                f"Checkpoint: quality memo seeded with "
                f"{len(self._quality_memo)} algo config(s) from {_seeded} record(s)"
            )

        # Set eval counter to continue from where we left off. Take the max
        # across BOTH the per-eval directories (valid_evals) AND the CSV —
        # if a previous crash wrote the CSV row but failed to create the
        # eval_NNNN/ dir, the dir-only max would let counter rewind onto an
        # id that's already in the CSV, producing duplicate rows on the
        # next resume's initialization. eval_id MUST be unique.
        self._eval_counter = max(
            self._eval_counter,
            max(ev["eval_id"] for ev in valid_evals) + 1,
        )

        # Hardware/system transfer has a stricter, transfer-specific commit
        # ledger.  Its caller asks us to restore the generic filesystem
        # bookkeeping (monotonic eval IDs, invalid registry and quality memo)
        # but deliberately defers optimizer restoration until source order,
        # target identity and per-record transfer commits have been verified.
        if not warm_start:
            self._checkpoint_loaded = True
            logger.info(
                "Checkpoint: deferred optimizer warm_start for strict "
                "system-transfer recovery (%d reconstructable records, "
                "eval_counter=%d)",
                len(valid_evals), self._eval_counter,
            )
            return

        # Warm-start optimizer
        try:
            self.optimizer.warm_start(valid_evals)
            self._checkpoint_loaded = True
            # Restore cumulative cm/gt counts so post-resume progress
            # dashboards reflect the project's lifetime totals, not just
            # this session. _total_cost is intentionally NOT restored —
            # in topo-decoupled mode it's a synthetic budget, and carrying
            # it across resumes would mislead.
            loaded_cm = sum(
                1 for ev in valid_evals if ev.get("eval_type") == EvalType.CM
            )
            # This is the optimizer's valid-candidate budget, not a count of
            # paid judge calls. Memo hits consume a seat on the live path and
            # therefore must consume the same seat after resume.
            loaded_gt = _restored_gt_budget_count(valid_evals)
            self._gt_memo_hits = sum(
                1 for ev in valid_evals if ev.get("quality_memo_hit")
            )
            # Summary counts can be stale in either direction: a crash can
            # leave them too low, while a bad resumed run can mark many evals
            # invalid and still complete the summary. Trust reconstructable
            # eval records as the checkpoint source of truth.
            self._cm_count = loaded_cm
            self._gt_count = loaded_gt
            logger.info(
                f"Checkpoint: loaded {len(valid_evals)} evaluations "
                f"(eval_counter={self._eval_counter}, "
                f"cm={self._cm_count}, gt={self._gt_count}, "
                f"penalty_replayed={n_penalty_replayed})"
            )
        except NotImplementedError as e:
            raise RuntimeError(
                f"Checkpoint: optimizer {type(self.optimizer).__name__} does not "
                "support warm_start; refusing to append a fresh optimization "
                "budget to an existing project"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Checkpoint: warm_start failed for "
                f"{type(self.optimizer).__name__}; refusing an unsafe fresh "
                f"restart: {e}"
            ) from e

    def _ensure_config(self):
        """Verify retained input; never rewrite it with runtime-derived data."""
        raw_content = verify_project_config(self.project_dir)
        if raw_content != self._project_config_raw:
            raise ProjectConfigError(
                "Project configuration no longer matches this controller's "
                "original input. Create a new project to change it."
            )
        if self.config != self._project_runtime_config:
            raise ProjectConfigError(
                "Controller configuration changed after initialization. "
                "Create a new project to change it."
            )

    def _make_quality_evaluator(self, dataset):
        """Construct the selected backend through the host-owned registry."""
        backend = str((self.config.get("global") or {}).get("eval_backend", "static_gt")).lower()
        return create_quality_evaluator(
            dataset, self.project_dir,
            dataset_manager=self._data,
            backend=backend,
            backend_options=configured_backend_options(self.config),
        )

    # ----------------------------------------------------------------------
    # Public properties
    # ----------------------------------------------------------------------

    @property
    def qa_data(self) -> pd.DataFrame:
        """QA dataframe (proxied to :class:`DatasetManager`)."""
        return self._data.qa_data

    @property
    def corpus_data(self) -> pd.DataFrame:
        """Corpus dataframe (proxied to :class:`DatasetManager`)."""
        return self._data.corpus_data

    @property
    def token_stats(self):
        """Token statistics (proxied to :class:`DatasetManager`)."""
        return self._data.token_stats

    # ----------------------------------------------------------------------
    # Public API: main entry points + state mutation
    # ----------------------------------------------------------------------

    def update_qa_data(self, new_qa_data: pd.DataFrame):
        """Update QA data on both the controller and the evaluator."""
        self._data.update_qa_data(new_qa_data, self.rag_quality_evaluator)

    def get_pareto(self, n_iterations: Optional[int] = None) -> dict:
        """Run optimization and return the Pareto frontier.

        In measured mode this wraps the whole run in the MeasuredProvider
        context so the ModelCache + vLLM subprocesses load ONCE and tear down on
        exit/crash. In cost_model mode the provider is None → direct call, no
        behavior change."""
        self._ensure_config()
        configured_iterations = int(
            (self.config.get("global") or {}).get("n_iterations", 100),
        )
        if n_iterations is not None and n_iterations != configured_iterations:
            raise ProjectConfigError(
                "n_iterations must match the frozen project configuration. "
                "Set global.n_iterations before starting a new project."
            )
        if self._measured_provider is None:
            return self._get_pareto_impl(n_iterations)
        with self._measured_provider:
            return self._get_pareto_impl(n_iterations)

    def _get_pareto_impl(self, n_iterations: Optional[int] = None) -> dict:
        """Run decoupled optimization and return the Pareto frontier.

        If all search space parameters are discrete and the total number of
        combinations is small enough (≤ n_both_init), exhaustively
        evaluates all combinations and returns the Pareto front directly,
        skipping surrogate-based optimization.

        Otherwise orchestrates the full optimization loop:
        Phase 1 (Both init): Random configs evaluated on ALL objectives.
        Phase 2 (Optimize): Iteratively evaluate candidates on all objectives.
        Phase 3:            Extract and return the Pareto frontier.

        Returns:
            Dict with pareto_configs, pareto_objectives,
            total_cost, cm_count, gt_count, assembly_results.
        """
        # --- Ensure config file exists / matches ---
        self._ensure_config()

        # --- Check if exhaustive evaluation is feasible ---
        # Initialize counters early — _evaluate_exhaustive (below) reads them
        # from save_summary, so they must exist before either branch runs.
        # _total_cost is per-run (always zeroed); cm/gt counts are cumulative
        # across resumes (preserved when _load_checkpoint already populated
        # them from summary.json).
        self._total_cost = 0.0
        if not self._checkpoint_loaded:
            self._gt_count = 0
            self._cm_count = 0

        exhaustive_configs = self._ssops.enumerate_all_configs()
        if exhaustive_configs is not None:
            optimizer = self.optimizer
            n_both_init = getattr(optimizer, "n_both_init", 1)

            if len(exhaustive_configs) <= n_both_init:
                logger.info(
                    f"Search space has only {len(exhaustive_configs)} discrete "
                    f"combination(s) (≤ both-objective init budget {n_both_init}). "
                    f"Exhaustively evaluating all."
                )
                return self._evaluate_exhaustive(exhaustive_configs)

        optimizer = self.optimizer
        obj_names = list(optimizer.objective_costs.keys())

        # Build expanded objective columns for CSV (once, up front)
        obj_columns = self._build_obj_columns(obj_names)

        if n_iterations is None:
            n_iterations = int((self.config.get("global") or {}).get("n_iterations", 100))
        # Instrumentation (07-04): let the agent optimizer persist its LLM
        # advice proposals under the project for post-hoc seed attribution.
        try:
            setattr(optimizer, "_advice_log_dir", self.project_dir)
        except Exception:  # noqa: BLE001
            pass
        # Advisory channel for infeasible arms (07-05): keep a ref so
        # _record_invalid can inform the agent's INFEASIBLE card, and replay
        # invalids loaded from the checkpoint into the fresh optimizer.
        self._opt_ref = optimizer
        # Resume-global-awareness (07-05, user directive): restored evals must
        # carry their PERF ATTRIBUTION tags too — replay each eval dir's
        # assembly.json into the optimizer (the live path registers these at
        # eval time; a resumed run otherwise leaves pre-resume rows untagged
        # and the LLM blind to their bound stages).
        self._restore_persisted_perf_attribution(optimizer)
        if hasattr(optimizer, "register_infeasible"):
            for _rec in self._invalid_configs:
                if not _invalid_record_is_config_verdict(_rec):
                    continue
                try:
                    optimizer.register_infeasible(
                        _rec.get("config") or {},
                        _rec.get("reason", ""),
                        eval_id=_rec.get("eval_id"),
                        optimizer_metadata=_rec.get("optimizer_metadata"),
                        replay=True,
                    )
                except Exception:  # noqa: BLE001
                    pass
        logger.info(
            f"[progress] optimization start: project={self.project_dir} "
            f"mode={'measured' if self._measured_provider is not None else 'cost_model'} "
            f"gt_budget={n_iterations} objectives={obj_names} "
            f"fail_fast={_fail_fast_eval_errors()}"
        )

        if self._checkpoint_loaded:
            # --- Checkpoint resume: finish Phase 1 if the
            # init phase was interrupted (07-08). Optimizers exposing
            # `init_chunks` (sobol-first agent init) report their REMAINING
            # init batches from the restored history; running them here keeps
            # a resumed run's init protocol identical to a fresh run's (the
            # previous behavior jumped straight to Phase 2, silently
            # truncating init — observed on the 07-08 s44 rerun: 7 probes,
            # Phase 1 never completed). Optimizers without init_chunks keep
            # the old skip bit for bit.
            _rem = list(getattr(optimizer, "init_chunks", None) or [])
            if _rem:
                logger.info(
                    f"Checkpoint resume: init phase incomplete — finishing "
                    f"Phase 1 (remaining chunks {_rem}) from "
                    f"eval_counter={self._eval_counter}"
                )
                self._run_phase_1(optimizer, obj_columns)
            else:
                logger.info(
                    f"Checkpoint resume: initialization complete, "
                    f"starting Phase 2 from eval_counter={self._eval_counter}"
                )
        else:
            # --- Phase 1: Both-objective init (all objectives evaluated) ---
            self._run_phase_1(optimizer, obj_columns)

        # --- Phase 2: Iterative optimization ---
        # n_iterations is the TOTAL GT-call budget (Phase 1 init + Phase 2 BO).
        # Seed `iteration` with the GT count already paid by Phase 1 (or the
        # checkpoint-loaded GT count on resume) so the loop runs exactly
        # `n_iterations - self._gt_count` successful GT budget seats in Phase 2.
        iteration = self._gt_count
        # Safety valve (07-06, valid-eval counting): with penalty/invalid evals
        # no longer consuming the success budget, a pathological proposer could
        # burn wall-clock forever on infeasible arms. Hard-cap RAW attempts per
        # session at 2x the budget and fail loudly (fail-closed spirit).
        _raw_start = self._eval_counter
        while iteration < n_iterations:
            if self._eval_counter - _raw_start >= max(2 * n_iterations, 20):
                raise RuntimeError(
                    f"raw-eval safety valve: {self._eval_counter - _raw_start} attempts this "
                    f"session yielded only {iteration}/{n_iterations} total valid GTs — aborting "
                    "instead of burning unbounded wall-clock on infeasible proposals"
                )
            t0 = time.monotonic()

            eval_id_start = self._eval_counter
            configs, eval_obj_names = optimizer.suggest()
            if configs is None or eval_obj_names is None:
                logger.info(
                    f"Search space exhausted after {iteration}/{n_iterations} "
                    "total valid GTs (Phase 1 init included)."
                )
                break
            logger.info(
                f"[progress] Phase 2 proposal for total GT budget seat "
                f"{iteration + 1}/{n_iterations} (Phase 1 init included): "
                f"{len(configs)} candidate(s), objectives={eval_obj_names}, "
                f"eval_ids={eval_id_start}-{eval_id_start + len(configs) - 1}"
            )
            obj_dict, valid_configs, n_success, free_flags = self._evaluate_objectives(configs, eval_obj_names, obj_columns)
            if not valid_configs:
                logger.info(
                    "[progress] Phase 2 proposal for total GT budget seat %d/%d "
                    "produced no valid objective rows; retrying the same seat "
                    "because GT_FP/invalid rows do not consume the GT budget.",
                    iteration + 1,
                    n_iterations,
                )
                continue
            eids = list(self._last_valid_eval_ids)
            if len(eids) != len(valid_configs):
                raise RuntimeError(
                    "Phase 2 evaluator returned misaligned valid eval ids: "
                    f"configs={len(valid_configs)}, ids={len(eids)}"
                )
            try:
                # Free rows (fingerprint-inherited quality, M3) are archive
                # observations that must not advance the optimizer's own
                # success accounting; optimizers without the kwarg (baselines)
                # never receive free rows anyway (flag is config-gated).
                optimizer.add_objectives(
                    valid_configs, obj_dict, eval_ids=eids,
                    free_mask=free_flags if any(free_flags) else None,
                )
            except TypeError:
                optimizer.add_objectives(valid_configs, obj_dict, eval_ids=eids)
            self._push_perf_attribution(optimizer, eids)

            new_cost = sum(optimizer.objective_costs[n] for n in eval_obj_names)
            self._total_cost += new_cost

            # n_iterations is the TOTAL GT-call budget (includes Phase 1 init).
            # Valid-eval counting: only SUCCESS evals consume the budget. A
            # batch that produced solely penalty rows feeds the GP but keeps
            # its seat; the raw-eval safety valve bounds total attempts.
            self._gt_count += n_success
            eval_type = EvalType.GT
            consumed_budget = n_success > 0

            t1 = time.monotonic()

            obj_strs = " ".join(
                format_objective_value(name, obj_dict.get(name, [None])[0])
                for name in obj_names
            )
            logger.info(
                f"  Iter {iteration:3d} ({eval_type}) | eval={eval_obj_names} | "
                f"{obj_strs} | "
                f"cost: {new_cost:.2f} total: {self._total_cost:.2f} | "
                f"time: {t1 - t0:.1f}s"
            )

            if consumed_budget:
                iteration += n_success
            else:
                logger.info(
                    "[progress] Total valid GT budget remains %d/%d; this "
                    "GT_FP/penalty result did not consume a budget seat.",
                    iteration,
                    n_iterations,
                )
            self._update_summary()

        # --- Extract Pareto frontier ---
        pareto_configs, pareto_objectives, pareto_eval_ids = optimizer.get_pareto()
        pareto_configs = self._ssops.enrich_configs(pareto_configs, self._base_decoded_config)
        logger.info(
            f"Optimization complete! Total valid GT budget consumed: "
            f"{iteration}/{n_iterations} (Phase 1 init included), "
            f"Cost: {self._total_cost:.2f}, CM: {self._cm_count}, GT: {self._gt_count} "
            f"({self._gt_memo_hits} memo-reused), "
            f"raw evals: {self._eval_counter} "
            f"(non-valid attempts kept on disk for audit), "
            f"Pareto points: {len(pareto_configs)}"
        )
        if self._failure_counts:
            infeasible = {k: v for k, v in self._failure_counts.items()
                          if k in ("CONTEXT_OVERFLOW", "OOM")}
            anomalous = {k: v for k, v in self._failure_counts.items()
                         if k not in ("CONTEXT_OVERFLOW", "OOM")}
            if infeasible:
                logger.info(f"budget-free infeasibility rows (normal): {infeasible}")
            if anomalous:
                logger.warning(
                    "run carries ANOMALOUS failure rows (should be zero in a "
                    f"healthy run; none consumed budget): {anomalous} — "
                    "check the eval_error column / eval dirs before trusting "
                    "operational comparisons"
                )

        # Save Pareto frontier (end-only — requires all evaluations)
        results = {
            "pareto_configs": pareto_configs,
            "pareto_objectives": pareto_objectives,
            "pareto_eval_ids": pareto_eval_ids,
            "total_cost": self._total_cost,
            "cm_count": self._cm_count,
            "gt_count": self._gt_count,
            "assembly_results": self._assembly_results,
        }
        self._save_pareto_frontier(results)
        self._update_summary(
            n_pareto_points=len(pareto_configs), status="completed",
        )

        return results

    def get_pareto_for_hardware(
        self,
        hardware_config: dict | SystemConfig,
        quality_store_path: str,
        n_refine_iterations: int = 0,
        performance_selection: dict | None = None,
        reuse_target_project: str | None = None,
        retransfer_source_indices: Sequence[int] | None = None,
    ) -> dict:
        """Compute Pareto frontier for a specific hardware configuration
        using pre-computed quality evaluations from a QualityStore.

        Phase 2a (``n_refine_iterations=0``):
            Load quality results → run CM for each config → direct Pareto.
            Runtime is workload dependent. For large dynamic sweeps,
            trace-driven CM screens exact deployments analytically and
            sends the stratified shortlist through authoritative DES.

        Phase 2b (``n_refine_iterations > 0``):
            After Phase 2a, run an optimisation loop that suggests new
            configs.  Quality evaluations are served from the store when
            available; otherwise new GT evaluations are run and written
            back to the store for future reuse.

        Args:
            hardware_config: Hardware specification.  Either a SystemConfig
                or a dict parseable by ``SystemConfig.from_dict()``.
            quality_store_path: Path to an existing QualityStore directory.
            n_refine_iterations: Total number of optimisation iterations after
                Phase 2a.  On resume, committed refinements count toward this
                total; the budget may be increased, but never decreased.
                0 = direct Pareto only.
            performance_selection: Optional explicit selection policy. It must
                match the target project's frozen system.performance_objectives.
            reuse_target_project: Optional prior target transfer project.  When
                provided, its validated Phase-2a observations are copied for
                every source index not explicitly selected for recomputation.
            retransfer_source_indices: Frozen zero-based source indices to run
                through current dynamic CM.  Must be provided together with
                ``reuse_target_project`` and exactly equal the active
                LLMLingua2 source dependency set.

        Returns:
            Dict with ``pareto_configs``, ``pareto_objectives``,
            ``total_cost``, ``cm_count``, ``gt_count``, ``assembly_results``.
        """
        import numpy as np
        from rag_stack.types import CMOutOfMemoryError

        self._ensure_config()
        if n_refine_iterations < 0:
            raise ValueError("n_refine_iterations must be >= 0")
        if not RAGCMAssembly.__module__.startswith("rag_stack.cost_model."):
            raise RuntimeError(
                "System transfer is pinned to production CM, but the "
                f"cost-model entry resolved {RAGCMAssembly.__module__}"
            )

        # --- Parse and cross-check the one target system source of truth ---
        raw_target_system = (self.config or {}).get("system")
        if not isinstance(raw_target_system, dict):
            raise ValueError("Target project config has no valid system block")
        configured_hw = SystemConfig.from_dict(raw_target_system)
        if isinstance(hardware_config, dict):
            hw = SystemConfig.from_dict(hardware_config)
        else:
            hw = hardware_config
        if hw.to_dict() != configured_hw.to_dict():
            raise ValueError(
                "System transfer hardware_config must describe the same target "
                "system as the target project's system block; independent "
                "hardware overrides would split assembly and pipeline policy"
            )
        self._transfer_target_system = deepcopy(raw_target_system)

        if performance_selection is not None:
            from rag_stack.types import PerformanceSelection
            requested_selection = PerformanceSelection.from_dict(performance_selection)
            configured_selection = hw.performance or PerformanceSelection()
            if requested_selection != configured_selection:
                raise ProjectConfigError(
                    "performance_selection must match the frozen project's "
                    "system.performance_objectives. Create a new project to change it."
                )

        # Hardware replay is defined over a strict canonical QualityStore.  Refinement
        # uses the Controller's already-initialised dataset/evaluator and the
        # same store as its exact quality+trace memo; it must never silently
        # downgrade to replay-only when the requested iteration budget is > 0.
        quality_store = QualityStore.load(quality_store_path)
        if not quality_store.requires_quality_trace:
            raise ValueError(
                "System transfer requires a QualityStore with a frozen "
                "quality trace for every record"
            )
        quality_results = quality_store.get_quality_results()
        if not quality_results:
            raise ValueError(f"Quality store at {quality_store_path} is empty")
        def _source_order(record: dict) -> tuple[int, str]:
            raw_eval_id = (((record.get("record") or {}).get("source") or {}).get(
                "eval_id"
            ))
            try:
                eval_id = int(raw_eval_id)
            except (TypeError, ValueError):
                eval_id = 1 << 60
            return eval_id, str(record["config_hash"])

        quality_results = sorted(quality_results, key=_source_order)
        quality_results = self._frozen_transfer_source_view(quality_results)
        selective_replay_plan = self._build_selective_transfer_replay_plan(
            quality_store=quality_store,
            quality_results=quality_results,
            reuse_target_project=reuse_target_project,
            retransfer_source_indices=retransfer_source_indices,
        )
        transfer_state, quality_results = self._prepare_system_transfer_state(
            quality_store=quality_store,
            quality_results=quality_results,
            n_refine_iterations=n_refine_iterations,
            replay_plan=selective_replay_plan,
        )

        # Inject workload-derived token dimensions into the target deployment
        # sweep.  These are trace/workload properties, not CM-version knobs.
        # Synthesised stores may not carry manifest-level token stats, while
        # every committed source pipeline still carries the exact dimensions
        # used by that evaluation.  Require one consensus value across the
        # frozen source set rather than silently falling back to CM defaults.
        token_stats = quality_store.get_token_stats()
        if hw.search_space is None:
            hw.search_space = {}
        committed_workload_axes: dict[str, set] = {
            "dec_steps": set(),
            "seq_len_inference_prefill": set(),
        }
        for source_record in quality_results:
            source_pipeline = quality_store.get_pipeline_config(
                source_record["config_hash"],
            )
            source_space = (
                (source_pipeline.get("system") or {}).get("cm_search_space")
                or {}
            )
            for axis in committed_workload_axes:
                value = source_space.get(axis)
                if value is not None:
                    committed_workload_axes[axis].add(value)
        for axis, values in committed_workload_axes.items():
            if len(values) > 1:
                raise ValueError(
                    "System transfer source pipelines disagree on workload "
                    f"axis {axis!r}: {sorted(values, key=str)}"
                )
            if values:
                hw.search_space.setdefault(axis, next(iter(values)))
        if token_stats:
            hw.search_space.setdefault(
                "dec_steps", token_stats.get("avg_output_tokens", 128),
            )
            hw.search_space.setdefault(
                "seq_len_inference_prefill",
                token_stats.get("prompt_template_tokens", 0)
                + 10 * token_stats.get("avg_chunk_tokens", 50),
            )

        # Keep the normal Controller evaluation path and the replay path on the
        # same target CM assembly.  The rag_stack.cost_model package owns
        # the only production CM; the trace contract is independently validated by the
        # QualityStore and evaluate_dynamic().
        assembly = RAGCMAssembly(hw)
        self.hardware = hw
        self.assembly = assembly
        self._assembly_results = []
        assembly_results = self._assembly_results
        self._total_cost = 0.0
        self._cm_count = 0
        self._gt_count = 0
        self._strict_eval_persistence = True
        # Full CM sweep payloads belong to each already-durable evaluation
        # directory.  Retain only a portable selected-deployment summary in
        # this long-lived Controller list during replay and refinement.
        self._retain_portable_assembly_results = True

        existing_observations = len(
            getattr(self.optimizer, "_configs", []) or []
        )
        if existing_observations:
            raise RuntimeError(
                "System transfer optimizer must be empty before strict "
                "transfer-ledger restoration; generic checkpoint warm_start "
                f"already installed {existing_observations} observations"
            )

        replay_evaluations, restored_replay_assemblies = (
            self._restore_transfer_replays(
                state=transfer_state,
                quality_store=quality_store,
                quality_results=quality_results,
            )
        )
        self._assembly_results.extend(restored_replay_assemblies)
        configs: List[dict] = [
            deepcopy(item["config"]) for item in replay_evaluations
        ]
        quality_scores: List[float] = [
            float(item["objectives"]["quality"]["combined"])
            for item in replay_evaluations
        ]
        performance_scores: List[float] = [
            float(item["objectives"]["performance"])
            for item in replay_evaluations
        ]
        config_hashes: List[str] = [
            quality_results[index]["config_hash"]
            for index in range(len(replay_evaluations))
        ]
        target_eval_ids: List[int] = list(range(len(replay_evaluations)))
        self._cm_count = len(replay_evaluations)
        obj_columns = self._build_obj_columns(
            list(self.optimizer.objective_costs.keys()),
        )

        # --- Phase 2a: trace-driven CM replay + full warm-start ledger ---
        for replay_idx in range(len(replay_evaluations), len(quality_results)):
            self._ensure_config()
            qr = quality_results[replay_idx]
            h = qr["config_hash"]
            cfg = qr["config"]
            combined = float(qr["quality_combined"])
            quality_metrics = qr.get("quality_metrics") or {}
            pipeline_config = None
            trace = None
            result = None
            target_eval_type = EvalType.GT_FP
            target_error = ""
            replay_action, replay_dependency = self._transfer_replay_action(
                transfer_state, replay_idx,
            )
            if replay_action == _TRANSFER_ACTION_REUSED:
                (
                    pipeline_config,
                    trace,
                    result,
                    target_eval_type,
                    target_error,
                    performance,
                ) = self._load_validated_prior_target_replay(
                    replay_idx=replay_idx,
                    quality_store=quality_store,
                    quality_record=qr,
                    replay_plan=transfer_state["replay_plan"],
                )
            else:
                if replay_dependency is not None:
                    self._assert_transfer_dependency_unchanged(
                        replay_dependency,
                    )
                try:
                    pipeline_config, trace, result = self._price_stored_quality_record(
                        quality_store=quality_store,
                        config_hash_value=h,
                        assembly=assembly,
                    )
                    performance = float(result.performance_score)
                except CMOutOfMemoryError as exc:
                    # A target-system all-candidate capacity proof is a legitimate
                    # transferred observation: preserve its real source quality and
                    # feed a zero-performance trough to the optimiser.  Every other
                    # exception is a trace/config/CM bug and aborts the whole replay.
                    if pipeline_config is None:
                        pipeline_config = self._target_replay_pipeline(
                            quality_store, h,
                        )
                    if trace is None:
                        trace = quality_store.get_execution_dag(h)
                    performance = 0.0
                    target_eval_type = EvalType.OOM
                    target_error = f"target CM: {_format_exception(exc)}"
                except Exception as exc:
                    raise RuntimeError(
                        "System transfer aborted: dynamic CM replay failed for "
                        f"record {replay_idx + 1}/{len(quality_results)} "
                        f"(algo_hash={h}): {_format_exception(exc)}"
                    ) from exc
                if replay_dependency is not None:
                    # Detect replacement during the (potentially long) CM
                    # sweep as well as before it starts. A paper benchmark may
                    # never mix coefficients within one frozen replay plan.
                    self._assert_transfer_dependency_unchanged(
                        replay_dependency,
                    )

            configs.append(cfg)
            quality_scores.append(combined)
            performance_scores.append(performance)
            config_hashes.append(h)
            target_eval_ids.append(replay_idx)
            quality_payload = self._transfer_optimizer_quality_payload(
                combined=combined,
                quality_metrics=quality_metrics,
            )
            replay_evaluations.append({
                "config": cfg,
                "objectives": {
                    "quality": quality_payload,
                    "performance": performance,
                },
                "eval_id": replay_idx,
                "eval_type": target_eval_type.value,
                "free": self._transfer_optimizer_history_free(
                    qr.get("record") or {}, target_eval_type,
                ),
                "quality_reused": True,
            })

            # A replay is a free quality reuse, hence GT_FP on a feasible
            # target.  The per-eval provenance file retains the source outcome
            # (including H100 CM-OOM) without poisoning the A100 objective.
            self._save_eval_record(
                eval_id=replay_idx,
                config=cfg,
                pipeline_config=pipeline_config,
                eval_type=target_eval_type,
                objectives={
                    "performance": performance,
                    "quality": quality_payload,
                },
                quality_metrics=quality_metrics,
                assembly_result=result,
                obj_columns=obj_columns,
                execution_dag=trace,
                eval_error=target_error,
            )
            if result is not None:
                # The full sweep is durable now; only its selected deployment
                # summary remains resident for counts and later attribution.
                self._assembly_results.append(_portable_assembly_result(result))
            self._save_transfer_provenance(
                replay_idx=replay_idx,
                config_hash_value=h,
                source_record=qr.get("record") or {},
                target_eval_type=target_eval_type,
                target_config_sha256=transfer_state["target_config_sha256"],
                source_record_marker_sha256=(
                    transfer_state["source_records"][replay_idx][
                        "record_marker_sha256"
                    ]
                ),
                replay_action=replay_action,
                llmlingua2_profile=replay_dependency,
                reuse_lineage=(
                    self._transfer_reuse_lineage(
                        transfer_state["replay_plan"], replay_idx,
                    )
                    if replay_action == _TRANSFER_ACTION_REUSED
                    else None
                ),
            )
            self._eval_counter = replay_idx + 1
            self._cm_count = replay_idx + 1
            self._update_summary()

        if len(configs) != len(quality_results):
            raise RuntimeError(
                "System transfer invariant failed: not every source quality "
                f"record was repriced ({len(configs)}/{len(quality_results)})"
            )

        (
            restored_refinements,
            restored_refinement_assemblies,
            restored_refinement_gt,
            restored_refinement_cost,
        ) = self._restore_transfer_refinements(
            state=transfer_state,
            quality_store=quality_store,
            source_count=len(quality_results),
        )
        for observation in restored_refinements:
            objectives = observation["objectives"]
            quality = objectives["quality"]
            configs.append(deepcopy(observation["config"]))
            quality_scores.append(float(
                quality["combined"] if isinstance(quality, dict) else quality
            ))
            performance_scores.append(float(objectives["performance"]))
            config_hashes.append(self._quality_algo_key(observation["config"]))
            target_eval_ids.append(int(observation["eval_id"]))
        self._assembly_results.extend(restored_refinement_assemblies)
        self._gt_count = restored_refinement_gt
        self._cm_count = len(configs)
        self._total_cost = restored_refinement_cost

        # Direct Pareto from observed points
        Y = np.array(
            [[q, p] for q, p in zip(quality_scores, performance_scores)]
        )
        pareto_mask = is_non_dominated_np(Y)
        pareto_idx = np.where(pareto_mask)[0]

        # Sort by performance (second column)
        sorted_idx = pareto_idx[np.argsort(Y[pareto_idx, 1])]

        pareto_configs = self._ssops.enrich_configs(
            [configs[i] for i in sorted_idx], self._base_decoded_config,
        )
        pareto_objectives = [
            {"quality": quality_scores[i], "performance": performance_scores[i]}
            for i in sorted_idx
        ]
        pareto_eval_ids = [target_eval_ids[i] for i in sorted_idx]
        pareto_algo_hashes = [config_hashes[i] for i in sorted_idx]

        cm_count = len(configs)
        gt_count = restored_refinement_gt

        logger.info(
            f"Phase 2a complete: {len(replay_evaluations)} configs replayed, "
            f"{len(pareto_configs)} Pareto points"
        )

        # --- Phase 2b: full warm-start, then exactly N landed iterations ---
        if n_refine_iterations > 0:
            optimizer = self.optimizer
            try:
                setattr(optimizer, "_advice_log_dir", self.project_dir)
            except Exception:  # noqa: BLE001
                pass
            # Keep invalid-arm reporting and advice persistence identical to
            # the normal optimization path before the first Phase-2b suggest.
            self._opt_ref = optimizer
            optimizer_history = replay_evaluations + restored_refinements
            try:
                optimizer.warm_start(optimizer_history)
            except Exception as exc:
                raise RuntimeError(
                    "System transfer aborted: optimizer warm_start failed; "
                    "refusing to continue with a partial/fresh history: "
                    f"{_format_exception(exc)}"
                ) from exc
            self._assert_exact_transfer_warm_start(optimizer_history)
            self._restore_persisted_perf_attribution(optimizer)
            logger.info(
                "Phase 2b warm-started optimizer with %d replayed and %d "
                "committed refinement observations",
                len(replay_evaluations), len(restored_refinements),
            )

            # Attach the store as the normal Controller's exact quality+trace
            # memo. New suggestions then use the production quality-first,
            # trace-driven dynamic CM path and are persisted like ordinary
            # evaluations; no second, divergent evaluator loop is maintained.
            self._attach_quality_store(quality_store_path)
            remaining_refinements = (
                n_refine_iterations - len(restored_refinements)
            )
            new_gt_count = self._run_refinement(
                assembly=assembly,
                quality_store=quality_store,
                n_iterations=remaining_refinements,
                configs=configs,
                quality_scores=quality_scores,
                performance_scores=performance_scores,
                config_hashes=config_hashes,
                target_eval_ids=target_eval_ids,
                assembly_results=self._assembly_results,
                obj_columns=obj_columns,
                transfer_state=transfer_state,
                refinement_index_offset=len(restored_refinements),
            )
            gt_count = restored_refinement_gt + new_gt_count

            # Recompute Pareto with new data
            Y = np.array(
                [[q, p] for q, p in zip(quality_scores, performance_scores)]
            )
            pareto_mask = is_non_dominated_np(Y)
            pareto_idx = np.where(pareto_mask)[0]
            sorted_idx = pareto_idx[np.argsort(Y[pareto_idx, 1])]

            pareto_configs = self._ssops.enrich_configs(
                [configs[i] for i in sorted_idx], self._base_decoded_config,
            )
            pareto_objectives = [
                {"quality": quality_scores[i], "performance": performance_scores[i]}
                for i in sorted_idx
            ]
            pareto_eval_ids = [target_eval_ids[i] for i in sorted_idx]
            pareto_algo_hashes = [config_hashes[i] for i in sorted_idx]
            cm_count = len(configs)

        # --- Save results ---
        refinement_iterations_completed = (
            len(configs) - len(replay_evaluations)
        )
        if refinement_iterations_completed != n_refine_iterations:
            raise RuntimeError(
                "System transfer refinement completed an inexact number of "
                "landed optimizer observations: "
                f"{refinement_iterations_completed}/{n_refine_iterations}"
            )
        replay_plan = transfer_state.get("replay_plan")
        if replay_plan is None:
            cm_repriced_observations = len(quality_results)
            imported_replay_observations = 0
        else:
            cm_repriced_observations = len(
                replay_plan["retransfer_source_indices"]
            )
            imported_replay_observations = len(
                replay_plan["reuse_source_indices"]
            )
        results = {
            "pareto_configs": pareto_configs,
            "pareto_objectives": pareto_objectives,
            "pareto_eval_ids": pareto_eval_ids,
            "pareto_algo_hashes": pareto_algo_hashes,
            "total_cost": self._total_cost,
            "cm_count": cm_count,
            "gt_count": gt_count,
            "assembly_results": assembly_results,
            "hardware": hw.to_dict(),
            "system_transfer": {
                "source_observations": len(quality_results),
                "repriced_observations": cm_repriced_observations,
                "cm_repriced_observations": cm_repriced_observations,
                "reused_target_observations": imported_replay_observations,
                "imported_replay_observations": imported_replay_observations,
                "refinement_iterations_requested": n_refine_iterations,
                "refinement_iterations_completed": (
                    refinement_iterations_completed
                ),
                "cost_model_version": 1,
            },
        }
        self._save_pareto_frontier(results)
        self._save_system_transfer_result(results)
        completed_state = deepcopy(transfer_state)
        completed_state["status"] = "completed"
        completed_state["refinement_iterations_completed"] = (
            refinement_iterations_completed
        )
        completed_state["result_sha256"] = _file_sha256(os.path.join(
            self.project_dir, "system_transfer_result.json",
        ))
        self._atomic_transfer_json(os.path.join(
            self.project_dir, _SYSTEM_TRANSFER_STATE_FILENAME,
        ), completed_state)
        self._update_summary(
            n_pareto_points=len(pareto_configs), status="completed",
        )

        return results

    def save_results(self, results: dict) -> str:
        """Persist final results to project_dir (backward-compatible API).

        Note: With incremental saving, most data is already on disk by the
        time this is called.  This method writes the Pareto frontier CSV and
        a final summary, which can also serve as a manual re-save.

        Args:
            results: Dict returned by get_pareto().

        Returns:
            Path to the output directory.
        """
        output_dir = self.project_dir
        os.makedirs(output_dir, exist_ok=True)

        self._save_pareto_frontier(results)
        self._total_cost = results.get("total_cost", self._total_cost)
        self._cm_count = results.get("cm_count", self._cm_count)
        self._gt_count = results.get("gt_count", self._gt_count)
        self._update_summary(
            n_pareto_points=len(results.get("pareto_configs", [])),
            status="completed",
        )

        logger.info(f"Results saved to {output_dir}")
        return output_dir

    # ----------------------------------------------------------------------
    # Private orchestration: evaluation loop
    # ----------------------------------------------------------------------

    def _evaluate_exhaustive(self, configs: List[dict]) -> dict:
        """Evaluate all configs on all objectives and return Pareto front."""
        import numpy as np

        obj_names = list(self.optimizer.objective_costs.keys())
        obj_columns = self._build_obj_columns(obj_names)

        valid_evaluations: List[dict] = []
        all_obj_values: List[List[float]] = []

        for cfg in configs:
            eval_id_before = self._eval_counter
            t0 = time.monotonic()
            obj_dict, valid_cfgs, _, _free = self._evaluate_objectives([cfg], obj_names, obj_columns)
            t1 = time.monotonic()

            if not valid_cfgs:
                continue

            # Composite objectives (e.g. `quality`) come back as dicts with a
            # `combined` scalar plus per-sub-metric values — same shape that
            # SMACDecoupledBase.add_objectives ultimately reduces to a scalar
            # via dict["combined"]. The exhaustive Pareto loop below assumes
            # numeric values, so apply the same reduction here.
            def _scalarize(v):
                if isinstance(v, dict):
                    return float(v.get("combined", 0.0))
                return float(v) if v is not None else 0.0
            obj_vals = [_scalarize(obj_dict[name][0]) for name in obj_names]
            all_obj_values.append(obj_vals)

            valid_evaluations.append({
                "config": cfg,
                # Keep the full dict in the per-eval record so downstream
                # consumers (frontend / pareto_frontier.csv) can introspect
                # sub-metrics. Only the Pareto computation below needs scalars.
                "objectives": {name: obj_dict[name][0] for name in obj_dict},
                "eval_type": EvalType.GT,
                "eval_id": eval_id_before,
            })

            obj_strs = " ".join(
                format_objective_value(name, obj_dict.get(name, [None])[0])
                for name in obj_names
            )
            logger.info(
                f"  Exhaustive [{len(valid_evaluations)}/{len(configs)}] | "
                f"{obj_strs} | time: {t1 - t0:.1f}s"
            )
            self._total_cost = sum(self.optimizer.objective_costs.values()) * len(valid_evaluations)
            self._gt_count = len(valid_evaluations)
            self._update_summary()

        self._total_cost = sum(self.optimizer.objective_costs.values()) * len(configs)

        # Compute Pareto front (maximization)
        Y = np.array(all_obj_values)
        n = len(Y)
        is_pareto = np.ones(n, dtype=bool)
        for i in range(n):
            if not is_pareto[i]:
                continue
            for j in range(n):
                if i == j or not is_pareto[j]:
                    continue
                if np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                    is_pareto[i] = False
                    break

        pareto_idx = np.where(is_pareto)[0]
        pareto_configs = self._ssops.enrich_configs(
            [configs[i] for i in pareto_idx], self._base_decoded_config,
        )
        pareto_objectives = [
            {name: all_obj_values[i][j] for j, name in enumerate(obj_names)}
            for i in pareto_idx
        ]
        pareto_eval_ids = [valid_evaluations[i]["eval_id"] for i in pareto_idx]
        logger.info(
            f"Exhaustive evaluation complete! "
            f"Configs: {len(configs)}, Pareto points: {len(pareto_configs)}, "
            f"Cost: {self._total_cost:.2f}"
        )

        self._gt_count = len(configs)
        results = {
            "pareto_configs": pareto_configs,
            "pareto_objectives": pareto_objectives,
            "pareto_eval_ids": pareto_eval_ids,
            "total_cost": self._total_cost,
            "cm_count": self._cm_count,
            "gt_count": self._gt_count,
            "assembly_results": self._assembly_results,
        }
        self._save_pareto_frontier(results)
        self._update_summary(
            n_pareto_points=len(pareto_configs), status="completed",
        )

        return results

    def _minimal_metric_set(self) -> Optional[List[dict]]:
        """The metric dicts to compute on evals that don't need full sub-metrics (ax-gen
        steps): the combined-quality OBJECTIVE metric(s) + the FREE local metrics
        (non-``deepeval_`` — token-overlap, ~0 cost). Drops the expensive deepeval
        sub-metrics. Returns None (→ caller computes the full set) when it can't be
        derived or wouldn't include the objective."""
        ebs = self.config.get("eval_backend_setting") or {}
        full = ebs.get("metrics") or []
        obj_names = set((self._combined_quality_spec or {}).get("metrics") or [])
        if not full or not obj_names:
            return None
        minimal = [
            m for m in full
            if isinstance(m, dict) and (
                m.get("metric_name") in obj_names
                or not str(m.get("metric_name", "")).startswith("deepeval_")
            )
        ]
        have = {m.get("metric_name") for m in minimal}
        if not obj_names.issubset(have):
            return None  # objective metric missing → fall back to full (never break the objective)
        return minimal

    def _restore_persisted_perf_attribution(self, optimizer) -> None:
        """Replay durable CM stage cards into an optimizer after warm-start."""
        register = getattr(optimizer, "register_perf_attribution", None)
        if register is None:
            return
        import glob as _glob

        pattern = os.path.join(
            self.project_dir, "evaluations", "eval_*", "assembly.json",
        )
        for assembly_path in _glob.glob(pattern):
            try:
                eval_dir = os.path.basename(os.path.dirname(assembly_path))
                eval_id = int(eval_dir.removeprefix("eval_"))
                with open(assembly_path) as stream:
                    hardware = (json.load(stream) or {}).get(
                        "best_hw_params"
                    ) or {}
                if not hardware:
                    continue
                attribution = _perf_attr_from_best(hardware)
                if attribution:
                    register(eval_id, attribution)
            except Exception:  # noqa: BLE001 — advisory channel only
                pass

    def _push_perf_attribution(self, optimizer, eids) -> None:
        """Hand each valid eval's per-stage CM attribution to optimizers that
        want it (currently the agent's LLM prompt). Positional zip with the
        eids the optimizer just received keeps its internal keying consistent
        even when some configs of the batch were skipped."""
        reg = getattr(optimizer, "register_perf_attribution", None)
        attrs = getattr(self, "_last_perf_attrs", None) or []
        if reg is None:
            return
        for eid, attr in zip(eids, attrs):
            if attr:
                try:
                    reg(int(eid), attr)
                except Exception:  # noqa: BLE001 — advisory only
                    pass

    def _evaluate_objectives(
        self, configs: List[dict], eval_objectives: List[str],
        obj_columns: List[str] = None,
    ) -> Dict[str, List[float]]:
        """Evaluate candidate configs on the complete objective set.

        Performance ALWAYS comes from cost model (RAGO), never from GT timing.
        Quality comes from GT evaluator metrics only.

        Each evaluation is saved to disk immediately via ``_save_eval_record()``.

        Returns:
            obj_dict: {objective_name: [value_per_config]}
        """
        self._ensure_config()
        required_objectives = set(self.optimizer.objective_names)
        if (
            len(eval_objectives) != len(required_objectives)
            or set(eval_objectives) != required_objectives
        ):
            raise ValueError(
                "_evaluate_objectives requires the complete objective set; "
                f"got {eval_objectives}, expected {self.optimizer.objective_names}"
            )
        obj_dict: Dict[str, List[float]] = {}
        eval_type = EvalType.GT

        quality_scores: List[float] = []
        performance_scores: List[float] = []
        valid_configs: List[dict] = []
        # SUCCESS evals only (eval_type GT/CM — real measured objectives).
        # valid_configs also carries genuine-infeasible penalty rows (they DO
        # feed the GP); n_success is the VALID-EVAL count the budget consumes
        # (07-06 two-counter rule: raw eval ids/folders count everything,
        # n_iterations counts only successes).
        n_success = 0
        # Positionally aligned with valid_configs: True when the row's quality
        # was fingerprint-inherited (M3) — real objectives, NO budget seat.
        free_flags: List[bool] = []
        # Positionally aligned with valid_configs: per-stage perf attribution of
        # each valid eval (None when the CM row is unavailable). Consumed by
        # _push_perf_attribution right after optimizer.add_objectives.
        self._last_perf_attrs: List[Optional[dict]] = []
        # Positionally aligned with valid_configs.  Raw eval ids are not
        # necessarily contiguous when an earlier item in a batch is INVALID;
        # callers must feed these exact ids into optimizer bookkeeping.
        self._last_valid_eval_ids: List[int] = []

        # Per-eval metric budget. The agent optimizer flags which evals it will later READ
        # sub-metrics from (LLM-driven steps + init); on the others (ax-gen exploitation), the
        # GP only needs the objective metric, so we skip the 5 expensive deepeval sub-metrics
        # (~6x the judge cost). Aligned by index with this suggestion batch.
        pending_full = list(getattr(self.optimizer, "_pending_full_metrics", []) or [])
        minimal_metrics = self._minimal_metric_set()

        for cfg_idx, cfg in enumerate(configs):
            self._ensure_config()
            eval_id = self._eval_counter
            self._eval_counter += 1
            eval_t0 = time.monotonic()
            cfg_hash = config_hash(cfg)
            obj_label = ",".join(eval_objectives)
            logger.info(
                f"[progress] eval_{eval_id:04d} START objectives={obj_label} "
                f"hash={cfg_hash[:12]}"
            )

            # Skip configs that already failed in previous iterations
            if self._is_known_invalid(cfg):
                logger.info(f"Config eval_{eval_id} skipped: known invalid config")
                self._save_eval_record(
                    eval_id=eval_id, config=cfg, pipeline_config=cfg,
                    eval_type=EvalType.DUPLICATE,
                    eval_error="config matches a previously-recorded invalid config",
                    objectives={}, quality_metrics=None,
                    assembly_result=None, obj_columns=obj_columns,
                )
                continue

            try:
                pipeline_config = self._resolve_config(cfg)
            except Exception as e:
                logger.warning(f"Config eval_{eval_id} resolve failed: {e}, marking INVALID")
                self._record_invalid(eval_id, cfg, str(e))
                self._save_eval_record(
                    eval_id=eval_id, config=cfg, pipeline_config=cfg,
                    eval_type=EvalType.INVALID, eval_error=f"resolve_config: {e}",
                    objectives={}, quality_metrics=None,
                    assembly_result=None, obj_columns=obj_columns,
                )
                continue
            logger.info(f"[progress] eval_{eval_id:04d} config resolved")

            # Per-eval corpus: resolve THIS config's chunker (hash-cached) and
            # stamp its cost-model inputs (_token_stats + FAISS N) into
            # pipeline_config. The quality evaluator independently resolves the
            # SAME cached chunk and activates data/corpus.parquet inside its own
            # evaluate(), so there's no config side-channel / evaluator mutation.
            try:
                chunker_params = (
                    (pipeline_config.get("corpus_runtime") or {}).get("chunker") or {}
                )
                corpus_view = self._data.resolve_corpus(chunker_params)
                self._data.apply_corpus_view(pipeline_config, corpus_view)
            except Exception as e:
                logger.warning(
                    f"Config eval_{eval_id} chunk-cache build failed: {e}, marking INVALID"
                )
                self._record_invalid(eval_id, cfg, f"chunk_cache: {e}")
                self._save_eval_record(
                    eval_id=eval_id, config=cfg, pipeline_config=pipeline_config,
                    eval_type=EvalType.INVALID, eval_error=f"chunk_cache: {e}",
                    objectives={}, quality_metrics=None,
                    assembly_result=None, obj_columns=obj_columns,
                )
                continue
            logger.info(
                f"[progress] eval_{eval_id:04d} corpus ready "
                f"chunk_hash={getattr(corpus_view, 'chunk_hash', '<unknown>')}"
            )

            # ─── Joint-space deployment resolve (owner=optimizer + cost_model) ─
            # The sampled deployment (deployment_preset / gpu_layout dims) is
            # expanded BEFORE the quality eval: an unfittable preset for this
            # pipeline shape is deterministic (the CM analog of TrialInvalid) —
            # penalize immediately, never burn a GPU+judge run on it.
            trial_system_config = None
            if self._cm_fixed_provider is not None:
                from rag_stack.search_space.deployment_presets import (
                    DeploymentPresetError,
                )
                try:
                    trial_system_config = self._ctx.resolve_trial_system_config(
                        cfg, pipeline_config,
                    )
                except DeploymentPresetError as e:
                    preset_text = f"deployment_preset: {_format_exception(e)}"
                    self._record_invalid(
                        eval_id,
                        cfg,
                        preset_text,
                        objectives_will_be_committed=True,
                    )
                    sub_metric_names = [
                        m.get("metric_name")
                        for m in (self.config.get("eval_backend_setting") or {}).get("metrics", [])
                        if m.get("metric_name")
                    ]
                    penalty_quality = {
                        "combined": 0.0,
                        **{k: 0.0 for k in sub_metric_names},
                    }
                    quality_scores.append(penalty_quality)
                    performance_scores.append(0.0)
                    penalty_objectives = {
                        "performance": 0.0,
                        "quality": penalty_quality,
                    }
                    self._save_eval_record(
                        eval_id=eval_id, config=cfg,
                        pipeline_config=pipeline_config,
                        eval_type=EvalType.INVALID,
                        eval_error=preset_text,
                        objectives=penalty_objectives,
                        quality_metrics=None,
                        assembly_result=None,
                        obj_columns=obj_columns,
                    )
                    self._update_summary()
                    valid_configs.append(cfg)
                    self._last_valid_eval_ids.append(eval_id)
                    # Penalty row: budget-FREE (no n_success), and free_flags
                    # MUST stay positionally aligned with valid_configs — the
                    # measured-failure penalty path below does the same.
                    free_flags.append(True)
                    self._last_perf_attrs.append(None)
                    logger.info(
                        f"[progress] eval_{eval_id:04d} INVALID after "
                        f"{time.monotonic() - eval_t0:.1f}s: {preset_text}"
                    )
                    continue
                logger.info(
                    f"[progress] eval_{eval_id:04d} deployment resolved: "
                    f"preset={trial_system_config.get('deployment_preset', '<raw>')} "
                    f"slots={(trial_system_config.get('layout') or {}).get('total_gpu_slots')}"
                )

            # ─── Measured-performance path (performance_source: measured) ────
            # One real on-hardware run yields BOTH quality + performance, so we
            # skip the cost model AND the separate GT eval. This branch is
            # self-contained and ``continue``s; the cost-model code below is
            # never reached in measured mode.
            if self._measured_provider is not None:
                from rag_stack_evaluator.static_rag_evaluator.measured.vllm_deployment import TrialInvalid
                m_run_dir = os.path.join(
                    self.project_dir, "evaluations", f"eval_{eval_id:04d}", "measured_run",
                )
                meas_err = None
                mres = None
                for attempt in range(1, _MAX_EVAL_RETRIES + 1):
                    try:
                        system_config = self._ctx.resolve_trial_system_config(
                            cfg, pipeline_config,
                        )
                        from rag_stack.system_layout import engine_devices, engine_info
                        _layout = system_config.get("layout") or {}
                        _gen = engine_info(system_config, "generator")
                        logger.info(
                            f"[progress] eval_{eval_id:04d} measured attempt "
                            f"{attempt}/{_MAX_EVAL_RETRIES}: "
                            f"serving={_gen.get('serving', '<unknown>')} "
                            f"slots={_layout.get('total_gpu_slots', '<unknown>')} "
                            f"gpus={engine_devices(system_config, 'generator')}"
                        )
                        self._save_measured_prelaunch_config(
                            eval_id, pipeline_config, system_config,
                        )
                        mres = self._measured_provider.evaluate(
                            pipeline_config, system_config, run_dir=m_run_dir,
                            require_admissible=True,
                        )
                        # A positive QPS is not sufficient publication proof.
                        # The measured runtime owns the saturation/stationarity
                        # contract; reject before memo/GT/AX bookkeeping and let
                        # the unified retry loop perform a fresh measurement.
                        from rag_stack_evaluator.static_rag_evaluator.measured.artifact_contract import (
                            require_measured_gt_admissible,
                        )
                        require_measured_gt_admissible(mres.raw_performance)
                        meas_err = None
                        break
                    except Exception as e:  # noqa: BLE001 (incl. TrialInvalid)
                        meas_err = e
                        meas_msg = _format_exception(e)
                        # UNIFIED retry policy (see _eval_retry_decision):
                        # TrialInvalid / OOM / context-overflow are deterministic
                        # — penalize immediately, never re-burn a real eval.
                        _ftype, _retryable = _eval_retry_decision(e)
                        if not _retryable:
                            outcome = (
                                "Stopping run."
                                if _should_abort_eval_failure(e)
                                else "Penalizing."
                            )
                            logger.warning(
                                f"Config eval_{eval_id} measured eval failed with "
                                f"non-retryable {_ftype}: {meas_msg}. {outcome}"
                            )
                            break
                        if attempt < _MAX_EVAL_RETRIES:
                            logger.warning(
                                f"Config eval_{eval_id} measured eval failed "
                                f"(attempt {attempt}/{_MAX_EVAL_RETRIES}): "
                                f"{meas_msg}. Retrying..."
                            )
                            time.sleep(5 * attempt)
                        else:
                            outcome = (
                                "Stopping run."
                                if _should_abort_eval_failure(e)
                                else "Penalizing."
                            )
                            logger.warning(
                                f"Config eval_{eval_id} measured eval failed after "
                                f"{_MAX_EVAL_RETRIES} attempts: {meas_msg}. {outcome}\n"
                                f"{traceback.format_exc()}"
                            )

                # Defensive: evaluate() is expected to RAISE on failure, but if
                # it ever yields no result (mres is None) without an error, never
                # crash the whole run. Synthesize an error so this arm is marked
                # INVALID and the optimization continues.
                if meas_err is None and mres is None:
                    _layout = (system_config or {}).get("layout") or {}
                    _hardware = (system_config or {}).get("hardware") or {}
                    meas_err = TrialInvalid(
                        "measured eval produced no result "
                        f"(total_gpu_slots={_layout.get('total_gpu_slots')} "
                        f"vs available={_hardware.get('available_gpu_count')})"
                    )

                if meas_err is not None:
                    meas_text = f"measured: {_format_exception(meas_err)}"
                    sub_metric_names = [
                        m.get("metric_name")
                        for m in (self.config.get("eval_backend_setting") or {}).get("metrics", [])
                        if m.get("metric_name")
                    ]
                    failure_type, penalty_objectives, penalty_quality = (
                        _measured_failure_penalty(
                            meas_err,
                            eval_gt=True,
                            metric_names=sub_metric_names,
                        )
                    )
                    if failure_type in (
                        EvalType.OOM, EvalType.CONTEXT_OVERFLOW,
                    ):
                        self._record_invalid(
                            eval_id,
                            cfg,
                            f"MEASURED: {_format_exception(meas_err)}",
                            objectives_will_be_committed=True,
                        )
                    self._save_eval_record(
                        eval_id=eval_id, config=cfg, pipeline_config=pipeline_config,
                        eval_type=failure_type,
                        eval_error=meas_text,
                        objectives=penalty_objectives,
                        quality_metrics=None, assembly_result=None, obj_columns=obj_columns,
                    )
                    self._update_summary()
                    # Every non-OOM/non-context measured failure is a bug or
                    # infrastructure fault, not a sampled-config verdict.  Stop
                    # after persisting INVALID so the operator can fix it and
                    # resume this same project; do not blacklist/register it as
                    # infeasible in the live optimizer.
                    if (
                        failure_type is EvalType.INVALID
                        or _should_abort_eval_failure(meas_err)
                    ):
                        logger.error(
                            f"[progress] eval_{eval_id:04d} FAILED after "
                            f"{time.monotonic() - eval_t0:.1f}s; stopping run: "
                            f"{meas_text}"
                        )
                        raise RuntimeError(
                            f"Stopping optimization: eval_{eval_id:04d} measured "
                            f"evaluation failed. {meas_text}"
                        ) from meas_err
                    if failure_type in (
                        EvalType.OOM, EvalType.CONTEXT_OVERFLOW,
                    ):
                        if penalty_quality is not None:
                            quality_scores.append(penalty_quality)
                        performance_scores.append(0.0)
                        valid_configs.append(cfg)
                        self._last_valid_eval_ids.append(eval_id)
                        # Genuine measured infeasibility rows are budget-FREE:
                        # n_success below never counts them, and free_flags keeps
                        # the optimizer's own phase/seat accounting aligned.
                        free_flags.append(True)
                        self._last_perf_attrs.append(None)
                    logger.info(
                        f"[progress] eval_{eval_id:04d} {failure_type.value} after "
                        f"{time.monotonic() - eval_t0:.1f}s: {meas_text}"
                    )
                    continue

                quality_metrics = mres.quality
                # ReAct (measured) carries its trace_v1 here for persistence;
                # pop it so it never leaks into the optimizer's quality_scores
                # and is saved to its own execution_dag.json instead (enables
                # measured-vs-CM accuracy checks on the SAME config). Measured
                # perf comes from real timing, so the trace is record-only here.
                measured_execution_dag = (
                    quality_metrics.pop("__execution_dag__", None)
                    if isinstance(quality_metrics, dict) else None
                )
                measured_performance_execution_trace = getattr(
                    mres, "performance_execution_trace", None
                )
                if not isinstance(measured_performance_execution_trace, dict):
                    raise RuntimeError(
                        f"eval_{eval_id:04d}: measured provider returned no "
                        "performance_execution_trace"
                    )
                from rag_stack_evaluator.static_rag_evaluator.measured.performance_trace import (
                    validate_performance_trace_envelope,
                )
                validate_performance_trace_envelope(
                    measured_performance_execution_trace
                )
                perf_score = mres.performance_score
                quality_scores.append({
                    "combined": aggregate_quality_metrics(
                        quality_metrics, self._combined_quality_spec,
                    ),
                    **{k: v for k, v in quality_metrics.items() if v is not None}  # None = judge-errored: omit, so it stays distinguishable from a true 0.0
                })
                performance_scores.append(perf_score)
                valid_configs.append(cfg)
                self._last_valid_eval_ids.append(eval_id)
                free_flags.append(False)
                # Real measured objectives consume a valid-eval budget seat —
                # the measured twin of the cost_model path's increment below.
                # Without it Phase 2's `iteration += n_success` never advances
                # and a measured run loops past n_iterations forever.
                n_success += 1
                self._last_perf_attrs.append(None)  # measured mode: no CM row

                eval_objectives_dict = {
                    "performance": perf_score,
                    "quality": quality_scores[-1],
                }

                # Persist measured perf for resume (the measured analog of
                # assembly.json) — _load_checkpoint reads it back.
                try:
                    perf_dir = os.path.join(
                        self.project_dir, "evaluations", f"eval_{eval_id:04d}",
                    )
                    os.makedirs(perf_dir, exist_ok=True)
                    with open(os.path.join(perf_dir, "performance.json"), "w") as _pf:
                        json.dump(
                            {"performance_score": perf_score,
                             "performance": mres.raw_performance},
                            _pf, default=str,
                        )
                except Exception as _e:  # noqa: BLE001
                    logger.warning(f"eval_{eval_id}: could not write performance.json: {_e}")

                self._save_eval_record(
                    eval_id=eval_id, config=cfg, pipeline_config=pipeline_config,
                    eval_type=eval_type, objectives=eval_objectives_dict,
                    quality_metrics=quality_metrics, assembly_result=None,
                    obj_columns=obj_columns,
                    execution_dag=measured_execution_dag,
                    performance_execution_trace=(
                        measured_performance_execution_trace
                    ),
                )
                self._update_summary()
                logger.info(
                    f"[progress] eval_{eval_id:04d} COMPLETE measured "
                    f"performance={perf_score:.6g} "
                    f"quality={quality_scores[-1].get('combined')} "
                    f"elapsed={time.monotonic() - eval_t0:.1f}s"
                )
                continue

            quality_metrics = None
            execution_dag = None
            cur_eval_type = eval_type
            fp_donor = None   # set when quality was fingerprint-inherited (M3)
            result = None
            gt_error = None
            cm_error = None

            self._configure_rag_ir_mode(pipeline_config)

            # ─── Quality memoization (quality ⊥ deployment) ─────────────────
            # Same algo sub-vector already measured → reuse its judge scores
            # AND its recorded trace (the CM's dynamic engine replays it);
            # only the system-dependent performance eval runs below.
            # ``_cm_overlap`` must exist on EVERY path — the CM step reads it;
            # on a memo hit no overlap slot is ever published ('current' stays
            # None → the cached trace prices through the synchronous branch).
            _cm_overlap = {'current': None}
            memo_hit = False
            _want_full = True
            if self._quality_memo_enabled:
                _want_full = (
                    pending_full[cfg_idx] if cfg_idx < len(pending_full) else True
                )
                _memo_key = self._quality_algo_key(cfg)
                _memo_entry = self._quality_memo.get(_memo_key)
                if _memo_entry is not None and _memo_entry.serves(_want_full):
                    # Dynamic CM pricing replays the quality run's trace — a
                    # hit without a usable trace would silently degrade this
                    # trial to static pricing. Treat it as a miss instead.
                    _memo_dag = _memo_entry.load_execution_dag()
                    _need_trace = (
                        (pipeline_config.get("_runtime") or {}).get("rag_ir_mode")
                        == "dynamic"
                    )
                    if _need_trace and not is_trace_payload(_memo_dag) and (
                        isinstance(_memo_dag, list)
                        or is_quality_trace_candidate(_memo_dag)
                    ):
                        # A resumed/seeded execution_dag in a PRE-FREEZE form
                        # (bare list / v1 trace_scope envelope). Re-running
                        # quality here would silently burn GPU+judge to
                        # reproduce a known number — fail loudly instead and
                        # point at the one-off converter.
                        raise RuntimeError(
                            f"eval_{eval_id:04d}: quality-memo execution_dag "
                            "is a legacy (retired) quality trace the CM no "
                            "longer accepts. Convert the project once before "
                            "resuming: python benchmarks/rag_cm_accuracy/"
                            "tools/convert_legacy_traces.py --project "
                            f"{self.project_dir} --scope auto"
                        )
                    if not _need_trace or is_trace_payload(_memo_dag):
                        quality_metrics = deepcopy(_memo_entry.quality_metrics)
                        if not _want_full and minimal_metrics is not None:
                            allowed_metric_names = {
                                str(metric.get("metric_name") or metric.get("metric"))
                                for metric in minimal_metrics
                                if isinstance(metric, dict)
                                and (metric.get("metric_name") or metric.get("metric"))
                            }
                            quality_metrics = {
                                name: value
                                for name, value in quality_metrics.items()
                                if name in allowed_metric_names
                            }
                        execution_dag = deepcopy(_memo_dag)
                        gt_error = None
                        memo_hit = True
                        self._gt_memo_hits += 1
                        logger.info(
                            f"[progress] eval_{eval_id:04d} quality MEMO HIT "
                            f"(algo_hash={_memo_key[:12]}, "
                            f"hits={self._gt_memo_hits}): reusing judge scores "
                            f"+ trace; skipping GPU+judge run"
                        )

            # ─── GT eval (LLM-judge pipeline): retry-able ───────────────────
            # Transient failures legit on GT side: vllm engine spawn races,
            # OpenRouter LLM-judge timeouts, retrieval flakes. Retry up to
            # _MAX_EVAL_RETRIES with backoff.
            if not memo_hit:
                # Judge/CM overlap (07-04): the trace is final before the judge
                # starts, and the CM is pure CPU while the judge is pure API wait
                # — price the trace CONCURRENTLY with scoring. Identity-matched
                # at the CM step below; retries re-fire with a fresh trace and
                # the stale thread's result is discarded.
                # One overlap slot PER ATTEMPT (07-06 race fix): _cm_overlap_start
                # binds a FRESH dict into its worker's closure and publishes it as
                # ['current']. A stale thread from an earlier (failed/retried)
                # attempt writes into its OWN dict and can never clobber the
                # accepted attempt's result/error — the previous single shared dict
                # let a slow attempt-1 thread's late error overwrite attempt-2's
                # valid CM result, spuriously failing a good config.
                # (_cm_overlap itself is initialized above the memo check —
                # every path reads it; this is the GT-path reset.)
                _cm_overlap = {'current': None}

                def _cm_overlap_start(_trace, _pc=None):
                    import threading as _threading

                    _pc = deepcopy(_pc or pipeline_config)
                    _backend = str((self.config.get("global") or {}).get("eval_backend", "static_gt")).lower()
                    normalize_backend_result(
                        {"__execution_dag__": _trace}, backend=_backend,
                        pipeline_config=_pc,
                        question_ids=(self.qa_data["qid"].tolist()
                                      if getattr(self, "qa_data", None) is not None else None),
                    )
                    _ov = {'thread': None, 'trace': _trace, 'result': None, 'error': None,
                           'workflow': deepcopy((_trace.get('provenance') or {}).get('workflow_schema'))}
                    def _work():
                        try:
                            if self._cm_fixed_provider is not None:
                                # Joint-space cell: price EXACTLY the sampled
                                # deployment (no sweep) — same overlap window,
                                # writing into this attempt's OWN slot (race fix).
                                _ov['result'] = self._cm_fixed_provider.evaluate(
                                    _pc, trial_system_config, trace=_trace,
                                )
                            else:
                                _schema = build_workflow_schema(_pc)
                                _ov['result'] = self.assembly.evaluate_dynamic(
                                    _pc, workflow_schema=_schema, trace=_trace,
                                )
                        except Exception as _e:  # noqa: BLE001 — joined at the CM step
                            _ov['error'] = _e
                    _t = _threading.Thread(target=_work, name=f'cm-overlap-{eval_id}', daemon=True)
                    _ov['thread'] = _t
                    _cm_overlap['current'] = _ov
                    _t.start()

                for attempt in range(1, _MAX_EVAL_RETRIES + 1):
                    try:
                        gt_run_dir = os.path.join(
                            self.project_dir, "evaluations",
                            f"eval_{eval_id:04d}", "static_run",
                        )
                        logger.info(
                            f"[progress] eval_{eval_id:04d} GT/judge attempt "
                            f"{attempt}/{_MAX_EVAL_RETRIES}"
                        )
                        # ax-gen evals (optimizer flags full=False) → objective + free local
                        # metrics only; skip the 5 deepeval sub-metrics the agent won't read.
                        _full = pending_full[cfg_idx] if cfg_idx < len(pending_full) else True
                        _mo = None if (_full or minimal_metrics is None) else minimal_metrics
                        if _mo is not None and attempt == 1:
                            logger.info(
                                f"[progress] eval_{eval_id:04d} minimal-metrics (ax-gen step): "
                                f"{[m.get('metric_name') for m in _mo]}"
                            )
                        quality_metrics = self.rag_quality_evaluator.evaluate(
                            pipeline_config, run_dir=gt_run_dir, metrics_override=_mo,
                            on_trace_ready=_cm_overlap_start,
                        )
                        quality_metrics = normalize_backend_result(
                            quality_metrics,
                            backend=(self.config.get("global") or {}).get("eval_backend", "static_gt"),
                            pipeline_config=pipeline_config,
                            question_ids=(self.qa_data["qid"].tolist()
                                          if getattr(self, "qa_data", None) is not None else None),
                        )
                        # DAG is for cost model only — pop so it never reaches
                        # optimizer via quality_scores.
                        execution_dag = quality_metrics.pop("__execution_dag__", None)
                        # Fingerprint-dedup marker (M3): quality inherited from
                        # this donor eval; pop so it never enters sub_metrics.
                        fp_donor = quality_metrics.pop("__fp_hit__", None)
                        gt_error = None
                        break
                    except Exception as e:
                        gt_error = e
                        gt_msg = _format_exception(e)
                        # UNIFIED retry policy (same as the measured arm):
                        # OOM / context-overflow are deterministic — no retry.
                        _ftype, _retryable = _eval_retry_decision(e)
                        # EXCEPTION (quality arm only): a CUDA OOM here is
                        # HARNESS pressure, not a config property — the GT
                        # evaluator may deploy however it likes (quality is
                        # deployment-independent), so running out of GPU says
                        # nothing about the sampled design. Retry (each
                        # attempt re-enters through the per-eval engine
                        # sweep); if it still cannot fit after all attempts,
                        # the post-loop abort below stops the run loudly
                        # instead of poisoning the search with a fake INVALID.
                        # (Measured-arm OOM keeps INVALID semantics: there the
                        # deployment IS the sampled config.)
                        if _ftype is EvalType.OOM:
                            _retryable = True
                        if not _retryable:
                            outcome = (
                                "Will stop run."
                                if _should_abort_eval_failure(e)
                                else "Will penalize quality."
                            )
                            logger.warning(
                                f"Config eval_{eval_id} GT eval failed with "
                                f"non-retryable {_ftype}: {gt_msg}. "
                                f"{outcome}"
                            )
                            break
                        if attempt < _MAX_EVAL_RETRIES:
                            logger.warning(
                                f"Config eval_{eval_id} GT eval failed "
                                f"(attempt {attempt}/{_MAX_EVAL_RETRIES}): "
                                f"{gt_msg}. Retrying..."
                            )
                            time.sleep(5 * attempt)
                        else:
                            outcome = (
                                "Will stop run."
                                if _should_abort_eval_failure(e)
                                or _ftype is EvalType.OOM
                                else "Will penalize quality."
                            )
                            logger.warning(
                                f"Config eval_{eval_id} GT eval failed after "
                                f"{_MAX_EVAL_RETRIES} attempts: {gt_msg}. "
                                f"{outcome}\n{traceback.format_exc()}"
                            )

                # The execution trace is finalized and published before the
                # quality metrics are judged.  A judge failure therefore does
                # not invalidate (or erase) the workload the CM must price.
                # ``evaluate()`` can only return ``__execution_dag__`` on its
                # success path, so recover the exact envelope already handed
                # to the overlap worker when scoring raised afterwards.  The
                # identity is intentionally preserved: the CM step below can
                # join the worker for this final attempt instead of launching
                # another evaluation or falling through to static pricing.
                if execution_dag is None:
                    _published_overlap = _cm_overlap.get("current")
                    _published_trace = (
                        _published_overlap.get("trace")
                        if isinstance(_published_overlap, dict)
                        else None
                    )
                    if (
                        is_trace_payload(_published_trace)
                        or is_quality_trace_candidate(_published_trace)
                    ):
                        execution_dag = _published_trace
                        logger.info(
                            "[progress] eval_%04d preserving the finalized "
                            "quality trace after judge failure for dynamic CM",
                            eval_id,
                        )

            _gt_oom = (
                gt_error is not None
                and EvalType.classify_failure(gt_error) is EvalType.OOM
            )
            if gt_error is not None and (
                _should_abort_eval_failure(gt_error) or _gt_oom
            ):
                gt_text = f"GT: {_format_exception(gt_error)}"
                # Static-GT runs use a quality-harness deployment independent
                # of the sampled system design.  Exhausting harness OOM retries
                # is therefore infrastructure failure, not evidence that this
                # config is infeasible.  Preserve an INVALID audit row and stop
                # at the last committed optimizer state; never blacklist it or
                # teach either the GP or the agent's infeasible advisory from
                # this failure.
                self._save_eval_record(
                    eval_id=eval_id, config=cfg, pipeline_config=pipeline_config,
                    eval_type=EvalType.INVALID,
                    eval_error=gt_text,
                    objectives={}, quality_metrics=quality_metrics,
                    assembly_result=None, obj_columns=obj_columns,
                    execution_dag=execution_dag,
                )
                self._update_summary()
                logger.error(
                    f"[progress] eval_{eval_id:04d} FAILED after "
                    f"{time.monotonic() - eval_t0:.1f}s; stopping run: "
                    f"{gt_text}"
                )
                raise RuntimeError(
                    f"Stopping optimization: eval_{eval_id:04d} GT/judge "
                    f"evaluation failed. {gt_text}"
                ) from gt_error

            # Memoize a FRESH successful quality eval (never failures, never
            # penalized zeros; a full entry overwrites a minimal one, never
            # the reverse — QualityMemoEntry.serves gates reads accordingly).
            if (
                self._quality_memo_enabled and not memo_hit
                and gt_error is None
                and isinstance(quality_metrics, dict) and quality_metrics
            ):
                from rag_stack.quality_memo import (
                    QualityMemoEntry,
                )
                _memo_key = self._quality_algo_key(cfg)
                _prev = self._quality_memo.get(_memo_key)
                if _prev is None or _want_full or not _prev.full:
                    self._quality_memo[_memo_key] = QualityMemoEntry(
                        quality_metrics=deepcopy(quality_metrics),
                        execution_dag=deepcopy(execution_dag),
                        full=_want_full,
                    )

                # The global store is a strict side cache, never optimizer
                # training data: only configs already chosen by the optimizer
                # reach this write-through. Persist metrics and the exact
                # frozen trace as one pair. Cache I/O cannot invalidate an
                # otherwise successful benchmark evaluation.
                if self._external_quality_store is not None:
                    try:
                        self._external_quality_store.append_record(
                            self._quality_algo_config(cfg),
                            pipeline_config,
                            quality_metrics,
                            execution_dag,
                            full=_want_full,
                            source={
                                "kind": "rag_stack_optimize",
                                "project_dir": self.project_dir,
                                "eval_id": eval_id,
                                "source_mtime_ns": time.time_ns(),
                            },
                            quality_spec=QualityStore.quality_spec(self.config),
                        )
                    except Exception as exc:  # noqa: BLE001 — optional cache
                        logger.warning(
                            "eval_%04d quality-store write-through skipped: %s",
                            eval_id, exc,
                        )

            # On resume, a project-local memo can be newer/more complete than
            # the attached global store. Copy it out only when the global
            # record is absent or cannot serve this full-metrics request. A
            # hit that originated in the global store is therefore read-only.
            if (
                memo_hit and self._external_quality_store is not None
                and isinstance(quality_metrics, dict) and quality_metrics
            ):
                _memo_key = self._quality_algo_key(cfg)
                external_record = self._external_quality_store.get_record(_memo_key)
                needs_copy = external_record is None or (
                    _want_full and not bool(external_record.get("full"))
                )
                if needs_copy:
                    try:
                        self._external_quality_store.append_record(
                            self._quality_algo_config(cfg),
                            pipeline_config,
                            quality_metrics,
                            execution_dag,
                            full=_want_full,
                            source={
                                "kind": "project_quality_memo",
                                "project_dir": self.project_dir,
                                "eval_id": eval_id,
                                "source_mtime_ns": time.time_ns(),
                            },
                            quality_spec=QualityStore.quality_spec(self.config),
                        )
                    except Exception as exc:  # noqa: BLE001 — optional cache
                        logger.warning(
                            "eval_%04d project-memo export skipped: %s",
                            eval_id, exc,
                        )

            # ─── Cost model: deterministic, NO retry ────────────────────────
            # Cost model is a pure analytical sweep over (batch, tp, pp,
            # num_devices). It either finds a feasible deployment or it
            # doesn't — retry never changes the outcome. Run once. (This is the
            # CM arm of the unified policy in _eval_retry_decision: an
            # analytically deterministic eval never enters a retry loop at all;
            # its failure is still classified with explicit CM provenance at
            # the record site below.)
            try:
                assembly_idx_before = len(self._assembly_results)
                bind_trace_workflow(pipeline_config, execution_dag)
                _gt_infeasible = gt_error is not None and EvalType.classify_failure(
                    gt_error
                ) in (EvalType.CONTEXT_OVERFLOW, EvalType.OOM)
                if _gt_infeasible:
                    # Skip CM for a config whose GT already failed with a GENUINE
                    # infeasibility (07-06): CONTEXT_OVERFLOW (prompt exceeds the
                    # window) / OOM means the SAMPLED CONFIG itself can't run, and
                    # its GT produced no trace — so an AGENTIC config here would
                    # mis-route to the sequential static engine and raise
                    # ("evaluate_static requires a sequential workflow_schema").
                    # Pricing a known-infeasible config is moot anyway. Leave
                    # result=None with NO cm_error → the failure handler penalizes
                    # it (quality=0, perf=0) as the genuine trough it is, so the
                    # GP learns to avoid the region (not excluded, not crashed).
                    result = None
                    cm_error = None
                elif (
                    is_trace_payload(execution_dag)
                    and _cm_overlap["current"] is not None
                    and _cm_overlap["current"]["trace"] is execution_dag
                    and _cm_overlap["current"].get("workflow") == (
                        execution_dag.get("provenance") or {}
                    ).get("workflow_schema")
                ):
                    # Judge/CM overlap hit: the CM was priced concurrently
                    # with the judge on EXACTLY this trace (identity match →
                    # it came from the attempt whose quality we accepted). Read
                    # the CURRENT attempt's own overlap dict, so a stale thread
                    # from a retried attempt can't inject its error/result here.
                    _ov = _cm_overlap["current"]
                    _ov["thread"].join(timeout=_CM_OVERLAP_TIMEOUT_S)
                    if _ov["thread"].is_alive():
                        raise OverlappedCMTimeoutError(
                            "overlapped CM did not finish within "
                            f"{_CM_OVERLAP_TIMEOUT_S:g}s"
                        )
                    if _ov["error"] is not None:
                        raise _ov["error"]
                    result = _ov["result"]
                    if result is None:
                        raise RuntimeError("overlapped CM returned no result")
                elif (
                    is_trace_payload(execution_dag)
                    or is_quality_trace_candidate(execution_dag)
                ):
                    # canonical envelope → priced; a malformed/legacy envelope
                    # candidate flows to normalize_quality_trace, whose
                    # directed error names the converter (fail closed —
                    # never silently down the static path).
                    if self._cm_fixed_provider is not None:
                        result = self._cm_fixed_provider.evaluate(
                            pipeline_config, trial_system_config,
                            trace=execution_dag,
                        )
                    else:
                        schema = build_workflow_schema(pipeline_config)
                        result = self.assembly.evaluate_dynamic(
                            pipeline_config,
                            workflow_schema=schema,
                            trace=execution_dag,
                        )
                elif execution_dag is not None and not isinstance(execution_dag, dict):
                    # Legacy bare-list trace (pre-freeze artifact): the static
                    # path would silently misread it as an aggregate. Directed
                    # error instead — convert the project once, offline.
                    raise RuntimeError(
                        f"eval_{eval_id:04d}: execution_dag holds a legacy "
                        "(retired) quality-trace payload the CM no longer "
                        "accepts. Convert it once: python benchmarks/"
                        "rag_cm_accuracy/tools/convert_legacy_traces.py "
                        f"--project {self.project_dir} --scope auto"
                    )
                elif getattr(
                    build_workflow_schema(pipeline_config),
                    "dataflow_type",
                    "sequential",
                ) != "sequential":
                    # Agentic/iterative performance is defined by the actual
                    # per-query stage visits and loop counts.  An aggregate or
                    # missing payload cannot be priced by StaticEngine.  Keep
                    # this guard at the controller dispatch boundary so neither
                    # a fixed-placement provider nor a future assembly change
                    # can silently reinterpret it as a sequential workflow.
                    raise RuntimeError(
                        f"eval_{eval_id:04d}: agentic/iterative workflow did "
                        "not publish a canonical quality trace; refusing "
                        "static CM dispatch"
                    )
                elif self._cm_fixed_provider is not None:
                    # NON-trace aggregate (static_gt rollup dict or None): the
                    # frozen contract carries it on schema.runtime_stats; the
                    # provider stamps it onto the schema it builds internally.
                    result = self._cm_fixed_provider.evaluate(
                        pipeline_config, trial_system_config,
                        runtime_stats=execution_dag,
                    )
                else:
                    schema = build_workflow_schema(pipeline_config)
                    schema.runtime_stats = execution_dag
                    result = self.assembly.evaluate_static(
                        pipeline_config,
                        workflow_schema=schema,
                    )
                if result is not None:
                    retained_result = (
                        _portable_assembly_result(result)
                        if getattr(
                            self, "_retain_portable_assembly_results", False,
                        )
                        else result
                    )
                    self._assembly_results.append(retained_result)
                cm_error = None
            except Exception as e:
                cm_error = e
                _cm_failure_action = (
                    "Will stop optimization."
                    if isinstance(e, OverlappedCMTimeoutError)
                    else "Will penalize performance."
                )
                logger.warning(
                    f"Config eval_{eval_id} cost-model failed (deterministic, no retry): {e}. "
                    f"{_cm_failure_action}"
                )
                # Full traceback (07-05): a CM CODE crash (e.g. a GenZ
                # pipeline-parallel indexing bug on disaggregated-PD configs)
                # is a modelling defect, not a genuine infeasibility — but the
                # penalty path buries it as a one-line message. Log the frame
                # so the exact GenZ site surfaces instead of a bare
                # "tuple index out of range".
                logger.warning(
                    "eval_%s CM traceback:\n%s", eval_id,
                    "".join(traceback.format_exception(type(e), e, e.__traceback__)),
                )

            # ─── Decide outcome ─────────────────────────────────────────────
            # Mixed-failure handling: GT and cost model are independent. Any
            # signal we DO have (GT or cost model) gets used; the missing one
            # is penalized to 0. The optimizer then learns where each
            # subsystem fails independently.
            if gt_error is not None or cm_error is not None:
                err_summary = []
                if gt_error is not None: err_summary.append(f"GT: {gt_error}")
                if cm_error is not None: err_summary.append(f"CM: {cm_error}")

                # Penalize-to-0 vs EXCLUDE (07-06, user directive): a 0 fed to
                # the GP says "this config is genuinely infeasible". That is
                # only TRUE for a config PROPERTY the sampler owns — the prompt
                # overflowing the generator's context window (CONTEXT_OVERFLOW,
                # GT-side) or the CM's own analytic deployment OOM (OOM,
                # CM-side). Everything else — a vLLM EngineCore death, a
                # GT-side harness OOM (the quality eval's deployment is the
                # harness's own, per _eval_retry_decision), a CM code crash, a
                # judge flake — is INFRASTRUCTURE noise that says nothing about
                # the config's quality/perf. Scoring those 0 POISONS the GP
                # (e.g. a 14B config that engine-dies gets quality=0, teaching
                # the surrogate that the BEST-quality region is the worst and
                # killing all quality-corner exploration). Such points are
                # recorded INVALID + blacklisted but NOT fed to the GP at all.
                failure_type, gt_genuine, cm_genuine, excluded = (
                    _classify_eval_arm_failures(gt_error, cm_error)
                )
                if excluded:
                    _err_text = " | ".join(err_summary)
                    self._save_eval_record(
                        eval_id=eval_id, config=cfg, pipeline_config=pipeline_config,
                        eval_type=failure_type,
                        eval_error=_err_text,
                        objectives={}, quality_metrics=quality_metrics,
                        assembly_result=None, obj_columns=obj_columns,
                        execution_dag=execution_dag,
                    )
                    self._update_summary()
                    logger.info(
                        f"[progress] eval_{eval_id:04d} INVALID (excluded from GP — "
                        f"infrastructure failure, not a config verdict) after "
                        f"{time.monotonic() - eval_t0:.1f}s: {_err_text}"
                    )
                    # This proposal has already advanced optimizer-side policy
                    # state (BO round, DOE/forced state, agent notes),
                    # but no replayable objective row exists to commit it.  Do
                    # not issue another suggestion in this process: that would
                    # silently spend policy state on infrastructure noise.  The
                    # durable commit boundary is add_objectives/forced_state;
                    # fail-fast lets a normal checkpoint resume reconstruct the
                    # exact last-landed optimizer state.  Also do NOT call
                    # _record_invalid/register_infeasible for an infrastructure
                    # failure: it is not a property of this sampled config.
                    logger.error(
                        f"[progress] eval_{eval_id:04d} infrastructure failure; "
                        "stopping after recording INVALID so optimizer proposal "
                        "state can be recovered from the last committed eval"
                    )
                    if isinstance(cm_error, OverlappedCMTimeoutError):
                        # A Python thread cannot be force-cancelled safely.  In
                        # production the CLI's exception path exits this process
                        # and its descendant reaper terminates the CM pool.
                        raise cm_error
                    raise RuntimeError(
                        f"Stopping optimization: eval_{eval_id:04d} had an "
                        f"infrastructure failure. {_err_text}"
                    ) from (gt_error or cm_error)

                # Only genuine config-property infeasibilities reach here — feed
                # the affected objective(s) a 0 penalty so the GP learns the
                # trough (CONTEXT_OVERFLOW → quality 0; CM OOM → perf 0).
                self._record_invalid(
                    eval_id,
                    cfg,
                    " | ".join(err_summary),
                    objectives_will_be_committed=True,
                )
                if gt_error is None and quality_metrics is not None:
                    # GT succeeded — use real quality.
                    quality_scores.append({
                        "combined": aggregate_quality_metrics(
                            quality_metrics, self._combined_quality_spec,
                        ),
                        **{k: v for k, v in quality_metrics.items() if v is not None}  # None = judge-errored: omit, so it stays distinguishable from a true 0.0
                    })
                else:
                    # GT failed — penalize quality to 0. Must include every
                    # declared metric, not just a partially returned subset.
                    sub_metric_names = [
                        m.get("metric_name")
                        for m in (self.config.get("eval_backend_setting") or {}).get("metrics", [])
                        if m.get("metric_name")
                    ]
                    if quality_metrics:
                        sub_metric_names = list({
                            *sub_metric_names, *quality_metrics.keys(),
                        })
                    quality_scores.append({
                        "combined": 0.0,
                        **{k: 0.0 for k in sub_metric_names},
                    })

                if cm_error is None and result is not None:
                    # Cost model succeeded — use real perf.
                    performance_scores.append(result.performance_score)
                else:
                    # Cost model failed — penalize perf to 0.
                    performance_scores.append(0.0)
                valid_configs.append(cfg)
                self._last_valid_eval_ids.append(eval_id)
                # Penalty rows are budget-FREE (see the measured-invalid site):
                # the GP gets the infeasibility trough, no seat is charged.
                free_flags.append(True)
                self._last_perf_attrs.append(None)  # failure path: no usable CM row
                _err_text = " | ".join(err_summary)
                # Persist the EXACT objectives fed to the optimizer (07-06,
                # valid-eval counting): the penalty row's perf/quality cells in
                # all_evaluations.csv now carry the 0-penalty (and any real
                # surviving objective), so a resume can replay this observation
                # into the GP (see _load_checkpoint + EvalType.reload_as_penalty)
                # instead of silently dropping the trough. Previously objectives
                # was saved as {} and the penalty vanished across resumes.
                _penalty_objectives: Dict[str, float] = {
                    "performance": performance_scores[-1],
                    "quality": quality_scores[-1],
                }
                self._save_eval_record(
                    eval_id=eval_id, config=cfg, pipeline_config=pipeline_config,
                    eval_type=failure_type,
                    eval_error=_err_text,
                    objectives=_penalty_objectives, quality_metrics=quality_metrics,
                    assembly_result=None, obj_columns=obj_columns,
                    execution_dag=execution_dag,
                )
                self._update_summary()
                logger.info(
                    f"[progress] eval_{eval_id:04d} {failure_type.value} after "
                    f"{time.monotonic() - eval_t0:.1f}s: {_err_text}"
                )
                continue

            quality_scores.append({
                "combined": aggregate_quality_metrics(
                    quality_metrics, self._combined_quality_spec,
                ),
                **{k: v for k, v in quality_metrics.items() if v is not None}  # None = judge-errored: omit, so it stays distinguishable from a true 0.0
            })
            perf_score = result.performance_score
            performance_scores.append(perf_score)

            valid_configs.append(cfg)
            self._last_valid_eval_ids.append(eval_id)
            if fp_donor is not None:
                # Fingerprint-inherited quality (M3): real objectives, but the
                # measurement carried zero new quality information — a FREE row
                # that never consumes a valid-eval budget seat.
                free_flags.append(True)
                cur_eval_type = EvalType.GT_FP
            else:
                free_flags.append(False)
                n_success += 1  # real measured objectives → consumes valid-eval budget

            # Get the assembly result if one was just appended
            assembly_result = None
            if len(self._assembly_results) > assembly_idx_before:
                # ``result`` is the full current CM object.  Transfer mode
                # stores only its portable summary in the long-lived list,
                # but persistence and immediate agent attribution must still
                # observe the full selected evaluation before it is released.
                assembly_result = result
            self._last_perf_attrs.append(_perf_attr_from_best(assembly_result))

            # Build objectives for this eval
            eval_objectives_dict: Dict[str, float] = {
                "performance": perf_score,
                "quality": quality_scores[-1],
            }

            # Save immediately to disk
            self._save_eval_record(
                eval_id=eval_id,
                config=cfg,
                pipeline_config=pipeline_config,
                eval_type=cur_eval_type,
                objectives=eval_objectives_dict,
                quality_metrics=quality_metrics,
                assembly_result=assembly_result,
                obj_columns=obj_columns,
                execution_dag=execution_dag,
                system_config=trial_system_config,
            )
            if gt_error is None and isinstance(quality_metrics, dict):
                # Memo marker on EVERY successful GT record: the algo-hash is
                # persisted WITH the record using the same fully enriched,
                # algo-only key as the global QualityStore, plus the reuse flag
                # for budget accounting (unique judge runs = GT records with
                # hit=false).
                try:
                    from rag_stack.quality_memo import (
                        MEMO_MARKER_FILENAME,
                    )
                    _mdir = os.path.join(
                        self.project_dir, "evaluations", f"eval_{eval_id:04d}",
                    )
                    os.makedirs(_mdir, exist_ok=True)
                    with open(os.path.join(_mdir, MEMO_MARKER_FILENAME), "w") as _mf:
                        json.dump(
                            {"algo_hash": self._quality_algo_key(cfg),
                             "hit": bool(memo_hit)},
                            _mf,
                        )
                except Exception as _e:  # noqa: BLE001 — marker is best-effort
                    logger.warning(
                        f"eval_{eval_id}: could not write memo marker: {_e}"
                    )

            # Heartbeat: update summary after each eval so the UI shows live progress
            self._update_summary()
            logger.info(
                f"[progress] eval_{eval_id:04d} COMPLETE "
                f"performance={perf_score:.6g} "
                f"quality={quality_scores[-1].get('combined')} "
                f"elapsed={time.monotonic() - eval_t0:.1f}s"
            )

        obj_dict["performance"] = performance_scores
        obj_dict["quality"] = quality_scores

        return obj_dict, valid_configs, n_success, free_flags

    def _target_replay_pipeline(
        self, quality_store: QualityStore, config_hash_value: str,
    ) -> dict:
        """Transplant one committed source pipeline onto the target system.

        Phase 2a transfers *evaluations*, not candidates in the new sampling
        space.  The strict canonical QualityStore commits the exact resolved pipeline
        together with its quality and trace; rebuilding its canonical optimizer
        config through today's search space can prune historical components or
        even change fixed module fields.  Therefore the committed pipeline is
        the replay source of truth.  Only target-owned deployment/runtime state
        is overlaid here: physical system, derived retrieval threads, optional
        local-chat transport, and authoritative target corpus facts.
        """
        record = quality_store.get_record(config_hash_value)
        if not isinstance(record, dict) or not isinstance(record.get("config"), dict):
            raise ValueError(
                "QualityStore record has no canonical algorithm config: "
                f"{config_hash_value}"
            )
        algo_config = QualityStore.algorithm_config(record["config"])
        if QualityStore.key_for_config(algo_config) != config_hash_value:
            raise ValueError(
                "QualityStore record hash does not match its canonical "
                f"algorithm config: {config_hash_value}"
            )
        try:
            pipeline_config = deepcopy(
                quality_store.get_pipeline_config(config_hash_value),
            )
        except Exception as stored_exc:
            raise RuntimeError(
                "System transfer could not load the committed resolved "
                f"pipeline for QualityStore record {config_hash_value}: "
                f"{_format_exception(stored_exc)}"
            ) from stored_exc
        if not isinstance(pipeline_config, dict):
            raise ValueError(
                "QualityStore committed pipeline is not a mapping for "
                f"record {config_hash_value}"
            )

        target_system = getattr(self, "_transfer_target_system", None)
        if target_system is None:
            target_system = (self.config or {}).get("system")
        if not isinstance(target_system, dict):
            raise ValueError("Target project config has no valid system block")
        resolved_target_system = deepcopy(target_system)

        # ``faiss_num_threads: null`` is a physical-policy input, not the
        # scalar consumed by CM.  Resolve only the target system runtime;
        # do not route the historical algorithm through the new search space.
        context = getattr(self, "_ctx", None)
        resolve_retrieval = getattr(context, "_resolve_system_retrieval", None)
        if callable(resolve_retrieval):
            target_container = {"system": resolved_target_system}
            resolve_retrieval(target_container, {})
            resolved_target_system = target_container["system"]
        pipeline_config["system"] = resolved_target_system

        # The transfer YAML's backend choice is materialised in the published
        # target config.  Apply that same transport to Phase 2a artifacts so a
        # local-chat transfer never persists or later executes a source URI.
        if config_uses_only_local_vllm(self.config or {}):
            pipeline_config, _ = project_to_vllm_chat(pipeline_config)
        # Match the production evaluation path: resolve the selected chunker
        # and stamp authoritative corpus/token facts (notably vectordb.N) onto
        # the replay pipeline before workflow-schema construction.
        chunker_params = (
            (pipeline_config.get("corpus_runtime") or {}).get("chunker") or {}
        )
        try:
            corpus_view = self._data.resolve_corpus(chunker_params)
            self._data.apply_corpus_view(pipeline_config, corpus_view)
        except Exception as exc:
            raise RuntimeError(
                "Target replay could not materialize authoritative corpus "
                f"facts for QualityStore record {config_hash_value}: "
                f"{_format_exception(exc)}"
            ) from exc
        target_system = getattr(self, "_transfer_target_system", None)
        if target_system is None:
            target_system = (self.config or {}).get("system")
        if not isinstance(target_system, dict):
            raise ValueError("Target project config has no valid system block")
        resolved_system = pipeline_config.get("system")
        if not isinstance(resolved_system, dict):
            raise ValueError(
                "Target resolver returned no valid system block for replay"
            )
        # _resolve_config starts from the target project and derives runtime
        # fields such as retrieval.faiss_num_threads.  Replacing that resolved
        # block with the raw YAML used to turn A100's target-derived 128 threads
        # back into null, which the CM adapters interpret as 1.  Verify that the
        # immutable physical/CM policy survived resolution, but preserve all
        # resolved runtime values.  performance_objectives is the sole public
        # replay override and is applied narrowly below.
        for key in (
            "system_hardware",
            "rag_cm",
            "performance_source",
            "system_space_owner",
            "gpu",
            "cpu",
            "available_gpus",
            "interconnect",
            "cm_search_space",
        ):
            if key in target_system and resolved_system.get(key) != target_system[key]:
                raise ValueError(
                    "Target resolver changed immutable system field "
                    f"{key!r} during replay"
                )
        if "performance_objectives" in target_system:
            resolved_system["performance_objectives"] = deepcopy(
                target_system["performance_objectives"],
            )
        return pipeline_config

    def _price_stored_quality_record(
        self,
        *,
        quality_store: QualityStore,
        config_hash_value: str,
        assembly: RAGCMAssembly,
    ) -> tuple[dict, dict, AssemblyResult]:
        """Price one exact quality trace with production CM.

        Missing, legacy or malformed traces fail in ``get_execution_dag``;
        there is deliberately no static fallback. The trace contract remains
        orthogonal to the cost-model selection.
        """
        pipeline_config = self._target_replay_pipeline(
            quality_store, config_hash_value,
        )
        trace = quality_store.get_execution_dag(config_hash_value)
        if trace is None:
            raise RuntimeError(
                f"QualityStore record {config_hash_value} has no canonical trace"
            )
        bind_trace_workflow(pipeline_config, trace)
        schema = build_workflow_schema(pipeline_config)
        result = assembly.evaluate_dynamic(
            pipeline_config,
            workflow_schema=schema,
            trace=trace,
        )
        try:
            performance = float(result.performance_score)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "Dynamic CM replay returned a non-numeric performance "
                f"for QualityStore record {config_hash_value}: "
                f"{getattr(result, 'performance_score', None)!r}"
            ) from exc
        if not math.isfinite(performance):
            raise ValueError(
                "Dynamic CM replay returned non-finite performance for "
                f"QualityStore record {config_hash_value}: {performance!r}"
            )
        return pipeline_config, trace, result

    def _atomic_transfer_json(self, path: str, payload: dict) -> None:
        """Commit one structurally validated transfer marker atomically."""

        validate_system_transfer_artifact(payload, source=path)
        import tempfile

        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            prefix=".system-transfer.", suffix=".tmp", dir=directory,
        )
        try:
            with os.fdopen(fd, "w") as stream:
                safe_json_dump(payload, stream, indent=2, default=str)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    def _transfer_record_marker_sha256(
        self, quality_store: QualityStore, config_hash_value: str,
    ) -> str:
        path = os.path.join(
            quality_store.store_path, "records", f"{config_hash_value}.json",
        )
        if not os.path.isfile(path):
            raise RuntimeError(
                "System transfer source record has no QualityStore commit "
                f"marker: {config_hash_value}"
            )
        return _file_sha256(path)

    @staticmethod
    def _transfer_eval_artifact_sha256(eval_dir: str) -> dict[str, str]:
        names = (
            "pipeline_config.json", "quality_metrics.json",
            "execution_dag.json", "assembly.json",
            _OPTIMIZER_OBSERVATION_FILENAME,
        )
        return {
            name: _file_sha256(os.path.join(eval_dir, name))
            for name in names
            if os.path.isfile(os.path.join(eval_dir, name))
        }

    def _validate_transfer_eval_artifact_sha256(
        self, *, marker: dict, eval_dir: str, expected_names: set[str],
    ) -> None:
        recorded = marker.get("artifact_sha256")
        if not isinstance(recorded, dict) or set(recorded) != expected_names:
            raise RuntimeError(
                f"System transfer commit at {eval_dir} has incomplete artifact checksums"
            )
        actual = self._transfer_eval_artifact_sha256(eval_dir)
        if actual != recorded:
            raise RuntimeError(
                f"System transfer committed artifacts changed at {eval_dir}"
            )

    @staticmethod
    def _resolved_pipeline_uses_llmlingua2(pipeline_config: dict) -> bool:
        """Inspect only the committed *resolved* pipeline dependency graph.

        ``algo_search_space`` deliberately contains every candidate component,
        so recursively searching the complete document would classify all 52
        paper observations as LLMLingua2 users.  Root-level ``node_lines`` is
        the committed active graph after optional-node pruning and scalar
        resolution; it is the only accepted dependency source here.
        """
        if not isinstance(pipeline_config, dict):
            raise RuntimeError(
                "System transfer source pipeline is not a mapping"
            )
        node_lines = pipeline_config.get("node_lines")
        if not isinstance(node_lines, list):
            raise RuntimeError(
                "System transfer source pipeline has no resolved root-level "
                "node_lines dependency graph"
            )
        for node_line in node_lines:
            if not isinstance(node_line, dict):
                raise RuntimeError(
                    "System transfer resolved node_lines contains a non-mapping"
                )
            nodes = node_line.get("nodes")
            if not isinstance(nodes, list):
                raise RuntimeError(
                    "System transfer resolved node line has no nodes list"
                )
            for node in nodes:
                if not isinstance(node, dict):
                    raise RuntimeError(
                        "System transfer resolved node is not a mapping"
                    )
                if node.get("stage") != "passage_compressor":
                    continue
                modules = node.get("modules")
                if not isinstance(modules, list) or not modules:
                    raise RuntimeError(
                        "Active passage_compressor has no resolved modules"
                    )
                for module in modules:
                    if not isinstance(module, dict):
                        raise RuntimeError(
                            "Resolved passage_compressor module is not a mapping"
                        )
                    if module.get("component") == "llmlingua2":
                        return True
        return False

    def _llmlingua2_transfer_profile_identity(self) -> dict:
        """Freeze the exact CM profile file consumed by this target."""
        from rag_stack.calibration_paths import calibration_profile_root

        target_system = getattr(self, "_transfer_target_system", None)
        if target_system is None:
            target_system = (self.config or {}).get("system")
        if not isinstance(target_system, dict):
            raise RuntimeError(
                "Cannot resolve LLMLingua2 dependency without target system"
            )
        raw_gpu = target_system.get("gpu")
        if isinstance(raw_gpu, str):
            gpu_key = raw_gpu.strip()
        elif isinstance(raw_gpu, dict):
            gpu_key = str(raw_gpu.get("hardware_key") or "").strip()
        else:
            gpu_key = ""
        if not gpu_key:
            raise RuntimeError(
                "Selective LLMLingua2 transfer requires a stable target GPU "
                "hardware key"
            )
        relative_path = os.path.join(
            "compressor_llmlingua2", f"{gpu_key}.json",
        )
        profile_path = os.path.realpath(
            os.path.join(str(calibration_profile_root()), relative_path),
        )
        if not os.path.isfile(profile_path):
            raise RuntimeError(
                "Selective system transfer cannot freeze missing CM "
                f"LLMLingua2 profile: {profile_path}"
            )
        return {
            "logical_path": os.path.join(
                "calibration_profiles", relative_path,
            ),
            "resolved_path": profile_path,
            "sha256": _file_sha256(profile_path),
        }

    @staticmethod
    def _assert_transfer_dependency_unchanged(dependency: dict) -> None:
        """Fail before CM when a frozen paper-benchmark dependency changed."""
        if not isinstance(dependency, dict):
            raise RuntimeError(
                "System transfer replay plan has a malformed profile dependency"
            )
        path = dependency.get("resolved_path")
        expected_sha = dependency.get("sha256")
        if not isinstance(path, str) or not isinstance(expected_sha, str):
            raise RuntimeError(
                "System transfer replay plan has an incomplete profile dependency"
            )
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Frozen system-transfer dependency disappeared: {path}"
            )
        actual_sha = _file_sha256(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                "Frozen system-transfer dependency changed before CM replay: "
                f"{path} ({expected_sha} -> {actual_sha})"
            )

    @staticmethod
    def _transfer_csv_rows_from_project(project_dir: str) -> dict[int, dict]:
        path = os.path.join(project_dir, "all_evaluations.csv")
        if not os.path.isfile(path):
            raise RuntimeError(
                f"Prior target transfer has no all_evaluations.csv: {project_dir}"
            )
        with open(path, newline="") as stream:
            rows = list(csv.DictReader(stream))
        indexed: dict[int, dict] = {}
        for row in rows:
            try:
                eval_id = int(row.get("eval_id", ""))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Prior target transfer CSV has a non-integer eval_id"
                ) from exc
            if eval_id in indexed:
                raise RuntimeError(
                    f"Prior target transfer CSV has duplicate eval_id {eval_id}"
                )
            indexed[eval_id] = row
        return indexed

    def _assert_transfer_pipeline_targets_current_system(
        self, *, pipeline_config: dict, context: str,
    ) -> None:
        if not isinstance(pipeline_config, dict):
            raise RuntimeError(f"{context} pipeline is not a mapping")
        resolved_system = pipeline_config.get("system") or {}
        target_system = getattr(self, "_transfer_target_system", {}) or {}
        if not isinstance(resolved_system, dict):
            raise RuntimeError(f"{context} pipeline has no target system block")
        for key in (
            "system_hardware", "rag_cm", "performance_source",
            "system_space_owner", "gpu", "cpu", "available_gpus",
            "interconnect", "cm_search_space",
        ):
            if key not in target_system:
                continue
            expected_value = _strip_secrets(target_system[key])
            actual_value = resolved_system.get(key)
            if key == "cm_search_space" and isinstance(
                expected_value, dict
            ) and isinstance(actual_value, dict):
                configured_mismatch = {
                    axis: (actual_value.get(axis), value)
                    for axis, value in expected_value.items()
                    if actual_value.get(axis) != value
                }
                unexpected_axes = set(actual_value) - set(expected_value) - {
                    "dec_steps", "seq_len_inference_prefill",
                }
                if not configured_mismatch and not unexpected_axes:
                    continue
            if actual_value != expected_value:
                raise RuntimeError(
                    f"{context} target system mismatch for {key!r}"
                )

    @staticmethod
    def _prior_transfer_state_identity(state: dict) -> tuple[dict, str]:
        """Hash only the prior transfer contract, never mutable run status.

        A fully committed Phase-2a source may legitimately advance from
        ``running`` to ``completed`` while a selective child is in progress.
        Status, result SHA, and refinement budgets/counters are therefore not
        lineage identity. CM/target/source order (and a prior selective
        plan, when present) are immutable and are the exact projection pinned.
        """
        if not isinstance(state, dict):
            raise RuntimeError("Prior system-transfer state is not a mapping")
        validate_system_transfer_artifact(
            state,
            expected_kind="system_transfer_state",
            source="prior system-transfer state",
        )
        fields = (
            "kind",
            "cost_model_version",
            "target_config_sha256",
            "target_system_sha256",
            "quality_store_path",
            "source_records",
            "replay_plan",
        )
        identity = {
            key: deepcopy(state[key])
            for key in fields
            if key in state
        }
        return identity, _canonical_payload_sha256(identity)

    @classmethod
    def _compatible_prior_transfer_identity_sha256s(
        cls, state: dict,
    ) -> set[str]:
        """Return canonical plus pre-budget-extension lineage identities.

        Early selective-transfer plans hashed the then-immutable requested
        refinement count.  Existing paper projects retain those hashes in
        their replay plans and per-eval lineage.  Accept the legacy hash for
        the current or any recorded historical budget while keeping every
        genuinely immutable identity field pinned.
        """
        identity, canonical_sha = cls._prior_transfer_state_identity(state)
        budgets: set[int] = set()
        current = state.get("refinement_iterations_requested")
        if isinstance(current, int) and not isinstance(current, bool):
            budgets.add(current)
        history = state.get("refinement_budget_extensions", [])
        if isinstance(history, list):
            for extension in history:
                if not isinstance(extension, dict):
                    continue
                for key in ("from", "to"):
                    value = extension.get(key)
                    if isinstance(value, int) and not isinstance(value, bool):
                        budgets.add(value)
        compatible = {canonical_sha}
        for budget in budgets:
            legacy_identity = deepcopy(identity)
            legacy_identity["refinement_iterations_requested"] = budget
            compatible.add(_canonical_payload_sha256(legacy_identity))
        return compatible

    def _validate_prior_target_replay(
        self,
        *,
        prior_project: str,
        prior_state: dict,
        prior_state_identity_sha256: str,
        replay_idx: int,
        quality_store: QualityStore,
        quality_record: dict,
        csv_row: dict,
    ) -> dict:
        """Validate one reusable prior observation and return frozen lineage."""
        algo_hash = quality_record["config_hash"]
        eval_dir = os.path.join(
            prior_project, "evaluations", f"eval_{replay_idx:04d}",
        )
        marker_path = os.path.join(eval_dir, "transfer_provenance.json")
        if not os.path.isfile(marker_path):
            raise RuntimeError(
                "Prior target transfer is missing replay provenance at "
                f"eval_{replay_idx:04d}"
            )
        with open(marker_path) as stream:
            marker = json.load(stream)
        validate_system_transfer_artifact(
            marker,
            expected_kind="system_transfer_replay",
            source=marker_path,
        )
        source_record = quality_record.get("record") or {}
        source_state = prior_state["source_records"][replay_idx]
        expected = {
            "kind": "system_transfer_replay",
            "source_index": replay_idx,
            "source_algo_hash": algo_hash,
            "source_record_marker_sha256": source_state[
                "record_marker_sha256"
            ],
            "target_config_sha256": prior_state["target_config_sha256"],
            "source": source_record.get("source") or {},
            "source_eval_type": source_record.get("source_eval_type"),
            "source_eval_error": source_record.get("source_eval_error"),
            "quality_reused": True,
            "cost_model_version": 1,
        }
        mismatch = {
            key: (marker.get(key), value)
            for key, value in expected.items()
            if marker.get(key) != value
        }
        if mismatch:
            raise RuntimeError(
                "Prior target replay provenance mismatch at "
                f"eval_{replay_idx:04d}: {mismatch}"
            )
        try:
            target_type = EvalType(marker["target_eval_type"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Prior target replay has invalid outcome at eval_{replay_idx:04d}"
            ) from exc
        if target_type not in (EvalType.GT_FP, EvalType.OOM):
            raise RuntimeError(
                "Prior target replay has unsupported outcome "
                f"{target_type.value} at eval_{replay_idx:04d}"
            )
        if csv_row.get("eval_type") != target_type.value:
            raise RuntimeError(
                "Prior target replay provenance/CSV outcome mismatch at "
                f"eval_{replay_idx:04d}"
            )

        artifact_names = {
            "pipeline_config.json", "quality_metrics.json",
            "execution_dag.json",
        }
        if target_type is not EvalType.OOM:
            artifact_names.add("assembly.json")
        recorded_artifacts = marker.get("artifact_sha256")
        if (
            isinstance(recorded_artifacts, dict)
            and _OPTIMIZER_OBSERVATION_FILENAME in recorded_artifacts
        ):
            artifact_names.add(_OPTIMIZER_OBSERVATION_FILENAME)
        if not isinstance(recorded_artifacts, dict) or set(
            recorded_artifacts
        ) != artifact_names:
            raise RuntimeError(
                "Prior target replay has incomplete artifact checksums at "
                f"eval_{replay_idx:04d}"
            )
        actual_artifacts = self._transfer_eval_artifact_sha256(eval_dir)
        if actual_artifacts != recorded_artifacts:
            raise RuntimeError(
                "Prior target replay artifacts changed at "
                f"eval_{replay_idx:04d}"
            )

        def _read_artifact(name: str) -> dict:
            path = os.path.join(eval_dir, name)
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"Prior target replay is missing {name} at eval_{replay_idx:04d}"
                )
            with open(path) as stream:
                payload = json.load(stream)
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"Prior target replay {name} is not a mapping at "
                    f"eval_{replay_idx:04d}"
                )
            return payload

        persisted_pipeline = _read_artifact("pipeline_config.json")
        persisted_quality = _read_artifact("quality_metrics.json")
        persisted_trace = _read_artifact("execution_dag.json")
        if persisted_quality != (quality_record.get("quality_metrics") or {}):
            raise RuntimeError(
                f"Prior target replay quality changed at eval_{replay_idx:04d}"
            )
        if persisted_trace != quality_store.get_execution_dag(algo_hash):
            raise RuntimeError(
                f"Prior target replay trace changed at eval_{replay_idx:04d}"
            )
        self._assert_transfer_pipeline_targets_current_system(
            pipeline_config=persisted_pipeline,
            context=f"Prior target replay eval_{replay_idx:04d}",
        )

        combined = float(quality_record["quality_combined"])
        if target_type is EvalType.OOM:
            performance = 0.0
            if os.path.isfile(os.path.join(eval_dir, "assembly.json")):
                raise RuntimeError(
                    "Prior target OOM replay unexpectedly has assembly at "
                    f"eval_{replay_idx:04d}"
                )
        else:
            assembly_payload = _read_artifact("assembly.json")
            try:
                performance = float(assembly_payload["performance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Prior target replay has invalid performance at "
                    f"eval_{replay_idx:04d}"
                ) from exc
            if not math.isfinite(performance):
                raise RuntimeError(
                    "Prior target replay has non-finite performance at "
                    f"eval_{replay_idx:04d}"
                )
        try:
            csv_performance = float(csv_row.get("performance", ""))
            csv_quality = float(csv_row.get("quality__combined", ""))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Prior target replay CSV objectives are missing at "
                f"eval_{replay_idx:04d}"
            ) from exc
        if not math.isclose(csv_performance, performance, abs_tol=5e-6):
            raise RuntimeError(
                "Prior target replay performance/CSV mismatch at "
                f"eval_{replay_idx:04d}"
            )
        if not math.isclose(csv_quality, combined, abs_tol=5e-6):
            raise RuntimeError(
                "Prior target replay quality/CSV mismatch at "
                f"eval_{replay_idx:04d}"
            )
        return {
            "source_index": replay_idx,
            "source_algo_hash": algo_hash,
            "prior_eval_id": replay_idx,
            "prior_target_project": prior_project,
            "prior_system_transfer_identity_sha256": (
                prior_state_identity_sha256
            ),
            "prior_transfer_provenance_sha256": _file_sha256(marker_path),
            "prior_artifact_sha256": deepcopy(recorded_artifacts),
            "target_eval_type": target_type.value,
            "target_eval_error": str(csv_row.get("eval_error") or ""),
            "performance": performance,
        }

    def _build_selective_transfer_replay_plan(
        self,
        *,
        quality_store: QualityStore,
        quality_results: List[dict],
        reuse_target_project: str | None,
        retransfer_source_indices: Sequence[int] | None,
    ) -> dict | None:
        """Build and fully validate the immutable selective replay plan."""
        if reuse_target_project is None and retransfer_source_indices is None:
            return None
        if reuse_target_project is None or retransfer_source_indices is None:
            raise ValueError(
                "reuse_target_project and retransfer_source_indices must be "
                "provided together"
            )
        if isinstance(retransfer_source_indices, (str, bytes)):
            raise ValueError("retransfer_source_indices must be integer indices")
        requested: list[int] = []
        for value in retransfer_source_indices:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    "retransfer_source_indices must contain only integers"
                )
            requested.append(value)
        if not requested:
            raise ValueError(
                "Selective system transfer requires at least one retransfer index"
            )
        if len(set(requested)) != len(requested):
            raise ValueError("retransfer_source_indices contains duplicates")
        invalid = [
            index for index in requested
            if index < 0 or index >= len(quality_results)
        ]
        if invalid:
            raise ValueError(
                "retransfer_source_indices are outside the frozen source set: "
                f"{invalid}"
            )
        requested = sorted(requested)

        active_indices: list[int] = []
        for source_index, record in enumerate(quality_results):
            pipeline = quality_store.get_pipeline_config(record["config_hash"])
            if self._resolved_pipeline_uses_llmlingua2(pipeline):
                active_indices.append(source_index)
        if requested != active_indices:
            raise RuntimeError(
                "Selective system transfer dependency mismatch: requested "
                f"CM indices {requested}, but the exact active LLMLingua2 "
                f"set is {active_indices}"
            )
        profile_identity = self._llmlingua2_transfer_profile_identity()
        self._assert_transfer_dependency_unchanged(profile_identity)

        prior_project = os.path.realpath(reuse_target_project)
        if not os.path.isdir(prior_project):
            raise ValueError(
                f"Prior target system-transfer project not found: {prior_project}"
            )
        if prior_project == os.path.realpath(self.project_dir):
            raise ValueError(
                "Selective replay prior target must differ from the new project"
            )
        prior_state_path = os.path.join(
            prior_project, _SYSTEM_TRANSFER_STATE_FILENAME,
        )
        if not os.path.isfile(prior_state_path):
            raise RuntimeError(
                "Prior target project has no immutable system-transfer state"
            )
        with open(prior_state_path) as stream:
            prior_state = json.load(stream)
        _, prior_state_identity_sha = self._prior_transfer_state_identity(
            prior_state,
        )
        target_config_sha, target_system_sha = (
            self._system_transfer_target_identity()
        )
        expected_prior_state = {
            "kind": "system_transfer_state",
            "cost_model_version": 1,
            "target_config_sha256": target_config_sha,
            "target_system_sha256": target_system_sha,
        }
        prior_mismatch = {
            key: (prior_state.get(key), value)
            for key, value in expected_prior_state.items()
            if prior_state.get(key) != value
        }
        if prior_mismatch:
            raise RuntimeError(
                "Prior target project does not match the current CM/target "
                f"target identity: {prior_mismatch}"
            )
        if prior_state.get("status") not in {"running", "completed"}:
            raise RuntimeError(
                "Prior target system-transfer state has invalid status"
            )
        prior_config_path = os.path.join(prior_project, CONFIG_FILENAME)
        if not os.path.isfile(prior_config_path):
            raise RuntimeError("Prior target project has no target config")
        prior_config = load_yaml_config(prior_config_path)
        if _canonical_payload_sha256(prior_config) != target_config_sha:
            raise RuntimeError(
                "Prior target config content does not match its transfer state"
            )
        if _canonical_payload_sha256(
            prior_config.get("system") or {}
        ) != target_system_sha:
            raise RuntimeError(
                "Prior target system content does not match its transfer state"
            )

        prior_source_records = prior_state.get("source_records")
        exact_source_order = (
            isinstance(prior_source_records, list)
            and len(prior_source_records) == len(quality_results)
            and all(
                isinstance(entry, dict)
                and entry.get("source_index") == source_index
                and entry.get("algo_hash") == record["config_hash"]
                and isinstance(entry.get("record_marker_sha256"), str)
                for source_index, (entry, record) in enumerate(zip(
                    prior_source_records, quality_results,
                ))
            )
        )
        if not exact_source_order:
            raise RuntimeError(
                "Prior target source-record order/identity does not exactly "
                "match the current frozen QualityStore"
            )
        prior_rows = self._transfer_csv_rows_from_project(prior_project)
        reused_indices = [
            index for index in range(len(quality_results))
            if index not in set(requested)
        ]
        reuse_lineage: list[dict] = []
        for replay_idx in reused_indices:
            row = prior_rows.get(replay_idx)
            if row is None:
                raise RuntimeError(
                    "Prior target transfer is missing reusable CSV row "
                    f"eval_{replay_idx:04d}"
                )
            reuse_lineage.append(self._validate_prior_target_replay(
                prior_project=prior_project,
                prior_state=prior_state,
                prior_state_identity_sha256=prior_state_identity_sha,
                replay_idx=replay_idx,
                quality_store=quality_store,
                quality_record=quality_results[replay_idx],
                csv_row=row,
            ))
        plan = {
            "kind": "selective_system_transfer_replay_plan",
            "prior_target_project": prior_project,
            "prior_system_transfer_identity_sha256": (
                prior_state_identity_sha
            ),
            "retransfer_source_indices": requested,
            "reuse_source_indices": reused_indices,
            "llmlingua2_active_source_indices": active_indices,
            "llmlingua2_profile": profile_identity,
            "reuse_lineage": reuse_lineage,
        }
        plan["plan_sha256"] = _canonical_payload_sha256(plan)
        self._validate_selective_replay_plan_contract(
            plan, n_sources=len(quality_results),
        )
        return plan

    @staticmethod
    def _validate_selective_replay_plan_contract(
        plan: dict | None, *, n_sources: int,
    ) -> None:
        if plan is None:
            return
        if not isinstance(plan, dict):
            raise RuntimeError("System transfer replay plan is not a mapping")
        validate_system_transfer_artifact(
            plan,
            expected_kind="selective_system_transfer_replay_plan",
            source="system-transfer replay plan",
        )
        expected_header = {"kind": "selective_system_transfer_replay_plan"}
        mismatch = {
            key: (plan.get(key), value)
            for key, value in expected_header.items()
            if plan.get(key) != value
        }
        if mismatch:
            raise RuntimeError(
                f"System transfer replay plan header mismatch: {mismatch}"
            )
        selected = plan.get("retransfer_source_indices")
        reused = plan.get("reuse_source_indices")
        active = plan.get("llmlingua2_active_source_indices")
        if (
            not isinstance(selected, list)
            or not isinstance(reused, list)
            or not isinstance(active, list)
            or selected != sorted(set(selected))
            or reused != sorted(set(reused))
            or active != selected
            or sorted(selected + reused) != list(range(n_sources))
            or set(selected).intersection(reused)
        ):
            raise RuntimeError(
                "System transfer replay plan index/dependency partition is invalid"
            )
        lineage = plan.get("reuse_lineage")
        if not isinstance(lineage, list) or [
            entry.get("source_index") if isinstance(entry, dict) else None
            for entry in lineage
        ] != reused:
            raise RuntimeError(
                "System transfer replay plan reuse lineage is not exact/ordered"
            )
        profile = plan.get("llmlingua2_profile")
        if not isinstance(profile, dict):
            raise RuntimeError(
                "System transfer replay plan has no LLMLingua2 dependency"
            )
        recorded_sha = plan.get("plan_sha256")
        unsigned_plan = {
            key: value for key, value in plan.items() if key != "plan_sha256"
        }
        actual_sha = _canonical_payload_sha256(unsigned_plan)
        if recorded_sha != actual_sha:
            raise RuntimeError(
                "System transfer replay plan SHA256 integrity check failed"
            )

    def _frozen_transfer_source_view(
        self, quality_results: List[dict],
    ) -> List[dict]:
        """Preview the frozen Phase-2a order before rebuilding a resume plan.

        Phase 2b may append records to the same QualityStore.  The authoritative
        validation remains in :meth:`_prepare_system_transfer_state`; this
        preview merely excludes those later cache records so regenerating the
        immutable selective plan observes the original source set.
        """
        state_path = os.path.join(
            self.project_dir, _SYSTEM_TRANSFER_STATE_FILENAME,
        )
        if not os.path.isfile(state_path):
            return quality_results
        with open(state_path) as stream:
            state = json.load(stream)
        entries = state.get("source_records")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError(
                "System transfer state has no frozen source record order"
            )
        indexed = {record.get("config_hash"): record for record in quality_results}
        if len(indexed) != len(quality_results):
            raise RuntimeError(
                "System transfer QualityStore contains duplicate config hashes"
            )
        frozen: List[dict] = []
        for source_index, entry in enumerate(entries):
            if not isinstance(entry, dict) or entry.get("source_index") != source_index:
                raise RuntimeError(
                    "System transfer state has malformed frozen source order"
                )
            record = indexed.get(entry.get("algo_hash"))
            if record is None:
                raise RuntimeError(
                    "System transfer QualityStore lost frozen source record "
                    f"{source_index}: {entry.get('algo_hash')!r}"
                )
            frozen.append(record)
        return frozen

    @staticmethod
    def _transfer_reuse_lineage(plan: dict, replay_idx: int) -> dict:
        entries = plan.get("reuse_lineage") if isinstance(plan, dict) else None
        if not isinstance(entries, list):
            raise RuntimeError(
                "Selective transfer replay plan has no reuse lineage"
            )
        matches = [
            entry for entry in entries
            if isinstance(entry, dict) and entry.get("source_index") == replay_idx
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "Selective transfer replay plan has missing/duplicate reuse "
                f"lineage for source index {replay_idx}"
            )
        return deepcopy(matches[0])

    @staticmethod
    def _transfer_replay_action(
        state: dict, replay_idx: int,
    ) -> tuple[str, dict | None]:
        plan = state.get("replay_plan")
        if plan is None:
            return _TRANSFER_ACTION_RECOMPUTED, None
        if not isinstance(plan, dict):
            raise RuntimeError("System transfer state has malformed replay plan")
        selected = plan.get("retransfer_source_indices")
        reused = plan.get("reuse_source_indices")
        if not isinstance(selected, list) or not isinstance(reused, list):
            raise RuntimeError("System transfer replay plan has incomplete index sets")
        in_selected = replay_idx in selected
        in_reused = replay_idx in reused
        if in_selected == in_reused:
            raise RuntimeError(
                "System transfer replay plan does not assign source index "
                f"{replay_idx} exactly once"
            )
        if in_selected:
            dependency = plan.get("llmlingua2_profile")
            if not isinstance(dependency, dict):
                raise RuntimeError(
                    "Selected replay has no frozen LLMLingua2 profile identity"
                )
            return _TRANSFER_ACTION_RECOMPUTED, deepcopy(dependency)
        return _TRANSFER_ACTION_REUSED, None

    def _load_validated_prior_target_replay(
        self,
        *,
        replay_idx: int,
        quality_store: QualityStore,
        quality_record: dict,
        replay_plan: dict,
    ) -> tuple[dict, dict, AssemblyResult | None, EvalType, str, float]:
        """Revalidate and load one lightweight prior-target observation."""
        prior_project = replay_plan["prior_target_project"]
        prior_state_path = os.path.join(
            prior_project, _SYSTEM_TRANSFER_STATE_FILENAME,
        )
        expected_state_identity_sha = replay_plan[
            "prior_system_transfer_identity_sha256"
        ]
        if not os.path.isfile(prior_state_path):
            raise RuntimeError("Prior target system-transfer state disappeared")
        with open(prior_state_path) as stream:
            prior_state = json.load(stream)
        if prior_state.get("status") not in {"running", "completed"}:
            raise RuntimeError(
                "Prior target system-transfer state has invalid status"
            )
        compatible_state_identities = (
            self._compatible_prior_transfer_identity_sha256s(prior_state)
        )
        if expected_state_identity_sha not in compatible_state_identities:
            raise RuntimeError(
                "Prior target immutable system-transfer identity changed after "
                "the replay plan was frozen"
            )
        prior_rows = self._transfer_csv_rows_from_project(prior_project)
        row = prior_rows.get(replay_idx)
        if row is None:
            raise RuntimeError(
                f"Prior target CSV row disappeared for eval_{replay_idx:04d}"
            )
        actual_lineage = self._validate_prior_target_replay(
            prior_project=prior_project,
            prior_state=prior_state,
            # Preserve the exact identity already frozen in this plan.  It may
            # be the legacy budget-bearing hash of an existing project.
            prior_state_identity_sha256=expected_state_identity_sha,
            replay_idx=replay_idx,
            quality_store=quality_store,
            quality_record=quality_record,
            csv_row=row,
        )
        frozen_lineage = self._transfer_reuse_lineage(
            replay_plan, replay_idx,
        )
        if actual_lineage != frozen_lineage:
            raise RuntimeError(
                "Prior target reuse lineage changed after replay plan was frozen "
                f"for eval_{replay_idx:04d}"
            )
        eval_dir = os.path.join(
            prior_project, "evaluations", f"eval_{replay_idx:04d}",
        )
        with open(os.path.join(eval_dir, "pipeline_config.json")) as stream:
            pipeline_config = json.load(stream)
        with open(os.path.join(eval_dir, "execution_dag.json")) as stream:
            trace = json.load(stream)
        target_type = EvalType(frozen_lineage["target_eval_type"])
        performance = float(frozen_lineage["performance"])
        result: AssemblyResult | None = None
        if target_type is not EvalType.OOM:
            with open(os.path.join(eval_dir, "assembly.json")) as stream:
                assembly_payload = json.load(stream)
            result = AssemblyResult(
                performance_score=performance,
                best_hw_params=deepcopy(
                    assembly_payload.get("best_hw_params") or {}
                ),
            )
        return (
            pipeline_config,
            trace,
            result,
            target_type,
            str(frozen_lineage.get("target_eval_error") or ""),
            performance,
        )

    def _prepare_system_transfer_state(
        self,
        *,
        quality_store: QualityStore,
        quality_results: List[dict],
        n_refine_iterations: int,
        replay_plan: dict | None = None,
    ) -> tuple[dict, List[dict]]:
        """Create or verify the immutable identity of a resumable transfer.

        The QualityStore is allowed to grow with Phase 2b cache records.  The
        initial Phase 2a source set is therefore frozen here as an ordered list
        of algo hashes plus each record's checksummed commit-marker identity.
        """
        self._validate_selective_replay_plan_contract(
            replay_plan, n_sources=len(quality_results),
        )
        state_path = os.path.join(
            self.project_dir, _SYSTEM_TRANSFER_STATE_FILENAME,
        )
        target_config_sha, target_system_sha = (
            self._system_transfer_target_identity()
        )
        store_path = os.path.realpath(quality_store.store_path)

        if os.path.isfile(state_path):
            with open(state_path) as stream:
                state = json.load(stream)
            validate_system_transfer_artifact(
                state,
                expected_kind="system_transfer_state",
                source=state_path,
            )
            expected_scalars = {
                "kind": "system_transfer_state",
                "cost_model_version": 1,
                "target_config_sha256": target_config_sha,
                "target_system_sha256": target_system_sha,
                "quality_store_path": store_path,
                "replay_plan": replay_plan,
            }
            mismatch = {
                key: (state.get(key), value)
                for key, value in expected_scalars.items()
                if state.get(key) != value
            }
            if mismatch:
                raise RuntimeError(
                    "System transfer resume identity mismatch; refusing to "
                    f"reuse target observations: {mismatch}"
                )
            status = state.get("status")
            if status not in {"running", "completed"}:
                raise RuntimeError(
                    f"System transfer state has invalid status {status!r}"
                )
            persisted_iterations = state.get(
                "refinement_iterations_requested",
            )
            if (
                isinstance(persisted_iterations, bool)
                or not isinstance(persisted_iterations, int)
                or persisted_iterations < 0
            ):
                raise RuntimeError(
                    "System transfer state has an invalid refinement budget: "
                    f"{persisted_iterations!r}"
                )
            if n_refine_iterations < persisted_iterations:
                raise RuntimeError(
                    "System transfer refinement budget cannot be decreased: "
                    f"{persisted_iterations} -> {n_refine_iterations}"
                )
            result_path = os.path.join(
                self.project_dir, "system_transfer_result.json",
            )
            if status == "completed":
                if (
                    state.get("refinement_iterations_completed")
                    != persisted_iterations
                    or not os.path.isfile(result_path)
                    or state.get("result_sha256") != _file_sha256(result_path)
                ):
                    raise RuntimeError(
                        "Completed system transfer state/result integrity "
                        "check failed"
                    )
            elif os.path.isfile(result_path):
                logger.warning(
                    "System transfer result exists while state is still "
                    "running; recovering the final state-commit crash window "
                    "from per-evaluation commits"
                )
            source_entries = state.get("source_records")
            if not isinstance(source_entries, list) or not source_entries:
                raise RuntimeError(
                    "System transfer state has no frozen source record order"
                )
            indexed = {item["config_hash"]: item for item in quality_results}
            restored: List[dict] = []
            seen: set[str] = set()
            for source_index, entry in enumerate(source_entries):
                if not isinstance(entry, dict):
                    raise RuntimeError(
                        "System transfer state has a malformed source record"
                    )
                if entry.get("source_index") != source_index:
                    raise RuntimeError(
                        "System transfer state source_index does not match its "
                        f"frozen position: {entry.get('source_index')!r} != "
                        f"{source_index}"
                    )
                config_hash_value = entry.get("algo_hash")
                if not isinstance(config_hash_value, str) or config_hash_value in seen:
                    raise RuntimeError(
                        "System transfer state source order contains a missing "
                        "or duplicate algo hash"
                    )
                seen.add(config_hash_value)
                record = indexed.get(config_hash_value)
                if record is None:
                    raise RuntimeError(
                        "System transfer QualityStore lost frozen source record "
                        f"{source_index}: {config_hash_value}"
                    )
                actual_marker = self._transfer_record_marker_sha256(
                    quality_store, config_hash_value,
                )
                if entry.get("record_marker_sha256") != actual_marker:
                    raise RuntimeError(
                        "System transfer frozen source record changed after "
                        f"replay began: {config_hash_value}"
                    )
                restored.append(record)
            if n_refine_iterations > persisted_iterations:
                # A completed budget is a sealed optimizer-history prefix.  It
                # must be audited under its OLD cap before widening the cap;
                # otherwise a missing final marker could be rolled back and
                # replaced, or a surplus marker could be silently legitimised
                # as part of the extension.
                if status == "completed":
                    completed_refinements, _, _, _ = (
                        self._restore_transfer_refinements(
                            state=state,
                            quality_store=quality_store,
                            source_count=len(source_entries),
                            exact_committed_count=persisted_iterations,
                        )
                    )
                    if len(completed_refinements) != persisted_iterations:
                        raise RuntimeError(
                            "Completed system transfer refinement ledger does "
                            "not exactly match its sealed budget: "
                            f"{len(completed_refinements)}/"
                            f"{persisted_iterations}"
                        )
                extension = {
                    "from": persisted_iterations,
                    "to": n_refine_iterations,
                    "prior_status": status,
                }
                if status == "completed":
                    extension.update({
                        "prior_completed_iterations": state[
                            "refinement_iterations_completed"
                        ],
                        "prior_result_sha256": state["result_sha256"],
                    })
                extension_history = state.get(
                    "refinement_budget_extensions", [],
                )
                if not isinstance(extension_history, list):
                    raise RuntimeError(
                        "System transfer state has malformed refinement "
                        "budget extension history"
                    )
                state = deepcopy(state)
                state["refinement_budget_extensions"] = [
                    *deepcopy(extension_history), extension,
                ]
                state["refinement_iterations_requested"] = (
                    n_refine_iterations
                )
                state["status"] = "running"
                state.pop("refinement_iterations_completed", None)
                state.pop("result_sha256", None)
                self._atomic_transfer_json(state_path, state)
                logger.info(
                    "Extended system-transfer refinement budget from %d to "
                    "%d after validating the sealed %d-observation ledger; "
                    "committed observations will be restored before running "
                    "the remainder",
                    persisted_iterations,
                    n_refine_iterations,
                    persisted_iterations,
                )
            return state, restored

        # A state file is written before the first CM call.  Artifacts without
        # it cannot be attributed to this exact source/target contract.
        evals_dir = os.path.join(self.project_dir, "evaluations")
        existing_eval_dirs = (
            [name for name in os.listdir(evals_dir) if name.startswith("eval_")]
            if os.path.isdir(evals_dir) else []
        )
        if (
            existing_eval_dirs
            or os.path.isfile(os.path.join(self.project_dir, "all_evaluations.csv"))
            or os.path.isfile(os.path.join(self.project_dir, "system_transfer_result.json"))
            or len(getattr(self.optimizer, "_configs", []) or [])
        ):
            raise RuntimeError(
                "System transfer target has observations but no immutable "
                "system_transfer_state.json; refusing unsafe reconstruction"
            )
        source_records = [
            {
                "source_index": source_index,
                "algo_hash": record["config_hash"],
                "record_marker_sha256": self._transfer_record_marker_sha256(
                    quality_store, record["config_hash"],
                ),
            }
            for source_index, record in enumerate(quality_results)
        ]
        state = {
            "kind": "system_transfer_state",
            "status": "running",
            "cost_model_version": 1,
            "target_config_sha256": target_config_sha,
            "target_system_sha256": target_system_sha,
            "quality_store_path": store_path,
            "refinement_iterations_requested": n_refine_iterations,
            "source_records": source_records,
        }
        if replay_plan is not None:
            state["replay_plan"] = deepcopy(replay_plan)
        self._atomic_transfer_json(state_path, state)
        return state, quality_results

    def _system_transfer_target_identity(self) -> tuple[str, str]:
        """Return the immutable config/system identity used by all commits.

        Dataset initialisation enriches ``self.config`` in memory with corpus
        N/token statistics.  Those are derived workload facts and can vary
        with tokenizer cache availability, so the published on-disk config is
        the transfer contract.  Programmatic callers without a config file
        retain the in-memory fallback.
        """
        config_path = os.path.join(self.project_dir, CONFIG_FILENAME)
        if os.path.isfile(config_path):
            identity_config = load_yaml_config(config_path)
        else:
            identity_config = deepcopy(getattr(self, "config", None) or {})
        identity_system = deepcopy(identity_config.get("system") or {})
        live_target_system = (
            getattr(self, "_transfer_target_system", None) or {}
        )
        if "performance_objectives" in live_target_system:
            identity_system["performance_objectives"] = deepcopy(
                live_target_system["performance_objectives"],
            )
            identity_config["system"] = deepcopy(identity_system)
        return (
            _canonical_payload_sha256(identity_config),
            _canonical_payload_sha256(identity_system),
        )

    def _transfer_csv_rows(self) -> dict[int, dict]:
        path = os.path.join(self.project_dir, "all_evaluations.csv")
        if not os.path.isfile(path):
            return {}
        with open(path, newline="") as stream:
            rows = list(csv.DictReader(stream))
        indexed: dict[int, dict] = {}
        for row in rows:
            try:
                eval_id = int(row.get("eval_id", ""))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "System transfer CSV has a non-integer eval_id"
                ) from exc
            if eval_id in indexed:
                raise RuntimeError(
                    f"System transfer CSV has duplicate eval_id {eval_id}"
                )
            indexed[eval_id] = row
        return indexed

    def _remove_transfer_csv_rows(self, eval_ids: set[int]) -> None:
        """Atomically roll back only uncommitted transfer CSV tail rows."""
        if not eval_ids:
            return
        path = os.path.join(self.project_dir, "all_evaluations.csv")
        if not os.path.isfile(path):
            return
        with open(path, newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        if "eval_id" not in fieldnames:
            raise RuntimeError(
                "Cannot roll back transfer CSV without an eval_id column"
            )
        kept: list[dict] = []
        removed: set[int] = set()
        for row in rows:
            try:
                eval_id = int(row.get("eval_id", ""))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Cannot roll back transfer CSV with invalid eval_id"
                ) from exc
            if eval_id in eval_ids:
                removed.add(eval_id)
            else:
                kept.append(row)
        if removed != eval_ids.intersection(set(self._transfer_csv_rows())):
            raise RuntimeError(
                "Transfer CSV rollback could not identify the requested rows"
            )
        temp_path = path + ".transfer-rollback.tmp"
        try:
            with open(temp_path, "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    def _transfer_config_from_csv_row(self, row: dict) -> dict:
        objective_names = set(self.optimizer.objective_costs)
        flat: dict = {}
        for column, value in row.items():
            if not value or column in {"eval_type", "eval_error", "eval_id"}:
                continue
            if any(
                column == name or column.startswith(name + "__")
                for name in objective_names
            ):
                continue
            flat[column] = _parse_checkpoint_csv_cell(value)
        decoded = self._ssops.decode_to_nested(flat)
        if not isinstance(decoded, dict) or not decoded:
            raise RuntimeError(
                "Cannot reconstruct system-transfer config from committed CSV row"
            )
        return decoded

    @staticmethod
    def _transfer_optimizer_history_free(
        source_record: dict, target_eval_type: EvalType,
    ) -> bool:
        """Restore source history phase semantics, not current-run GT cost.

        Every replay reuses quality for free *in this transfer*, but marking
        every historical observation ``free=True`` makes hybrid optimizers
        believe they have seen zero paid-quality history and re-enter their
        greedy bootstrap.  A source GT was a real history seat; inherited GT_FP
        and penalty/OOM observations remain phase-free.
        """
        source_type = str(
            source_record.get("source_eval_type")
            or (source_record.get("source") or {}).get("eval_type")
            or ""
        )
        return source_type != EvalType.GT.value or target_eval_type is EvalType.OOM

    def _transfer_optimizer_quality_payload(
        self, *, combined: float, quality_metrics: dict,
    ) -> dict:
        """Expose only target-declared diagnostics to the optimizer.

        Phase 2a intentionally preserves the source quality scalar and writes
        the source ``quality_metrics.json`` byte-for-semantics into the target
        replay.  The target experiment may nevertheless use a different set of
        diagnostic sub-metrics.  Filtering only this optimizer-facing payload
        prevents retired source metrics from steering the agent without
        fabricating values for newly declared metrics or changing quality.
        """
        configured_names = {
            str(metric.get("metric_name") or metric.get("metric"))
            for metric in (
                (self.config.get("eval_backend_setting") or {}).get("metrics")
                or []
            )
            if isinstance(metric, dict)
            and (metric.get("metric_name") or metric.get("metric"))
        }
        payload = {"combined": float(combined)}
        payload.update({
            key: value
            for key, value in (quality_metrics or {}).items()
            if value is not None
            and (not configured_names or key in configured_names)
        })
        return payload

    def _save_transfer_provenance(
        self,
        *,
        replay_idx: int,
        config_hash_value: str,
        source_record: dict,
        target_eval_type: EvalType,
        source_record_marker_sha256: str | None = None,
        target_config_sha256: str | None = None,
        replay_action: str = _TRANSFER_ACTION_RECOMPUTED,
        llmlingua2_profile: dict | None = None,
        reuse_lineage: dict | None = None,
    ) -> None:
        """Persist source identity beside a target replay evaluation."""
        if replay_action not in {
            _TRANSFER_ACTION_RECOMPUTED, _TRANSFER_ACTION_REUSED,
        }:
            raise RuntimeError(
                f"Unsupported system-transfer replay action {replay_action!r}"
            )
        if replay_action == _TRANSFER_ACTION_RECOMPUTED:
            if reuse_lineage is not None:
                raise RuntimeError(
                    "Recomputed replay provenance cannot claim prior-target lineage"
                )
        else:
            if llmlingua2_profile is not None:
                raise RuntimeError(
                    "Reused replay provenance cannot claim the current CM profile"
                )
            if not isinstance(reuse_lineage, dict):
                raise RuntimeError(
                    "Reused replay provenance requires immutable prior lineage"
                )
        eval_dir = os.path.join(
            self.project_dir, "evaluations", f"eval_{replay_idx:04d}",
        )
        os.makedirs(eval_dir, exist_ok=True)
        payload = {
            "kind": "system_transfer_replay",
            "source_index": replay_idx,
            "source_algo_hash": config_hash_value,
            "source_record_marker_sha256": source_record_marker_sha256,
            "target_config_sha256": (
                target_config_sha256
                if target_config_sha256 is not None
                else self._system_transfer_target_identity()[0]
            ),
            "source": deepcopy(source_record.get("source") or {}),
            "source_eval_type": source_record.get("source_eval_type"),
            "source_eval_error": source_record.get("source_eval_error"),
            "target_eval_type": target_eval_type.value,
            "quality_reused": True,
            "cost_model_version": 1,
            "performance_origin": replay_action,
            "artifact_sha256": self._transfer_eval_artifact_sha256(eval_dir),
        }
        if llmlingua2_profile is not None:
            payload["llmlingua2_profile"] = deepcopy(llmlingua2_profile)
        if reuse_lineage is not None:
            payload["reuse_lineage"] = deepcopy(reuse_lineage)
        path = os.path.join(eval_dir, "transfer_provenance.json")
        validate_system_transfer_artifact(payload, source=path)
        import tempfile

        fd, temp_path = tempfile.mkstemp(
            prefix=".transfer_provenance.", suffix=".tmp", dir=eval_dir,
        )
        try:
            with os.fdopen(fd, "w") as stream:
                safe_json_dump(payload, stream, indent=2, default=str)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    def _restore_transfer_replays(
        self,
        *,
        state: dict,
        quality_store: QualityStore,
        quality_results: List[dict],
    ) -> tuple[List[dict], List[AssemblyResult]]:
        """Validate and reconstruct the committed contiguous Phase 2a prefix."""
        rows = self._transfer_csv_rows()
        replay_evaluations: List[dict] = []
        restored_assemblies: List[AssemblyResult] = []
        first_missing: int | None = None
        # Before the immutable on-disk identity was introduced, replay
        # provenance hashed the DatasetManager-enriched in-memory config.  It
        # is safe to recognise exactly that deterministic legacy value, but
        # only migrate it after every source, artifact, target-system and CSV
        # integrity check below succeeds.
        legacy_mutable_target_sha = _canonical_payload_sha256(
            getattr(self, "config", None) or {}
        )
        for replay_idx, quality_record in enumerate(quality_results):
            replay_action, replay_dependency = self._transfer_replay_action(
                state, replay_idx,
            )
            reuse_lineage = (
                self._transfer_reuse_lineage(state["replay_plan"], replay_idx)
                if replay_action == _TRANSFER_ACTION_REUSED
                else None
            )
            eval_dir = os.path.join(
                self.project_dir, "evaluations", f"eval_{replay_idx:04d}",
            )
            marker_path = os.path.join(eval_dir, "transfer_provenance.json")
            if not os.path.isfile(marker_path):
                if os.path.isdir(eval_dir) or replay_idx in rows:
                    # Crash-window recovery: _save_eval_record commits the
                    # complete eval artifacts and atomically upserts its CSV
                    # row before the small provenance marker is written.  When
                    # all those durable inputs are present, promote them by
                    # writing the missing marker; the normal validation below
                    # then proves their exact source/target identity.  We never
                    # recompute or overwrite the expensive CM result.
                    row = rows.get(replay_idx)
                    required = [
                        os.path.join(eval_dir, "pipeline_config.json"),
                        os.path.join(eval_dir, "quality_metrics.json"),
                        os.path.join(eval_dir, "execution_dag.json"),
                    ]
                    target_type_value = (row or {}).get("eval_type")
                    if (
                        row is None
                        or target_type_value not in {
                            EvalType.GT_FP.value, EvalType.OOM.value,
                        }
                        or not all(os.path.isfile(path) for path in required)
                        or (
                            target_type_value == EvalType.GT_FP.value
                            and not os.path.isfile(os.path.join(eval_dir, "assembly.json"))
                        )
                    ):
                        later_state = []
                        for later_idx in range(replay_idx + 1, len(quality_results)):
                            later_dir = os.path.join(
                                self.project_dir, "evaluations",
                                f"eval_{later_idx:04d}",
                            )
                            if os.path.isdir(later_dir) or later_idx in rows:
                                later_state.append(later_idx)
                        if later_state:
                            raise RuntimeError(
                                "System transfer replay partial observation is "
                                "not a tail; later state exists at indices "
                                f"{later_state}"
                            )
                        # The record never reached its commit marker and no
                        # later transfer record depends on it. Roll back this
                        # transaction tail only; the next loop invocation
                        # reprices the same source index. Already committed
                        # source observations are never touched.
                        self._remove_transfer_csv_rows({replay_idx})
                        if os.path.isdir(eval_dir):
                            shutil.rmtree(eval_dir)
                        rows.pop(replay_idx, None)
                        first_missing = replay_idx
                        logger.warning(
                            "Rolled back incomplete uncommitted replay tail "
                            "eval_%04d; replay will resume from this source",
                            replay_idx,
                        )
                        break
                    self._save_transfer_provenance(
                        replay_idx=replay_idx,
                        config_hash_value=quality_record["config_hash"],
                        source_record=quality_record.get("record") or {},
                        target_eval_type=EvalType(target_type_value),
                        target_config_sha256=state["target_config_sha256"],
                        source_record_marker_sha256=state["source_records"][
                            replay_idx
                        ]["record_marker_sha256"],
                        replay_action=replay_action,
                        llmlingua2_profile=replay_dependency,
                        reuse_lineage=reuse_lineage,
                    )
                    logger.info(
                        "Recovered replay commit marker for complete "
                        "eval_%04d after interrupted provenance write",
                        replay_idx,
                    )
                else:
                    first_missing = replay_idx
                    break

            if first_missing is not None:
                break

            with open(marker_path) as stream:
                marker = json.load(stream)
            validate_system_transfer_artifact(
                marker,
                expected_kind="system_transfer_replay",
                source=marker_path,
            )
            source_state = state["source_records"][replay_idx]
            expected = {
                "kind": "system_transfer_replay",
                "source_index": replay_idx,
                "source_algo_hash": quality_record["config_hash"],
                "source_record_marker_sha256": source_state[
                    "record_marker_sha256"
                ],
                "target_config_sha256": state["target_config_sha256"],
                "quality_reused": True,
                "cost_model_version": 1,
            }
            if state.get("replay_plan") is not None:
                expected["performance_origin"] = replay_action
                if replay_action == _TRANSFER_ACTION_RECOMPUTED:
                    expected["llmlingua2_profile"] = replay_dependency
                    if "reuse_lineage" in marker:
                        raise RuntimeError(
                            "Recomputed selective replay claims prior-target "
                            f"lineage at eval_{replay_idx:04d}"
                        )
                else:
                    expected["reuse_lineage"] = reuse_lineage
                    if "llmlingua2_profile" in marker:
                        raise RuntimeError(
                            "Reused selective replay claims current CM profile "
                            f"at eval_{replay_idx:04d}"
                        )
            migrate_legacy_target_identity = (
                marker.get("target_config_sha256")
                != state["target_config_sha256"]
                and marker.get("target_config_sha256")
                == legacy_mutable_target_sha
            )
            mismatch = {
                key: (marker.get(key), value)
                for key, value in expected.items()
                if marker.get(key) != value
                if not (
                    key == "target_config_sha256"
                    and migrate_legacy_target_identity
                )
            }
            if mismatch:
                raise RuntimeError(
                    "System transfer replay commit mismatch at "
                    f"eval_{replay_idx:04d}: {mismatch}"
                )
            try:
                target_type = EvalType(marker["target_eval_type"])
            except (KeyError, ValueError) as exc:
                raise RuntimeError(
                    f"Invalid replay outcome at eval_{replay_idx:04d}"
                ) from exc
            if target_type not in (EvalType.GT_FP, EvalType.OOM):
                raise RuntimeError(
                    f"Unsupported replay outcome {target_type} at "
                    f"eval_{replay_idx:04d}"
                )
            row = rows.get(replay_idx)
            if row is None or row.get("eval_type") != target_type.value:
                raise RuntimeError(
                    f"Replay commit/CSV mismatch at eval_{replay_idx:04d}"
                )
            required_paths = {
                "pipeline": os.path.join(eval_dir, "pipeline_config.json"),
                "quality": os.path.join(eval_dir, "quality_metrics.json"),
                "trace": os.path.join(eval_dir, "execution_dag.json"),
            }
            missing = [name for name, path in required_paths.items() if not os.path.isfile(path)]
            if missing:
                raise RuntimeError(
                    f"Committed replay eval_{replay_idx:04d} is missing {missing}"
                )
            with open(required_paths["quality"]) as stream:
                persisted_quality = json.load(stream)
            if persisted_quality != (quality_record.get("quality_metrics") or {}):
                raise RuntimeError(
                    f"Replay quality changed at eval_{replay_idx:04d}"
                )
            with open(required_paths["trace"]) as stream:
                persisted_trace = json.load(stream)
            if persisted_trace != quality_store.get_execution_dag(
                quality_record["config_hash"]
            ):
                raise RuntimeError(
                    f"Replay trace changed at eval_{replay_idx:04d}"
                )
            with open(required_paths["pipeline"]) as stream:
                persisted_pipeline = json.load(stream)
            self._assert_transfer_pipeline_targets_current_system(
                pipeline_config=persisted_pipeline,
                context=f"Committed replay eval_{replay_idx:04d}",
            )

            combined = float(quality_record["quality_combined"])
            quality_payload = self._transfer_optimizer_quality_payload(
                combined=combined,
                quality_metrics=persisted_quality,
            )
            assembly_path = os.path.join(eval_dir, "assembly.json")
            if target_type is EvalType.OOM:
                if os.path.isfile(assembly_path):
                    raise RuntimeError(
                        f"OOM replay unexpectedly has assembly at eval_{replay_idx:04d}"
                    )
                performance = 0.0
            else:
                if not os.path.isfile(assembly_path):
                    raise RuntimeError(
                        f"Replay assembly missing at eval_{replay_idx:04d}"
                    )
                with open(assembly_path) as stream:
                    assembly_payload = json.load(stream)
                performance = float(assembly_payload["performance_score"])
                if not math.isfinite(performance):
                    raise RuntimeError(
                        f"Non-finite replay performance at eval_{replay_idx:04d}"
                    )
                restored_assemblies.append(AssemblyResult(
                    performance_score=performance,
                    best_hw_params=deepcopy(
                        assembly_payload.get("best_hw_params") or {}
                    ),
                ))
            expected_artifacts = {
                "pipeline_config.json", "quality_metrics.json",
                "execution_dag.json",
            }
            if target_type is not EvalType.OOM:
                expected_artifacts.add("assembly.json")
            recorded_artifacts = marker.get("artifact_sha256") or {}
            if _OPTIMIZER_OBSERVATION_FILENAME in recorded_artifacts:
                expected_artifacts.add(_OPTIMIZER_OBSERVATION_FILENAME)
            self._validate_transfer_eval_artifact_sha256(
                marker=marker,
                eval_dir=eval_dir,
                expected_names=expected_artifacts,
            )
            optimizer_metadata = None
            optimizer_marker_path = os.path.join(
                eval_dir,
                _OPTIMIZER_OBSERVATION_FILENAME,
            )
            if os.path.isfile(optimizer_marker_path):
                with open(optimizer_marker_path) as stream:
                    optimizer_metadata = json.load(stream)
            try:
                csv_performance = float(row.get("performance", ""))
                csv_quality = float(row.get("quality__combined", ""))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Replay CSV objectives missing at eval_{replay_idx:04d}"
                ) from exc
            if not math.isclose(csv_performance, performance, abs_tol=5e-6):
                raise RuntimeError(
                    f"Replay performance/CSV mismatch at eval_{replay_idx:04d}"
                )
            if not math.isclose(csv_quality, combined, abs_tol=5e-6):
                raise RuntimeError(
                    f"Replay quality/CSV mismatch at eval_{replay_idx:04d}"
                )
            if migrate_legacy_target_identity:
                marker = deepcopy(marker)
                marker["target_config_sha256"] = state[
                    "target_config_sha256"
                ]
                self._atomic_transfer_json(marker_path, marker)
                logger.warning(
                    "Migrated legacy mutable target-config identity for "
                    "fully validated replay eval_%04d to the immutable "
                    "system-transfer state identity",
                    replay_idx,
                )
            replay_evaluations.append({
                "config": quality_record["config"],
                "objectives": {
                    "quality": quality_payload,
                    "performance": performance,
                },
                "eval_id": replay_idx,
                "eval_type": target_type.value,
                "free": self._transfer_optimizer_history_free(
                    quality_record.get("record") or {}, target_type,
                ),
                "quality_reused": True,
                "optimizer_metadata": optimizer_metadata,
            })

        if first_missing is not None:
            for later_idx in range(first_missing + 1, len(quality_results)):
                later_marker = os.path.join(
                    self.project_dir, "evaluations", f"eval_{later_idx:04d}",
                    "transfer_provenance.json",
                )
                later_dir = os.path.dirname(later_marker)
                if (
                    os.path.isfile(later_marker)
                    or os.path.isdir(later_dir)
                    or later_idx in rows
                ):
                    raise RuntimeError(
                        "System transfer replay state is not a contiguous "
                        f"prefix (missing {first_missing}, found {later_idx})"
                    )
        self._eval_counter = max(self._eval_counter, len(replay_evaluations))
        return replay_evaluations, restored_assemblies

    def _save_transfer_refinement_commit(
        self,
        *,
        state: dict,
        refinement_index: int,
        eval_id: int,
        config: dict,
        performance: float,
        quality: Any,
        algo_hash: str,
        eval_type: str,
        free: bool,
        gt_budget_count: int,
        quality_store: QualityStore,
    ) -> None:
        eval_dir = os.path.join(
            self.project_dir, "evaluations", f"eval_{eval_id:04d}",
        )
        payload = {
            "kind": "system_transfer_refinement",
            "refinement_index": refinement_index,
            "eval_id": eval_id,
            "algo_hash": algo_hash,
            "config": _strip_secrets(deepcopy(config)),
            "objectives": {
                "performance": float(performance),
                "quality": deepcopy(quality),
            },
            "eval_type": str(eval_type),
            "optimizer_history_free": bool(free),
            "gt_budget_count": int(gt_budget_count),
            "objective_cost": float(sum(self.optimizer.objective_costs.values())),
            "quality_store_record_marker_sha256": (
                self._transfer_record_marker_sha256(quality_store, algo_hash)
            ),
            "target_config_sha256": state["target_config_sha256"],
            "cost_model_version": 1,
            "artifact_sha256": self._transfer_eval_artifact_sha256(eval_dir),
        }
        replay_plan = state.get("replay_plan")
        if replay_plan is not None:
            dependency = replay_plan.get("llmlingua2_profile")
            if not isinstance(dependency, dict):
                raise RuntimeError(
                    "Selective refinement commit has no frozen CM profile"
                )
            payload["llmlingua2_profile"] = deepcopy(dependency)
        self._atomic_transfer_json(
            os.path.join(eval_dir, _TRANSFER_REFINEMENT_MARKER_FILENAME),
            payload,
        )

    def _recover_refinement_quality_store_record(
        self,
        *,
        quality_store: QualityStore,
        config: dict,
        eval_id: int,
    ) -> dict | None:
        """Recover a complete eval whose QualityStore append was interrupted."""
        eval_dir = os.path.join(
            self.project_dir, "evaluations", f"eval_{eval_id:04d}",
        )
        paths = {
            "pipeline": os.path.join(eval_dir, "pipeline_config.json"),
            "quality": os.path.join(eval_dir, "quality_metrics.json"),
            "trace": os.path.join(eval_dir, "execution_dag.json"),
        }
        if not all(os.path.isfile(path) for path in paths.values()):
            return None
        with open(paths["pipeline"]) as stream:
            pipeline_config = json.load(stream)
        with open(paths["quality"]) as stream:
            quality_metrics = json.load(stream)
        with open(paths["trace"]) as stream:
            execution_dag = json.load(stream)
        configured_metrics = [
            metric.get("metric_name") or metric.get("metric")
            for metric in (self.config.get("eval_backend_setting") or {}).get(
                "metrics", []
            )
            if isinstance(metric, dict)
            and (metric.get("metric_name") or metric.get("metric"))
        ]
        full = all(
            quality_metrics.get(metric_name) is not None
            for metric_name in configured_metrics
        )
        record_hash, _ = quality_store.append_record(
            self._quality_algo_config(config),
            pipeline_config,
            quality_metrics,
            execution_dag,
            full=full,
            source={
                "kind": "system_transfer_resume_recovery",
                "project_dir": self.project_dir,
                "eval_id": eval_id,
                "source_mtime_ns": max(
                    os.stat(path).st_mtime_ns for path in paths.values()
                ),
            },
            quality_spec=QualityStore.quality_spec(self.config),
        )
        record = quality_store.get_record(record_hash)
        if record is not None:
            quality_store.get_execution_dag(record_hash)
        return record

    def _restore_transfer_refinements(
        self,
        *,
        state: dict,
        quality_store: QualityStore,
        source_count: int,
        exact_committed_count: int | None = None,
    ) -> tuple[List[dict], List[AssemblyResult], int, float]:
        """Restore every strictly committed landed Phase 2b observation.

        ``exact_committed_count`` is the non-mutating preflight used before a
        completed budget is extended.  In that mode missing/surplus markers and
        every uncommitted tail fail closed instead of entering crash recovery.
        """
        evals_dir = os.path.join(self.project_dir, "evaluations")
        markers: list[tuple[int, str]] = []
        if os.path.isdir(evals_dir):
            for dirname in os.listdir(evals_dir):
                marker_path = os.path.join(
                    evals_dir, dirname, _TRANSFER_REFINEMENT_MARKER_FILENAME,
                )
                if not os.path.isfile(marker_path):
                    continue
                with open(marker_path) as stream:
                    payload = json.load(stream)
                index = payload.get("refinement_index")
                if isinstance(index, bool) or not isinstance(index, int):
                    raise RuntimeError(
                        f"Malformed refinement commit at {marker_path}"
                    )
                markers.append((index, marker_path))
        markers.sort()
        expected_indices = list(range(len(markers)))
        if [index for index, _ in markers] != expected_indices:
            raise RuntimeError(
                "System transfer refinement commits are duplicated or non-contiguous"
            )
        if (
            exact_committed_count is not None
            and len(markers) != exact_committed_count
        ):
            raise RuntimeError(
                "Completed system transfer refinement marker ledger does not "
                "exactly match its sealed budget: "
                f"{len(markers)}/{exact_committed_count}"
            )
        requested = state["refinement_iterations_requested"]
        if len(markers) > requested:
            raise RuntimeError(
                "System transfer has more committed refinement observations "
                f"than requested ({len(markers)}/{requested})"
            )

        rows = self._transfer_csv_rows()
        refinement_eval_highwater = max(
            [self._eval_counter, source_count]
            + [eval_id + 1 for eval_id in rows if eval_id >= source_count]
        )
        if os.path.isdir(evals_dir):
            for dirname in os.listdir(evals_dir):
                if not dirname.startswith("eval_"):
                    continue
                try:
                    refinement_eval_highwater = max(
                        refinement_eval_highwater,
                        int(dirname.removeprefix("eval_")) + 1,
                    )
                except ValueError:
                    pass
        committed_eval_ids: set[int] = set()
        for _, marker_path in markers:
            with open(marker_path) as stream:
                marker_payload = json.load(stream)
            eval_id = marker_payload.get("eval_id")
            if isinstance(eval_id, bool) or not isinstance(eval_id, int):
                raise RuntimeError(
                    f"Malformed refinement eval_id at {marker_path}"
                )
            committed_eval_ids.add(eval_id)

        # Same crash window as Phase 2a: target eval + CSV + QualityStore can
        # all be durable before the tiny refinement marker. A selective replay
        # marker is also the post-CM profile attestation, so an unmarked landed
        # tail must be re-priced rather than promoted. Non-selective transfers
        # may still promote one complete valid tail. Later observations make an
        # unmarked earlier row an impossible/non-prefix history and fail closed.
        next_refinement_index = len(markers)
        max_committed_eval_id = max(committed_eval_ids, default=source_count - 1)
        for eval_id, row in sorted(rows.items()):
            if eval_id < source_count or eval_id in committed_eval_ids:
                continue
            eval_type = row.get("eval_type") or ""
            if eval_type in {
                EvalType.INVALID.value, EvalType.DUPLICATE.value,
                EvalType.CONTEXT_OVERFLOW.value, "SKIPPED", "",
            }:
                continue
            if exact_committed_count is not None:
                raise RuntimeError(
                    "Completed system transfer has an uncommitted landed "
                    f"refinement at eval_{eval_id:04d}"
                )
            if eval_id < max_committed_eval_id:
                raise RuntimeError(
                    "System transfer refinement history has an uncommitted "
                    f"landed row before a later commit: eval_{eval_id:04d}"
                )
            if eval_type not in {
                EvalType.GT.value, EvalType.GT_FP.value, EvalType.OOM.value,
            }:
                raise RuntimeError(
                    "System transfer cannot recover unsupported uncommitted "
                    f"refinement type {eval_type!r} at eval_{eval_id:04d}"
                )
            eval_dir = os.path.join(
                self.project_dir, "evaluations", f"eval_{eval_id:04d}",
            )
            if state.get("replay_plan") is not None:
                later_landed = [
                    later_id for later_id, later_row in rows.items()
                    if later_id > eval_id
                    and (later_row.get("eval_type") or "") not in {
                        EvalType.INVALID.value, EvalType.DUPLICATE.value,
                        EvalType.CONTEXT_OVERFLOW.value, "SKIPPED", "",
                    }
                ]
                later_commits = [
                    committed_id for committed_id in committed_eval_ids
                    if committed_id > eval_id
                ]
                if later_landed or later_commits:
                    raise RuntimeError(
                        "Selective system transfer has an unattested "
                        "refinement before later history: "
                        f"eval_{eval_id:04d}, landed={later_landed}, "
                        f"commits={later_commits}"
                    )
                self._remove_transfer_csv_rows({eval_id})
                if os.path.isdir(eval_dir):
                    shutil.rmtree(eval_dir)
                rows.pop(eval_id, None)
                logger.warning(
                    "Rolled back selective refinement eval_%04d without a "
                    "durable post-CM profile attestation; its QualityStore "
                    "quality/trace record remains reusable",
                    eval_id,
                )
                continue
            required = [
                os.path.join(eval_dir, "pipeline_config.json"),
                os.path.join(eval_dir, "quality_metrics.json"),
                os.path.join(eval_dir, "execution_dag.json"),
            ]
            if (
                not all(os.path.isfile(path) for path in required)
                or (
                    eval_type != EvalType.OOM.value
                    and not os.path.isfile(os.path.join(eval_dir, "assembly.json"))
                )
            ):
                later_landed = [
                    later_id for later_id, later_row in rows.items()
                    if later_id > eval_id
                    and (later_row.get("eval_type") or "") not in {
                        EvalType.INVALID.value, EvalType.DUPLICATE.value,
                        EvalType.CONTEXT_OVERFLOW.value, "SKIPPED", "",
                    }
                ]
                later_commits = [
                    committed_id for committed_id in committed_eval_ids
                    if committed_id > eval_id
                ]
                if later_landed or later_commits:
                    raise RuntimeError(
                        "System transfer incomplete refinement is not a tail; "
                        f"later landed={later_landed}, commits={later_commits}"
                    )
                self._remove_transfer_csv_rows({eval_id})
                if os.path.isdir(eval_dir):
                    shutil.rmtree(eval_dir)
                rows.pop(eval_id, None)
                logger.warning(
                    "Rolled back incomplete uncommitted refinement tail "
                    "eval_%04d; optimizer history and eval-id gaps preserved",
                    eval_id,
                )
                continue
            config = self._transfer_config_from_csv_row(row)
            algo_hash = self._quality_algo_key(config)
            if quality_store.get_record(algo_hash) is None:
                try:
                    self._recover_refinement_quality_store_record(
                        quality_store=quality_store,
                        config=config,
                        eval_id=eval_id,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "System transfer could not recover the missing "
                        "QualityStore commit for complete refinement "
                        f"eval_{eval_id:04d}: {_format_exception(exc)}"
                    ) from exc
            if quality_store.get_record(algo_hash) is None:
                raise RuntimeError(
                    "System transfer uncommitted refinement has no matching "
                    f"QualityStore record after recovery: {algo_hash}"
                )
            quality_store.get_execution_dag(algo_hash)
            try:
                performance = float(row.get("performance", ""))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Missing refinement performance at eval_{eval_id:04d}"
                ) from exc
            quality: dict[str, float] = {}
            for column, value in row.items():
                if not column.startswith("quality__") or value in (None, ""):
                    continue
                try:
                    quality[column.removeprefix("quality__")] = float(value)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"Invalid refinement quality at eval_{eval_id:04d}"
                    ) from exc
            if "combined" not in quality:
                raise RuntimeError(
                    f"Missing combined quality at eval_{eval_id:04d}"
                )
            self._save_transfer_refinement_commit(
                state=state,
                refinement_index=next_refinement_index,
                eval_id=eval_id,
                config=config,
                performance=performance,
                quality=quality,
                algo_hash=algo_hash,
                eval_type=eval_type,
                free=eval_type != EvalType.GT.value,
                gt_budget_count=int(eval_type == EvalType.GT.value),
                quality_store=quality_store,
            )
            marker_path = os.path.join(
                eval_dir, _TRANSFER_REFINEMENT_MARKER_FILENAME,
            )
            markers.append((next_refinement_index, marker_path))
            committed_eval_ids.add(eval_id)
            max_committed_eval_id = eval_id
            next_refinement_index += 1
            logger.info(
                "Recovered refinement commit marker for complete eval_%04d "
                "after interrupted marker write", eval_id,
            )

        if os.path.isdir(evals_dir):
            untracked_dirs: list[tuple[int, str]] = []
            for dirname in os.listdir(evals_dir):
                if not dirname.startswith("eval_"):
                    continue
                try:
                    eval_id = int(dirname.removeprefix("eval_"))
                except ValueError:
                    continue
                marker_path = os.path.join(
                    evals_dir, dirname, _TRANSFER_REFINEMENT_MARKER_FILENAME,
                )
                if (
                    eval_id >= source_count
                    and eval_id not in rows
                    and not os.path.isfile(marker_path)
                ):
                    untracked_dirs.append((eval_id, os.path.join(evals_dir, dirname)))
            for eval_id, eval_dir in sorted(untracked_dirs):
                if exact_committed_count is not None:
                    raise RuntimeError(
                        "Completed system transfer has an untracked refinement "
                        f"directory at eval_{eval_id:04d}"
                    )
                later_dirs = [
                    later_id for later_id, _ in untracked_dirs
                    if later_id > eval_id
                ]
                later_rows = [later_id for later_id in rows if later_id > eval_id]
                later_commits = [
                    committed_id for committed_id in committed_eval_ids
                    if committed_id > eval_id
                ]
                if later_dirs or later_rows or later_commits:
                    raise RuntimeError(
                        "System transfer untracked refinement directory is not "
                        f"a tail at eval_{eval_id:04d}"
                    )
                shutil.rmtree(eval_dir)
                logger.warning(
                    "Rolled back incomplete refinement directory eval_%04d "
                    "that never reached its CSV/commit marker",
                    eval_id,
                )
        markers.sort()
        if [index for index, _ in markers] != list(range(len(markers))):
            raise RuntimeError(
                "Recovered system-transfer refinement commits are non-contiguous"
            )
        if len(markers) > requested:
            raise RuntimeError(
                "Recovered more refinement observations than requested "
                f"({len(markers)}/{requested})"
            )

        evaluations: List[dict] = []
        assemblies: List[AssemblyResult] = []
        restored_gt = 0
        restored_cost = 0.0
        committed_ids: set[int] = set()
        last_eval_id = source_count - 1
        for expected_index, marker_path in markers:
            with open(marker_path) as stream:
                marker = json.load(stream)
            validate_system_transfer_artifact(
                marker,
                expected_kind="system_transfer_refinement",
                source=marker_path,
            )
            expected = {
                "kind": "system_transfer_refinement",
                "refinement_index": expected_index,
                "target_config_sha256": state["target_config_sha256"],
                "cost_model_version": 1,
            }
            replay_plan = state.get("replay_plan")
            if replay_plan is not None:
                expected["llmlingua2_profile"] = replay_plan.get(
                    "llmlingua2_profile"
                )
            mismatch = {
                key: (marker.get(key), value)
                for key, value in expected.items()
                if marker.get(key) != value
            }
            if mismatch:
                raise RuntimeError(
                    f"System transfer refinement commit mismatch: {mismatch}"
                )
            eval_id = marker.get("eval_id")
            if (
                isinstance(eval_id, bool) or not isinstance(eval_id, int)
                or eval_id <= last_eval_id or eval_id < source_count
            ):
                raise RuntimeError(
                    "System transfer refinement eval IDs are not strictly increasing"
                )
            last_eval_id = eval_id
            committed_ids.add(eval_id)
            config = marker.get("config")
            algo_hash = marker.get("algo_hash")
            if not isinstance(config, dict) or self._quality_algo_key(config) != algo_hash:
                raise RuntimeError(
                    f"Refinement config/hash mismatch at eval_{eval_id:04d}"
                )
            record = quality_store.get_record(algo_hash)
            if record is None:
                raise RuntimeError(
                    f"Refinement QualityStore record missing: {algo_hash}"
                )
            quality_store.get_execution_dag(algo_hash)
            actual_record_sha = self._transfer_record_marker_sha256(
                quality_store, algo_hash,
            )
            if marker.get("quality_store_record_marker_sha256") != actual_record_sha:
                raise RuntimeError(
                    f"Refinement QualityStore record changed: {algo_hash}"
                )
            row = rows.get(eval_id)
            if row is None or row.get("eval_type") != marker.get("eval_type"):
                raise RuntimeError(
                    f"Refinement commit/CSV mismatch at eval_{eval_id:04d}"
                )
            eval_dir = os.path.join(
                self.project_dir, "evaluations", f"eval_{eval_id:04d}",
            )
            artifact_paths = {
                "pipeline": os.path.join(eval_dir, "pipeline_config.json"),
                "quality": os.path.join(eval_dir, "quality_metrics.json"),
                "trace": os.path.join(eval_dir, "execution_dag.json"),
            }
            missing = [
                name for name, path in artifact_paths.items()
                if not os.path.isfile(path)
            ]
            if missing:
                raise RuntimeError(
                    f"Committed refinement eval_{eval_id:04d} is missing {missing}"
                )
            with open(artifact_paths["quality"]) as stream:
                persisted_quality = json.load(stream)
            if persisted_quality != (record.get("quality_metrics") or {}):
                raise RuntimeError(
                    f"Refinement quality changed at eval_{eval_id:04d}"
                )
            with open(artifact_paths["trace"]) as stream:
                persisted_trace = json.load(stream)
            if persisted_trace != quality_store.get_execution_dag(algo_hash):
                raise RuntimeError(
                    f"Refinement trace changed at eval_{eval_id:04d}"
                )
            with open(artifact_paths["pipeline"]) as stream:
                persisted_pipeline = json.load(stream)
            resolved_system = persisted_pipeline.get("system") or {}
            target_system = getattr(self, "_transfer_target_system", {}) or {}
            for key in (
                "system_hardware", "rag_cm", "performance_source",
                "system_space_owner", "gpu", "cpu", "available_gpus",
                "interconnect", "cm_search_space",
            ):
                if key in target_system and resolved_system.get(key) != _strip_secrets(
                    target_system[key]
                ):
                    raise RuntimeError(
                        "Committed refinement target system mismatch for "
                        f"{key!r} at eval_{eval_id:04d}"
                    )
            objectives = marker.get("objectives") or {}
            performance = float(objectives.get("performance"))
            quality = objectives.get("quality")
            quality_combined = float(
                quality.get("combined") if isinstance(quality, dict) else quality
            )
            try:
                csv_performance = float(row.get("performance", ""))
                csv_quality = float(row.get("quality__combined", ""))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Refinement CSV objectives missing at eval_{eval_id:04d}"
                ) from exc
            if not (
                math.isclose(performance, csv_performance, abs_tol=5e-6)
                and math.isclose(quality_combined, csv_quality, abs_tol=5e-6)
            ):
                raise RuntimeError(
                    f"Refinement objective/CSV mismatch at eval_{eval_id:04d}"
                )
            assembly_path = os.path.join(
                eval_dir, "assembly.json",
            )
            eval_type = marker.get("eval_type")
            if eval_type == EvalType.OOM.value:
                if os.path.isfile(assembly_path):
                    raise RuntimeError(
                        f"OOM refinement unexpectedly has assembly at eval_{eval_id:04d}"
                    )
            else:
                if not os.path.isfile(assembly_path):
                    raise RuntimeError(
                        f"Refinement assembly missing at eval_{eval_id:04d}"
                    )
                with open(assembly_path) as stream:
                    assembly_payload = json.load(stream)
                if not math.isclose(
                    float(assembly_payload["performance_score"]), performance,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        f"Refinement assembly mismatch at eval_{eval_id:04d}"
                    )
                assemblies.append(AssemblyResult(
                    performance_score=performance,
                    best_hw_params=deepcopy(
                        assembly_payload.get("best_hw_params") or {}
                    ),
                ))
            expected_artifacts = {
                "pipeline_config.json", "quality_metrics.json",
                "execution_dag.json",
            }
            if eval_type != EvalType.OOM.value:
                expected_artifacts.add("assembly.json")
            recorded_artifacts = marker.get("artifact_sha256") or {}
            if _OPTIMIZER_OBSERVATION_FILENAME in recorded_artifacts:
                expected_artifacts.add(_OPTIMIZER_OBSERVATION_FILENAME)
            self._validate_transfer_eval_artifact_sha256(
                marker=marker,
                eval_dir=eval_dir,
                expected_names=expected_artifacts,
            )
            optimizer_metadata = None
            optimizer_marker_path = os.path.join(
                eval_dir,
                _OPTIMIZER_OBSERVATION_FILENAME,
            )
            if os.path.isfile(optimizer_marker_path):
                with open(optimizer_marker_path) as stream:
                    optimizer_metadata = json.load(stream)
            evaluations.append({
                "config": config,
                "objectives": deepcopy(objectives),
                "eval_id": eval_id,
                "eval_type": marker["eval_type"],
                "free": bool(marker.get("optimizer_history_free")),
                "optimizer_metadata": optimizer_metadata,
            })
            restored_gt += int(marker.get("gt_budget_count", 0))
            restored_cost += float(marker.get("objective_cost", 0.0))

        # Rows after Phase 2a that have no landed commit are safe only when
        # they were explicitly excluded from optimizer data. A GT/GT_FP/OOM
        # row without a commit means the process died mid-transaction.
        for eval_id, row in rows.items():
            if eval_id < source_count or eval_id in committed_ids:
                continue
            if row.get("eval_type") not in {
                EvalType.INVALID.value, EvalType.DUPLICATE.value,
                EvalType.CONTEXT_OVERFLOW.value, "SKIPPED", "",
            }:
                raise RuntimeError(
                    "System transfer has an uncommitted landed refinement at "
                    f"eval_{eval_id:04d} (type={row.get('eval_type')})"
                )
        self._eval_counter = max(self._eval_counter, refinement_eval_highwater)
        return evaluations, assemblies, restored_gt, restored_cost

    def _save_system_transfer_result(self, results: dict) -> None:
        """Atomically persist the completed system-transfer result ledger.

        ``assembly_results`` can contain large, implementation-specific CM
        objects, so the durable public artifact records their count and keeps
        the portable Pareto/config/objective data.  A transfer is never marked
        complete here unless the landed refinement count exactly matches the
        requested iteration budget.
        """
        transfer = results.get("system_transfer")
        if not isinstance(transfer, dict):
            raise RuntimeError(
                "System transfer result is missing system_transfer metadata"
            )
        requested = transfer.get("refinement_iterations_requested")
        completed = transfer.get("refinement_iterations_completed")
        if (
            isinstance(requested, bool)
            or isinstance(completed, bool)
            or not isinstance(requested, int)
            or not isinstance(completed, int)
            or requested < 0
            or completed != requested
        ):
            raise RuntimeError(
                "Refusing to persist an incomplete system transfer result: "
                f"refinement iterations {completed}/{requested}"
            )

        pareto_configs = results.get("pareto_configs") or []
        pareto_objectives = results.get("pareto_objectives") or []
        pareto_eval_ids = results.get("pareto_eval_ids") or []
        pareto_algo_hashes = results.get("pareto_algo_hashes") or []
        sizes = {
            len(pareto_configs),
            len(pareto_objectives),
            len(pareto_eval_ids),
            len(pareto_algo_hashes),
        }
        if len(sizes) != 1:
            raise RuntimeError(
                "Refusing to persist a misaligned system transfer Pareto "
                "ledger: configs/objectives/eval_ids/algo_hashes have sizes "
                f"{sorted(sizes)}"
            )
        if any(
            isinstance(eval_id, bool) or not isinstance(eval_id, int)
            for eval_id in pareto_eval_ids
        ):
            raise RuntimeError(
                "System transfer pareto_eval_ids must be target integer eval "
                "IDs, not source algorithm hashes"
            )

        assembly_results = results.get("assembly_results") or []
        payload = {
            "kind": "system_transfer_result",
            "status": "completed",
            "pareto_configs": deepcopy(pareto_configs),
            "pareto_objectives": deepcopy(pareto_objectives),
            "pareto_eval_ids": list(pareto_eval_ids),
            "pareto_algo_hashes": list(pareto_algo_hashes),
            "total_cost": results.get("total_cost", 0.0),
            "cm_count": results.get("cm_count", 0),
            "gt_count": results.get("gt_count", 0),
            "assembly_results_count": len(assembly_results),
            "hardware": deepcopy(results.get("hardware") or {}),
            "system_transfer": deepcopy(transfer),
        }
        validate_system_transfer_artifact(
            payload,
            expected_kind="system_transfer_result",
            source="system_transfer_result.json",
        )

        import tempfile

        os.makedirs(self.project_dir, exist_ok=True)
        path = os.path.join(self.project_dir, "system_transfer_result.json")
        fd, temp_path = tempfile.mkstemp(
            prefix=".system_transfer_result.", suffix=".tmp",
            dir=self.project_dir,
        )
        try:
            with os.fdopen(fd, "w") as stream:
                safe_json_dump(payload, stream, indent=2, default=str)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, path)
        except Exception:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            raise

    def _assert_exact_transfer_warm_start(
        self, replay_evaluations: List[dict],
    ) -> None:
        """Fail closed unless the optimiser restored every replayed point."""
        restored = getattr(self.optimizer, "_configs", None)
        if not isinstance(restored, list):
            raise RuntimeError(
                f"Optimizer {type(self.optimizer).__name__} does not expose "
                "a verifiable _configs observation ledger for system transfer"
            )
        expected_configs = [item["config"] for item in replay_evaluations]
        if len(restored) != len(expected_configs):
            raise RuntimeError(
                "System transfer warm_start was partial: restored "
                f"{len(restored)}/{len(expected_configs)} observations"
            )
        expected_hashes = [self._quality_algo_key(cfg) for cfg in expected_configs]
        restored_hashes = [self._quality_algo_key(cfg) for cfg in restored]
        if restored_hashes != expected_hashes:
            raise RuntimeError(
                "System transfer warm_start changed, reordered or omitted the "
                "source observation history"
            )

    def _run_refinement(
        self,
        assembly: RAGCMAssembly,
        quality_store: QualityStore,
        n_iterations: int,
        configs: List[dict],
        quality_scores: List[float],
        performance_scores: List[float],
        config_hashes: List[str],
        target_eval_ids: List[int],
        assembly_results: List[AssemblyResult],
        obj_columns: List[str],
        transfer_state: dict | None = None,
        refinement_index_offset: int = 0,
    ) -> int:
        """Phase 2b: optimizer-guided refinement.

        Uses the existing optimiser's suggest→evaluate→add cycle.  Quality
        evaluations are served from the store when available; new ones are
        run and written back.

        Mutates ``configs``, ``quality_scores``, ``performance_scores``,
        ``config_hashes``, ``target_eval_ids``, and ``assembly_results`` in
        place.

        Returns the number of new GT evaluations performed.
        """
        optimizer = self.optimizer
        if hasattr(self, "assembly") and assembly is not self.assembly:
            raise RuntimeError(
                "Refinement assembly does not match the Controller target assembly"
            )
        external_store = getattr(self, "_external_quality_store", None)
        if (
            hasattr(quality_store, "store_path")
            and hasattr(external_store, "store_path")
            and os.path.abspath(quality_store.store_path)
            != os.path.abspath(external_store.store_path)
        ):
            raise RuntimeError(
                "Refinement QualityStore does not match the attached target store"
            )
        required_objectives = set(optimizer.objective_costs)
        refinement_profile_dependency: dict | None = None
        if transfer_state is not None and transfer_state.get("replay_plan") is not None:
            refinement_profile_dependency = (
                transfer_state["replay_plan"].get("llmlingua2_profile")
            )
            if not isinstance(refinement_profile_dependency, dict):
                raise RuntimeError(
                    "Selective system-transfer refinement has no frozen "
                    "CM profile dependency"
                )
            self._assert_transfer_dependency_unchanged(
                refinement_profile_dependency,
            )
        landed = 0
        gt_count = 0
        raw_attempts = 0
        raw_cap = max(2 * n_iterations, 20)

        # The complete replay history already satisfies the optimiser's init
        # requirement.  There is intentionally no extra n_both_init suggest:
        # YAML N means exactly N new landed optimiser observations.
        while landed < n_iterations:
            if raw_attempts >= raw_cap:
                raise RuntimeError(
                    "System transfer refinement safety valve: "
                    f"{raw_attempts} raw attempts produced only "
                    f"{landed}/{n_iterations} landed observations"
                )
            t0 = time.monotonic()
            eval_id_start = self._eval_counter
            suggested_configs, eval_obj_names = optimizer.suggest(
                n_suggestions=1,
            )
            if suggested_configs is None or eval_obj_names is None:
                raise RuntimeError(
                    "System transfer refinement exhausted the search space "
                    f"after {landed}/{n_iterations} iterations"
                )
            if set(eval_obj_names) != required_objectives or "quality" not in eval_obj_names:
                raise RuntimeError(
                    "Dynamic system-transfer refinement requires every new "
                    "candidate to evaluate quality+trace and performance; "
                    f"optimizer requested {eval_obj_names}"
                )
            if len(suggested_configs) != 1:
                raise RuntimeError(
                    "System-transfer refinement requires one candidate per "
                    f"iteration; optimizer returned {len(suggested_configs)}"
                )
            raw_attempts += 1

            before_assembly = len(self._assembly_results)
            if refinement_profile_dependency is not None:
                self._assert_transfer_dependency_unchanged(
                    refinement_profile_dependency,
                )
            try:
                (
                    obj_dict, valid_configs, n_success, free_flags,
                ) = self._evaluate_objectives(
                    suggested_configs, eval_obj_names, obj_columns,
                )
            finally:
                if refinement_profile_dependency is not None:
                    # Covers replacement during GT work, an overlapped CM, or
                    # the CM sweep itself. The optimizer/refinement commit is
                    # still untouched at this point, so any mismatch aborts the
                    # paper run without landing a mixed-profile observation.
                    self._assert_transfer_dependency_unchanged(
                        refinement_profile_dependency,
                    )
            if not valid_configs:
                logger.warning(
                    "Refinement attempt %d produced no optimiser observation; "
                    "retrying without consuming the requested iteration budget",
                    raw_attempts,
                )
                continue

            perf_values = obj_dict.get("performance") or []
            quality_values = obj_dict.get("quality") or []
            if not (
                len(valid_configs) == len(perf_values) == len(quality_values)
                == len(free_flags)
            ):
                raise RuntimeError(
                    "Refinement evaluator returned misaligned configs/objectives: "
                    f"configs={len(valid_configs)}, performance={len(perf_values)}, "
                    f"quality={len(quality_values)}, free={len(free_flags)}"
                )
            if len(valid_configs) != 1:
                raise RuntimeError(
                    "System-transfer refinement must land exactly one "
                    "optimizer observation per iteration; evaluator returned "
                    f"{len(valid_configs)}"
                )
            eids = list(self._last_valid_eval_ids)
            if len(eids) != len(valid_configs):
                raise RuntimeError(
                    "Refinement evaluator returned misaligned valid eval ids: "
                    f"configs={len(valid_configs)}, ids={len(eids)}"
                )
            persisted_rows = (
                self._transfer_csv_rows() if transfer_state is not None else {}
            )
            if transfer_state is not None and any(
                (persisted_rows.get(eval_id) or {}).get("eval_type")
                == EvalType.CONTEXT_OVERFLOW.value
                for eval_id in eids
            ):
                # CONTEXT_OVERFLOW is a useful blacklist/eval-ledger event, but
                # it has no successful quality trace and therefore cannot be a
                # dynamic system-transfer observation. Keep its eval-id gap and
                # persisted invalid record, discard any in-memory CM summary,
                # and continue toward exactly N trace-backed landed points.
                del self._assembly_results[before_assembly:]
                logger.warning(
                    "Refinement attempt %d recorded CONTEXT_OVERFLOW without "
                    "a quality trace; excluded from transfer optimizer history",
                    raw_attempts,
                )
                continue
            committed_records: list[tuple[dict, float, Any, str]] = []
            for position, (cfg, perf, quality) in enumerate(zip(
                valid_configs, perf_values, quality_values,
            )):
                eval_id = eids[position]
                record_hash = self._quality_algo_key(cfg)
                persisted = self._external_quality_store.get_record(record_hash)
                if persisted is None:
                    try:
                        persisted = self._recover_refinement_quality_store_record(
                            quality_store=quality_store,
                            config=cfg,
                            eval_id=eval_id,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            "Refinement quality+trace recovery into the "
                            "QualityStore failed for "
                            f"eval_{eval_id:04d} (algo_hash={record_hash}): "
                            f"{_format_exception(exc)}"
                        ) from exc
                if persisted is None:
                    raise RuntimeError(
                        "Refinement quality+trace was not committed to the "
                        f"QualityStore (algo_hash={record_hash})"
                    )
                self._external_quality_store.get_execution_dag(record_hash)
                committed_records.append((cfg, float(perf), quality, record_hash))

            # Only mutate the optimizer after objective alignment and both the
            # target evaluation ledger and QualityStore trace commit succeeded.
            if transfer_state is not None:
                for position, (cfg, perf, quality, record_hash) in enumerate(
                    committed_records
                ):
                    eval_id = eids[position]
                    row = persisted_rows.get(eval_id)
                    if row is None:
                        raise RuntimeError(
                            "Refinement target evaluation has no CSV commit at "
                            f"eval_{eval_id:04d}"
                        )
                    marker_path = os.path.join(
                        self.project_dir, "evaluations", f"eval_{eval_id:04d}",
                        _TRANSFER_REFINEMENT_MARKER_FILENAME,
                    )
                    if os.path.exists(marker_path):
                        raise RuntimeError(
                            "Refusing to overwrite an existing refinement "
                            f"commit at eval_{eval_id:04d}"
                        )
                    self._save_transfer_refinement_commit(
                        state=transfer_state,
                        refinement_index=refinement_index_offset + landed + position,
                        eval_id=eval_id,
                        config=cfg,
                        performance=perf,
                        quality=quality,
                        algo_hash=record_hash,
                        eval_type=row.get("eval_type") or "",
                        free=bool(free_flags[position]),
                        gt_budget_count=int(n_success),
                        quality_store=quality_store,
                    )
            try:
                optimizer.add_objectives(
                    valid_configs,
                    obj_dict,
                    eval_ids=eids,
                    free_mask=free_flags if any(free_flags) else None,
                )
            except TypeError:
                optimizer.add_objectives(
                    valid_configs, obj_dict, eval_ids=eids,
                )
            self._push_perf_attribution(optimizer, eids)

            for cfg, perf, quality, record_hash in committed_records:
                configs.append(cfg)
                performance_scores.append(perf)
                if isinstance(quality, dict):
                    quality_scores.append(float(quality["combined"]))
                else:
                    quality_scores.append(float(quality))
                config_hashes.append(record_hash)
            target_eval_ids.extend(eids)

            # _evaluate_objectives writes full target CM evaluation records
            # and appends only their portable summaries to the shared list.
            if assembly_results is not self._assembly_results:
                assembly_results.extend(self._assembly_results[before_assembly:])
            landed += len(valid_configs)
            gt_count += n_success
            self._gt_count += n_success
            self._cm_count += len(valid_configs)
            self._total_cost += sum(
                optimizer.objective_costs[name] for name in eval_obj_names
            )
            self._update_summary()

            logger.info(
                "  Transfer refine %d/%d | eval=%s | target eval_id=%d | "
                "time=%.1fs",
                landed, n_iterations, eval_obj_names, eval_id_start,
                time.monotonic() - t0,
            )

        return gt_count

    def _resolve_config(self, decoded_config: dict) -> dict:
        """Apply decoded optimizer params to the base YAML config.

        Delegates to :meth:`PerformanceContext.resolve_config` (the body was
        extracted there so the live path and the historical-surrogate benchmark
        share identical config-resolution). ``self._ctx`` holds the same
        ``config`` / ``_ssops`` / ``_data`` objects aliased onto ``self``, so
        the output is byte-identical to the pre-extraction behaviour.
        """
        return self._ctx.resolve_config(decoded_config)

    def _configure_rag_ir_mode(self, pipeline_config: dict) -> None:
        """Select the configured CM mode for the uniform evaluator result.

        ``global.rag_ir_mode`` is ``dynamic`` by default (the quality run
        returns trace + workflow-schema CM feedback for this eval). ``static``
        selects aggregate token-stat pricing explicitly.
        """
        from rag_stack.performance_context import stamp_rag_ir_mode
        stamp_rag_ir_mode(self.config, pipeline_config)

    def _cm_corpus_id(self) -> str:
        """The run's dataset identifier — the EXPLICIT mandatory ``dataset.dataset_name``
        from the config, the SAME value the eval backend keys its saved imbalance
        profile by. Never derived from a file path. The IVF cost model combines it
        with each config's embedding_model via the shared ``imbalance_key``.
        """
        from rag_stack.cost_model.workload_profile import workload_corpus_id

        return workload_corpus_id(self.config, self._data)

    def _cm_workload_search_space(self) -> Optional[dict]:
        """Enumerate embedding_models / nlist / nprobe for the FAISS workload-profile
        prebuild, or ``None`` when the run uses no FAISS retrieval.

        ``nlist`` is DERIVED from the corpus size (round(factor·√N) for swept/fixed
        ``nlist_factor``, plus any absolute ``nlist``) using the raw corpus row
        count as the chunk-agnostic stand-in (cell-imbalance is ~chunk-invariant).
        ``embedding_model`` falls back to the FAISS vectordbs' configured value
        when it isn't a search dim.
        """
        from rag_stack.cost_model.workload_profile import (
            build_workload_search_space,
        )

        return build_workload_search_space(
            self.config, self.search_space, self.corpus_data
        )

    def _make_cm_vector_provider(self):
        """Build the ``embedding_model -> (base_vecs, query_vecs)`` closure handed to
        the cost model via the CmInitContext.

        Reuses the PRODUCTION SRE encode path (``EmbeddingModel`` registry +
        ``embedding_cache.get_or_encode``) so the corpus encode hits the global
        cache. Base = the corpus contents; query = up to 512 QA queries. Fully
        guarded — any failure returns ``None`` so the CM falls back to load-only.
        """
        from rag_stack.cost_model.workload_profile import (
            make_workload_vector_provider,
        )

        return make_workload_vector_provider(
            corpus_data=self.corpus_data,
            qa_data=self.qa_data,
            corpus_key=self._cm_corpus_id(),
        )

    def _init_cost_models(self) -> None:
        """Preflight only the selected component modules, then freeze them.

        This is deliberately fail-closed.  A missing/stale required component
        profile makes a cost-model run unpriceable; it is never converted into
        an IVF ``f=1`` assumption, a generic HNSW predictor, or another module's
        default.  Each selected component owns its own build-on-miss path and a
        failure in that path aborts startup before any estimate is consumed.
        """
        from rag_stack.cost_model.base import CmInitContext
        from rag_stack.cost_model.sweep_inputs import (
            build_performance_stage_cost_models,
        )
        from rag_stack.performance_context import PERFORMANCE_SOURCE_COST_MODEL

        if (
            getattr(self._ctx, "performance_source", None)
            != PERFORMANCE_SOURCE_COST_MODEL
        ):
            return

        ss = self._cm_workload_search_space() or {}
        vector_provider = self._make_cm_vector_provider() if ss else None

        ctx = CmInitContext(
            config=self.config,
            corpus_key=self._cm_corpus_id(),  # corpus part; embedding resolved per-config
            query_sample=None,
            index_builder=None,  # legacy (eval backend can still populate lazily)
            search_space=ss,
            vector_provider=vector_provider,
        )
        selected = build_performance_stage_cost_models(self.config)
        # One LLM CostModelModule instance may own both its prefill and decode
        # PerformanceStages.  Preflight that component exactly once.
        initialized: set[int] = set()
        for performance_stage, module in selected.items():
            identity = id(module)
            if identity in initialized:
                continue
            try:
                module.init(ctx)
            except Exception as exc:
                raise RuntimeError(
                    "cost-model component preflight failed for "
                    f"{performance_stage!r} ({type(module).__name__}): {exc}"
                ) from exc
            initialized.add(identity)

    # ----------------------------------------------------------------------
    # Private persistence: disk I/O
    # ----------------------------------------------------------------------

    def _save_measured_prelaunch_config(
        self,
        eval_id: int,
        pipeline_config: dict,
        system_config: dict,
    ) -> None:
        """Persist the resolved measured deployment before vLLM launch."""
        eval_dir = os.path.join(
            self.project_dir, "evaluations", f"eval_{eval_id:04d}"
        )
        try:
            os.makedirs(eval_dir, exist_ok=True)
            with open(os.path.join(eval_dir, "pipeline_config.json"), "w") as f:
                safe_json_dump(pipeline_config, f, indent=2, default=str)
            with open(os.path.join(eval_dir, "system_config_resolved.json"), "w") as f:
                safe_json_dump(system_config, f, indent=2, default=str)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"eval_{eval_id:04d}: could not write measured prelaunch config: {e}"
            )

        from rag_stack.system_layout import (
            engine_devices,
            engine_info,
            engine_role_devices,
            stage_devices,
        )

        layout = system_config.get("layout") or {}
        gen = engine_info(system_config, "generator")
        gen_devices = engine_devices(system_config, "generator")
        if gen.get("pd_serving") == "disagg_pd":
            gen_prefill = engine_role_devices(system_config, "generator", "prefill")
            gen_decode = engine_role_devices(system_config, "generator", "decode")
        else:
            gen_prefill = []
            gen_decode = []
        qe_devices = (
            engine_devices(system_config, "query_expansion")
            if "query_expansion" in (layout.get("engines") or {})
            else []
        )
        logger.info(
            f"eval_{eval_id:04d} measured deployment: "
            f"slots={layout.get('total_gpu_slots')} "
            f"(bounds={layout.get('min_total_gpu_slots')}-"
            f"{layout.get('max_total_gpu_slots')}, "
            f"available={(system_config.get('hardware') or {}).get('available_gpu_count')}), "
            f"serving={gen.get('serving')}, "
            f"generator_devices={gen_devices}, "
            f"prefill={gen_prefill}, "
            f"decode={gen_decode}, "
            f"query_expansion_devices={qe_devices}, "
            f"encode={stage_devices(system_config, 'semantic_retrieval')}"
        )

    def _pending_optimizer_observation_metadata(self) -> Optional[dict]:
        provider = getattr(
            getattr(self, "optimizer", None),
            "pending_observation_metadata",
            None,
        )
        if not callable(provider):
            return None
        metadata = provider()
        if not isinstance(metadata, dict):
            raise RuntimeError(
                "pending_observation_metadata() must return a mapping"
            )
        return metadata

    def _save_eval_record(
        self,
        eval_id: int,
        config: dict,
        pipeline_config: dict,
        eval_type: EvalType,
        objectives: Dict[str, float],
        quality_metrics: Optional[dict],
        assembly_result: Optional[AssemblyResult],
        obj_columns: List[str],
        execution_dag: Optional[Any] = None,
        performance_execution_trace: Optional[Any] = None,
        eval_error: str = "",
        system_config: Optional[dict] = None,
        strict: bool = False,
    ) -> bool:
        """Persist a single evaluation record to disk immediately.

        Writes per-evaluation JSON files and appends to the shared CSV.
        Normal runs keep best-effort persistence for backward compatibility.
        System transfer sets ``_strict_eval_persistence`` (or passes
        ``strict=True``) so a partial target ledger aborts immediately.

        The ``execution_dag`` is written to its OWN file separate from
        ``quality_metrics.json`` so the metric file stays a pure
        ``{metric_name: float}`` payload. Quality producers emit THE canonical
        quality-trace envelope; static_gt may emit aggregate dicts with
        rag_ir runtime token stats. Measured mode additionally writes its
        independent completion-window sample to
        ``performance_execution_trace.json``; it is never substituted by the
        quality trace.

        ``system_config``: the trial's RESOLVED deployment (joint-space cell —
        the optimizer sampled it). Written verbatim to ``system_config.json``;
        the CM-selected reconstruction below stays the cm-swept fallback.
        """
        optimizer_metadata = self._pending_optimizer_observation_metadata()

        # Failure-type ledger (07-08 user rule): INVALID / DUPLICATE / any
        # non-infeasibility failure never consumes a paid seat (two-counter
        # rule) but must be LOUD — each occurrence warns, and the completion
        # summary repeats the tally so a scarred run can't look clean.
        et = str(eval_type)
        if et not in ("GT", "GT_FP", "CM"):
            self._failure_counts[et] = self._failure_counts.get(et, 0) + 1
            if et not in ("CONTEXT_OVERFLOW", "OOM"):
                logger.warning(
                    "eval_%04d recorded as %s (budget-free, NOT a valid eval) "
                    "— this failure type should be zero in a healthy run; "
                    "eval_error: %.160s",
                    eval_id, et, eval_error or "<none>",
                )
        try:
            eval_dir = os.path.join(
                self.project_dir, "evaluations", f"eval_{eval_id:04d}"
            )
            os.makedirs(eval_dir, exist_ok=True)

            # 1. Pipeline config (secrets stripped via safe_json_dump)
            with open(os.path.join(eval_dir, "pipeline_config.json"), "w") as f:
                safe_json_dump(pipeline_config, f, indent=2, default=str)

            if optimizer_metadata is not None:
                with open(
                    os.path.join(
                        eval_dir,
                        _OPTIMIZER_OBSERVATION_FILENAME,
                    ),
                    "w",
                ) as f:
                    safe_json_dump(
                        optimizer_metadata,
                        f,
                        indent=2,
                        default=str,
                    )

            # 1b. Exact resolved deployment.  Measured evaluations do not
            # have an AssemblyResult: their concrete system is resolved
            # before launch and passed directly to this recorder.  Persist it
            # independently of assembly so every successful measured eval is
            # a self-contained, replayable (pipeline, system) pair.
            if system_config is not None:
                with open(os.path.join(eval_dir, "system_config.json"), "w") as f:
                    safe_json_dump(system_config, f, indent=2, default=str)

            # 2. GT metrics (if available)
            if quality_metrics is not None:
                with open(os.path.join(eval_dir, "quality_metrics.json"), "w") as f:
                    safe_json_dump(quality_metrics, f, indent=2)

            # 2b. Execution DAG / aggregate runtime payload; never mixed into metrics.
            if execution_dag is not None:
                with open(os.path.join(eval_dir, "execution_dag.json"), "w") as f:
                    safe_json_dump(execution_dag, f, indent=2, default=str)

            # 2c. Independent measured performance-workload trace. Keeping a
            # distinct filename prevents resume/replay readers from confusing
            # first-completion quality winners with the completion-window
            # workload that a CM comparison must price.
            if performance_execution_trace is not None:
                with open(
                    os.path.join(eval_dir, "performance_execution_trace.json"),
                    "w",
                ) as f:
                    safe_json_dump(
                        performance_execution_trace,
                        f,
                        indent=2,
                        default=str,
                    )

            # 3. Assembly result details
            if assembly_result is not None:
                serialized = assembly_result.to_dict()
                with open(os.path.join(eval_dir, "assembly.json"), "w") as f:
                    safe_json_dump(serialized, f, indent=2, default=str)

                # 3a. A direct resolved system was already persisted above.
                # For a CM-swept eval without one, reconstruct the SELECTED
                # deployment through the SAME resolve_system_config that the
                # measured path consumes.
                if system_config is None:
                    try:
                        dims = self._measured_dims_from_selection(
                            assembly_result.best_hw_params or {}, pipeline_config,
                        )
                        if dims:
                            scfg = self._ctx.resolve_system_config(
                                {}, resolved_config=pipeline_config,
                                system_dims_override=dims,
                            )
                            with open(os.path.join(eval_dir, "system_config.json"), "w") as f:
                                safe_json_dump(scfg, f, indent=2, default=str)
                    except Exception as exc:  # noqa: BLE001 — recording is best-effort
                        logger.debug(f"eval_{eval_id} system_config.json skipped: {exc}")

                # Save RAGO sweep DataFrame
                if (assembly_result.rago_result_df is not None
                        and len(assembly_result.rago_result_df) > 0):
                    assembly_result.rago_result_df.to_csv(
                        os.path.join(eval_dir, "rago_sweep.csv"),
                        index=False,
                    )
                # Save per-stage sweep DataFrames
                if assembly_result.sweep_dfs:
                    sweep_dir = os.path.join(eval_dir, "sweep_dfs")
                    os.makedirs(sweep_dir, exist_ok=True)
                    for stage_name, df in assembly_result.sweep_dfs.items():
                        df.to_csv(
                            os.path.join(sweep_dir, f"{stage_name}.csv"),
                            index=False,
                        )

            # 4. Append to all_evaluations.csv (includes INVALID)
            from rag_stack.optimizer.utils import append_evaluation_to_csv
            enriched_config = self._ssops.enrich_configs([config], self._base_decoded_config)[0]
            enriched_config = _strip_secrets(enriched_config)
            csv_entry = {
                "config": enriched_config,
                "objectives": objectives,
                "eval_type": eval_type,
                "eval_error": eval_error,
                "eval_id": eval_id,
                "quality_metrics": quality_metrics,
            }
            csv_path = os.path.join(
                self.project_dir, "all_evaluations.csv"
            )
            append_evaluation_to_csv(csv_entry, csv_path, obj_columns)
            return True

        except Exception as e:
            if (
                strict
                or optimizer_metadata is not None
                or getattr(self, "_strict_eval_persistence", False)
            ):
                raise RuntimeError(
                    f"Failed to persist eval_{eval_id:04d}: "
                    f"{_format_exception(e)}"
                ) from e
            logger.warning(f"Failed to save eval record {eval_id}: {e}")
            return False

    @staticmethod
    def _measured_dims_from_selection(
        best_hw_params: dict, pipeline_config: dict,
    ) -> dict | None:
        """CM-selected sweep row → the measured system dims vocabulary.

        Translates ``best_hw_params`` (placement groups, per-stage chips,
        stamped TP/PP, batches, timeout) into the ``system_dims_override``
        dict :meth:`PerformanceContext.resolve_system_config` accepts, so the
        SELECTED deployment can be serialized in the exact schema the
        measured runtime consumes.
        """
        import ast as _ast
        from rag_stack.search_space.placement import (
            measured_gpu_stages, stage_engine, cut_after_key, num_gpus_key,
        )
        bhp = best_hw_params or {}
        if not bhp:
            return None
        candidate_layout = None
        if bhp.get("candidate_system_layout") is not None:
            from rag_stack.cost_model.candidate_layout import (
                candidate_system_layout_from_row,
            )
            candidate_layout = candidate_system_layout_from_row(bhp)
        stages = measured_gpu_stages(pipeline_config)
        if not stages:
            return None
        groups = bhp.get("collocation_strategy")
        if candidate_layout is not None:
            groups = [
                list(group.get("performance_stages") or [])
                for group in candidate_layout.get("resource_groups", [])
            ]
        if isinstance(groups, str):
            try:
                groups = _ast.literal_eval(groups)
            except Exception:  # noqa: BLE001
                groups = None
        gidx: dict = {}
        if isinstance(groups, (list, tuple)):
            for gi, grp in enumerate(groups):
                members = grp if isinstance(grp, (list, tuple, set)) else [grp]
                for m in members:
                    gidx[str(m)] = gi
        dims: dict = {}
        for a, b in zip(stages, stages[1:]):
            ga, gb = gidx.get(a), gidx.get(b)
            dims[cut_after_key(a)] = int(
                ga is not None and gb is not None and ga != gb
            )
        # Per-stage chips: the engine key carries the max (compat fallback);
        # when an LLM's prefill/decode chips DIFFER the role-qualified keys
        # (num_gpus_<stage>) carry the true asymmetric counts — derive_layout
        # prefers them, so a CM-selected 1P2D row round-trips faithfully
        # instead of being max()ed into 2P2D.
        stage_chips: dict = {}
        candidate_stages = (
            candidate_layout.get("performance_stages", {})
            if candidate_layout is not None else {}
        )
        for st in stages:
            eng = stage_engine(st)
            candidate_info = candidate_stages.get(st) or {}
            chips = candidate_info.get("num_chips", bhp.get(f"num_chips_{st}"))
            if chips:
                stage_chips[st] = int(chips)
                dims[num_gpus_key(eng)] = max(
                    int(chips), int(dims.get(num_gpus_key(eng), 1)),
                )
        for st, chips in stage_chips.items():
            eng = stage_engine(st)
            if st != eng and chips != int(dims.get(num_gpus_key(eng), chips)):
                dims[num_gpus_key(st)] = chips
        for st in stages:
            eng = stage_engine(st)
            candidate_info = candidate_stages.get(st) or {}
            pp = candidate_info.get(
                "pp",
                bhp.get(f"candidate_pp_{st}", bhp.get(f"pp_{st}")),
            )
            if pp:
                role = "prefill" if st.endswith("_prefill") else (
                    "decode" if st.endswith("_decode") else None)
                if role:
                    dims[f"pipeline_parallel_{eng}_{role}"] = int(pp)
                    dims.setdefault(f"pipeline_parallel_{eng}", int(pp))
        if bhp.get("batch_size_request"):
            dims["batch_size_request"] = int(bhp["batch_size_request"])
        if bhp.get("batch_size_decode"):
            dims["batch_size_decode"] = int(bhp["batch_size_decode"])
        if bhp.get("dynamic_batch_timeout_s") is not None:
            try:
                dims["dynamic_batch_timeout_s"] = float(bhp["dynamic_batch_timeout_s"])
            except (TypeError, ValueError):
                pass
        # RAGO names the CPU retrieval deployment axis
        # ``num_retrieval_servers``; PerformanceContext's measured runtime
        # vocabulary is ``retrieval.num_servers`` (carried as the flat
        # ``num_servers`` override).  Preserve it just like GPU chip counts.
        # When a concrete candidate layout is present, its CPU resource is the
        # physical source of truth and must agree with the flat provenance.
        layout_retrieval_servers: set[int] = set()
        if candidate_layout is not None:
            for resource in candidate_layout.get("resource_groups", []):
                if (
                    isinstance(resource, dict)
                    and resource.get("kind") == "cpu_retrieval"
                    and resource.get("num_servers") is not None
                ):
                    layout_retrieval_servers.add(
                        max(1, int(resource["num_servers"]))
                    )
            for info in candidate_stages.values():
                if (
                    isinstance(info, dict)
                    and info.get("kind") == "cpu_retrieval"
                    and info.get("num_servers") is not None
                ):
                    layout_retrieval_servers.add(max(1, int(info["num_servers"])))
            if len(layout_retrieval_servers) > 1:
                raise ValueError(
                    "Selected candidate_system_layout has inconsistent CPU "
                    f"retrieval server counts: {sorted(layout_retrieval_servers)}"
                )
        flat_retrieval_servers = bhp.get("num_retrieval_servers")
        if flat_retrieval_servers is not None:
            flat_retrieval_servers = max(1, int(flat_retrieval_servers))
        if layout_retrieval_servers:
            exact_retrieval_servers = next(iter(layout_retrieval_servers))
            if (
                flat_retrieval_servers is not None
                and flat_retrieval_servers != exact_retrieval_servers
            ):
                raise ValueError(
                    "Selected num_retrieval_servers disagrees with concrete CPU "
                    f"resource: {flat_retrieval_servers} != {exact_retrieval_servers}"
                )
            dims["num_servers"] = exact_retrieval_servers
        elif flat_retrieval_servers is not None:
            dims["num_servers"] = flat_retrieval_servers
        retr = ((pipeline_config.get("system") or {}).get("retrieval") or {})
        if (
            not isinstance(retr.get("faiss_ivf_parallel_mode"), (list, dict))
            and retr.get("faiss_ivf_parallel_mode") is not None
        ):
            dims["faiss_ivf_parallel_mode"] = int(retr["faiss_ivf_parallel_mode"])
        n_slots = int(
            (candidate_layout or {}).get("total_gpu_slots")
            or bhp.get("num_chips_e2e") or 0
        )
        if n_slots < 1:
            n_slots = max(
                (int(bhp.get(f"num_chips_{s}", 0) or 0) for s in stages),
                default=1,
            )
        if candidate_layout is not None:
            available = [
                str(device)
                for device in candidate_layout.get("available_devices", [])
            ]
            if not available:
                raise ValueError(
                    "Selected candidate_system_layout has no available_devices"
                )
            dims["available_gpus"] = available
            # Reserved bridge consumed by PerformanceContext after it derives
            # the ordinary runtime/vLLM fields.  This prevents a selected
            # same-pair/cross-pair/rank order from being reconstructed from
            # lossy chip counts.
            dims["candidate_system_layout"] = candidate_layout
        else:
            dims["available_gpus"] = [
                f"cuda:{i}" for i in range(max(1, n_slots))
            ]
        return dims

    def _selected_system_design(self, eval_id) -> dict | None:
        """Curated CM-selected deployment of ``eval_id`` (assembly.json
        best_hw_params → design keys only: exact physical layout, placement /
        collocation / chips / TP/PP / batches / timeout + headline metrics)."""
        try:
            eid = int(eval_id)
        except (TypeError, ValueError):
            return None
        path = os.path.join(
            self.project_dir, "evaluations", f"eval_{eid:04d}", "assembly.json",
        )
        try:
            with open(path) as f:
                bhp = (json.load(f) or {}).get("best_hw_params") or {}
        except Exception:  # noqa: BLE001 — missing/failed eval has no design
            return None
        keep_prefixes = (
            "num_chips_", "tp_", "pp_", "batch_size_",
            "candidate_devices_", "candidate_tp_", "candidate_pp_",
            "candidate_replicas_",
        )
        keep_exact = {
            "placement_policy", "collocation_strategy", "num_retrieval_servers",
            "scheduling_policy", "dynamic_batch_timeout_s", "qps", "latency_s",
            "candidate_system_layout", "device_layout_id", "stage_devices",
            "resource_group_devices", "available_devices",
        }
        design = {
            k: bhp[k] for k in sorted(bhp)
            if k.startswith(keep_prefixes) or k in keep_exact
        }
        return design or None

    def _persisted_pareto_pipeline_config(
        self,
        *,
        eval_id: int,
    ) -> dict:
        """Load the exact resolved pipeline that produced a Pareto eval.

        Pareto export happens after every evaluation has been persisted.  The
        per-eval ``pipeline_config.json`` is therefore the authoritative
        recipe: it already contains the selected component's own fixed fields
        plus that eval's resolved corpus metadata (``_token_stats`` and FAISS
        ``N``).  Re-resolving an optimizer-facing/enriched config here can
        reintroduce base-module fields or base-corpus metadata that were never
        used by the evaluation.
        """
        if eval_id < 0:
            raise RuntimeError(
                "Pareto pipeline export requires a concrete non-negative eval_id"
            )
        source_path = os.path.join(
            self.project_dir,
            "evaluations",
            f"eval_{eval_id:04d}",
            "pipeline_config.json",
        )
        try:
            with open(source_path) as stream:
                pipeline_config = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Pareto pipeline export requires the exact persisted config "
                f"for eval_{eval_id:04d}: {source_path}: {exc}"
            ) from exc
        if not isinstance(pipeline_config, dict):
            raise RuntimeError(
                "Pareto pipeline export expected a JSON object for "
                f"eval_{eval_id:04d}: {source_path}"
            )
        return pipeline_config

    def _persisted_pareto_system_config(
        self,
        *,
        eval_id: int,
    ) -> dict:
        """Load the exact concrete deployment used by a Pareto eval.

        New eval records persist ``system_config.json`` directly.  Historical
        measured records wrote the same resolved deployment before launch as
        ``system_config_resolved.json``; accept that filename only when the
        canonical file is absent so completed measured projects remain
        publishable without remeasurement.
        """
        if eval_id < 0:
            raise RuntimeError(
                "Pareto system export requires a concrete non-negative eval_id"
            )
        eval_dir = os.path.join(
            self.project_dir,
            "evaluations",
            f"eval_{eval_id:04d}",
        )
        candidates = (
            os.path.join(eval_dir, "system_config.json"),
            os.path.join(eval_dir, "system_config_resolved.json"),
        )
        source_path = next(
            (path for path in candidates if os.path.lexists(path)),
            None,
        )
        if source_path is None:
            raise RuntimeError(
                "Pareto system export requires the exact persisted system "
                f"config for eval_{eval_id:04d}; checked: "
                + ", ".join(candidates)
            )
        try:
            with open(source_path) as stream:
                system_config = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Pareto system export requires a valid persisted system "
                f"config for eval_{eval_id:04d}: {source_path}: {exc}"
            ) from exc
        if not isinstance(system_config, dict):
            raise RuntimeError(
                "Pareto system export expected a JSON object for "
                f"eval_{eval_id:04d}: {source_path}"
            )
        return system_config

    def _commit_pareto_publication(
        self,
        *,
        transaction_dir: str,
        staged_csv: str | None,
        staged_configs_dir: str | None,
    ) -> None:
        """Replace the CSV/config set as one rollback-safe publication.

        Both new artifacts are fully materialized before this method runs.
        Existing artifacts move into the transaction directory first; any
        failed install restores them.  Replacing the complete config directory
        also guarantees that a resumed run cannot leave stale Pareto files.
        Passing two ``None`` staged paths atomically clears an obsolete empty
        publication.
        """
        target_csv = os.path.join(self.project_dir, "pareto_frontier.csv")
        target_configs = os.path.join(self.project_dir, "pareto_configs")
        previous_csv = os.path.join(transaction_dir, "previous_pareto_frontier.csv")
        previous_configs = os.path.join(transaction_dir, "previous_pareto_configs")

        previous_csv_moved = False
        previous_configs_moved = False
        staged_csv_installed = False
        staged_configs_installed = False
        try:
            if os.path.lexists(target_configs):
                os.replace(target_configs, previous_configs)
                previous_configs_moved = True
            if os.path.lexists(target_csv):
                os.replace(target_csv, previous_csv)
                previous_csv_moved = True

            if staged_configs_dir is not None:
                os.replace(staged_configs_dir, target_configs)
                staged_configs_installed = True
            if staged_csv is not None:
                os.replace(staged_csv, target_csv)
                staged_csv_installed = True
        except Exception as exc:
            rollback_errors: list[str] = []

            def _rollback(source: str, destination: str, label: str) -> None:
                try:
                    if os.path.lexists(source):
                        os.replace(source, destination)
                except Exception as rollback_exc:  # noqa: BLE001
                    rollback_errors.append(f"{label}: {rollback_exc}")

            if staged_csv_installed:
                _rollback(target_csv, staged_csv, "remove staged CSV")
            if staged_configs_installed:
                _rollback(
                    target_configs,
                    staged_configs_dir,
                    "remove staged config directory",
                )
            if previous_csv_moved:
                _rollback(previous_csv, target_csv, "restore previous CSV")
            if previous_configs_moved:
                _rollback(
                    previous_configs,
                    target_configs,
                    "restore previous config directory",
                )
            if rollback_errors:
                raise RuntimeError(
                    "Pareto publication failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from exc
            raise

    def _save_pareto_frontier(self, results: dict):
        """Publish one exact, replayable Pareto snapshot or fail closed."""
        pareto_configs = list(results["pareto_configs"])
        pareto_objectives = list(results["pareto_objectives"])
        pareto_eval_ids = list(results.get("pareto_eval_ids", []))
        lengths = {
            "configs": len(pareto_configs),
            "objectives": len(pareto_objectives),
            "eval_ids": len(pareto_eval_ids),
        }
        if len(set(lengths.values())) != 1:
            raise RuntimeError(
                f"Pareto publication inputs must have equal lengths: {lengths}"
            )

        normalized_eval_ids: list[int] = []
        for raw_eval_id in pareto_eval_ids:
            if isinstance(raw_eval_id, bool):
                raise RuntimeError(
                    f"Pareto publication has invalid eval_id: {raw_eval_id!r}"
                )
            try:
                eval_id = int(raw_eval_id)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Pareto publication has invalid eval_id: {raw_eval_id!r}"
                ) from exc
            if eval_id < 0 or raw_eval_id != eval_id:
                raise RuntimeError(
                    f"Pareto publication has invalid eval_id: {raw_eval_id!r}"
                )
            normalized_eval_ids.append(eval_id)
        if len(set(normalized_eval_ids)) != len(normalized_eval_ids):
            raise RuntimeError(
                "Pareto publication requires unique eval IDs: "
                f"{normalized_eval_ids}"
            )
        pareto_eval_ids = normalized_eval_ids

        if not pareto_configs:
            transaction_dir = tempfile.mkdtemp(
                prefix=".pareto-publication-",
                dir=self.project_dir,
            )
            try:
                self._commit_pareto_publication(
                    transaction_dir=transaction_dir,
                    staged_csv=None,
                    staged_configs_dir=None,
                )
            finally:
                shutil.rmtree(transaction_dir, ignore_errors=True)
            return

        # Validate and load every authoritative config before staging any
        # frontier artifact.  Missing/corrupt persisted configs make the
        # publication incomplete and must prevent a completed project state.
        persisted_pipeline_configs = [
            self._persisted_pareto_pipeline_config(eval_id=eval_id)
            for eval_id in pareto_eval_ids
        ]
        persisted_system_configs = [
            self._persisted_pareto_system_config(eval_id=eval_id)
            for eval_id in pareto_eval_ids
        ]

        transaction_dir = tempfile.mkdtemp(
            prefix=".pareto-publication-",
            dir=self.project_dir,
        )
        try:
            os.makedirs(os.path.join(self.project_dir, "optimizer"), exist_ok=True)
            objective_names = list(pareto_objectives[0].keys()) if pareto_objectives else []

            # Both layers of defense for the saved CSV: structural strip of
            # secret-keyed entries from each config, AND per-cell pattern
            # scrub for any provider-API-key strings hiding in serialized
            # blobs.
            sanitized_configs = [_strip_secrets(cfg) for cfg in pareto_configs]

            # Centralize the ``<top>__<leaf>`` 2-deep flatten convention
            # in RagSearchSpace.flatten_2deep so column headers and per-row
            # cell lookups share one source of truth.
            flat_configs = [self._ssops.flatten_2deep(cfg) for cfg in sanitized_configs]
            all_keys: list = []
            for flat in flat_configs:
                for col in flat:
                    if col not in all_keys:
                        all_keys.append(col)

            staged_csv = os.path.join(transaction_dir, "pareto_frontier.csv")
            with open(staged_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(all_keys + objective_names + ["eval_id"])
                for idx, (flat, objs) in enumerate(zip(flat_configs, pareto_objectives)):
                    row = [flat.get(col, "") for col in all_keys]
                    for name in objective_names:
                        val = objs.get(name, 0.0)
                        if isinstance(val, dict):
                            row.append(f"{val.get('combined', 0.0):.6f}")
                        else:
                            row.append(f"{val:.6f}" if isinstance(val, (int, float)) else str(val))
                    row.append(pareto_eval_ids[idx])
                    writer.writerow(safe_csv_row(row))

            # Save each point as a full recipe plus the exact JSON pair
            # accepted by ``rag_stack eval-one``.
            staged_configs_dir = os.path.join(transaction_dir, "pareto_configs")
            os.makedirs(staged_configs_dir)
            for i, objs in enumerate(pareto_objectives):
                eid = pareto_eval_ids[i]
                exact_pipeline_config = persisted_pipeline_configs[i]
                exact_system_config = persisted_system_configs[i]
                with open(
                    os.path.join(
                        staged_configs_dir,
                        f"pareto_eval_{eid:04d}_pipeline_config.json",
                    ),
                    "w",
                ) as pipeline_stream:
                    safe_json_dump(
                        exact_pipeline_config,
                        pipeline_stream,
                        indent=2,
                        default=str,
                    )
                with open(
                    os.path.join(
                        staged_configs_dir,
                        f"pareto_eval_{eid:04d}_system_config.json",
                    ),
                    "w",
                ) as system_stream:
                    safe_json_dump(
                        exact_system_config,
                        system_stream,
                        indent=2,
                        default=str,
                    )

                # Convert to plain Python types for YAML serialization
                pipeline_config = json.loads(
                    json.dumps(exact_pipeline_config, default=str)
                )
                pipeline_config["_pareto_objectives"] = {
                    k: float(v) for k, v in objs.items()
                }
                pipeline_config["_eval_id"] = eid
                design = self._selected_system_design(eid)
                if design:
                    # The CM-selected deployment point for this eval (from
                    # assembly.json best_hw_params) — placement, per-stage
                    # chips, TP/PP, batches — so the Pareto artifact is a
                    # complete (pipeline recipe + system design) pair.
                    pipeline_config["_system_design"] = design
                config_path = os.path.join(
                    staged_configs_dir,
                    f"pareto_eval_{eid:04d}.yaml",
                )
                # safe_yaml_dump strips secrets before writing.
                with open(config_path, "w") as yf:
                    safe_yaml_dump(pipeline_config, yf, default_flow_style=False)

            self._commit_pareto_publication(
                transaction_dir=transaction_dir,
                staged_csv=staged_csv,
                staged_configs_dir=staged_configs_dir,
            )
            logger.info(
                "Saved %d Pareto points to %s",
                len(pareto_configs),
                os.path.join(self.project_dir, "pareto_frontier.csv"),
            )

            # Save invalid configs if any
            if self._invalid_configs:
                invalid_path = os.path.join(self.project_dir, "invalid_configs.json")
                with open(invalid_path, "w") as f:
                    safe_json_dump(self._invalid_configs, f, indent=2, default=str)
                logger.info(
                    f"Saved {len(self._invalid_configs)} invalid configs to {invalid_path}"
                )
        finally:
            shutil.rmtree(transaction_dir, ignore_errors=True)

    def _update_summary(
        self,
        n_pareto_points: int = 0,
        status: str = "running",
    ):
        """Overwrite summary.json with current progress."""
        try:
            summary = ProjectSummary(
                status=ProjectStatus(status),
                timestamp=datetime.datetime.now().isoformat(),
                pid=os.getpid(),
                total_cost=self._total_cost,
                cm_count=self._cm_count,
                gt_count=self._gt_count,
                n_evaluations=self._eval_counter,
                n_pareto_points=n_pareto_points,
                n_invalid_configs=len(self._invalid_configs),
                n_assembly_results=len(self._assembly_results),
                valid_eval_count=self._gt_count + self._cm_count,
            )
            save_project_summary(self.project_dir, summary)
        except Exception as e:
            logger.warning(f"Failed to update summary: {e}")

    # ----------------------------------------------------------------------
    # Private bookkeeping: in-memory state
    # ----------------------------------------------------------------------

    def _record_invalid(
        self,
        eval_id: int,
        config: dict,
        reason: str,
        *,
        objectives_will_be_committed: bool = False,
    ):
        """Persist one sampled-config verdict and retire it from future pools.

        A deterministic failure either produces no optimizer observation
        (resolve/chunk failures) or lands an explicit zero-objective penalty.
        Only the former is committed here as an unscored optimizer attempt;
        the latter is accounted exactly once by ``add_objectives``.
        """
        # Every caller reaches this helper only after proving a deterministic
        # sampled-config verdict.  Infrastructure failures are audit rows in
        # all_evaluations.csv and deliberately never call _record_invalid.
        # Persist the proof result structurally: a typed CMOutOfMemoryError is
        # otherwise degraded to an unprovable string across resume.
        optimizer_metadata = self._pending_optimizer_observation_metadata()
        record = {
            "eval_id": eval_id,
            "config": config,
            "reason": reason,
            "config_verdict": True,
        }
        if optimizer_metadata is not None:
            record["optimizer_metadata"] = optimizer_metadata
        self._invalid_configs.append(record)
        self._add_invalid_key(config)
        logger.info(f"Invalid config recorded: eval_{eval_id} — {reason}")
        # Persist INCREMENTALLY (07-04): the graceful-shutdown save never runs
        # on SIGINT/crash, so every checkpoint resume re-proposed and re-burned
        # the same deterministic INVALID arm (observed: one CONTEXT_OVERFLOW
        # arm burned 3x across two resumes). Written per-record so a resume
        # (which loads this file, see _load_checkpoint) skips known-bad arms.
        try:
            invalid_path = os.path.join(self.project_dir, "invalid_configs.json")
            with open(invalid_path, "w") as f:
                safe_json_dump(self._invalid_configs, f, indent=2, default=str)
        except Exception as exc:  # noqa: BLE001 — bookkeeping must not break evals
            if optimizer_metadata is not None:
                raise RuntimeError(
                    "invalid_configs.json must be durable before optimizer "
                    f"state advances for eval_{eval_id:04d}"
                ) from exc
            logger.warning(f"invalid_configs.json incremental save failed: {exc}")
            return

        # The invalid ledger is now the durable commit for this unscored
        # attempt. Only after it exists may the optimizer advance its round or
        # forced-channel state.
        _opt = getattr(self, "_opt_ref", None)
        if _opt is not None and hasattr(_opt, "register_infeasible"):
            _opt.register_infeasible(
                config,
                reason,
                eval_id=None if objectives_will_be_committed else eval_id,
                optimizer_metadata=optimizer_metadata,
            )

    def _is_known_invalid(self, cfg: dict) -> bool:
        """Check if this config (or an identical one) already failed."""
        cfg_key = str(sorted(cfg.items()))
        return cfg_key in self._invalid_config_keys

    def _add_invalid_key(self, cfg: dict):
        """Remember this config as invalid."""
        cfg_key = str(sorted(cfg.items()))
        self._invalid_config_keys.add(cfg_key)

    def _build_obj_columns(self, obj_names: List[str]) -> List[str]:
        """Build expanded CSV column names from objective names + composite config."""
        composite = self.config.get("optimizer", {}).get("params", {}).get("composite_objectives", {})
        obj_columns: list[str] = []
        for name in obj_names:
            if name in composite:
                sub_metrics = composite[name].get("sub_metrics", {})
                obj_columns.append(f"{name}__combined")
                for m in sub_metrics:
                    obj_columns.append(f"{name}__{m}")
            else:
                obj_columns.append(name)
        return obj_columns
