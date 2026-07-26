"""Independent project previews: observed evaluator RAG-IR -> real CM compilation.

This module never constructs an optimizer or writes its project evaluation state.
A preview owns a subprocess session and a private dataset workspace. Job state,
inputs, observed RAG-IR and a display-only CM description survive restarts.
Executable CM objects stay in the worker and cannot be restored from the preview.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

SCHEMA_VERSION = "rag-stack.cm-preview.v1"
ACTIVE = frozenset({"queued", "running"})
ARTIFACTS = frozenset({
    "state.json", "request.json", "source_config.yaml", "algo_config.json",
    "system_config.json", "runtime_config.json", "rag_ir.json",
    "preview.log", "RUN.md", "source_trace.json", "compiled_description.json",
})
_CANDIDATE_KEYS = frozenset({
    "system_design_space", "measured_search_space", "deployment_presets",
    "cm_search_space", "batch_presets", "system_space_owner",
})


class PreviewError(ValueError):
    def __init__(self, code: str, message: str, *, phase: str = "preparing", details=None):
        super().__init__(message)
        self.code, self.phase, self.details = code, phase, details or {}

    def to_dict(self):
        return {"code": self.code, "phase": self.phase,
                "message": str(self), "details": self.details}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def requested_count(request: dict) -> int:
    ids = request.get("question_ids")
    return len(ids) if ids is not None else request.get("sample_count", 8)


def _json_default(value):
    from numbers import Integral, Real
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False,
                                    default=_json_default) + "\n")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def _contained(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise PreviewError("unsafe_path", "Artifact path is outside its owning project")
    return resolved


def preview_directory(project: Path, preview_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", preview_id):
        raise PreviewError("invalid_preview_id", "Invalid preview identifier")
    directory = _contained(project, project / "cm_previews" / preview_id)
    if not directory.is_dir():
        raise FileNotFoundError("Preview not found")
    return directory


def artifact_path(directory: Path, name: str) -> Path:
    if name not in ARTIFACTS:
        raise PreviewError("invalid_artifact", "Unknown preview artifact")
    if name == "compiled_description.json" and recover_state(directory)["status"] != "succeeded":
        raise FileNotFoundError("Compiled preview description is unavailable")
    path = _contained(directory, directory / name)
    if not path.is_file():
        raise FileNotFoundError("Preview artifact not found")
    return path


@contextmanager
def _state_lock(directory: Path):
    with (directory / ".state.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def update_state(directory: Path, **changes):
    with _state_lock(directory):
        state = read_json(directory / "state.json")
        # Cancellation is final even when an evaluator callback races with it.
        if state["status"] == "cancelled":
            return state
        state.update(changes, updated_at=utc_now())
        write_json(directory / "state.json", state)
        return state


def _complete_preview(directory: Path, run_id: str, description: dict) -> bool:
    """Publish only the CM display projection, atomically with respect to cancellation."""
    with _state_lock(directory):
        state = read_json(directory / "state.json")
        if state["run_id"] != run_id or state["status"] not in ACTIVE:
            return False
        write_json(directory / "compiled_description.json", description)
        now = utc_now()
        state.update(status="succeeded", phase="complete", completed_at=now, updated_at=now)
        write_json(directory / "state.json", state)
        return True


def _process_identity(pid: int) -> dict | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": stat[19], "pgid": os.getpgid(pid)}
    except (OSError, IndexError, ValueError):
        return None


def _owns_worker(state: dict) -> bool:
    owner = state.get("worker") or {}
    pid = owner.get("pid")
    if not isinstance(pid, int) or pid <= 1:
        return False
    actual = _process_identity(pid)
    if actual != owner or actual["pgid"] != pid:
        return False
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return (b"rag_stack.cm_preview" in argv and
            state["run_id"].encode() in argv and b"--run-id" in argv)


def recover_state(directory: Path) -> dict:
    with _state_lock(directory):
        state = read_json(directory / "state.json")
        if state["status"] in ACTIVE:
            # Read status and ownership under one lock: a late GET must never
            # replace a worker's just-committed terminal result with interrupted.
            age = time.time() - datetime.fromisoformat(state["updated_at"]).timestamp()
            if not _owns_worker(state) and (state.get("worker") or age > 30):
                state.update(
                    status="interrupted", completed_at=utc_now(), updated_at=utc_now(),
                    error={"code": "worker_interrupted", "phase": state["phase"],
                           "message": "The preview worker is no longer running; saved artifacts remain available.",
                           "details": {}},
                )
                write_json(directory / "state.json", state)
    return state


def detail(directory: Path, *, include_payloads: bool = True) -> dict:
    state = recover_state(directory)
    result = {k: v for k, v in state.items() if k != "worker"}
    result["artifacts"] = [
        {"name": name, "size_bytes": path.stat().st_size}
        for name in sorted(ARTIFACTS)
        if name != "compiled_description.json" or state["status"] == "succeeded"
        if (path := directory / name).is_file() and path.resolve().is_relative_to(directory.resolve())
    ]
    log = directory / "preview.log"
    result["log_tail"] = ""
    if log.is_file() and log.resolve().is_relative_to(directory.resolve()):
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 32_768))
            result["log_tail"] = stream.read().decode("utf-8", errors="replace")
    result["compiled_plan"] = result["rag_ir"] = None
    result["compiled_path_status"] = None
    if state["status"] == "succeeded":
        path = directory / "compiled_description.json"
        available = path.is_file() and path.resolve().is_relative_to(directory.resolve())
        result["compiled_path_status"] = "available" if available else "unavailable"
        if available and include_payloads:
            result["compiled_plan"] = read_json(path)
    if include_payloads:
        path = directory / "rag_ir.json"
        if path.is_file() and path.resolve().is_relative_to(directory.resolve()):
            result["rag_ir"] = read_json(path)
    return result


def list_previews(project: Path) -> list[dict]:
    root = _contained(project, project / "cm_previews")
    if not root.is_dir():
        return []
    rows = []
    for directory in root.iterdir():
        if re.fullmatch(r"[a-f0-9]{32}", directory.name) and directory.is_dir():
            directory = _contained(project, directory)
            if (directory / "state.json").is_file():
                rows.append(detail(directory, include_payloads=False))
    return sorted(rows, key=lambda row: row["created_at"], reverse=True)


def split_config_pair(algo: dict, system: dict | None) -> tuple[dict, dict]:
    """Snapshot algorithm/deployment separately; dataset statistics stay runtime-only."""
    algo = deepcopy(algo)
    physical = algo.pop("system", {}) or {}
    physical.update(deepcopy(system or {}))
    for config in (algo, physical):
        for key in _CANDIDATE_KEYS | {"dataset", "algo_search_space", "optimizer", "_token_stats"}:
            config.pop(key, None)
    for store in algo.get("vectordb", []):
        store.pop("N", None)
    return algo, physical


def create_preview(project: Path, request: dict, *, launch: bool = True) -> dict:
    """Snapshot the selected source, then launch an independent worker session."""
    from rag_stack.replay_cli import validate_resolved_algorithm, validate_resolved_system
    request = deepcopy(request)
    selectors = [request.get(key) is not None for key in ("eval_id", "parameters", "algo_config")]
    if sum(selectors) != 1:
        raise PreviewError("selection_required", "Choose exactly one of eval_id, parameters, or algo_config")
    if request.get("reuse_trace") and request.get("eval_id") is None:
        raise PreviewError("trace_source_required", "Trace reuse requires an existing eval_id")
    if request.get("algo_config") is not None and not request.get("system_config"):
        raise PreviewError("deployment_selection_required", "An explicit algorithm requires system_config")
    if request.get("system_config") is not None:
        validate_resolved_system(request["system_config"])
    source = {"kind": "evaluation" if request.get("eval_id") is not None else
              "parameters" if request.get("parameters") is not None else "resolved",
              "eval_id": request.get("eval_id"), "reuse_trace": bool(request.get("reuse_trace"))}
    config_path = _contained(project, project / "rag_stack_config.yaml")
    if not config_path.is_file():
        raise PreviewError("project_config_missing", "Project configuration is missing")
    source_algo = source_system = source_trace = None
    if source["kind"] == "evaluation":
        eid = request["eval_id"]
        if isinstance(eid, bool) or not isinstance(eid, int) or eid < 0:
            raise PreviewError("invalid_eval_id", "Evaluation id must be a non-negative integer")
        evaluation = _contained(project, project / "evaluations" / f"eval_{eid:04d}")
        algo_path = _contained(project, evaluation / "pipeline_config.json")
        if not algo_path.is_file():
            raise PreviewError("evaluation_config_missing", "This evaluation has no resolved pipeline config")
        source_algo = read_json(algo_path)
        system_path = _contained(project, evaluation / "system_config.json")
        source_system = read_json(system_path) if system_path.is_file() else None
        if request.get("system_config") is not None:
            source_system = request["system_config"]
        source.update(config_path=str(algo_path.relative_to(project)))
        if request.get("reuse_trace"):
            trace_path = _contained(project, evaluation / "execution_dag.json")
            if not trace_path.is_file():
                raise PreviewError("trace_missing", "This evaluation has no saved canonical RAG-IR")
            source_trace = read_json(trace_path)
            source.update(trace_path=str(trace_path.relative_to(project)),
                          trace_recorded_at=datetime.fromtimestamp(trace_path.stat().st_mtime,
                                                                  timezone.utc).isoformat())
    elif source["kind"] == "resolved":
        source_algo, source_system = request["algo_config"], request["system_config"]
        runtime = source_algo.get("pipeline_runtime") or {}
        if not isinstance(runtime, dict):
            raise PreviewError("invalid_resolved_config", "pipeline_runtime must be an object")
        backend = (source_algo.get("global") or {}).get("eval_backend", "static_gt")
        mode = str(runtime.get("rag_dataflow") or runtime.get("mode", "sequential")).lower()
        if backend == "static_gt" and mode not in {"sequential", "react"}:
            raise PreviewError("invalid_resolved_config", "static_gt supports sequential or react execution")
        # Caller-supplied resolved JSON must already meet the public contract;
        # never discard an accidentally submitted dataset or search space.
        validate_resolved_algorithm(source_algo)
    pair = split_config_pair(source_algo, source_system) if source_algo is not None else None
    if pair is not None:
        validate_resolved_algorithm(pair[0])
    preview_id = uuid.uuid4().hex
    root = _contained(project, project / "cm_previews")
    root.mkdir(exist_ok=True)
    directory = root / preview_id
    directory.mkdir()
    (directory / "source_config.yaml").write_text(config_path.read_text())
    if source_algo is not None:
        write_json(directory / "runtime_config.json", source_algo)
        algo, system = pair
        write_json(directory / "algo_config.json", algo)
        write_json(directory / "system_config.json", system)
    if source_trace is not None:
        write_json(directory / "source_trace.json", source_trace)
    try:
        version = importlib.metadata.version("rag-stack")
    except importlib.metadata.PackageNotFoundError:
        version = "source-checkout"
    now = utc_now()
    state = {"schema_version": SCHEMA_VERSION, "preview_id": preview_id, "project": project.name,
             "run_id": uuid.uuid4().hex, "status": "queued", "phase": "preparing",
             "created_at": now, "updated_at": now, "started_at": None, "completed_at": None,
             "source": source, "cohort": {"requested_count": requested_count(request),
                                         "question_ids": [], "count": 0},
             "skip_judge": request.get("skip_judge", True), "error": None, "worker": None,
             "versions": {"preview_schema": SCHEMA_VERSION, "rag_stack": version}}
    write_json(directory / "request.json", request)
    write_json(directory / "state.json", state)
    (directory / "preview.log").touch()
    (directory / "RUN.md").write_text(
        "# CM preview\n\nThis isolated preview does not update optimizer evaluations or Pareto results.\n"
        f"\nRun ID: `{state['run_id']}`. Source: `{source['kind']}`.\n"
        f"Trace reuse requested: `{source['reuse_trace']}`. Judge skipped: `{state['skip_judge']}`.\n"
        f"QA: `{request.get('qa_path')}`. Corpus: `{request.get('corpus_path')}`.\n"
        f"Corpus profile: `{request.get('dataset_name')}`.\n")
    if launch:
        launch_worker(directory)
    return detail(directory)


def launch_worker(directory: Path) -> None:
    state = read_json(directory / "state.json")
    cmd = [sys.executable, "-u", "-m", "rag_stack.cm_preview", "--job-dir", str(directory),
           "--run-id", state["run_id"]]
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    for key in ("RAG_STACK_PROJECT_LOG_FILE", "RAG_STACK_PROJECT_LOG_OWNED_BY_PARENT",
                "RAG_STACK_PROJECT_LOG_STDIO_REDIRECTED"):
        env.pop(key, None)
    # Each worker's data/cache ownership belongs to its private runtime directory.
    env["PROJECT_DIR"] = str(directory / "runtime")
    process = None
    try:
        with (directory / "preview.log").open("ab") as log:
            process = subprocess.Popen(cmd, cwd=root, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        identity = _process_identity(process.pid)
        with _state_lock(directory):
            current = read_json(directory / "state.json")
            if current["status"] in ACTIVE:
                current.update(worker=identity, updated_at=utc_now())
                write_json(directory / "state.json", current)
        # Reap the child without retaining its executable CM objects in the server.
        threading.Thread(target=process.wait, daemon=True).start()
    except Exception as exc:
        if process is not None:
            # This is our just-spawned session, not a PID recovered from disk.
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        update_state(directory, status="failed", completed_at=utc_now(),
                     error={"code": "worker_launch_failed", "phase": "preparing",
                            "message": str(exc), "details": {}})


def cancel_preview(directory: Path) -> dict:
    owned = False
    with _state_lock(directory):
        state = read_json(directory / "state.json")
        if state["status"] in ACTIVE:
            owned = _owns_worker(state)
            state.update(status="cancelled", phase="cancelled", updated_at=utc_now(),
                         completed_at=utc_now(), error=None)
            write_json(directory / "state.json", state)
    if owned and _owns_worker(state):
        # Identity + session check prevents signalling a reused PID or another job.
        try:
            os.killpg(state["worker"]["pgid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
    return detail(directory)


def select_question_ids(available: list, request: dict) -> list:
    names = [str(value) for value in available]
    if len(set(names)) != len(names):
        raise PreviewError("duplicate_question_ids", "The source contains duplicate question IDs", phase="sampling")
    requested = request.get("question_ids")
    if requested is not None:
        if not requested or len({str(x) for x in requested}) != len(requested):
            raise PreviewError("invalid_question_ids", "Question IDs must be non-empty and unique", phase="sampling")
        unknown = [x for x in requested if str(x) not in names]
        if unknown:
            raise PreviewError("unknown_question_ids", "Some question IDs are absent from the source",
                               phase="sampling", details={"question_ids": unknown})
        lookup = dict(zip(names, available))
        return [lookup[str(x)] for x in requested]
    if not available:
        raise PreviewError("empty_dataset", "The source has no questions", phase="sampling")
    return available[:request.get("sample_count", 8)]


def _sample_qa(directory: Path, request: dict) -> tuple[Path, list]:
    import pandas as pd
    path = Path(request["qa_path"])
    frame = pd.read_parquet(path)
    if "qid" not in frame.columns:
        raise PreviewError("question_ids_missing", "QA dataset requires its qid column", phase="sampling")
    ids = select_question_ids(frame["qid"].tolist(), request)
    by_id = {str(value): i for i, value in enumerate(frame["qid"].tolist())}
    subset = frame.iloc[[by_id[str(value)] for value in ids]].copy()
    target = directory / "runtime_qa.parquet"
    subset.to_parquet(target, index=False)
    return target, ids


def _resolve_parameters(directory: Path, request: dict, qa: Path) -> tuple[dict, dict | None]:
    from rag_stack.performance_context import PerformanceContext
    context = PerformanceContext(
        str(directory / "source_config.yaml"), str(directory / "runtime"),
        qa_data_path=str(qa), corpus_data_path=request["corpus_path"],
        validate=False, check_model_cache=False, check_performance_support=False,
    )
    params = request["parameters"]
    try:
        context.search_space.check_membership(params, raise_error=True,
                                              check_all_parameters_present=True)
    except (ValueError, KeyError, RuntimeError) as exc:
        raise PreviewError("invalid_parameters", str(exc), phase="resolving") from exc
    decoded = context._ssops.decode_to_config(context._ssops.encode_arm_params(params))
    algo = context.resolve_config(decoded)
    system = request.get("system_config")
    if system is None and context.system_space_owner == "optimizer":
        system = context.resolve_trial_system_config(decoded, algo)
    # A CM-owned deployment sweep is not a concrete physical plan. The caller
    # must choose a layout; collect the observed trace before reporting this.
    return algo, system


def _fresh_trace(directory: Path, request: dict, algo: dict, system: dict, qa: Path, ids: list) -> dict:
    from rag_stack.evaluator_contract import normalize_backend_result
    from rag_stack.replay_cli import _runtime
    from rag_stack.rag_ir.trace import validate_quality_trace_envelope
    runtime = deepcopy(algo)
    runtime["system"] = deepcopy(system)
    runtime.setdefault("_runtime", {})["rag_ir_mode"] = "dynamic"
    if request.get("skip_judge", True):
        runtime.setdefault("eval_backend_setting", {})["performance_only"] = True
    else:
        runtime.setdefault("eval_backend_setting", {}).pop("performance_only", None)
    # Environment interpolation is private to the worker, after snapshots.
    from rag_stack.utils.config import convert_env_in_dict
    convert_env_in_dict(runtime)
    # Dataset ownership and corpus/token derivation are exactly the public replay boundary.
    manager, evaluator, backend = _runtime(
        runtime, directory / "runtime", qa_path=qa,
        corpus_path=Path(request["corpus_path"]), dataset_name=request.get("dataset_name"),
    )
    view = manager.resolve_corpus((runtime.get("corpus_runtime") or {}).get("chunker") or {})
    manager.apply_corpus_view(runtime, view)
    write_json(directory / "runtime_config.json", runtime)
    captured = []
    def on_trace_ready(trace):
        validate_quality_trace_envelope(trace)
        if (trace.get("provenance") or {}).get("trace_origin") == "reconstructed":
            raise PreviewError("observed_trace_missing", "The evaluator published a reconstructed trace", phase="evaluating")
        snapshot = deepcopy(trace)
        snapshot["provenance"] = {**(snapshot.get("provenance") or {}), "trace_origin": "observed"}
        write_json(directory / "rag_ir.json", snapshot)
        captured.append(trace)
    raw = evaluator.evaluate(runtime, run_dir=str(directory / "runtime" / "evaluation"),
                             metrics_override=[] if request.get("skip_judge", True) else None,
                             on_trace_ready=on_trace_ready)
    normalized = normalize_backend_result(raw, backend=backend, pipeline_config=runtime, question_ids=ids)
    trace = normalized["__execution_dag__"]
    write_json(directory / "runtime_config.json", runtime)
    if backend == "static_gt" and not captured:
        raise PreviewError("observed_trace_missing", "The evaluator did not publish an observed serving trace",
                           phase="evaluating")
    if captured and trace is not captured[-1]:
        raise PreviewError("trace_identity_mismatch", "The evaluator returned a different trace from its callback",
                           phase="evaluating")
    if not captured and (trace.get("provenance") or {}).get("trace_origin") != "observed":
        raise PreviewError("trace_origin_unverified", "The evaluator trace does not establish observed execution",
                           phase="evaluating")
    trace["provenance"] = {**(trace.get("provenance") or {}), "trace_origin": "observed"}
    write_json(directory / "rag_ir.json", trace)
    return trace


def _compile(directory: Path, request: dict, algo: dict, system: dict, trace: dict) -> dict:
    from rag_stack.cost_model.assembly import RAGCMAssembly
    from rag_stack.cost_model.compiled import describe
    from rag_stack.evaluator_contract import bind_trace_workflow
    from rag_stack.system_layout import cm_fixed_pipeline_config
    from rag_stack.types import SystemConfig
    if not isinstance(system.get("layout"), dict) or not system["layout"]:
        raise PreviewError("deployment_selection_required", "Select a concrete system_config; this project owns a deployment search space",
                           phase="compiling")
    runtime = deepcopy(algo)
    if request.get("dataset_name"):
        runtime["dataset"] = {"dataset_name": request["dataset_name"]}
    bind_trace_workflow(runtime, trace)
    runtime = cm_fixed_pipeline_config(runtime, system)
    assembly = RAGCMAssembly(SystemConfig.from_dict(runtime["system"]))
    plan = assembly.compile_fixed(runtime, system, trace=trace)
    return describe(plan)


def run_worker(directory: Path, run_id: str) -> int:
    """Worker entry, also usable with injected real-boundary adapters in CPU tests."""
    state = read_json(directory / "state.json")
    if state["run_id"] != run_id or state["status"] not in ACTIVE:
        return 1
    request = read_json(directory / "request.json")
    state = update_state(directory, status="running", phase="sampling", started_at=utc_now(),
                         worker=_process_identity(os.getpid()))
    if state["status"] == "cancelled":
        return 1
    phase = "sampling"
    try:
        print("Selecting preview questions", flush=True)
        if request.get("reuse_trace"):
            from rag_stack.rag_ir.trace import validate_quality_trace_envelope
            trace = read_json(directory / "source_trace.json")
            validate_quality_trace_envelope(trace)
            origin = (trace.get("provenance") or {}).get("trace_origin")
            if origin != "observed":
                raise PreviewError(
                    "trace_origin_unverified", "The saved trace has no verified observed origin; create a fresh preview",
                    phase="sampling", details={"trace_origin": origin, "fresh_run_required": True},
                )
            ids = select_question_ids([row["question_id"] for row in trace["queries"]], request)
            by_id = {str(row["question_id"]): row for row in trace["queries"]}
            trace["queries"] = [by_id[str(qid)] for qid in ids]
            write_json(directory / "rag_ir.json", trace)
        else:
            qa, ids = _sample_qa(directory, request)
        update_state(directory, cohort={"requested_count": requested_count(request),
                                        "question_ids": ids, "count": len(ids)})
        phase = "resolving"
        update_state(directory, phase=phase)
        if request.get("parameters") is not None:
            algo, system = _resolve_parameters(directory, request, qa)
            original_runtime = deepcopy(algo)
            algo, system = split_config_pair(algo, system)
            from rag_stack.replay_cli import validate_resolved_algorithm
            validate_resolved_algorithm(algo)
            write_json(directory / "algo_config.json", algo)
            write_json(directory / "system_config.json", system)
            write_json(directory / "runtime_config.json", original_runtime)
        else:
            algo = read_json(directory / "algo_config.json")
            system = read_json(directory / "system_config.json")
        if not request.get("reuse_trace"):
            phase = "evaluating"
            update_state(directory, phase=phase)
            print(f"Running real evaluator for {len(ids)} questions", flush=True)
            trace = _fresh_trace(directory, request, algo, system, qa, ids)
        phase = "compiling"
        update_state(directory, phase=phase)
        if read_json(directory / "state.json")["status"] == "cancelled":
            return 1
        print("Compiling observed RAG-IR through RAG-CM", flush=True)
        runtime = read_json(directory / "runtime_config.json")
        description = _compile(directory, request, runtime, system, trace)
        if not _complete_preview(directory, run_id, description):
            return 1
        print("CM preview compiled", flush=True)
        # DatasetManager's copies and evaluator intermediates are worker scratch.
        # Keep failed runs for diagnosis, and every durable preview artifact above.
        shutil.rmtree(directory / "runtime", ignore_errors=True)
        (directory / "runtime_qa.parquet").unlink(missing_ok=True)
        return 0
    except Exception as exc:
        if isinstance(exc, PreviewError):
            error = exc.to_dict()
            status = "blocked"
        elif hasattr(exc, "to_dict") and type(exc).__name__ == "CompilationBlocked":
            error = exc.to_dict()
            error.pop("status", None)
            status = "blocked"
        else:
            error = {"code": "preview_failed", "phase": phase,
                     "message": f"{type(exc).__name__}: {exc}", "details": {}}
            status = "failed"
        print(f"{status}: {error['message']}", flush=True)
        update_state(directory, status=status, phase=error.get("phase", phase),
                     error=error, completed_at=utc_now())
        return 1
    finally:
        state = read_json(directory / "state.json")
        with (directory / "RUN.md").open("a") as stream:
            stream.write(f"\n- {utc_now()}: {state['status']} ({state['phase']}).\n")
            stream.write(f"  Questions: {json.dumps(state['cohort']['question_ids'], default=_json_default)}.\n")
            if state.get("error"):
                stream.write(f"  {state['error']['code']}: {state['error']['message']}\n")


def release_worker_children() -> None:
    """Release only this isolated worker's children before Python joins them.

    Evaluators may cache model processes for adjacent evaluations. A preview
    has no next evaluation, and multiprocessing's interpreter shutdown can
    otherwise wait forever for those cached engines. This is a worker process
    lifecycle boundary, independent of any backend's private cache API.
    """
    import psutil

    children = psutil.Process().children(recursive=True)
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(children, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    try:
        result = run_worker(args.job_dir.resolve(), args.run_id)
    finally:
        release_worker_children()
    raise SystemExit(result)


if __name__ == "__main__":
    main()
