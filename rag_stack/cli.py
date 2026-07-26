"""RAG-Stack CLI.

Unified command-line interface with subcommands:

    python -m rag_stack optimize \\
        --project-dir /path/to/project \\
        --qa-data /path/to/qa.parquet \\
        --corpus-data /path/to/corpus.parquet

    python -m rag_stack quality-sweep \\
        --config configs/rag_stack/sgs_config.yaml \\
        --qa-data datasets/eli5/qa_test.parquet \\
        --corpus-data datasets/eli5/corpus.parquet \\
        --store-path quality_stores/eli5_base

    python -m rag_stack eval-one \\
        path/to/pipeline_config.json \\
        path/to/system_config.json

    python -m rag_stack convert-optimizer-export \\
        --source-project benchmarks/.../msmarco_agent_v2_s45 \\
        --output-project tmp_outputs/rag_stack_projects/msmarco_agent_v2_s45
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
import traceback

from rag_stack.logging_utils import (
    PROJECT_LOG_FILE_ENV,
    PROJECT_LOG_OWNED_BY_PARENT_ENV,
    attach_project_log as _attach_project_log,
    setup_logging as _setup_logging,
)
from rag_stack.project_summary import (
    CONFIG_FILENAME, ProjectStatus,
    load_project_summary, save_project_summary, create_project_summary,
    pid_alive,
)
from rag_stack.project_config import (
    freeze_project_config,
    project_config_lock,
    verify_project_config,
    write_project_config,
)

logger = logging.getLogger("RAG-Stack")


def _sha256_path(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quality_store_for_project_resume(
    *, source_project: str, store_path: str,
) -> tuple[str, bool]:
    """Create a local source store once, or strictly reuse the same store.

    A transfer QualityStore grows during Phase 2b, so rerunning
    ``QualityStore.from_project`` is neither a harmless refresh nor a merge:
    it would reject the directory at best and could blur the frozen Phase 2a
    source set at worst.  Reuse is allowed only when the v2 manifest still
    names this exact source project, its copied source config is byte-identical,
    and every original imported record still carries the source artifact mtime
    captured by the manifest-backed commit marker.

    Returns ``(absolute_store_path, reused)``.
    """
    from rag_stack.quality_store import QualityStore

    source_project = os.path.abspath(source_project)
    store_path = os.path.abspath(store_path)
    manifest_path = os.path.join(store_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        if os.path.isdir(store_path) and any(
            name != ".quality_store.lock" for name in os.listdir(store_path)
        ):
            raise RuntimeError(
                "Existing transfer QualityStore has no manifest.json; "
                f"refusing to rebuild or merge it in place: {store_path}"
            )
        return QualityStore.from_project(source_project, store_path), False

    store = QualityStore.load(store_path)
    manifest = dict(getattr(store, "_manifest", {}) or {})
    source_dirs = [
        os.path.abspath(str(path))
        for path in (manifest.get("source_project_dirs") or [])
    ]
    legacy_source = manifest.get("source_project_dir")
    if legacy_source and os.path.abspath(str(legacy_source)) not in source_dirs:
        source_dirs.append(os.path.abspath(str(legacy_source)))
    if source_dirs != [source_project]:
        raise RuntimeError(
            "Existing transfer QualityStore source-project manifest mismatch: "
            f"expected {[source_project]}, found {source_dirs}"
        )

    source_config = os.path.join(source_project, CONFIG_FILENAME)
    store_config = os.path.join(store_path, "rag_stack_config.yaml")
    if not os.path.isfile(source_config) or not os.path.isfile(store_config):
        raise RuntimeError(
            "Cannot verify existing transfer QualityStore source config: "
            f"source={source_config}, store={store_config}"
        )
    if _sha256_path(source_config) != _sha256_path(store_config):
        raise RuntimeError(
            "Existing transfer QualityStore source config no longer matches "
            f"{source_config}; reset the target project for a new transfer"
        )

    original_hashes = list(manifest.get("last_imported_hashes") or [])
    if not original_hashes:
        raise RuntimeError(
            "Existing transfer QualityStore does not identify its original "
            "from-project records; refusing unsafe reuse"
        )
    for record_hash in original_hashes:
        record = store.get_record(str(record_hash))
        if record is None:
            raise RuntimeError(
                "Existing transfer QualityStore lost an original source "
                f"record: {record_hash}"
            )
        metadata = record.get("record") or {}
        source = metadata.get("source") or {}
        try:
            eval_id = int(source["eval_id"])
            expected_mtime = int(metadata["source_mtime_ns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Existing transfer QualityStore has incomplete source "
                f"identity for {record_hash}"
            ) from exc
        eval_dir = os.path.join(
            source_project, "evaluations", f"eval_{eval_id:04d}",
        )
        source_artifacts = [
            os.path.join(eval_dir, name)
            for name in (
                "quality_metrics.json", "pipeline_config.json",
                "execution_dag.json",
            )
        ]
        if not all(os.path.isfile(path) for path in source_artifacts):
            raise RuntimeError(
                "Existing transfer QualityStore source artifacts are missing "
                f"for {record_hash} (eval_{eval_id:04d})"
            )
        actual_mtime = max(os.stat(path).st_mtime_ns for path in source_artifacts)
        if actual_mtime != expected_mtime:
            raise RuntimeError(
                "Existing transfer QualityStore source artifacts changed "
                f"after import for {record_hash} (eval_{eval_id:04d})"
            )
    return store_path, True


def _mark_project_failed(project_dir: str) -> None:
    """Mark this process's project failed without clobbering a live owner."""
    summary = load_project_summary(project_dir)
    if summary is None:
        return
    owner = summary.pid
    if owner is not None and owner != os.getpid() and pid_alive(owner):
        logger.warning(
            f"summary.json owned by live PID {owner}; "
            f"leaving status='{summary.status.value}' intact."
        )
        return
    summary.status = ProjectStatus.FAILED
    summary.timestamp = datetime.datetime.now().isoformat()
    summary.pid = None
    save_project_summary(project_dir, summary)


def _attach_command_log(output_dir: str, command: str) -> None:
    """Attach a command log after its immutable project input is prepared."""
    _setup_logging()
    try:
        parent_log = os.environ.get(PROJECT_LOG_FILE_ENV)
        if os.environ.get(PROJECT_LOG_OWNED_BY_PARENT_ENV) == "1" and parent_log:
            logger.info("[logging] using parent-owned project log -> %s", parent_log)
        else:
            _attach_project_log(os.path.abspath(output_dir), command or "run")
    except Exception as exc:  # logging must never break the command
        logger.warning("[logging] could not attach project log: %s", exc)


def _install_subproc_reaper():
    """Reap all descendant processes on signal-driven exit.

    The optimize subcommand spins up vllm.LLM instances which in turn spawn
    `VLLM::EngineCore` multiprocessing subprocesses. On SIGTERM/SIGINT
    Python's atexit + __del__ hooks do not reliably fire before the
    interpreter tears down, so those EngineCore children orphan to init and
    keep their GPU memory pinned. We install an explicit handler that walks
    the descendant tree and reaps everything before exiting.

    Also registered as an atexit hook so normal shutdown paths benefit too.
    """
    import atexit
    import os
    import signal
    import sys

    def _reap_self_descendants():
        try:
            import psutil
        except ImportError:
            return
        try:
            me = psutil.Process()
            descendants = me.children(recursive=True)
        except psutil.NoSuchProcess:
            return

        # Resource trackers are lifecycle helpers, not worker processes.  They
        # intentionally ignore SIGTERM and exit when every process holding the
        # tracker pipe closes it.  Killing one here corrupts that protocol: a
        # later SemLock finalizer relaunches an empty tracker and its UNREGISTER
        # messages surface as noisy KeyError tracebacks.  Leave both CPython's
        # tracker and loky's equivalent to their normal pipe-EOF shutdown while
        # continuing to reap every real worker / EngineCore descendant.
        def _is_resource_tracker(process):
            try:
                command = " ".join(process.cmdline())
            except (
                psutil.AccessDenied,
                psutil.NoSuchProcess,
                psutil.ZombieProcess,
                ProcessLookupError,
            ):
                return False
            return (
                "multiprocessing.resource_tracker" in command
                or "joblib.externals.loky.backend.resource_tracker" in command
            )

        descendants = [p for p in descendants if not _is_resource_tracker(p)]
        if not descendants:
            return
        for p in reversed(descendants):
            try:
                p.send_signal(signal.SIGTERM)
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
        _, alive = psutil.wait_procs(descendants, timeout=5.0)
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
        logger.info(
            f"cli: reaped {len(descendants)} subprocess(es) on shutdown "
            f"(SIGKILL: {len(alive)})"
        )

    def _handle(signum, _frame):
        logger.warning(f"cli: received signal {signum}, reaping children before exit")
        _reap_self_descendants()
        # Re-raise as default behaviour after cleanup so the parent sees the
        # expected exit code.
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)
    atexit.register(_reap_self_descendants)


# ── optimize ──────────────────────────────────────────────────────────────

def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _add_optimize_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--project-dir", default=None,
        help=("Project (output) directory. If omitted, it is resolved from "
              ".rag_stack_settings.json 'rag_optimization_project_dir' (treated "
              "as a ROOT — a per-run subdir <config-stem>_<timestamp> is created "
              "under it); when that setting is empty, defaults to "
              "tmp_outputs/rag_stack_projects/ under the repo root."))
    parser.add_argument("--qa-data", default=None,
                        help=("Path to QA parquet file. Optional: falls back to the config's "
                              "`data.qa` when omitted. Also optional when the project already "
                              "has data/qa.parquet from a prior run (resume case — Controller "
                              "always uses the in-project copy to keep quality comparable)."))
    parser.add_argument("--corpus-data", default=None,
                        help=("Path to corpus parquet file (raw corpus when the chunker is in "
                              "the search space, else a final corpus). Optional: falls back to "
                              "the config's `data.corpus` when omitted. Also optional on resume "
                              "(Controller always uses the in-project data/corpus.parquet)."))
    parser.add_argument("--config", default=None,
                        help="Path to config YAML (copied into a draft project; "
                             "must match the frozen config on resume)")
    parser.add_argument("--n-iterations", type=int, default=None,
                        help="Set global.n_iterations before first launch; "
                             "must match the frozen value on resume")
    parser.add_argument("--performance-source", default=None,
                        choices=["cost_model", "measured"],
                        help="Set system.performance_source before first launch. 'measured' = real "
                             "on-hardware timing via the static evaluator (needs GPUs "
                             "+ system.system_design_space incl. available_gpus); "
                             "'cost_model' (default) = RAGO cost-model prediction. "
                             "Orthogonal to --mode.")
    parser.add_argument(
        "--cm-calibration-mode",
        default=None,
        choices=["strict", "uncalibrated_spec"],
        help=("Set system.rag_cm.calibration_mode before first launch and persist the choice "
              "in the project's rag_stack_config.yaml. 'strict' requires exact "
              "calibration coverage; 'uncalibrated_spec' allows the CM to use "
              "physical hardware specifications where a component profile is "
              "absent."),
    )
    parser.add_argument("--mode", default="full", choices=["full", "hardware_replay"],
                        help="Run mode: 'full' = standard MOBO, 'hardware_replay' = use quality store")
    parser.add_argument("--quality-store", default=None,
                        help=("Path to a QualityStore. In full optimize mode it is an "
                              "exact-config read/write cache for quality+trace only; "
                              "it never seeds optimizer observations or candidates. "
                              "Required for hardware_replay unless --from-project is given."))
    parser.add_argument("--from-project", default=None,
                        help=("Path to a previous project directory whose GT evaluations "
                              "will be synthesized into a QualityStore at "
                              "<project-dir>/quality_store/ before hardware_replay runs."))
    parser.add_argument(
        "--n-refine-iterations",
        type=_nonnegative_int,
        default=None,
        help=("Cumulative refinement-iteration target after hardware replay "
              "(Phase 2b); a resumed transfer runs requested minus committed. "
              "Omitted means 0 for direct CLI runs; the system-transfer API "
              "passes the selected YAML default explicitly."),
    )
    parser.add_argument("--hardware-config", default=None,
                        help="Path to hardware JSON (must match the project's system block)")
    parser.add_argument(
        "--reuse-target-project",
        default=None,
        help=("Validated prior target system-transfer project whose replay "
              "artifacts/performance may be copied for source indices not "
              "listed by --retransfer-source-index. Hardware replay only."),
    )
    parser.add_argument(
        "--retransfer-source-index",
        action="append",
        type=_nonnegative_int,
        default=None,
        help=("Zero-based frozen source index to recompute with dynamic CM. "
              "Repeat once per index. Requires --reuse-target-project; the "
              "complete requested set must exactly match the active "
              "LLMLingua2 dependency set."),
    )


def _load_rag_stack_settings() -> dict:
    """Read ``.rag_stack_settings.json`` (``RAG_STACK_SETTINGS_PATH`` override or
    the repo root), returning ``{}`` if absent/unreadable. Mirrors the path
    logic in :mod:`rag_stack.server.settings_store` without importing the server
    package (keeps the CLI light)."""
    import json
    from pathlib import Path
    env = os.environ.get("RAG_STACK_SETTINGS_PATH")
    path = Path(env) if env else Path(__file__).resolve().parents[1] / ".rag_stack_settings.json"
    try:
        if path.is_file():
            with open(path) as f:
                return json.load(f) or {}
    except Exception:  # noqa: BLE001 — settings are best-effort, never fatal
        pass
    return {}


def _resolve_project_dir(args) -> str:
    """Resolve the optimize project directory. Precedence:

      1. ``--project-dir`` — used verbatim (the explicit / resume case).
      2. ``.rag_stack_settings.json`` → ``rag_optimization_project_dir`` — treated
         as a ROOT; a per-run subdir is created under it.
      3. default ROOT ``tmp_outputs/rag_stack_projects`` (under the repo root).

    For 2 & 3 the per-run subdir is ``<config-stem>_<timestamp>`` (or
    ``run_<timestamp>`` when ``--config`` is absent) so repeated runs never
    collide. Returns an absolute path."""
    if args.project_dir:
        return os.path.abspath(args.project_dir)
    from pathlib import Path
    from datetime import datetime
    repo_root = Path(__file__).resolve().parents[1]
    configured = str(
        _load_rag_stack_settings().get("rag_optimization_project_dir", "") or ""
    ).strip()
    if configured:
        root = Path(configured)
        if not root.is_absolute():
            root = repo_root / root
    else:
        root = repo_root / "tmp_outputs" / "rag_stack_projects"
    stem = Path(args.config).stem if getattr(args, "config", None) else "run"
    run_name = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return os.path.abspath(str(root / run_name))


def _apply_optimize_config_overrides(config_path: str, args) -> None:
    """Persist explicit optimize CLI policy overrides in the project YAML.

    The in-project config is the resume and provenance source of truth.  Keep
    ``None`` as "no override" so a resumed project retains its prior choice.
    Once launched, an explicit option must match the frozen configuration.
    Identical options do not reserialize YAML or change its retained text.
    """

    performance_source = getattr(args, "performance_source", None)
    calibration_mode = getattr(args, "cm_calibration_mode", None)
    n_iterations = getattr(args, "n_iterations", None)

    import yaml as _yaml

    project_dir = os.path.dirname(os.path.abspath(config_path))
    with project_config_lock(project_dir):
        config = _yaml.safe_load(verify_project_config(project_dir))
        overrides = [
            (("system", "performance_source"), performance_source, "cost_model"),
            (("system", "rag_cm", "calibration_mode"), calibration_mode, "strict"),
            (("global", "n_iterations"), n_iterations, 100),
        ]
        changed = False
        for keys, value, default in overrides:
            if value is None:
                continue
            section = config
            for key in keys[:-1]:
                section = section.setdefault(key, {})
                if not isinstance(section, dict):
                    raise ValueError(f"{'.'.join(keys[:-1])} must be a YAML mapping")
            if section.get(keys[-1], default) == value:
                continue
            section[keys[-1]] = value
            changed = True
        if changed:
            write_project_config(
                project_dir, _yaml.safe_dump(config, sort_keys=False),
            )


def _prepare_optimize_project_config(project_dir: str, args) -> str:
    """Resolve draft CLI options and freeze the canonical input in one lock."""
    config_path = os.path.join(project_dir, CONFIG_FILENAME)
    with project_config_lock(project_dir):
        if args.config:
            with open(args.config, encoding="utf-8", newline="") as stream:
                write_project_config(project_dir, stream.read())
        _apply_optimize_config_overrides(config_path, args)
        freeze_project_config(project_dir)
        # Preserve checkpoint ownership and counters on resume.
        if load_project_summary(project_dir) is None:
            save_project_summary(project_dir, create_project_summary())
    return config_path


_RAG_LLM_PREFLIGHT_TIMEOUT_S = 5.0


def _config_choice_values(value) -> list[str]:
    """Return scalar/list/``{ordered: [...]}`` config values as strings."""
    if isinstance(value, dict):
        if "ordered" in value:
            return _config_choice_values(value["ordered"])
        values: list[str] = []
        for nested in value.values():
            values.extend(_config_choice_values(nested))
        return values
    if isinstance(value, (list, tuple)):
        values: list[str] = []
        for nested in value:
            values.extend(_config_choice_values(nested))
        return values
    if value is None:
        return []
    return [str(value)]


def _required_vllm_endpoints(config: dict) -> dict[str, set[str]]:
    """Collect endpoint → model requirements for every vllm_api module.

    This walks the declarative search space rather than a sampled pipeline, so
    the preflight proves that every Qwen choice the ten refinement iterations
    may select is advertised before any source-store or corpus work starts.
    """
    requirements: dict[str, set[str]] = {}

    def _walk(value) -> None:
        if isinstance(value, list):
            for item in value:
                _walk(item)
            return
        if not isinstance(value, dict):
            return

        backend_values = {
            item.strip().lower()
            for key in ("component", "generator_backend")
            for item in _config_choice_values(value.get(key))
        }
        if "vllm_api" in backend_values:
            raw_uris = _config_choice_values(value.get("uri"))
            models = {
                model.strip()
                for model in _config_choice_values(value.get("model"))
                if model.strip()
            }
            if not raw_uris:
                raise ValueError("vllm_api module is missing its uri")
            if not models:
                raise ValueError("vllm_api module is missing its model")
            for raw_uri in raw_uris:
                if "${RAG_LLM_BASE_URL}" in raw_uri:
                    endpoint = str(
                        os.environ.get("RAG_LLM_BASE_URL") or ""
                    ).strip()
                    if not endpoint:
                        raise ValueError(
                            "RAG_LLM_BASE_URL is required for refinement but "
                            "is empty"
                        )
                    resolved = raw_uri.replace(
                        "${RAG_LLM_BASE_URL}", endpoint,
                    )
                else:
                    resolved = os.path.expandvars(raw_uri)
                resolved = resolved.strip().rstrip("/")
                if "$" in resolved or not resolved.startswith(
                    ("http://", "https://")
                ):
                    raise ValueError(
                        "vllm_api uri did not resolve to an HTTP endpoint: "
                        f"{raw_uri!r}"
                    )
                requirements.setdefault(resolved, set()).update(models)

        for nested in value.values():
            _walk(nested)

    _walk(config)
    return requirements


def _preflight_hardware_replay_llm_config(config: dict) -> None:
    """Fail fast unless every refinement vLLM endpoint is ready.

    This is compatibility validation for direct hardware replay configs that
    explicitly retain ``vllm_api``.  System-transfer targets are projected to
    local ``vllm`` chat before publication, so their requirement set is empty
    and this helper performs no HTTP requests.
    """
    import requests

    if not isinstance(config, dict):
        raise ValueError("RAG config must be a YAML mapping")

    requirements = _required_vllm_endpoints(config)
    for endpoint, required_models in sorted(requirements.items()):
        models_url = f"{endpoint}/v1/models"
        try:
            response = requests.get(
                models_url, timeout=_RAG_LLM_PREFLIGHT_TIMEOUT_S,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # requests + malformed JSON from the service
            raise ValueError(
                f"RAG LLM endpoint preflight failed for {models_url}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ValueError(
                f"RAG LLM endpoint {models_url} returned no data model list"
            )
        advertised = {
            str(entry.get("id"))
            for entry in data
            if isinstance(entry, dict) and entry.get("id")
        }
        missing = sorted(required_models - advertised)
        if missing:
            raise ValueError(
                f"RAG LLM endpoint {endpoint} is missing required model(s): "
                f"{missing}; advertised={sorted(advertised)}"
            )
        logger.info(
            "RAG LLM preflight OK: %s advertises %d/%d required model(s)",
            endpoint, len(required_models), len(required_models),
        )


def _preflight_hardware_replay_llm(config_path: str) -> None:
    """Load a persisted config and apply the refinement endpoint preflight.

    The check intentionally happens before ``QualityStore.from_project`` and
    ``Controller`` construction: both can process gigabytes before the first
    quality candidate would otherwise discover a missing/dead endpoint.
    """
    import yaml as _yaml

    with open(config_path) as stream:
        config = _yaml.safe_load(stream) or {}
    _preflight_hardware_replay_llm_config(config)


def cmd_optimize(args):
    """Run RAG optimization (full MOBO or hardware replay)."""
    _setup_logging()
    _install_subproc_reaper()

    project_dir = os.path.abspath(args.project_dir)
    os.makedirs(project_dir, exist_ok=True)
    try:
        config_path = _prepare_optimize_project_config(project_dir, args)
    except (OSError, ValueError) as exc:
        print(f"Error: cannot start project: {exc}", file=sys.stderr)
        sys.exit(1)
    _attach_command_log(project_dir, "optimize")

    hardware_replay_refine_iterations = 0
    reuse_target_project = getattr(args, "reuse_target_project", None)
    retransfer_source_indices = getattr(
        args, "retransfer_source_index", None,
    )
    selective_replay_requested = (
        reuse_target_project is not None
        or retransfer_source_indices is not None
    )
    if selective_replay_requested and args.mode != "hardware_replay":
        print(
            "Error: --reuse-target-project/--retransfer-source-index are "
            "valid only in hardware_replay mode",
            file=sys.stderr,
        )
        _mark_project_failed(project_dir)
        sys.exit(1)
    if (reuse_target_project is None) != (retransfer_source_indices is None):
        print(
            "Error: --reuse-target-project and at least one "
            "--retransfer-source-index must be provided together",
            file=sys.stderr,
        )
        _mark_project_failed(project_dir)
        sys.exit(1)
    if retransfer_source_indices is not None:
        if not retransfer_source_indices:
            print(
                "Error: selective replay requires at least one "
                "--retransfer-source-index",
                file=sys.stderr,
            )
            _mark_project_failed(project_dir)
            sys.exit(1)
        if len(set(retransfer_source_indices)) != len(retransfer_source_indices):
            print(
                "Error: --retransfer-source-index values must be unique",
                file=sys.stderr,
            )
            _mark_project_failed(project_dir)
            sys.exit(1)
        retransfer_source_indices = sorted(retransfer_source_indices)
        reuse_target_project = os.path.realpath(reuse_target_project)
    if args.mode == "hardware_replay":
        if not args.quality_store and not args.from_project:
            print(
                "Error: --quality-store or --from-project is required for "
                "hardware_replay mode",
                file=sys.stderr,
            )
            _mark_project_failed(project_dir)
            sys.exit(1)
        hardware_replay_refine_iterations = (
            args.n_refine_iterations
            if args.n_refine_iterations is not None
            else 0
        )
        if hardware_replay_refine_iterations > 0:
            try:
                _preflight_hardware_replay_llm(config_path)
            except Exception as exc:
                print(
                    f"Error: hardware-replay refinement preflight failed: {exc}",
                    file=sys.stderr,
                )
                _mark_project_failed(project_dir)
                sys.exit(1)

    # If --from-project is given, synthesize a QualityStore inside the new
    # project before running hardware_replay against it. The store lives at
    # <project_dir>/quality_store/ so it's discoverable alongside the run.
    if args.mode == "hardware_replay" and args.from_project:
        prev = os.path.abspath(args.from_project)
        if not os.path.isdir(prev):
            print(f"Error: --from-project path not found: {prev}", file=sys.stderr)
            sys.exit(1)
        store_path = os.path.join(project_dir, "quality_store")
        try:
            store_path, reused = _quality_store_for_project_resume(
                source_project=prev, store_path=store_path,
            )
        except Exception as e:
            print(f"Error: failed to prepare QualityStore from {prev}: {e}",
                  file=sys.stderr)
            sys.exit(1)
        args.quality_store = store_path
        logger.info(
            "%s QualityStore from %s → %s",
            "Reused strict" if reused else "Synthesized", prev, store_path,
        )

    # Resolve QA/corpus data paths. Explicit CLI flags win; otherwise fall back
    # to the config's top-level `data:` section (data.qa / data.corpus) so a
    # config can be fully self-contained — no need to pass --qa-data/--corpus-data.
    # On resume the Controller ignores both and uses the in-project data/ copy.
    qa_data_path = args.qa_data
    corpus_data_path = args.corpus_data
    if qa_data_path is None or corpus_data_path is None:
        import yaml as _yaml
        with open(config_path) as _f:
            _cfg_data = (_yaml.safe_load(_f) or {}).get("dataset") or {}
        if qa_data_path is None:
            qa_data_path = _cfg_data.get("qa")
        if corpus_data_path is None:
            corpus_data_path = _cfg_data.get("corpus")

    logger.info(f"Starting RAG optimization in {project_dir} (mode={args.mode})")
    logger.info(f"Config: {config_path}")
    logger.info(f"QA data: {qa_data_path}")
    logger.info(f"Corpus data: {corpus_data_path}")

    try:
        from rag_stack.controller import Controller

        controller = Controller(
            config_path=config_path,
            qa_data_path=qa_data_path,
            corpus_data_path=corpus_data_path,
            project_dir=project_dir,
            quality_store_path=(
                args.quality_store if args.mode == "full" else None
            ),
            defer_checkpoint_warm_start=(args.mode == "hardware_replay"),
        )

        # Mark as running
        my_pid = os.getpid()
        summary = load_project_summary(project_dir)
        if summary is None:
            summary = create_project_summary()
        summary.status = ProjectStatus.RUNNING
        summary.pid = my_pid
        summary.timestamp = datetime.datetime.now().isoformat()
        save_project_summary(project_dir, summary)
        logger.info(f"PID {my_pid} registered in summary.json")

        if args.mode == "hardware_replay":
            # Project configs remain strict six-section RAG configs. The server
            # resolves transfer-only YAML metadata and passes it explicitly;
            # direct CLI replay without the flag performs Phase 2a only.
            n_refine_iterations = hardware_replay_refine_iterations
            hw_config = None
            if args.hardware_config:
                with open(args.hardware_config) as f:
                    hw_config = json.load(f)
                logger.info(f"Hardware config override: {args.hardware_config}")

            logger.info(f"Quality store: {args.quality_store}")
            logger.info(f"Refine iterations: {n_refine_iterations}")
            if reuse_target_project is not None:
                logger.info(
                    "Selective replay source: %s; dynamic CM indices: %s",
                    os.path.realpath(reuse_target_project),
                    sorted(retransfer_source_indices),
                )

            result = controller.get_pareto_for_hardware(
                hardware_config=hw_config or controller.hardware.to_dict(),
                quality_store_path=args.quality_store,
                n_refine_iterations=n_refine_iterations,
                reuse_target_project=reuse_target_project,
                retransfer_source_indices=retransfer_source_indices,
            )
        else:
            result = controller.get_pareto(n_iterations=args.n_iterations)

        logger.info(
            f"Optimization complete: "
            f"{result.get('gt_count', 0)} GT evals, "
            f"{result.get('cm_count', 0)} CM evals, "
            f"{len(result.get('pareto_configs', []))} Pareto points"
        )

    except (KeyboardInterrupt, SystemExit):
        # The signal reaper exits via SystemExit (a BaseException), so the
        # generic Exception handler below never saw SIGTERM/SIGINT and left a
        # dead run advertised as "running".  Persist terminal state, then keep
        # the original signal-derived exit code/semantics.
        _mark_project_failed(project_dir)
        raise
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        traceback.print_exc()
        _mark_project_failed(project_dir)
        sys.exit(1)


# ── quality-sweep ─────────────────────────────────────────────────────────

def _add_quality_sweep_args(parser: argparse.ArgumentParser):
    parser.add_argument("--config", required=True, help="Path to RAG config YAML")
    parser.add_argument("--qa-data", required=True, help="Path to QA parquet file")
    parser.add_argument("--corpus-data", required=True, help="Path to corpus parquet file")
    parser.add_argument("--store-path", required=True,
                        help="Output directory for quality store")
    parser.add_argument("--n-grid", type=int, default=5,
                        help="Grid points per continuous parameter (exhaustive)")
    parser.add_argument("--n-configs", type=int, default=0,
                        help="Number of configs to sample (sobol)")
    parser.add_argument("--strategy", default="exhaustive",
                        choices=["exhaustive", "sobol"],
                        help="Config generation strategy")


def cmd_quality_sweep(args):
    """Run quality store sweep."""
    _setup_logging()

    store_path = os.path.abspath(args.store_path)
    logger.info(f"Starting quality sweep -> {store_path}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Strategy: {args.strategy}, n_grid: {args.n_grid}")

    store = None
    try:
        from rag_stack.quality_store import QualityStore

        store = QualityStore(
            config_path=args.config,
            qa_data_path=args.qa_data,
            corpus_data_path=args.corpus_data,
            store_path=store_path,
        )

        store.update_status(
            "running",
            pid=os.getpid(),
            strategy=args.strategy,
            n_grid=args.n_grid,
            n_configs_requested=args.n_configs,
        )

        result = store.sweep(
            n_grid=args.n_grid,
            n_configs=args.n_configs,
            strategy=args.strategy,
        )

        store.update_status(
            "completed",
            pid=None,
            strategy=args.strategy,
            n_grid=args.n_grid,
            n_configs_requested=args.n_configs,
        )

        logger.info(
            f"Quality sweep complete: "
            f"{result['n_configs']} configs evaluated, "
            f"{result['n_failed']} failed"
        )

    except Exception as e:
        logger.error(f"Quality sweep failed: {e}")
        traceback.print_exc()
        try:
            if store is not None:
                store.update_status("failed", pid=None)
        except Exception:
            pass
        sys.exit(1)


# ── Main entry point ──────────────────────────────────────────────────────

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "replay":
        from rag_stack.replay_cli import main as _replay_main

        _install_subproc_reaper()
        raise SystemExit(_replay_main(sys.argv[2:]))

    # Export conversion is a file/schema-only operation. Dispatch it before
    # importing optimizer benchmark modules so it cannot pull in torch/CUDA.
    if len(sys.argv) >= 2 and sys.argv[1] == "convert-optimizer-export":
        from rag_stack.optimizer_export import main as _convert_export_main

        _convert_export_main(sys.argv[2:])
        return

    # Calibration owns a self-contained parser. Dispatch it before importing
    # optimizer benchmark modules: ``calibrate status/plan`` must work in a
    # lightweight deployment environment and must not import Ax/torch/CUDA.
    if len(sys.argv) >= 2 and sys.argv[1] == "calibrate":
        from rag_stack.calibration_cli import main as _calibrate_main

        _calibrate_main(sys.argv[2:] or ["--help"])
        return

    # A one-point measured replay has its own small parser and then loads the
    # GPU runtime lazily. Dispatch it before optimizer benchmark imports so the
    # command needs only the two saved JSON files the user is measuring.
    if len(sys.argv) >= 2 and sys.argv[1] == "eval-one":
        from rag_stack.eval_one_cli import main as _eval_one_main

        _install_subproc_reaper()
        raise SystemExit(_eval_one_main(sys.argv[2:]))

    # Lazy import: the benchmark module pulls in heavy ML deps (torch,
    # botorch, ax, smac), so only load them when a user actually runs a
    # benchmark-related subcommand.
    from rag_stack.optimizer.benchmark.cli import (
        add_args as _add_benchmark_args,
        cmd_benchmark,
    )
    from rag_stack.optimizer.benchmark.build_pareto_cache import (
        add_args as _add_pareto_cache_args,
        cmd_build_pareto_cache,
    )
    from rag_stack.optimizer.benchmark.recompute_metrics_from_evals import (
        add_args as _add_recompute_args,
        cmd_recompute_metrics,
    )

    parser = argparse.ArgumentParser(
        prog="rag_stack",
        description="RAG-Stack CLI",
    )
    sub = parser.add_subparsers(dest="command")

    # optimize
    p_opt = sub.add_parser(
        "optimize",
        help="Run RAG optimization (full MOBO or hardware replay)",
    )
    _add_optimize_args(p_opt)

    # quality-sweep
    p_qs = sub.add_parser(
        "quality-sweep",
        help="Run quality store sweep (exhaustive or sobol)",
    )
    _add_quality_sweep_args(p_qs)

    # Kept in the top-level command list for discoverability. Invocations are
    # dispatched above through the lightweight one-point parser.
    sub.add_parser(
        "eval-one",
        add_help=False,
        help="Measure one saved pipeline_config + system_config deployment",
    )
    sub.add_parser(
        "replay", add_help=False,
        help="Replay an algo_config + system_config pair for quality, measured performance or CM",
    )

    # Kept in the top-level command list for discoverability. Invocations are
    # dispatched above through the converter's lightweight parser.
    sub.add_parser(
        "convert-optimizer-export",
        add_help=False,
        help="Convert an optimizer export seed into a standard project",
    )

    # benchmark
    p_bench = sub.add_parser(
        "benchmark",
        help="Run optimizer benchmark sweeps (YAML-driven matrix or single-run)",
    )
    _add_benchmark_args(p_bench)

    # pareto-cache
    p_pareto = sub.add_parser(
        "pareto-cache",
        help="Build / refresh frozen reference Pareto for non-analytical benchmarks",
    )
    _add_pareto_cache_args(p_pareto)

    # recompute-metrics
    p_recompute = sub.add_parser(
        "recompute-metrics",
        help="Backfill hv_diff / igd in snapshot CSVs against the current cache",
    )
    _add_recompute_args(p_recompute)

    # calibrate — unified cost-model calibration (measure-once / fit-on-CM).
    # Self-contained arg parsing (its own verbs/components); pass the rest through
    # verbatim. Heavy deps (faiss/scipy) load lazily inside it.
    p_cal = sub.add_parser(
        "calibrate",
        add_help=False,
        help="Calibrate the cost model: calibrate {measure|fit} {ivf|hnsw|...}",
    )
    p_cal.add_argument(
        "-h", "--help",
        action="store_true",
        dest="calibrate_help",
        help="show calibrate help and exit",
    )
    p_cal.add_argument("calibrate_args", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Resolve a default optimize project dir when --project-dir is omitted
    # (.rag_stack_settings.json → tmp_outputs/rag_stack_projects), and write it
    # back onto args so BOTH the per-project log attach below AND cmd_optimize
    # use the same concrete path (resolve once → one timestamped dir).
    if args.command == "optimize" and not getattr(args, "project_dir", None):
        args.project_dir = _resolve_project_dir(args)
        print(f"[rag_stack] --project-dir not given; using {args.project_dir}")

    # Persistent per-project logging: every command that has an output directory
    # (optimize: --project-dir, quality-sweep: --store-path, benchmark/pareto-cache/
    # recompute: --output-dir) gets a timestamped log file under <dir>/logs/ that
    # captures the whole run's console output. Durable and reusable across runs.
    _out = next(
        (getattr(args, a) for a in ("project_dir", "store_path", "output_dir")
         if getattr(args, a, None)),
        None,
    )
    if _out and args.command != "optimize":
        _attach_command_log(_out, args.command)

    if args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "quality-sweep":
        cmd_quality_sweep(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    elif args.command == "pareto-cache":
        cmd_build_pareto_cache(args)
    elif args.command == "recompute-metrics":
        cmd_recompute_metrics(args)
    elif args.command == "calibrate":
        from rag_stack.calibration_cli import main as _calibrate_main
        _calibrate_main(["--help"] if getattr(args, "calibrate_help", False) else args.calibrate_args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
