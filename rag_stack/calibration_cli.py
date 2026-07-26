"""Unified, staged ``rag_stack calibrate`` dispatcher.

``status``/``plan`` are metadata-only and intentionally do not import the cost
model package.  Real component implementations are imported only after a
measure or fit command is selected.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from rag_stack.calibration_paths import (
    configure_calibration_roots,
    resolve_calibration_roots,
)


VERBS = ("measure", "fit", "all-faiss", "all", "workload")
COMPONENTS = ("ivf", "hnsw")
PASSTHROUGH = (
    "llm", "hnsw-workload", "generator", "generator-tp",
    "generator-disagg", "retrieval-worker", "llmlingua2",
)
META_COMMANDS = ("status", "plan")


def _extract_root_options(argv: list[str]) -> tuple[list[str], str | None, str | None]:
    """Remove global staging-root flags wherever the user placed them."""
    remaining: list[str] = []
    roots: dict[str, str | None] = {"data": None, "profile": None}
    index = 0
    while index < len(argv):
        token = argv[index]
        matched = False
        for flag, key in (("--data-dir", "data"), ("--profile-dir", "profile")):
            if token == flag:
                if index + 1 >= len(argv):
                    raise SystemExit(f"{flag} requires a path")
                roots[key] = argv[index + 1]
                index += 2
                matched = True
                break
            if token.startswith(flag + "="):
                roots[key] = token.split("=", 1)[1]
                index += 1
                matched = True
                break
        if not matched:
            remaining.append(token)
            index += 1
    return remaining, roots["data"], roots["profile"]


def _absolutize_input_paths(argv: list[str], relative_to: Path) -> list[str]:
    """Resolve file inputs before component CLIs change to the repo root."""
    flags = {
        "--config", "--qa-data", "--corpus-data", "--telemetry",
        "--component-store", "--stage-store", "--output-dir",
        "--candidate", "--candidate-output",
        "--pipeline-config", "--hardware-config", "--project-dir", "--output",
    }

    def absolute(value: str) -> str:
        path = Path(value).expanduser()
        return str((relative_to / path).resolve() if not path.is_absolute() else path.resolve())

    result = list(argv)
    index = 0
    while index < len(result):
        token = result[index]
        if token in flags and index + 1 < len(result):
            result[index + 1] = absolute(result[index + 1])
            index += 2
            continue
        matched = next((flag for flag in flags if token.startswith(flag + "=")), None)
        if matched:
            result[index] = matched + "=" + absolute(token.split("=", 1)[1])
        if token == "--cohort" and index + 1 < len(result):
            fields = result[index + 1].split(",")
            if len(fields) == 5:
                fields[2] = absolute(fields[2])
                result[index + 1] = ",".join(fields)
            index += 2
            continue
        if token.startswith("--cohort="):
            fields = token.split("=", 1)[1].split(",")
            if len(fields) == 5:
                fields[2] = absolute(fields[2])
                result[index] = "--cohort=" + ",".join(fields)
        index += 1
    return result


def _route_passthrough(head: str, rest: list[str]) -> bool:
    if head == "llm":
        from rag_stack.cost_model.llm_sim.cli import main as llm_main
        llm_main(
            rest
            if rest and rest[0] in {"benchmark", "calibrate", "refit"}
            else ["calibrate"] + rest
        )
        return True
    if head == "hnsw-workload":
        from rag_stack.cost_model.faiss_hnsw_sim.train import main as train_main
        train_main(rest)
        return True
    if head == "generator":
        from rag_stack.cost_model.llm_sim.calibration.main_llm.generator import (
            main as generator_main,
        )
        generator_main(rest)
        return True
    if head == "generator-tp":
        from rag_stack.cost_model.llm_sim.calibration.main_llm.generator_parallel import (
            main as generator_parallel_main,
        )
        generator_parallel_main(rest)
        return True
    if head == "generator-disagg":
        from rag_stack.cost_model.llm_sim.calibration.main_llm.generator_disagg import (
            main as generator_disagg_main,
        )
        generator_disagg_main(rest)
        return True
    if head == "retrieval-worker":
        from rag_stack.retrieval_worker_calibrator import main as retrieval_worker_main
        retrieval_worker_main(rest)
        return True
    if head == "llmlingua2":
        parser = argparse.ArgumentParser(prog="rag_stack calibrate llmlingua2")
        parser.add_argument("--config", required=True)
        parser.add_argument("--device", default="cuda:0")
        parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
        parser.add_argument("--seq-lens", default="64,128,256,448")
        parser.add_argument("--warmup", type=int, default=3)
        parser.add_argument("--measurements", type=int, default=10)
        args = parser.parse_args(rest)
        from rag_stack.cost_model.llm_sim.calibration import hardware_from_config
        from rag_stack.cost_model.llm_sim.compressor_llmlingua2_sim import (
            calibrate_llmlingua2_linear,
        )
        hardware, _ = hardware_from_config(args.config)
        calibrate_llmlingua2_linear(
            hardware,
            device=args.device,
            batch_sizes=tuple(int(value) for value in args.batch_sizes.split(",") if value),
            seq_lens=tuple(int(value) for value in args.seq_lens.split(",") if value),
            warmup=args.warmup,
            measurements=args.measurements,
        )
        return True
    return False


def _make(component: str, system: str, **kwargs):
    if component == "ivf":
        from rag_stack.cost_model.faiss_ivf_sim.hardware_calibrator import IVFHardwareCalibrator
        return IVFHardwareCalibrator(system=system, **kwargs)
    if component == "hnsw":
        from rag_stack.cost_model.faiss_hnsw_sim.hardware_calibrator import HNSWHardwareCalibrator
        return HNSWHardwareCalibrator(system=system, **kwargs)
    raise SystemExit(f"unknown calibrate component {component!r} (have {COMPONENTS})")


def _register_config_hardware(component: str, config: str) -> str:
    from rag_stack.cost_model.llm_sim.calibration import hardware_from_config, cpu_hardware_key

    hardware, system = hardware_from_config(config)
    key = cpu_hardware_key(system)
    if not key:
        raise SystemExit(
            "[calibrate] config.system.cpu must be a preset string or a dict "
            "with a `hardware_key`"
        )
    if isinstance(system.get("cpu"), dict):
        from rag_stack.cost_model.faiss_system_registry import register_configured_faiss_preset
        register_configured_faiss_preset(component, key, hardware)
    return key


def _run(verb: str, component: str, args) -> None:
    if component not in COMPONENTS:
        raise SystemExit(f"unknown framework component {component!r}; choose {COMPONENTS}")
    if not args.config:
        raise SystemExit(
            f"`calibrate {component}` needs --config <hardware config>"
        )
    kwargs = {"maxiter": args.maxiter} if args.maxiter is not None else {}
    if component == "hnsw":
        hnsw_parameters = {
            "simd_efficiency": args.hnsw_simd_efficiency,
            "dram_random_access_latency_ns": (
                args.hnsw_dram_random_access_latency_ns
            ),
            "scattered_bw_ceiling": args.hnsw_scattered_bw_ceiling,
            "numa_local_bytes": args.hnsw_numa_local_bytes,
            "numa_remote_bw_ceiling": args.hnsw_numa_remote_bw_ceiling,
        }
        supplied = {
            name: value for name, value in hnsw_parameters.items()
            if value is not None
        }
        if supplied and len(supplied) != len(hnsw_parameters):
            missing = sorted(set(hnsw_parameters) - set(supplied))
            raise SystemExit(
                "custom HNSW calibration parameters are all-or-nothing; "
                f"missing {missing}"
            )
        if supplied:
            kwargs["hardware_model_parameters"] = supplied
    calibrator = _make(
        component, _register_config_hardware(component, args.config), **kwargs
    )
    if verb == "measure":
        calibrator.measure(force=args.force)
    elif verb == "fit":
        calibrator.fit()
    else:
        calibrator.measure(force=args.force)
        calibrator.fit()


def _run_workload(args) -> None:
    if not args.config:
        raise SystemExit("`calibrate workload` needs a full RAG --config")
    if not args.project_dir:
        raise SystemExit(
            "`calibrate workload` needs --project-dir <cache project>"
        )
    from rag_stack.cost_model.workload_profile import build_workload_profiles

    summary = build_workload_profiles(
        config_path=args.config,
        project_dir=args.project_dir,
        qa_data_path=args.qa_data,
        corpus_data_path=args.corpus_data,
    )
    print(
        "[calibrate workload] build-on-miss attempt completed; rerun "
        "pipeline status to verify every artifact: "
        f"corpus={summary['corpus_key']} "
        f"embeddings={','.join(summary['embedding_models'])} "
        f"project={summary['project_dir']}",
        flush=True,
    )


def _status_parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"rag_stack calibrate {command}")
    parser.add_argument("--config", required=True, help="system-only or full RAG hardware config")
    parser.add_argument(
        "--scope", default="core",
        help=(
            "comma-separated pipeline,ivf,hnsw,generator,generator-tp,"
            "retrieval-worker; pipeline expands a full RAG config; default: core"
        ),
    )
    parser.add_argument(
        "--models", default=None,
        help="comma-separated generator model IDs; default: canonical Qwen2.5 family",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def _run_status(
    command: str,
    argv: list[str],
    *,
    data_dir: str | None,
    profile_dir: str | None,
) -> None:
    args = _status_parser(command).parse_args(argv)
    scopes = tuple(value.strip() for value in args.scope.split(",") if value.strip())
    models = (
        tuple(value.strip() for value in args.models.split(",") if value.strip())
        if args.models else None
    )
    previous_logging_disable = logging.root.manager.disable
    # Metadata commands should not inherit noisy import-time INFO messages.
    # JSON additionally suppresses warnings/errors so stdout is exactly one
    # machine-readable document; raised exceptions still carry failures.
    metadata_disable = logging.CRITICAL if args.json else logging.INFO
    logging.disable(max(previous_logging_disable, metadata_disable))
    try:
        from rag_stack.calibration_inventory import build_calibration_inventory

        inventory = build_calibration_inventory(
            args.config,
            data_dir=data_dir,
            profile_dir=profile_dir,
            scopes=scopes,
            generator_models=models,
        )
    finally:
        logging.disable(previous_logging_disable)
    if args.json:
        print(json.dumps(inventory.to_dict(), indent=2, sort_keys=True))
        if command == "status" and not inventory.deploy_ready:
            raise SystemExit(2)
        return
    elif command == "status":
        verdict = "READY" if inventory.deploy_ready else "INCOMPLETE"
        gpu_count = inventory.gpu_count if inventory.gpu_count is not None else "?"
        print(
            f"Calibration SELECTED SCOPES {verdict}: cpu={inventory.cpu_key or '-'} "
            f"gpu={inventory.gpu_key or '-'} x{gpu_count}"
        )
        print(f"  frozen data: {inventory.data_root}")
        print(f"  profiles:    {inventory.profile_root}")
        for item in inventory.items:
            print(f"  [{item.state.upper():14s}] {item.name}: {item.detail}")
            print(f"                     profile: {item.profile_path}")
            if item.frozen_path:
                print(f"                     frozen:  {item.frozen_path}")
    else:
        print("Calibration plan:")
        for item in inventory.items:
            if item.required_for_deploy and item.state != "ready":
                print(f"  [{item.state.upper():14s}] {item.name}: {item.detail}")
    if inventory.pending_commands:
        print("\nNext commands (in dependency order):")
        for index, pending in enumerate(inventory.pending_commands, 1):
            print(f"  {index}. {pending}")
    else:
        print("\nNo executable calibration commands are pending.")
    if command == "status" and not inventory.deploy_ready:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_stack calibrate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Inspect, stage, measure, and fit RAG cost-model profiles.",
        epilog="""\
Safe first step on every machine:
  rag_stack calibrate status --config <hardware.yaml> \\
      --data-dir <staging>/calibration_data \\
      --profile-dir <staging>/calibration_profiles

Commands:
  status | plan                      read-only readiness / next commands (no CUDA import)
  measure ivf|hnsw                   real CPU hardware -> frozen parquet
  fit ivf|hnsw                       frozen parquet -> system profile (CPU-only)
  ivf | hnsw                         measure-if-needed, then fit
  all-faiss                          IVF + HNSW only (legacy alias: all)
  generator                          generator component roofline calibration
  generator-tp                       collocated hardware×stage×batch campaign
  generator-disagg                   direct P/D prefill+decode stage campaign
  llm                                encoder/reranker component measure + fit
  llmlingua2                         exact GPU+CPU compressor measure + fit
  retrieval-worker                   standalone CPU worker operators (RRF)
  workload                           build corpus×embedding profiles without optimizer startup
Global staging flags ``--data-dir`` and ``--profile-dir`` may appear anywhere.
Defaults are calibration_data/ and calibration_profiles/.
""",
    )
    parser.add_argument("target", help="status/plan, verb, or component")
    parser.add_argument("component", nargs="?", help="component after measure/fit")
    parser.add_argument("--config", default=None)
    parser.add_argument("--qa-data", default=None)
    parser.add_argument("--corpus-data", default=None)
    parser.add_argument("--project-dir", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--hnsw-simd-efficiency", type=float, default=None)
    parser.add_argument(
        "--hnsw-dram-random-access-latency-ns", type=float, default=None,
    )
    parser.add_argument("--hnsw-scattered-bw-ceiling", type=float, default=None)
    parser.add_argument("--hnsw-numa-local-bytes", type=int, default=None)
    parser.add_argument("--hnsw-numa-remote-bw-ceiling", type=float, default=None)
    return parser


def _dispatch(argv: list[str], data_dir: str | None, profile_dir: str | None) -> None:
    if argv and argv[0] in META_COMMANDS:
        _run_status(
            argv[0], argv[1:], data_dir=data_dir, profile_dir=profile_dir
        )
        return
    if argv and argv[0] in PASSTHROUGH:
        _route_passthrough(argv[0], argv[1:])
        return

    args = build_parser().parse_args(argv)
    if args.target in VERBS:
        verb = args.target
        if verb == "workload":
            _run_workload(args)
            return
        if verb in ("all", "all-faiss"):
            if verb == "all":
                print("[calibrate] `all` is a legacy alias for `all-faiss`; it does not calibrate GPU components.")
            for component in COMPONENTS:
                print(f"\n########## calibrate {component} ##########", flush=True)
                _run("auto", component, args)
            return
        if not args.component:
            raise SystemExit(f"`calibrate {verb}` needs a component {COMPONENTS}")
        _run(verb, args.component, args)
        return
    _run("auto", args.target, args)


def main(argv=None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    raw, data_dir, profile_dir = _extract_root_options(raw)
    caller_cwd = Path.cwd()
    raw = _absolutize_input_paths(raw, caller_cwd)
    meta_only = bool(raw and raw[0] in META_COMMANDS)
    resolver = resolve_calibration_roots if meta_only else configure_calibration_roots
    resolved_data, resolved_profiles = resolver(
        data_dir=data_dir, profile_dir=profile_dir, relative_to=caller_cwd
    )
    # Pass absolute roots to metadata commands even when only environment
    # defaults supplied; generated plans are unambiguous and reproducible.
    data_dir, profile_dir = str(resolved_data), str(resolved_profiles)
    if meta_only:
        _dispatch(raw, data_dir, profile_dir)
        return
    repo_root = Path(__file__).resolve().parents[1]
    try:
        os.chdir(repo_root)
        _dispatch(raw, data_dir, profile_dir)
    finally:
        os.chdir(caller_cwd)


if __name__ == "__main__":
    main()
