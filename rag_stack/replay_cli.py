"""Replay one resolved algorithm/deployment pair without archived run artifacts.

Quality uses the public evaluator interface. Measured performance uses the
existing local deployment adapter. CM obtains a fresh canonical evaluator trace
and prices the supplied deployment. Temporary runtime files are not publications.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
from numbers import Integral, Real
from pathlib import Path
import shlex
import tempfile
from typing import Sequence


MODES = ("quality", "measured", "cm")
_SEARCH_SETTINGS = frozenset({
    "algo_search_space", "optimizer", "system_design_space", "measured_search_space",
    "deployment_presets", "cm_search_space", "batch_presets", "system_space_owner",
})


def validate_resolved_algorithm(algo: dict) -> None:
    """Shared replay/preview boundary for one dataset-independent algorithm."""
    if not isinstance(algo, dict):
        raise ValueError("algo_config must be a JSON object")
    lines = algo.get("node_lines")
    if not isinstance(lines, list) or not lines:
        raise ValueError("algo_config must contain resolved node_lines, not a search space")
    for line in lines:
        if not isinstance(line, dict) or not isinstance(line.get("nodes"), list):
            raise ValueError("Each resolved node line requires nodes")
        for node in line["nodes"]:
            modules = node.get("modules") if isinstance(node, dict) else None
            if not isinstance(modules, list) or len(modules) != 1:
                raise ValueError("Each resolved node must select exactly one module")
            if not isinstance(modules[0], dict) or not isinstance(modules[0].get("component"), str):
                raise ValueError("Each resolved module requires a component name")
    if "dataset" in algo:
        raise ValueError("algo_config must not bind a dataset; supply --qa and --corpus")
    if _SEARCH_SETTINGS.intersection(algo):
        raise ValueError("algo_config must contain a resolved pipeline, not optimizer settings")
    if "system" in algo:
        raise ValueError("Deployment/hardware settings belong in system_config, not algo_config.system")


def validate_resolved_system(system: dict) -> None:
    """Shared replay/preview boundary for one concrete external deployment."""
    if not isinstance(system, dict):
        raise ValueError("system_config must be a JSON object")
    if "dataset" in system:
        raise ValueError("system_config must not bind a dataset; supply --qa and --corpus")
    if _SEARCH_SETTINGS.intersection(system):
        raise ValueError("system_config must select a concrete deployment, without search spaces")
    if not isinstance(system.get("layout"), dict) or not system["layout"]:
        raise ValueError("system_config must contain a concrete deployment layout")


def load_replay_inputs(algo_path: Path, system_path: Path) -> tuple[dict, dict]:
    """Read resolved algorithm/deployment inputs independently of any dataset."""
    from rag_stack.eval_one_cli import _read_json_object

    algo, _ = _read_json_object(algo_path, label="algorithm config")
    system, _ = _read_json_object(system_path, label="system config")
    validate_resolved_algorithm(algo)
    validate_resolved_system(system)
    return algo, system


def _dataset_path(path: Path) -> Path:
    """Resolve caller-supplied paths from the working directory, never a config."""
    resolved = Path(os.path.expandvars(str(path))).expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"Dataset file does not exist: {resolved}")
    if resolved.suffix != ".parquet":
        raise ValueError(f"Dataset input must be a Parquet file: {resolved}")
    return resolved


def _runtime(algo: dict, project: Path, *, qa_path: Path, corpus_path: Path,
             dataset_name: str | None = None):
    from rag_stack.dataset_manager import DatasetManager
    from rag_stack.evaluation import create_quality_evaluator
    from rag_stack.evaluator_contract import configured_backend_options, dataset_owner_config_from_resolved

    owner_config = dataset_owner_config_from_resolved(algo)
    if dataset_name:
        owner_config["dataset"] = {"dataset_name": dataset_name}
    manager = DatasetManager(
        project_dir=str(project), config=owner_config,
        qa_data_path=str(qa_path), corpus_data_path=str(corpus_path),
    )
    manager.inject_corpus_size()
    manager.inject_token_stats()
    algo["_token_stats"] = deepcopy(owner_config["_token_stats"])
    algo["vectordb"] = deepcopy(owner_config["algo_search_space"]["vectordb"])
    backend = (algo.get("global") or {}).get("eval_backend", "static_gt")
    evaluator = create_quality_evaluator(
        dataset_manager=manager, project_dir=str(project), backend=backend,
        backend_options=configured_backend_options(algo),
    )
    return manager, evaluator, backend


def _price(algo: dict, system: dict, trace: dict, *, dataset_name: str | None = None) -> dict:
    from rag_stack.cost_model.assembly import RAGCMAssembly
    from rag_stack.system_layout import cm_fixed_pipeline_config
    from rag_stack.types import SystemConfig

    # CM's existing data-profile adapters use this runtime corpus identity.
    # It is supplied by the invocation, never recovered from either JSON.
    priced_algo = deepcopy(algo)
    if dataset_name:
        priced_algo["dataset"] = {"dataset_name": dataset_name}
    hardware = SystemConfig.from_dict(cm_fixed_pipeline_config(priced_algo, system)["system"])
    result = RAGCMAssembly(hardware).evaluate_fixed(priced_algo, system, trace=trace)
    return {"status": "ok", "performance_score": float(result.performance_score),
            "best_hw_params": result.best_hw_params}


def _measure(algo: dict, system: dict, evaluator, project: Path) -> dict:
    from rag_stack.eval_one_cli import _available_gpus, _publication_rejections
    from rag_stack_evaluator.static_rag_evaluator.measured.provider import MeasuredProvider

    selection = ((algo.get("system") or {}).get("performance_objectives") or {}).get(
        "selection", "max_throughput",
    )
    with MeasuredProvider(evaluator, _available_gpus(system), n_queries=None,
                          selection=selection) as provider:
        measured = provider.evaluate(algo, system, run_dir=str(project / "measured"),
                                     require_admissible=False)
    performance = dict(measured.raw_performance)
    reasons = _publication_rejections(
        measured.performance_score, performance, measured.performance_execution_trace,
    )
    result = {"status": "inadmissible" if reasons else "ok",
              "performance_score": float(measured.performance_score),
              "qps": performance.get("qps"), "performance": performance}
    if reasons:
        result["reasons"] = reasons
    return result


def _project_measured_algo(algo: dict) -> dict:
    """Keep resolved sampling choices across the local measured transport.

    API generation defaults to 4096 tokens/temperature 0.4; local measured
    generation defaults to 512/1.0. Resolve these before changing transport,
    then protect the selected cap from the provider's historical 512 policy.
    The provider consumes the private marker; it is never an archive parameter.
    """
    from rag_stack.system_transfer_backend import project_to_vllm_chat

    config = deepcopy(algo)
    for line in config.get("node_lines", []):
        for node in line.get("nodes", []):
            key = "generator_backend" if node.get("stage") == "query_expansion" else "component"
            if node.get("stage") not in {"generator", "query_expansion"}:
                continue
            for module in node.get("modules", []):
                backend = module.get(key)
                if backend not in {"vllm", "vllm_api"}:
                    continue
                api = backend == "vllm_api"
                module.setdefault("temperature", 0.4 if api else 1.0)
                if api:
                    module["max_tokens"] = module.get("max_tokens") or 4096
                else:
                    module.setdefault("max_tokens", 512)
                module["_measured_source_max_tokens"] = module["max_tokens"]
    projected, _ = project_to_vllm_chat(config)
    return projected


def evaluate_pair(algo: dict, system: dict, mode: str, project: Path, *,
                  qa_path: Path, corpus_path: Path, dataset_name: str | None = None) -> dict:
    """Execute the selected real backend; caller owns the temporary directory."""
    if mode not in MODES:
        raise ValueError(f"Unknown replay mode: {mode}")
    algo, system = deepcopy(algo), deepcopy(system)
    # Internal execution still reads pipeline.system; persisted inputs have one
    # authoritative deployment mapping, supplied separately by the caller.
    algo["system"] = deepcopy(system)
    if mode == "quality":
        settings = algo.setdefault("eval_backend_setting", {})
        settings.pop("performance_only", None)
        if not isinstance(settings.get("metrics"), list) or not settings["metrics"]:
            raise ValueError("Quality replay requires non-empty eval_backend_setting.metrics; "
                             "this configuration contains no scoring protocol")
    if mode == "measured":
        if (algo.get("global") or {}).get("eval_backend", "static_gt") != "static_gt":
            raise ValueError("Measured replay currently requires the static_gt backend")
        algo = _project_measured_algo(algo)
        algo["eval_backend_setting"] = {"performance_only": True}
    elif mode == "cm":
        from rag_stack.cost_model.support import require_cm_pipeline_support
        require_cm_pipeline_support(algo)
        if (not dataset_name and any(store.get("db_type") in {"faiss_ivf", "faiss_hnsw"}
                                     for store in algo.get("vectordb", []))):
            raise ValueError("FAISS CM replay requires --dataset-name to select the corpus workload "
                             "profiles; use the corpus identity recorded in RUN.md")
        algo.setdefault("_runtime", {})["rag_ir_mode"] = "dynamic"

    manager, evaluator, backend = _runtime(algo, project, qa_path=qa_path, corpus_path=corpus_path,
                                           dataset_name=dataset_name)
    if mode == "measured":
        return _measure(algo, system, evaluator, project)

    from rag_stack.evaluator_contract import normalize_backend_result
    raw = evaluator.evaluate(algo, run_dir=str(project / "quality"),
                             metrics_override=[] if mode == "cm" else None)
    question_ids = manager.qa_data["qid"].tolist()
    normalized = normalize_backend_result(
        raw, backend=backend, pipeline_config=algo, question_ids=question_ids,
    )
    if mode == "cm":
        result = _price(algo, system, normalized["__execution_dag__"], dataset_name=dataset_name)
    else:
        result = {"status": "ok", "metrics": {
            key: value for key, value in normalized.items() if not key.startswith("__")
        }}
        if not result["metrics"]:
            raise ValueError("Quality evaluator returned no scores for the configured metrics")
        # Scores are results. Judge requests, receipts and model logs remain
        # temporary, while any per-query score vector is retained explicitly.
        scores = {}
        for path in sorted((project / "quality").rglob("*_scores.json")):
            payload = json.loads(path.read_text())
            if isinstance(payload, dict) and isinstance(payload.get("scores"), list):
                scores[str(path.relative_to(project / "quality"))] = payload["scores"]
        if scores:
            result["per_query_scores"] = scores
    result["question_ids"] = question_ids
    return result


def _json_default(value):
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported result value: {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=_json_default) + "\n")
    temporary.replace(path)


def run_replay(algo_path: Path, system_path: Path, *, qa_path: Path, corpus_path: Path,
               mode: str, output: Path, dataset_name: str | None = None) -> int:
    algo, system = load_replay_inputs(algo_path, system_path)
    qa_path, corpus_path = _dataset_path(qa_path), _dataset_path(corpus_path)
    output.mkdir(parents=True, exist_ok=True)
    original_project = os.environ.get("PROJECT_DIR")
    started = datetime.now(timezone.utc).isoformat()
    try:
        # A fresh manager cannot silently reuse a previous run's QA binding.
        with tempfile.TemporaryDirectory(prefix="rag-replay-") as scratch:
            os.environ["PROJECT_DIR"] = scratch
            result = evaluate_pair(algo, system, mode, Path(scratch),
                                   qa_path=qa_path, corpus_path=corpus_path, dataset_name=dataset_name)
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if original_project is None:
            os.environ.pop("PROJECT_DIR", None)
        else:
            os.environ["PROJECT_DIR"] = original_project
    result.update(mode=mode, algo_config=str(algo_path), system_config=str(system_path),
                  started_at_utc=started, completed_at_utc=datetime.now(timezone.utc).isoformat())
    _write_json(output / f"{mode}_result.json", result)
    notes = output / "RUN.md"
    if not notes.exists():
        notes.write_text("# Replay events\n\nInputs are the referenced algorithm/system JSON pair. "
                         "QA and corpus files are explicit invocation arguments.\n")
    with notes.open("a") as stream:
        stream.write(f"\n- {started}: `{mode}` — **{result['status']}**. "
                     f"Result: `{mode}_result.json`.\n")
        command = shlex.join(["python", "-m", "rag_stack", "replay", str(algo_path),
                              str(system_path), "--qa", str(qa_path), "--corpus", str(corpus_path),
                              "--mode", mode] + (["--dataset-name", dataset_name] if dataset_name else []))
        stream.write(f"\n  Command: `{command}`\n")
        if result.get("error"):
            stream.write(f"  {result['error']}\n")
    return 0 if result["status"] == "ok" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, prog="rag_stack replay")
    parser.add_argument("algo_config", type=Path)
    parser.add_argument("system_config", type=Path)
    parser.add_argument("--qa", type=Path, required=True, help="Caller-selected QA Parquet file")
    parser.add_argument("--corpus", type=Path, required=True, help="Caller-selected corpus Parquet file")
    parser.add_argument("--dataset-name", help="Corpus identity for workload profiles; required for FAISS CM replay")
    parser.add_argument("--mode", choices=MODES, default="measured")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Check inputs without loading models or running evaluation")
    args = parser.parse_args(argv)
    algo_path = args.algo_config.expanduser().resolve()
    system_path = args.system_config.expanduser().resolve()
    try:
        if args.dry_run:
            load_replay_inputs(algo_path, system_path)
            _dataset_path(args.qa)
            _dataset_path(args.corpus)
            print("Resolved algorithm/system inputs and dataset files are present; no models were run.")
            return 0
        output = args.output_dir or (Path("tmp_outputs/replays") /
                                    (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"))
        return run_replay(algo_path, system_path, qa_path=args.qa, corpus_path=args.corpus,
                          mode=args.mode, output=output.resolve(), dataset_name=args.dataset_name)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"rag_stack replay: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
