"""CLI entrypoints for the LLM cost model benchmark + calibration framework.

Two subcommands share the same benchmark backend:

- ``benchmark`` — measure real latency vs predicted (ceff=meff=1.0),
  print a diagnostic table and verdict. Does NOT save coefficients.
  Without ``--profile`` this is an explicitly uncalibrated physical-roofline
  diagnostic, not a production price. Use it to decide whether the ModelConfig
  is structurally correct before attempting calibration.
- ``calibrate`` — run the same benchmark, fit
  ``(compute_efficiency, memory_efficiency)`` via scipy L-BFGS-B,
  and persist one JSON per (model, hardware) under
  ``calibration_profiles/llm_sim/``.

Examples:

    rag_stack calibrate llm benchmark \\
        --config configs/rag_stack/sgs_openrouter.yaml \\
        --cost-model passage_reranker_sim --module tart

    rag_stack calibrate llm \\
        --config configs/rag_stack/sgs_openrouter.yaml \\
        --cost-model passage_reranker_sim --module tart \\
        --batch-sizes 1,2,4,8 --seq-lens 128,256,512
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("RAG-Stack")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x]


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--cost-model", required=True,
        help=(
            "Registered cost model name (see COST_MODEL_REGISTRY). "
            "Examples: passage_reranker_sim, passage_reranker_llm_sim, llm_sim, query_expansion_sim, "
            "passage_compressor_sim, passage_filter_sim."
        ),
    )
    p.add_argument(
        "--module", required=True,
        help=(
            "component that selects the concrete runner. "
            "Example: 'tart' for passage_reranker_sim."
        ),
    )
    p.add_argument(
        "--config", required=True,
        help=(
            "Path to a rag_stack YAML config (e.g. configs/rag_stack/"
            "sgs_openrouter.yaml). The ``system.gpu`` section provides peak "
            "physical specs and ``hardware_key`` (calibration registry id). "
            "YAML GPU efficiency fields are ignored."
        ),
    )
    p.add_argument(
        "--batch-sizes", type=_int_list, default=[1, 8, 32, 64, 128, 256],
        help="Comma-separated batch sizes. Default: 1,8,32,64,128,256",
    )
    p.add_argument(
        "--seq-lens", type=_int_list, default=[128, 256, 512, 1024],
        help="Comma-separated per-passage seq lengths. Default: 128,256,512,1024",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--measurements", type=int, default=20)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="rag_stack calibrate llm",
        description=(
            "Benchmark + calibrate LLM cost models against real-hardware "
            "latency."
        ),
    )
    sub = ap.add_subparsers(dest="command", required=True)

    bench = sub.add_parser(
        "benchmark",
        help="Measure real vs predicted latency; emit diagnostic report. "
             "Does not persist coefficients.",
    )
    _add_common_args(bench)
    bench.add_argument(
        "--output-report", default=None,
        help="Optional CSV path for the per-point table.",
    )
    bench.add_argument(
        "--profile", default=None,
        help="Optional existing calibration JSON. Either a per-system file with a "
             "record-key fragment (calibration_profiles/llm_sim/<hw>.json"
             "#<module>__<model>__<bits>) or a standalone single-record JSON. "
             "Highest-priority source for (compute_efficiency, memory_efficiency). "
             "When absent, benchmark uses the uncalibrated physical roofline "
             "(1.0, 1.0) for diagnostics only; production pricing requires an "
             "exact calibration record.",
    )

    cal = sub.add_parser(
        "calibrate",
        help="Same measurement, plus fit (compute_efficiency, "
             "memory_efficiency) and save to JSON.",
    )
    _add_common_args(cal)
    cal.add_argument(
        "--output-dir", default=None,
        help="Directory holding one <hw>.json per system (records keyed by "
             "<module>__<model>__<bits>). Default: calibration_profiles/llm_sim/",
    )

    refit = sub.add_parser(
        "refit",
        help="Re-fit an EXISTING record from its stored measured grid "
             "(offline, no GPU) — for fitter upgrades like the affine "
             "per-call-overhead term. Guarded by a prediction-reproduction "
             "check against the stored per-point predictions.",
    )
    _add_common_args(refit)
    refit.add_argument(
        "--output-dir", default=None,
        help="Profile directory (default: calibration_profiles/llm_sim/).",
    )
    refit.add_argument(
        "--bits", default=None,
        help="Record bits discriminator (e.g. f32). Default: glob with "
             "bf16→f32 preference.",
    )
    refit.add_argument(
        "--reproduction-tolerance", type=float, default=0.02,
        help="Max allowed relative deviation when reproducing the record's "
             "stored predicted_s with its saved coefficients (default 0.02).",
    )
    refit.add_argument(
        "--form", default="affine", choices=["affine", "linear"],
        help="Fit form: 'affine' = GenZ roofline x (ceff, meff) + per-call "
             "overhead (default); 'linear' = per-item linear law "
             "t = fixed + B*(per_item + per_token*L) for models whose "
             "latency is not roofline-shaped (tiny cross-encoders).",
    )

    return ap


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

def _resolve_cost_model(name: str):
    from rag_stack.cost_model import COST_MODEL_REGISTRY
    cls = COST_MODEL_REGISTRY.get(name)
    if cls is None:
        raise SystemExit(
            f"Unknown cost model: {name!r}. "
            f"Available: {sorted(COST_MODEL_REGISTRY.keys())}"
        )
    return cls()


def _build_pipeline_config(component: str, cost_model_name: str = "") -> dict:
    """Build a minimal pipeline_config that selects this component.

    The cost model's _find_module() looks for a node with a matching
    stage whose first module has the given component. We provide
    just enough structure for that + _token_stats to compute input_tokens
    for the benchmark.

    ``retrieval_encode_sim`` is special: its ``_find_module`` returns the
    VECTORDB entry the semantic_retrieval node points at (the embedding
    model lives there, not on a node module), so ``component`` is the
    embedding-model name and the scaffold carries a one-entry ``vectordb``
    section. Shape injection for it goes through the cost model's
    ``_inject_calibration_shape`` override (query-token + vectordb batch).
    """
    if cost_model_name == "retrieval_encode_sim":
        return {
            "node_lines": [{
                "node_line_name": "calibration",
                "nodes": [{
                    "stage": "semantic_retrieval",
                    "top_k": 4,
                    "modules": [{
                        "component": "vectordb",
                        "vectordb": "calibration_db",
                        "batch_size": 1,
                    }],
                }],
            }],
            "vectordb": [{
                "name": "calibration_db",
                "embedding_model": component,
                "batch_size": 1,
            }],
            "_token_stats": {
                "avg_chunk_tokens": 128,
                "prompt_template_tokens": 0,
                "avg_output_tokens": 1,
                "avg_query_tokens": 16,
            },
        }
    # Map known components to their owning stage
    stage = _node_type_for_module(component)
    return {
        "node_lines": [{
            "node_line_name": "calibration",
            "nodes": [{
                "stage": stage,
                "top_k": 4,
                "modules": [{
                    "component": component,
                    "batch_size": 1,
                }],
            }],
        }],
        "_token_stats": {
            "avg_chunk_tokens": 128,
            "prompt_template_tokens": 32,
            "avg_output_tokens": 64,
            "avg_query_tokens": 16,
        },
    }


def _node_type_for_module(component: str) -> str:
    # Used only at CLI scaffolding time; extend as new modules add runners.
    _MAP = {
        # reranker
        "tart": "passage_reranker",
        "monot5": "passage_reranker",
        "sentence_transformer_reranker": "passage_reranker",
        "flag_embedding_reranker": "passage_reranker",
        "colbert_reranker": "passage_reranker",
        "flashrank_reranker": "passage_reranker",
        "openvino_reranker": "passage_reranker",
        "upr": "passage_reranker",
        "flag_embedding_llm_reranker": "passage_reranker",
        # compressor
        "llmlingua2": "passage_compressor",
        # rewriter / generator / filter go here as needed
    }
    return _MAP.get(component, "passage_reranker")


def _build_hardware(args) -> "SystemConfig":
    """Load ``SystemConfig`` from the rag_stack YAML at ``args.config``.

    The YAML's ``system.gpu`` section drives both peak specs and the
    calibration registry key (``gpu.hardware_key``).
    """
    from rag_stack.types import SystemConfig
    from rag_stack.utils.config import load_yaml_config

    config = load_yaml_config(args.config)
    return SystemConfig.from_dict(config["system"])


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _write_report_csv(report, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["batch_size", "seq_len", "measured_s",
                    "predicted_s", "ratio", "rel_err"])
        for p in report.points:
            w.writerow([p.batch_size, p.seq_len, p.measured_s,
                        p.predicted_s, p.ratio, p.rel_err])


def cmd_benchmark(args) -> int:
    import json as _json

    cost_model = _resolve_cost_model(args.cost_model)
    pipeline_config = _build_pipeline_config(args.module, args.cost_model)
    hardware = _build_hardware(args)

    # Resolve diagnostic coefficients.  An explicit profile verifies calibrated
    # behavior; without one, benchmark deliberately shows the raw physical
    # roofline.  Production CM resolution is stricter and never enters here.
    if args.profile:
        spec, _, key = args.profile.partition("#")
        with open(spec) as f:
            data = _json.load(f)
        if "compute_efficiency" in data:
            rec = data                          # standalone single-record JSON
        else:
            from rag_stack.llm_sim_profile_schema import (
                unwrap_llm_sim_profile_document,
            )

            try:
                profiles = unwrap_llm_sim_profile_document(
                    data, source=str(spec),
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            if key and key not in profiles:
                raise SystemExit(
                    f"Record key {key!r} not in {spec}. "
                    f"Available: {sorted(profiles)}"
                )
            if key:
                rec = profiles[key]
            elif len(profiles) == 1:
                rec = next(iter(profiles.values()))  # unambiguous system file
            else:
                raise SystemExit(
                    f"{spec} holds multiple records; append "
                    "#<module>__<model>__<bits>. "
                    f"Available: {sorted(profiles)}"
                )
        compute_eff = float(rec["compute_efficiency"])
        memory_eff = float(rec["memory_efficiency"])
        logger.info(
            f"Using profile {args.profile}: ceff={compute_eff:.3f}, "
            f"meff={memory_eff:.3f}"
        )
    else:
        compute_eff = memory_eff = 1.0
        logger.warning(
            "No --profile supplied: using the uncalibrated physical roofline "
            "(ceff=meff=1.0) for this diagnostic benchmark only. Production "
            "GenZ pricing requires an exact calibration record."
        )

    report = cost_model.benchmark(
        hardware=hardware,
        pipeline_config=pipeline_config,
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        warmup=args.warmup,
        measurements=args.measurements,
        compute_eff=compute_eff,
        memory_eff=memory_eff,
    )

    print(report.format())
    if args.output_report:
        _write_report_csv(report, Path(args.output_report))
        print(f"\nReport CSV written: {args.output_report}")

    # Exit code 0 if params look reasonable; non-zero if shape_mismatch.
    return 0 if report.verdict != "shape_mismatch" else 2


def cmd_calibrate(args) -> int:
    cost_model = _resolve_cost_model(args.cost_model)
    pipeline_config = _build_pipeline_config(args.module, args.cost_model)
    hardware = _build_hardware(args)

    output_dir = Path(args.output_dir) if args.output_dir else None
    record = cost_model.calibrate(
        hardware=hardware,
        pipeline_config=pipeline_config,
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        warmup=args.warmup,
        measurements=args.measurements,
        output_dir=output_dir,
    )

    print(
        f"Calibration saved for {record.model_key}@{record.hardware_key}:\n"
        f"  compute_efficiency = {record.compute_efficiency:.3f}\n"
        f"  memory_efficiency  = {record.memory_efficiency:.3f}\n"
        f"  fit_mape           = {record.fit_mape * 100:.1f} %\n"
        f"  correlation        = {record.correlation:.3f}\n"
        f"  grid_points        = {len(record.grid)}"
    )
    return 0


def cmd_refit(args) -> int:
    """Re-fit an existing record from its STORED measured grid — no GPU.

    Use after a fitter improvement (e.g. the affine per-call-overhead term /
    multistart) to upgrade records without re-measuring. Safety gate: before
    fitting, the stored per-point ``predicted_s`` must be reproducible with
    the record's saved coefficients through TODAY's prediction path — if it
    is not, the (cost_model, scaffold) pair here differs from whatever
    produced the record, and refitting would silently change semantics.
    ``--reproduction-tolerance`` bounds the allowed max relative deviation.
    """
    from rag_stack.cost_model.llm_sim.calibration import (
        _inject_shape,
        detect_hardware_key,
        _hardware_to_dict,
        fit_and_save,
        grid_seq_axis,
        load_record,
    )
    from rag_stack.cost_model.llm_sim.calibration._predict import predict_latency

    cost_model = _resolve_cost_model(args.cost_model)
    pipeline_config = _build_pipeline_config(args.module, args.cost_model)
    hardware = _build_hardware(args)
    hw_key = detect_hardware_key(hardware)
    model_key = cost_model._get_model_name_for_df(pipeline_config)
    # Use the same model-owned identity contract as online publication.  This
    # keeps refit and first-time calibration on exactly the same record key.
    component = cost_model._calibration_component(pipeline_config)

    directory = Path(args.output_dir) if args.output_dir else None
    rec = load_record(component, model_key, hw_key, bits=args.bits,
                      directory=directory)
    if rec is None:
        raise SystemExit(
            f"[refit] no record {component}__{model_key}__{args.bits or '*'} "
            f"in llm_sim/{hw_key}.json — nothing to refit"
        )

    measured = {
        (int(g["batch_size"]), grid_seq_axis(g["seq_len"])): float(g["measured_s"])
        for g in rec.grid if "batch_size" in g and g.get("measured_s")
    }
    validation = {
        (int(g["batch_size"]), grid_seq_axis(g["seq_len"])): float(g["measured_s"])
        for g in rec.validation_grid
        if "batch_size" in g and g.get("measured_s")
    }
    if not measured:
        raise SystemExit(
            f"[refit] record {component}__{model_key}__{rec.bits} stores no "
            f"measured grid — cannot refit offline; re-run `calibrate` on the GPU"
        )

    # Reproduction gate: today's path at the SAVED coefficients vs the
    # SAVED predictions.
    inject = getattr(cost_model, "_inject_calibration_shape", None) or _inject_shape
    hw_dict = _hardware_to_dict(hardware)
    old_oh = max(0.0, float(getattr(rec, "per_call_overhead_s", 0.0) or 0.0))
    max_dev, n_checked = 0.0, 0
    for g in rec.grid:
        if "batch_size" not in g or not g.get("predicted_s"):
            continue
        cfg = inject(pipeline_config, int(g["batch_size"]), grid_seq_axis(g["seq_len"]))
        pred = predict_latency(
            cost_model, cfg, hw_dict,
            compute_eff=rec.compute_efficiency,
            memory_eff=rec.memory_efficiency,
            bits=rec.bits, per_call_overhead_s=old_oh,
        )
        dev = abs(pred - float(g["predicted_s"])) / float(g["predicted_s"])
        max_dev = max(max_dev, dev)
        n_checked += 1
    print(f"[refit] reproduction check: {n_checked} points, "
          f"max |pred_now - pred_stored| / pred_stored = {max_dev:.4f}")
    if max_dev > args.reproduction_tolerance:
        raise SystemExit(
            f"[refit] reproduction deviation {max_dev:.4f} exceeds tolerance "
            f"{args.reproduction_tolerance} — the prediction path no longer "
            f"matches the one that produced this record. Investigate (wrong "
            f"--cost-model? scaffold drift? deliberate model change?) before "
            f"refitting; raise --reproduction-tolerance ONLY for a deliberate,"
            f" understood change."
        )

    if getattr(args, "form", "affine") == "linear":
        from rag_stack.cost_model.llm_sim.calibration import fit_linear_and_save
        new_rec = fit_linear_and_save(
            cost_model=cost_model,
            hardware=hardware,
            pipeline_config=pipeline_config,
            record=rec,
            output_dir=directory,
        )
        ll = new_rec.linear_law or {}
        print(
            f"Linear-law refit {component}__{model_key}__{rec.bits}@{hw_key} "
            f"({len(measured)} stored points):\n"
            f"  fixed_ms     = {ll.get('fixed_ms', 0.0):.3f}\n"
            f"  per_item_ms  = {ll.get('per_item_ms', 0.0):.4f}\n"
            f"  per_token_ms = {ll.get('per_token_ms', 0.0):.6f}\n"
            f"  fit_mape     = {rec.fit_mape * 100:.1f} % → "
            f"{new_rec.fit_mape * 100:.1f} %\n"
            f"  correlation  = {new_rec.correlation:.3f}"
        )
        return 0

    new_rec = fit_and_save(
        cost_model=cost_model,
        hardware=hardware,
        pipeline_config=pipeline_config,
        measured=measured,
        component=component,
        model_key=model_key,
        output_dir=directory,
        bits=rec.bits,
        validation_points=validation,
    )
    print(
        f"Refit {component}__{model_key}__{rec.bits}@{hw_key} "
        f"({len(measured)} stored points):\n"
        f"  compute_efficiency = {rec.compute_efficiency:.4f} → {new_rec.compute_efficiency:.4f}\n"
        f"  memory_efficiency  = {rec.memory_efficiency:.4f} → {new_rec.memory_efficiency:.4f}\n"
        f"  per_call_overhead  = {old_oh * 1000:.2f} ms → {new_rec.per_call_overhead_s * 1000:.2f} ms\n"
        f"  fit_mape           = {rec.fit_mape * 100:.1f} % → {new_rec.fit_mape * 100:.1f} %\n"
        f"  correlation        = {new_rec.correlation:.3f}"
    )
    return 0


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.command == "benchmark":
        return cmd_benchmark(args)
    if args.command == "calibrate":
        return cmd_calibrate(args)
    if args.command == "refit":
        return cmd_refit(args)
    parser.error(f"Unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
