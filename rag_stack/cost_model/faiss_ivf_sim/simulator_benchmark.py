"""
Benchmark vs Simulator Comparison

Compares the analytical cost model against actual FAISS results on SIFT1M.

Usage:
    python -m rag_stack.cost_model.faiss_ivf_sim benchmark [options]
    python -m rag_stack.cost_model.faiss_ivf_sim.simulator_benchmark [options]
"""

import argparse
import collections
import csv
import io
import json
import os

import numpy as np

from .faiss_benchmark import FaissBenchmarkRunner
from .ivfpq_config import IVFPQConfig
from .cost_model import estimate_search_cost
from .systems import SystemConfig
from .cost_model import QueryCostBreakdown
from .types import BenchmarkResult, SimulatorResult, ComparisonResult
from .calibrator import parse_index_config


class SimulatorRunner:
    """Runs the analytical simulator with matching configurations."""

    def __init__(self, hardware: SystemConfig, warm_cache: bool = False):
        self.hardware = hardware
        self.warm_cache = warm_cache

    def simulate(
        self,
        d: int,
        N: int,
        nlist: int,
        M: int,
        nbits: int,
        k: int,
        nprobe_values: list[int],
        batch_size: int = 1,
        by_residual: bool = True,
        parallel_mode: int = 0,
        index_type: str = "pq",
        cell_imbalance: "float | dict[int, float]" = 1.0,
    ) -> list[tuple[SimulatorResult, QueryCostBreakdown]]:
        """Returns list of (SimulatorResult, full breakdown) tuples.

        ``cell_imbalance`` may be a single float (applied to every nprobe) or a
        {nprobe: imbalance} dict (per-nprobe, the correct form — the imbalance
        decays toward 1.0 as nprobe→nlist; a constant over-counts at high nprobe).
        """
        results = []
        for nprobe in nprobe_values:
            imb = (cell_imbalance.get(nprobe, 1.0)
                   if isinstance(cell_imbalance, dict) else cell_imbalance)
            config = IVFPQConfig(
                index_type=index_type,
                d=d, N=N, nlist=nlist, nprobe=nprobe, M=M, nbits=nbits,
                k=k, batch_size=batch_size, by_residual=by_residual,
                parallel_mode=parallel_mode, cell_imbalance=imb,
            )
            cost = estimate_search_cost(config, self.hardware, warm_cache=self.warm_cache)
            sim_result = SimulatorResult(
                nprobe=nprobe, latency_ms=cost.latency_ms,
                qps=cost.throughput_qps, bottleneck=cost.overall_bottleneck,
            )
            results.append((sim_result, cost))
        return results


def compare_results(
    benchmark_results: list[BenchmarkResult],
    simulator_results: list[tuple[SimulatorResult, QueryCostBreakdown]],
) -> list[ComparisonResult]:
    sim_by_nprobe = {r.nprobe: r for r, _ in simulator_results}
    comparisons = []
    for bench in benchmark_results:
        sim = sim_by_nprobe.get(bench.nprobe)
        if not sim:
            continue
        comparisons.append(ComparisonResult(
            nprobe=bench.nprobe,
            actual_qps=bench.qps, predicted_qps=sim.qps,
            actual_latency_ms=bench.latency_ms, predicted_latency_ms=sim.latency_ms,
            qps_error_pct=(sim.qps - bench.qps) / bench.qps * 100,
            latency_error_pct=(sim.latency_ms - bench.latency_ms) / bench.latency_ms * 100,
        ))
    return comparisons


def _spearman(a, b) -> float:
    """Spearman rank correlation (Pearson of ranks), numpy-only (no scipy dep).

    The load-bearing metric for a cost model feeding optimization: it measures
    whether the CM ORDERS configs correctly (does it know which is faster),
    which is what a Pareto search needs — absolute MAPE matters less. Within one
    config (over nprobe) latency is monotonic so this is ~1; the informative
    value is the GLOBAL one pooled across all configs (index types / threads /
    batches mixed), where ranking is genuinely non-trivial.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2:
        return 1.0
    ra = np.argsort(np.argsort(a))  # ranks (ties broken by order; fine for latencies)
    rb = np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return 1.0
    return float(np.corrcoef(ra, rb)[0, 1])


def _qerror(actual, predicted):
    """q-error = max(pred/actual, actual/pred) ≥ 1 (Moerkotte et al., VLDB'09).

    The standard cost/cardinality-estimator metric: MULTIPLICATIVE and SYMMETRIC
    (a 2x over- and a 2x under-estimate both score 2), and bounding it bounds plan
    quality — more apt for a cost model than additive MAPE. Reported as a
    DISTRIBUTION (median / p90 / p95 / p99 / max / geomean) since the tail is what
    hurts an optimizer.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    return np.maximum(p / a, a / p)


def calculate_accuracy_metrics(comparisons: list[ComparisonResult]) -> dict:
    if not comparisons:
        return {}
    qps_errors = [abs(c.qps_error_pct) for c in comparisons]
    latency_errors = [abs(c.latency_error_pct) for c in comparisons]
    actual_qps = np.array([c.actual_qps for c in comparisons])
    predicted_qps = np.array([c.predicted_qps for c in comparisons])
    actual_lat = np.array([c.actual_latency_ms for c in comparisons])
    predicted_lat = np.array([c.predicted_latency_ms for c in comparisons])
    qe = _qerror(actual_lat, predicted_lat)
    correlation = np.corrcoef(actual_qps, predicted_qps)[0, 1] if len(actual_qps) > 1 else 1.0
    return {
        "mape_qps": np.mean(qps_errors),
        "max_ape_qps": np.max(qps_errors),
        "mape_latency": np.mean(latency_errors),
        "max_ape_latency": np.max(latency_errors),
        "qerror_median": float(np.median(qe)),            # q-error (VLDB'09)
        "qerror_max": float(qe.max()),
        "correlation": correlation,                       # Pearson (qps)
        "spearman_latency": _spearman(actual_lat, predicted_lat),
        "num_points": len(comparisons),
    }


def aggregate_metrics(all_comparison_data: list[dict]) -> dict:
    """Pool EVERY comparison point across all configs → global accuracy.

    The per-config metrics average rank-trivial within-config series; this pools
    all (actual, predicted) points so the global Spearman reflects ranking across
    the whole config space (the number that matters for optimizer fidelity).
    """
    comps = [c for e in all_comparison_data for c in e["comparisons"]]
    if not comps:
        return {}
    al = np.array([c.actual_latency_ms for c in comps])
    pl = np.array([c.predicted_latency_ms for c in comps])
    aq = np.array([c.actual_qps for c in comps])
    pq = np.array([c.predicted_qps for c in comps])
    qe = _qerror(al, pl)  # q-error distribution (the tail is what matters)
    return {
        "n": len(comps),
        "mape_latency": float(np.mean(np.abs(pl / al - 1) * 100)),
        "mape_qps": float(np.mean(np.abs(pq / aq - 1) * 100)),
        "qerror_median": float(np.median(qe)),
        "qerror_p90": float(np.percentile(qe, 90)),
        "qerror_p95": float(np.percentile(qe, 95)),
        "qerror_p99": float(np.percentile(qe, 99)),
        "qerror_max": float(qe.max()),
        "qerror_gmean": float(np.exp(np.mean(np.log(qe)))),
        "pearson_latency": float(np.corrcoef(al, pl)[0, 1]) if len(al) > 1 else 1.0,
        "spearman_latency": _spearman(al, pl),
    }


def _format_report(
    all_comparison_data: list[dict],
    all_metrics: list[dict],
    config_info: dict | None = None,
) -> str:
    """Build a human-readable plain-text report."""
    buf = io.StringIO()
    buf.write("=" * 70 + "\n")
    buf.write("FAISS Benchmark vs Simulator Comparison Report\n")
    buf.write("=" * 70 + "\n")

    if config_info:
        if config_info.get("datasets"):
            buf.write(f"  Datasets:      {config_info['datasets']}\n")
        buf.write(f"  Index configs: {config_info['index_configs']}\n")
        buf.write(f"  nprobe values: {config_info['nprobe_values']}\n")
        buf.write(f"  Batch sizes:   {config_info['batch_sizes']}\n")
        buf.write(f"  Hardware:      {config_info['hardware_preset']}\n")
        buf.write(f"  Num threads:   {config_info['num_threads']}\n")
        buf.write(f"  Parallel modes:{config_info['parallel_modes']}\n")
        buf.write(f"  Cache model:   {config_info['cache_model']}\n")

    for entry in all_comparison_data:
        label = entry["label"]
        comparisons = entry["comparisons"]
        breakdown_by_nprobe = entry["breakdown_by_nprobe"]
        metrics = entry["metrics"]

        buf.write(f"\n{'─' * 70}\n")
        buf.write(f"{label}\n")
        buf.write("─" * 70 + "\n")

        for c in comparisons:
            buf.write(
                f"  nprobe={c.nprobe:3d}: "
                f"actual={c.actual_latency_ms:.4f}ms  "
                f"pred={c.predicted_latency_ms:.4f}ms  "
                f"err={c.latency_error_pct:+.1f}%\n"
            )
            cost = breakdown_by_nprobe.get(c.nprobe)
            if cost:
                for op in cost.operators:
                    buf.write(
                        f"    {op.name:20s}  "
                        f"T={op.time_s*1e3:8.3f}ms  "
                        f"compute={op.compute_time_s*1e3:8.3f}ms  "
                        f"memory={op.memory_time_s*1e3:8.3f}ms  "
                        f"[{op.bottleneck}]\n"
                    )

        buf.write(
            f"\n  MAPE(latency): {metrics['mape_latency']:.2f}%  "
            f"Max APE: {metrics['max_ape_latency']:.2f}%  "
            f"Corr: {metrics['correlation']:.4f}\n"
        )

    # Summary
    buf.write(f"\n{'=' * 70}\n")
    buf.write("SUMMARY (per config)\n")
    buf.write("=" * 70 + "\n")
    for m in all_metrics:
        buf.write(f"  {m['label']:46s}  lat MAPE={m['mape_latency']:5.1f}%  "
                  f"q-err med={m.get('qerror_median', 1):.2f} max={m.get('qerror_max', 1):.2f}  "
                  f"spearman={m.get('spearman_latency', 1.0):.3f}\n")

    # GLOBAL: pool every point → the ranking + q-error metrics that matter.
    g = aggregate_metrics(all_comparison_data)
    overall_mape = float(np.mean([m["mape_latency"] for m in all_metrics]))
    buf.write(f"\n  Overall latency MAPE (per-config mean): {overall_mape:.2f}%\n")
    if g:
        buf.write(f"\n{'-' * 70}\n  GLOBAL (all {g['n']} points pooled):\n")
        buf.write(f"    latency MAPE : {g['mape_latency']:.2f}%      qps MAPE : {g['mape_qps']:.2f}%\n")
        buf.write(f"    q-error (Moerkotte VLDB'09, max(pred/act, act/pred)):\n")
        buf.write(f"        median {g['qerror_median']:.2f}  p90 {g['qerror_p90']:.2f}  "
                  f"p95 {g['qerror_p95']:.2f}  p99 {g['qerror_p99']:.2f}  "
                  f"max {g['qerror_max']:.2f}  gmean {g['qerror_gmean']:.2f}\n")
        buf.write(f"    Pearson  (latency) : {g['pearson_latency']:.4f}\n")
        buf.write(f"    Spearman (latency) : {g['spearman_latency']:.4f}   "
                  f"<- rank correlation; the optimizer-fidelity metric\n")
    return buf.getvalue()


def _write_csv(
    all_comparison_data: list[dict],
    output_path: str,
) -> None:
    """Write per-nprobe comparison results to a CSV file."""
    fieldnames = [
        "dataset", "d", "index_name", "num_threads", "batch_size", "parallel_mode", "nprobe",
        "actual_latency_ms", "predicted_latency_ms", "latency_error_pct",
        "actual_qps", "predicted_qps", "qps_error_pct",
    ]
    # Collect the UNION of operator names across all entries (different index
    # types have different ops: Flat → flat_scan, PQ → lut_build+pq_scan, etc.).
    op_names: list[str] = []
    seen: set[str] = set()
    for entry in all_comparison_data:
        for cost in entry["breakdown_by_nprobe"].values():
            for op in cost.operators:
                if op.name not in seen:
                    seen.add(op.name)
                    op_names.append(op.name)

    for name in op_names:
        fieldnames.extend([
            f"{name}_time_ms",
            f"{name}_compute_ms",
            f"{name}_memory_ms",
            f"{name}_bottleneck",
        ])

    file_exists = os.path.exists(output_path) and os.path.getsize(output_path) > 0
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for entry in all_comparison_data:
            for c in entry["comparisons"]:
                row = {
                    "dataset": entry.get("dataset", ""),
                    "d": entry.get("d", ""),
                    "index_name": entry["index_name"],
                    "num_threads": entry.get("num_threads", ""),
                    "batch_size": entry["batch_size"],
                    "parallel_mode": entry["parallel_mode"],
                    "nprobe": c.nprobe,
                    "actual_latency_ms": f"{c.actual_latency_ms:.6f}",
                    "predicted_latency_ms": f"{c.predicted_latency_ms:.6f}",
                    "latency_error_pct": f"{c.latency_error_pct:.2f}",
                    "actual_qps": f"{c.actual_qps:.2f}",
                    "predicted_qps": f"{c.predicted_qps:.2f}",
                    "qps_error_pct": f"{c.qps_error_pct:.2f}",
                }
                cost = entry["breakdown_by_nprobe"].get(c.nprobe)
                if cost:
                    for op in cost.operators:
                        row[f"{op.name}_time_ms"] = f"{op.time_s*1e3:.6f}"
                        row[f"{op.name}_compute_ms"] = f"{op.compute_time_s*1e3:.6f}"
                        row[f"{op.name}_memory_ms"] = f"{op.memory_time_s*1e3:.6f}"
                        row[f"{op.name}_bottleneck"] = op.bottleneck
                writer.writerow(row)


def benchmark_main(args=None):
    """CLI entry point for benchmark comparison. Can be called from __main__.py."""
    from .cli import add_common_args, resolve_hardware_preset, resolve_num_threads, build_hardware, print_config_header

    parser = argparse.ArgumentParser(description="Compare FAISS benchmark vs simulator")
    add_common_args(parser)
    parser.add_argument("--calibrated", type=str, default=None,
                        help="Path to calibrated_coefficients.json")
    parser.add_argument("--output-dir", default="tmp_outputs/faiss_sim_bench", type=str,
                        help="Output directory for plots")
    parser.add_argument("--actuals-csv", type=str, default=None,
                        help="Reuse measured actuals from this CSV (columns: dataset, "
                             "index_name, num_threads, batch_size, nprobe, "
                             "actual_latency_ms). Points found here are NOT re-measured; "
                             "only the missing (dataset,index,threads,batch,nprobe) "
                             "combinations are benchmarked live. Lets a run extend a "
                             "frozen campaign without redoing the slow >100ms / large-"
                             "batch / multi-thread points.")

    args = parser.parse_args(args)
    warm_cache = not args.cold_cache
    thread_counts = resolve_num_threads(args)        # list (sweep, default [all, 1])
    hardware_preset = resolve_hardware_preset(args)

    os.makedirs(args.output_dir, exist_ok=True)

    # Optional measured-actuals reuse, keyed
    # (dataset, index, threads, batch, parallel_mode, nprobe) -> latency_ms.
    # parallel_mode defaults to 0 if the CSV omits it (the calibration campaigns).
    actuals: dict = {}
    if args.actuals_csv:
        for r in csv.DictReader(open(args.actuals_csv)):
            actuals[(r["dataset"], r["index_name"], int(r["num_threads"]),
                     int(r["batch_size"]), int(r.get("parallel_mode") or 0),
                     int(r["nprobe"]))] = float(r["actual_latency_ms"])
        print(f"  Reusing {len(actuals)} measured actual(s) from {args.actuals_csv} "
              f"(those points will NOT be re-measured)")

    print("=" * 60)
    print("FAISS Benchmark vs Simulator Comparison")
    print("=" * 60)
    print_config_header(args, thread_counts, hardware_preset)

    all_metrics = []
    all_comparison_data = []

    # Outer sweep: dataset (→ dimension d / N) × thread count. The simulator's d/N
    # come from the per-dataset runner; the hardware's active_threads must match
    # the measured thread count (the CM is calibrated at active_threads=1).
    for dataset in args.datasets:
        benchmark_runner = FaissBenchmarkRunner(
            num_threads=thread_counts[0],
            dataset=dataset,
            data_dir=args.data_dir,
            download_url=args.download_url,
        )

        for num_threads in thread_counts:
            benchmark_runner.num_threads = num_threads
            hardware = build_hardware(hardware_preset, num_threads, args.calibrated)
            simulator_runner = SimulatorRunner(hardware, warm_cache=warm_cache)

            for index_name in args.index_configs:
                idx_config = parse_index_config(index_name)
                # Data-aware scan count: real clustered data scans more than the
                # balanced nprobe·N/nlist. Compute the EXACT per-nprobe imbalance
                # from real query→centroid assignments on the (already-built)
                # index — the factor decays toward 1.0 as nprobe→nlist (a single
                # constant over-counts at high nprobe, e.g. gist Flat np1024 +62%).
                cell_imbalance = benchmark_runner.cell_imbalance_by_nprobe(
                    index_name, args.nprobe_values)
                _imb_str = " ".join(f"np{p}={cell_imbalance[p]:.2f}"
                                    for p in sorted(cell_imbalance))
                print(f"  cell_imbalance({index_name}): {_imb_str}")

                for batch_size in args.batch_sizes:
                    if batch_size > benchmark_runner.nq:
                        continue  # not enough queries for this batch size
                    for parallel_mode in args.parallel_modes:
                        label = (f"{dataset} d={benchmark_runner.d} t={num_threads} "
                                 f"{index_name} bs={batch_size} pm={parallel_mode}")
                        print(f"\n{'─' * 60}")
                        print(f"{label}")
                        print("─" * 60)

                        # Run FAISS benchmark — but reuse any actuals supplied via
                        # --actuals-csv; only the missing nprobe points are measured
                        # live (skips the slow >100ms / large-batch / MT tail).
                        provided = {
                            n: actuals[(dataset, index_name, num_threads, batch_size, parallel_mode, n)]
                            for n in args.nprobe_values
                            if (dataset, index_name, num_threads, batch_size, parallel_mode, n) in actuals}
                        to_measure = [n for n in args.nprobe_values if n not in provided]
                        benchmark_results = [
                            BenchmarkResult(index_name=index_name, nprobe=n,
                                            recall_at_k=0.0, latency_ms=lat,
                                            qps=batch_size / lat * 1000.0,
                                            num_queries=benchmark_runner.nq,
                                            batch_size=batch_size)
                            for n, lat in sorted(provided.items())]
                        if provided:
                            print(f"  reused {len(provided)} measured actual(s): "
                                  f"nprobe {sorted(provided)} (not re-measured)")
                        if to_measure:
                            benchmark_results += benchmark_runner.benchmark_index(
                                index_name=index_name,
                                nprobe_values=to_measure,
                                k=args.k,
                                warmup=warm_cache,
                                batch_size=batch_size,
                                parallel_mode=parallel_mode,
                            )

                        # Run simulator (d/N from the per-dataset runner)
                        simulator_results = simulator_runner.simulate(
                            d=benchmark_runner.d,
                            N=benchmark_runner.nb,
                            nlist=idx_config["nlist"],
                            M=idx_config["M"],
                            nbits=idx_config["nbits"],
                            k=args.k,
                            nprobe_values=args.nprobe_values,
                            batch_size=batch_size,
                            by_residual=idx_config["by_residual"],
                            parallel_mode=parallel_mode,
                            index_type=idx_config["index_type"],
                            cell_imbalance=cell_imbalance,
                        )

                        # Compare
                        comparisons = compare_results(benchmark_results, simulator_results)
                        metrics = calculate_accuracy_metrics(comparisons)

                        # Per-point details with operator breakdown
                        breakdown_by_nprobe = {cost.config.nprobe: cost for _, cost in simulator_results}
                        for c in comparisons:
                            print(f"  nprobe={c.nprobe:3d}: "
                                  f"actual={c.actual_latency_ms:.4f}ms "
                                  f"pred={c.predicted_latency_ms:.4f}ms "
                                  f"err={c.latency_error_pct:+.1f}%")

                        print(f"\n  MAPE(latency): {metrics['mape_latency']:.2f}%  "
                              f"Max APE: {metrics['max_ape_latency']:.2f}%  "
                              f"Corr: {metrics['correlation']:.4f}")

                        all_metrics.append({"label": label, **metrics})
                        all_comparison_data.append({
                            "label": label,
                            "dataset": dataset,
                            "d": benchmark_runner.d,
                            "index_name": index_name,
                            "num_threads": num_threads,
                            "batch_size": batch_size,
                            "parallel_mode": parallel_mode,
                            "comparisons": comparisons,
                            "breakdown_by_nprobe": breakdown_by_nprobe,
                            "metrics": metrics,
                        })

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    for m in all_metrics:
        print(f"  {m['label']:40s}  MAPE={m['mape_latency']:.1f}%  MaxAPE={m['max_ape_latency']:.1f}%")

    overall_mape = np.mean([m["mape_latency"] for m in all_metrics])
    print(f"\n  Overall latency MAPE (per-config mean): {overall_mape:.2f}%")
    g = aggregate_metrics(all_comparison_data)
    if g:
        print(f"  GLOBAL ({g['n']} pts): lat MAPE {g['mape_latency']:.2f}%  qps MAPE {g['mape_qps']:.2f}%  "
              f"q-err med {g['qerror_median']:.2f}/p95 {g['qerror_p95']:.2f}/max {g['qerror_max']:.2f}  "
              f"Spearman {g['spearman_latency']:.4f}")

    # Write output files
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    txt_path = os.path.join(args.output_dir, "benchmark_report.txt")

    csv_existed = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    txt_existed = os.path.exists(txt_path) and os.path.getsize(txt_path) > 0

    config_info = {
        "datasets": args.datasets,
        "index_configs": args.index_configs,
        "nprobe_values": args.nprobe_values,
        "batch_sizes": args.batch_sizes,
        "hardware_preset": hardware_preset,
        "num_threads": thread_counts,
        "parallel_modes": args.parallel_modes,
        "cache_model": "warm" if warm_cache else "cold",
    }

    _write_csv(all_comparison_data, csv_path)
    report = _format_report(all_comparison_data, all_metrics, config_info=config_info)
    with open(txt_path, "a") as f:
        if txt_existed:
            f.write("\n\n")
        f.write(report)

    mode = "appended to" if csv_existed or txt_existed else "saved to"
    print(f"\n  Results {mode}:")
    print(f"    CSV:  {csv_path}")
    print(f"    Text: {txt_path}")


def replay_from_measured(
    measured_csv: str,
    imbalance_json: str,
    output_dir: str,
    *,
    hardware_preset: str = "epyc_9124",
    profile: str | None = None,
    k: int = 10,
) -> list[dict]:
    """Drive the full benchmark pipeline from MEASURED actuals (no re-measure).

    Runs the identical CM + comparison + writers as ``benchmark_main`` —
    ``simulate`` → ``compare_results`` → ``calculate_accuracy_metrics`` →
    ``_write_csv`` → ``_format_report`` — but takes the actual latencies from a
    CSV instead of ``FaissBenchmarkRunner``. Lets the infra reproduce the rich
    benchmark CSV + report offline from a frozen ground-truth campaign (the
    >100ms / large-batch / multi-thread tail costs hours to re-measure).

    Args:
        measured_csv: columns dataset, index_name, num_threads, batch_size,
            nprobe, actual_latency_ms (extra columns ignored).
        imbalance_json: index geometry {dataset: {d, N, imbalance: {nprobe: f}}}
            — the per-nprobe cell imbalance the CM needs (NOT measured latency).
        output_dir: where benchmark_results.csv + benchmark_report.txt are written.
        profile: calibration profile path (passed to build_hardware).
    Returns the per-config metrics list.
    """
    from .cli import build_hardware  # local import avoids cli<->this circular

    imb = json.load(open(imbalance_json))
    groups: "collections.OrderedDict" = collections.OrderedDict()
    for r in csv.DictReader(open(measured_csv)):
        key = (r["dataset"], int(r["num_threads"]), r["index_name"], int(r["batch_size"]))
        groups.setdefault(key, {})[int(r["nprobe"])] = float(r["actual_latency_ms"])

    runners: dict[int, SimulatorRunner] = {}
    all_comparison_data, all_metrics = [], []
    seen = collections.defaultdict(set)
    for (ds, thr, idx, bs), nps in groups.items():
        ic = parse_index_config(idx)
        nprobe_values = sorted(nps)
        if thr not in runners:
            runners[thr] = SimulatorRunner(
                build_hardware(hardware_preset, thr, profile), warm_cache=True)
        bench = [BenchmarkResult(index_name=idx, nprobe=n, recall_at_k=0.0,
                                 latency_ms=nps[n], qps=bs / nps[n] * 1000.0,
                                 num_queries=0, batch_size=bs) for n in nprobe_values]
        cell_imb = {n: imb[ds]["imbalance"][str(n)] for n in nprobe_values}
        sim = runners[thr].simulate(
            d=imb[ds]["d"], N=imb[ds]["N"], nlist=ic["nlist"], M=ic["M"],
            nbits=ic["nbits"], k=k, nprobe_values=nprobe_values, batch_size=bs,
            by_residual=ic["by_residual"], parallel_mode=0,
            index_type=ic["index_type"], cell_imbalance=cell_imb)
        comparisons = compare_results(bench, sim)
        metrics = calculate_accuracy_metrics(comparisons)
        label = f"{ds} d={imb[ds]['d']} t={thr} {idx} bs={bs}"
        all_metrics.append({"label": label, **metrics})
        all_comparison_data.append({
            "label": label, "dataset": ds, "d": imb[ds]["d"], "index_name": idx,
            "num_threads": thr, "batch_size": bs, "parallel_mode": 0,
            "comparisons": comparisons,
            "breakdown_by_nprobe": {c.config.nprobe: c for _, c in sim},
            "metrics": metrics})
        seen["datasets"].add(ds); seen["index_configs"].add(idx)
        seen["nprobe_values"].update(nprobe_values); seen["batch_sizes"].add(bs)
        seen["num_threads"].add(thr)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "benchmark_results.csv")
    txt_path = os.path.join(output_dir, "benchmark_report.txt")
    for p in (csv_path, txt_path):
        if os.path.exists(p):
            os.remove(p)
    _write_csv(all_comparison_data, csv_path)
    config_info = {
        "datasets": sorted(seen["datasets"]), "index_configs": sorted(seen["index_configs"]),
        "nprobe_values": sorted(seen["nprobe_values"]), "batch_sizes": sorted(seen["batch_sizes"]),
        "hardware_preset": hardware_preset, "num_threads": sorted(seen["num_threads"]),
        "parallel_modes": [0], "cache_model": "warm (replay from measured actuals)"}
    with open(txt_path, "w") as f:
        f.write(_format_report(all_comparison_data, all_metrics, config_info=config_info))
    overall = float(np.mean([m["mape_latency"] for m in all_metrics]))
    g = aggregate_metrics(all_comparison_data)
    print(f"  replay: {len(all_metrics)} config groups, per-config latency MAPE {overall:.2f}%")
    if g:
        print(f"  GLOBAL ({g['n']} pts): lat MAPE {g['mape_latency']:.2f}%  qps MAPE {g['mape_qps']:.2f}%  "
              f"q-err med {g['qerror_median']:.2f}/p95 {g['qerror_p95']:.2f}/max {g['qerror_max']:.2f}  "
              f"Spearman {g['spearman_latency']:.4f}")
    print(f"    CSV:  {csv_path}\n    Text: {txt_path}")
    return all_metrics


if __name__ == "__main__":
    benchmark_main()
