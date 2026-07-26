export interface OptimizerInfo {
  key: string;
  class_name: string;
  description: string;
}

export interface OptimizerBenchmarkRow {
  iteration: number;
  cumulative_gt_calls: number;
  cumulative_cm_calls: number;
  cumulative_optimizer_time_s: number;
  hv_diff: number;
  igd: number;
  get_pareto_time_s: number;
  optimizer: string;
  synthetic_function: string;
  seed: number;
}

export interface OptimizerBenchmarkSummary {
  optimizer: string;
  synthetic_function: string;
  /** Number of seeds aggregated into this row. */
  n_seeds: number;
  /** Sorted list of the actual seeds — surfaced in the SummaryTable's
   *  Seeds-cell tooltip so users can see which seeds were run. */
  seeds: number[];
  hv_diff_mean: number;
  hv_diff_std: number;
  igd_mean: number;
  igd_std: number;
  gt_calls: number;
  gt_calls_std: number;
  optimizer_time_s: number;
  optimizer_time_s_std: number;
}

export interface OptimizerEvalPoint {
  seed: number;
  iteration: number;
  /** Monotonic GT-call budget axis used by the progress slider. */
  cumulative_gt_calls: number;
  /** Ground-truth values keyed by objective_name (matches the function's
   *  `objectives()` declaration). Includes every objective on every eval —
   *  cheap-only iters still get the full reading because the runner re-evals
   *  the suggested config through `function.evaluate(...)`. */
  obj_values: Record<string, number>;
  phase: string;
}

export interface OptimizerEvalDistribution {
  name: string;
  evals: OptimizerEvalPoint[];
}

export interface EvalDistributionPayload {
  function: string;
  /** Canonical objective ordering (= `function.objectives()` order on the
   *  server). UI uses obj_names[0] as x-axis, obj_names[1] as y-axis. */
  obj_names: string[];
  seeds: number[];
  /** One entry per Pareto point, dict-keyed by objective_name. Empty when
   *  the true Pareto is still being computed in the background
   *  (`true_pareto_pending` is true) or when computation failed
   *  (`true_pareto_error` is set). */
  true_pareto: Array<Record<string, number>>;
  /** Non-null when true Pareto computation was skipped — e.g. the
   *  synthetic function couldn't be instantiated. Surfaced as a small
   *  note next to the chart. */
  true_pareto_error: string | null;
  /** True when the server is still computing the Pareto in a background
   *  thread (the heavy load is async). Client should poll until false. */
  true_pareto_pending: boolean;
  optimizers: OptimizerEvalDistribution[];
}

export interface BenchmarkConfig {
  filename: string;
  optimizer_type: string;
  synthetic_function: string;
  n_iterations: number;
  seed: number;
  output_dir: string;
  content: Record<string, unknown>;
}

export interface BenchmarkJob {
  job_id: string;
  config_path: string;
  config: Record<string, unknown>;
  pid: number;
  status: "running" | "completed" | "failed";
  started_at: string;
  finished_at: string | null;
  error: string | null;
  output_dir: string;
}
