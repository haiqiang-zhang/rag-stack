export type CostModelFamily = "llm_sim" | "retrieval" | "other";

/** Cost model registry entry from the API. */
export interface CostModelInfo {
  name: string;
  class_name: string;
  description: string;
  family: CostModelFamily;
  mapped_module_types: string[];
  capabilities: string[];
  extensions: EndpointInfo[];
}

/** Read-only descriptors owned by RAG-CM; never a compiled project plan. */
export interface CostModelBinding {
  component: string;
  dispatch_stage: string;
  db_type: string | null;
  cost_model: string | null;
  status: "priced" | "unpriced" | "unsupported";
  reason: string | null;
  execution_kind: "llm" | "forward" | null;
  performance_stages: { name: string; kind: string; resource: string }[];
  resources: string[];
}

export interface CostModelCatalog {
  schema_version: "rag-stack.cost-model-catalog.v1";
  read_only: true;
  scope: "dispatch_catalog";
  stages: {
    stage: string;
    user_selectable: boolean;
    bindings: CostModelBinding[];
  }[];
  contract: {
    rag_ir: string;
    compiler: string;
    compiled_plan_schema: string;
    readiness: string;
  };
}

export interface EndpointInfo {
  path: string;
  method: string;
  summary: string;
}

/** Calibration profile JSON (faiss_ivf). */
export interface CalibrationProfile {
  filename: string;
  profile_kind: "faiss_ivf_exact_cpu";
  evidence_scope: string;
  fit_method: string;
  hardware_preset: string;
  mape_latency: number;
  max_ape_latency: number;
  timestamp: string;
  index_configs: string[];
  batch_sizes: number[];
  fitted_fields: string[];
  source_path: string;
  source_sha256: string;
  raw_hardware_source: string;
  raw_hardware_source_sha256: string;
  num_trials: number;
  blas_efficiency: number;
  simd_efficiency: number;
  pq_fastscan_efficiency: number;
  pq_fastscan_batch_efficiency: number;
  pq_standard_efficiency: number;
  pq_fastscan_per_probe_overhead_us: number;
  pq_standard_per_probe_overhead_us: number;
  pq_rap_standard_c1: number;
  pq_rap_standard_c2: number;
  pq_rap_standard_c3: number;
  scattered_bw_ceiling: number;
  numa_local_bytes: number;
  flat_per_probe_overhead_us: number;
  flat_scan_efficiency: number;
  flat_fixed_overhead_us: number;
  pq_setup_per_dim_us: number;
  pq_fastscan_qbs_floor: number;
  pq_fastscan_overhead_alpha: number;
  pq_fastscan_scan_bw_ceiling: number;
  [key: string]: unknown;
}

/** Benchmark comparison row (faiss_ivf). */
export interface BenchmarkRow {
  index_name: string;
  batch_size: number;
  parallel_mode: number;
  nprobe: number;
  actual_latency_ms: number;
  predicted_latency_ms: number;
  latency_error_pct: number;
  actual_qps: number;
  predicted_qps: number;
  qps_error_pct: number;
  [key: string]: unknown;
}

/** Benchmark summary per index config group. */
export interface BenchmarkSummary {
  label: string;
  mape_latency: number;
  max_ape_latency: number;
  num_points: number;
}

/** Hardware preset (faiss_ivf). */
export interface HardwarePreset {
  name: string;
  description: string;
}

/** A single point in the calibration grid (measured vs predicted). */
export interface CalibrationGridPoint {
  batch_size: number;
  seq_len: number;
  measured_s: number;
  predicted_s: number;
  rel_err: number;
}

/** Calibration profile JSON (one CalibrationRecord). */
export interface CalibrationProfileRecord {
  filename: string;
  component: string;
  model_key: string;
  hardware_key: string;
  bits: string;
  compute_efficiency?: number | null;
  memory_efficiency?: number | null;
  fit_mape: number;
  correlation?: number | null;
  cuda_name?: string;
  timestamp?: string;
  grid: CalibrationGridPoint[];
  error?: string;
  can_delete?: boolean;
  delete_reason?: string;
  record_key?: string;
  match_scope?: string;
  linear_law?: unknown;
}

/** Background benchmark / calibration job record. */
export interface CalibrationJob {
  job_id: string;
  job_type: string;       // "cost_model_benchmark" | "cost_model_calibrate"
  status: "running" | "completed" | "failed";
  started_at: string;
  finished_at: string | null;
  error: string | null;
  output_dir: string;
  pid?: number;
  config: {
    cost_model?: string;
    kind?: "benchmark" | "calibrate";
    module?: string;
    config_path?: string;
    batch_sizes?: number[];
    seq_lens?: number[];
    warmup?: number;
    measurements?: number;
    calibrate_command?: string;
    cli_args?: string[];
    command?: string;
    [key: string]: unknown;
  };
}

export interface CalibrationRunRequest {
  kind: "benchmark" | "calibrate";
  module: string;
  config_path: string;
  batch_sizes: number[];
  seq_lens: number[];
  warmup: number;
  measurements: number;
}

export type UnifiedCalibrationAxis = "system" | "corpus_embedding";

export interface UnifiedCalibrationProfile {
  axis: UnifiedCalibrationAxis;
  component: string;
  kind: string;
  key: string;
  corpus: string;
  embedding: string;
  system: string;
  filename: string;
  relative_path: string;
  path: string;
  size_bytes: number;
  modified_at: string;
  record_count: number;
}

export interface CalibrationDataFile {
  component: string;
  system: string;
  format: string;
  filename: string;
  relative_path: string;
  path: string;
  size_bytes: number;
  modified_at: string;
}

export interface UnifiedCalibrationStatus {
  scope: "artifact_inventory";
  profiles_root: string;
  configured_profiles_root?: string | null;
  root_note?: string;
  data_root: string;
  commands: CalibrationCommandOption[];
  profiles: UnifiedCalibrationProfile[];
  data_files: CalibrationDataFile[];
  summary: {
    profiles: number;
    system_profiles: number;
    corpus_embedding_profiles: number;
    data_files: number;
  };
}

export interface CalibrationCommandOption {
  value: string;
  label: string;
  needs_config: boolean;
  supports_force: boolean;
  supports_maxiter: boolean;
  supports_data_paths: boolean;
  requires_project_dir: boolean;
  supports_data_dir: boolean;
  supports_profile_dir: boolean;
  read_only: boolean;
  required_extra_args: string[];
  extra_placeholder: string;
}

export interface UnifiedCalibrationProfileDetail {
  profile: UnifiedCalibrationProfile;
  content_type: "json" | "binary";
  content: unknown;
  error?: string;
}

export interface UnifiedCalibrationRunRequest {
  command: string;
  config_path: string;
  force: boolean;
  maxiter?: number | null;
  qa_data?: string;
  corpus_data?: string;
  project_dir?: string;
  data_dir?: string;
  profile_dir?: string;
  extra_args: string[];
}

export interface CalibrationJobLog {
  job: CalibrationJob;
  log: string;
  log_path: string;
}

/** Estimate API response. */
export interface OperatorBreakdown {
  name: string;
  compute_time_s: number;
  memory_time_s: number;
  time_s: number;
  bottleneck: string;
}

export interface EstimateResponse {
  scope?: "component_estimate";
  latency_s: number;
  throughput_qps: number;
  operators: OperatorBreakdown[];
}
