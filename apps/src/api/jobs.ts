import { apiFetch } from "./base";

export interface GlobalJob {
  pid: number;
  ppid: number | null;
  pgid: number | null;
  sid: number | null;
  kind: string;
  name: string;
  status: string;
  started_at: string;
  path: string;
  job_id?: string | null;
  command: string;
  argv: string[];
}

export interface GlobalJobsOverview {
  running_count: number;
  jobs: GlobalJob[];
  process_scan: {
    strategy: string;
    dedupe: string;
  };
}

export interface GlobalProcessSnapshot {
  pid: number;
  ppid: number | null;
  pgid: number | null;
  sid: number | null;
  depth: number;
  name: string;
  state: string;
  threads: number;
  rss_bytes: number;
  cpu_time_seconds: number;
  command: string;
  argv0: string;
}

export interface GlobalJobProcessDetail {
  pid: number;
  alive: boolean;
  scanned_at: string;
  summary: {
    total_processes: number;
    descendant_processes: number;
    total_rss_bytes: number;
    total_threads: number;
    total_cpu_time_seconds: number;
    max_depth: number;
  };
  root: GlobalProcessSnapshot | null;
  processes: GlobalProcessSnapshot[];
  collection: {
    mode: string;
    max_processes: number;
    truncated: boolean;
    cpu: string;
  };
}

export async function fetchGlobalJobsOverview(): Promise<GlobalJobsOverview> {
  return apiFetch<GlobalJobsOverview>("/api/jobs/overview");
}

export async function fetchGlobalJobProcessDetail(
  pid: number,
  maxProcesses = 160,
): Promise<GlobalJobProcessDetail> {
  return apiFetch<GlobalJobProcessDetail>(
    `/api/jobs/process/${encodeURIComponent(pid)}?max_processes=${encodeURIComponent(maxProcesses)}`,
  );
}
