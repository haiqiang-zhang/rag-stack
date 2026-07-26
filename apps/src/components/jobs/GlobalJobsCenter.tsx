import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Activity,
  ChevronDown,
  ChevronRight,
  Clock3,
  Cpu,
  HardDrive,
  Loader2,
  RefreshCw,
  TerminalSquare,
} from "lucide-react";

import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import { StatusPill } from "@/components/ui/StatusPill";
import {
  fetchGlobalJobProcessDetail,
  fetchGlobalJobsOverview,
  type GlobalJob,
  type GlobalJobProcessDetail,
  type GlobalJobsOverview,
  type GlobalProcessSnapshot,
} from "@/api/jobs";
import { cn } from "@/lib/utils";

function formatKind(kind: string): string {
  const normalized = kind.replace(/^cost_model_/, "cost model ");
  return normalized.replace(/[_/.-]+/g, " ").trim() || "job";
}

function shortCommand(job: GlobalJob): string {
  if (job.command) return job.command;
  if (job.argv.length) return job.argv.join(" ");
  return job.path || `PID ${job.pid}`;
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  let current = value;
  let unit = 0;
  while (current >= 1024 && unit < units.length - 1) {
    current /= 1024;
    unit += 1;
  }
  return `${current >= 10 || unit === 0 ? current.toFixed(0) : current.toFixed(1)} ${units[unit]}`;
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0s";
  if (seconds < 60) return `${seconds.toFixed(seconds >= 10 ? 0 : 1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60);
  if (minutes < 60) return `${minutes}m ${rest}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function formatScannedAt(value: string): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleTimeString();
}

function DetailMetric({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
}) {
  return (
    <div className="min-w-0 rounded-md border bg-background px-2.5 py-2">
      <div className="flex min-w-0 items-center gap-1.5 text-[10px] font-medium uppercase text-muted-foreground">
        <Icon className="size-3 shrink-0" />
        <span className="min-w-0 truncate">{label}</span>
      </div>
      <div className="mt-1 truncate font-mono text-sm font-semibold tabular-nums text-foreground">
        {value}
      </div>
    </div>
  );
}

function JobMetaChip({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex min-w-0 max-w-full items-center overflow-hidden rounded-md border bg-background px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
      <span className="min-w-0 max-w-full truncate">{children}</span>
    </span>
  );
}

interface ProcessTreeNode {
  process: GlobalProcessSnapshot;
  children: ProcessTreeNode[];
}

interface VisibleProcessNode {
  node: ProcessTreeNode;
  treeDepth: number;
}

function buildProcessTree(processes: GlobalProcessSnapshot[]): ProcessTreeNode[] {
  const byPid = new Map<number, ProcessTreeNode>();
  for (const process of processes) {
    byPid.set(process.pid, { process, children: [] });
  }

  const roots: ProcessTreeNode[] = [];
  for (const process of processes) {
    const node = byPid.get(process.pid);
    if (!node) continue;
    const parent = process.ppid == null ? null : byPid.get(process.ppid);
    if (parent && parent.process.pid !== process.pid) {
      parent.children.push(node);
    } else {
      roots.push(node);
    }
  }

  return roots;
}

function flattenProcessTree(
  nodes: ProcessTreeNode[],
  expanded: Set<number>,
  depth = 0,
): VisibleProcessNode[] {
  const result: VisibleProcessNode[] = [];
  for (const node of nodes) {
    result.push({ node, treeDepth: depth });
    if (node.children.length > 0 && expanded.has(node.process.pid)) {
      result.push(...flattenProcessTree(node.children, expanded, depth + 1));
    }
  }
  return result;
}

function expandableProcessIds(nodes: ProcessTreeNode[]): number[] {
  const ids: number[] = [];
  for (const node of nodes) {
    if (node.children.length > 0) ids.push(node.process.pid);
    ids.push(...expandableProcessIds(node.children));
  }
  return ids;
}

function ProcessRow({
  item,
  expanded,
  onToggle,
}: {
  item: VisibleProcessNode;
  expanded: boolean;
  onToggle: (pid: number) => void;
}) {
  const { node, treeDepth } = item;
  const { process } = node;
  const isRoot = process.depth === 0;
  const kind = isRoot ? "root" : "sub";
  const hasChildren = node.children.length > 0;

  return (
    <tr className="border-t align-top hover:bg-muted/30">
      <td className="px-3 py-2">
        <div
          className="flex items-center gap-2 whitespace-nowrap"
          style={{ paddingLeft: `${treeDepth * 18}px` }}
        >
          {hasChildren ? (
            <button
              type="button"
              onClick={() => onToggle(process.pid)}
              className="flex size-5 shrink-0 items-center justify-center rounded-sm text-muted-foreground hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              aria-label={expanded ? "Collapse process" : "Expand process"}
            >
              {expanded ? (
                <ChevronDown className="size-3.5" />
              ) : (
                <ChevronRight className="size-3.5" />
              )}
            </button>
          ) : (
            <span className="size-5 shrink-0" />
          )}
          <span
            className={cn(
              "rounded-sm px-1.5 py-0.5 text-[9px] font-medium uppercase",
              isRoot ? "bg-primary/10 text-primary" : "bg-muted text-muted-foreground",
            )}
          >
            {kind}
          </span>
          <span className="text-xs font-medium text-foreground">{process.name}</span>
          {hasChildren && (
            <span className="text-[10px] text-muted-foreground">
              {node.children.length} child{node.children.length === 1 ? "" : "ren"}
            </span>
          )}
        </div>
        {process.command && (
          <div
            className="mt-0.5 whitespace-nowrap font-mono text-[10px] leading-4 text-muted-foreground"
            style={{ paddingLeft: `${treeDepth * 18 + 28}px` }}
          >
            {process.command}
          </div>
        )}
      </td>
      <td className="whitespace-nowrap px-3 py-2 font-mono text-[11px] font-medium tabular-nums text-foreground">
        {process.pid}
      </td>
      <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-[11px] tabular-nums text-muted-foreground">
        {process.state.split(/\s+/)[0] || "-"}
      </td>
      <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-[11px] tabular-nums text-muted-foreground">
        {formatBytes(process.rss_bytes)}
      </td>
      <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-[11px] tabular-nums text-muted-foreground">
        {formatDuration(process.cpu_time_seconds)}
      </td>
      <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-[11px] tabular-nums text-muted-foreground">
        {process.threads}
      </td>
      <td className="whitespace-nowrap px-3 py-2 text-right font-mono text-[11px] tabular-nums text-muted-foreground">
        {process.ppid ?? "-"}
      </td>
    </tr>
  );
}

function ProcessDetailsContent({
  detail,
  loading,
  error,
  onRefresh,
}: {
  detail: GlobalJobProcessDetail | null;
  loading: boolean;
  error: string | null;
  onRefresh: () => void;
}) {
  const processTree = useMemo(
    () => buildProcessTree(detail?.processes ?? []),
    [detail?.processes],
  );
  const expandableIds = useMemo(
    () => expandableProcessIds(processTree),
    [processTree],
  );
  const [expandedPids, setExpandedPids] = useState<Set<number>>(new Set());

  useEffect(() => {
    setExpandedPids(new Set(expandableIds));
  }, [detail?.pid, detail?.scanned_at, expandableIds]);

  const visibleProcesses = useMemo(
    () => flattenProcessTree(processTree, expandedPids),
    [expandedPids, processTree],
  );

  const toggleProcess = useCallback((pid: number) => {
    setExpandedPids((current) => {
      const next = new Set(current);
      if (next.has(pid)) {
        next.delete(pid);
      } else {
        next.add(pid);
      }
      return next;
    });
  }, []);

  const expandAll = useCallback(() => {
    setExpandedPids(new Set(expandableIds));
  }, [expandableIds]);

  const collapseAll = useCallback(() => {
    setExpandedPids(new Set());
  }, []);

  if (loading && !detail) {
    return (
      <div className="rounded-md border bg-muted/20 px-3 py-8">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <Loader2 className="size-3.5 animate-spin" />
          Loading process snapshot
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
        {error}
      </div>
    );
  }

  if (!detail) return null;

  if (!detail.alive) {
    return (
      <div className="rounded-md border bg-muted/30 px-3 py-6 text-sm text-muted-foreground">
        Process {detail.pid} is no longer running.
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 w-full min-w-0 max-w-full flex-col gap-3 overflow-hidden">
      <div className="grid min-w-0 grid-cols-2 gap-2 lg:grid-cols-5">
        <DetailMetric
          icon={TerminalSquare}
          label="Processes"
          value={`${detail.summary.total_processes}`}
        />
        <DetailMetric
          icon={Activity}
          label="Sub proc"
          value={`${detail.summary.descendant_processes}`}
        />
        <DetailMetric
          icon={HardDrive}
          label="RSS"
          value={formatBytes(detail.summary.total_rss_bytes)}
        />
        <DetailMetric
          icon={Cpu}
          label="CPU time"
          value={formatDuration(detail.summary.total_cpu_time_seconds)}
        />
        <DetailMetric
          icon={Activity}
          label="Threads"
          value={`${detail.summary.total_threads}`}
        />
      </div>

      <div className="flex min-w-0 items-center justify-between gap-3 rounded-md border bg-muted/20 px-3 py-2">
        <div className="min-w-0 text-[11px] text-muted-foreground">
          <span className="font-medium text-foreground">Snapshot</span>
          <span className="mx-2 text-muted-foreground/70">/</span>
          <span className="font-mono tabular-nums">{formatScannedAt(detail.scanned_at)}</span>
          <span className="mx-2 text-muted-foreground/70">/</span>
          <span>{detail.collection.truncated ? "truncated" : "complete"}</span>
          <span className="mx-2 text-muted-foreground/70">/</span>
          <span>max depth {detail.summary.max_depth}</span>
        </div>
        <Button
          type="button"
          variant="outline"
          size="xs"
          onClick={onRefresh}
          disabled={loading}
          className="h-7 shrink-0 px-2 text-[11px]"
        >
          <RefreshCw className={cn("size-3", loading && "animate-spin")} />
          Refresh
        </Button>
      </div>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-md border bg-card">
        <div className="flex min-w-0 items-center justify-between gap-3 border-b px-3 py-2">
          <div className="min-w-0">
            <div className="truncate text-xs font-medium text-foreground">
              Process tree · {detail.summary.descendant_processes} sub proc
            </div>
            <div className="truncate text-[11px] text-muted-foreground">
              Parent-linked snapshot from /proc; CPU is cumulative, no sampling.
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-1">
            <Button
              type="button"
              variant="ghost"
              size="xs"
              onClick={expandAll}
              disabled={expandableIds.length === 0}
              className="h-6 px-1.5 text-[11px]"
            >
              Expand all
            </Button>
            <Button
              type="button"
              variant="ghost"
              size="xs"
              onClick={collapseAll}
              disabled={expandableIds.length === 0}
              className="h-6 px-1.5 text-[11px]"
            >
              Collapse all
            </Button>
          </div>
        </div>
        <div className="min-h-0 min-w-0 flex-1 overflow-auto">
          <table className="w-max min-w-full border-collapse text-left">
            <thead className="sticky top-0 z-10 bg-muted/80 backdrop-blur">
              <tr className="border-b text-[10px] font-medium uppercase text-muted-foreground">
                <th className="min-w-[40rem] whitespace-nowrap px-3 py-1.5">
                  Process / command
                </th>
                <th className="w-24 whitespace-nowrap px-3 py-1.5">PID</th>
                <th className="w-24 whitespace-nowrap px-3 py-1.5 text-right">State</th>
                <th className="w-28 whitespace-nowrap px-3 py-1.5 text-right">RSS</th>
                <th className="w-24 whitespace-nowrap px-3 py-1.5 text-right">CPU</th>
                <th className="w-20 whitespace-nowrap px-3 py-1.5 text-right">Threads</th>
                <th className="w-24 whitespace-nowrap px-3 py-1.5 text-right">PPID</th>
              </tr>
            </thead>
            <tbody>
              {visibleProcesses.map((item) => (
                <ProcessRow
                  key={item.node.process.pid}
                  item={item}
                  expanded={expandedPids.has(item.node.process.pid)}
                  onToggle={toggleProcess}
                />
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function JobCard({
  job,
  onOpen,
}: {
  job: GlobalJob;
  onOpen: (job: GlobalJob) => void;
}) {
  const command = shortCommand(job);
  const kindLabel = formatKind(job.kind);

  return (
    <button
      type="button"
      onClick={() => onOpen(job)}
      className="group w-full min-w-0 max-w-full overflow-hidden rounded-md border bg-card p-3 text-left transition-colors hover:bg-muted/35 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      <div className="flex w-full min-w-0 max-w-full items-start gap-3 overflow-hidden">
        <div className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
          <TerminalSquare className="size-4" />
        </div>
        <div className="min-w-0 flex-1 overflow-hidden">
          <div className="flex min-w-0 max-w-full items-center gap-2 overflow-hidden">
            <span className="min-w-0 flex-1 truncate text-sm font-medium">
              {job.name || formatKind(job.kind)}
            </span>
            <span className="shrink-0">
              <StatusPill status={job.status || "running"} showLabel={false} />
            </span>
          </div>
          <div className="mt-1 flex w-full min-w-0 max-w-full flex-wrap items-center gap-1.5 overflow-hidden">
            <JobMetaChip>
              <span className="tabular-nums">PID {job.pid}</span>
            </JobMetaChip>
            <JobMetaChip>{kindLabel}</JobMetaChip>
            {job.job_id && <JobMetaChip>{job.job_id}</JobMetaChip>}
          </div>
          {job.started_at && (
            <div className="mt-2 flex min-w-0 items-center gap-1.5 overflow-hidden text-[11px] text-muted-foreground">
              <Clock3 className="size-3 shrink-0" />
              <span className="min-w-0 truncate">{job.started_at}</span>
            </div>
          )}
          {command && (
            <div className="mt-2 min-w-0 truncate rounded-md bg-muted/30 px-2.5 py-1.5 font-mono text-[11px] leading-5 text-muted-foreground">
              {command}
            </div>
          )}
          <div className="mt-2 text-[11px] font-medium text-primary opacity-80 group-hover:opacity-100">
            Open process details
          </div>
        </div>
      </div>
    </button>
  );
}

function JobProcessDialog({
  job,
  detail,
  loading,
  error,
  onClose,
  onRefresh,
}: {
  job: GlobalJob | null;
  detail: GlobalJobProcessDetail | null;
  loading: boolean;
  error: string | null;
  onClose: () => void;
  onRefresh: () => void;
}) {
  const kindLabel = job ? formatKind(job.kind) : "";

  return (
    <Dialog open={Boolean(job)} onOpenChange={(nextOpen) => !nextOpen && onClose()}>
      <DialogContent className="flex h-[min(46rem,calc(100vh-2rem))] w-[min(72rem,calc(100vw-2rem))] max-w-none min-w-0 flex-col gap-0 overflow-hidden p-0 sm:max-w-none">
        <DialogHeader className="min-w-0 max-w-full overflow-hidden border-b p-4 pr-12 text-left">
          <div className="flex min-w-0 items-start gap-3 overflow-hidden">
            <div className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-md bg-primary/10 text-primary">
              <TerminalSquare className="size-4" />
            </div>
            <div className="min-w-0 flex-1 overflow-hidden">
              <DialogTitle className="truncate text-base">
                {job?.name || (job ? formatKind(job.kind) : "Process details")}
              </DialogTitle>
              <DialogDescription className="mt-1 truncate">
                PID {job?.pid ?? "-"} · {kindLabel || "job"} · on-demand snapshot, no sampling
              </DialogDescription>
              <div className="mt-2 flex w-full min-w-0 max-w-full flex-wrap items-center gap-1.5 overflow-hidden">
                {job && (
                  <>
                    <JobMetaChip>
                      <span className="tabular-nums">PID {job.pid}</span>
                    </JobMetaChip>
                    <JobMetaChip>{kindLabel}</JobMetaChip>
                    {job.job_id && <JobMetaChip>{job.job_id}</JobMetaChip>}
                  </>
                )}
              </div>
            </div>
          </div>
        </DialogHeader>

        <div className="min-h-0 min-w-0 flex-1 overflow-hidden p-4">
          <ProcessDetailsContent
            detail={detail}
            loading={loading}
            error={error}
            onRefresh={onRefresh}
          />
        </div>
      </DialogContent>
    </Dialog>
  );
}

export function GlobalJobsCenter() {
  const [open, setOpen] = useState(false);
  const [overview, setOverview] = useState<GlobalJobsOverview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<GlobalJob | null>(null);
  const [detail, setDetail] = useState<GlobalJobProcessDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      setOverview(await fetchGlobalJobsOverview());
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    const interval = window.setInterval(load, 5000);
    return () => window.clearInterval(interval);
  }, [load]);

  const loadDetail = useCallback(async (job: GlobalJob) => {
    try {
      setDetailLoading(true);
      setDetailError(null);
      setDetail(await fetchGlobalJobProcessDetail(job.pid));
    } catch (err) {
      setDetailError(err instanceof Error ? err.message : String(err));
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const openJobDetail = useCallback(
    (job: GlobalJob) => {
      setSelectedJob(job);
      setDetail(null);
      setDetailError(null);
      void loadDetail(job);
    },
    [loadDetail],
  );

  const closeJobDetail = useCallback(() => {
    setSelectedJob(null);
    setDetail(null);
    setDetailError(null);
    setDetailLoading(false);
  }, []);

  const refreshSelectedDetail = useCallback(() => {
    if (selectedJob) {
      void loadDetail(selectedJob);
    }
  }, [loadDetail, selectedJob]);

  const runningCount = overview?.running_count ?? 0;
  const jobs = useMemo(() => overview?.jobs ?? [], [overview]);
  const jobsTooltip =
    runningCount > 0
      ? `${runningCount} job${runningCount === 1 ? "" : "s"} running`
      : "Jobs";

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SidebarMenuItem>
        <SheetTrigger asChild>
          <SidebarMenuButton
            tooltip={jobsTooltip}
            className={cn(
              "relative",
              runningCount > 0 && "text-primary hover:text-primary",
            )}
          >
            <Activity className={runningCount > 0 ? "text-primary" : ""} />
            <span>Jobs</span>
            {runningCount > 0 && (
              <>
                <span className="ml-auto rounded-md bg-primary px-1.5 py-0.5 text-[10px] font-medium text-primary-foreground group-data-[collapsible=icon]:hidden">
                  {runningCount}
                </span>
                {/* Collapsed rail: the count pill would sit on the centered
                    icon (32px button), so show a corner dot instead — the
                    tooltip (only visible when collapsed) carries the count. */}
                <span
                  aria-hidden="true"
                  className="absolute right-1 top-1 hidden size-2 rounded-full bg-primary ring-2 ring-sidebar group-data-[collapsible=icon]:block"
                />
              </>
            )}
          </SidebarMenuButton>
        </SheetTrigger>
      </SidebarMenuItem>

      <SheetContent className="box-border w-[min(42rem,calc(100vw-1rem))] max-w-none min-w-0 gap-0 overflow-hidden p-0 sm:max-w-none">
        <SheetHeader className="min-w-0 max-w-full overflow-hidden border-b p-4">
          <div className="flex items-start justify-between gap-3 pr-8">
            <div className="min-w-0">
              <SheetTitle>Jobs</SheetTitle>
              <SheetDescription>
                Running RAG-Stack main processes. Forked child processes are deduplicated.
              </SheetDescription>
            </div>
            <Button
              type="button"
              variant="outline"
              size="icon"
              onClick={load}
              disabled={loading}
              aria-label="Refresh jobs"
              className="size-8"
            >
              <RefreshCw className={cn("size-4", loading && "animate-spin")} />
            </Button>
          </div>
        </SheetHeader>

        <div className="min-w-0 max-w-full overflow-hidden border-b px-4 py-3">
          <div className="flex min-w-0 items-center gap-2 overflow-hidden">
            <Badge variant={runningCount > 0 ? "default" : "secondary"}>
              {runningCount} running
            </Badge>
            {overview?.process_scan && (
              <span className="min-w-0 truncate text-xs text-muted-foreground">
                main processes only
              </span>
            )}
          </div>
        </div>

        <div className="min-h-0 min-w-0 flex-1 overflow-y-auto overflow-x-hidden">
          <div className="w-full min-w-0 max-w-full space-y-3 overflow-hidden p-4">
            {error && (
              <div className="w-full min-w-0 max-w-full overflow-hidden rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
                {error}
              </div>
            )}
            {jobs.length === 0 && !error ? (
              <div className="w-full min-w-0 max-w-full rounded-md border bg-muted/30 px-3 py-6 text-center text-sm text-muted-foreground">
                No running jobs
              </div>
            ) : (
              jobs.map((job) => (
                <JobCard
                  key={`${job.pid}-${job.kind}`}
                  job={job}
                  onOpen={openJobDetail}
                />
              ))
            )}
          </div>
        </div>
      </SheetContent>

      <JobProcessDialog
        job={selectedJob}
        detail={detail}
        loading={detailLoading}
        error={detailError}
        onClose={closeJobDetail}
        onRefresh={refreshSelectedDetail}
      />
    </Sheet>
  );
}
