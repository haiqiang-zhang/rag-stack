import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  Database,
  FileText,
  Loader2,
  Play,
  RefreshCw,
  Square,
  Terminal,
} from "lucide-react";

import {
  fetchUnifiedCalibrationProfileDetail,
  fetchUnifiedCalibrationJobLog,
  fetchUnifiedCalibrationJobs,
  fetchUnifiedCalibrationStatus,
  killJob,
  runUnifiedCalibrationJob,
} from "@/api/costModelClient";
import type {
  CalibrationDataFile,
  CalibrationJob,
  UnifiedCalibrationProfile,
  UnifiedCalibrationProfileDetail,
  UnifiedCalibrationStatus,
} from "@/types/costModel";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

function formatBytes(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = n;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

function formatDate(s: string): string {
  if (!s) return "";
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return s;
  return d.toLocaleString();
}

function splitArgs(value: string): string[] {
  return value
    .split(/\s+/)
    .map((x) => x.trim())
    .filter(Boolean);
}

function statusBadge(job: CalibrationJob) {
  if (job.status === "completed") return <Badge variant="secondary">completed</Badge>;
  if (job.status === "failed") return <Badge variant="destructive">failed</Badge>;
  return <Badge variant="outline">running</Badge>;
}

function jobLabel(job: CalibrationJob): string {
  const cfg = job.config as unknown as Record<string, unknown>;
  if (cfg.calibrate_command) return String(cfg.calibrate_command);
  const cliArgs = cfg.cli_args;
  if (Array.isArray(cliArgs) && cliArgs.length > 0) return cliArgs.join(" ");
  if (cfg.command) return String(cfg.command);
  const target = String(cfg.target ?? cfg.cost_model ?? "calibrate");
  const mode = String(cfg.mode ?? cfg.kind ?? "");
  return mode ? `${target} · ${mode}` : target;
}

function SummaryTile({ label, value }: { label: string; value: number | string }) {
  return (
    <Card className="min-w-[140px] flex-1 gap-1 py-3">
      <CardContent className="px-4">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
        <span className="mt-1 block text-xl font-semibold tabular-nums text-foreground">{value}</span>
      </CardContent>
    </Card>
  );
}

type ProfileSortKey =
  | "modified_desc"
  | "modified_asc"
  | "component_asc"
  | "kind_asc"
  | "key_asc"
  | "size_desc";

type ProfileGroupKey = "none" | "component" | "kind";

function profileText(p: UnifiedCalibrationProfile): string {
  return [
    p.axis,
    p.component,
    p.kind,
    p.key,
    p.corpus,
    p.embedding,
    p.system,
    p.filename,
    p.path,
  ].join(" ").toLowerCase();
}

function uniqueProfileValues(
  profiles: UnifiedCalibrationProfile[],
  field: "component" | "kind",
): string[] {
  return Array.from(new Set(profiles.map((p) => p[field]).filter(Boolean))).sort();
}

function sortProfiles(
  rows: UnifiedCalibrationProfile[],
  sortBy: ProfileSortKey,
): UnifiedCalibrationProfile[] {
  const sorted = [...rows];
  sorted.sort((a, b) => {
    if (sortBy === "modified_desc") {
      return Date.parse(b.modified_at) - Date.parse(a.modified_at);
    }
    if (sortBy === "modified_asc") {
      return Date.parse(a.modified_at) - Date.parse(b.modified_at);
    }
    if (sortBy === "component_asc") {
      return a.component.localeCompare(b.component) || a.key.localeCompare(b.key);
    }
    if (sortBy === "kind_asc") {
      return a.kind.localeCompare(b.kind) || a.key.localeCompare(b.key);
    }
    if (sortBy === "size_desc") {
      return b.size_bytes - a.size_bytes;
    }
    return a.key.localeCompare(b.key);
  });
  return sorted;
}

function groupProfiles(
  rows: UnifiedCalibrationProfile[],
  groupBy: ProfileGroupKey,
): Array<{ key: string; rows: UnifiedCalibrationProfile[] }> {
  if (groupBy === "none") return [{ key: "", rows }];
  const groups = new Map<string, UnifiedCalibrationProfile[]>();
  for (const row of rows) {
    const key = groupBy === "component" ? row.component : row.kind;
    const label = key || "unknown";
    groups.set(label, [...(groups.get(label) ?? []), row]);
  }
  return Array.from(groups.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, groupedRows]) => ({ key, rows: groupedRows }));
}

function ProfileTable({
  title,
  axis,
  profiles,
  onOpenProfile,
}: {
  title: string;
  axis: UnifiedCalibrationProfile["axis"];
  profiles: UnifiedCalibrationProfile[];
  onOpenProfile: (profile: UnifiedCalibrationProfile) => void;
}) {
  const [search, setSearch] = useState("");
  const [component, setComponent] = useState("all");
  const [kind, setKind] = useState("all");
  const [sortBy, setSortBy] = useState<ProfileSortKey>("modified_desc");
  const [groupBy, setGroupBy] = useState<ProfileGroupKey>("component");

  const componentOptions = useMemo(
    () => uniqueProfileValues(profiles, "component"),
    [profiles],
  );
  const kindOptions = useMemo(
    () => uniqueProfileValues(profiles, "kind"),
    [profiles],
  );

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const filtered = profiles.filter((p) => {
      if (component !== "all" && p.component !== component) return false;
      if (kind !== "all" && p.kind !== kind) return false;
      if (needle && !profileText(p).includes(needle)) return false;
      return true;
    });
    return sortProfiles(filtered, sortBy);
  }, [component, kind, profiles, search, sortBy]);

  const grouped = useMemo(
    () => groupProfiles(visible, groupBy),
    [groupBy, visible],
  );

  const colSpan = 7;
  const isSystem = axis === "system";

  return (
    <Card>
      <CardHeader className="border-b">
        <CardTitle className="flex items-center gap-2 text-sm">
          <FileText className="size-4 text-primary" />
          {title}
          <Badge variant="secondary">{visible.length}/{profiles.length}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="grid gap-2 lg:grid-cols-[minmax(220px,1fr)_160px_160px_160px_160px]">
          <Input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search profiles..."
          />
          <Select value={component} onValueChange={setComponent}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Component" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All components</SelectItem>
              {componentOptions.map((value) => (
                <SelectItem key={value} value={value}>{value}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={kind} onValueChange={setKind}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Kind" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All kinds</SelectItem>
              {kindOptions.map((value) => (
                <SelectItem key={value} value={value}>{value}</SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={sortBy} onValueChange={(v) => setSortBy(v as ProfileSortKey)}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Sort" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="modified_desc">Newest first</SelectItem>
              <SelectItem value="modified_asc">Oldest first</SelectItem>
              <SelectItem value="component_asc">Component A-Z</SelectItem>
              <SelectItem value="kind_asc">Kind A-Z</SelectItem>
              <SelectItem value="key_asc">Key A-Z</SelectItem>
              <SelectItem value="size_desc">Largest first</SelectItem>
            </SelectContent>
          </Select>
          <Select value={groupBy} onValueChange={(v) => setGroupBy(v as ProfileGroupKey)}>
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Group" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">No grouping</SelectItem>
              <SelectItem value="component">Group by component</SelectItem>
              <SelectItem value="kind">Group by kind</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardContent>
      <CardContent className="px-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Component</TableHead>
              <TableHead>Kind</TableHead>
              {isSystem ? (
                <TableHead>System</TableHead>
              ) : (
                <>
                  <TableHead>Corpus</TableHead>
                  <TableHead>Embedding</TableHead>
                </>
              )}
              <TableHead>Records</TableHead>
              <TableHead>File</TableHead>
              <TableHead>{isSystem ? "Size" : "Modified"}</TableHead>
              {isSystem && <TableHead>Modified</TableHead>}
            </TableRow>
          </TableHeader>
          <TableBody>
            {visible.length === 0 ? (
              <TableRow>
                <TableCell colSpan={colSpan} className="py-8 text-center text-xs text-muted-foreground">
                  No profiles match the current filters.
                </TableCell>
              </TableRow>
            ) : grouped.map((group) => (
              <React.Fragment key={group.key || "all"}>
                {groupBy !== "none" && (
                  <TableRow>
                    <TableCell colSpan={colSpan} className="bg-muted/50 text-xs font-medium text-muted-foreground">
                      {group.key} · {group.rows.length}
                    </TableCell>
                  </TableRow>
                )}
                {group.rows.map((p) => (
                  <TableRow
                    key={p.relative_path}
                    onClick={() => onOpenProfile(p)}
                    tabIndex={0}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        onOpenProfile(p);
                      }
                    }}
                    className="cursor-pointer"
                  >
                    <TableCell className="font-mono text-xs">{p.component}</TableCell>
                    <TableCell>
                      <Badge variant="outline">{p.kind}</Badge>
                    </TableCell>
                    {isSystem ? (
                      <TableCell className="font-mono text-xs">{p.system}</TableCell>
                    ) : (
                      <>
                        <TableCell className="max-w-[220px] truncate text-xs" title={p.corpus}>
                          {p.corpus || "-"}
                        </TableCell>
                        <TableCell className="max-w-[260px] truncate text-xs" title={p.embedding}>
                          {p.embedding || "-"}
                        </TableCell>
                      </>
                    )}
                    <TableCell className="font-mono text-xs tabular-nums">{p.record_count}</TableCell>
                    <TableCell className="max-w-[320px] truncate font-mono text-xs" title={p.path}>
                      {p.relative_path}
                    </TableCell>
                    <TableCell className={isSystem ? "font-mono text-xs" : "text-xs text-muted-foreground"}>
                      {isSystem ? formatBytes(p.size_bytes) : formatDate(p.modified_at)}
                    </TableCell>
                    {isSystem && (
                      <TableCell className="text-xs text-muted-foreground">
                        {formatDate(p.modified_at)}
                      </TableCell>
                    )}
                  </TableRow>
                ))}
              </React.Fragment>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}

function DetailField({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <span className="break-all font-mono text-xs text-foreground">{value || "-"}</span>
    </div>
  );
}

function ProfileDetailDialog({
  detail,
  loading,
  open,
  onOpenChange,
}: {
  detail: UnifiedCalibrationProfileDetail | null;
  loading: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const profile = detail?.profile;
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-h-[85vh] overflow-hidden sm:max-w-4xl">
        <DialogHeader>
          <DialogTitle>Profile Details</DialogTitle>
          <DialogDescription>
            {profile ? profile.relative_path : "Loading profile metadata"}
          </DialogDescription>
        </DialogHeader>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="size-5 animate-spin text-muted-foreground" />
          </div>
        ) : !detail || !profile ? (
          <div className="py-8 text-sm text-muted-foreground">No profile selected.</div>
        ) : (
          <div className="grid min-h-0 gap-4">
            <div className="grid gap-3 rounded-md border p-4 sm:grid-cols-3">
              <DetailField label="Axis" value={profile.axis} />
              <DetailField label="Component" value={profile.component} />
              <DetailField label="Kind" value={profile.kind} />
              <DetailField label="Key" value={profile.key} />
              <DetailField label="System" value={profile.system} />
              <DetailField label="Corpus" value={profile.corpus} />
              <DetailField label="Embedding" value={profile.embedding} />
              <DetailField label="Records" value={profile.record_count} />
              <DetailField label="Size" value={formatBytes(profile.size_bytes)} />
              <DetailField label="Modified" value={formatDate(profile.modified_at)} />
              <DetailField label="File" value={profile.path} />
              <DetailField label="Content Type" value={detail.content_type} />
            </div>

            {detail.error && (
              <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-sm text-destructive">
                {detail.error}
              </div>
            )}

            {detail.content_type === "json" ? (
              <pre className="max-h-[45vh] overflow-auto rounded-md border bg-muted/40 p-3 font-mono text-xs leading-5 text-foreground">
                {JSON.stringify(detail.content, null, 2)}
              </pre>
            ) : (
              <div className="rounded-md border bg-muted/40 p-4 text-sm text-muted-foreground">
                Binary calibration artifact. The app shows file metadata; the
                model payload is loaded by the calibration/runtime code.
              </div>
            )}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}

function DataTable({ files }: { files: CalibrationDataFile[] }) {
  return (
    <Card>
      <CardHeader className="border-b">
        <CardTitle className="flex items-center gap-2 text-sm">
          <Database className="size-4 text-primary" />
          calibration_data
          <Badge variant="secondary">{files.length}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="px-0">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Component</TableHead>
            <TableHead>System</TableHead>
            <TableHead>Format</TableHead>
            <TableHead>Path</TableHead>
            <TableHead>Size</TableHead>
            <TableHead>Modified</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {files.length === 0 ? (
            <TableRow>
              <TableCell colSpan={6} className="py-8 text-center text-xs text-muted-foreground">
                No frozen measured data found.
              </TableCell>
            </TableRow>
          ) : files.map((f) => (
            <TableRow key={f.relative_path}>
              <TableCell className="font-mono text-xs">{f.component}</TableCell>
              <TableCell className="font-mono text-xs">{f.system}</TableCell>
              <TableCell>
                <Badge variant="outline">{f.format}</Badge>
              </TableCell>
              <TableCell className="max-w-[420px] truncate font-mono text-xs" title={f.path}>
                {f.path}
              </TableCell>
              <TableCell className="font-mono text-xs">{formatBytes(f.size_bytes)}</TableCell>
              <TableCell className="text-xs text-muted-foreground">
                {formatDate(f.modified_at)}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      </CardContent>
    </Card>
  );
}

export function UnifiedCalibrationView() {
  const [status, setStatus] = useState<UnifiedCalibrationStatus | null>(null);
  const [jobs, setJobs] = useState<CalibrationJob[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string>("");
  const [log, setLog] = useState("");
  const [loading, setLoading] = useState(true);
  const [launching, setLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [profileDetail, setProfileDetail] = useState<UnifiedCalibrationProfileDetail | null>(null);

  const [command, setCommand] = useState("plan");
  const [projectDir, setProjectDir] = useState("");
  const [dataDir, setDataDir] = useState("");
  const [profileDir, setProfileDir] = useState("");
  const [inventoryError, setInventoryError] = useState<string | null>(null);
  const [configPath, setConfigPath] = useState("configs/rag_stack/config_reference.yaml");
  const [qaData, setQaData] = useState("");
  const [corpusData, setCorpusData] = useState("");
  const [maxiter, setMaxiter] = useState("");
  const [force, setForce] = useState(false);
  const [extraArgs, setExtraArgs] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [nextStatus, nextJobs] = await Promise.all([
        fetchUnifiedCalibrationStatus(),
        fetchUnifiedCalibrationJobs(),
      ]);
      setStatus(nextStatus);
      setCommand((previous) => nextStatus.commands.some((item) => item.value === previous) ? previous : nextStatus.commands[0]?.value || "");
      setJobs(nextJobs);
      setSelectedJobId((prev) => prev || nextJobs[0]?.job_id || "");
      setInventoryError(null);
    } catch (err) {
      setInventoryError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const selectedJob = useMemo(
    () => jobs.find((j) => j.job_id === selectedJobId) ?? jobs[0] ?? null,
    [jobs, selectedJobId],
  );

  const refreshLog = useCallback(async () => {
    if (!selectedJob) {
      setLog("");
      return;
    }
    try {
      const data = await fetchUnifiedCalibrationJobLog(selectedJob.job_id);
      setLog(data.log);
    } catch (err) {
      setLog(err instanceof Error ? err.message : String(err));
    }
  }, [selectedJob]);

  useEffect(() => { refreshLog(); }, [refreshLog]);

  useEffect(() => {
    const hasRunning = jobs.some((j) => j.status === "running");
    if (!hasRunning) return;
    const timer = setInterval(() => {
      load();
      refreshLog();
    }, 2000);
    return () => clearInterval(timer);
  }, [jobs, load, refreshLog]);

  const selectedCommand = useMemo(
    () => status?.commands.find((item) => item.value === command),
    [command, status],
  );
  const missingArguments = (selectedCommand?.required_extra_args ?? []).filter((flag) => {
    const tokens = splitArgs(extraArgs);
    return !tokens.some((token, index) => token.startsWith(`${flag}=`) && token.length > flag.length + 1 || token === flag && !!tokens[index + 1] && !tokens[index + 1].startsWith("--"));
  });

  const launch = useCallback(async () => {
    if (!selectedCommand) return;
    setLaunching(true);
    setError(null);
    try {
      const job = await runUnifiedCalibrationJob({
        command,
        config_path: selectedCommand.needs_config ? configPath : "",
        force: selectedCommand.supports_force ? force : false,
        maxiter: selectedCommand.supports_maxiter && maxiter.trim() ? Number(maxiter) : null,
        qa_data: selectedCommand.supports_data_paths ? qaData : "",
        corpus_data: selectedCommand.supports_data_paths ? corpusData : "",
        project_dir: selectedCommand.requires_project_dir ? projectDir : "",
        data_dir: selectedCommand.supports_data_dir ? dataDir : "",
        profile_dir: selectedCommand.supports_profile_dir ? profileDir : "",
        extra_args: splitArgs(extraArgs),
      });
      setSelectedJobId(job.job_id);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLaunching(false);
    }
  }, [command, configPath, corpusData, extraArgs, force, load, maxiter, qaData, projectDir, dataDir, profileDir, selectedCommand]);

  const stopSelected = useCallback(async () => {
    if (!selectedJob || selectedJob.status !== "running") return;
    try {
      await killJob(selectedJob.job_id);
      await load();
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
  }, [load, selectedJob]);

  const openProfileDetail = useCallback(async (profile: UnifiedCalibrationProfile) => {
    setDetailOpen(true);
    setDetailLoading(true);
    setProfileDetail({
      profile,
      content_type: profile.filename.endsWith(".json") ? "json" : "binary",
      content: null,
    });
    try {
      const detail = await fetchUnifiedCalibrationProfileDetail(profile.relative_path);
      setProfileDetail(detail);
    } catch (err) {
      setProfileDetail({
        profile,
        content_type: profile.filename.endsWith(".json") ? "json" : "binary",
        content: null,
        error: err instanceof Error ? err.message : String(err),
      });
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const profiles = status?.profiles ?? [];
  const corpusEmbeddingProfiles = useMemo(
    () => profiles.filter((p) => p.axis === "corpus_embedding"),
    [profiles],
  );
  const systemProfiles = useMemo(
    () => profiles.filter((p) => p.axis === "system"),
    [profiles],
  );
  const dataFiles = status?.data_files ?? [];

  if (!status || !selectedCommand) return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-center">
      {loading ? <><Loader2 className="size-5 animate-spin text-muted-foreground" /><p className="text-sm text-muted-foreground">Loading calibration inventory…</p></>
        : <><p role="alert" className="text-sm text-destructive">{inventoryError || "No calibration commands are available from the server."}</p><Button variant="outline" onClick={() => void load()}>Try again</Button></>}
    </div>
  );

  return (
    <div className="flex flex-1 min-w-0 overflow-auto bg-background">
      <div className="mx-auto flex w-full max-w-[1280px] flex-col gap-5 px-4 pt-8 pb-16 sm:px-8">
        <div className="space-y-2">
          <h2 className="text-xl font-semibold tracking-tight">Calibration inventory</h2>
          <p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">Browse saved artifacts and run calibration commands. Artifact counts do not indicate project coverage; CM Preview validates the profiles selected for a compiled workload.</p>
          <p className="break-all text-xs text-muted-foreground">Runtime profile root: <span className="font-mono">{status.profiles_root}</span></p>
          {status.root_note && <p className="text-xs text-muted-foreground">{status.root_note}</p>}
        </div>
        {inventoryError && <p role="alert" className="rounded-lg border border-destructive/30 p-3 text-sm text-destructive">Inventory refresh failed: {inventoryError}</p>}
        <div className="flex flex-wrap gap-3">
          <SummaryTile label="Profile files" value={status?.summary.profiles ?? 0} />
          <SummaryTile label="System Axis" value={status?.summary.system_profiles ?? 0} />
          <SummaryTile
            label="Corpus x Embedding"
            value={status?.summary.corpus_embedding_profiles ?? 0}
          />
          <SummaryTile label="Measurement files" value={status?.summary.data_files ?? 0} />
        </div>

        <div className="grid gap-5 xl:grid-cols-[minmax(420px,0.75fr)_minmax(520px,1fr)]">
          <Card>
            <CardHeader className="border-b">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Play className="size-4 text-primary" />
                Calibration command
              </CardTitle>
              <CardAction>
              <Button type="button" variant="ghost" size="icon" onClick={load} className="size-7">
                {loading ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <RefreshCw className="size-4 text-muted-foreground" />
                )}
              </Button>
              </CardAction>
            </CardHeader>
            <CardContent className="flex flex-col gap-3.5">
              <label className="flex flex-col gap-1.5">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  Command
                </span>
                <Select value={command} onValueChange={setCommand}>
                  <SelectTrigger className="w-full" aria-label="Calibration command">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {status.commands.map((item) => (
                      <SelectItem key={item.value} value={item.value}>
                        {item.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </label>

              {selectedCommand.needs_config && (
                <label className="flex flex-col gap-1.5">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Config Path
                  </span>
                  <Input
                    value={configPath}
                    onChange={(e) => setConfigPath(e.target.value)}
                    className="font-mono"
                  />
                </label>
              )}

              {selectedCommand.requires_project_dir && <label className="flex flex-col gap-1.5">
                <span className="text-xs font-medium">Project directory</span>
                <Input value={projectDir} onChange={(event) => setProjectDir(event.target.value)} placeholder="Project containing observed workload" className="font-mono" />
              </label>}
              {(selectedCommand.supports_data_dir || selectedCommand.supports_profile_dir) && <details className="rounded-md border p-3">
                <summary className="cursor-pointer text-xs font-medium">Output directories (optional)</summary>
                <div className="mt-3 space-y-3">
                  {selectedCommand.supports_data_dir && <label className="flex flex-col gap-1.5"><span className="text-xs text-muted-foreground">Measurement directory</span><Input value={dataDir} onChange={(event) => setDataDir(event.target.value)} placeholder="Default measurement root" className="font-mono" /></label>}
                  {selectedCommand.supports_profile_dir && <label className="flex flex-col gap-1.5"><span className="text-xs text-muted-foreground">Profile output directory</span><Input value={profileDir} onChange={(event) => setProfileDir(event.target.value)} placeholder={status.profiles_root} className="font-mono" /></label>}
                  <p className="text-xs leading-relaxed text-muted-foreground">Custom output directories stage artifacts separately. CM reads profiles from the runtime root shown above.</p>
                </div>
              </details>}

              {selectedCommand.supports_data_paths && (
                <div className="grid gap-3 sm:grid-cols-2">
                  <label className="flex flex-col gap-1.5">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      QA Data
                    </span>
                    <Input value={qaData} onChange={(e) => setQaData(e.target.value)} className="font-mono" />
                  </label>
                  <label className="flex flex-col gap-1.5">
                    <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                      Corpus Data
                    </span>
                    <Input value={corpusData} onChange={(e) => setCorpusData(e.target.value)} className="font-mono" />
                  </label>
                </div>
              )}

              <div className={selectedCommand.supports_maxiter ? "grid gap-3 sm:grid-cols-[120px_1fr]" : "grid gap-3"}>
                {selectedCommand.supports_maxiter && (
                <label className="flex flex-col gap-1.5">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Max Iter
                  </span>
                  <Input value={maxiter} onChange={(e) => setMaxiter(e.target.value)} type="number" className="font-mono" />
                </label>
                )}
                <label className="flex flex-col gap-1.5">
                  <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Extra Args
                  </span>
                  <Input
                    value={extraArgs}
                    onChange={(e) => setExtraArgs(e.target.value)}
                    placeholder={selectedCommand.extra_placeholder || "Additional CLI arguments (optional)"}
                    className="font-mono"
                  />
                  {selectedCommand.required_extra_args.length > 0 && <span className="text-xs text-muted-foreground">Required: {selectedCommand.required_extra_args.join(", ")}</span>}
                </label>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3">
                {selectedCommand.supports_force ? (
                  <label className="flex items-center gap-2 text-xs text-foreground">
                    <Checkbox checked={force} onCheckedChange={(v) => setForce(v === true)} />
                    Force measure
                  </label>
                ) : (
                  <span className="text-xs text-muted-foreground" />
                )}
                <Button type="button" onClick={launch} disabled={launching || !command || missingArguments.length > 0 || (selectedCommand.needs_config && !configPath.trim()) || (selectedCommand.requires_project_dir && !projectDir.trim())}>
                  {launching ? <Loader2 className="size-4 animate-spin" /> : <Play className="size-4" />}
                  {selectedCommand.read_only ? "Check configuration" : "Run calibration"}
                </Button>
              </div>
              <div className="rounded-md bg-muted px-3 py-2">
                <span className="font-mono text-xs text-muted-foreground">
                  rag_stack calibrate {command}
                  {selectedCommand.needs_config && configPath ? ` --config ${configPath}` : ""}
                  {selectedCommand.requires_project_dir && projectDir ? ` --project-dir ${projectDir}` : ""}
                  {selectedCommand.supports_data_dir && dataDir ? ` --data-dir ${dataDir}` : ""}
                  {selectedCommand.supports_profile_dir && profileDir ? ` --profile-dir ${profileDir}` : ""}
                  {selectedCommand.supports_force && force ? " --force" : ""}
                  {selectedCommand.supports_maxiter && maxiter.trim() ? ` --maxiter ${maxiter.trim()}` : ""}
                  {selectedCommand.supports_data_paths && qaData ? ` --qa-data ${qaData}` : ""}
                  {selectedCommand.supports_data_paths && corpusData ? ` --corpus-data ${corpusData}` : ""}
                  {extraArgs.trim() ? ` ${extraArgs.trim()}` : ""}
                </span>
              </div>
              {error && (
                <div className="rounded-md border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
                  {error}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="border-b">
              <CardTitle className="flex items-center gap-2 text-sm">
                <Terminal className="size-4 text-primary" />
                Output
              </CardTitle>
              <CardAction className="flex items-center gap-2">
                {selectedJob && statusBadge(selectedJob)}
                {selectedJob?.status === "running" && (
                  <Button type="button" variant="ghost" size="icon" onClick={stopSelected} className="size-7 hover:bg-destructive/10">
                    <Square className="size-3 fill-destructive text-destructive" />
                  </Button>
                )}
              </CardAction>
            </CardHeader>
            <CardContent className="grid min-h-[310px] grid-rows-[auto_1fr] px-0 pb-0">
              <div className="flex gap-2 overflow-x-auto px-3 pb-3">
                {jobs.length === 0 ? (
                  <span className="px-1 py-1 text-xs text-muted-foreground">No calibration jobs.</span>
                ) : jobs.slice(0, 12).map((job) => (
                  <Button
                    key={job.job_id}
                    type="button"
                    variant={selectedJob?.job_id === job.job_id ? "secondary" : "outline"}
                    onClick={() => setSelectedJobId(job.job_id)}
                    className="h-auto shrink-0 flex-col items-start gap-0 px-2.5 py-1.5 text-left"
                  >
                    <span className="block max-w-[170px] truncate font-medium">
                      {jobLabel(job)}
                    </span>
                    <span className="font-mono text-[10px]">{job.job_id}</span>
                  </Button>
                ))}
              </div>
              <Separator />
              <pre className="min-h-[260px] overflow-auto bg-muted/40 p-3 font-mono text-[11px] leading-5 text-foreground">
                {log || "No output."}
              </pre>
            </CardContent>
          </Card>
        </div>

        <ProfileTable
          title="Corpus x Embedding Profiles"
          axis="corpus_embedding"
          profiles={corpusEmbeddingProfiles}
          onOpenProfile={openProfileDetail}
        />
        <ProfileTable
          title="System Profiles"
          axis="system"
          profiles={systemProfiles}
          onOpenProfile={openProfileDetail}
        />
        <DataTable files={dataFiles} />
      </div>
      <ProfileDetailDialog
        detail={profileDetail}
        loading={detailLoading}
        open={detailOpen}
        onOpenChange={setDetailOpen}
      />
    </div>
  );
}
