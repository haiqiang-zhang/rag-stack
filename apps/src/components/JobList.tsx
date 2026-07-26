import { RefreshCw, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { StatusPill } from "./ui/StatusPill";

export interface Job {
  job_id: string;
  job_type?: string;
  config: Record<string, unknown>;
  pid: number;
  status: "running" | "completed" | "failed";
  started_at: string;
  finished_at: string | null;
  error: string | null;
  output_dir: string;
}

interface JobListProps {
  jobs: Job[];
  onRefresh: () => void;
  onKill?: (jobId: string) => void;
  /** How to render the job title from its config. Default: shows optimizer type. */
  renderTitle?: (job: Job) => string;
}

function defaultTitle(job: Job): string {
  const opt = job.config?.optimizer;
  const optType =
    typeof opt === "object" && opt ? ((opt as any).type ?? "") : "";
  return optType || job.job_id;
}

export function JobList({
  jobs,
  onRefresh,
  onKill,
  renderTitle = defaultTitle,
}: JobListProps) {
  return (
    <Card>
      <CardContent>
        <div className="mb-2.5 flex items-center justify-between">
          <span className="text-[11px] font-semibold tracking-[0.5px] text-muted-foreground">
            JOBS
          </span>
          <Button type="button" variant="ghost" size="icon" onClick={onRefresh} className="size-6">
            <RefreshCw className="size-3 text-muted-foreground" />
          </Button>
        </div>

        {jobs.length === 0 ? (
          <span className="text-xs text-muted-foreground">No jobs yet.</span>
        ) : (
          <div className="flex flex-col gap-1.5">
            {jobs.map((job) => (
              <div
                key={job.job_id}
                className="flex items-center gap-2.5 rounded-lg bg-muted p-2.5"
              >
                <span
                  className={`size-1.5 shrink-0 rounded-full ${
                    job.status === "running"
                      ? "bg-amber-500"
                      : job.status === "completed"
                        ? "bg-emerald-500"
                        : "bg-destructive"
                  }`}
                />
                <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className="truncate text-xs font-medium text-foreground">
                    {renderTitle(job)}
                  </span>
                  <span className="text-[10px] text-muted-foreground">
                    {job.job_id} · PID {job.pid} ·{" "}
                    {job.started_at?.slice(0, 19).replace("T", " ")}
                  </span>
                  {job.status === "failed" && job.error && (
                    <span className="line-clamp-2 text-[10px] text-destructive">
                      {job.error}
                    </span>
                  )}
                </div>
                <StatusPill status={job.status} />
                {job.status === "running" && onKill && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => onKill(job.job_id)}
                    className="size-6 hover:bg-destructive/10"
                  >
                    <Square className="size-2.5 fill-destructive text-destructive" />
                  </Button>
                )}
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
