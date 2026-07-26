import { Loader2, CheckCircle } from "lucide-react";

interface StatusBadgeProps {
  status: "running" | "completed" | "unknown";
  nEvaluations?: number;
}

export function StatusBadge({ status, nEvaluations }: StatusBadgeProps) {
  if (status === "running") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-100 px-2.5 py-1 text-xs font-medium text-amber-700 dark:bg-amber-500/15 dark:text-amber-300">
        <Loader2 className="size-3 animate-spin" />
        Running{nEvaluations != null ? ` · ${nEvaluations} evals` : ""}
      </span>
    );
  }

  if (status === "completed") {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-2.5 py-1 text-xs font-medium text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300">
        <CheckCircle className="size-3" />
        Completed
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 text-xs font-medium text-muted-foreground">
      <span className="size-1.5 rounded-full bg-muted-foreground" />
      Loading
    </span>
  );
}
