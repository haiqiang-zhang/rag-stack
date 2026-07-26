import { useMemo, useState } from "react";
import { ChevronDown, ChevronUp, ChevronsUpDown } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useEvalData } from "@/context/EvalDataContext";
import { useSelection } from "@/context/SelectionContext";
import { isEvalFailure, type Evaluation } from "@/types";
import { cn } from "@/lib/utils";

type SortDir = "asc" | "desc";

/** Every evaluation as a sortable, clickable table. Click a row to select the
 *  eval for the Performance / Config tabs. */
export function HistoryTab() {
  const { evaluations, getPerformanceResult } = useEvalData();
  const { selectedEvalId, setSelectedEvalId } = useSelection();

  const { qualityKeys, configKeys } = useMemo(() => {
    const q = new Set<string>();
    const c = new Set<string>();
    for (const ev of evaluations) {
      for (const k of Object.keys(ev.quality_metrics)) q.add(k);
      for (const k of Object.keys(ev.config)) c.add(k);
    }
    return { qualityKeys: [...q], configKeys: [...c] };
  }, [evaluations]);

  const [sortKey, setSortKey] = useState<string>("__id");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const sortVal = (ev: Evaluation, key: string): number | string => {
    if (key === "__id") return ev.eval_id;
    if (key === "__type") return ev.eval_type;
    if (key === "__perf") {
      if (isEvalFailure(ev.eval_type)) return -Infinity;
      const r = getPerformanceResult(ev);
      return r.feasible ? r.score : -Infinity;
    }
    if (key.startsWith("q:")) return ev.quality_metrics[key.slice(2)] ?? -Infinity;
    const v = ev.config[key.slice(2)];
    return v == null ? "" : typeof v === "number" ? v : String(v);
  };

  const sorted = useMemo(() => {
    const rows = [...evaluations];
    rows.sort((a, b) => {
      const va = sortVal(a, sortKey);
      const vb = sortVal(b, sortKey);
      let cmp: number;
      if (typeof va === "number" && typeof vb === "number") cmp = va - vb;
      else cmp = String(va).localeCompare(String(vb));
      return sortDir === "asc" ? cmp : -cmp;
    });
    return rows;
  }, [evaluations, sortKey, sortDir]); // eslint-disable-line react-hooks/exhaustive-deps

  const onSort = (key: string) => {
    if (sortKey === key) setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    else {
      setSortKey(key);
      setSortDir(key === "__id" ? "desc" : "asc");
    }
  };

  if (evaluations.length === 0) {
    return (
      <div className="flex flex-1 min-h-0 items-center justify-center p-6">
        <span className="text-sm text-muted-foreground">No evaluations yet.</span>
      </div>
    );
  }

  const columns: { key: string; label: string; numeric?: boolean }[] = [
    { key: "__id", label: "ID", numeric: true },
    { key: "__type", label: "Type" },
    { key: "__perf", label: "Performance", numeric: true },
    ...qualityKeys.map((k) => ({ key: `q:${k}`, label: k.replace(/_/g, " "), numeric: true })),
    ...configKeys.map((k) => ({ key: `c:${k}`, label: k.replace(/__/g, " › ") })),
  ];

  return (
    <div className="flex-1 min-w-0 overflow-auto p-6 pb-12">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold text-foreground">
          All evaluations{" "}
          <span className="font-mono text-xs font-normal text-muted-foreground">
            ({evaluations.length})
          </span>
        </h3>
        <span className="text-[11px] text-muted-foreground">
          Click a row to inspect it in Performance / Config
        </span>
      </div>
      <div className="min-w-0 overflow-auto rounded-md border bg-card">
        <Table className="min-w-[720px]">
          <TableHeader className="bg-muted/60">
            <TableRow className="hover:bg-transparent">
              {columns.map((col) => (
                <TableHead
                  key={col.key}
                  onClick={() => onSort(col.key)}
                  className={cn(
                    "sticky top-0 z-10 h-9 cursor-pointer select-none whitespace-nowrap bg-muted/60 px-3 text-[10px] uppercase text-muted-foreground hover:text-foreground",
                    col.numeric && "text-right",
                  )}
                >
                  <span
                    className={cn(
                      "inline-flex items-center gap-1",
                      col.numeric && "flex-row-reverse",
                    )}
                  >
                    {col.label}
                    <SortIcon active={sortKey === col.key} dir={sortDir} />
                  </span>
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {sorted.map((ev) => {
              const failed = isEvalFailure(ev.eval_type);
              const selected = ev.eval_id === selectedEvalId;
              return (
                <TableRow
                  key={ev.eval_id}
                  onClick={() => setSelectedEvalId(ev.eval_id)}
                  className={cn(
                    "cursor-pointer",
                    selected && "bg-primary/10 hover:bg-primary/15",
                    !selected && failed && "bg-destructive/[0.04]",
                  )}
                >
                  <TableCell
                    className={cn(
                      "px-3 py-2 text-right font-mono text-xs tabular-nums text-foreground",
                      selected && "border-l-2 border-l-primary",
                    )}
                  >
                    {ev.eval_id}
                  </TableCell>
                  <TableCell className="px-3 py-2">
                    <span
                      title={ev.eval_error || undefined}
                      className={cn(
                        "font-mono text-[11px] font-semibold",
                        ev.eval_type === "GT"
                          ? "text-emerald-600 dark:text-emerald-400"
                          : failed
                            ? "text-destructive"
                            : "text-muted-foreground",
                        ev.eval_error && "cursor-help",
                      )}
                    >
                      {ev.eval_type}
                    </span>
                  </TableCell>
                  <TableCell className="px-3 py-2 text-right font-mono text-xs tabular-nums">
                    <PerfCell ev={ev} getPerformanceResult={getPerformanceResult} />
                  </TableCell>
                  {qualityKeys.map((k) => (
                    <TableCell
                      key={k}
                      className="px-3 py-2 text-right font-mono text-xs tabular-nums text-foreground"
                    >
                      {ev.quality_metrics[k] != null
                        ? ev.quality_metrics[k].toFixed(3)
                        : "—"}
                    </TableCell>
                  ))}
                  {configKeys.map((k) => (
                    <TableCell
                      key={k}
                      title={String(ev.config[k] ?? "")}
                      className="max-w-[180px] truncate px-3 py-2 font-mono text-[11px] text-muted-foreground"
                    >
                      {String(ev.config[k] ?? "—")}
                    </TableCell>
                  ))}
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

function PerfCell({
  ev,
  getPerformanceResult,
}: {
  ev: Evaluation;
  getPerformanceResult: (ev: Evaluation) => { score: number; feasible: boolean };
}) {
  if (isEvalFailure(ev.eval_type)) {
    return (
      <span
        title={ev.eval_error || undefined}
        className="text-muted-foreground"
      >
        —
      </span>
    );
  }
  const r = getPerformanceResult(ev);
  if (!r.feasible)
    return (
      <span className="italic text-destructive" title="No SLO-feasible point in sweep">
        infeasible
      </span>
    );
  return <span className="text-foreground">{r.score.toFixed(3)}</span>;
}

function SortIcon({ active, dir }: { active: boolean; dir: SortDir }) {
  if (!active)
    return <ChevronsUpDown className="size-3 text-muted-foreground/40" />;
  return dir === "asc" ? (
    <ChevronUp className="size-3 text-foreground" />
  ) : (
    <ChevronDown className="size-3 text-foreground" />
  );
}
