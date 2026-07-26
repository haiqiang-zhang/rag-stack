/**
 * SummaryTable — final-checkpoint stats per (optimizer, function), linked
 * to the chart grid above via a shared selection set.
 *
 * Each row carries a checkbox; the master checkbox in the header toggles
 * all-or-none with an indeterminate state when partially selected. Rows
 * also display a color dot keyed to the same CHART_COLORS index used by
 * BenchmarkChartGrid, so a glance maps each row to its chart line. A
 * compact toolbar above the table offers "All", "None", and "Best 3"
 * quick filters.
 *
 * Selection state is owned by the parent (OptimizerScreen). When a row
 * is unchecked the parent passes the new set into BenchmarkChartGrid,
 * which hides that optimizer's series via `legend.selected` — colors
 * stay stable.
 */
import * as React from "react";
import {
  ArrowDown,
  ArrowUp,
  ChevronsUpDown,
  Search,
  Trash2,
  X,
} from "lucide-react";
import type { OptimizerBenchmarkSummary } from "@/types/optimizer";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/ConfirmDialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { CHART_COLORS } from "@/lib/theme";
import { cn } from "@/lib/utils";

type SortKey =
  | "optimizer"
  | "synthetic_function"
  | "n_seeds"
  | "hv_diff_mean"
  | "igd_mean"
  | "gt_calls"
  | "optimizer_time_s";

type SortDir = "asc" | "desc";

interface Props {
  summary: OptimizerBenchmarkSummary[];
  /**
   * Full sorted list of optimizer names — used to assign stable chart
   * colors regardless of which rows the table is filtered to.
   */
  optimizers: string[];
  /**
   * The set of optimizer names currently included in the linked chart.
   * Defaults to the full optimizer list when omitted.
   */
  selected?: Set<string>;
  onSelectedChange?: (next: Set<string>) => void;
  /**
   * Optional row-level deletion. When provided, a trash button is rendered
   * at the end of each row; clicking it opens a confirmation dialog and
   * the parent is responsible for calling the API + refetching.
   */
  onDelete?: (row: OptimizerBenchmarkSummary) => Promise<void> | void;
}

const NUMERIC_KEYS: ReadonlySet<SortKey> = new Set([
  "n_seeds",
  "hv_diff_mean",
  "igd_mean",
  "gt_calls",
  "optimizer_time_s",
]);

/**
 * Adaptive numeric formatter.
 *   - Mid-range values (|x| ≥ 0.01): plain decimal, ~5 significant figures.
 *   - Small values (|x| < 0.01): scientific notation with 4 mantissa
 *     decimals (5 sig figs total), e.g. 2.0000e-4 / 4.2000e-3 — precision
 *     is applied AFTER the scientific conversion, so display digits stay
 *     consistent regardless of magnitude.
 *   - Very large (|x| ≥ 1e5): same scientific path so wide values don't
 *     blow up the column width.
 */
function fmt(x: number): string {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return "0";
  const abs = Math.abs(x);
  if (abs >= 1e5) return x.toExponential(4);
  if (abs < 1e-2) return x.toExponential(4);
  if (abs >= 100) return x.toFixed(2);
  if (abs >= 10) return x.toFixed(3);
  if (abs >= 1) return x.toFixed(4);
  // 0.01 ≤ |x| < 1 → 4–5 decimal places (~5 sig figs), trim trailing zeros.
  const decimals = Math.min(6, -Math.floor(Math.log10(abs)) + 4);
  const s = x.toFixed(decimals);
  return s.includes(".") ? s.replace(/\.?0+$/, "") || "0" : s;
}

export function SummaryTable({
  summary,
  optimizers,
  selected,
  onSelectedChange,
  onDelete,
}: Props) {
  const [sortKey, setSortKey] = React.useState<SortKey>("hv_diff_mean");
  const [sortDir, setSortDir] = React.useState<SortDir>("asc");
  const [search, setSearch] = React.useState("");
  const [deleteTarget, setDeleteTarget] =
    React.useState<OptimizerBenchmarkSummary | null>(null);
  const [deleting, setDeleting] = React.useState(false);

  const handleDeleteConfirm = async () => {
    if (!onDelete || !deleteTarget) return;
    try {
      setDeleting(true);
      await onDelete(deleteTarget);
      setDeleteTarget(null);
    } finally {
      setDeleting(false);
    }
  };

  const colorOf = React.useMemo(() => {
    const m = new Map<string, string>();
    optimizers.forEach((opt, i) => {
      m.set(opt, CHART_COLORS[i % CHART_COLORS.length]);
    });
    return m;
  }, [optimizers]);

  // Effective selection — defaults to all optimizers when none provided.
  const effective = React.useMemo(
    () => selected ?? new Set(optimizers),
    [selected, optimizers],
  );

  const onHeaderClick = (key: SortKey) => {
    if (key === sortKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir(NUMERIC_KEYS.has(key) ? "asc" : "asc");
    }
  };

  const filtered = React.useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return summary;
    return summary.filter(
      (r) =>
        r.optimizer.toLowerCase().includes(q) ||
        r.synthetic_function.toLowerCase().includes(q),
    );
  }, [summary, search]);

  const sorted = React.useMemo(() => {
    const arr = [...filtered];
    arr.sort((a, b) => {
      const av = a[sortKey] as string | number;
      const bv = b[sortKey] as string | number;
      let cmp: number;
      if (typeof av === "number" && typeof bv === "number") {
        cmp = av - bv;
      } else {
        cmp = String(av).localeCompare(String(bv));
      }
      return sortDir === "asc" ? cmp : -cmp;
    });
    return arr;
  }, [filtered, sortKey, sortDir]);

  const best = React.useMemo(() => {
    if (summary.length === 0) {
      return { hv: "", igd: "", time: "" };
    }
    let hv = 0, igd = 0, time = 0;
    for (let i = 1; i < summary.length; i++) {
      if (summary[i].hv_diff_mean < summary[hv].hv_diff_mean) hv = i;
      if (summary[i].igd_mean < summary[igd].igd_mean) igd = i;
      if (summary[i].optimizer_time_s < summary[time].optimizer_time_s) time = i;
    }
    return {
      hv: summary[hv].optimizer + "|" + summary[hv].synthetic_function,
      igd: summary[igd].optimizer + "|" + summary[igd].synthetic_function,
      time: summary[time].optimizer + "|" + summary[time].synthetic_function,
    };
  }, [summary]);

  // Master checkbox + selection counts operate on the filtered (search-
  // matched) rows, so checking "all" only toggles what the user can see.
  const visibleNames = React.useMemo(
    () => new Set(filtered.map((r) => r.optimizer)),
    [filtered],
  );
  const visibleSelectedCount = React.useMemo(() => {
    let n = 0;
    for (const o of visibleNames) if (effective.has(o)) n++;
    return n;
  }, [visibleNames, effective]);
  const masterState: boolean | "indeterminate" =
    visibleSelectedCount === 0
      ? false
      : visibleSelectedCount === visibleNames.size
        ? true
        : "indeterminate";

  const toggleOne = (optimizer: string) => {
    if (!onSelectedChange) return;
    const next = new Set(effective);
    if (next.has(optimizer)) next.delete(optimizer);
    else next.add(optimizer);
    onSelectedChange(next);
  };

  const onMasterToggle = () => {
    if (!onSelectedChange) return;
    if (masterState === true) {
      // Currently all visible are selected → drop them from selection.
      const next = new Set(effective);
      for (const n of visibleNames) next.delete(n);
      onSelectedChange(next);
    } else {
      // Add all visible names to selection.
      const next = new Set(effective);
      for (const n of visibleNames) next.add(n);
      onSelectedChange(next);
    }
  };

  if (summary.length === 0) {
    return (
      <div className="py-8 text-center text-sm text-muted-foreground">
        No summary data
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {/* Toolbar: title + count + search */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-baseline gap-2">
          <h3 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
            Final Checkpoint
          </h3>
          <span className="text-[11px] tabular-nums text-muted-foreground/80">
            ·&nbsp;
            {search.trim()
              ? `${filtered.length}/${summary.length}`
              : summary.length}{" "}
            {summary.length === 1 ? "row" : "rows"}
          </span>
          {onSelectedChange && (
            <span className="text-[11px] text-muted-foreground/80">
              ·&nbsp;
              <span className="tabular-nums text-foreground">
                {visibleSelectedCount}/{visibleNames.size}
              </span>{" "}
              shown in chart
            </span>
          )}
        </div>
        <div className="relative w-64 max-w-full">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search optimizers or functions…"
            className="h-8 pl-8 pr-8 text-xs"
          />
          {search && (
            <button
              type="button"
              onClick={() => setSearch("")}
              aria-label="Clear search"
              className="absolute right-1 top-1/2 -translate-y-1/2 rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              <X className="size-3" />
            </button>
          )}
        </div>
      </div>

      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            {onSelectedChange && (
              <TableHead className="w-8">
                <Checkbox
                  checked={masterState}
                  onCheckedChange={onMasterToggle}
                  aria-label="Toggle all"
                />
              </TableHead>
            )}
            <SortableHead
              label="Optimizer"
              sortKey="optimizer"
              activeKey={sortKey}
              dir={sortDir}
              onClick={onHeaderClick}
            />
            <SortableHead
              label="Function"
              sortKey="synthetic_function"
              activeKey={sortKey}
              dir={sortDir}
              onClick={onHeaderClick}
            />
            <SortableHead
              label="Seeds"
              sortKey="n_seeds"
              activeKey={sortKey}
              dir={sortDir}
              onClick={onHeaderClick}
              align="right"
            />
            <SortableHead
              label="HV Diff"
              sortKey="hv_diff_mean"
              activeKey={sortKey}
              dir={sortDir}
              onClick={onHeaderClick}
              align="right"
            />
            <SortableHead
              label="IGD"
              sortKey="igd_mean"
              activeKey={sortKey}
              dir={sortDir}
              onClick={onHeaderClick}
              align="right"
            />
            <SortableHead
              label="GT"
              sortKey="gt_calls"
              activeKey={sortKey}
              dir={sortDir}
              onClick={onHeaderClick}
              align="right"
            />
            <SortableHead
              label="Time (s)"
              sortKey="optimizer_time_s"
              activeKey={sortKey}
              dir={sortDir}
              onClick={onHeaderClick}
              align="right"
            />
            {onDelete && <TableHead className="w-8" />}
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((row) => {
            const id = `${row.optimizer}|${row.synthetic_function}`;
            const color = colorOf.get(row.optimizer) ?? "#9CA3AF";
            const checked = effective.has(row.optimizer);
            return (
              <TableRow
                key={id}
                data-state={checked ? "selected" : undefined}
                className={cn(!checked && "opacity-55")}
              >
                {onSelectedChange && (
                  <TableCell className="w-8">
                    <Checkbox
                      checked={checked}
                      onCheckedChange={() => toggleOne(row.optimizer)}
                      aria-label={`Toggle ${row.optimizer}`}
                    />
                  </TableCell>
                )}
                <TableCell className="font-medium text-foreground">
                  <span className="flex items-center gap-2">
                    <span
                      aria-hidden
                      className={cn(
                        "size-2 shrink-0 rounded-full transition-opacity",
                        !checked && "opacity-40",
                      )}
                      style={{ backgroundColor: color }}
                    />
                    <span className="truncate">{row.optimizer}</span>
                  </span>
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {row.synthetic_function}
                </TableCell>
                <TableCell className="text-right tabular-nums text-muted-foreground">
                  {row.seeds && row.seeds.length > 0 ? (
                    <TooltipProvider delayDuration={200}>
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="cursor-help underline decoration-dotted underline-offset-2">
                            {row.n_seeds}
                          </span>
                        </TooltipTrigger>
                        <TooltipContent>
                          <span className="font-mono text-[11px]">
                            {row.seeds.join(", ")}
                          </span>
                        </TooltipContent>
                      </Tooltip>
                    </TooltipProvider>
                  ) : (
                    row.n_seeds
                  )}
                </TableCell>
                <TableCell
                  className={cn(
                    "text-right tabular-nums",
                    id === best.hv
                      ? "font-semibold text-primary"
                      : "text-foreground",
                  )}
                >
                  {fmt(row.hv_diff_mean)}
                  {row.hv_diff_std > 0 && (
                    <span className="ml-1 text-muted-foreground/70">
                      ±{fmt(row.hv_diff_std)}
                    </span>
                  )}
                </TableCell>
                <TableCell
                  className={cn(
                    "text-right tabular-nums",
                    id === best.igd
                      ? "font-semibold text-primary"
                      : "text-foreground",
                  )}
                >
                  {fmt(row.igd_mean)}
                  {row.igd_std > 0 && (
                    <span className="ml-1 text-muted-foreground/70">
                      ±{fmt(row.igd_std)}
                    </span>
                  )}
                </TableCell>
                <TableCell className="text-right tabular-nums text-muted-foreground">
                  {row.gt_calls}
                  {row.gt_calls_std > 0 && (
                    <span className="ml-1 text-muted-foreground/70">
                      ±{fmt(row.gt_calls_std)}
                    </span>
                  )}
                </TableCell>
                <TableCell
                  className={cn(
                    "text-right tabular-nums",
                    id === best.time
                      ? "font-semibold text-primary"
                      : "text-muted-foreground",
                  )}
                >
                  {fmt(row.optimizer_time_s)}
                  {row.optimizer_time_s_std > 0 && (
                    <span className="ml-1 text-muted-foreground/70">
                      ±{fmt(row.optimizer_time_s_std)}
                    </span>
                  )}
                </TableCell>
                {onDelete && (
                  <TableCell className="w-8 p-1">
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      onClick={() => setDeleteTarget(row)}
                      aria-label={`Delete ${row.optimizer} · ${row.synthetic_function}`}
                      className="size-7 text-muted-foreground/70 hover:bg-destructive/10 hover:text-destructive"
                    >
                      <Trash2 className="size-3.5" />
                    </Button>
                  </TableCell>
                )}
              </TableRow>
            );
          })}
        </TableBody>
      </Table>

      <ConfirmDialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open && !deleting) setDeleteTarget(null);
        }}
        title="Delete benchmark result"
        description={
          deleteTarget
            ? `Permanently delete the CSV for ${deleteTarget.optimizer} on ${deleteTarget.synthetic_function}? This removes every seed for that pair.`
            : ""
        }
        confirmLabel="Delete"
        destructive
        loading={deleting}
        onConfirm={handleDeleteConfirm}
      />
    </div>
  );
}

interface SortableHeadProps {
  label: string;
  sortKey: SortKey;
  activeKey: SortKey;
  dir: SortDir;
  onClick: (key: SortKey) => void;
  align?: "left" | "right";
}

function SortableHead({
  label,
  sortKey,
  activeKey,
  dir,
  onClick,
  align = "left",
}: SortableHeadProps) {
  const isActive = sortKey === activeKey;
  const Icon = !isActive ? ChevronsUpDown : dir === "asc" ? ArrowUp : ArrowDown;
  return (
    <TableHead
      className={cn(
        "h-9 text-[11px] font-semibold uppercase tracking-wider",
        align === "right" ? "text-right" : "text-left",
        isActive ? "text-foreground" : "text-muted-foreground",
      )}
    >
      <button
        type="button"
        onClick={() => onClick(sortKey)}
        className={cn(
          "inline-flex items-center gap-1 select-none rounded outline-none",
          "hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring",
          align === "right" && "flex-row-reverse",
        )}
      >
        <Icon
          className={cn(
            "size-3 shrink-0",
            isActive ? "text-foreground" : "text-muted-foreground/50",
          )}
        />
        <span>{label}</span>
      </button>
    </TableHead>
  );
}
