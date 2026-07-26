/**
 * BenchmarkTable — actual-vs-predicted MAPE summary, shadcn <Table>, no card.
 */
import React from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { BenchmarkSummary } from "@/types/costModel";

interface Props {
  summary: BenchmarkSummary[];
}

export function BenchmarkTable({ summary }: Props) {
  if (summary.length === 0) {
    return (
      <div className="py-8 text-center">
        <span className="text-sm text-muted-foreground">
          No benchmark results available
        </span>
      </div>
    );
  }

  return (
    <section className="space-y-2">
      <div className="flex items-baseline justify-between">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Benchmark Summary
        </h3>
        <span className="text-[11px] text-muted-foreground/80">
          {summary.length} {summary.length === 1 ? "config" : "configs"}
        </span>
      </div>
      <Table>
        <TableHeader>
          <TableRow className="hover:bg-transparent">
            <TableHead className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              Config
            </TableHead>
            <TableHead className="text-right text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              MAPE
            </TableHead>
            <TableHead className="text-right text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              Max APE
            </TableHead>
            <TableHead className="text-right text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              Points
            </TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {summary.map((row) => (
            <TableRow key={row.label}>
              <TableCell className="truncate font-mono text-xs text-muted-foreground">
                {row.label}
              </TableCell>
              <TableCell className="text-right tabular-nums text-xs font-medium text-foreground">
                {row.mape_latency.toFixed(1)}%
              </TableCell>
              <TableCell className="text-right tabular-nums text-xs text-muted-foreground">
                {row.max_ape_latency.toFixed(1)}%
              </TableCell>
              <TableCell className="text-right tabular-nums text-xs text-muted-foreground">
                {row.num_points}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </section>
  );
}
