/**
 * ProfileDetail — calibration profile inspector. Two big stat tiles up top,
 * then key/value sections grouped by category. No card chrome — sections are
 * delineated by uppercase eyebrow + thin border.
 */
import React from "react";
import type { CalibrationProfile } from "@/types/costModel";

interface Props {
  profile: CalibrationProfile;
}

interface CoeffGroup {
  title: string;
  items: { label: string; key: string; format?: (v: number) => string }[];
}

const GROUPS: CoeffGroup[] = [
  {
    title: "Efficiency",
    items: [
      { label: "BLAS", key: "blas_efficiency" },
      { label: "SIMD", key: "simd_efficiency" },
      { label: "FastScan", key: "pq_fastscan_efficiency" },
      { label: "FastScan Batch", key: "pq_fastscan_batch_efficiency" },
      { label: "Standard PQ", key: "pq_standard_efficiency" },
      { label: "IVF-Flat", key: "flat_scan_efficiency" },
    ],
  },
  {
    title: "Overhead",
    items: [
      { label: "Search Fixed (us)", key: "flat_fixed_overhead_us", format: (v) => v.toFixed(2) },
      { label: "FastScan / Probe (us)", key: "pq_fastscan_per_probe_overhead_us", format: (v) => v.toFixed(3) },
      { label: "Standard / Probe (us)", key: "pq_standard_per_probe_overhead_us", format: (v) => v.toFixed(3) },
      { label: "Flat / Probe (us)", key: "flat_per_probe_overhead_us", format: (v) => v.toFixed(3) },
      { label: "PQ Setup / Dim (us)", key: "pq_setup_per_dim_us", format: (v) => v.toFixed(5) },
    ],
  },
  {
    title: "RAP — Standard",
    items: [
      { label: "c1", key: "pq_rap_standard_c1", format: (v) => v.toFixed(2) },
      { label: "c2", key: "pq_rap_standard_c2", format: (v) => v.toFixed(2) },
      { label: "c3", key: "pq_rap_standard_c3", format: (v) => v.toFixed(2) },
    ],
  },
  {
    title: "Memory Model",
    items: [
      { label: "Scattered BW Ceiling", key: "scattered_bw_ceiling" },
      { label: "NUMA Local (bytes)", key: "numa_local_bytes", format: (v) => v.toLocaleString() },
      { label: "FastScan BW Ceiling", key: "pq_fastscan_scan_bw_ceiling" },
    ],
  },
  {
    title: "FastScan Batch",
    items: [
      { label: "QBS Floor", key: "pq_fastscan_qbs_floor" },
      { label: "Overhead Alpha", key: "pq_fastscan_overhead_alpha", format: (v) => v.toFixed(4) },
    ],
  },
];

function StatTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-1 flex-col">
      <span className="text-2xl font-semibold tabular-nums tracking-tight text-foreground">
        {value}
      </span>
      <span className="text-[11px] uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
    </div>
  );
}

export function ProfileDetail({ profile }: Props) {
  return (
    <div className="space-y-8">
      {/* Top stats */}
      <div className="flex items-end gap-12">
        <StatTile
          label="MAPE (Latency)"
          value={`${profile.mape_latency.toFixed(2)}%`}
        />
        <StatTile
          label="Max APE"
          value={`${profile.max_ape_latency.toFixed(2)}%`}
        />
      </div>

      {/* Coefficient groups */}
      {GROUPS.map((group) => {
        const visibleItems = group.items.filter(
          (item) => (profile[item.key] as number) !== undefined,
        );
        if (visibleItems.length === 0) return null;
        return (
          <section key={group.title} className="space-y-2">
            <h3 className="border-b border-border pb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
              {group.title}
            </h3>
            <ul className="divide-y divide-border">
              {visibleItems.map((item) => {
                const value = profile[item.key] as number;
                const fmt = item.format ?? ((v: number) => v.toFixed(4));
                return (
                  <li
                    key={item.key}
                    className="flex items-center justify-between py-1.5"
                  >
                    <span className="text-xs text-muted-foreground">
                      {item.label}
                    </span>
                    <span className="font-mono text-xs font-medium tabular-nums text-foreground">
                      {fmt(value)}
                    </span>
                  </li>
                );
              })}
            </ul>
          </section>
        );
      })}
    </div>
  );
}
