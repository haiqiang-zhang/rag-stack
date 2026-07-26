import React, { useMemo, useCallback } from "react";
import { X, Download, SlidersHorizontal } from "lucide-react";

import { usePostAnalysis } from "@/context/PostAnalysisContext";
import { usePerformancePolicy } from "@/context/PerformancePolicyContext";
import { useEvaluations } from "@/hooks/useEvaluations";
import { useSummary } from "@/hooks/useSummary";
import { Dropdown, type DropdownOption } from "@/components/selectors/Dropdown";
import { DebouncedSlider, DebouncedNumberInput } from "@/components/selectors/DebouncedSlider";
import { computePareto, exportParetoCSV, type FormulaType } from "@/utils/pareto";
import {
  METRIC_LABEL,
  METRIC_SPEC,
  PERFORMANCE_METRIC_KEYS,
  type PerformanceMetricKey,
  type SelectionPolicy,
} from "@/utils/performanceSelection";
import { Separator } from "@/components/ui/separator";

// from old theme.ts
const BORDER_DEFAULT = "#E2E6EF";

const FORMULA_OPTIONS: DropdownOption[] = [
  { key: "mean", label: "Mean" },
  { key: "weighted_mean", label: "Weighted Mean" },
  { key: "min", label: "Min" },
  { key: "max", label: "Max" },
  { key: "geometric_mean", label: "Geometric Mean" },
];

const SELECTION_POLICY_OPTIONS: DropdownOption[] = [
  { key: "min_latency", label: "Min Latency (E2E)" },
  { key: "min_ttft", label: "Min TTFT" },
  { key: "min_tpot", label: "Min TPOT" },
  { key: "max_throughput", label: "Max Throughput" },
  { key: "weighted", label: "Weighted Sum" },
  { key: "constrained", label: "Constrained" },
];

const PERF_METRIC_OPTIONS: DropdownOption[] = PERFORMANCE_METRIC_KEYS.map((k) => ({
  key: k,
  label: METRIC_LABEL[k],
}));

interface PostAnalysisPanelProps {
  onClose: () => void;
}

export function PostAnalysisPanel({ onClose }: PostAnalysisPanelProps) {
  const {
    qualityKey, setQualityKey,
    formula, setFormula,
    weights, setWeights,
    getQuality,
  } = usePostAnalysis();
  const {
    selection: perfSelection,
    weights: perfWeights,
    constraint: perfConstraint,
    optimize: perfOptimize,
    setSelection: setPerfSelection,
    setWeights: setPerfWeights,
    setConstraint: setPerfConstraint,
    setOptimize: setPerfOptimize,
    getPerformance,
    getPerformanceResult,
  } = usePerformancePolicy();

  const { data: summary } = useSummary();
  const { evaluations } = useEvaluations(summary?.n_evaluations ?? 0);

  // Derive available quality metric keys from evaluation data
  const metricOptions: DropdownOption[] = useMemo(() => {
    const keys = new Set<string>();
    for (const ev of evaluations) {
      for (const k of Object.keys(ev.quality_metrics)) keys.add(k);
    }
    const sorted = [...keys].filter((k) => k !== "combined").sort();
    sorted.unshift("combined");
    return sorted.map((k) => ({
      key: k,
      label: k.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
    }));
  }, [evaluations]);

  // Sub-metric names (excluding "combined")
  const subMetricNames = useMemo(
    () => metricOptions.filter((o) => o.key !== "combined").map((o) => o.key),
    [metricOptions],
  );

  // Computed Pareto count for display. Mirrors DashboardContent so the count
  // shown in the panel matches what's drawn on the Pareto chart.
  const paretoCount = useMemo(
    () =>
      computePareto(
        evaluations,
        getQuality,
        getPerformance,
        (ev) => getPerformanceResult(ev).feasible,
      ).length,
    [evaluations, getQuality, getPerformance, getPerformanceResult],
  );

  const handleWeightChange = useCallback(
    (metric: string, value: number) => {
      setWeights({ ...weights, [metric]: value });
    },
    [weights, setWeights],
  );

  const handleExportCSV = useCallback(() => {
    const pareto = computePareto(
      evaluations,
      getQuality,
      getPerformance,
      (ev) => getPerformanceResult(ev).feasible,
    );
    const csv = exportParetoCSV(pareto, getQuality, getPerformance);
    if (!csv) return;

    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pareto_frontier_${qualityKey}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [evaluations, getQuality, getPerformance, getPerformanceResult, qualityKey]);

  return (
    <div className="flex h-full w-[300px] flex-col border-l border-border bg-card shadow-md">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div className="flex items-center gap-2">
          <SlidersHorizontal className="size-[15px] text-primary" />
          <span className="text-[13px] font-semibold text-foreground">Analysis Settings</span>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="cursor-pointer rounded-md p-1 hover:bg-primary/10"
        >
          <X className="size-[15px] text-muted-foreground" />
        </button>
      </div>

      <div className="flex-1 overflow-auto p-4" style={{ gap: 20 }}>
        <div className="flex flex-col gap-5">
          {/* Quality Metric group — metric + combination formula + weights */}
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-2">
              <span className="text-[10px] font-bold tracking-[0.8px] text-muted-foreground">
                QUALITY METRIC
              </span>
              <Dropdown
                options={metricOptions}
                selectedKey={qualityKey}
                onSelect={setQualityKey}
                placeholder="Select metric..."
              />
              <span className="text-[11px] text-muted-foreground">
                {paretoCount} Pareto point{paretoCount !== 1 ? "s" : ""} with current metric
              </span>
            </div>

            {qualityKey === "combined" && (
              <div className="flex flex-col gap-2">
                <span className="text-[10px] font-bold tracking-[0.8px] text-muted-foreground">
                  COMBINATION FORMULA
                </span>
                <Dropdown
                  options={FORMULA_OPTIONS}
                  selectedKey={formula}
                  onSelect={(k) => setFormula(k as FormulaType)}
                  placeholder="Select formula..."
                />
              </div>
            )}

            {qualityKey === "combined" && formula === "weighted_mean" && subMetricNames.length > 0 && (
              <div className="flex flex-col gap-2.5">
                <span className="text-[10px] font-bold tracking-[0.8px] text-muted-foreground">
                  SUB-METRIC WEIGHTS
                </span>
                {subMetricNames.map((metric) => (
                  <DebouncedSlider
                    key={metric}
                    label={metric.replace(/_/g, " ")}
                    value={weights[metric] ?? 1}
                    onCommit={(v) => handleWeightChange(metric, v)}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Performance Policy group — policy + weights/constraints */}
          <Separator className="bg-border" />
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-2">
              <span className="text-[10px] font-bold tracking-[0.8px] text-muted-foreground">
                PERFORMANCE POLICY
              </span>
              <Dropdown
                options={SELECTION_POLICY_OPTIONS}
                selectedKey={perfSelection}
                onSelect={(k) => setPerfSelection(k as SelectionPolicy)}
                placeholder="Select policy..."
              />
              <span className="text-[11px] text-muted-foreground">
                Picks the best row from the sweep per policy (real-time).
              </span>
            </div>

            {perfSelection === "weighted" && (
              <div className="flex flex-col gap-2.5">
                <span className="text-[10px] font-bold tracking-[0.8px] text-muted-foreground">
                  METRIC WEIGHTS
                </span>
                {PERF_METRIC_OPTIONS.map((opt) => {
                  const key = opt.key as PerformanceMetricKey;
                  return (
                    <DebouncedSlider
                      key={key}
                      label={opt.label}
                      value={perfWeights?.[key] ?? 0}
                      onCommit={(v) =>
                        setPerfWeights({ ...perfWeights, [key]: v })
                      }
                    />
                  );
                })}
              </div>
            )}

            {perfSelection === "constrained" && (
              <div className="flex flex-col gap-2.5">
                <span className="text-[10px] font-bold tracking-[0.8px] text-muted-foreground">
                  CONSTRAINTS
                </span>
                {PERFORMANCE_METRIC_KEYS.map((key) => {
                  const spec = METRIC_SPEC[key];
                  const threshold = perfConstraint?.[key] ?? "";
                  // Natural-sense thresholds: lower-is-better → "≤", higher-is-better → "≥".
                  const direction = spec.higherIsBetter ? "≥" : "≤";
                  return (
                    <div key={key} className="flex items-center gap-2.5">
                      <span className="flex-1 truncate text-[11px] text-muted-foreground">
                        {METRIC_LABEL[key]} {direction}
                      </span>
                      <DebouncedNumberInput
                        value={threshold as number | ""}
                        borderColor={BORDER_DEFAULT}
                        onCommit={(v) => {
                          const next = { ...perfConstraint };
                          if (v === undefined) delete next[key];
                          else next[key] = v;
                          setPerfConstraint(next);
                        }}
                      />
                      <span className="text-[10px] text-muted-foreground">{spec.unit}</span>
                    </div>
                  );
                })}
                <span className="mt-1 text-[10px] font-bold tracking-[0.8px] text-muted-foreground">
                  OPTIMIZE
                </span>
                <Dropdown
                  options={PERF_METRIC_OPTIONS}
                  selectedKey={perfOptimize ?? "throughput_qps"}
                  onSelect={(k) => setPerfOptimize(k as PerformanceMetricKey)}
                />
              </div>
            )}
          </div>

          {/* Export */}
          <Separator className="bg-border" />
          <div className="flex flex-col gap-2">
            <span className="text-[10px] font-bold tracking-[0.8px] text-muted-foreground">
              EXPORT
            </span>
            <button
              type="button"
              onClick={handleExportCSV}
              className="flex cursor-pointer items-center gap-2 rounded-lg border border-border bg-card px-3 py-2 hover:border-primary hover:bg-primary/10"
            >
              <Download className="size-[13px] text-primary" />
              <span className="text-xs font-medium text-foreground">Export Pareto CSV</span>
            </button>
          </div>
        </div>
      </div>

    </div>
  );
}
