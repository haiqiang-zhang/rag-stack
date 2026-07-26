import { useCallback, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SummaryPanel } from "@/components/cards/SummaryPanel";
import { ConvergenceCard } from "./ConvergenceCard";
import { useEvalData } from "@/context/EvalDataContext";
import { useProject } from "@/context/ProjectContext";
import { usePostAnalysis } from "@/context/PostAnalysisContext";
import { createDataClient } from "@/lib/config";

/**
 * Project-level landing: aggregate health + optimizer convergence, plus a
 * one-click export of the current Pareto frontier's resolved configs.
 * Per-eval detail lives in the Performance / Config tabs (pick an eval in the
 * left navigator).
 */
export function OverviewTab() {
  const {
    summary,
    enrichedEvals,
    clientPareto,
    getQuality,
    getPerformance,
    getPerformanceResult,
  } = useEvalData();
  const { projectBaseUrl } = useProject();
  const { qualityKey } = usePostAnalysis();
  const [exporting, setExporting] = useState(false);

  const handleExport = useCallback(async () => {
    if (!projectBaseUrl || clientPareto.length === 0) return;
    setExporting(true);
    try {
      const client = createDataClient(projectBaseUrl);
      const configs: Record<string, unknown>[] = [];
      for (const ev of clientPareto) {
        try {
          const c = await client.getPipelineConfig(ev.eval_id);
          configs.push({
            _eval_id: ev.eval_id,
            _performance: getPerformance(ev),
            _quality: getQuality(ev),
            ...c,
          });
        } catch {
          /* skip evals whose config is unavailable */
        }
      }
      const yaml = await import("js-yaml");
      const content = configs
        .map(
          (c, i) =>
            `# Pareto point ${i} (Eval #${c._eval_id})\n` +
            yaml.dump(c, { lineWidth: -1 }),
        )
        .join("\n---\n\n");
      const blob = new Blob([content], { type: "text/yaml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `pareto_configs_${qualityKey}.yaml`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  }, [clientPareto, getPerformance, getQuality, projectBaseUrl, qualityKey]);

  return (
    <div className="flex-1 min-w-0 overflow-auto p-6 pb-12">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-foreground">
            Project overview
          </h2>
          <p className="text-xs text-muted-foreground">
            Search health and optimizer progress across all evaluations.
          </p>
        </div>
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={handleExport}
          disabled={exporting || clientPareto.length === 0}
          title="Download every Pareto-frontier config as YAML"
        >
          {exporting ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <Download className="size-4" />
          )}
          Export Pareto configs ({clientPareto.length})
        </Button>
      </div>

      <SummaryPanel
        summary={summary}
        evaluations={enrichedEvals}
        paretoCount={clientPareto.length}
        getQuality={getQuality}
        getPerformanceResult={getPerformanceResult}
      />
      <div className="mt-4">
        <ConvergenceCard />
      </div>
    </div>
  );
}
