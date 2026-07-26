/**
 * QualityTab — the `eval_backend_setting` section, conditional on
 * global.eval_backend:
 *   static_gt → combined_quality + a metrics list (name + judge model)
 *   flashrag  → a metrics multiselect + the flashrag framework block
 * The valid metric vocabulary per backend comes from the server registry.
 */
import React from "react";
import { Target, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import type { SectionTabProps } from "./SetupTab";
import {
  SectionTitle, Row, TextField, SelectField, ChipMultiSelect,
} from "./SectionFields";

interface MetricEntry { metric_name?: string; model?: string; [k: string]: unknown; }

export function QualityTab({ config, patch, registries }: SectionTabProps) {
  const backend = ((config.global ?? {}) as any).eval_backend ?? "static_gt";
  const ebs = (config.eval_backend_setting ?? {}) as Record<string, any>;
  const metrics: MetricEntry[] = Array.isArray(ebs.metrics) ? ebs.metrics : [];
  const vocab = backend === "flashrag" ? registries.metrics.flashrag : registries.metrics.static_gt;

  const setEbs = (k: string, v: unknown) => patch((c) => { c.eval_backend_setting = { ...(c.eval_backend_setting ?? {}), [k]: v }; });
  const setMetrics = (m: MetricEntry[]) => setEbs("metrics", m);

  return (
    <div className="mx-auto max-w-2xl space-y-7 p-7 pb-12">
      <div className="flex items-center gap-2.5">
        <Target size={18} className="text-primary" />
        <span className="text-lg font-semibold tracking-tight text-foreground">Quality ({backend})</span>
      </div>

      {backend === "flashrag" ? (
        <>
          <section className="space-y-2">
            <SectionTitle>Metrics (FlashRAG vocabulary)</SectionTitle>
            <Row label="Metrics" hint="eval_backend_setting.metrics">
              <ChipMultiSelect options={vocab}
                value={metrics.map((m) => m.metric_name).filter(Boolean) as string[]}
                onChange={(names) => setMetrics(names.map((n) => ({ metric_name: n })))} />
            </Row>
          </section>
          <Separator />
          <section className="space-y-2">
            <SectionTitle>FlashRAG framework (eval_backend_setting.flashrag)</SectionTitle>
            <Row label="Framework" hint="flashrag.framework">
              <SelectField value={(ebs.flashrag ?? {}).framework} options={["vllm", "openai", "hf", "fschat"]}
                onChange={(v) => setEbs("flashrag", { ...(ebs.flashrag ?? {}), framework: v })} placeholder="vllm" />
            </Row>
            <p className="pt-1 text-[11px] text-muted-foreground/70">
              model2path and advanced flashrag options stay editable in the YAML tab.
            </p>
          </section>
        </>
      ) : (
        <>
          <section className="space-y-2">
            <SectionTitle>Combined quality (the optimizer's scalar)</SectionTitle>
            <Row label="Mode" hint="combined_quality.mode">
              <SelectField value={(ebs.combined_quality ?? {}).mode ?? "mean"} options={registries.enums.combined_quality_mode}
                onChange={(v) => setEbs("combined_quality", { ...(ebs.combined_quality ?? {}), mode: v })} />
            </Row>
            <Row label="Target metrics" hint="combined_quality.metrics (subset folded into `quality`)">
              <ChipMultiSelect options={metrics.map((m) => m.metric_name).filter(Boolean) as string[]}
                value={Array.isArray((ebs.combined_quality ?? {}).metrics) ? ebs.combined_quality.metrics : []}
                onChange={(v) => setEbs("combined_quality", { ...(ebs.combined_quality ?? {}), metrics: v })} />
            </Row>
          </section>

          <Separator />

          <section className="space-y-3">
            <div className="flex items-center justify-between">
              <SectionTitle>Metrics ({metrics.length})</SectionTitle>
              <Button type="button" variant="outline" size="sm"
                onClick={() => setMetrics([...metrics, { metric_name: vocab[0] ?? "", model: "openai/deepseek/deepseek-v4-flash", param_deps: null }])}>
                <Plus size={10} className="text-primary" />
                Add
              </Button>
            </div>
            {metrics.length === 0 && <span className="text-xs italic text-muted-foreground">No metrics</span>}
            {metrics.map((m, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="flex-[0.45]">
                  <SelectField value={m.metric_name} options={vocab}
                    onChange={(v) => setMetrics(metrics.map((x, j) => (j === i ? { ...x, metric_name: v } : x)))} />
                </div>
                <div className="flex-[0.5]">
                  <TextField value={m.model ?? ""} mono placeholder="judge model (optional)"
                    onChange={(v) => setMetrics(metrics.map((x, j) => (j === i ? { ...x, model: v || undefined } : x)))} />
                </div>
                <Button type="button" variant="ghost" size="icon" className="size-[22px] hover:bg-destructive/15"
                  onClick={() => setMetrics(metrics.filter((_, j) => j !== i))}>
                  <Trash2 size={11} className="text-destructive" />
                </Button>
              </div>
            ))}
            <p className="pt-1 text-[11px] text-muted-foreground/70">
              Per-metric param_deps (stage attribution) stay editable in the YAML tab.
            </p>
          </section>
        </>
      )}
    </div>
  );
}
