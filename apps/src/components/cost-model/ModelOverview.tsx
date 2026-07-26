/**
 * ModelOverview — flat sections, no card chrome.
 * Description / Mapped Modules / Capabilities each lead with a small
 * uppercase eyebrow + thin border-b, then the content sits directly on
 * the page background.
 */
import React from "react";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import type { CostModelInfo } from "@/types/costModel";

interface Props {
  model: CostModelInfo;
}

const CAPABILITY_HINT: Record<string, string> = {
  calibration_profiles:
    "Saved component calibration records — see Profiles. Matching a record to a project is checked during CM binding.",
  benchmarking:
    "Run measured-vs-predicted benchmark grids — see Benchmarks tab",
  calibration_runs:
    "Launch benchmark / calibration jobs — see Calibrate tab",
  presets: "Predefined hardware presets available for this cost model",
};

function Eyebrow({ children }: { children: React.ReactNode }) {
  return (
    <span className="block border-b border-border pb-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
      {children}
    </span>
  );
}

export function ModelOverview({ model }: Props) {
  return (
    <div className="space-y-8">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold tracking-tight text-foreground">
          {model.name}
        </h2>
        <p className="font-mono text-xs text-muted-foreground">
          {model.class_name}
        </p>
      </div>

      <Separator />

      <section className="space-y-2">
        <Eyebrow>Description</Eyebrow>
        <p className="text-sm leading-relaxed text-muted-foreground">
          {model.description}
        </p>
      </section>

      <p className="rounded-lg border bg-muted/20 p-4 text-sm leading-relaxed text-muted-foreground">
        This is a registered component cost model. A project’s CM Preview shows
        the real RAG-IR calls, compiled operations, deployment, and calibration
        selected for that workload.
      </p>

      {model.mapped_module_types.length > 0 && (
        <section className="space-y-2">
          <Eyebrow>Mapped module types</Eyebrow>
          <div className="flex flex-wrap gap-1.5 pt-1">
            {model.mapped_module_types.map((t) => (
              <Badge
                key={t}
                variant="secondary"
                className="font-mono text-[11px]"
              >
                {t}
              </Badge>
            ))}
          </div>
        </section>
      )}

      {model.capabilities.length > 0 && (
        <section className="space-y-3">
          <Eyebrow>Capabilities</Eyebrow>
          <ul className="divide-y divide-border">
            {model.capabilities.map((c) => (
              <li key={c} className="flex items-center gap-3 py-2">
                <Badge
                  variant="secondary"
                  className="bg-emerald-500/10 font-mono text-[11px] text-emerald-700 dark:text-emerald-300"
                >
                  {c}
                </Badge>
                {CAPABILITY_HINT[c] && (
                  <span className="flex-1 text-xs text-muted-foreground">
                    {CAPABILITY_HINT[c]}
                  </span>
                )}
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
