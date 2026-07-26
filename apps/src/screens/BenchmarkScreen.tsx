/**
 * BenchmarkScreen — the global Benchmark page. Two modes:
 *   - RAG Projects        — compare finished optimization projects by
 *     normalized hypervolume (ProjectBenchmark). The default: comparing real
 *     runs is the page's primary daily use.
 *   - Synthetic Functions — optimizer benchmark over the project's synthetic
 *     function(s) (the existing OptimizerScreen, embedded headerless).
 */
import React, { useState } from "react";
import { FlaskConical, FolderGit2, BarChart3 } from "lucide-react";
import { PageHeader, PageHeaderTitle } from "@/components/ui/PageHeader";
import { cn } from "@/lib/utils";
import OptimizerScreen from "./OptimizerScreen";
import { ProjectBenchmark } from "@/components/optimizer/ProjectBenchmark";

type Mode = "synthetic" | "projects";

export default function BenchmarkScreen() {
  const [mode, setMode] = useState<Mode>("projects");

  return (
    <div className="flex h-svh flex-col overflow-hidden min-h-0">
      <PageHeader>
        <PageHeaderTitle icon={BarChart3} title="Benchmark" />
        <div className="flex gap-0.5 rounded-lg bg-muted p-0.5">
          {([
            { k: "projects" as const, label: "RAG Projects", Icon: FolderGit2 },
            { k: "synthetic" as const, label: "Optimizer Benchmark", Icon: FlaskConical },
          ]).map(({ k, label, Icon }) => {
            const active = mode === k;
            return (
              <button
                key={k}
                type="button"
                onClick={() => setMode(k)}
                className={cn(
                  "flex cursor-pointer items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[11px] transition-colors",
                  active ? "bg-primary/10 font-semibold text-primary" : "text-muted-foreground hover:bg-muted",
                )}
              >
                <Icon size={11} className={active ? "text-primary" : "text-muted-foreground"} />
                {label}
              </button>
            );
          })}
        </div>
      </PageHeader>

      <div className="flex flex-1 min-h-0 flex-col">
        {mode === "synthetic" ? <OptimizerScreen hideHeader /> : <ProjectBenchmark />}
      </div>
    </div>
  );
}
