/**
 * CostModelsScreen — top-level page for the Cost Models route.
 *
 * Page-level views:
 *   - Mapping: full-page CostModelMapping table
 *   - Models:  sidebar (ModelSelector) + per-model sub-tabs
 *              (Simulate · Overview · Profiles · Benchmarks)
 *   - Calibrate: unified cost-model calibration status + launcher
 *
 * Redesigned around shadcn primitives — same typography-driven, no-card-
 * chrome pattern as Settings / Quality Stores / Optimizer / RAG Optimize.
 *   - shadcn <Tabs> for the per-model sub-tabs
 *   - <SidePanel> (consistent w/ other split-view screens)
 *   - <Badge> for the active-model chip
 *   - PageHeaderTitle / PageHeaderActions slots
 */
import React, { useState } from "react";
import { Loader2, Cpu, ArrowLeft, SlidersHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  PageHeader,
  PageHeaderTitle,
  PageHeaderActions,
} from "@/components/ui/PageHeader";
import { Badge } from "@/components/ui/badge";
import { SidePanel } from "@/components/ui/SidePanel";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

import { useRegistry } from "@/hooks/useRegistry";
import { useProfiles } from "@/hooks/useProfiles";
import { useBenchmarkResults } from "@/hooks/useBenchmarkResults";

import { ModelSelector } from "@/components/cost-model/ModelSelector";
import { ModelOverview } from "@/components/cost-model/ModelOverview";
import { ProfileList } from "@/components/cost-model/ProfileList";
import { ProfileDetail } from "@/components/cost-model/ProfileDetail";
import { ProfileComparison } from "@/components/cost-model/ProfileComparison";
import { BenchmarkTable } from "@/components/cost-model/BenchmarkTable";
import { BenchmarkChart } from "@/components/cost-model/BenchmarkChart";
import { StageBreakdownChart } from "@/components/cost-model/StageBreakdownChart";
import { SimulatePanel } from "@/components/cost-model/SimulatePanel";
import { hasComponentEstimateForm } from "@/components/cost-model/estimateForms";
import { CostModelMapping } from "@/components/cost-model/CostModelMapping";
import { CalibrationProfilesView } from "@/components/cost-model/calibration/CalibrationProfilesView";
import { UnifiedCalibrationView } from "@/components/cost-model/calibration/UnifiedCalibrationView";

type PageView = "mapping" | "models" | "calibrate";
type ModelSubTab =
  | "simulate"
  | "overview"
  | "profiles"
  | "benchmarks";

const ALL_SUB_TABS: { key: ModelSubTab; label: string; capability?: string }[] =
  [
    { key: "overview", label: "Overview" },
    { key: "simulate", label: "Component estimate" },
    { key: "profiles", label: "Profiles", capability: "calibration_profiles" },
    { key: "benchmarks", label: "Benchmarks", capability: "benchmarking" },
  ];

export default function CostModelsScreen() {
  const { models, loading, error, refetch } = useRegistry();

  const [pageView, setPageView] = useState<PageView>("mapping");
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [modelSubTab, setModelSubTab] = useState<ModelSubTab>("overview");
  const [selectedProfile, setSelectedProfile] = useState<number | null>(null);

  React.useEffect(() => {
    if (!selectedModel && models.length > 0) {
      setSelectedModel(models[0].name);
    }
  }, [models, selectedModel]);

  const model = models.find((m) => m.name === selectedModel) ?? null;
  const shouldLoadFaissProfiles =
    pageView === "models" &&
    modelSubTab === "profiles" &&
    model?.name === "faiss_ivf";
  const shouldLoadBenchmarks =
    pageView === "models" &&
    modelSubTab === "benchmarks" &&
    !!model?.capabilities.includes("benchmarking");
  const { profiles } = useProfiles(shouldLoadFaissProfiles);
  const { rows, summary } = useBenchmarkResults(undefined, shouldLoadBenchmarks);
  const subTabs = ALL_SUB_TABS.filter(
    (t) => (t.key !== "simulate" || !!model && hasComponentEstimateForm(model.name)) &&
      (!t.capability || (model && model.capabilities.includes(t.capability))),
  );

  const handleSelectModel = (name: string) => {
    setSelectedModel(name);
    setPageView("models");
    setModelSubTab("overview");
    setSelectedProfile(null);
  };

  // ── Loading / Error states ─────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex flex-1 flex-col bg-background">
        <PageHeader>
          <PageHeaderTitle icon={Cpu} title="Cost Models" />
        </PageHeader>
        <div className="flex flex-1 flex-col items-center justify-center gap-3">
          <Loader2 className="size-6 animate-spin text-muted-foreground" />
          <span className="text-sm text-muted-foreground">
            Loading cost models…
          </span>
        </div>
      </div>
    );
  }
  if (error) {
    return (
      <div className="flex flex-1 flex-col bg-background">
        <PageHeader>
          <PageHeaderTitle icon={Cpu} title="Cost Models" />
        </PageHeader>
        <div className="flex flex-1 flex-col items-center justify-center gap-2 p-5">
          <p className="text-base font-semibold text-destructive">
            Could not load cost models
          </p>
          <p className="text-xs text-muted-foreground">{error.message}</p>
          <Button variant="outline" onClick={() => void refetch()}>Try again</Button>
        </div>
      </div>
    );
  }

  // ── Mapping page (full-width table) ────────────────────────────────
  if (pageView === "mapping") {
    return (
      <div className="flex flex-1 flex-col bg-background">
        <PageHeader>
          <PageHeaderTitle
            icon={Cpu}
            title="Cost Models"
            badge={
              <Badge variant="secondary" className="text-[11px]">
                {models.length}
              </Badge>
            }
          />
          <PageHeaderActions>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setPageView("calibrate")}
            >
              <SlidersHorizontal className="size-4" />
              Calibrate
            </Button>
          </PageHeaderActions>
        </PageHeader>
        <CostModelMapping models={models} onOpenModel={handleSelectModel} />
      </div>
    );
  }

  if (pageView === "calibrate") {
    return (
      <div className="flex flex-1 min-w-0 flex-col bg-background">
        <PageHeader>
          <PageHeaderTitle
            icon={SlidersHorizontal}
            title="Calibration"
            prefix={
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={() => setPageView("mapping")}
                className="size-8"
                aria-label="Back to cost models"
              >
                <ArrowLeft className="size-4 text-muted-foreground" />
              </Button>
            }
          />
        </PageHeader>
        <UnifiedCalibrationView />
      </div>
    );
  }

  // ── Models page (sidebar + tab content) ────────────────────────────
  return (
    <Tabs
      value={modelSubTab}
      onValueChange={(v) => setModelSubTab(v as ModelSubTab)}
      className="flex h-svh max-h-svh min-h-0 min-w-0 flex-col gap-0 overflow-hidden bg-background"
    >
      <PageHeader className="shrink-0 gap-3 px-4">
        <PageHeaderTitle
          icon={Cpu}
          title="Cost Models"
          prefix={
            <Button
              type="button"
              variant="ghost"
              size="icon"
              onClick={() => setPageView("mapping")}
              className="size-8"
              aria-label="Back to cost models"
            >
              <ArrowLeft className="size-4 text-muted-foreground" />
            </Button>
          }
          badge={
            model && (
              <>
                <div className="h-4 w-px bg-border" />
                <Badge
                  variant="secondary"
                  className="hidden max-w-[220px] truncate font-mono text-[11px] font-medium sm:block"
                >
                  {model.name}
                </Badge>
              </>
            )
          }
        />
        {model && (
          <PageHeaderActions>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setPageView("calibrate")}
            >
              <SlidersHorizontal className="size-4" />
              Calibrate
            </Button>
          </PageHeaderActions>
        )}
      </PageHeader>

      <div className="flex min-w-0 shrink-0 flex-wrap items-center gap-3 border-b px-4 py-2">
        <div className="min-w-0 w-full sm:max-w-[260px] xl:hidden">
          <Select value={selectedModel ?? undefined} onValueChange={handleSelectModel}>
            <SelectTrigger aria-label="Cost model" className="w-full min-w-0 font-mono text-xs"><SelectValue placeholder="Select a cost model" /></SelectTrigger>
            <SelectContent>{models.map((item) => <SelectItem key={item.name} value={item.name}>{item.name}</SelectItem>)}</SelectContent>
          </Select>
        </div>
        <div className="min-w-0 max-w-full overflow-x-auto">
          <TabsList>
            {subTabs.map((tab) => <TabsTrigger key={tab.key} value={tab.key}>{tab.label}</TabsTrigger>)}
          </TabsList>
        </div>
      </div>

      <div className="flex flex-1 min-h-0 min-w-0 overflow-hidden">
        <SidePanel className="hidden min-h-0 w-[220px] xl:flex">
          <ModelSelector
            models={models}
            selected={selectedModel}
            onSelect={handleSelectModel}
          />
        </SidePanel>

        {model ? (
          <div className="flex flex-1 min-h-0 min-w-0 flex-col overflow-hidden">
            {/* Simulate / Profiles take their own scroll layout */}
            <TabsContent
              value="simulate"
              className="flex flex-1 min-h-0 min-w-0 flex-col m-0 overflow-hidden"
            >
              <SimulatePanel key={model.name} model={model} />
            </TabsContent>

            {model.capabilities.includes("calibration_profiles") &&
              model.name !== "faiss_ivf" && (
                <TabsContent
                  value="profiles"
                  className="flex flex-1 min-h-0 min-w-0 flex-col m-0 overflow-hidden"
                >
                  <CalibrationProfilesView key={model.name} costModel={model.name} />
                </TabsContent>
              )}

            {/* Overview / Benchmarks / fallback Profiles are simple scrollable */}
            <TabsContent
              value="overview"
              className="flex-1 min-w-0 m-0 overflow-auto"
            >
              <div className="mx-auto max-w-[1100px] px-8 pt-8 pb-20">
                <ModelOverview model={model} />
              </div>
            </TabsContent>

            {!(
              model.capabilities.includes("calibration_profiles") &&
              model.name !== "faiss_ivf"
            ) && (
              <TabsContent
                value="profiles"
                className="flex-1 min-w-0 m-0 overflow-auto"
              >
                <div className="mx-auto max-w-[1100px] px-8 pt-8 pb-20 space-y-6">
                  <ProfileList
                    profiles={profiles}
                    selectedIndex={selectedProfile}
                    onSelect={setSelectedProfile}
                  />
                  {selectedProfile !== null && profiles[selectedProfile] && (
                    <ProfileDetail profile={profiles[selectedProfile]} />
                  )}
                  <ProfileComparison profiles={profiles} />
                </div>
              </TabsContent>
            )}

            <TabsContent
              value="benchmarks"
              className="flex-1 min-w-0 m-0 overflow-auto"
            >
              <div className="mx-auto max-w-[1200px] px-8 pt-8 pb-20 space-y-6">
                <BenchmarkChart rows={rows} />
                <StageBreakdownChart rows={rows} />
                <BenchmarkTable summary={summary} />
              </div>
            </TabsContent>
          </div>
        ) : (
          <div className="flex flex-1 items-center justify-center">
            <p className="text-sm text-muted-foreground">
              Select a cost model to view details
            </p>
          </div>
        )}
      </div>
    </Tabs>
  );
}
