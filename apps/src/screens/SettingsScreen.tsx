/**
 * SettingsScreen — top-level page for the Settings route.
 *
 * Web port of apps/app/(tabs)/settings.tsx. Single layout with a left
 * sidebar of section categories (wide ≥600px) or a horizontal pill bar
 * (narrow), and a right pane that renders the selected section's form.
 *
 * Sections: connection, generators (Claude Code SDK + Codex CLI health
 * checks), keys (env keys editor + .env path), rag, dataset, cost-model
 * (with discovered benchmark folders), optimizer.
 *
 * All hooks, save/dirty tracking, and side effects are preserved verbatim.
 * Only the JSX shell changed (Tamagui → Tailwind divs/spans, ScrollView →
 * overflow-auto div, Spinner → `<Loader2 className="animate-spin" />`,
 * TextInput → `<Input>` from shadcn). Width detection now uses a
 * ResizeObserver on the content wrapper (was `onLayout` in RN).
 */
import React, { useState, useEffect, useCallback } from "react";
import { useNavigate, useSearch } from "@tanstack/react-router";
import {
  Loader2,
  Check,
  RefreshCw,
  Cpu,
  Zap,
  BarChart3,
  Database,
  FolderOpen,
  Key,
  Plus,
  X,
  Eye,
  EyeOff,
  Wifi,
  WifiOff,
  Bot,
  CircleCheckBig,
  CircleX,
  Settings as SettingsIcon,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";

import {
  fetchSettings, updateSettings, fetchBenchmarkFolders,
  fetchEnvKeys, updateEnvKeys,
  checkClaudeCodeHealth, type ClaudeCodeHealth,
  checkCodexCliHealth, type CodexCliHealth,
  type AppSettings, type BenchmarkFolder,
} from "@/api/costModelClient";

import { useConnection } from "@/context/ConnectionContext";


// -- Types -----------------------------------------------------------

type SectionKey = "connection" | "generators" | "rag" | "dataset" | "cost-model" | "optimizer" | "keys";

const SECTION_DESCRIPTIONS: Record<SectionKey, string> = {
  connection: "Backend API connection and authentication.",
  generators: "LLM client configuration for generation tasks.",
  keys: "API keys and secrets stored in the local .env file.",
  rag: "Directories used by the RAG optimization workflow.",
  dataset: "Configuration for the dataset generator.",
  "cost-model":
    "Where benchmark outputs and calibration profiles are read from / written to.",
  optimizer:
    "Where optimizer benchmark configs are read from and CSV results are written to.",
};

const SECTIONS: { key: SectionKey; label: string; icon: React.ElementType }[] = [
  { key: "connection", label: "Connection", icon: Wifi },
  { key: "generators", label: "Generator Clients", icon: Bot },
  { key: "keys", label: "Keys & Secrets", icon: Key },
  { key: "rag", label: "RAG Optimization", icon: BarChart3 },
  { key: "dataset", label: "Dataset Generation", icon: Database },
  { key: "cost-model", label: "Cost Model", icon: Cpu },
  { key: "optimizer", label: "Optimizer", icon: Zap },
];

const COMMON_ENV_KEYS = [
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "COHERE_API_KEY",
  "HUGGINGFACE_TOKEN",
  "VOYAGE_API_KEY",
  "JINA_API_KEY",
];

// -- Setting field ---------------------------------------------------

/**
 * Canonical settings row — Vercel/Stripe-style. Label on top, input below,
 * helper text last. No card wrapper. Use inside a `space-y-6` parent.
 */
function SettingField({
  label,
  description,
  value,
  onChange,
  placeholder,
  type = "text",
  mono = true,
}: {
  label: string;
  description?: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
  mono?: boolean;
}) {
  return (
    <div className="space-y-2">
      <Label className="text-sm font-medium">{label}</Label>
      <Input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className={mono ? "font-mono text-sm" : "text-sm"}
      />
      {description && (
        <p className="text-xs text-muted-foreground">{description}</p>
      )}
    </div>
  );
}

/** Section heading — title + optional description, followed by a separator. */
function SettingsSectionHeader({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <>
      <div className="flex items-start justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-xl font-semibold tracking-tight">{title}</h2>
          {description && (
            <p className="text-sm text-muted-foreground">{description}</p>
          )}
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </div>
      <Separator className="my-6" />
    </>
  );
}

// -- Connection Section ---------------------------------------------

function ConnectionSection() {
  const { backendUrl, isConnected, authRequired, disconnect } = useConnection();
  return (
    <div className="space-y-4">
      <DefRow label="Status">
        <Badge
          variant={isConnected ? "default" : "destructive"}
          className="gap-1.5"
        >
          {isConnected ? <Wifi className="size-3" /> : <WifiOff className="size-3" />}
          {isConnected ? "Connected" : "Disconnected"}
        </Badge>
      </DefRow>
      <Separator />
      <DefRow label="Backend URL">
        <span className="font-mono text-sm">
          {backendUrl || (
            <span className="text-muted-foreground italic">Not configured</span>
          )}
        </span>
      </DefRow>
      <Separator />
      <DefRow label="Authentication">
        <span className="text-sm">
          {authRequired ? "Token required" : "No authentication"}
        </span>
      </DefRow>
      {isConnected && (
        <>
          <Separator />
          <div className="flex justify-end">
            <Button type="button" variant="ghost" onClick={disconnect}>
              <WifiOff className="size-[13px] text-destructive" />
              <span className="text-xs font-medium text-destructive">Disconnect</span>
            </Button>
          </div>
        </>
      )}
    </div>
  );
}

/** Two-column key-value row — label-left muted, value-right normal. */
function DefRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd className="text-right">{children}</dd>
    </div>
  );
}

// -- Generator Clients Section --------------------------------------

function GeneratorClientsSection() {
  const [health, setHealth] = useState<ClaudeCodeHealth | null>(null);
  const [checking, setChecking] = useState(false);
  const [testModel, setTestModel] = useState("sonnet");

  const runCheck = useCallback(async () => {
    setChecking(true);
    try {
      setHealth(await checkClaudeCodeHealth(testModel));
    } catch (e) {
      setHealth({ installed: false, working: false, error: e instanceof Error ? e.message : String(e), model: null });
    } finally {
      setChecking(false);
    }
  }, [testModel]);

  // Auto-check on mount
  useEffect(() => { runCheck(); }, [runCheck]);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2.5">
        <Bot className="size-4 text-foreground" />
        <div>
          <h3 className="text-sm font-semibold">Claude Code SDK</h3>
          <p className="text-xs text-muted-foreground">
            Primary generator client for RAG evaluation
          </p>
        </div>
      </div>

      <div className="flex items-end gap-2">
        <div className="flex-1 space-y-2">
          <Label className="text-xs text-muted-foreground">Test Model</Label>
          <Input
            type="text"
            value={testModel}
            onChange={(e) => setTestModel(e.target.value)}
            placeholder="sonnet"
            autoCapitalize="none"
            autoCorrect="off"
            className="font-mono text-sm"
          />
        </div>
        <Button type="button" variant="outline" size="sm" onClick={runCheck} disabled={checking}>
          {checking ? <Loader2 className="size-4 animate-spin" /> : "Test"}
        </Button>
      </div>

      {health === null && checking ? (
        <div className="flex items-center gap-2 py-2">
          <Loader2 className="size-4 animate-spin text-muted-foreground" />
          <span className="text-xs text-muted-foreground">Checking...</span>
        </div>
      ) : health ? (
        <div className="space-y-3 pt-2">
          <DefRow label="SDK Installed">
            <HealthBadge ok={health.installed} okLabel="Yes" failLabel="No" />
          </DefRow>
          <Separator />
          <DefRow label="Health Check">
            <HealthBadge ok={health.working} okLabel="Passed" failLabel="Failed" />
          </DefRow>
          {health.response && (
            <>
              <Separator />
              <DefRow label="Response">
                <span className="font-mono text-xs">&quot;{health.response}&quot;</span>
              </DefRow>
            </>
          )}
          {health.error && (
            <div className="rounded-md border border-destructive/20 bg-destructive/5 p-3">
              <p className="font-mono text-xs text-destructive">{health.error}</p>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}

/** Tiny status pill — green check / red x. Shared by Generators + Codex sections. */
function HealthBadge({
  ok,
  okLabel,
  failLabel,
}: {
  ok: boolean;
  okLabel: string;
  failLabel: string;
}) {
  return ok ? (
    <span className="flex items-center gap-1 text-xs font-medium text-emerald-600">
      <CircleCheckBig className="size-[13px]" />
      {okLabel}
    </span>
  ) : (
    <span className="flex items-center gap-1 text-xs font-medium text-destructive">
      <CircleX className="size-[13px]" />
      {failLabel}
    </span>
  );
}

// -- Codex CLI Client Section --------------------------------------

function CodexCliClientSection() {
  const [health, setHealth] = useState<CodexCliHealth | null>(null);
  const [checking, setChecking] = useState(false);
  const [testModel, setTestModel] = useState("gpt-5.4");

  const runCheck = useCallback(async () => {
    setChecking(true);
    try {
      setHealth(await checkCodexCliHealth(testModel));
    } catch (e) {
      setHealth({ installed: false, working: false, error: e instanceof Error ? e.message : String(e), model: null });
    } finally {
      setChecking(false);
    }
  }, [testModel]);

  useEffect(() => { runCheck(); }, [runCheck]);

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2.5">
        <Bot className="size-4 text-foreground" />
        <div>
          <h3 className="text-sm font-semibold">Codex CLI</h3>
          <p className="text-xs text-muted-foreground">
            OpenAI <code className="font-mono">codex exec</code> wrapper
          </p>
        </div>
      </div>

      <div className="flex items-end gap-2">
        <div className="flex-1 space-y-2">
          <Label className="text-xs text-muted-foreground">Test Model</Label>
          <Input
            type="text"
            value={testModel}
            onChange={(e) => setTestModel(e.target.value)}
            placeholder="gpt-5.4"
            autoCapitalize="none"
            autoCorrect="off"
            className="font-mono text-sm"
          />
        </div>
        <Button type="button" variant="outline" size="sm" onClick={runCheck} disabled={checking}>
          {checking ? <Loader2 className="size-4 animate-spin" /> : "Test"}
        </Button>
      </div>

      {health === null && checking ? (
        <div className="flex items-center gap-2 py-2">
          <Loader2 className="size-4 animate-spin text-muted-foreground" />
          <span className="text-xs text-muted-foreground">Checking...</span>
        </div>
      ) : health ? (
        <div className="space-y-3 pt-2">
          <DefRow label="CLI Installed">
            <HealthBadge ok={health.installed} okLabel="Yes" failLabel="No" />
          </DefRow>
          <Separator />
          <DefRow label="Health Check">
            <HealthBadge ok={health.working} okLabel="Passed" failLabel="Failed" />
          </DefRow>
          {health.response && (
            <>
              <Separator />
              <DefRow label="Response">
                <span className="font-mono text-xs">&quot;{health.response}&quot;</span>
              </DefRow>
            </>
          )}
          {health.error && (
            <div className="rounded-md border border-destructive/20 bg-destructive/5 p-3">
              <p className="font-mono text-xs text-destructive">{health.error}</p>
            </div>
          )}
        </div>
      ) : null}
    </div>
  );
}

// -- Env Keys Section -----------------------------------------------

function EnvKeysSection({ envKeys, setEnvKeys, newKeyName, setNewKeyName, revealedKeys, setRevealedKeys, envFilePath, setEnvFilePath }: {
  envKeys: Record<string, string>;
  setEnvKeys: React.Dispatch<React.SetStateAction<Record<string, string>>>;
  newKeyName: string;
  setNewKeyName: (v: string) => void;
  revealedKeys: Set<string>;
  setRevealedKeys: React.Dispatch<React.SetStateAction<Set<string>>>;
  envFilePath: string;
  setEnvFilePath: (v: string) => void;
}) {
  const updateKey = (name: string, value: string) => setEnvKeys((prev) => ({ ...prev, [name]: value }));
  const removeKey = (name: string) => setEnvKeys((prev) => {
    const next = { ...prev };
    delete next[name];
    return next;
  });
  const toggleReveal = (name: string) => setRevealedKeys((prev) => {
    const next = new Set(prev);
    if (next.has(name)) next.delete(name); else next.add(name);
    return next;
  });
  const addKey = (name: string) => {
    const trimmed = name.trim().toUpperCase().replace(/[^A-Z0-9_]/g, "_");
    if (!trimmed || trimmed in envKeys) return;
    setEnvKeys((prev) => ({ ...prev, [trimmed]: "" }));
    setNewKeyName("");
  };

  // Keys already configured
  const configuredKeys = Object.keys(envKeys);
  // Suggest common keys not yet added
  const suggestions = COMMON_ENV_KEYS.filter((k) => !(k in envKeys));

  return (
    <div className="space-y-10">
      {/* .env file path */}
      <SettingField
        label=".env File Path"
        description="Path to the .env file. Leave empty for project root default."
        value={envFilePath}
        onChange={setEnvFilePath}
        placeholder=".env"
      />

      {/* Configured keys */}
      {configuredKeys.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold">Configured Keys</h3>
          <ul className="divide-y divide-border rounded-md border border-border">
            {configuredKeys.map((name) => {
              const revealed = revealedKeys.has(name);
              return (
                <li key={name} className="flex flex-col gap-2 px-3 py-3">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-semibold text-primary">
                      {name}
                    </span>
                    <div className="flex gap-1">
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => toggleReveal(name)}
                        className="size-[26px]"
                      >
                        {revealed ? (
                          <EyeOff className="size-3 text-muted-foreground" />
                        ) : (
                          <Eye className="size-3 text-muted-foreground" />
                        )}
                      </Button>
                      <Button
                        type="button"
                        variant="ghost"
                        size="icon"
                        onClick={() => removeKey(name)}
                        className="size-[26px] hover:bg-destructive/10"
                      >
                        <X className="size-3 text-destructive" />
                      </Button>
                    </div>
                  </div>
                  <Input
                    type={revealed ? "text" : "password"}
                    value={envKeys[name]}
                    onChange={(e) => updateKey(name, e.target.value)}
                    placeholder="Enter value..."
                    autoCorrect="off"
                    autoCapitalize="none"
                    className="font-mono text-sm"
                  />
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* Add custom key */}
      <div className="space-y-2">
        <Label className="text-sm font-medium">Add Custom Key</Label>
        <div className="flex items-center gap-2">
          <Input
            type="text"
            value={newKeyName}
            onChange={(e) => setNewKeyName(e.target.value)}
            placeholder="MY_CUSTOM_KEY"
            autoCapitalize="characters"
            autoCorrect="off"
            className="font-mono text-sm"
          />
          <Button
            type="button"
            onClick={() => addKey(newKeyName)}
            disabled={!newKeyName.trim()}
          >
            <Plus className="size-4" />
            Add
          </Button>
        </div>
      </div>

      {/* Quick-add suggestions */}
      {suggestions.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">Quick Add</h3>
          <p className="text-xs text-muted-foreground">Common API keys</p>
          <div className="flex flex-wrap gap-2 pt-1">
            {suggestions.map((name) => (
              <Badge
                key={name}
                variant="secondary"
                className="cursor-pointer gap-1 hover:bg-primary/10 hover:text-primary"
                onClick={() => addKey(name)}
              >
                <Plus className="size-3" />
                <span className="font-mono text-[11px]">{name}</span>
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// -- Main ------------------------------------------------------------

export default function SettingsScreen() {
  // Section nav now lives in the global sidebar (collapsible group), so the
  // screen no longer needs its own rail or width-detection. URL is the
  // source of truth for activeSection — the sidebar links navigate to
  // /settings?section=<key>, and SettingsScreen re-reads the search param.
  const navigate = useNavigate({ from: "/settings" });
  const search = useSearch({ from: "/settings" });
  const activeSection: SectionKey = search.section ?? "connection";
  const setActiveSection = useCallback(
    (sec: SectionKey) =>
      navigate({ search: { section: sec }, replace: true }),
    [navigate],
  );

  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [benchDir, setBenchDir] = useState("");
  const [optimizerBenchDir, setOptimizerBenchDir] = useState("");
  const [optimizerConfigsDir, setOptimizerConfigsDir] = useState("");
  const [ragProjectDir, setRagProjectDir] = useState("");
  const [ragConfigsDir, setRagConfigsDir] = useState("");
  const [datasetsDir, setDatasetsDir] = useState("");
  const [datasetGenConfigsDir, setDatasetGenConfigsDir] = useState("");
  const [envFilePath, setEnvFilePath] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [envKeys, setEnvKeys] = useState<Record<string, string>>({});
  const [savedEnvKeys, setSavedEnvKeys] = useState<Record<string, string>>({});
  const [newKeyName, setNewKeyName] = useState("");
  const [revealedKeys, setRevealedKeys] = useState<Set<string>>(new Set());

  const [folders, setFolders] = useState<BenchmarkFolder[]>([]);
  const [benchRoot, setBenchRoot] = useState("");
  const [foldersLoading, setFoldersLoading] = useState(false);

  const loadSettings = useCallback(async () => {
    try {
      setLoading(true); setError(null);
      const s = await fetchSettings();
      setSettings(s);
      setBenchDir(s.benchmark_results_dir ?? "");
      setOptimizerBenchDir(s.optimizer_benchmark_dir ?? "");
      setOptimizerConfigsDir(s.optimizer_configs_dir ?? "");
      setRagProjectDir(s.rag_optimization_project_dir ?? "");
      setRagConfigsDir(s.rag_optimization_configs_dir ?? "");
      setDatasetsDir(s.datasets_dir ?? "");
      setDatasetGenConfigsDir(s.dataset_generator_configs_dir ?? "");
      setEnvFilePath(s.env_file_path ?? "");
      const keys = await fetchEnvKeys();
      setEnvKeys(keys);
      setSavedEnvKeys(keys);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setLoading(false); }
  }, []);

  const loadFolders = useCallback(async () => {
    try { setFoldersLoading(true); const data = await fetchBenchmarkFolders(); setFolders(data.folders); setBenchRoot(data.root); }
    catch { setFolders([]); }
    finally { setFoldersLoading(false); }
  }, []);

  useEffect(() => { loadSettings(); loadFolders(); }, [loadSettings, loadFolders]);

  const isDirty = settings != null && (
    benchDir !== settings.benchmark_results_dir ||
    optimizerBenchDir !== settings.optimizer_benchmark_dir ||
    optimizerConfigsDir !== settings.optimizer_configs_dir ||
    ragProjectDir !== settings.rag_optimization_project_dir ||
    ragConfigsDir !== settings.rag_optimization_configs_dir ||
    datasetsDir !== settings.datasets_dir ||
    datasetGenConfigsDir !== settings.dataset_generator_configs_dir ||
    envFilePath !== (settings.env_file_path ?? "") ||
    JSON.stringify(envKeys) !== JSON.stringify(savedEnvKeys)
  );

  const handleSave = async () => {
    try {
      setSaving(true); setSaved(false); setError(null);
      const updated = await updateSettings({
        benchmark_results_dir: benchDir,
        calibration_profiles_dir: settings?.calibration_profiles_dir ?? "",
        optimizer_benchmark_dir: optimizerBenchDir, optimizer_configs_dir: optimizerConfigsDir,
        rag_optimization_project_dir: ragProjectDir, rag_optimization_configs_dir: ragConfigsDir,
        datasets_dir: datasetsDir, dataset_generator_configs_dir: datasetGenConfigsDir,
        env_file_path: envFilePath,
      });
      const savedKeys = await updateEnvKeys(envKeys);
      setEnvKeys(savedKeys); setSavedEnvKeys(savedKeys);
      setSettings(updated); setSaved(true); loadFolders();
      setTimeout(() => setSaved(false), 2000);
    } catch (err) { setError(err instanceof Error ? err.message : String(err)); }
    finally { setSaving(false); }
  };

  if (loading) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-3 bg-background">
        <Loader2 className="size-6 animate-spin text-muted-foreground" />
        <span className="text-[13px] text-muted-foreground">Loading settings...</span>
      </div>
    );
  }

  if (error && !settings) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-2 bg-background p-5">
        <span className="text-[15px] font-semibold text-destructive">Failed to connect to API</span>
        <span className="text-xs text-muted-foreground">{error}</span>
        <div className="mt-3 flex rounded-lg bg-muted p-3">
          <span className="break-all font-mono text-xs text-muted-foreground">uvicorn rag_stack.server.cost_model_api:app --port 8000 --reload --reload-dir rag_stack</span>
        </div>
      </div>
    );
  }

  const sec = SECTIONS.find((s) => s.key === activeSection)!;
  const sectionDescription = SECTION_DESCRIPTIONS[activeSection];

  return (
    <div className="flex flex-1 min-w-0 flex-col bg-background">
      {/* Mobile-only chrome — sidebar trigger + page title so the drawer is
          reachable AND the user knows what page they're on. Mirrors
          <PageHeaderTitle> styling on the other routes. */}
      <div className="flex h-[52px] items-center gap-2.5 border-b border-border bg-card px-3 md:hidden">
        <SidebarTrigger className="size-8 shrink-0" />
        <SettingsIcon className="size-4 shrink-0 text-foreground" />
        <span className="truncate text-base font-semibold tracking-tight text-foreground">
          Settings
        </span>
      </div>
      <div className="flex-1 min-w-0 overflow-auto">
        <div className="mx-auto max-w-[760px] px-8 pt-10 pb-20">
          <SettingsSectionHeader
            title={sec.label}
            description={sectionDescription}
            action={
              <div className="flex items-center gap-3">
                {saved && (
                  <span className="flex items-center gap-1 text-xs text-emerald-600">
                    <Check className="size-[13px]" />
                    Saved
                  </span>
                )}
                {isDirty && (
                  <Button
                    type="button"
                    onClick={handleSave}
                    disabled={saving}
                  >
                    {saving ? <Loader2 className="size-4 animate-spin" /> : "Save Changes"}
                  </Button>
                )}
              </div>
            }
          />

          {activeSection === "connection" && <ConnectionSection />}

          {activeSection === "generators" && (
            <div className="space-y-10">
              <GeneratorClientsSection />
              <Separator />
              <CodexCliClientSection />
            </div>
          )}

          {activeSection === "keys" && (
            <EnvKeysSection
              envKeys={envKeys}
              setEnvKeys={setEnvKeys}
              newKeyName={newKeyName}
              setNewKeyName={setNewKeyName}
              revealedKeys={revealedKeys}
              setRevealedKeys={setRevealedKeys}
              envFilePath={envFilePath}
              setEnvFilePath={setEnvFilePath}
            />
          )}

          {activeSection === "rag" && (
            <div className="space-y-6">
              <SettingField label="Project Directory" description="Root folder for optimization projects (each subfolder is a project)" value={ragProjectDir} onChange={setRagProjectDir} placeholder="tmp_outputs/rag_stack_projects" />
              <SettingField label="Configs Directory" description="YAML config templates for creating new projects" value={ragConfigsDir} onChange={setRagConfigsDir} placeholder="configs/rag_stack" />
              <SettingField label="Datasets Directory" description="Root folder with dataset subfolders containing parquet files" value={datasetsDir} onChange={setDatasetsDir} placeholder="datasets" />
            </div>
          )}

          {activeSection === "dataset" && (
            <div className="space-y-6">
              <SettingField label="Generator Configs Directory" description="YAML configs for dataset generation strategies" value={datasetGenConfigsDir} onChange={setDatasetGenConfigsDir} placeholder="configs/dataset_generator" />
            </div>
          )}

          {activeSection === "cost-model" && (
            <div className="space-y-10">
              <div className="space-y-6">
                <SettingField label="Benchmark Results Directory" description="Root folder for cost model benchmark outputs (per-model subdirs inside)" value={benchDir} onChange={setBenchDir} placeholder="tmp_outputs" />
                <div className="space-y-1.5">
                  <p className="text-sm font-medium">Calibration profiles</p>
                  <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">Apps and RAG-CM use the same runtime profile directory. View its active path in Cost Models → Calibration. Set <code className="break-all text-xs">RAG_STACK_CALIBRATION_PROFILE_DIR</code> on the server to change it.</p>
                </div>
              </div>

              <Separator />

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <h3 className="text-sm font-semibold">Discovered Benchmark Folders</h3>
                    {benchRoot && (
                      <p className="font-mono text-xs text-muted-foreground/70">
                        {benchRoot}
                      </p>
                    )}
                  </div>
                  <Button type="button" variant="ghost" size="icon" onClick={loadFolders} className="size-6">
                    <RefreshCw className="size-3.5 text-muted-foreground" />
                  </Button>
                </div>
                {foldersLoading ? (
                  <div className="flex items-center justify-center py-4">
                    <Loader2 className="size-4 animate-spin text-muted-foreground" />
                  </div>
                ) : folders.length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    No subfolders found
                  </p>
                ) : (
                  <ul className="divide-y divide-border rounded-md border border-border">
                    {folders.map((f) => (
                      <li
                        key={f.name}
                        className="flex items-center gap-3 px-3 py-2 text-sm"
                      >
                        <FolderOpen className="size-4 text-muted-foreground" />
                        <span className="flex-1 truncate font-medium">{f.name}</span>
                        <Badge variant="secondary" className="font-mono text-xs">
                          {f.n_rows} rows
                        </Badge>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          )}

          {activeSection === "optimizer" && (
            <div className="space-y-6">
              <SettingField label="Benchmark Configs Directory" description="YAML config files for launching benchmark runs" value={optimizerConfigsDir} onChange={setOptimizerConfigsDir} placeholder="configs/optimizer_benchmark" />
              <SettingField label="Benchmark Results Directory" description="CSV results from optimizer benchmarks" value={optimizerBenchDir} onChange={setOptimizerBenchDir} placeholder="tmp_outputs/optimizer_benchmark_results" />
            </div>
          )}

          {error && settings && (
            <p className="mt-6 text-xs text-destructive">{error}</p>
          )}
        </div>
      </div>
    </div>
  );
}
