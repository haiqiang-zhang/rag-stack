import React, { useCallback, useMemo } from "react";
import { Layers, FileBox, Plus, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { IconCircle } from "@/components/ui/IconCircle";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { Separator } from "@/components/ui/separator";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import type { CorpusConfig } from "./types";

/** Hardcoded list of chunker methods, mirroring the python-side
 * `chunk_modules` registry at
 * rag_stack/static_rag_quality_evaluator/chunk/_registry.py. Keep in sync
 * if new chunkers are added.
 */
const CHUNK_METHODS = [
  "recursivecharacter",
  "character",
  "token",
  "sentence",
  "sentencewindow",
  "semantic_llama_index",
  "semanticdoublemerging",
  "sentencetransformerstoken",
  "konlpy",
  "simplefile",
] as const;

type RangeSpec = { range: [number, number]; n_grid?: number };
type DiscreteValue = number | number[];
type ParamValue = DiscreteValue | RangeSpec | undefined;
type ParamMode = "off" | "discrete" | "range";

function isRangeSpec(v: unknown): v is RangeSpec {
  return (
    !!v &&
    typeof v === "object" &&
    "range" in (v as Record<string, unknown>) &&
    Array.isArray((v as { range: unknown }).range)
  );
}

function modeOf(v: unknown): ParamMode {
  if (v === undefined) return "off";
  if (isRangeSpec(v)) return "range";
  return "discrete";
}

/** Normalize a discrete value to a number[] for editing (length >= 1). */
function asValueArray(v: ParamValue): number[] {
  if (Array.isArray(v)) return v.length > 0 ? v : [0];
  if (typeof v === "number") return [v];
  return [];
}

/** Collapse a number[] back to scalar / array / undefined for the YAML. */
function collapseDiscrete(arr: number[]): DiscreteValue | undefined {
  if (arr.length === 0) return undefined;
  if (arr.length === 1) return arr[0];
  return arr;
}

interface CorpusTabProps {
  corpus: CorpusConfig | undefined;
  onChange: (next: CorpusConfig | undefined) => void;
}

export function CorpusTab({ corpus, onChange }: CorpusTabProps) {
  const safeCorpus: CorpusConfig = corpus ?? {};
  const chunker = safeCorpus.chunker ?? {};

  const setRawPath = useCallback(
    (raw_path: string) => {
      const next: CorpusConfig = { ...safeCorpus };
      if (raw_path) next.raw_path = raw_path;
      else delete next.raw_path;
      onChange(emptyIfBlank(next));
    },
    [safeCorpus, onChange],
  );

  const updateChunker = useCallback(
    (patch: Partial<CorpusConfig["chunker"]>) => {
      const nextChunker = { ...chunker, ...patch };
      for (const k of Object.keys(nextChunker)) {
        if (nextChunker[k as keyof typeof nextChunker] === undefined) {
          delete nextChunker[k as keyof typeof nextChunker];
        }
      }
      const next: CorpusConfig = { ...safeCorpus };
      if (Object.keys(nextChunker).length > 0) next.chunker = nextChunker;
      else delete next.chunker;
      onChange(emptyIfBlank(next));
    },
    [chunker, safeCorpus, onChange],
  );

  // component: multi-select. Single → scalar, multiple → list (sweep dim).
  const selectedMethods = useMemo<string[]>(() => {
    const m = chunker.component;
    if (m === undefined) return [];
    if (Array.isArray(m)) return m.map(String);
    return [String(m)];
  }, [chunker.component]);

  const toggleMethod = (method: string) => {
    const set = new Set(selectedMethods);
    if (set.has(method)) set.delete(method);
    else set.add(method);
    const arr = Array.from(set);
    if (arr.length === 0) updateChunker({ component: undefined });
    else if (arr.length === 1) updateChunker({ component: arr[0] });
    else updateChunker({ component: arr });
  };

  return (
    // The `absolute inset-0 overflow-auto` wrapper in ConfigVisualEditor
    // owns the geometry and the scrollbar; we only lay out content here.
    <div className="mx-auto w-full max-w-[860px] p-7 pb-12">
        {/* Header */}
        <div className="mb-6 flex items-center gap-2.5">
          <IconCircle size="md" variant="accent">
            <Layers size={16} className="text-primary" />
          </IconCircle>
          <div className="flex flex-col">
            <span className="text-lg font-semibold tracking-tight text-foreground">
              Corpus &amp; Chunking
            </span>
            <span className="text-xs text-muted-foreground">
              Optional. Add a <code className="rounded bg-muted px-1 font-mono text-[10px]">corpus.raw_path</code> and chunker
              sweep spec to search over chunk_size / chunk_overlap / chunk_method.
            </span>
          </div>
        </div>

        {/* Source */}
        <section className="space-y-3">
          <SectionHeader
            title="Raw Source"
            right={<FileBox size={12} className="text-primary" />}
          />
          <div className="grid gap-2">
            <Label htmlFor="corpus-raw-path" className="text-xs text-muted-foreground">
              raw_path
            </Label>
            <Input
              id="corpus-raw-path"
              className="font-mono text-xs"
              value={safeCorpus.raw_path ?? ""}
              onChange={(e) => setRawPath(e.target.value)}
              placeholder="${PROJECT_DIR}/data/raw_corpus.parquet"
            />
            <p className="text-xs leading-snug text-muted-foreground">
              Long-document parquet (one row per doc, with a{" "}
              <code className="rounded bg-muted px-1 font-mono text-[10px]">contents</code> or{" "}
              <code className="rounded bg-muted px-1 font-mono text-[10px]">texts</code> column).
              Copied to{" "}
              <code className="rounded bg-muted px-1 font-mono text-[10px]">
                &lt;project&gt;/data/raw_corpus.parquet
              </code>{" "}
              on first run. Leave empty to disable chunking sweeps and use the project's
              pre-chunked{" "}
              <code className="rounded bg-muted px-1 font-mono text-[10px]">corpus.parquet</code>{" "}
              as-is.
            </p>
          </div>
        </section>

        <Separator className="my-6" />

        {/* Chunker */}
        <section className="space-y-5">
          <SectionHeader title="Chunker" />

          {/* Methods */}
          <div className="space-y-2">
            <Label className="text-xs text-muted-foreground">
              method
              <span className="font-normal text-muted-foreground/70">
                (one fixed · multiple sweep)
              </span>
            </Label>
            <div className="grid grid-cols-2 gap-1.5">
              {CHUNK_METHODS.map((m) => {
                const checked = selectedMethods.includes(m);
                return (
                  <Label
                    key={m}
                    className={cn(
                      "flex cursor-pointer items-center gap-2 rounded-md border px-2.5 py-1.5 transition-colors",
                      checked
                        ? "border-primary/40 bg-primary/5"
                        : "border-input bg-background hover:bg-muted/40",
                    )}
                  >
                    <Checkbox
                      checked={checked}
                      onCheckedChange={() => toggleMethod(m)}
                    />
                    <span className="font-mono text-xs text-foreground">{m}</span>
                  </Label>
                );
              })}
            </div>
          </div>

          <Separator />

          {/* chunk_size */}
          <ParamRow
            label="chunk_size"
            value={chunker.chunk_size as ParamValue}
            discreteDefault={512}
            rangeDefault={{ range: [128, 1024] }}
            onChange={(v) => updateChunker({ chunk_size: v })}
          />

          <Separator />

          {/* chunk_overlap */}
          <ParamRow
            label="chunk_overlap"
            value={chunker.chunk_overlap as ParamValue}
            discreteDefault={50}
            rangeDefault={{ range: [0, 200] }}
            onChange={(v) => updateChunker({ chunk_overlap: v })}
          />
        </section>

        <p className="mt-6 text-xs leading-relaxed text-muted-foreground">
          Each evaluation re-chunks on demand and caches under{" "}
          <code className="rounded bg-muted px-1 font-mono text-[10px]">
            &lt;project&gt;/preprocessed/chunks/&lt;chunk_hash&gt;/
          </code>
          . FAISS index dirs are keyed on chunk_hash + embedding_model so
          different chunkings stay isolated.
        </p>
    </div>
  );
}

// ── Single param row: off / discrete / range toggle ──────────────────

interface ParamRowProps {
  label: string;
  value: ParamValue;
  /** Initial scalar value when the user switches to discrete mode. */
  discreteDefault: number;
  rangeDefault: RangeSpec;
  onChange: (v: ParamValue) => void;
}

function ParamRow({ label, value, discreteDefault, rangeDefault, onChange }: ParamRowProps) {
  const mode = modeOf(value);

  const setMode = (next: ParamMode) => {
    if (next === mode) return;
    if (next === "off") onChange(undefined);
    else if (next === "discrete") onChange(discreteDefault);
    else onChange(rangeDefault);
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <Label className="text-xs text-muted-foreground">{label}</Label>
        <ToggleGroup
          type="single"
          size="sm"
          variant="outline"
          value={mode}
          onValueChange={(v) => v && setMode(v as ParamMode)}
        >
          <ToggleGroupItem value="off" aria-label="off">off</ToggleGroupItem>
          <ToggleGroupItem value="discrete" aria-label="discrete">discrete</ToggleGroupItem>
          <ToggleGroupItem value="range" aria-label="range">range</ToggleGroupItem>
        </ToggleGroup>
      </div>

      {mode === "discrete" && (
        <DiscreteEditor
          value={value as DiscreteValue}
          discreteDefault={discreteDefault}
          onChange={(arr) => onChange(collapseDiscrete(arr))}
        />
      )}
      {mode === "range" && (
        <RangeEditor
          value={value as RangeSpec}
          onChange={(v) => onChange(v)}
        />
      )}
      {mode === "off" && (
        <span className="text-xs italic text-muted-foreground/70">
          unset — uses the chunker's library default
        </span>
      )}
    </div>
  );
}

// ── Discrete value editor: 1+ numbers, scalar collapses to a fixed value ──

interface DiscreteEditorProps {
  value: DiscreteValue;
  discreteDefault: number;
  onChange: (next: number[]) => void;
}

function DiscreteEditor({ value, discreteDefault, onChange }: DiscreteEditorProps) {
  const arr = asValueArray(value);

  const setAt = (idx: number, n: number) => {
    const next = arr.slice();
    next[idx] = n;
    onChange(next);
  };
  const removeAt = (idx: number) => {
    if (arr.length <= 1) return;
    onChange(arr.filter((_, i) => i !== idx));
  };
  const append = () => {
    const last = arr[arr.length - 1];
    onChange([...arr, Number.isFinite(last) ? last : discreteDefault]);
  };

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {arr.map((n, i) => (
        <div key={i} className="flex items-center">
          <Input
            type="number"
            className="w-24 rounded-r-none font-mono text-xs"
            value={n}
            onChange={(e) => {
              const next = Number(e.target.value);
              if (Number.isFinite(next)) setAt(i, next);
            }}
          />
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="size-9 rounded-l-none border-l-0"
            disabled={arr.length <= 1}
            onClick={() => removeAt(i)}
            aria-label={`remove value ${i + 1}`}
          >
            <X className="size-3 text-muted-foreground" />
          </Button>
        </div>
      ))}
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={append}
      >
        <Plus className="size-3 text-primary" />
        add value
      </Button>
      <span className="ml-2 text-[10px] text-muted-foreground/70">
        {arr.length === 1 ? "single value (fixed)" : `${arr.length} values (sweep)`}
      </span>
    </div>
  );
}

interface RangeEditorProps {
  value: RangeSpec;
  onChange: (v: RangeSpec) => void;
}

function RangeEditor({ value, onChange }: RangeEditorProps) {
  const [lo, hi] = value.range;
  const setLo = (n: number) => onChange({ ...value, range: [n, hi] });
  const setHi = (n: number) => onChange({ ...value, range: [lo, n] });

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Input
        type="number"
        className="w-24 font-mono text-xs"
        value={lo}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isFinite(n)) setLo(n);
        }}
      />
      <span className="text-xs text-muted-foreground">–</span>
      <Input
        type="number"
        className="w-24 font-mono text-xs"
        value={hi}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (Number.isFinite(n)) setHi(n);
        }}
      />
    </div>
  );
}

/** Strip empty objects so we don't emit `corpus: {}` to YAML when the
 * user toggles everything off. */
function emptyIfBlank(c: CorpusConfig): CorpusConfig | undefined {
  if (!c.raw_path && (!c.chunker || Object.keys(c.chunker).length === 0)) {
    return undefined;
  }
  return c;
}
