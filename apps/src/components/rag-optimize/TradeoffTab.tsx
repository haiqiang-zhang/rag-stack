/**
 * Trade-off tab — the project-level quality ⟷ performance synergy view.
 *
 * The knob-association table IS the content here and leads the page: for
 * every searched knob, how it moves quality vs how it moves performance.
 * There is deliberately NO second objective-space chart — picking a knob
 * (row click or the dropdown) recolors the sidebar map, which is the one
 * canonical (q, p) chart. One map, more function; no duplication.
 */
import { useMemo } from "react";
import { ArrowLeft } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useEvalData } from "@/context/EvalDataContext";
import { buildKnobDimensions, varyingDims, formatKnobValue } from "@/utils/searchKnobs";
import { buildKnobImpacts, type KnobImpact } from "@/utils/knobStats";
import { cn } from "@/lib/utils";

// Diverging pair for signed ρ bars. Sign is primarily encoded by direction
// (left/right of the zero line); color only reinforces it.
const POS = "#0D9488";
const NEG = "#BE123C";

interface TradeoffTabProps {
  /** Active coloring: "none" | "progress" | a knob key. Lifted to the screen
   *  so the sidebar map (the actual chart) can consume it too. */
  colorKey: string;
  onColorKeyChange: (key: string) => void;
}

export function TradeoffTab({ colorKey, onColorKeyChange }: TradeoffTabProps) {
  const {
    enrichedEvals,
    clientPareto,
    getQuality,
    getPerformance,
    isFeasible,
    sweepsReady,
    sweepsResolvedCount,
    sweepsTotalCount,
    loading,
  } = useEvalData();

  const dims = useMemo(
    () => varyingDims(buildKnobDimensions(enrichedEvals)),
    [enrichedEvals],
  );
  const impacts = useMemo(
    () => buildKnobImpacts(dims, enrichedEvals, getQuality, getPerformance, isFeasible),
    [dims, enrichedEvals, getQuality, getPerformance, isFeasible],
  );

  const nGT = useMemo(
    () =>
      new Set(
        enrichedEvals
          .filter((ev) => ev.eval_type === "GT" && getQuality(ev) != null)
          .map((ev) => ev.eval_id),
      ).size,
    [enrichedEvals, getQuality],
  );

  // Group the color-by options by knob group for a scannable dropdown.
  const dimGroups = useMemo(() => {
    const groups = new Map<string, typeof dims>();
    for (const d of dims) {
      (groups.get(d.group) ?? groups.set(d.group, []).get(d.group)!).push(d);
    }
    return [...groups.entries()];
  }, [dims]);

  return (
    <div className="flex-1 min-w-0 overflow-auto p-6 pb-12">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-foreground">
            Quality ⟷ performance trade-off
          </h2>
          <p className="flex items-center gap-1 text-xs text-muted-foreground">
            <ArrowLeft className="size-3 shrink-0" />
            <span>
              pick a knob (row or dropdown) — the map on the left recolors by
              it · {nGT} GT evals · frontier {clientPareto.length} points
              {!sweepsReady && sweepsTotalCount > 0 && (
                <>
                  {" "}
                  · policy scores hydrating {sweepsResolvedCount}/{sweepsTotalCount}
                </>
              )}
            </span>
          </p>
        </div>
        <Select value={colorKey} onValueChange={onColorKeyChange}>
          <SelectTrigger size="sm" className="w-56">
            <SelectValue placeholder="Color map by…" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">Map color: none</SelectItem>
            <SelectItem value="progress">Map color: search order</SelectItem>
            {dimGroups.map(([group, ds]) => (
              <SelectGroup key={group}>
                <SelectLabel>{group.replace(/_/g, " ")}</SelectLabel>
                {ds.map((d) => (
                  <SelectItem key={d.key} value={d.key}>
                    {d.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Card className="rounded-lg py-0">
        <CardContent className="p-4">
          <div className="mb-1 flex flex-wrap items-baseline justify-between gap-2">
            <h3 className="text-sm font-semibold text-foreground">
              What moves quality vs performance
            </h3>
          </div>
          <p className="mb-3 text-xs text-muted-foreground">
            Per-knob association with each objective under the active quality
            metric + performance policy. Numeric knobs: Spearman ρ (signed,
            monotone trend). Choice knobs: correlation ratio η (unsigned share
            of variance) with the best-mean value. Quality uses GT evals;
            performance also counts CM evals.
          </p>
          {loading ? (
            <span className="text-xs text-muted-foreground">Loading…</span>
          ) : impacts.length === 0 ? (
            <span className="text-xs text-muted-foreground">
              No varying knobs found yet — associations appear once evals land.
            </span>
          ) : (
            <KnobImpactTable
              impacts={impacts}
              activeKey={colorKey}
              onPick={(key) => onColorKeyChange(key === colorKey ? "none" : key)}
            />
          )}
        </CardContent>
      </Card>
    </div>
  );
}

// ── Impact table ────────────────────────────────────────────────────────────

function KnobImpactTable({
  impacts,
  activeKey,
  onPick,
}: {
  impacts: KnobImpact[];
  activeKey: string;
  onPick: (dimKey: string) => void;
}) {
  return (
    <div className="min-w-0 overflow-x-auto rounded-md border">
      <table className="w-full min-w-[720px] border-collapse text-xs">
        <thead className="bg-muted/50 text-[10px] uppercase text-muted-foreground">
          <tr>
            <th className="px-3 py-2 text-left font-medium">Knob</th>
            <th className="px-3 py-2 text-left font-medium" title="Association with the active quality metric (GT evals)">
              → Quality
            </th>
            <th className="px-3 py-2 text-left font-medium" title="Association with the active performance policy score (GT + CM evals)">
              → Performance
            </th>
            <th className="px-3 py-2 text-right font-medium" title="Sample sizes: quality pairs / performance pairs">
              n
            </th>
          </tr>
        </thead>
        <tbody>
          {impacts.map((im) => {
            const active = im.dim.key === activeKey;
            return (
              <tr
                key={im.dim.key}
                onClick={() => onPick(im.dim.key)}
                title="Click to recolor the sidebar map by this knob"
                className={cn(
                  "cursor-pointer border-t transition-colors",
                  active ? "bg-primary/10" : "hover:bg-muted/50",
                )}
              >
                <td className="px-3 py-2">
                  <div className="flex min-w-0 flex-col">
                    <span className="truncate font-mono font-medium text-foreground">
                      {im.dim.label}
                    </span>
                    <span className="truncate text-[10px] text-muted-foreground">
                      {im.dim.group.replace(/_/g, " ")}
                      {" · "}
                      {im.dim.distinct.length} values
                    </span>
                  </div>
                </td>
                <td className="px-3 py-2">
                  <AssocCell
                    rho={im.qualityRho}
                    eta={im.qualityEta}
                    bestValue={im.bestQualityValue}
                  />
                </td>
                <td className="px-3 py-2">
                  <AssocCell
                    rho={im.perfRho}
                    eta={im.perfEta}
                    bestValue={im.bestPerfValue}
                  />
                </td>
                <td className="whitespace-nowrap px-3 py-2 text-right font-mono tabular-nums text-muted-foreground">
                  {im.nQuality}/{im.nPerf}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** One objective cell: diverging ρ bar (numeric) or single η bar (choice). */
function AssocCell({
  rho,
  eta,
  bestValue,
}: {
  rho: number | null;
  eta: number | null;
  bestValue: string | null;
}) {
  if (rho == null && eta == null) {
    return <span className="text-muted-foreground">— too few points</span>;
  }
  if (rho != null) {
    const mag = Math.min(1, Math.abs(rho));
    const positive = rho >= 0;
    return (
      <div
        className="flex items-center gap-2"
        title={`Spearman ρ = ${rho.toFixed(3)} (${positive ? "higher knob → higher objective" : "higher knob → lower objective"})`}
      >
        <div className="relative h-1.5 w-28 shrink-0 rounded-full bg-muted">
          <span className="absolute inset-y-0 left-1/2 w-px bg-border" />
          <span
            className="absolute inset-y-0 rounded-full"
            style={{
              backgroundColor: positive ? POS : NEG,
              left: positive ? "50%" : `${50 - mag * 50}%`,
              width: `${mag * 50}%`,
            }}
          />
        </div>
        <span
          className="w-12 shrink-0 font-mono tabular-nums"
          style={{ color: positive ? POS : NEG }}
        >
          {rho >= 0 ? "+" : ""}
          {rho.toFixed(2)}
        </span>
      </div>
    );
  }
  const e = Math.min(1, eta!);
  return (
    <div
      className="flex min-w-0 items-center gap-2"
      title={`Correlation ratio η = ${eta!.toFixed(3)} — best mean value: ${bestValue ?? "—"}`}
    >
      <div className="relative h-1.5 w-28 shrink-0 rounded-full bg-muted">
        <span
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ backgroundColor: POS, width: `${e * 100}%`, opacity: 0.7 }}
        />
      </div>
      <span className="w-12 shrink-0 font-mono tabular-nums text-muted-foreground">
        η {eta!.toFixed(2)}
      </span>
      {bestValue && (
        <span className="min-w-0 truncate text-[10px] text-muted-foreground">
          best: <span className="font-mono text-foreground">{formatKnobValue(bestValue)}</span>
        </span>
      )}
    </div>
  );
}
