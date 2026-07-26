/**
 * ProjectBenchmark — compare RAG-Stack optimization projects by hypervolume.
 *
 * Pick any set of finished projects; the backend (POST /rag-project/compare-hv,
 * porting analysis/compare_projects.py) computes the running 2-objective
 * hypervolume (performance × quality) for each, normalized to [0,1] on bounds
 * SHARED across the selected projects so the HV numbers are directly
 * comparable.
 *
 * Runs of the SAME experiment under different seeds can be merged into one
 * series via the SeriesTray: the user arms chips and merges them (explicit
 * choice, never name-convention matching). A merged series draws as mean ±
 * 1 std across its runs — the same convention as the synthetic-functions
 * tab — aggregated client-side from the per-project HV curves, so
 * regrouping is instant. Renders an HV-convergence chart + a Pareto/eval
 * scatter + a per-series summary table.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "@tanstack/react-router";
import type { ECharts } from "echarts/core";
import {
  Loader2, BarChart3, CircleAlert, CornerDownRight,
  ArrowUp, ArrowDown, ArrowUpDown,
} from "lucide-react";
import { SidePanel, SidePanelHeader } from "@/components/ui/SidePanel";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";
import EChartsView from "@/components/charts/EChartsView";
import { useProject } from "@/context/ProjectContext";
import {
  fetchProjectHvComparison,
  type ProjectHvComparison,
  type ProjectHvSeries,
} from "@/api/costModelClient";
import type { EChartsOption } from "echarts";
import { SeriesTray } from "./SeriesTray";
import {
  aggregateHv, buildSeriesList, finalHvStats, loadGroups, loadHidden,
  loadSelection, newGroupId, saveGroups, saveHidden, saveSelection,
  suggestGroupName, type SeriesGroup,
} from "./projectSeries";

const fmt3 = (v: number) => v.toFixed(3);

// ── Final-metrics table sorting ─────────────────────────────────────────────
type TableSortKey = "name" | "hv" | "qual" | "perf" | "valid" | "init";
interface TableSort {
  key: TableSortKey;
  dir: 1 | -1;
}
/** Natural first direction per column: names A→Z, numbers big-first. */
const SORT_DEFAULT_DIR: Record<TableSortKey, 1 | -1> = {
  name: 1, hv: -1, qual: -1, perf: -1, valid: -1, init: -1,
};

function SortableTh({
  label,
  k,
  sort,
  onCycle,
  className,
}: {
  label: string;
  k: TableSortKey;
  sort: TableSort | null;
  onCycle: (k: TableSortKey) => void;
  className?: string;
}) {
  const active = sort?.key === k;
  const Icon = !active ? ArrowUpDown : sort!.dir === 1 ? ArrowUp : ArrowDown;
  return (
    <th className={cn("py-1 font-medium", className)}>
      <button
        type="button"
        onClick={() => onCycle(k)}
        title="Click to sort (third click restores tray order)"
        className={cn(
          "inline-flex cursor-pointer select-none items-center gap-0.5 transition-colors hover:text-foreground",
          active && "text-foreground",
        )}
      >
        {label}
        <Icon size={10} className={active ? "" : "opacity-40"} />
      </button>
    </th>
  );
}

export function ProjectBenchmark() {
  const { projects } = useProject();
  const [selected, setSelected] = useState<string[]>(() => loadSelection());
  const [groups, setGroups] = useState<SeriesGroup[]>(() => loadGroups());
  const [hiddenKeys, setHiddenKeys] = useState<string[]>(() => loadHidden());
  const [armed, setArmed] = useState<string[]>([]);
  const [expandedGroupId, setExpandedGroupId] = useState<string | null>(null);
  const [showRuns, setShowRuns] = useState(false);
  const [metric, setMetric] = useState<string | undefined>(undefined);
  const [data, setData] = useState<ProjectHvComparison | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => saveGroups(groups), [groups]);
  useEffect(() => saveSelection(selected), [selected]);
  useEffect(() => saveHidden(hiddenKeys), [hiddenKeys]);

  // Visual-only hide per series (chart legend semantics): the series keeps
  // its selection, grouping, color, and the shared normalization bounds —
  // unlike deselecting every member in the project rail and re-picking them
  // later. Toggled from the tray chip's eye.
  const hiddenSet = useMemo(() => new Set(hiddenKeys), [hiddenKeys]);
  const toggleHidden = (key: string) =>
    setHiddenKeys((h) => (h.includes(key) ? h.filter((k) => k !== key) : [...h, key]));

  const toggle = (name: string) =>
    setSelected((s) => (s.includes(name) ? s.filter((x) => x !== name) : [...s, name]));

  // Draw only projects that still exist, in the project list's sort order.
  const orderedSelected = useMemo(
    () => projects.filter((p) => selected.includes(p.name)).map((p) => p.name),
    [projects, selected],
  );
  const series = useMemo(() => buildSeriesList(orderedSelected, groups), [orderedSelected, groups]);
  const armedValid = useMemo(
    () => armed.filter((k) => series.some((s) => s.key === k)),
    [armed, series],
  );
  /** project name → color of the series it belongs to (left panel + table). */
  const seriesColorOf = useMemo(() => {
    const m = new Map<string, string>();
    for (const s of series) for (const p of s.members) m.set(p, s.color);
    return m;
  }, [series]);

  const selectionKey = orderedSelected.join("\n");
  useEffect(() => {
    if (orderedSelected.length === 0) { setData(null); return; }
    let cancelled = false;
    setLoading(true);
    fetchProjectHvComparison(orderedSelected, metric)
      .then((d) => { if (!cancelled) { setData(d); setLoading(false); if (!metric && d.metric) setMetric(d.metric); } })
      .catch((e) => { if (!cancelled) { setData({ available_metrics: [], error: String(e) }); setLoading(false); } });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectionKey, metric]);

  const byLabel = useMemo(
    () => new Map((data?.projects ?? []).map((p) => [p.label, p])),
    [data],
  );

  // ── Final-metrics table: per-series stats + user sorting ─────────────────
  // Sorting reorders the TABLE only — tray order still drives colors and
  // chart z-order, so re-sorting never repaints a series. Cycle per column:
  // natural direction → reversed → back to tray order.
  const [tableSort, setTableSort] = useState<TableSort | null>(null);
  const cycleSort = (key: TableSortKey) =>
    setTableSort((cur) => {
      if (!cur || cur.key !== key) return { key, dir: SORT_DEFAULT_DIR[key] };
      if (cur.dir === SORT_DEFAULT_DIR[key]) return { key, dir: (cur.dir * -1) as 1 | -1 };
      return null;
    });

  const tableRows = useMemo(() => {
    const rows = series.map((s) => {
      const members = s.members
        .map((m) => byLabel.get(m))
        .filter((p): p is ProjectHvSeries => !!p);
      const present = members.filter((p) => !p.missing);
      const finals = finalHvStats(present.map((p) => p.final_hv ?? 0));
      const bestQ = present.reduce<number | null>(
        (acc, p) => (p.best_qual != null && (acc == null || p.best_qual > acc) ? p.best_qual : acc),
        null,
      );
      const bestP = present.reduce<number | null>(
        (acc, p) => (p.best_perf != null && (acc == null || p.best_perf > acc) ? p.best_perf : acc),
        null,
      );
      const nValid = present.reduce((a, p) => a + (p.n_valid ?? 0), 0);
      const nEvals = present.reduce((a, p) => a + (p.n_evals ?? 0), 0);
      const nInits = Array.from(new Set(present.map((p) => p.n_init ?? 0)));
      return { s, members, present, finals, bestQ, bestP, nValid, nEvals, nInits };
    });
    if (!tableSort) return rows;
    const { key, dir } = tableSort;
    const val = (r: (typeof rows)[number]): string | number | null => {
      if (key === "name") return r.s.name;
      if (r.present.length === 0) return null; // no data → always bottom
      switch (key) {
        case "hv": return r.finals.mean;
        case "qual": return r.bestQ;
        case "perf": return r.bestP;
        case "valid": return r.nValid;
        case "init": return r.nInits[0] ?? 0;
      }
    };
    return [...rows].sort((a, b) => {
      const va = val(a);
      const vb = val(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      const cmp =
        typeof va === "string"
          ? va.localeCompare(String(vb))
          : (va as number) - (vb as number);
      return cmp * dir;
    });
  }, [series, byLabel, tableSort]);
  /** A series' members that have data on the chart (csv present, ≥1 HV point). */
  const presentOf = (members: string[]): ProjectHvSeries[] =>
    members
      .map((m) => byLabel.get(m))
      .filter((p): p is ProjectHvSeries => !!p && !p.missing && (p.hv?.length ?? 0) > 0);
  const hasMultiRunSeries = series.some((s) => presentOf(s.members).length > 1);

  // ── Grouping actions ──────────────────────────────────────────────────

  const toggleArm = (key: string) =>
    setArmed((a) => (a.includes(key) ? a.filter((k) => k !== key) : [...a, key]));

  const mergeArmed = () => {
    const armedSeries = series.filter((s) => armedValid.includes(s.key));
    if (armedSeries.length < 2) return;
    // Full group definitions merge (not just the selected subset), so the
    // series stays intact when members are re-checked later.
    const memberUnion: string[] = [];
    for (const s of armedSeries) {
      for (const m of s.group ? s.group.members : s.members) {
        if (!memberUnion.includes(m)) memberUnion.push(m);
      }
    }
    const target = armedSeries.find((s) => s.group)?.group;
    if (target) {
      const armedGroupIds = new Set(armedSeries.filter((s) => s.group).map((s) => s.group!.id));
      setGroups((gs) =>
        gs
          .filter((g) => g.id === target.id || !armedGroupIds.has(g.id))
          .map((g) => (g.id === target.id ? { ...g, members: memberUnion } : g)));
      setExpandedGroupId(target.id);
    } else {
      const id = newGroupId();
      setGroups((gs) => [...gs, { id, name: suggestGroupName(memberUnion), members: memberUnion }]);
      setExpandedGroupId(id);
    }
    setArmed([]);
  };

  const renameGroup = (id: string, name: string) =>
    setGroups((gs) => gs.map((g) => (g.id === id ? { ...g, name } : g)));

  const splitGroup = (id: string) => {
    setGroups((gs) => gs.filter((g) => g.id !== id));
    setExpandedGroupId((e) => (e === id ? null : e));
  };

  const removeMember = (id: string, member: string) => {
    const g = groups.find((x) => x.id === id);
    if (!g) return;
    const members = g.members.filter((m) => m !== member);
    if (members.length < 2) { // a 1-member series is just the run itself
      splitGroup(id);
    } else {
      setGroups((gs) => gs.map((x) => (x.id === id ? { ...x, members } : x)));
    }
  };

  // ── Charts ────────────────────────────────────────────────────────────

  const hvOption: EChartsOption = useMemo(() => {
    const chartSeries: any[] = [];
    const stdLookup = new Map<string, Map<number, number>>();
    const nRunsByName = new Map<string, number>();

    for (const s of series) {
      if (hiddenSet.has(s.key)) continue;
      const present = presentOf(s.members);
      if (present.length === 0) continue;
      nRunsByName.set(s.name, present.length);

      if (present.length === 1) {
        chartSeries.push({
          name: s.name, id: s.key, type: "line", showSymbol: false, z: 3,
          lineStyle: { width: 2, color: s.color }, itemStyle: { color: s.color },
          emphasis: { focus: "series" },
          data: present[0].hv!.map((h) => [h.i, h.hv]),
        });
        continue;
      }

      const agg = aggregateHv(present.map((p) => p.hv!));
      stdLookup.set(s.name, agg.stdByX);
      // Mean line first so the deduping tooltip keeps it (not a member run).
      chartSeries.push({
        name: s.name, id: s.key, type: "line", showSymbol: false, z: 3,
        lineStyle: { width: 2, color: s.color }, itemStyle: { color: s.color },
        emphasis: { focus: "series" },
        data: agg.mean,
      });
      // ±1 std band as a closed polygon (same idiom as BenchmarkChartGrid;
      // shares the series `name` so the legend toggles line + band together).
      if (agg.lower.some((lo, i) => agg.upper[i][1] !== lo[1])) {
        const upperArr = agg.upper;
        const lowerArr = agg.lower;
        const fillColor = s.color;
        // Invisible line tracing the band's lower edge: only real series data
        // enters ECharts' axis extent, and the polygon itself contributes just
        // its upper edge — without this the tight-fit y-axis could clip the
        // bottom of the band. Same name → toggles with the group's legend
        // entry; the tooltip dedup keeps the mean line (pushed first).
        chartSeries.push({
          name: s.name, id: `${s.key}__bandlo`, type: "line",
          showSymbol: false, silent: true, z: 0,
          lineStyle: { width: 0, opacity: 0 }, itemStyle: { color: s.color },
          data: lowerArr,
        });
        chartSeries.push({
          name: s.name, id: `${s.key}__band`, type: "custom", z: 0,
          data: upperArr, // only drives dataIndex iteration
          renderItem: (params: any, api: any) => {
            if (params.dataIndex !== 0) return null; // draw the polygon once
            const points: [number, number][] = [];
            for (let i = 0; i < upperArr.length; i++) points.push(api.coord(upperArr[i]));
            for (let i = lowerArr.length - 1; i >= 0; i--) points.push(api.coord(lowerArr[i]));
            return {
              type: "polygon",
              shape: { points },
              style: { fill: fillColor, opacity: 0.15, stroke: "transparent" },
              emphasisDisabled: false,
              emphasis: { style: { opacity: 0.32 } },
            };
          },
          tooltip: { show: false },
        });
      }
      if (showRuns) {
        for (const p of present) {
          chartSeries.push({
            name: s.name, id: `${s.key}__run__${p.label}`, type: "line",
            showSymbol: false, silent: true, z: 1,
            lineStyle: { width: 1, color: s.color, opacity: 0.3 },
            itemStyle: { color: s.color },
            data: p.hv!.map((h) => [h.i, h.hv]),
          });
        }
      }
    }

    return {
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(17,24,39,0.92)",
        borderColor: "transparent",
        textStyle: { color: "#F9FAFB", fontSize: 11 },
        // Filter band polygons, dedup member-run rows (they share the series
        // name — first hit is the mean line), append "± std · n runs".
        formatter: (params: any) => {
          const seen = new Set<string>();
          const rows = (Array.isArray(params) ? params : [params]).filter((p: any) => {
            if (!p || !p.seriesName || p.seriesType === "custom") return false;
            if (seen.has(p.seriesName)) return false;
            seen.add(p.seriesName);
            return true;
          });
          if (rows.length === 0) return "";
          const header = `<div style="margin-bottom:4px;opacity:0.75">eval ${rows[0].axisValue}</div>`;
          const body = rows.map((p: any) => {
            const x = Array.isArray(p.value) ? p.value[0] : p.axisValue;
            const y = Array.isArray(p.value) ? p.value[1] : p.value;
            const sd = stdLookup.get(p.seriesName)?.get(x);
            const n = nRunsByName.get(p.seriesName) ?? 1;
            const dot = `<span style="display:inline-block;width:8px;height:8px;background:${p.color};border-radius:50%;margin-right:6px"></span>`;
            const suffix = n > 1
              ? ` <span style="opacity:0.65">± ${sd !== undefined ? fmt3(sd) : "0.000"} · ${n} runs</span>`
              : "";
            return `${dot}${p.seriesName}: <b>${fmt3(y)}</b>${suffix}`;
          }).join("<br/>");
          return header + body;
        },
      },
      // No in-chart legend: the Final-metrics table on the right carries the
      // color → series mapping (hover a row to highlight the series here).
      grid: { left: 48, right: 16, top: 16, bottom: 40 },
      xAxis: { type: "value", name: "eval", nameLocation: "middle", nameGap: 24, minInterval: 1 },
      // Tight-fit y to the drawn envelope instead of a fixed [0,1] — with few
      // low-HV runs a fixed axis is mostly blank. Functional min/max are
      // re-evaluated by ECharts on legend toggles, so hiding a dominant
      // series re-fits the axis to what remains. Clamped to [0,1] (norm HV).
      yAxis: {
        type: "value",
        name: "norm HV",
        // Center + rotate the axis title along the axis (like the Pareto chart)
        // so it doesn't collide with the legend pinned at top: 0.
        nameLocation: "middle",
        nameGap: 36,
        min: (v: { min: number; max: number }) =>
          Number.isFinite(v.min) ? Math.max(0, v.min - Math.max((v.max - v.min) * 0.05, 0.005)) : 0,
        max: (v: { min: number; max: number }) =>
          Number.isFinite(v.max) ? Math.min(1, v.max + Math.max((v.max - v.min) * 0.05, 0.005)) : 1,
        axisLabel: { formatter: (val: number) => val.toFixed(2) },
      },
      series: chartSeries,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, series, showRuns, hiddenSet]);

  const paretoOption: EChartsOption = useMemo(() => {
    const chartSeries: any[] = [];
    for (const s of series) {
      if (hiddenSet.has(s.key)) continue;
      const present = s.members
        .map((m) => byLabel.get(m))
        .filter((p): p is ProjectHvSeries => !!p && !p.missing);
      if (present.length === 0) continue;
      const grouped = present.length > 1;
      // `focus: "series"` mirrors the HV chart: highlighting a series (from
      // the table rows or in-chart hover) blurs every other series, so the
      // hovered one visibly pops instead of the highlight being a no-op.
      chartSeries.push({ // all valid evals of all runs, faint
        name: s.name, id: `${s.key}__pts`, type: "scatter", symbolSize: 5,
        itemStyle: { color: s.color, opacity: 0.3 },
        emphasis: { focus: "series", itemStyle: { opacity: 0.9 } },
        cursor: "pointer",
        data: present.flatMap((p) =>
          (p.points ?? []).filter((pt) => pt.valid).map((pt) =>
            ({ value: [pt.perf, pt.qual], name: p.label, evalId: pt.eval_id }))),
      });
      for (const p of present) { // one Pareto front per run (same legend name → toggles together)
        chartSeries.push({
          name: s.name, id: `${s.key}__front__${p.label}`, type: "line",
          showSymbol: true, symbolSize: grouped ? 5 : 7,
          lineStyle: { color: s.color, width: grouped ? 1.5 : 2 },
          itemStyle: { color: s.color },
          emphasis: { focus: "series", lineStyle: { width: grouped ? 2.5 : 3 } },
          cursor: "pointer",
          data: (p.pareto ?? []).map((pt) =>
            ({ value: [pt.perf, pt.qual], name: p.label, evalId: pt.eval_id ?? undefined })),
        });
      }
    }
    return {
      tooltip: {
        trigger: "item",
        formatter: (pp: any) => {
          const member = pp.name && pp.name !== pp.seriesName
            ? `<span style="opacity:.7"> · ${pp.name}</span>` : "";
          const evalId = pp.data?.evalId;
          const idLine = typeof evalId === "number" ? ` · eval #${evalId}` : "";
          return `${pp.seriesName}${member}${idLine}<br/>perf ${Number(pp.value[0]).toFixed(2)} · qual ${Number(pp.value[1]).toFixed(3)}` +
            `<br/><span style="opacity:.6">click → open ${typeof evalId === "number" ? "this eval" : "project"}</span>`;
        },
      },
      // No in-chart legend — the Final-metrics table is the legend.
      grid: { left: 48, right: 16, top: 16, bottom: 40 },
      xAxis: { type: "value", name: "performance", nameLocation: "middle", nameGap: 24, scale: true },
      yAxis: { type: "value", name: "quality", nameLocation: "middle", nameGap: 36, scale: true },
      series: chartSeries,
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, series, hiddenSet]);

  // ── Click-through: series → its project page; eval point → that eval ──
  const navigate = useNavigate();
  const goToProject = useCallback(
    (label: string) => {
      if (!label) return;
      navigate({ to: "/", search: { project: label } });
    },
    [navigate],
  );
  const goToEval = useCallback(
    (label: string, evalId: number) => {
      if (!label) return;
      navigate({
        to: "/",
        search: { project: label, tab: "performance", eval: evalId },
      });
    },
    [navigate],
  );

  /** Resolve an HV-chart series click to ONE project: single-member series
   *  directly; a member run line of a group via its `__run__<label>` id. A
   *  group's mean line maps to several projects — no navigation. */
  const projectOfHvClick = useCallback(
    (params: any): string | null => {
      const id = String(params?.seriesId ?? "");
      const runIdx = id.indexOf("__run__");
      if (runIdx >= 0) return id.slice(runIdx + "__run__".length);
      const s = series.find((x) => x.key === id || x.name === params?.seriesName);
      return s && s.members.length === 1 ? s.members[0] : null;
    },
    [series],
  );

  // Hovering a mean line also lights up its band (they share the name; the
  // band is a separate custom series, so emphasis must be dispatched by name).
  const hvEvents = {
    mouseover: (params: any, chart: any) => {
      if (params.componentType === "series" && params.seriesName && chart) {
        chart.dispatchAction({ type: "highlight", seriesName: params.seriesName });
      }
    },
    mouseout: (params: any, chart: any) => {
      if (params.componentType === "series" && params.seriesName && chart) {
        chart.dispatchAction({ type: "downplay", seriesName: params.seriesName });
      }
    },
    click: (params: any) => {
      const label = projectOfHvClick(params);
      if (label) goToProject(label);
    },
  };

  // Pareto scatter/front points carry {name: project label, evalId} — click
  // opens that eval's Performance view in the project. Hovering any point
  // lights the whole series by NAME (scatter + every front line of the
  // group), same idiom as the HV chart's mouseover.
  const paretoEvents = {
    mouseover: (params: any, chart: any) => {
      if (params.componentType === "series" && params.seriesName && chart) {
        chart.dispatchAction({ type: "highlight", seriesName: params.seriesName });
      }
    },
    mouseout: (params: any, chart: any) => {
      if (params.componentType === "series" && params.seriesName && chart) {
        chart.dispatchAction({ type: "downplay", seriesName: params.seriesName });
      }
    },
    click: (params: any) => {
      const d = params?.data;
      const label = typeof d?.name === "string" ? d.name : "";
      const evalId = d?.evalId;
      if (label && typeof evalId === "number" && Number.isFinite(evalId)) {
        goToEval(label, evalId);
      } else if (label) {
        goToProject(label);
      }
    },
  };

  // The Final-metrics table doubles as the legend: hovering a row highlights
  // that series in BOTH charts (the in-chart legends are gone).
  const hvChartRef = useRef<ECharts | null>(null);
  const paretoChartRef = useRef<ECharts | null>(null);
  const onHvReady = useCallback((c: ECharts) => {
    hvChartRef.current = c;
  }, []);
  const onParetoReady = useCallback((c: ECharts) => {
    paretoChartRef.current = c;
  }, []);
  const highlightSeries = useCallback(
    (s: { key: string; name: string }, on: boolean) => {
      if (hiddenSet.has(s.key)) return; // nothing drawn to highlight
      for (const chart of [hvChartRef.current, paretoChartRef.current]) {
        chart?.dispatchAction({
          type: on ? "highlight" : "downplay",
          seriesName: s.name,
        });
      }
    },
    [hiddenSet],
  );

  // ── Render ────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-1 min-h-0">
      {/* Project picker */}
      <SidePanel className="w-[260px]">
        <SidePanelHeader>
          <span className="text-[10px] font-semibold uppercase tracking-[1px] text-muted-foreground">
            PROJECTS ({selected.length}/{projects.length})
          </span>
        </SidePanelHeader>
        <div className="flex-1 overflow-auto">
          {projects.length === 0 && (
            <div className="px-3 py-6 text-center text-[11px] text-muted-foreground">No projects found.</div>
          )}
          {projects.map((pr) => {
            const on = selected.includes(pr.name);
            return (
              <button
                key={pr.name}
                type="button"
                title={pr.name || "(root)"}
                onClick={() => toggle(pr.name)}
                className={cn(
                  "flex w-full items-center gap-2.5 px-3 py-2 text-left transition-colors hover:bg-muted/60",
                  on && "bg-primary/5",
                )}
              >
                <Checkbox checked={on} className="pointer-events-none" />
                {on && (
                  <span
                    className="size-2 shrink-0 rounded-full"
                    style={{ background: seriesColorOf.get(pr.name) }}
                  />
                )}
                <div className="flex min-w-0 flex-1 flex-col">
                  <span className={cn("truncate text-xs", on ? "font-semibold text-primary" : "text-foreground")}>
                    {pr.name || "(root)"}
                  </span>
                  <span className="text-[10px] text-muted-foreground">{pr.n_evaluations} evals · {pr.status}</span>
                </div>
              </button>
            );
          })}
        </div>
      </SidePanel>

      {/* Series tray + charts + summary */}
      <div className="flex flex-1 min-h-0 flex-col">
        <div className="flex items-center gap-3 border-b border-border bg-card px-4 py-2">
          <BarChart3 size={15} className="text-primary" />
          <span className="text-[12px] font-semibold text-foreground">Hypervolume comparison</span>
          {loading && <Loader2 size={13} className="animate-spin text-muted-foreground" />}
          <div className="ml-auto flex items-center gap-4">
            {hasMultiRunSeries && (
              <label className="flex cursor-pointer items-center gap-1.5 text-[11px] text-muted-foreground">
                <Switch checked={showRuns} onCheckedChange={setShowRuns} className="scale-75" />
                individual runs
              </label>
            )}
            <div className="flex items-center gap-2">
              <span className="text-[11px] text-muted-foreground">quality metric</span>
              <Select value={metric} onValueChange={setMetric}>
                <SelectTrigger className="h-7 w-[230px] text-[11px]"><SelectValue placeholder="metric" /></SelectTrigger>
                <SelectContent>
                  {(data?.available_metrics ?? []).map((m) => <SelectItem key={m} value={m}>{m}</SelectItem>)}
                </SelectContent>
              </Select>
            </div>
          </div>
        </div>

        {series.length > 0 && (
          <SeriesTray
            series={series}
            armed={armedValid}
            onToggleArm={toggleArm}
            onMerge={mergeArmed}
            hidden={hiddenSet}
            onToggleHide={toggleHidden}
            expandedGroupId={expandedGroupId}
            onToggleExpand={setExpandedGroupId}
            onRenameGroup={renameGroup}
            onSplitGroup={splitGroup}
            onRemoveMember={removeMember}
          />
        )}

        <div className="flex min-h-0 flex-1 flex-col overflow-auto">
        {selected.length === 0 ? (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
            <BarChart3 size={34} className="text-muted-foreground/40" />
            <span className="text-sm text-muted-foreground">Select projects to compare their hypervolume</span>
            <span className="text-[11px] text-muted-foreground/70">
              Seeds of the same config? Merge them in the series strip — one line, mean ± std.
            </span>
          </div>
        ) : data?.error ? (
          <div className="m-4 flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-3 py-2">
            <CircleAlert size={14} className="mt-0.5 shrink-0 text-amber-600 dark:text-amber-400" />
            <span className="text-[11px] text-amber-700 dark:text-amber-300">{data.error}</span>
          </div>
        ) : (
          <div className="space-y-4 p-4">
            {/* Benchmark chart (left) + concrete final metrics (right) —
                charts slightly narrower than 50% so the table breathes. */}
            <div className="grid items-start gap-4 lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.1fr)]">
              {/* Left column — both benchmark charts, stacked */}
              <div className="min-w-0 space-y-4">
                {/* HV convergence */}
                <div>
                  <div className="mb-1 text-[11px] font-semibold text-muted-foreground">
                    HV convergence (normalized, vs eval budget{hasMultiRunSeries ? " — merged series: mean ± 1 std" : ""})
                  </div>
                  <EChartsView option={hvOption} notMerge style={{ height: 300 }} onEvents={hvEvents} onChartReady={onHvReady} />
                </div>
                {/* Evals + Pareto fronts per run */}
                <div>
                  <div className="mb-1 text-[11px] font-semibold text-muted-foreground">
                    Evals + Pareto fronts per run (raw performance × quality{data?.metric ? ` — ${data.metric}` : ""})
                  </div>
                  <EChartsView option={paretoOption} notMerge style={{ height: 320 }} onEvents={paretoEvents} onChartReady={onParetoReady} />
                </div>
              </div>
              {/* Right column — concrete final metrics per series */}
              <div className="min-w-0 overflow-x-auto">
                <div className="mb-1 text-[11px] font-semibold text-muted-foreground">
                  Final metrics
                </div>
                <table className="w-full text-[11px]">
              <thead>
                <tr className="text-left text-muted-foreground">
                  <SortableTh label="series" k="name" sort={tableSort} onCycle={cycleSort} className="w-[260px]" />
                  <SortableTh label="final norm HV" k="hv" sort={tableSort} onCycle={cycleSort} />
                  <SortableTh label="best quality" k="qual" sort={tableSort} onCycle={cycleSort} />
                  <SortableTh label="best perf" k="perf" sort={tableSort} onCycle={cycleSort} />
                  <SortableTh label="valid/evals" k="valid" sort={tableSort} onCycle={cycleSort} />
                  <SortableTh label="n_init" k="init" sort={tableSort} onCycle={cycleSort} />
                </tr>
              </thead>
              <tbody>
                {tableRows.flatMap(({ s, members, present, finals, bestQ, bestP, nValid, nEvals, nInits }) => {
                  // Group-backed series keep their group row even with a
                  // single selected member, so the table name matches the
                  // chart legend (which uses the series name).
                  const isGroup = !!s.group;

                  const isHidden = hiddenSet.has(s.key);
                  const memberRow = (p: ProjectHvSeries, indent: boolean) => (
                    <tr
                      key={`${s.key}:${p.label}`}
                      onMouseEnter={() => highlightSeries(s, true)}
                      onMouseLeave={() => highlightSeries(s, false)}
                      onClick={() => !p.missing && goToProject(p.label ?? "")}
                      title={p.missing ? undefined : "Open this project"}
                      className={cn(
                        "border-t border-border/60",
                        indent && "text-muted-foreground",
                        !p.missing && "cursor-pointer hover:bg-muted/50",
                        isHidden && "opacity-45",
                      )}
                    >
                      <td className="py-1" title={p.label}>
                        <div className={cn("flex max-w-[260px] items-center gap-1.5", indent && "pl-4")}>
                          {indent
                            ? <CornerDownRight size={10} className="shrink-0 text-muted-foreground/60" />
                            : !p.missing && <span className="size-2 shrink-0 rounded-full" style={{ background: s.color }} />}
                          <span className="min-w-0 truncate font-mono">{p.label}</span>
                        </div>
                      </td>
                      {p.missing ? (
                        <td colSpan={5} className="py-1 italic text-muted-foreground">no all_evaluations.csv</td>
                      ) : (
                        <>
                          <td className={cn("py-1 font-mono", !indent && "font-semibold text-primary")}>{p.final_hv?.toFixed(3)}</td>
                          <td className="py-1 font-mono">{p.best_qual?.toFixed(3) ?? "—"}</td>
                          <td className="py-1 font-mono">{p.best_perf?.toFixed(1) ?? "—"}</td>
                          <td className="py-1 font-mono">{p.n_valid}/{p.n_evals}</td>
                          <td className="py-1 font-mono">{p.n_init}</td>
                        </>
                      )}
                    </tr>
                  );

                  if (!isGroup) return members.map((p) => memberRow(p, false));

                  const groupRow = (
                    <tr
                      key={s.key}
                      onMouseEnter={() => highlightSeries(s, true)}
                      onMouseLeave={() => highlightSeries(s, false)}
                      className={cn("border-t border-border/60", isHidden && "opacity-45")}
                    >
                      <td className="py-1" title={s.name}>
                        <div className="flex max-w-[260px] items-center gap-1.5">
                          <span className="size-2 shrink-0 rounded-full" style={{ background: s.color }} />
                          <span className="min-w-0 truncate font-mono font-semibold text-foreground/90">{s.name}</span>
                          <span className="shrink-0 rounded bg-muted px-1 text-[10px] font-semibold text-muted-foreground">×{present.length}</span>
                        </div>
                      </td>
                      {present.length === 0 ? (
                        <td colSpan={5} className="py-1 italic text-muted-foreground">no data in any run</td>
                      ) : (
                        <>
                          <td className="py-1 font-mono font-semibold text-primary">
                            {fmt3(finals.mean)}
                            {present.length > 1 && <span className="font-normal text-muted-foreground"> ± {fmt3(finals.std)}</span>}
                          </td>
                          <td className="py-1 font-mono">{bestQ != null ? fmt3(bestQ) : "—"}</td>
                          <td className="py-1 font-mono">{bestP != null ? bestP.toFixed(1) : "—"}</td>
                          <td className="py-1 font-mono">{nValid}/{nEvals}</td>
                          <td className="py-1 font-mono">{nInits.join("/")}</td>
                        </>
                      )}
                    </tr>
                  );
                  return [groupRow, ...members.map((p) => memberRow(p, true))];
                })}
              </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
        </div>
      </div>
    </div>
  );
}
