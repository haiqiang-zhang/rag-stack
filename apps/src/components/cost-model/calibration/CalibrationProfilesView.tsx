import React, { useEffect, useMemo, useState } from "react";
import { Loader2, Archive, RefreshCw, Trash2, Inbox } from "lucide-react";

import { useCalibrationProfiles } from "@/hooks/useCalibrationProfiles";
import { deleteCalibrationProfile } from "@/api/costModelClient";
import { CalibrationProfileList, mapeColor } from "./CalibrationProfileList";
import { CalibrationProfileDetail } from "./CalibrationProfileDetail";
import { Dropdown, type DropdownOption } from "@/components/selectors/Dropdown";
import { Button } from "@/components/ui/button";
import { Badge, BadgeText } from "@/components/ui/LegacyBadge";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

// from old theme.ts
const ERROR = "#EF4444";

interface Props {
  costModel: string;
}

type SortKey = "mape_asc" | "timestamp_desc" | "module_asc";

const SORT_OPTIONS: DropdownOption[] = [
  { key: "mape_asc", label: "MAPE · best first" },
  { key: "timestamp_desc", label: "Newest first" },
  { key: "module_asc", label: "Module · A-Z" },
];

export function CalibrationProfilesView({ costModel }: Props) {
  const { profiles, supported, loading, error, refetch } = useCalibrationProfiles(costModel);
  const [selected, setSelected] = useState<number | null>(null);
  const [sortBy, setSortBy] = useState<SortKey>("mape_asc");
  const [hwFilter, setHwFilter] = useState<string>("all");
  const [deleteError, setDeleteError] = useState<string | null>(null);

  // Hardware filter options derived from loaded profiles
  const hwOptions: DropdownOption[] = useMemo(() => {
    if (!profiles) return [{ key: "all", label: "All hardware" }];
    const hws = Array.from(new Set(profiles.map((p) => p.hardware_key))).filter(Boolean).sort();
    return [
      { key: "all", label: `All hardware (${profiles.length})` },
      ...hws.map((hw) => ({ key: hw, label: hw })),
    ];
  }, [profiles]);

  // Filter + sort pipeline (keeps selection stable by filename)
  const visible = useMemo(() => {
    if (!profiles) return [];
    const filtered = hwFilter === "all"
      ? profiles
      : profiles.filter((p) => p.hardware_key === hwFilter);
    const sorted = [...filtered];
    if (sortBy === "mape_asc") {
      sorted.sort((a, b) => (a.fit_mape ?? Infinity) - (b.fit_mape ?? Infinity));
    } else if (sortBy === "timestamp_desc") {
      sorted.sort((a, b) => (b.timestamp ?? "").localeCompare(a.timestamp ?? ""));
    } else {
      sorted.sort((a, b) => (a.component || a.filename).localeCompare(b.component || b.filename));
    }
    return sorted;
  }, [profiles, hwFilter, sortBy]);

  // Aggregate health: count + avg MAPE + # poor-fit
  const health = useMemo(() => {
    if (!visible.length) return null;
    const mapes = visible.filter((p) => !p.error).map((p) => p.fit_mape).filter((m) => Number.isFinite(m));
    if (!mapes.length) return null;
    const avg = mapes.reduce((s, m) => s + m, 0) / mapes.length;
    const poor = mapes.filter((m) => m >= 0.2).length;
    return { count: visible.length, avgMape: avg, poor };
  }, [visible]);

  // Keep selection valid: prefer current filename if still present; else pick the first
  const selectedProfile = selected != null ? visible[selected] : null;
  useEffect(() => {
    if (!visible.length) {
      if (selected != null) setSelected(null);
      return;
    }
    if (selected == null || selected >= visible.length) {
      setSelected(0);
    }
  }, [visible, selected]);

  const handleDelete = async () => {
    if (!selectedProfile || selectedProfile.can_delete !== true) return;
    if (!confirm(`Delete calibration profile ${selectedProfile.filename}?`)) return;
    setDeleteError(null);
    try {
      await deleteCalibrationProfile(costModel, selectedProfile.filename);
      await refetch();
    } catch (err) { setDeleteError(err instanceof Error ? err.message : String(err)); }
  };

  // ── Unsupported cost model ─────────────────────────────────────────────
  if (!supported) {
    return (
      <div className="flex flex-1 items-center justify-center p-8">
        <div className="flex max-w-[420px] flex-col items-center gap-3 rounded-xl bg-card p-8 shadow-sm">
          <Archive className="size-7 text-muted-foreground/60" />
          <span className="text-sm font-semibold text-foreground">
            Calibration profiles not available
          </span>
          <span className="text-center text-xs text-muted-foreground">
            The {costModel} cost model does not expose persisted profiles via
            the generic endpoint.
          </span>
        </div>
      </div>
    );
  }

  // ── Main layout ────────────────────────────────────────────────────────
  return (
    <div className="flex flex-1 min-h-0 min-w-0 flex-col overflow-hidden lg:flex-row">
      <div className="min-w-0 shrink-0 space-y-2 border-b p-3 lg:hidden">
        <div className="flex items-center gap-2">
          <Select value={selected == null ? undefined : String(selected)} onValueChange={(value) => setSelected(Number(value))}>
            <SelectTrigger aria-label="Calibration profile" className="min-w-0 flex-1 text-xs"><SelectValue placeholder={loading ? "Loading profiles…" : "Select a profile"} /></SelectTrigger>
            <SelectContent>{visible.map((profile, index) => <SelectItem key={`${profile.filename}:${profile.record_key ?? index}`} value={String(index)}>
              {profile.model_key || profile.component || profile.filename} · {profile.hardware_key || "Unknown hardware"}{profile.bits ? ` · ${profile.bits}` : ""}
            </SelectItem>)}</SelectContent>
          </Select>
          <Button type="button" variant="ghost" size="icon" className="size-8 shrink-0" aria-label="Refresh profiles" onClick={() => void refetch()}><RefreshCw className="size-3.5" /></Button>
        </div>
        {error && <p role="alert" className="break-words text-xs text-destructive">{error.message}</p>}
        {!loading && !error && profiles?.length === 0 && <p className="text-xs text-muted-foreground">No saved profiles for this cost model.</p>}
        <details className="text-xs text-muted-foreground"><summary className="cursor-pointer">{visible.length} profiles · filter and sort</summary>
          <div className="mt-2 grid min-w-0 gap-2 sm:grid-cols-2"><Dropdown options={SORT_OPTIONS} selectedKey={sortBy} onSelect={(value) => setSortBy(value as SortKey)} />{hwOptions.length > 2 && <Dropdown options={hwOptions} selectedKey={hwFilter} onSelect={setHwFilter} />}</div>
        </details>
      </div>
      {/* ── Sidebar ───────────────────────────────────────────────── */}
      <div className="hidden min-h-0 w-[240px] shrink-0 flex-col overflow-hidden border-r border-border bg-card lg:flex">
        {/* Header row */}
        <div className="flex items-center justify-between gap-1.5 px-3.5 pt-3.5 pb-2">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold uppercase tracking-[1.2px] text-muted-foreground">
              PROFILES
            </span>
            <Badge variant="accent">
              <BadgeText>{profiles?.length ?? 0}</BadgeText>
            </Badge>
          </div>
          <Button type="button" variant="ghost" size="icon" onClick={() => refetch()} className="size-6">
            {loading ? (
              <Loader2 className="size-3 animate-spin text-primary" />
            ) : (
              <RefreshCw className="size-3 text-muted-foreground" />
            )}
          </Button>
        </div>

        {/* Health banner */}
        {health && (
          <div className="mx-2.5 mb-2 flex flex-col gap-1 rounded-lg border border-border bg-card px-2.5 py-2">
            <div className="flex items-baseline gap-2.5">
              <span className="font-mono text-sm font-bold text-foreground">
                {(health.avgMape * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] text-muted-foreground">avg MAPE</span>
            </div>
            {health.poor > 0 ? (
              <div className="flex items-center gap-1.5">
                <div
                  className="size-1.5 rounded-full"
                  style={{ backgroundColor: ERROR }}
                />
                <span className="text-[10px] text-muted-foreground">
                  {health.poor} poor fit ({'>'}20%)
                </span>
              </div>
            ) : (
              <span className="text-[10px] text-muted-foreground">
                Recorded fit errors below 20%
              </span>
            )}
          </div>
        )}

        {/* Sort + hardware filter */}
        <div className="flex flex-col gap-1.5 px-2.5 pb-2">
          <Dropdown
            options={SORT_OPTIONS}
            selectedKey={sortBy}
            onSelect={(k) => setSortBy(k as SortKey)}
          />
          {hwOptions.length > 2 && (
            <Dropdown
              options={hwOptions}
              selectedKey={hwFilter}
              onSelect={setHwFilter}
            />
          )}
        </div>

        {/* Divider */}
        <div className="h-px bg-border" />

        {/* Scrollable list */}
        {loading ? (
          <div className="flex items-center justify-center p-5">
            <Loader2 className="size-4 animate-spin text-muted-foreground" />
          </div>
        ) : error ? (
          <div className="px-3.5 py-3.5">
            <span className="text-[11px] text-muted-foreground">
              Load failed: {error.message}
            </span>
          </div>
        ) : profiles && profiles.length === 0 ? (
          <div className="flex flex-col items-center gap-2 p-5">
            <Inbox className="size-5 text-muted-foreground/60" />
            <span className="text-center text-[11px] text-muted-foreground">
              No profiles yet. Run a calibration from the
            </span>
            <span className="text-[11px] font-semibold text-primary">
              Calibrate
            </span>
            <span className="-mt-1 text-center text-[11px] text-muted-foreground">
              tab to create one.
            </span>
          </div>
        ) : (
          <div className="min-h-0 min-w-0 flex-1 overflow-auto">
            <CalibrationProfileList
              profiles={visible}
              selectedIndex={selected}
              onSelect={setSelected}
            />
          </div>
        )}
      </div>

      {/* ── Detail pane ───────────────────────────────────────────── */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        {selectedProfile ? (
          <>
            {/* Sticky detail header */}
            <div className="flex min-w-0 shrink-0 items-center gap-3 border-b border-border bg-card px-4 py-3.5">
              {/* MAPE color chip */}
              <div
                className="h-7 w-1 shrink-0 rounded-[2px]"
                style={{
                  backgroundColor: selectedProfile.error
                    ? ERROR
                    : mapeColor(selectedProfile.fit_mape),
                }}
              />
              <div className="flex min-w-0 flex-1 flex-col gap-0.5">
                <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-0.5">
                  <span className="break-all text-sm font-semibold text-foreground">
                    {selectedProfile.component}
                  </span>
                  <span className="text-[13px] text-muted-foreground">/</span>
                  <span className="min-w-0 break-all font-mono text-xs text-muted-foreground">
                    {selectedProfile.model_key}
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-1.5">
                  <Badge variant="accent">
                    <BadgeText>{selectedProfile.hardware_key}</BadgeText>
                  </Badge>
                  <Badge>
                    <BadgeText>{selectedProfile.bits}</BadgeText>
                  </Badge>
                  {selectedProfile.timestamp && (
                    <span className="font-mono text-[10px] text-muted-foreground/60">
                      {selectedProfile.timestamp}
                    </span>
                  )}
                </div>
              </div>
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={handleDelete}
                disabled={selectedProfile.can_delete !== true}
                aria-label="Delete calibration profile"
                title={selectedProfile.delete_reason || (selectedProfile.can_delete !== true ? "This record cannot be deleted independently." : "Delete calibration profile")}
                className="size-6 shrink-0 hover:bg-destructive/10"
              >
                <Trash2 className="size-[13px] text-destructive" />
              </Button>
            </div>

            {/* Detail content */}
            <div className="min-h-0 min-w-0 flex-1 overflow-auto p-4 pb-12">
              {deleteError && <p role="alert" className="mb-4 text-sm text-destructive">{deleteError}</p>}
              <p className="mb-4 text-xs leading-relaxed text-muted-foreground">Saved calibration records are an inventory. Fit error describes their measured grid; CM binding validates compatibility with a project.</p>
              <CalibrationProfileDetail profile={selectedProfile} />
            </div>
          </>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center gap-2.5 p-8">
            <Inbox className="size-7 text-muted-foreground/60" />
            <span className="text-[13px] text-muted-foreground">
              Select a profile to view its calibration grid
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
