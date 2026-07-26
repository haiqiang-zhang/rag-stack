import React, { useEffect, useMemo, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { SectionHeader } from "@/components/ui/SectionHeader";
import { RagoSweepChart } from "@/components/charts/RagoSweepChart";
import { ResetZoomPill, type ZoomState } from "@/components/charts/zoomControls";
import { AxisSelector } from "@/components/selectors/AxisSelector";
import { RAGO_AXIS_OPTIONS, RAGO_COLOR_OPTIONS } from "@/types";
import type { RagoSweepPoint, AssemblyResult } from "@/types";

interface SweepCardProps {
  evalId: number | null;
  sweepPoints: RagoSweepPoint[];
  assembly: AssemblyResult | null;
  selectedSweepRow: RagoSweepPoint | null;
  selectedBestRow: RagoSweepPoint | null;
  onPointClick: (pt: RagoSweepPoint) => void;
  height?: number;
}

export function SweepCard({
  evalId,
  sweepPoints,
  assembly,
  selectedSweepRow,
  selectedBestRow,
  onPointClick,
  height = 320,
}: SweepCardProps) {
  const [xKey, setXKey] = useState("batch_size_decode");
  const [yKey, setYKey] = useState("qps");
  const [colorKey, setColorKey] = useState("placement_policy");
  const [zoom, setZoom] = useState<ZoomState | null>(null);

  const availableAxisOptions = useMemo(() => {
    if (sweepPoints.length === 0) return RAGO_AXIS_OPTIONS;
    return RAGO_AXIS_OPTIONS.filter((opt) =>
      sweepPoints.some((pt) => {
        const value = Number(pt[opt.key]);
        return Number.isFinite(value);
      }),
    );
  }, [sweepPoints]);

  const availableColorOptions = useMemo(() => {
    if (sweepPoints.length === 0) return RAGO_COLOR_OPTIONS;
    return RAGO_COLOR_OPTIONS.filter((opt) =>
      sweepPoints.some((pt) => {
        const value = pt[opt.key];
        return value != null && String(value).trim() !== "";
      }),
    );
  }, [sweepPoints]);

  useEffect(() => {
    if (availableAxisOptions.length === 0) return;
    if (!availableAxisOptions.some((o) => o.key === xKey)) {
      setXKey(
        availableAxisOptions.find((o) => o.key === "batch_size_decode")?.key ??
          availableAxisOptions.find((o) => o.key === "latency_s")?.key ??
          availableAxisOptions[0].key,
      );
    }
    if (!availableAxisOptions.some((o) => o.key === yKey)) {
      setYKey(
        availableAxisOptions.find((o) => o.key === "qps")?.key ??
          availableAxisOptions[0].key,
      );
    }
  }, [availableAxisOptions, xKey, yKey]);

  useEffect(() => {
    if (availableColorOptions.length === 0) return;
    if (!availableColorOptions.some((o) => o.key === colorKey)) {
      setColorKey(
        availableColorOptions.find((o) => o.key === "placement_policy")?.key ??
          availableColorOptions[0].key,
      );
    }
  }, [availableColorOptions, colorKey]);

  const xOpt = availableAxisOptions.find((o) => o.key === xKey);
  const yOpt = availableAxisOptions.find((o) => o.key === yKey);

  return (
    <Card className="min-w-0 flex-1 basis-[420px]">
      <CardContent>
        <SectionHeader
          eyebrow={evalId != null ? `EVAL #${String(evalId).padStart(4, "0")}` : "NO SELECTION"}
          title="HARDWARE SWEEP"
          right={
            evalId != null && (
              <ResetZoomPill
                visible={!!zoom?.isZoomed}
                onClick={() => zoom?.reset()}
              />
            )
          }
        />
        {evalId == null ? (
          <div
            className="flex items-center justify-center"
            style={{ height }}
          >
            <span className="text-xs text-muted-foreground">
              Click an evaluation in the Pareto chart to see its hardware sweep
            </span>
          </div>
        ) : (
          <div className="flex flex-col gap-2.5">
            <div className="flex flex-wrap gap-1.5">
              <AxisSelector label="X" options={availableAxisOptions} selectedKey={xKey} onSelect={setXKey} />
              <AxisSelector label="Y" options={availableAxisOptions} selectedKey={yKey} onSelect={setYKey} />
              <AxisSelector label="Color" options={availableColorOptions} selectedKey={colorKey} onSelect={setColorKey} />
            </div>
            <RagoSweepChart
              sweepPoints={sweepPoints}
              assembly={assembly}
              xKey={xKey}
              yKey={yKey}
              colorKey={colorKey}
              xLabel={xOpt?.label ?? xKey}
              yLabel={yOpt?.label ?? yKey}
              height={height}
              selectedBestRow={selectedBestRow}
              selectedSweepRow={selectedSweepRow}
              onPointClick={onPointClick}
              onZoomStateChange={setZoom}
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
