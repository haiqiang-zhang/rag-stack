import React from "react";
import type { ModuleConfig } from "../types";
import type { ComponentCapabilities, ComponentParameter } from "@/api/registry";
import { ConfigValueInput } from "./ConfigValueInput";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface Props {
  module: ModuleConfig;
  onChange: (updated: ModuleConfig) => void;
  onRemove: () => void;
  vectordbNames?: string[];
  schema?: Record<string, ComponentParameter>;
  capabilities?: ComponentCapabilities;
}

export function ModuleParamForm({ module, onChange, onRemove, vectordbNames, schema = {}, capabilities }: Props) {
  const params = [...new Set([...Object.keys(module), ...Object.keys(schema)])]
    .filter((key) => key !== "component").map((key) => [key, module[key]] as const);
  const isVectordbModule = module.component === "vectordb";

  const handleParamChange = (key: string, value: unknown) => {
    const next = { ...module };
    if (value === undefined) delete next[key];
    else next[key] = value;
    onChange(next);
  };

  return (
    <div style={{ padding: "8px 0", borderBottom: "1px solid #E2E8F0" }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 6 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: "#0F766E" }}>
          {module.component}
        </span>
        <button
          onClick={onRemove}
          style={{
            fontSize: 10,
            color: "#EF4444",
            background: "#FEF2F2",
            border: "1px solid #FECACA",
            borderRadius: 4,
            padding: "2px 6px",
            cursor: "pointer",
          }}
        >
          Remove
        </button>
      </div>

      <CapabilityStatus capabilities={capabilities} />

      {params.length === 0 && (
        <div style={{ fontSize: 11, color: "#94A3B8", fontStyle: "italic" }}>No parameters</div>
      )}

      {params.map(([key, value]) => (
        <div key={key} style={{ marginBottom: 8 }}>
          <label title={key} style={{ display: "block", fontSize: 10, fontWeight: 500, color: "#64748B", marginBottom: 4 }}>
            {schema[key]?.label ?? key}
          </label>
          {/* Suggestions apply only to a scalar; list/range selectors stay intact. */}
          {isVectordbModule && key === "vectordb" && (value === undefined || typeof value === "string") && !!vectordbNames?.length && (
            <div style={{ marginBottom: 4 }}>
              <Select
                value={String(value ?? "")}
                onValueChange={(v) => onChange({ ...module, vectordb: v })}
              >
                <SelectTrigger className="h-7 w-full text-[11px]">
                  <SelectValue placeholder="Select vectordb..." />
                </SelectTrigger>
                <SelectContent>
                  {vectordbNames!.map((name) => (
                    <SelectItem key={name} value={name}>
                      {name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}
            <ConfigValueInput
              label={schema[key]?.label ?? key}
              value={value}
              onChange={(next) => handleParamChange(key, next)}
              placeholder={schema[key]?.default !== undefined ? `Default: ${JSON.stringify(schema[key].default)}` : "Value or JSON selector"}
              style={{
                flex: 1,
                fontSize: 11,
                padding: "4px 8px",
                border: "1px solid #E2E8F0",
                borderRadius: 6,
                outline: "none",
                background: "#F8FAFC",
                color: "#1E293B",
                fontFamily: "monospace",
              }}
            />
          {schema[key]?.description && (
            <div style={{ fontSize: 10, color: "#64748B", marginTop: 3 }}>{schema[key].description}</div>
          )}
        </div>
      ))}
    </div>
  );
}

export function CapabilityStatus({ capabilities }: { capabilities?: ComponentCapabilities }) {
  const label = (value: boolean | null | undefined) => value == null ? "unknown" : value ? "supported" : "unavailable";
  return (
    <div style={{ fontSize: 10, color: "#64748B", marginBottom: 8, lineHeight: 1.5 }}>
      <div>Evaluator: {label(capabilities?.execution_supported)} · Measured: {label(capabilities?.measured_supported)} · CM: {label(capabilities?.cost_model_supported)}</div>
      {capabilities?.supported_dataflows && <div>Dataflows: {capabilities.supported_dataflows.join(", ") || "none"}</div>}
      {capabilities?.reason && <div>{capabilities.reason}</div>}
      {capabilities?.reasons?.map((reason) => <div key={reason}>{reason}</div>)}
    </div>
  );
}
