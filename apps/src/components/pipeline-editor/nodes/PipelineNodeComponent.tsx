import React, { memo } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import type { PipelineNodeData } from "../types";
import { NODE_TYPE_COLORS } from "../constants";
import { useDataflowLens } from "../DataflowLensContext";
import { REACT_LOOP_COLOR } from "../DataflowBar";
import { CircleDashed } from "lucide-react";

function PipelineNodeInner({ data, selected }: NodeProps & { data: PipelineNodeData }) {
  const d = data as PipelineNodeData;
  const colors = NODE_TYPE_COLORS[d.nodeType] || NODE_TYPE_COLORS.generator;

  // react dataflow lens: ghost the stages the react loop strips from a
  // trial's node_lines; ring the stages the loop actually drives each round.
  const { lens, reactActiveStages } = useDataflowLens();
  const stripped = lens === "react" && !reactActiveStages.has(d.nodeType);
  const looped = lens === "react" && reactActiveStages.has(d.nodeType);

  return (
    <div
      style={{
        background: "#fff",
        border: `2px ${d.optional ? "dashed" : "solid"} ${
          selected ? colors.accent : stripped ? "#E2E8F0" : colors.border
        }`,
        borderRadius: 12,
        width: 260,
        boxShadow: selected
          ? `0 0 0 2px ${colors.accent}33`
          : looped
            ? `0 0 0 2px ${REACT_LOOP_COLOR}40`
            : "0 1px 4px rgba(15, 23, 42, 0.06)",
        cursor: "pointer",
        transition: "border-color 0.15s, box-shadow 0.15s",
        position: "relative",
      }}
    >
      {/* react lens — stripped-stage tag (kept at full opacity, floats on
          the ghosted content below) */}
      {stripped && (
        <div style={{
          position: "absolute",
          top: -9,
          left: 10,
          fontSize: 9,
          fontWeight: 600,
          color: "#64748B",
          background: "#fff",
          border: "1px solid #E2E8F0",
          borderRadius: 4,
          padding: "1px 5px",
          zIndex: 2,
          whiteSpace: "nowrap",
        }}>
          stripped in react
        </div>
      )}
      {/* Optional indicator — top right */}
      {d.optional && (
        <div style={{
          position: "absolute",
          top: -8,
          right: -8,
          width: 20,
          height: 20,
          borderRadius: 10,
          background: "#8B5CF6",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          boxShadow: "0 1px 4px rgba(139, 92, 246, 0.35)",
          zIndex: 1,
        }}>
          <CircleDashed size={11} color="#fff" strokeWidth={2.5} />
        </div>
      )}

      {/* Ghosting wrapper — dims header + modules under the react lens while
          the tag above stays crisp. */}
      <div
        style={{
          opacity: stripped ? 0.35 : 1,
          filter: stripped ? "grayscale(0.7)" : "none",
          transition: "opacity 0.15s, filter 0.15s",
        }}
      >
      {/* Header */}
      <div
        style={{
          background: colors.bg,
          borderBottom: `1px solid ${colors.border}`,
          padding: "8px 14px",
          borderRadius: "10px 10px 0 0",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <span
          style={{
            fontSize: 12,
            fontWeight: 700,
            color: colors.accent,
            letterSpacing: 0.3,
          }}
        >
          {d.nodeIndex + 1}. {d.nodeType.replace(/_/g, " ").toUpperCase()}
        </span>
        {d.topK !== undefined && (
          <span
            style={{
              fontSize: 10,
              fontWeight: 600,
              color: colors.accent,
              background: `${colors.accent}15`,
              padding: "2px 6px",
              borderRadius: 4,
            }}
          >
            k={typeof d.topK === "object" ? JSON.stringify(d.topK) : d.topK}
          </span>
        )}
      </div>

      {/* Modules */}
      <div style={{ padding: "8px 14px 10px" }}>
        {d.modules.length === 0 ? (
          <div style={{ fontSize: 11, color: "#94A3B8", fontStyle: "italic" }}>
            No modules
          </div>
        ) : (
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {d.modules.map((mod, i) => (
              <span
                key={i}
                style={{
                  fontSize: 10,
                  fontWeight: 500,
                  color: "#475569",
                  background: "#F1F5F9",
                  padding: "3px 8px",
                  borderRadius: 6,
                  border: "1px solid #E2E8F0",
                }}
              >
                {mod.component}
              </span>
            ))}
          </div>
        )}

        {/* Strategy metrics */}
        {d.strategy?.metrics && d.strategy.metrics.length > 0 && (
          <div style={{ marginTop: 6, display: "flex", gap: 3, flexWrap: "wrap" }}>
            {d.strategy.metrics.map((m: string) => (
              <span
                key={m}
                style={{
                  fontSize: 9,
                  color: "#94A3B8",
                  background: "#F8FAFC",
                  padding: "1px 5px",
                  borderRadius: 3,
                }}
              >
                {m}
              </span>
            ))}
          </div>
        )}
      </div>
      </div>

      <Handle
        type="target"
        position={Position.Left}
        style={{
          width: 8,
          height: 8,
          background: stripped ? "#CBD5E1" : colors.accent,
          border: `2px solid #fff`,
        }}
      />
      <Handle
        type="source"
        position={Position.Right}
        style={{
          width: 8,
          height: 8,
          background: stripped ? "#CBD5E1" : colors.accent,
          border: `2px solid #fff`,
        }}
      />
    </div>
  );
}

export const PipelineNodeComponent = memo(PipelineNodeInner);
