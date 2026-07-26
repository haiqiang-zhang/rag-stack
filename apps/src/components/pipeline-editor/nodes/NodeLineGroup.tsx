import React, { memo } from "react";
import { Handle, Position } from "@xyflow/react";
import type { NodeProps } from "@xyflow/react";
import type { NodeLineLabelData } from "../types";

function NodeLineGroupInner({ data, selected }: NodeProps & { data: NodeLineLabelData }) {
  const d = data as NodeLineLabelData;

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        background: selected ? "rgba(13, 148, 136, 0.03)" : "rgba(241, 245, 249, 0.4)",
        border: `1.5px dashed ${selected ? "#0F766E" : "#CBD5E1"}`,
        borderRadius: 16,
        padding: "10px 16px",
        transition: "border-color 0.15s, background 0.15s",
        cursor: "pointer",
        boxSizing: "border-box",
      }}
    >
      <div
        style={{
          display: "inline-block",
          fontSize: 10,
          fontWeight: 700,
          color: selected ? "#0F766E" : "#64748B",
          letterSpacing: 1,
          textTransform: "uppercase",
          padding: "2px 8px",
          background: selected ? "rgba(13, 148, 136, 0.08)" : "rgba(100, 116, 139, 0.06)",
          borderRadius: 4,
          userSelect: "none",
        }}
      >
        {d.nodeLineName}
      </div>

      <Handle
        type="target"
        position={Position.Top}
        style={{ width: 8, height: 8, background: "#94A3B8", border: "2px solid #fff", top: -5 }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ width: 8, height: 8, background: "#94A3B8", border: "2px solid #fff", bottom: -5 }}
      />
    </div>
  );
}

export const NodeLineGroup = memo(NodeLineGroupInner);
