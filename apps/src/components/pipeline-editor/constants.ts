// from old theme.ts
const ACCENT = "#0F766E";

/** Colors for each node type */
export const NODE_TYPE_COLORS: Record<string, { bg: string; border: string; accent: string }> = {
  query_expansion:    { bg: "#F0F9FF", border: "#BAE6FD", accent: "#0284C7" },
  semantic_retrieval: { bg: "#F0FDF4", border: "#BBF7D0", accent: "#16A34A" },
  lexical_retrieval:  { bg: "#FFFBEB", border: "#FDE68A", accent: "#D97706" },
  hybrid_retrieval:   { bg: "#FFF7ED", border: "#FED7AA", accent: "#EA580C" },
  passage_augmenter:  { bg: "#FAF5FF", border: "#E9D5FF", accent: "#9333EA" },
  passage_reranker:   { bg: "#FFF1F2", border: "#FECDD3", accent: "#E11D48" },
  passage_filter:     { bg: "#F5F3FF", border: "#DDD6FE", accent: "#7C3AED" },
  passage_compressor: { bg: "#ECFDF5", border: "#A7F3D0", accent: "#0F766E" },
  prompt_maker:       { bg: "#FEF9C3", border: "#FDE047", accent: "#CA8A04" },
  generator:          { bg: "#F0FDFA", border: "#99F6E4", accent: ACCENT },
};

export const VECTORDB_COLOR = { bg: "#F8FAFC", border: "#CBD5E1", accent: "#475569" };

/** Layout constants */
export const NODE_WIDTH = 260;
export const NODE_HEIGHT_BASE = 80;
export const NODE_HEIGHT_PER_MODULE = 28;
export const ROW_GAP = 180;
export const COL_GAP = 320;
export const VECTORDB_Y = 20;
export const PIPELINE_Y_START = 250;

/** Default node type for new nodes */
export const DEFAULT_NODE_TYPE = "semantic_retrieval";
