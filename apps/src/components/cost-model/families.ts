import type { CostModelFamily, CostModelInfo } from "@/types/costModel";

export interface FamilyMeta {
  key: CostModelFamily;
  label: string;
  /** Lucide icon name (key in lucide-react). */
  iconName: "Cpu" | "Database" | "Layers";
  order: number;
}

/** Display metadata and sort order for each cost-model family. */
export const FAMILIES: FamilyMeta[] = [
  { key: "llm_sim",   label: "LLM Simulation", iconName: "Cpu",      order: 0 },
  { key: "retrieval", label: "Retrieval",      iconName: "Database", order: 1 },
  { key: "other",     label: "Other",          iconName: "Layers",   order: 2 },
];

export function familyOf(family: CostModelFamily): FamilyMeta {
  return FAMILIES.find((f) => f.key === family) ?? FAMILIES[FAMILIES.length - 1];
}

/** Group cost models by family and return sections in canonical order. */
export function groupByFamily(models: CostModelInfo[]): Array<FamilyMeta & { items: CostModelInfo[] }> {
  const bucket: Record<string, CostModelInfo[]> = {};
  for (const m of models) {
    const key = m.family ?? "other";
    (bucket[key] = bucket[key] ?? []).push(m);
  }
  for (const list of Object.values(bucket)) {
    list.sort((a, b) => a.name.localeCompare(b.name));
  }
  return FAMILIES
    .filter((f) => bucket[f.key] && bucket[f.key].length > 0)
    .map((f) => ({ ...f, items: bucket[f.key] }));
}
