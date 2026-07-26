import type { CostModelBinding, CostModelCatalog } from "@/types/costModel";

export interface BindingGroup extends Omit<CostModelBinding, "component"> {
  components: string[];
}

/** Only combine visually identical backend bindings; no client-side dispatch rules. */
export function groupBindings(bindings: CostModelBinding[]): BindingGroup[] {
  const groups = new Map<string, BindingGroup>();
  for (const { component, ...binding } of bindings) {
    const key = JSON.stringify(binding);
    const group = groups.get(key) ?? { ...binding, components: [] };
    group.components.push(component);
    groups.set(key, group);
  }
  return [...groups.values()];
}

export function filterCatalogStages(catalog: CostModelCatalog, search: string) {
  const query = search.trim().toLowerCase();
  if (!query) return catalog.stages;
  return catalog.stages.flatMap((stage) => {
    if (stage.stage.toLowerCase().includes(query)) return [stage];
    const bindings = stage.bindings.filter((binding) => [
      binding.component,
      binding.dispatch_stage,
      binding.db_type,
      binding.cost_model,
      binding.status,
      binding.reason,
      ...binding.performance_stages.map((phase) => phase.name),
    ].some((value) => value?.toLowerCase().includes(query)));
    return bindings.length ? [{ ...stage, bindings }] : [];
  });
}
