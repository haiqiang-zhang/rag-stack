/** Dedicated forms speak the selected adapter's inputs. Unknown models have no fallback. */
const COMPONENT_ESTIMATE_FORMS = new Set([
  "faiss_ivf",
  "llm_sim",
  "query_expansion_sim",
  "passage_compressor_sim",
  "passage_compressor_token_pruning_sim",
  "passage_reranker_sim",
]);

export function hasComponentEstimateForm(name: string): boolean {
  return COMPONENT_ESTIMATE_FORMS.has(name);
}
