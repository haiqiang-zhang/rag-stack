import type {
  CMParameter,
  CMPreviewRequest,
  CMOperation,
  CMTraceCall,
  CMRagIR,
  PreviewStatus,
} from "@/api/cmPreview";

export function isPreviewActive(status?: PreviewStatus): boolean {
  return status === "queued" || status === "running";
}

export function parseObject(
  text: string,
  label: string,
): Record<string, unknown> {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error(`${label} must be valid JSON.`);
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${label} must be a JSON object.`);
  }
  return value as Record<string, unknown>;
}

/** JSON preserves string IDs (including leading zeroes) and numeric IDs. */
export function parseQuestionIds(text: string): Array<string | number> {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error(
      'Question IDs must be a JSON array, for example ["q1", "q2"].',
    );
  }
  if (
    !Array.isArray(value) ||
    value.length === 0 ||
    value.some(
      (id) =>
        !(typeof id === "string" && id.length > 0) &&
        !(typeof id === "number" && Number.isSafeInteger(id)),
    )
  ) {
    throw new Error(
      "Enter at least one question ID. IDs must be nonempty strings or safe integers.",
    );
  }
  if (value.length > 128) throw new Error("Select at most 128 question IDs.");
  if (new Set(value.map(String)).size !== value.length) {
    throw new Error("Question IDs must be unique.");
  }
  return value;
}

export interface PreviewForm {
  source: "evaluation" | "parameters";
  evalId: number | null;
  parameters: string;
  queryMode: "sample" | "ids";
  sampleCount: string;
  questionIds: string;
  reuseTrace: boolean;
  skipJudge: boolean;
  qaData: string;
  corpusData: string;
  datasetName: string;
  systemConfig: string;
  systemRequired: boolean;
  parameterSchema?: CMParameter[];
}

export function buildPreviewRequest(form: PreviewForm): CMPreviewRequest {
  const request: CMPreviewRequest = {
    reuse_trace: form.reuseTrace,
    skip_judge: form.skipJudge,
  };
  if (form.source === "evaluation") {
    if (
      form.evalId === null ||
      !Number.isSafeInteger(form.evalId) ||
      form.evalId < 0
    ) {
      throw new Error("Select an evaluation to use its resolved pipeline.");
    }
    request.eval_id = form.evalId;
  } else {
    const selected = parseObject(form.parameters, "Point parameters");
    if (form.parameterSchema) {
      for (const parameter of activeParameters(
        form.parameterSchema,
        selected,
      )) {
        if (parameter.type === "fixed") {
          if (
            Object.hasOwn(selected, parameter.name) &&
            JSON.stringify(selected[parameter.name]) !==
              JSON.stringify(parameter.value)
          ) {
            throw new Error(
              `${parameter.name} is fixed at ${JSON.stringify(parameter.value)}.`,
            );
          }
          selected[parameter.name] = parameter.value;
        } else if (!Object.hasOwn(selected, parameter.name)) {
          throw new Error(`Choose a value for ${parameter.name}.`);
        }
      }
    }
    request.parameters = selected;
    if (form.reuseTrace)
      throw new Error("Trace reuse requires an existing evaluation.");
  }
  if (form.queryMode === "ids") {
    request.question_ids = parseQuestionIds(form.questionIds);
  } else {
    const count = Number(form.sampleCount);
    if (
      !form.sampleCount.trim() ||
      !Number.isSafeInteger(count) ||
      count < 1 ||
      count > 128
    ) {
      throw new Error("Sample count must be an integer between 1 and 128.");
    }
    request.sample_count = count;
  }
  if (form.qaData.trim()) request.qa_data = form.qaData.trim();
  if (form.corpusData.trim()) request.corpus_data = form.corpusData.trim();
  if (form.reuseTrace && (request.qa_data || request.corpus_data)) {
    throw new Error(
      "A reused trace keeps its original dataset. Clear QA/corpus overrides or run a fresh evaluation.",
    );
  }
  if (form.datasetName.trim()) request.dataset_name = form.datasetName.trim();
  if (form.systemConfig.trim())
    request.system_config = parseObject(form.systemConfig, "System config");
  else if (form.source === "parameters" && form.systemRequired) {
    throw new Error(
      "This project needs a concrete system config for CM compilation.",
    );
  }
  return request;
}

export function jsonText(value: unknown): string {
  return value === undefined ? "Not recorded" : JSON.stringify(value, null, 2);
}

/** Select only active Ax dimensions; conditional children follow the chosen value. */
export function activeParameters(
  parameters: CMParameter[],
  values: Record<string, unknown>,
): CMParameter[] {
  const children = new Set(
    parameters.flatMap((p) => Object.values(p.dependents ?? {}).flat()),
  );
  const byName = new Map(parameters.map((p) => [p.name, p]));
  const active = new Set<string>();
  function visit(parameter: CMParameter) {
    if (active.has(parameter.name)) return;
    active.add(parameter.name);
    const chosen = values[parameter.name] ?? parameter.value;
    const dependentNames =
      parameter.dependents?.[String(chosen)] ??
      (typeof chosen === "boolean"
        ? parameter.dependents?.[chosen ? "True" : "False"]
        : undefined) ??
      [];
    for (const name of dependentNames) {
      const child = byName.get(name);
      if (child) visit(child);
    }
  }
  parameters.filter((p) => !children.has(p.name)).forEach(visit);
  return parameters.filter((p) => active.has(p.name));
}

/** Compiler indices address raw calls by array position, independent of step_idx. */
export function operationSources(
  operation: CMOperation,
  calls: CMTraceCall[],
): Array<{ call: CMTraceCall; index: number }> {
  const ids = new Set(operation.source_call_ids);
  const indices = new Set(operation.source_call_indices);
  return calls
    .map((call, index) => ({ call, index }))
    .filter(
      ({ call, index }) =>
        (call.call_id !== undefined && ids.has(call.call_id)) ||
        indices.has(index),
    );
}

/** Preview compilation preserves cohort order, including repeated source IDs. */
export function rawQueryForCompiledQuery(
  queries: NonNullable<CMRagIR["queries"]>,
  questionId: string | number,
  index: number,
) {
  const sourceId = (query: (typeof queries)[number]) =>
    query.source_question_id ?? query.question_id;
  if (queries[index] && sourceId(queries[index]) === questionId)
    return queries[index];
  const matches = queries.filter((query) => sourceId(query) === questionId);
  return matches.length === 1 ? matches[0] : undefined;
}
