import assert from "node:assert/strict";
import test from "node:test";
import { loadTs } from "../support/load-ts.mjs";

const api = loadTs("src/api/cmPreview.ts", {
  "@/lib/config": {
    getApiUrl: () => "https://backend.test",
    getAuthToken: () => "secret",
  },
});

test("preview API uses project-scoped routes, explicit payloads and bearer headers", async (context) => {
  const calls = [];
  context.mock.method(globalThis, "fetch", async (url, options) => {
    calls.push({ url, options });
    return new Response(JSON.stringify({ preview_id: "p1" }));
  });
  const request = {
    eval_id: 2,
    sample_count: 8,
    skip_judge: true,
    reuse_trace: false,
  };
  await api.createCMPreview("project / 1", request);
  await api.fetchCMPreviewOptions("project / 1");
  await api.fetchCMPreview("project / 1", "p1");
  await api.cancelCMPreview("project / 1", "p1");
  assert.equal(
    calls[0].url,
    "https://backend.test/api/rag-project/projects/project%20%2F%201/cm-previews",
  );
  assert.deepEqual(JSON.parse(calls[0].options.body), request);
  assert.equal(calls[0].options.headers.Authorization, "Bearer secret");
  assert.ok(calls[1].url.endsWith("/cm-previews/options"));
  assert.equal(calls[2].options.method, "GET");
  assert.ok(calls[3].url.endsWith("/p1/cancel"));
  assert.equal(calls[3].options.method, "POST");
});

test("server contract errors remain visible rather than collapsing to statusText", async (context) => {
  context.mock.method(
    globalThis,
    "fetch",
    async () =>
      new Response(
        JSON.stringify({
          detail: {
            code: "unresolved_parameter",
            message: "Choose generator model",
          },
        }),
        { status: 422, statusText: "Unprocessable Entity" },
      ),
  );
  await assert.rejects(
    api.createCMPreview("project", {}),
    /unresolved_parameter.*Choose generator model/,
  );
});

test("download contract permits saved display descriptions and rejects executable plan artifacts", async (context) => {
  const fetch = context.mock.method(globalThis, "fetch", async () => { throw new Error("Unexpected request"); });
  for (const filename of ["compiled_plan.json", "compiled_path.json"]) {
    assert.equal(api.isCMPreviewArtifactDownloadable(filename), false);
    await assert.rejects(api.downloadCMPreviewArtifact("project", "p1", filename), /display description can be exported/);
  }
  for (const filename of ["compiled_description.json", "rag_ir.json", "algo_config.json", "system_config.json", "preview.log", "state.json"]) {
    assert.equal(api.isCMPreviewArtifactDownloadable(filename), true);
  }
  assert.equal(fetch.mock.callCount(), 0);
});

test("reopening and downloading a saved description issue only read requests", async (context) => {
  const requests = [];
  const projection = { schema_version: "rag-cm.compiled-plan.v1", status: "compiled", queries: [] };
  context.mock.method(globalThis, "fetch", async (url, options) => {
    requests.push({ url, options });
    return new Response(JSON.stringify(url.endsWith("compiled_description.json")
      ? projection
      : { status: "succeeded", compiled_path_status: "available", compiled_plan: projection }));
  });
  const originalDocument = Object.getOwnPropertyDescriptor(globalThis, "document");
  const link = { href: "", download: "", click() {}, remove() {} };
  Object.defineProperty(globalThis, "document", { configurable: true, value: {
    createElement: () => link,
    body: { appendChild() {} },
  } });
  context.after(() => originalDocument
    ? Object.defineProperty(globalThis, "document", originalDocument)
    : delete globalThis.document);
  context.mock.method(URL, "createObjectURL", () => "blob:saved-preview");
  context.mock.method(URL, "revokeObjectURL", () => {});
  context.mock.method(globalThis, "setTimeout", (callback) => { callback(); return 0; });

  const reopened = await api.fetchCMPreview("project", "p1");
  await api.downloadCMPreviewArtifact("project", "p1", "compiled_description.json");
  assert.deepEqual(reopened.compiled_plan, projection);
  assert.equal(link.download, "compiled_description.json");
  assert.equal(requests.length, 2);
  assert.ok(requests.every(({ options }) => !options.method || options.method === "GET"));
  assert.ok(requests.every(({ options }) => options.headers.Authorization === "Bearer secret"));
  assert.ok(requests[1].url.endsWith("/p1/artifacts/compiled_description.json"));
});
