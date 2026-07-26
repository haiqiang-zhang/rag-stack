import assert from "node:assert/strict";
import test from "node:test";
import { loadTs } from "../support/load-ts.mjs";

const api = loadTs("src/api/base.ts", {
  "@/lib/config": { getApiUrl: () => "http://backend", getAuthToken: () => "" },
});

for (const name of ["apiFetch", "apiPost", "apiPut", "apiDelete"]) {
  test(`${name} preserves the backend's CM blocker`, async (t) => {
    t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify({
      detail: { code: "unsupported_component", message: "BM25 has no CM pricing policy." },
    }), { status: 422, statusText: "Unprocessable Entity" }));
    await assert.rejects(api[name]("/cost-model", {}), /422: BM25 has no CM pricing policy/);
  });
}

test("validation errors retain the invalid field and reason", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response(JSON.stringify({
    detail: [{ loc: ["body", "project_dir"], msg: "Field required" }],
  }), { status: 422 }));
  await assert.rejects(api.apiPost("/calibration/run", {}), /body.project_dir: Field required/);
});

test("non-JSON proxy errors fall back to HTTP status", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response("Gateway unavailable", {
    status: 502, statusText: "Bad Gateway",
  }));
  await assert.rejects(api.apiFetch("/catalog"), /502: Bad Gateway/);
});

test("auth failures retain their dedicated error type", async (t) => {
  t.mock.method(globalThis, "fetch", async () => new Response(null, { status: 401 }));
  await assert.rejects(api.apiFetch("/catalog"), api.AuthError);
});
