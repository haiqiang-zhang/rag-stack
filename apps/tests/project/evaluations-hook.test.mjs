import assert from "node:assert/strict";
import test from "node:test";
import { loadTs } from "../support/load-ts.mjs";

// A minimal effect driver lets deferred network responses arrive in arbitrary
// order without adding a browser/test-renderer dependency to the application.
function mountEvaluations() {
  const slots = [];
  let cursor = 0;
  let pending = [];
  let project = "first";
  const requests = [];
  const { useEvaluations } = loadTs("src/hooks/useEvaluations.ts", {
    react: {
      useState(initial) {
        const index = cursor++;
        slots[index] ??= { value: typeof initial === "function" ? initial() : initial };
        return [slots[index].value, (update) => {
          slots[index].value = typeof update === "function" ? update(slots[index].value) : update;
        }];
      },
      useEffect(effect, dependencies) {
        const index = cursor++;
        const slot = slots[index];
        if (!slot || dependencies.some((value, i) => !Object.is(value, slot.dependencies[i]))) {
          pending.push(() => {
            slot?.cleanup?.();
            slots[index] = { dependencies, cleanup: effect() };
          });
        }
      },
    },
    "@/context/ProjectContext": { useProject: () => ({ currentProject: project }) },
    "@/api/ragProject": {
      fetchProjectEvaluations(name) {
        return new Promise((resolve, reject) => requests.push({ name, resolve, reject }));
      },
    },
  });
  return {
    requests,
    setProject(value) { project = value; },
    render(count = 0) { cursor = 0; return useEvaluations(count); },
    effects() { const effects = pending; pending = []; effects.forEach((effect) => effect()); },
    unmount() { slots.forEach((slot) => slot.cleanup?.()); },
  };
}

const row = (id) => ({
  eval_id: id, eval_type: "GT", performance: 2, quality: { combined: 0.8 }, config: {}, gt_metrics: null,
});

test("a completed empty response ends evaluation loading", async () => {
  const hook = mountEvaluations();
  assert.equal(hook.render().loading, true);
  hook.effects();
  hook.requests[0].resolve([]);
  await Promise.resolve();
  const state = hook.render();
  assert.equal(state.loading, false);
  assert.equal(state.evaluations.length, 0);
  assert.equal(state.error, null);
  hook.unmount();
});

test("request errors end loading and remain distinguishable from an empty response", async () => {
  const hook = mountEvaluations();
  hook.render();
  hook.effects();
  hook.requests[0].reject(new Error("unavailable"));
  await Promise.resolve();
  assert.equal(hook.render().loading, false);
  assert.equal(hook.render().error.message, "unavailable");
  hook.unmount();
});

test("switching to a new project immediately hides old rows before its fetch effect", async () => {
  const hook = mountEvaluations();
  hook.render(1);
  hook.effects();
  hook.requests[0].resolve([row(7)]);
  await Promise.resolve();
  assert.equal(hook.render(1).evaluations[0].eval_id, 7);
  hook.setProject("new-project");
  const switched = hook.render(0);
  assert.equal(switched.evaluations.length, 0);
  assert.equal(switched.loading, true);
  hook.effects();
  hook.requests[1].resolve([]);
  await Promise.resolve();
  assert.equal(hook.render().loading, false);
  assert.equal(hook.render().evaluations.length, 0);
  hook.unmount();
});

test("a late response from the previous project cannot replace the new empty project", async () => {
  const hook = mountEvaluations();
  hook.render(1);
  hook.effects();
  hook.setProject("new-project");
  hook.render();
  hook.effects();
  hook.requests[1].resolve([]);
  await Promise.resolve();
  hook.requests[0].resolve([row(9)]);
  await Promise.resolve();
  assert.equal(hook.render().evaluations.length, 0);
  assert.equal(hook.render().loading, false);
  hook.unmount();
});

test("a stale request failure cannot overwrite a newer evaluation response", async () => {
  const hook = mountEvaluations();
  hook.render(1);
  hook.effects();
  hook.render(2);
  hook.effects();
  hook.requests[1].resolve([row(1), row(2)]);
  await Promise.resolve();
  hook.requests[0].reject(new Error("stale failure"));
  await Promise.resolve();
  assert.equal(hook.render(2).evaluations.length, 2);
  assert.equal(hook.render(2).error, null);
  hook.unmount();
});

test("no selected project has neither evaluations nor loading work", async () => {
  const hook = mountEvaluations();
  hook.render();
  hook.effects();
  hook.setProject(null);
  hook.render();
  hook.effects();
  hook.requests[0].resolve([row(3)]);
  await Promise.resolve();
  assert.equal(hook.render().evaluations.length, 0);
  assert.equal(hook.render().loading, false);
});
