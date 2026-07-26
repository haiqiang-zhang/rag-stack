import assert from "node:assert/strict";
import test from "node:test";
import { loadTs } from "../support/load-ts.mjs";

const { parseConfigValue, formatConfigValue } = loadTs("src/components/pipeline-editor/panels/ConfigValueInput.tsx");

test("parameter editors preserve scalar/list/dict types and reject incomplete selectors", () => {
  for (const value of [32, [32, 64], { range: [8, 64], n_grid: 3 }, "dense", ["dense", "dense2"], { ordered: ["dense", "dense2"] }, false, "false", "32", ""]) {
    assert.deepEqual(parseConfigValue(formatConfigValue(value)), value);
  }
  assert.equal(parseConfigValue(""), undefined);
  assert.throws(() => parseConfigValue("[32,"));
  assert.throws(() => parseConfigValue('{"range":'));
});
