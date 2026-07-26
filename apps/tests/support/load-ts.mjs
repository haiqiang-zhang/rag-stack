import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import ts from "typescript";

// Execute the actual TypeScript modules with Node's built-in test runner.
// Paths are relative to apps/, independent of the invoking working directory.
export function loadTs(relativePath, dependencies = {}) {
  const file = fileURLToPath(new URL(`../../${relativePath}`, import.meta.url));
  const code = ts.transpileModule(readFileSync(file, "utf8"), {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      jsx: ts.JsxEmit.ReactJSX,
      target: ts.ScriptTarget.ES2022,
    },
  }).outputText;
  const exports = {};
  const run = vm.runInThisContext(`(function(require, exports) {${code}\n})`, { filename: file });
  const require = createRequire(file);
  run((name) => Object.hasOwn(dependencies, name) ? dependencies[name] : require(name), exports);
  return exports;
}
