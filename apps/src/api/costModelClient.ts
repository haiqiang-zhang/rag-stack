/**
 * Barrel re-export for backwards compatibility.
 *
 * All API functions and types have been split into feature-based modules.
 * Import from the individual modules for new code, or continue importing
 * from this file — both work identically.
 */

export * from "./base";
export * from "./ragProject";
export * from "./qualityStore";
export * from "./costModel"; // canonical `killJob` (calibration job)
export * from "./dataset";
export * from "./registry";

// `optimizer.ts` also has a `killJob` for benchmark jobs — explicit
// enumeration (with the optimizer one aliased) avoids `export *` collision.
export {
  fetchOptimizerRegistry,
  fetchOptimizerBenchmarkResults,
  fetchOptimizerBenchmarkSummary,
  fetchOptimizerConfigs,
  saveOptimizerConfig,
  launchBenchmarkRun,
  fetchBenchmarkJobs,
  killJob as killOptimizerJob,
} from "./optimizer";
