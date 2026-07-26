import assert from "node:assert/strict";
import test from "node:test";
import { createElement, Fragment } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { loadTs } from "../support/load-ts.mjs";

const utils = loadTs("src/lib/utils.ts");
const tabs = loadTs("src/components/ui/tabs.tsx", { "@/lib/utils": utils });
const passthrough = ({ children }) => createElement(Fragment, null, children);
const marker = (name) => ({ status }) => createElement("div", { "data-component": name, "data-status": status });
const ui = new Proxy({}, {
  get: (_, name) => name === "Dialog" ? ({ open, children }) => open ? children : null : passthrough,
});

function renderScreen({ count = 0, status = "created", tab, knownProject = true, selected = true } = {}) {
  const dependencies = {
    "@/lib/utils": utils,
    "@/lib/systemTransfer": loadTs("src/lib/systemTransfer.ts"),
    "@/components/rag-optimize/projectView": loadTs("src/components/rag-optimize/projectView.ts"),
    "@tanstack/react-router": { useSearch: () => ({ tab }), useNavigate: () => () => {} },
    "@/context/ProjectContext": {
      useProject: () => ({
        currentProject: selected ? "my-project" : null,
        projects: knownProject ? [{ name: "my-project", status, n_evaluations: count }] : [],
        setCurrentProject() {}, refreshProjects() {}, setSortKey() {}, setSortDirection() {},
      }),
    },
    "@/context/SelectionContext": { useSelection: () => ({ setSelectedEvalId() {} }) },
    "@/context/PostAnalysisContext": { PostAnalysisProvider: passthrough },
    "@/context/PerformancePolicyContext": { PerformancePolicyProvider: passthrough },
    "@/context/EvalDataContext": { EvalDataProvider: passthrough },
    "@/context/SweepCacheBridge": { SweepCacheBridge: () => null },
    "@/api/ragProject": {},
    "@/api/qualityStore": {},
    "@/components/ui/tabs": tabs,
  };
  for (const module of ["badge", "button", "ConfirmDialog", "dropdown-menu", "dialog", "input", "label", "PageHeader", "select", "StatusPill"]) {
    dependencies[`@/components/ui/${module}`] = ui;
  }
  // Keep the actual screen, navigation model, and Radix tabs; isolate network
  // providers, editors, and charts so these tests exercise view composition.
  for (const name of ["ProjectList", "ProjectSetupTab", "ProjectConfigContent", "EvalNavigator", "OverviewTab", "TradeoffTab", "PerformanceTab", "QualityTab", "ConfigTab", "HistoryTab", "CMPreviewTab", "PostAnalysisPanel", "SystemYamlPicker"]) {
    dependencies[`@/components/rag-optimize/${name}`] = { [name]: marker(name) };
  }
  const { default: Screen } = loadTs("src/screens/RagOptimizeScreen.tsx", dependencies);
  return renderToStaticMarkup(createElement(Screen));
}

function activeTab(html) {
  return html.match(/<button\b[^>]*role="tab"[^>]*aria-selected="true"[^>]*>([^<]*)<\/button>/)?.[1];
}

test("a newly created project mounts setup and run controls, without result dashboards or navigator", () => {
  const html = renderScreen();
  assert.equal(activeTab(html), "Start eval");
  assert.match(html, /data-component="ProjectSetupTab" data-status="created"/);
  assert.match(html, /data-component="ProjectConfigContent"/);
  assert.doesNotMatch(html, /data-component="(?:EvalNavigator|OverviewTab|ConfigTab)"/);
  assert.doesNotMatch(html, /Loading project|animate-spin/);
});

test("an old overview bookmark on an empty project opens setup", () => {
  const html = renderScreen({ tab: "overview" });
  assert.equal(activeTab(html), "Start eval");
  assert.match(html, /data-component="ProjectSetupTab"/);
  assert.doesNotMatch(html, /data-component="OverviewTab"/);
});

test("projects with evaluations retain their overview and evaluation navigator", () => {
  const html = renderScreen({ count: 3, status: "completed" });
  assert.equal(activeTab(html), "Overview");
  assert.match(html, /data-component="OverviewTab"/);
  assert.match(html, /data-component="EvalNavigator"/);
  assert.doesNotMatch(html, /data-component="ProjectSetupTab"/);
});

test("a saved start tab becomes overview once evaluations exist", () => {
  const html = renderScreen({ count: 1, status: "running", tab: "start" });
  assert.equal(activeTab(html), "Overview");
  assert.match(html, /data-component="OverviewTab"/);
});

test("CM preview deep links remain selected before the first evaluation", () => {
  const html = renderScreen({ tab: "cm-preview" });
  assert.equal(activeTab(html), "CM Preview");
  assert.match(html, /data-component="CMPreviewTab"/);
  assert.doesNotMatch(html, /data-component="(?:EvalNavigator|OverviewTab)"/);
});

test("a running project with zero results passes the running state to setup", () => {
  const html = renderScreen({ status: "running" });
  assert.equal(activeTab(html), "Start eval");
  assert.match(html, /data-component="ProjectSetupTab" data-status="running"/);
  assert.doesNotMatch(html, /data-component="OverviewTab"/);
});

test("an unknown project waits for its status instead of treating missing data as zero evaluations", () => {
  const html = renderScreen({ knownProject: false });
  assert.match(html, /Loading project/);
  assert.doesNotMatch(html, /data-component="(?:ProjectSetupTab|OverviewTab|EvalNavigator)"/);
});

test("no selected project still shows the project list", () => {
  const html = renderScreen({ selected: false });
  assert.match(html, /data-component="ProjectList"/);
  assert.doesNotMatch(html, /data-component="(?:ProjectSetupTab|OverviewTab|EvalNavigator)"/);
});
