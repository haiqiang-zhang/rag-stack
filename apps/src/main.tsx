import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider, createRouter } from "@tanstack/react-router";
import { registerSW } from "virtual:pwa-register";
import { routeTree } from "./routeTree.gen";
import "./index.css";

// Register the production PWA service worker. Dev PWA registration is handled
// by vite-plugin-pwa when VITE_PWA_DEV=1 enables devOptions.
// `autoUpdate` strategy: on new build, refresh the page so users always run
// the latest assets without prompting.
if (import.meta.env.PROD) {
  registerSW({ immediate: true });
}

const router = createRouter({
  routeTree,
  defaultPreload: "intent",
});

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

const rootEl = document.getElementById("root");
if (!rootEl) throw new Error("Missing #root in index.html");

createRoot(rootEl).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
