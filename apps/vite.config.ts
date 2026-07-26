import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { TanStackRouterVite } from "@tanstack/router-plugin/vite";
import { VitePWA } from "vite-plugin-pwa";
import path from "node:path";

const enableDevPwa = process.env.VITE_PWA_DEV === "1";

// File-based routing under src/routes — TanStack Router scans the dir and
// generates `src/routeTree.gen.ts` automatically. Mirrors expo-router DX.
export default defineConfig({
  plugins: [
    TanStackRouterVite({
      routesDirectory: "src/routes",
      generatedRouteTree: "src/routeTree.gen.ts",
    }),
    react(),
    tailwindcss(),
    VitePWA({
      // Auto-update strategy: when the SW finds a new build, swap it in
      // immediately. We never serve cached API responses (only assets), so
      // the user always sees fresh data.
      registerType: "autoUpdate",
      injectRegister: false,
      includeAssets: ["favicon.svg", "icon-512.svg"],
      manifest: {
        name: "RAG-Stack",
        short_name: "RAG-Stack",
        description: "RAG-Stack optimization dashboard",
        start_url: "/",
        display: "standalone",
        // Match the shadcn neutral palette (near-black primary, white bg).
        background_color: "#ffffff",
        theme_color: "#000000",
        icons: [
          {
            src: "/icon-512.svg",
            sizes: "512x512",
            type: "image/svg+xml",
            purpose: "any maskable",
          },
        ],
      },
      workbox: {
        // Pre-cache app shell + static assets (JS/CSS/SVG/woff2/etc).
        globPatterns: ["**/*.{js,css,html,svg,png,woff2}"],
        // The dashboard bundle is currently larger than Workbox's 2 MiB
        // default. Keep the app shell installable while runtime API data stays
        // uncached by the denylist below.
        maximumFileSizeToCacheInBytes: 5 * 1024 * 1024,
        // NEVER cache API calls — runtime data must always be fresh.
        // (ConnectionContext talks to the backend over /api/*; no SW.)
        navigateFallbackDenylist: [/^\/api\//],
        cleanupOutdatedCaches: true,
      },
      devOptions: {
        // Dev SW is opt-in because it can cache over Vite HMR during UI work.
        // Use `VITE_PWA_DEV=1 npm run dev` when validating installability.
        enabled: enableDevPwa,
        type: "module",
      },
    }),
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    // Pinned to 2333 (project convention) but overridable via APP_PORT env
    // or Vite's `--port <n>` CLI flag (CLI wins). `strictPort: true` makes
    // Vite error out instead of silently falling back to the next free port,
    // so the URL stays predictable.
    port: Number(process.env.APP_PORT) || 2333,
    strictPort: true,
    // Proxy /api to the rag-stack backend (rag_stack/server runs on 8000 by default,
    // started via `uvicorn rag_stack.server.cost_model_api:app --port 8000 --reload --reload-dir rag_stack`).
    // Override via APP_API_URL env at dev time.
    proxy: {
      "/api": {
        target: process.env.APP_API_URL || "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
