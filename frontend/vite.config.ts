import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The product name shown in the header, on the auth screens, and in the browser
// tab. Set APP_NAME in the deployment's .env; compose passes it to the proxy
// image as a build arg, which exports it here as VITE_APP_NAME.
//
// The default is deliberately generic. This repo is public and a fork should
// not ship branded as somebody else's deployment, so the name lives in the
// (gitignored) .env rather than in tracked source.
const APP_NAME = process.env.VITE_APP_NAME || "Graph RAG";

// The landing page's links, from CONTACT_EMAIL / SOURCE_URL in .env for the same
// reason: a deployment's contact address does not belong in a public repo.
// No address hides the contact link; no source URL points at upstream.
const CONTACT_EMAIL = process.env.VITE_CONTACT_EMAIL || "";
const SOURCE_URL =
  process.env.VITE_SOURCE_URL || "https://github.com/M-azero/agentic-graph-rag";

// In dev, proxy /api -> the local backend and strip the /api prefix.
export default defineConfig({
  plugins: [
    react(),
    {
      // index.html is a static file, so the name is substituted at build time.
      // Setting document.title from JS instead would flash the placeholder on
      // first paint, and would leave the title empty for anything that reads
      // the markup without executing it.
      name: "app-name-html",
      transformIndexHtml: (html) => html.replace(/%APP_NAME%/g, APP_NAME),
    },
  ],
  // Inlined at build time so the components have no runtime lookup and no
  // undefined case. Declared in src/vite-env.d.ts.
  define: {
    __APP_NAME__: JSON.stringify(APP_NAME),
    __CONTACT_EMAIL__: JSON.stringify(CONTACT_EMAIL),
    __SOURCE_URL__: JSON.stringify(SOURCE_URL),
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
