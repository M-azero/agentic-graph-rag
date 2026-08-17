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
  define: { __APP_NAME__: JSON.stringify(APP_NAME) },
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
