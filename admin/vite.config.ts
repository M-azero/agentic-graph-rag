import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The console is served at /admin, not at the root, so `base` has to say so:
// without it every hashed asset URL in the built index.html points at /assets/…
// and resolves against the chat app's root, which serves the wrong bundle.
//
// Port 5174 so `npm run dev` here and in ../frontend can run side by side.
// Same product name the chat app uses — see frontend/vite.config.ts for why it
// comes from the environment rather than tracked source.
const APP_NAME = process.env.VITE_APP_NAME || "Graph RAG";

export default defineConfig({
  base: "/admin/",
  plugins: [
    react(),
    {
      name: "app-name-html",
      transformIndexHtml: (html) => html.replace(/%APP_NAME%/g, APP_NAME),
    },
  ],
  define: { __APP_NAME__: JSON.stringify(APP_NAME) },
  server: {
    port: 5174,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
