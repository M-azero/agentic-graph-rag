import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The console is served at /admin, not at the root, so `base` has to say so:
// without it every hashed asset URL in the built index.html points at /assets/…
// and resolves against the chat app's root, which serves the wrong bundle.
//
// Port 5174 so `npm run dev` here and in ../frontend can run side by side.
export default defineConfig({
  base: "/admin/",
  plugins: [react()],
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
