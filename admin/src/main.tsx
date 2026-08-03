import React, { lazy, Suspense } from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom";

import "./index.css";
import { AuthGate } from "./lib/auth";
import { ThemeProvider } from "./lib/theme";
import { AdminShell } from "./routes/AdminShell";
import Users from "./routes/Users";

// Only Users ships in the entry chunk. Everything else is split, and Overview
// in particular has to be: it imports recharts, which is ~400 kB on its own —
// eagerly importing it put the entry chunk at 635 kB and made every page load
// pay for a chart most visits never render.
const Overview = lazy(() => import("./routes/Overview"));
const UserDetail = lazy(() => import("./routes/UserDetail"));
const Limits = lazy(() => import("./routes/Limits"));
const Audit = lazy(() => import("./routes/Audit"));
const System = lazy(() => import("./routes/System"));

function Loading() {
  return (
    <div className="flex h-40 items-center justify-center">
      <div className="h-5 w-5 animate-spin rounded-full border-2 border-border border-t-accent" />
    </div>
  );
}

const page = (element: React.ReactNode) => <Suspense fallback={<Loading />}>{element}</Suspense>;

// basename="/admin": Caddy serves this bundle under that prefix, and without it
// every route the router builds would point at the chat app's root instead.
const router = createBrowserRouter(
  [
    {
      path: "/",
      element: <AdminShell />,
      children: [
        { index: true, element: page(<Overview />) },
        { path: "users", element: <Users /> },
        { path: "users/:userId", element: page(<UserDetail />) },
        { path: "limits", element: page(<Limits />) },
        { path: "audit", element: page(<Audit />) },
        { path: "system", element: page(<System />) },
        { path: "*", element: <Navigate to="/" replace /> },
      ],
    },
  ],
  { basename: "/admin" },
);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <AuthGate>
        <RouterProvider router={router} />
      </AuthGate>
    </ThemeProvider>
  </React.StrictMode>,
);
