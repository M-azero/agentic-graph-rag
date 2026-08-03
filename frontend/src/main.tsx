import React, { lazy, Suspense } from "react";
import ReactDOM from "react-dom/client";
import { createBrowserRouter, Navigate, RouterProvider } from "react-router-dom";

import { AppShell } from "./components/AppShell";
import "./index.css";
import { AuthProvider, RequireAuth } from "./lib/auth";
import { ThemeProvider } from "./lib/theme";
import ForgotPassword from "./routes/ForgotPassword";
import Login from "./routes/Login";
import ResetPassword from "./routes/ResetPassword";
import Signup from "./routes/Signup";
import Verify from "./routes/Verify";

// Chat carries react-markdown and highlight.js — around two thirds of the
// bundle, and none of it is needed to render a sign-in form. Split so the
// first page an unauthenticated visitor sees is not the heaviest one.
const Chat = lazy(() => import("./routes/Chat"));
const Account = lazy(() => import("./routes/Account"));

function Loading() {
  return (
    <div className="flex h-full items-center justify-center">
      <div className="h-5 w-5 animate-spin rounded-full border-2 border-border border-t-accent" />
    </div>
  );
}

/** Wraps a page in the signed-in shell. */
const app = (element: React.ReactNode) => (
  <RequireAuth>
    <AppShell>
      <Suspense fallback={<Loading />}>{element}</Suspense>
    </AppShell>
  </RequireAuth>
);

// /admin is deliberately absent. The console is a separate app served from
// /admin by the proxy, so navigating there is a full page load, not a route —
// which is what keeps its bundle, its dependencies and its route names out of
// this one entirely.
const router = createBrowserRouter([
  { path: "/login", element: <Login /> },
  { path: "/signup", element: <Signup /> },
  { path: "/verify", element: <Verify /> },
  { path: "/forgot-password", element: <ForgotPassword /> },
  { path: "/reset-password", element: <ResetPassword /> },

  { path: "/", element: <Navigate to="/chat" replace /> },
  { path: "/chat", element: app(<Chat />) },
  // The same component for both: the thread id is a route param, so opening a
  // conversation is a navigation rather than a state change — which makes the
  // back button and a shared link both work.
  { path: "/chat/:threadId", element: app(<Chat />) },
  { path: "/account", element: app(<Account />) },

  { path: "*", element: <Navigate to="/chat" replace /> },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ThemeProvider>
      <AuthProvider>
        <RouterProvider router={router} />
      </AuthProvider>
    </ThemeProvider>
  </React.StrictMode>,
);
