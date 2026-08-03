import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

import { ApiError, session, type Me } from "../api";

interface AuthValue {
  me: Me;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthValue | null>(null);

type State =
  | { status: "loading" }
  | { status: "ready"; me: Me }
  | { status: "forbidden"; me: Me }
  | { status: "error"; message: string };

/**
 * The console's front door.
 *
 * Note what this is and is not. It decides what to *render*; it is not the
 * access control. Every `/api/admin/*` call is gated server-side by
 * `require_admin_user`, which fails closed — so a determined visitor who loads
 * this bundle gets a UI that can fetch nothing. That is why serving the static
 * files at a public path is not a leak.
 *
 * Signing in is the chat app's job: there is exactly one login form, and
 * duplicating it here would mean two places to get password handling wrong.
 * A 401 therefore leaves the SPA entirely, and `?next=` brings the admin back.
 */
export function AuthGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<State>({ status: "loading" });

  useEffect(() => {
    let alive = true;
    session
      .me()
      .then((me) => {
        if (!alive) return;
        setState(me.role === "admin" ? { status: "ready", me } : { status: "forbidden", me });
      })
      .catch((err: unknown) => {
        if (!alive) return;
        if (err instanceof ApiError && err.status === 401) {
          window.location.replace("/login?next=/admin/");
          return;
        }
        setState({
          status: "error",
          message: err instanceof Error ? err.message : "Could not reach the server.",
        });
      });
    return () => {
      alive = false;
    };
  }, []);

  if (state.status === "loading") {
    return (
      <div className="flex h-full items-center justify-center">
        <div className="h-5 w-5 animate-spin rounded-full border-2 border-border border-t-accent" />
      </div>
    );
  }

  if (state.status === "error") {
    return <Notice title="Something went wrong" detail={state.message} />;
  }

  if (state.status === "forbidden") {
    return (
      <Notice
        title="Not authorised"
        detail={`${state.me.email} does not have the admin role on this deployment.`}
      />
    );
  }

  const signOut = async () => {
    try {
      await session.logout();
    } finally {
      window.location.assign("/login");
    }
  };

  return (
    <AuthContext.Provider value={{ me: state.me, signOut }}>{children}</AuthContext.Provider>
  );
}

function Notice({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="flex h-full items-center justify-center p-6">
      <div className="max-w-sm rounded-lg border border-border bg-surface p-6 text-center shadow-card">
        <h1 className="text-lg font-semibold text-strong">{title}</h1>
        <p className="mt-2 text-sm text-muted">{detail}</p>
        <a
          href="/"
          className="mt-5 inline-flex items-center rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-accent-text"
        >
          Back to the app
        </a>
      </div>
    </div>
  );
}

export function useAuth(): AuthValue {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside AuthGate");
  return value;
}
