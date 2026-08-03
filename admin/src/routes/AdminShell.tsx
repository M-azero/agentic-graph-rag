import { Outlet, useLocation } from "react-router-dom";

import { Sidebar } from "../components/Sidebar";

const TITLES: Record<string, string> = {
  "/": "Overview",
  "/users": "Users",
  "/limits": "Limits",
  "/audit": "Audit log",
  "/system": "System",
};

function titleFor(pathname: string): string {
  if (TITLES[pathname]) return TITLES[pathname];
  if (pathname.startsWith("/users/")) return "User";
  return "Admin";
}

/**
 * The console frame: a fixed sidebar and a scrolling work area.
 *
 * Only the main column scrolls. In a tool that is mostly long tables, a page
 * that scrolls as a whole takes the navigation off-screen exactly when someone
 * has read far enough to want to go somewhere else.
 */
export function AdminShell() {
  const { pathname } = useLocation();

  return (
    <div className="flex h-full bg-canvas">
      <Sidebar />
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center border-b border-border px-5">
          <h1 className="text-sm font-semibold text-strong">{titleFor(pathname)}</h1>
        </header>
        <main className="min-h-0 flex-1 overflow-y-auto p-5">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
