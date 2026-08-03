import clsx from "clsx";
import {
  Activity,
  ExternalLink,
  LayoutDashboard,
  LogOut,
  Monitor,
  Moon,
  ScrollText,
  Server,
  SlidersHorizontal,
  Sun,
  Users,
} from "lucide-react";
import { NavLink } from "react-router-dom";

import { useAuth } from "../lib/auth";
import { useTheme } from "../lib/theme";

const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard, end: true },
  { to: "/users", label: "Users", icon: Users, end: false },
  { to: "/limits", label: "Limits", icon: SlidersHorizontal, end: false },
  { to: "/audit", label: "Audit", icon: ScrollText, end: false },
  { to: "/system", label: "System", icon: Server, end: false },
];

const THEMES = [
  { value: "light", icon: Sun, label: "Light" },
  { value: "dark", icon: Moon, label: "Dark" },
  { value: "system", icon: Monitor, label: "System" },
] as const;

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  return (
    <div className="flex rounded-md bg-raised p-0.5" role="radiogroup" aria-label="Colour theme">
      {THEMES.map(({ value, icon: Icon, label }) => (
        <button
          key={value}
          role="radio"
          aria-checked={theme === value}
          aria-label={label}
          title={label}
          onClick={() => setTheme(value)}
          className={clsx(
            "flex-1 rounded p-1.5 transition-colors",
            theme === value ? "bg-surface text-strong shadow-card" : "text-muted hover:text-body",
          )}
        >
          <Icon className="mx-auto h-3.5 w-3.5" />
        </button>
      ))}
    </div>
  );
}

export function Sidebar() {
  const { me, signOut } = useAuth();

  return (
    <aside className="flex w-sidebar shrink-0 flex-col border-r border-border bg-surface">
      <div className="flex items-center gap-2 px-4 py-3.5">
        <span className="flex h-6 w-6 items-center justify-center rounded-md bg-accent text-accent-text">
          <Activity className="h-3.5 w-3.5" />
        </span>
        <span className="text-sm font-semibold tracking-tight text-strong">Admin</span>
      </div>

      <nav className="flex-1 space-y-0.5 px-2 py-2">
        {NAV.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              clsx(
                "flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm font-medium transition-colors",
                isActive ? "bg-accent/10 text-accent" : "text-muted hover:bg-raised hover:text-body",
              )
            }
          >
            <Icon className="h-4 w-4" />
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="space-y-2 border-t border-border p-2">
        <ThemeToggle />
        <p className="truncate px-1 text-2xs text-muted" title={me.email}>
          {me.email}
        </p>
        {/* A plain anchor, not a Link: the chat app is a different bundle at a
            different root, so this has to be a full page load. */}
        <a
          href="/"
          className="flex items-center gap-2 rounded-md px-2.5 py-1.5 text-sm text-muted hover:bg-raised hover:text-body"
        >
          <ExternalLink className="h-3.5 w-3.5" />
          Exit to chat
        </a>
        <button
          onClick={signOut}
          className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-sm text-muted hover:bg-raised hover:text-body"
        >
          <LogOut className="h-3.5 w-3.5" />
          Sign out
        </button>
      </div>
    </aside>
  );
}
