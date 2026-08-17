import clsx from "clsx";
import {
  LogOut,
  MessageSquare,
  Monitor,
  Moon,
  Network,
  Sun,
  UserCircle,
} from "lucide-react";
import { useState, type ReactNode } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";

import { Button } from "./ui";
import { useAuth } from "../lib/auth";
import { useTheme } from "../lib/theme";

const THEMES = [
  { value: "light", icon: Sun, label: "Light" },
  { value: "dark", icon: Moon, label: "Dark" },
  { value: "system", icon: Monitor, label: "System" },
] as const;

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  return (
    <div
      className="flex rounded-md bg-raised p-0.5"
      role="radiogroup"
      aria-label="Color theme"
    >
      {THEMES.map(({ value, icon: Icon, label }) => (
        <button
          key={value}
          role="radio"
          aria-checked={theme === value}
          aria-label={label}
          title={label}
          onClick={() => setTheme(value)}
          className={clsx(
            "rounded p-1.5 transition-colors",
            theme === value
              ? "bg-surface text-strong shadow-card"
              : "text-muted hover:text-body",
          )}
        >
          <Icon className="h-3.5 w-3.5" />
        </button>
      ))}
    </div>
  );
}

function NavItem({ to, icon, children }: { to: string; icon: ReactNode; children: ReactNode }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        clsx(
          "inline-flex items-center gap-2 rounded-md px-2.5 py-1.5 text-sm font-medium transition-colors",
          isActive ? "bg-raised text-strong" : "text-muted hover:text-body",
        )
      }
    >
      {icon}
      {children}
    </NavLink>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const { me, signOut } = useAuth();
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);

  async function handleSignOut() {
    await signOut();
    navigate("/login", { replace: true });
  }

  return (
    <div className="flex h-full flex-col bg-canvas">
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-border bg-surface px-4">
        <Link to="/chat" className="flex items-center gap-2.5">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent text-accent-text">
            <Network className="h-3.5 w-3.5" />
          </span>
          <span className="hidden text-base font-semibold tracking-tight text-strong sm:block">
            {__APP_NAME__}
          </span>
        </Link>

        {/* No link to the admin console here. It is a separate app at /admin,
            and an operator navigates to it directly — putting an entry point in
            the end-user chrome would advertise the surface to every visitor who
            reads the markup, for the sake of one URL an admin already knows. */}
        <nav className="ml-2 flex items-center gap-1">
          <NavItem to="/chat" icon={<MessageSquare className="h-3.5 w-3.5" />}>
            Chat
          </NavItem>
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <ThemeToggle />
          <div className="relative">
            <button
              onClick={() => setMenuOpen((v) => !v)}
              className="flex items-center gap-2 rounded-md px-2 py-1.5 text-sm text-muted transition-colors hover:bg-raised hover:text-body"
              aria-haspopup="menu"
              aria-expanded={menuOpen}
            >
              <UserCircle className="h-4 w-4" />
              <span className="hidden max-w-[16ch] truncate sm:block">{me?.email}</span>
            </button>

            {menuOpen && (
              <>
                {/* Click-away layer: a menu that only closes via its own items
                    strands the user if they change their mind. */}
                <div className="fixed inset-0 z-40" onClick={() => setMenuOpen(false)} />
                <div className="absolute right-0 z-50 mt-1 w-48 rounded-lg bg-surface p-1 border border-border shadow-pop animate-slide-up">
                  <Link
                    to="/account"
                    onClick={() => setMenuOpen(false)}
                    className="flex items-center gap-2 rounded-md px-2.5 py-2 text-sm text-body hover:bg-raised"
                  >
                    <UserCircle className="h-3.5 w-3.5" />
                    Account &amp; usage
                  </Link>
                  <button
                    onClick={handleSignOut}
                    className="flex w-full items-center gap-2 rounded-md px-2.5 py-2 text-left text-sm text-body hover:bg-raised"
                  >
                    <LogOut className="h-3.5 w-3.5" />
                    Sign out
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      </header>

      <main className="min-h-0 flex-1">{children}</main>
    </div>
  );
}

export { Button };
