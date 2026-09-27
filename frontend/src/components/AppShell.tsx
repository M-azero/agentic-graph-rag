import clsx from "clsx";
import {
  GitBranch,
  LogOut,
  MessageSquare,
  Monitor,
  Moon,
  Sun,
  UserCircle,
} from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { Link, NavLink, useNavigate } from "react-router-dom";

import { Button, Wordmark } from "./ui";
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
      className="flex rounded-full bg-raised p-0.5"
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
            "rounded-full p-1.5 transition-colors",
            theme === value
              ? "bg-surface text-accent shadow-card"
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
          "inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium transition-colors",
          isActive ? "bg-accent/15 text-strong" : "text-muted hover:text-body",
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
  const menuRef = useRef<HTMLDivElement>(null);

  // Close on any press outside the menu. A full-screen click-away layer would
  // be trapped inside the glass header (backdrop-filter makes it the containing
  // block for fixed children), so this listens on the document instead.
  useEffect(() => {
    if (!menuOpen) return;
    const onDown = (e: PointerEvent) => {
      if (!menuRef.current?.contains(e.target as Node)) setMenuOpen(false);
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setMenuOpen(false);
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [menuOpen]);

  async function handleSignOut() {
    await signOut();
    navigate("/login", { replace: true });
  }

  return (
    <div className="flex h-full flex-col gap-3 p-3">
      <header className="glass relative z-30 flex h-14 shrink-0 items-center gap-3 rounded-xl px-4">
        <Link to="/chat" aria-label={__APP_NAME__}>
          <Wordmark className="[&>span:last-child]:hidden sm:[&>span:last-child]:inline" />
        </Link>

        {/* No link to the admin console here. It is a separate app at /admin,
            and an operator navigates to it directly — putting an entry point in
            the end-user chrome would advertise the surface to every visitor who
            reads the markup, for the sake of one URL an admin already knows. */}
        <nav className="ml-2 flex items-center gap-1">
          <NavItem to="/chat" icon={<MessageSquare className="h-3.5 w-3.5" />}>
            Chat
          </NavItem>
          <NavItem to="/pipeline" icon={<GitBranch className="h-3.5 w-3.5" />}>
            Pipeline
          </NavItem>
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <ThemeToggle />
          <div ref={menuRef} className="relative">
            <button
              onClick={() => setMenuOpen((v) => !v)}
              className="flex items-center gap-2 rounded-full px-2.5 py-1.5 text-sm text-muted transition-colors hover:bg-raised hover:text-body"
              aria-haspopup="menu"
              aria-expanded={menuOpen}
            >
              <UserCircle className="h-4 w-4" />
              <span className="hidden max-w-[16ch] truncate sm:block">{me?.email}</span>
            </button>

            {menuOpen && (
              <div className="glass-pop absolute right-0 z-50 mt-2 w-52 rounded-lg p-1 animate-slide-up">
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
            )}
          </div>
        </div>
      </header>

      <main className="min-h-0 flex-1">{children}</main>
    </div>
  );
}

export { Button };
