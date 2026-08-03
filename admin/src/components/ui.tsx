// The console's primitives. Leaner than the chat app's set and shaped for
// tables and forms rather than conversation — no Meter, no prose styling.

import clsx from "clsx";
import { X } from "lucide-react";
import {
  useEffect,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
} from "react";

type Variant = "primary" | "secondary" | "ghost" | "danger";

const VARIANTS: Record<Variant, string> = {
  primary: "bg-accent text-accent-text hover:opacity-90",
  secondary: "border border-border bg-surface text-body hover:bg-raised",
  ghost: "text-muted hover:bg-raised hover:text-body",
  // Destructive actions are outlined rather than filled: a solid red button is
  // the easiest thing on the page to hit by accident, and everything wearing
  // this class deletes or revokes something.
  danger: "border border-danger/40 text-danger hover:bg-danger/10",
};

export function Button({
  variant = "secondary",
  loading = false,
  className,
  children,
  disabled,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; loading?: boolean }) {
  return (
    <button
      {...props}
      // Disabled while in flight: every button here fires a mutation, and a
      // double-click on "delete user" is a request you cannot take back.
      disabled={disabled || loading}
      className={clsx(
        "inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium",
        "transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        VARIANTS[variant],
        className,
      )}
    >
      {loading && <Spinner className="h-3.5 w-3.5" />}
      {children}
    </button>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <div
      className={clsx(
        "h-4 w-4 animate-spin rounded-full border-2 border-border border-t-accent",
        className,
      )}
    />
  );
}

const FIELD =
  "w-full rounded-md border border-border bg-surface px-2.5 py-1.5 text-sm text-strong " +
  "placeholder:text-muted disabled:opacity-50";

export function Input({ className, ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return <input {...props} className={clsx(FIELD, className)} />;
}

export function Select({
  className,
  children,
  ...props
}: SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select {...props} className={clsx(FIELD, className)}>
      {children}
    </select>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block space-y-1">
      <span className="text-xs font-medium text-body">{label}</span>
      {children}
      {hint && <span className="block text-2xs text-muted">{hint}</span>}
    </label>
  );
}

export function Card({
  title,
  actions,
  children,
  className,
}: {
  title?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={clsx(
        "rounded-lg border border-border bg-surface shadow-card",
        className,
      )}
    >
      {(title || actions) && (
        <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-2.5">
          {title && <h2 className="text-sm font-semibold text-strong">{title}</h2>}
          {actions}
        </header>
      )}
      <div className="p-4">{children}</div>
    </section>
  );
}

type Tone = "neutral" | "positive" | "caution" | "danger" | "accent";

const TONES: Record<Tone, string> = {
  neutral: "bg-raised text-muted",
  positive: "bg-positive/10 text-positive",
  caution: "bg-caution/10 text-caution",
  danger: "bg-danger/10 text-danger",
  accent: "bg-accent/10 text-accent",
};

export function Badge({ tone = "neutral", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center rounded px-1.5 py-0.5 text-2xs font-medium",
        TONES[tone],
      )}
    >
      {children}
    </span>
  );
}

export function Alert({ tone = "danger", children }: { tone?: Tone; children: ReactNode }) {
  return (
    <div
      className={clsx(
        "rounded-md px-3 py-2 text-sm",
        tone === "danger" && "bg-danger/10 text-danger",
        tone === "positive" && "bg-positive/10 text-positive",
        tone === "caution" && "bg-caution/10 text-caution",
        (tone === "neutral" || tone === "accent") && "bg-raised text-body",
      )}
      role="status"
    >
      {children}
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail?: string }) {
  return (
    <div className="py-12 text-center">
      <p className="text-sm font-medium text-body">{title}</p>
      {detail && <p className="mt-1 text-xs text-muted">{detail}</p>}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={clsx("animate-pulse rounded bg-raised", className)} />;
}

export function Modal({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  // Escape closes it. A modal that can only be dismissed by its own buttons
  // strands anyone who opened it by accident.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="relative z-10 w-full max-w-md rounded-lg border border-border bg-surface shadow-pop animate-slide-up"
      >
        <header className="flex items-center justify-between border-b border-border px-4 py-2.5">
          <h2 className="text-sm font-semibold text-strong">{title}</h2>
          <button
            onClick={onClose}
            aria-label="Close"
            className="rounded p-1 text-muted hover:bg-raised hover:text-body"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </header>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
}

/** A timestamp rendered as "3 minutes ago", falling back to a date past a week. */
export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return "never";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "—";
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 60) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(iso).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** 1234567 -> "1.2M". Keeps token counts from dominating a table column. */
export function compactNumber(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Math.abs(value) < 1000) return String(value);
  return new Intl.NumberFormat(undefined, {
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}
