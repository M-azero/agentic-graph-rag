import { Network } from "lucide-react";
import type { ReactNode } from "react";

/** The frame around sign-in, sign-up and verification. Deliberately plain:
 *  a single centred card, no marketing panel, no gradient. */
export function AuthLayout({
  title,
  subtitle,
  children,
  footer,
}: {
  title: string;
  subtitle?: string;
  children: ReactNode;
  footer?: ReactNode;
}) {
  return (
    <div className="flex min-h-full items-center justify-center bg-canvas px-4 py-12">
      <div className="w-full max-w-sm">
        <div className="mb-8 flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-accent-text">
            <Network className="h-4 w-4" />
          </span>
          <span className="text-base font-semibold tracking-tight text-strong">
            Graph RAG
          </span>
        </div>

        <h1 className="text-xl font-semibold tracking-tight text-strong">{title}</h1>
        {subtitle && <p className="mt-1.5 text-sm text-muted">{subtitle}</p>}

        <div className="mt-6 space-y-4">{children}</div>

        {footer && <div className="mt-6 text-sm text-muted">{footer}</div>}
      </div>
    </div>
  );
}
