import type { ReactNode } from "react";

import { Wordmark } from "../components/ui";

/** The frame around sign-in, sign-up and verification: one glass pane on the
 *  open sky, no marketing panel. The sky is the welcome; the pane is the task. */
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
    <div className="flex min-h-full items-center justify-center px-4 py-12">
      <div className="w-full max-w-sm">
        <Wordmark className="mb-6 justify-center" />

        <div className="glass rounded-2xl p-7">
          <h1 className="font-display text-2xl font-bold tracking-tight text-strong">{title}</h1>
          {subtitle && <p className="mt-2 text-sm text-muted">{subtitle}</p>}

          <div className="mt-6 space-y-4">{children}</div>
        </div>

        {footer && <div className="mt-5 text-center text-sm text-muted">{footer}</div>}
      </div>
    </div>
  );
}
