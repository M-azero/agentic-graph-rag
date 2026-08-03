import { AlertTriangle } from "lucide-react";
import { Link } from "react-router-dom";

import type { LimitDetail } from "../../api";

const NAMES: Record<string, string> = {
  messages_per_minute: "messages per minute",
  messages_per_day: "messages today",
  tokens_per_day: "tokens today",
  tokens_per_month: "tokens this month",
  max_files: "documents",
  max_storage_mb: "storage",
  max_chunks: "indexed content",
  max_threads: "conversations",
};

/** Rendered when the server refuses a request for quota reasons. It reports
 *  the specific limit and when it resets, because "try again later" leaves
 *  the user guessing at both. */
export function QuotaBanner({ detail }: { detail: LimitDetail }) {
  const name = NAMES[detail.limit] ?? detail.limit.replace(/_/g, " ");
  return (
    <div className="flex items-start gap-2.5 rounded-lg border border-caution/20 bg-caution/10 px-3.5 py-3 text-sm">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-caution" />
      <div className="min-w-0">
        <p className="font-medium text-strong">
          You've used all {detail.max.toLocaleString()} {name}.
        </p>
        <p className="mt-0.5 text-muted">
          {detail.retry_after > 0 ? (
            <>Resets in {formatDuration(detail.retry_after)}. </>
          ) : (
            <>Free some space to continue. </>
          )}
          <Link to="/account" className="font-medium text-accent hover:underline">
            View usage
          </Link>
        </p>
      </div>
    </div>
  );
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)} min`;
  if (seconds < 86400) return `${Math.ceil(seconds / 3600)} hours`;
  return `${Math.ceil(seconds / 86400)} days`;
}
