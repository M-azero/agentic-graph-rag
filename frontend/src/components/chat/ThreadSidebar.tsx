import clsx from "clsx";
import { MessageSquarePlus, Trash2 } from "lucide-react";
import { useMemo } from "react";

import type { ThreadInfo } from "../../api";
import { Button, Skeleton } from "../ui";

/**
 * Group conversations the way someone looks for them: by when they last said
 * something, not alphabetically and not by creation date. A flat list of forty
 * titles is a list you scan; four short lists are ones you scan the right part
 * of.
 */
function group(threads: ThreadInfo[]): { label: string; items: ThreadInfo[] }[] {
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startOfYesterday = startOfToday - 86_400_000;
  const startOfWeek = startOfToday - 6 * 86_400_000;

  const buckets: Record<string, ThreadInfo[]> = {
    Today: [],
    Yesterday: [],
    "Previous 7 days": [],
    Earlier: [],
  };

  for (const thread of threads) {
    const at = new Date(thread.updated_at || thread.created_at).getTime();
    if (Number.isNaN(at) || at >= startOfToday) buckets.Today.push(thread);
    else if (at >= startOfYesterday) buckets.Yesterday.push(thread);
    else if (at >= startOfWeek) buckets["Previous 7 days"].push(thread);
    else buckets.Earlier.push(thread);
  }

  return Object.entries(buckets)
    .filter(([, items]) => items.length > 0)
    .map(([label, items]) => ({ label, items }));
}

export function ThreadSidebar({
  threads,
  activeId,
  loading,
  onSelect,
  onCreate,
  onDelete,
}: {
  threads: ThreadInfo[];
  activeId: string | null;
  loading: boolean;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onDelete: (id: string) => void;
}) {
  const groups = useMemo(() => group(threads), [threads]);

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-border bg-surface">
      <div className="min-h-0 flex-1 overflow-y-auto px-2 py-3">
        {loading ? (
          <div className="space-y-1.5 px-1">
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} className="h-8 w-full" />
            ))}
          </div>
        ) : threads.length === 0 ? (
          <p className="px-2 py-8 text-center text-xs text-muted">
            No conversations yet.
          </p>
        ) : (
          groups.map(({ label, items }) => (
            <section key={label} className="mb-4 last:mb-0">
              <h2 className="px-2 pb-1 text-2xs font-medium uppercase tracking-wider text-muted">
                {label}
              </h2>
              <ul className="space-y-px">
                {items.map((thread) => (
                  <li key={thread.id} className="group relative">
                    <button
                      onClick={() => onSelect(thread.id)}
                      className={clsx(
                        "w-full truncate rounded-md py-1.5 pl-2 pr-8 text-left text-sm transition-colors",
                        thread.id === activeId
                          ? "bg-raised font-medium text-strong"
                          : "text-body hover:bg-raised/60",
                      )}
                      title={thread.title}
                    >
                      {thread.title}
                    </button>
                    <button
                      onClick={() => onDelete(thread.id)}
                      aria-label={`Delete ${thread.title}`}
                      // Hidden until hover/focus so the list stays calm, but
                      // reachable by keyboard.
                      className="absolute right-1 top-1/2 -translate-y-1/2 rounded p-1 text-muted opacity-0 transition-opacity hover:text-danger focus-visible:opacity-100 group-hover:opacity-100"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          ))
        )}
      </div>

      {/* Pinned to the bottom rather than the top: the list above is what the
          eye is here for, and a button that never moves is easier to hit than
          one that sits above a list of varying length. */}
      <div className="border-t border-border p-2">
        <Button variant="secondary" onClick={onCreate} className="w-full justify-start">
          <MessageSquarePlus className="h-3.5 w-3.5" />
          New chat
        </Button>
      </div>
    </aside>
  );
}
