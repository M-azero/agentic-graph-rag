import clsx from "clsx";
import { MoreVertical } from "lucide-react";
import { useState, type ReactNode } from "react";

export interface Action {
  label: string;
  onSelect: () => void;
  danger?: boolean;
  disabled?: boolean;
  icon?: ReactNode;
}

/** The ⋮ menu at the end of a table row. */
export function RowMenu({ actions, label = "Actions" }: { actions: Action[]; label?: string }) {
  const [open, setOpen] = useState(false);
  const usable = actions.filter((a) => !a.disabled);
  if (!usable.length) return null;

  return (
    <div className="relative inline-block text-left">
      <button
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={(e) => {
          // Rows are clickable; opening the menu must not also navigate.
          e.stopPropagation();
          setOpen((v) => !v);
        }}
        className="rounded p-1 text-muted hover:bg-raised hover:text-body"
      >
        <MoreVertical className="h-4 w-4" />
      </button>

      {open && (
        <>
          {/* Click-away layer — a menu that only closes via its own items
              strands anyone who changes their mind. */}
          <div className="fixed inset-0 z-40" onClick={(e) => { e.stopPropagation(); setOpen(false); }} />
          <div
            role="menu"
            className="absolute right-0 z-50 mt-1 w-52 rounded-lg border border-border bg-surface p-1 shadow-pop animate-slide-up"
          >
            {usable.map((action) => (
              <button
                key={action.label}
                role="menuitem"
                onClick={(e) => {
                  e.stopPropagation();
                  setOpen(false);
                  action.onSelect();
                }}
                className={clsx(
                  "flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-sm",
                  action.danger ? "text-danger hover:bg-danger/10" : "text-body hover:bg-raised",
                )}
              >
                {action.icon}
                {action.label}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
