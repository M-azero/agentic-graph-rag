import clsx from "clsx";
import { MoreVertical } from "lucide-react";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

export interface Action {
  label: string;
  onSelect: () => void;
  danger?: boolean;
  disabled?: boolean;
  icon?: ReactNode;
}

/** The ⋮ menu at the end of a table row. */
export function RowMenu({ actions, label = "Actions" }: { actions: Action[]; label?: string }) {
  // Where the menu opens, in viewport coordinates. The menu is portaled to
  // <body> and positioned from the button: rows sit inside glass cards, and a
  // backdrop-filter pane both clips fixed children and paints later cards over
  // anything that spills out of it.
  const [anchor, setAnchor] = useState<{ top: number; right: number } | null>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const open = anchor !== null;

  useEffect(() => {
    if (!open) return;
    const close = () => setAnchor(null);
    // A press outside closes it — a menu that only closes via its own items
    // strands anyone who changes their mind. Scrolling closes it too, since
    // the menu is pinned to where the button *was*.
    const onDown = (e: PointerEvent) => {
      const target = e.target as Node;
      if (!menuRef.current?.contains(target) && !buttonRef.current?.contains(target)) close();
    };
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    document.addEventListener("pointerdown", onDown);
    document.addEventListener("keydown", onKey);
    window.addEventListener("scroll", close, true);
    window.addEventListener("resize", close);
    return () => {
      document.removeEventListener("pointerdown", onDown);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", close, true);
      window.removeEventListener("resize", close);
    };
  }, [open]);

  const usable = actions.filter((a) => !a.disabled);
  if (!usable.length) return null;

  return (
    <div className="relative inline-block text-left">
      <button
        ref={buttonRef}
        aria-label={label}
        aria-haspopup="menu"
        aria-expanded={open}
        onClick={(e) => {
          // Rows are clickable; opening the menu must not also navigate.
          e.stopPropagation();
          if (open) return setAnchor(null);
          const rect = e.currentTarget.getBoundingClientRect();
          setAnchor({ top: rect.bottom + 4, right: window.innerWidth - rect.right });
        }}
        className="rounded p-1 text-muted hover:bg-raised hover:text-body"
      >
        <MoreVertical className="h-4 w-4" />
      </button>

      {anchor &&
        createPortal(
          <div
            ref={menuRef}
            role="menu"
            style={{ top: anchor.top, right: anchor.right }}
            className="glass-pop fixed z-50 w-52 rounded-lg p-1 animate-slide-up"
          >
            {usable.map((action) => (
              <button
                key={action.label}
                role="menuitem"
                onClick={(e) => {
                  e.stopPropagation();
                  setAnchor(null);
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
          </div>,
          document.body,
        )}
    </div>
  );
}
