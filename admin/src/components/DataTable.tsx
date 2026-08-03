import clsx from "clsx";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import { EmptyState, Skeleton } from "./ui";

export interface Column<T> {
  key: string;
  label: string;
  align?: "left" | "right";
  sortable?: boolean;
  /** Sort key, when the rendered cell is not what you sort on. */
  value?: (row: T) => string | number;
  render: (row: T) => ReactNode;
  className?: string;
}

interface Props<T> {
  columns: Column<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  loading?: boolean;
  empty?: { title: string; detail?: string };
  onRowClick?: (row: T) => void;
}

/**
 * A dense table with client-side sorting.
 *
 * Client-side on purpose: every list that reaches this component is already
 * bounded by a server-side page (the users endpoint caps `size` at 100), so
 * sorting is reordering at most a screen or two of rows. Pushing it to the
 * server would add a round trip and a query parameter to every column header
 * for no benefit at this scale.
 */
export function DataTable<T>({
  columns,
  rows,
  rowKey,
  loading,
  empty,
  onRowClick,
}: Props<T>) {
  const [sort, setSort] = useState<{ key: string; desc: boolean } | null>(null);

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const column = columns.find((c) => c.key === sort.key);
    if (!column?.value) return rows;
    const read = column.value;
    // Copy before sorting: `rows` belongs to the caller's state, and sorting in
    // place mutates it without React knowing.
    return [...rows].sort((a, b) => {
      const x = read(a);
      const y = read(b);
      const order = typeof x === "number" && typeof y === "number"
        ? x - y
        : String(x).localeCompare(String(y));
      return sort.desc ? -order : order;
    });
  }, [rows, sort, columns]);

  const toggle = (key: string) =>
    setSort((prev) =>
      prev?.key === key ? { key, desc: !prev.desc } : { key, desc: false },
    );

  if (loading) {
    return (
      <div className="space-y-2 p-1">
        {Array.from({ length: 6 }).map((_, i) => (
          <Skeleton key={i} className="h-8 w-full" />
        ))}
      </div>
    );
  }

  if (!sorted.length) {
    return <EmptyState title={empty?.title ?? "Nothing here yet"} detail={empty?.detail} />;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b border-border">
            {columns.map((c) => (
              <th
                key={c.key}
                scope="col"
                className={clsx(
                  "whitespace-nowrap px-3 py-2 text-2xs font-medium uppercase tracking-wider text-muted",
                  c.align === "right" ? "text-right" : "text-left",
                )}
              >
                {c.sortable && c.value ? (
                  <button
                    onClick={() => toggle(c.key)}
                    className={clsx(
                      "inline-flex items-center gap-1 hover:text-body",
                      sort?.key === c.key && "text-body",
                    )}
                  >
                    {c.label}
                    {sort?.key === c.key &&
                      (sort.desc ? (
                        <ChevronDown className="h-3 w-3" />
                      ) : (
                        <ChevronUp className="h-3 w-3" />
                      ))}
                  </button>
                ) : (
                  c.label
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr
              key={rowKey(row)}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              className={clsx(
                "border-b border-border/60 last:border-0",
                onRowClick && "cursor-pointer hover:bg-raised",
              )}
            >
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={clsx(
                    "whitespace-nowrap px-3 py-2 text-body",
                    c.align === "right" && "text-right",
                    c.className,
                  )}
                >
                  {c.render(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
