import { useEffect, useState } from "react";

import { admin, type AuditEntry } from "../api";
import { Timeline } from "../components/Timeline";
import { Alert, Card, Select, Skeleton } from "../components/ui";

/**
 * The admin action log.
 *
 * Its own page rather than a panel at the bottom of System: "who suspended this
 * account" is the first question asked when something looks wrong, and it is
 * not a question about system health.
 */
export default function Audit() {
  const [entries, setEntries] = useState<AuditEntry[] | null>(null);
  const [limit, setLimit] = useState(100);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    setEntries(null);
    admin
      .audit(limit)
      .then((rows) => alive && setEntries(rows))
      .catch((err: unknown) => {
        if (alive) {
          setError(err instanceof Error ? err.message : "Could not load the audit log.");
        }
      });
    return () => {
      alive = false;
    };
  }, [limit]);

  return (
    <div className="space-y-5">
      {error && <Alert>{error}</Alert>}
      <Card
        title="Admin actions"
        actions={
          <Select
            value={String(limit)}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="h-7 !w-auto py-0 text-xs"
            aria-label="How many entries"
          >
            <option value="50">Last 50</option>
            <option value="100">Last 100</option>
            <option value="500">Last 500</option>
          </Select>
        }
      >
        {!entries ? (
          <div className="space-y-2">
            {Array.from({ length: 8 }).map((_, i) => (
              <Skeleton key={i} className="h-6 w-full" />
            ))}
          </div>
        ) : (
          <Timeline entries={entries} />
        )}
      </Card>
    </div>
  );
}
