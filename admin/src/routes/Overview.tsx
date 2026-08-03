import { useEffect, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { admin, type SystemStatus, type UsageSeries } from "../api";
import { Stat, StatRail } from "../components/Stat";
import { Alert, Card, EmptyState, Select, Skeleton, compactNumber } from "../components/ui";

export default function Overview() {
  const [usage, setUsage] = useState<UsageSeries | null>(null);
  const [system, setSystem] = useState<SystemStatus | null>(null);
  const [days, setDays] = useState(30);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [u, s] = await Promise.all([admin.usage(days), admin.system()]);
        if (!alive) return;
        setUsage(u);
        setSystem(s);
      } catch (err) {
        if (alive) {
          setError(err instanceof Error ? err.message : "Could not load the overview.");
        }
      }
    })();
    return () => {
      alive = false;
    };
  }, [days]);

  const number = (v?: number) => (v === undefined ? "—" : v.toLocaleString());

  return (
    <div className="space-y-5">
      {error && <Alert>{error}</Alert>}

      <StatRail>
        <Stat
          label="Users"
          value={number(system?.users)}
          sub={`${system?.active_users ?? 0} active in 30d`}
        />
        <Stat label="Conversations" value={number(system?.threads)} />
        <Stat label="Documents" value={number(system?.files)} />
        <Stat
          label={`Messages · ${days}d`}
          value={number(usage?.totals.messages)}
          sub={`${compactNumber(usage?.totals.tokens ?? 0)} tokens`}
        />
      </StatRail>

      <Card
        title="Activity"
        actions={
          <Select
            value={String(days)}
            onChange={(e) => setDays(Number(e.target.value))}
            className="h-7 !w-auto py-0 text-xs"
            aria-label="Time range"
          >
            <option value="7">7 days</option>
            <option value="30">30 days</option>
            <option value="90">90 days</option>
          </Select>
        }
      >
        {!usage ? (
          <Skeleton className="h-56 w-full" />
        ) : usage.points.length === 0 ? (
          <EmptyState title="No activity in this period" />
        ) : (
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={usage.points} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="messages" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="rgb(var(--accent))" stopOpacity={0.25} />
                    <stop offset="100%" stopColor="rgb(var(--accent))" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="rgb(var(--border))" strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="bucket"
                  tickFormatter={shortDate}
                  tick={{ fontSize: 11, fill: "rgb(var(--text-muted))" }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "rgb(var(--text-muted))" }}
                  axisLine={false}
                  tickLine={false}
                  allowDecimals={false}
                />
                <Tooltip
                  contentStyle={{
                    background: "rgb(var(--surface))",
                    border: "1px solid rgb(var(--border))",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  labelFormatter={(label) => shortDate(String(label))}
                />
                <Area
                  type="monotone"
                  dataKey="messages"
                  stroke="rgb(var(--accent))"
                  strokeWidth={2}
                  fill="url(#messages)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card>
    </div>
  );
}

function shortDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
