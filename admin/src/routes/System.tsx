import { useEffect, useState } from "react";

import { admin, type ModelOption, type SystemStatus } from "../api";
import { StatusDot } from "../components/Stat";
import { Alert, Badge, Button, Card, Skeleton } from "../components/ui";

export default function System() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [models, setModels] = useState<{ available: ModelOption[]; enabled: string[] } | null>(
    null,
  );
  const [error, setError] = useState("");
  const [saving, setSaving] = useState("");

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [s, m] = await Promise.all([admin.system(), admin.models()]);
        if (!alive) return;
        setStatus(s);
        setModels(m);
      } catch (err) {
        if (alive) {
          setError(err instanceof Error ? err.message : "Could not load system status.");
        }
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  async function toggleModel(id: string) {
    if (!models) return;
    const next = models.enabled.includes(id)
      ? models.enabled.filter((m) => m !== id)
      : [...models.enabled, id];
    // The picker has to offer something. Emptying the list would leave every
    // chat request falling back to a default the UI cannot name.
    if (next.length === 0) {
      setError("At least one model must stay enabled.");
      return;
    }
    setError("");
    setSaving(id);
    try {
      setModels(await admin.setModels(next));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not update models.");
    } finally {
      setSaving("");
    }
  }

  return (
    <div className="space-y-5">
      {error && <Alert>{error}</Alert>}

      <Card title="Services">
        {!status ? (
          <Skeleton className="h-6 w-full" />
        ) : (
          <div className="flex flex-wrap gap-x-8 gap-y-2">
            <StatusDot ok={status.database} label="Postgres" />
            <StatusDot ok={status.neo4j} label="Neo4j" />
            <StatusDot ok={status.redis} label="Redis" />
          </div>
        )}
      </Card>

      <Card title="Configuration">
        {!status ? (
          <Skeleton className="h-24 w-full" />
        ) : (
          <dl className="grid gap-x-8 gap-y-1 text-sm sm:grid-cols-2">
            <Row label="Version" value={status.version} />
            <Row label="Default model" value={status.default_model} />
            <Row label="Vector store" value={status.vector_provider} />
            <Row label="Agent memory" value={status.memory_backend} />
          </dl>
        )}
      </Card>

      <Card title="Available models">
        <p className="mb-3 text-xs text-muted">
          Which models users can choose in chat. Disabling one hides it from the picker;
          requests naming it fall back to the default.
        </p>
        {!models ? (
          <Skeleton className="h-24 w-full" />
        ) : (
          <ul className="space-y-1.5">
            {models.available.map((m) => {
              const on = models.enabled.includes(m.model);
              return (
                <li
                  key={m.model}
                  className="flex items-center gap-3 rounded-md border border-border px-3 py-2"
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium text-strong">{m.label}</p>
                    <p className="truncate font-mono text-2xs text-muted">
                      {m.provider} · {m.model}
                    </p>
                  </div>
                  <Badge tone={on ? "positive" : "neutral"}>{on ? "enabled" : "disabled"}</Badge>
                  <Button
                    onClick={() => toggleModel(m.model)}
                    loading={saving === m.model}
                    disabled={Boolean(saving)}
                  >
                    {on ? "Disable" : "Enable"}
                  </Button>
                </li>
              );
            })}
          </ul>
        )}
      </Card>
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-3 border-b border-border py-1.5 last:border-0">
      <dt className="text-muted">{label}</dt>
      <dd className="font-mono text-xs text-strong">{value || "—"}</dd>
    </div>
  );
}
