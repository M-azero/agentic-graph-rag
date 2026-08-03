import { useEffect, useState } from "react";

import { admin } from "../api";
import { LimitsForm } from "../components/LimitsForm";
import { Alert, Button, Card, Modal, Skeleton } from "../components/ui";

export default function Limits() {
  const [globals, setGlobals] = useState<Record<string, number> | null>(null);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [saving, setSaving] = useState(false);
  const [confirmClear, setConfirmClear] = useState(false);

  async function load() {
    try {
      setGlobals(await admin.globalLimits());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load limits.");
    }
  }

  useEffect(() => {
    void load();
  }, []);

  async function run(fn: () => Promise<unknown>, done: string) {
    setError("");
    setNotice("");
    setSaving(true);
    try {
      await fn();
      setNotice(done);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : "That didn't work.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="space-y-5">
      {error && <Alert>{error}</Alert>}
      {notice && <Alert tone="positive">{notice}</Alert>}

      <Card title="Default limits">
        <p className="mb-4 text-xs text-muted">
          What every user gets unless they have an override. Changes apply immediately,
          including to existing accounts. Token ceilings count prompt and completion
          together — budget roughly 10k per question, not the few hundred the visible
          answer suggests.
        </p>
        {!globals ? (
          <Skeleton className="h-40 w-full" />
        ) : (
          <LimitsForm
            mode="global"
            values={globals}
            saving={saving}
            onSave={(values) =>
              run(
                () => admin.setGlobalLimits(values as Record<string, number>),
                "Default limits updated.",
              )
            }
          />
        )}
      </Card>

      <Card title="Apply to everyone">
        <p className="mb-4 text-xs text-muted">
          Clearing every override puts all users back on the defaults above. Individual
          limits set on a user's page are discarded.
        </p>
        <Button variant="danger" onClick={() => setConfirmClear(true)}>
          Clear all per-user overrides
        </Button>
      </Card>

      {confirmClear && (
        <Modal title="Clear every override?" onClose={() => setConfirmClear(false)}>
          <p className="text-sm text-muted">
            Every user returns to the default limits. Any custom allowance you have granted
            is removed.
          </p>
          <div className="mt-4 flex justify-end gap-2">
            <Button onClick={() => setConfirmClear(false)}>Cancel</Button>
            <Button
              variant="danger"
              onClick={async () => {
                setConfirmClear(false);
                await run(() => admin.bulkLimits({ clear: true }), "All overrides cleared.");
              }}
            >
              Clear overrides
            </Button>
          </div>
        </Modal>
      )}
    </div>
  );
}
