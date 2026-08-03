import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";

import { auth } from "../api";
import { Alert, Button, Field, Input } from "../components/ui";
import { useAuth } from "../lib/auth";
import { AuthLayout } from "./AuthLayout";

const RESEND_SECONDS = 45;

export default function Verify() {
  const navigate = useNavigate();
  const location = useLocation();
  const { setMe } = useAuth();
  const passedEmail = (location.state as { email?: string } | null)?.email ?? "";

  const [email, setEmail] = useState(passedEmail);
  const [code, setCode] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [cooldown, setCooldown] = useState(RESEND_SECONDS);
  const codeRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    codeRef.current?.focus();
  }, []);

  // A countdown rather than an always-live button: resending invalidates the
  // previous code, so an impatient double-click would otherwise void the code
  // the user is about to type.
  useEffect(() => {
    if (cooldown <= 0) return;
    const timer = setTimeout(() => setCooldown((s) => s - 1), 1000);
    return () => clearTimeout(timer);
  }, [cooldown]);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      setMe(await auth.verify(email, code));
      navigate("/chat", { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : "That code is not valid.");
      setCode("");
      codeRef.current?.focus();
    } finally {
      setBusy(false);
    }
  }

  async function resend() {
    setError("");
    setNotice("");
    try {
      await auth.resend(email);
      setNotice("A new code is on its way. The previous one no longer works.");
      setCooldown(RESEND_SECONDS);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not send a new code.");
    }
  }

  return (
    <AuthLayout
      title="Check your email"
      subtitle={
        email
          ? `Enter the 6-digit code sent to ${email}.`
          : "Enter your email and the 6-digit code we sent."
      }
      footer={
        <Link to="/login" className="font-medium text-accent hover:underline">
          Back to sign in
        </Link>
      }
    >
      <form onSubmit={submit} className="space-y-4">
        {error && <Alert>{error}</Alert>}
        {notice && <Alert tone="positive">{notice}</Alert>}

        {!passedEmail && (
          <Field label="Email">
            <Input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              autoComplete="email"
              required
            />
          </Field>
        )}

        <Field label="Verification code">
          <Input
            ref={codeRef}
            value={code}
            // Digits only, capped at six: the field can't hold anything the
            // server would reject.
            onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
            inputMode="numeric"
            autoComplete="one-time-code"
            placeholder="000000"
            className="text-center font-mono text-lg tracking-[0.4em]"
            required
          />
        </Field>

        <Button
          type="submit"
          variant="primary"
          loading={busy}
          disabled={code.length !== 6}
          className="w-full"
        >
          Verify
        </Button>

        <div className="text-center text-sm text-muted">
          {cooldown > 0 ? (
            <span>Didn't get it? You can resend in {cooldown}s.</span>
          ) : (
            <button
              type="button"
              onClick={resend}
              className="font-medium text-accent hover:underline"
            >
              Send a new code
            </button>
          )}
        </div>
      </form>
    </AuthLayout>
  );
}
