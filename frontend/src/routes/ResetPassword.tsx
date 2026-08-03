import { useState, type FormEvent } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";

import { ApiError, auth } from "../api";
import { Alert, Button, Field, Input } from "../components/ui";
import { AuthLayout } from "./AuthLayout";

/**
 * Step two: the code plus a new password.
 *
 * Signing in is deliberately NOT automatic. The server does not open a session
 * here, so a stolen code alone is not a login — the user has to type the new
 * password once on the sign-in page, which proves they know it.
 */
export default function ResetPassword() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const [email, setEmail] = useState(params.get("email") ?? "");
  const [code, setCode] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    // Checked here rather than server-side: the confirmation field exists to
    // catch a typo, and a round trip would spend one of five attempts on it.
    if (password !== confirm) {
      setError("Those passwords don't match.");
      return;
    }
    setError("");
    setBusy(true);
    try {
      await auth.resetPassword(email, code, password);
      navigate("/login", {
        replace: true,
        state: { notice: "Password updated. Sign in with your new password." },
      });
    } catch (err) {
      if (err instanceof ApiError && err.code === "no_code") {
        setError("That code has been used or never existed. Request a new one.");
      } else {
        setError(err instanceof Error ? err.message : "Could not reset the password.");
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthLayout
      title="Choose a new password"
      subtitle="Enter the code we emailed, then pick a password."
      footer={
        <>
          Didn't get a code?{" "}
          <Link to="/forgot-password" className="font-medium text-accent hover:underline">
            Send another
          </Link>
        </>
      }
    >
      <form onSubmit={submit} className="space-y-4">
        {error && <Alert>{error}</Alert>}
        <Field label="Email">
          <Input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            required
          />
        </Field>
        <Field label="Code">
          <Input
            value={code}
            onChange={(e) => setCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
            inputMode="numeric"
            autoComplete="one-time-code"
            placeholder="123456"
            className="text-center font-mono text-lg tracking-[0.4em]"
            autoFocus={Boolean(email)}
            required
          />
        </Field>
        <Field label="New password" hint="At least 10 characters.">
          <Input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="new-password"
            required
          />
        </Field>
        <Field label="Confirm password">
          <Input
            type="password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
            autoComplete="new-password"
            required
          />
        </Field>
        <Button type="submit" variant="primary" loading={busy} className="w-full">
          Set password
        </Button>
      </form>
    </AuthLayout>
  );
}
