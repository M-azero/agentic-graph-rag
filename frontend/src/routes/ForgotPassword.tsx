import { useState, type FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";

import { auth } from "../api";
import { Alert, Button, Field, Input } from "../components/ui";
import { AuthLayout } from "./AuthLayout";

/**
 * Step one of a password reset: ask for a code.
 *
 * The server answers the same way for every address, registered or not, so
 * this page cannot report "no such account" — and deliberately does not try.
 * Saying anything more specific would turn it into a way to test whether
 * someone has an account here.
 */
export default function ForgotPassword() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      await auth.forgotPassword(email);
      navigate(`/reset-password?email=${encodeURIComponent(email)}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not send a code.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthLayout
      title="Reset your password"
      subtitle="We'll email you a six-digit code."
      footer={
        <>
          Remembered it?{" "}
          <Link to="/login" className="font-medium text-accent hover:underline">
            Sign in
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
            autoFocus
            required
          />
        </Field>
        <Button type="submit" variant="primary" loading={busy} className="w-full">
          Send code
        </Button>
      </form>
    </AuthLayout>
  );
}
