import { useState, type FormEvent } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";

import { ApiError } from "../api";
import { Alert, Button, Field, Input } from "../components/ui";
import { safeNext, useAuth } from "../lib/auth";
import { AuthLayout } from "./AuthLayout";

export default function Login() {
  const { me, signIn, loading } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const state = location.state as { from?: string; notice?: string } | null;
  const next = safeNext(location.search);

  /** Where to go once signed in. `?next=` wins because it is the admin
   *  console's hand-off, and it is a different app rather than a route here. */
  function goOnwards() {
    if (next) {
      // A different bundle at a different root — a router navigation would try
      // to resolve it inside this SPA and land on the catch-all.
      window.location.assign(next);
      return;
    }
    navigate(state?.from ?? "/chat", { replace: true });
  }

  if (!loading && me) {
    if (next) {
      window.location.replace(next);
      return null;
    }
    return <Navigate to={state?.from ?? "/chat"} replace />;
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      await signIn(email, password);
      goOnwards();
    } catch (err) {
      // An unverified account isn't a failed login — it's an unfinished
      // signup, so send them to the step they stopped at.
      if (err instanceof ApiError && err.code === "email_unverified") {
        navigate("/verify", { state: { email } });
        return;
      }
      setError(err instanceof Error ? err.message : "Could not sign in.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <AuthLayout
      title="Sign in"
      subtitle="Continue to your knowledge base."
      footer={
        <>
          No account?{" "}
          <Link to="/signup" className="font-medium text-accent hover:underline">
            Create one
          </Link>
        </>
      }
    >
      <form onSubmit={submit} className="space-y-4">
        {/* Carried from a completed reset, so the outcome is stated where the
            user next looks rather than lost in the navigation. */}
        {state?.notice && <Alert tone="positive">{state.notice}</Alert>}
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
        <Field label="Password">
          <Input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
        </Field>
        <div className="flex justify-end">
          <Link
            to="/forgot-password"
            className="text-sm text-muted hover:text-accent hover:underline"
          >
            Forgot your password?
          </Link>
        </div>
        <Button type="submit" variant="primary" loading={busy} className="w-full">
          Sign in
        </Button>
      </form>
    </AuthLayout>
  );
}
