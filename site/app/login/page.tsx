"use client";

import { useRouter } from "next/navigation";
import { FormEvent, useState } from "react";

export default function LoginPage() {
  const router = useRouter();
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const response = await fetch("/api/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (response.ok) {
        router.replace("/");
        router.refresh();
        return;
      }
      const body = (await response.json().catch(() => ({}))) as {
        error?: string;
      };
      setError(body.error || "That did not work.");
    } catch {
      setError("Could not reach the door.");
    }
    setBusy(false);
  }

  return (
    <main className="gate">
      <div className="gate__card">
        <h1>
          Members
          <br />
          only
        </h1>
        <p>
          One listener, one password. This station is private on purpose, which
          is the whole reason it gets to play whatever it likes.
        </p>
        <form onSubmit={submit}>
          <input
            className="field"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            placeholder="password"
            autoFocus
            autoComplete="current-password"
            aria-label="Password"
          />
          <button className="submit" type="submit" disabled={busy || !password}>
            {busy ? "checking" : "let me in"}
          </button>
        </form>
        {error ? <p className="error">{error}</p> : null}
      </div>
    </main>
  );
}
