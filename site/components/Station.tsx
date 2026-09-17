"use client";

import { useEffect, useMemo, useState } from "react";

import Player from "@/components/Player";
import Stamp from "@/components/Stamp";
import { ago, clock, elapsed, StationState } from "@/lib/station";

type Props = {
  initial: StationState | null;
  initialError: string;
  streamUrl: string;
  embedUrl: string;
};

// The radio pushes every twenty seconds, so asking more often than this only
// burns GitHub's rate limit for the same answer.
const POLL_MS = 12_000;

// After this long with no update the laptop is almost certainly not running.
const STALE_SECONDS = 180;

export default function Station({
  initial,
  initialError,
  streamUrl,
  embedUrl,
}: Props) {
  const [state, setState] = useState<StationState | null>(initial);
  const [error, setError] = useState(initialError);
  // Null until the browser has mounted. Reading the clock during render would
  // give the server one answer and the browser another a moment later, and
  // React calls that a hydration mismatch. Until then the station's own
  // publish time stands in, which is within a few seconds of right anyway.
  const [clientNow, setClientNow] = useState<number | null>(null);
  const now = clientNow ?? state?.updated_at ?? 0;

  useEffect(() => {
    let alive = true;

    async function poll() {
      try {
        const response = await fetch("/api/state", { cache: "no-store" });
        if (!alive) return;
        if (!response.ok) {
          // The route knows why -- "GIST_ID is not set" is a far better thing
          // to read than "something went wrong".
          const body = (await response.json().catch(() => ({}))) as {
            error?: string;
          };
          setError(body.error || "the station is not answering");
          return;
        }
        setState((await response.json()) as StationState);
        setError("");
      } catch {
        if (alive) setError("lost the connection");
      }
    }

    const poller = setInterval(poll, POLL_MS);
    // A ticking clock, so the progress bar moves between polls instead of
    // jumping twelve seconds at a time.
    const ticker = setInterval(() => setClientNow(Date.now() / 1000), 1000);
    setClientNow(Date.now() / 1000);
    poll();

    return () => {
      alive = false;
      clearInterval(poller);
      clearInterval(ticker);
    };
  }, []);

  const stale = useMemo(() => {
    if (!state?.updated_at) return true;
    return now - state.updated_at > STALE_SECONDS;
  }, [state, now]);

  const marquee = useMemo(() => {
    const tracks = (state?.history ?? []).map(
      (item) => `${item.artist} — ${item.title}`,
    );
    if (!tracks.length) return ["warming up"];
    return tracks;
  }, [state]);

  const current = state?.now_playing;
  const played = current ? elapsed(current, now) : 0;
  const progress =
    current && current.duration > 0
      ? Math.min(100, (played / current.duration) * 100)
      : 0;
  const talking = current?.kind === "patter";

  const name = state?.station.name || "Radio";
  const [first, ...rest] = name.split(" ");

  return (
    <>
      <div className="ticker">
        <div className="ticker__track">
          {[...marquee, ...marquee].map((line, index) => (
            <span key={index}>{line}</span>
          ))}
        </div>
      </div>

      <div className="shell">
        <header className="masthead">
          <h1 className="wordmark">
            {first} {rest.length ? <em>{rest.join(" ")}</em> : null}
          </h1>
          <div className="masthead__aside">
            <div className={`onair${stale ? " is-off" : ""}`}>
              <span className="onair__dot" />
              {stale ? "off air" : "on air"}
            </div>
            <div>{state?.station.description || " "}</div>
            <div>
              {state?.show.slot ? `${state.show.slot} rotation` : " "}
            </div>
            <div>
              {state?.station.hosts?.length
                ? `with ${state.station.hosts.join(" & ")}`
                : " "}
            </div>
          </div>
        </header>

        {!state ? (
          <div className="offline">
            <strong>Nothing coming through.</strong>
            <br />
            {error || "The station has not published anything yet."}
          </div>
        ) : (
          <>
            <div className="deck">
              <section
                className={`block block--paper now${talking ? " now--talk" : ""}`}
              >
                <h2 className="block__label">
                  <span>{talking ? "On the mic" : "Now playing"}</span>
                  <span>{stale ? "last seen " + ago(state.updated_at, now) : "live"}</span>
                </h2>

                {talking ? <div className="talkbadge">Talk</div> : null}

                <h3 className="now__title">
                  {current?.title || "—"}
                </h3>
                <p className="now__artist">{current?.artist || ""}</p>

                <div className="now__spacer" />

                {current && current.duration > 0 ? (
                  <>
                    <div className="now__meter">
                      <i style={{ width: `${progress}%` }} />
                    </div>
                    <div className="now__times">
                      <span>{clock(played)}</span>
                      <span>{clock(current.duration)}</span>
                    </div>
                  </>
                ) : null}
              </section>

              <div style={{ display: "grid", gap: 18 }}>
                <Player streamUrl={streamUrl} embedUrl={embedUrl} />

                {state.weather ? (
                  <section className="block">
                    <h2 className="block__label">
                      <span>Outside</span>
                      <span>{state.weather.place}</span>
                    </h2>
                    <div className="weather">
                      <b>
                        {state.weather.temp_c ?? "—"}
                        {"°"}
                      </b>
                      <div>
                        <div>{state.weather.condition || "unknown"}</div>
                        <div className="note note--quiet">
                          high {state.weather.high_c ?? "?"}
                          {"°"} / low {state.weather.low_c ?? "?"}
                          {"°"}
                        </div>
                      </div>
                    </div>
                  </section>
                ) : null}
              </div>
            </div>

            <div className="columns">
              <section className="block">
                <h2 className="block__label">
                  <span>Up next</span>
                  <span>{state.queue.length} queued</span>
                </h2>
                {state.queue.length ? (
                  <ol className="list">
                    {state.queue.map((item, index) => (
                      <li
                        key={index}
                        className={item.kind === "song" ? "" : "is-talk"}
                      >
                        <span className="list__index">
                          {String(index + 1).padStart(2, "0")}
                        </span>
                        <span className="list__body">
                          {item.kind === "song" ? (
                            <>
                              <span className="list__title">{item.title}</span>
                              <span className="list__sub">{item.artist}</span>
                            </>
                          ) : (
                            <>
                              <span className="list__title">
                                {item.kind === "banter"
                                  ? "The hosts, talking"
                                  : "Station link"}
                              </span>
                              <span className="list__sub">
                                {item.hosts || "on the mic"}
                              </span>
                            </>
                          )}
                        </span>
                        <span className="list__meta">
                          {item.duration > 0 ? clock(item.duration) : ""}
                        </span>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="note note--quiet">
                    Nothing queued yet. The next hour is still being written.
                  </p>
                )}
              </section>

              <section className="block">
                <h2 className="block__label">
                  <span>Just played</span>
                  <span>{state.history.length}</span>
                </h2>
                {state.history.length ? (
                  <ol className="list">
                    {state.history.map((item, index) => (
                      <li key={`${item.title}-${index}`}>
                        <span className="list__body">
                          <span className="list__title">{item.title}</span>
                          <span className="list__sub">{item.artist}</span>
                        </span>
                        <span className="list__meta">
                          {ago(item.played_at, now)}
                        </span>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <p className="note note--quiet">Nothing yet tonight.</p>
                )}
              </section>

              <section className="block">
                <h2 className="block__label">
                  <span>This hour</span>
                  <span>{state.show.source || ""}</span>
                </h2>
                <p className="note">
                  {state.show.note || "No notes on this one."}
                </p>
                {state.show.mood ? (
                  <p className="note note--quiet" style={{ marginTop: 12 }}>
                    You asked for: {state.show.mood}
                  </p>
                ) : null}

                <div style={{ height: 18 }} />

                <div className="stats">
                  <div className="stat">
                    <b>{state.library.tracks}</b>
                    <span>tracks</span>
                  </div>
                  <div className="stat">
                    <b>{state.library.artists}</b>
                    <span>artists</span>
                  </div>
                  <div className="stat">
                    <b>{Math.round(state.library.seconds / 3600)}</b>
                    <span>hours</span>
                  </div>
                </div>
              </section>
            </div>
          </>
        )}

        <footer className="foot">
          <div>
            {error ? `⚠ ${error}` : `updated ${state ? ago(state.updated_at, now) : "never"}`}
          </div>
          <Stamp />
        </footer>
      </div>
    </>
  );
}
