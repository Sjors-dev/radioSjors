"use client";

import { useEffect, useRef, useState } from "react";

type Props = {
  streamUrl: string;
  embedUrl: string;
};

type Status = "idle" | "connecting" | "playing" | "error";

export default function Player({ streamUrl, embedUrl }: Props) {
  const audio = useRef<HTMLAudioElement | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [volume, setVolume] = useState(0.8);

  useEffect(() => {
    if (audio.current) audio.current.volume = volume;
  }, [volume]);

  // caster.fm gives an embeddable player of its own. It works, but it drags
  // its own styling in with it, so it is only used when there is no direct
  // stream URL to build a nicer button around.
  if (!streamUrl && embedUrl) {
    return (
      <div className="block block--hot">
        <h2 className="block__label">
          <span>Listen</span>
        </h2>
        <iframe
          className="embed"
          src={embedUrl}
          title="Radio player"
          allow="autoplay"
        />
      </div>
    );
  }

  if (!streamUrl) {
    return (
      <div className="block">
        <h2 className="block__label">
          <span>Listen</span>
        </h2>
        <p className="note note--quiet">
          No stream URL configured. Set <code>STREAM_URL</code> to the listen
          link from caster.fm, or <code>PLAYER_EMBED_URL</code> to their embed,
          and redeploy.
        </p>
      </div>
    );
  }

  async function toggle() {
    const element = audio.current;
    if (!element) return;

    if (status === "playing" || status === "connecting") {
      element.pause();
      // A paused live stream keeps buffering, and then plays minutes-old
      // audio when you come back. Dropping the source stops the world.
      element.removeAttribute("src");
      element.load();
      setStatus("idle");
      return;
    }

    setStatus("connecting");
    element.src = streamUrl;
    try {
      await element.play();
      setStatus("playing");
    } catch {
      setStatus("error");
    }
  }

  const label = {
    idle: "Tap to tune in",
    connecting: "Connecting",
    playing: "Live",
    error: "Could not connect",
  }[status];

  return (
    <div className="block block--hot">
      <h2 className="block__label">
        <span>Listen</span>
        <span>{streamUrl.startsWith("https") ? "secure" : "direct"}</span>
      </h2>

      <div className="player">
        <div className="player__row">
          <button
            className="playbtn"
            onClick={toggle}
            disabled={status === "connecting"}
            aria-label={status === "playing" ? "Stop" : "Play"}
          >
            {status === "playing" ? (
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <rect x="5" y="4" width="5" height="16" />
                <rect x="14" y="4" width="5" height="16" />
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" aria-hidden="true">
                <path d="M6 3l15 9-15 9z" />
              </svg>
            )}
          </button>

          <div className="player__state">
            {label}
            <div
              className={`bars${status === "playing" ? " is-live" : ""}`}
              aria-hidden="true"
            >
              <i />
              <i />
              <i />
              <i />
              <i />
              <i />
            </div>
          </div>
        </div>

        <input
          className="volume"
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={volume}
          onChange={(event) => setVolume(Number(event.target.value))}
          aria-label="Volume"
        />
      </div>

      <audio ref={audio} preload="none" />
    </div>
  );
}
