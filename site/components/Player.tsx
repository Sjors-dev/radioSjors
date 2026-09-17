"use client";

import Script from "next/script";
import { useEffect, useRef, useState } from "react";

import type { EmbedConfig } from "@/lib/embed";

type Props = {
  streamUrl: string;
  embed: EmbedConfig;
};

type Status = "idle" | "connecting" | "playing" | "error";

export default function Player({ streamUrl, embed }: Props) {
  const audio = useRef<HTMLAudioElement | null>(null);
  const [status, setStatus] = useState<Status>("idle");
  const [volume, setVolume] = useState(0.8);

  useEffect(() => {
    if (audio.current) audio.current.volume = volume;
  }, [volume]);

  // caster.fm gives an embeddable player of its own. It works, but it drags
  // its own styling in with it, so it is only used when there is no direct
  // stream URL to build a nicer button around.
  if (!streamUrl && embed?.kind === "widget") {
    return (
      <div className="block block--hot">
        <h2 className="block__label">
          <span>Listen</span>
        </h2>
        <div className="embed">
          {/* caster.fm's loader script scans the page for this div by class
             and renders the actual player into it. React must never touch
             its children after that, so nothing here is React-managed. */}
          <div
            className="cstrEmbed"
            data-type="newStreamPlayer"
            // React only passes a data-* attribute through as-is when it is
            // spelled all-lowercase; caster.fm's snippet writes these in
            // camelCase, but a browser parsing that snippet as raw HTML
            // would lowercase it anyway (that is what HTML5 parsing does to
            // attribute names), and their own script reads it that way --
            // confirmed working. Writing it lowercase here just matches
            // reality and drops React's warning about it.
            data-publictoken={embed.publicToken}
            data-theme={embed.theme}
            data-color={embed.color}
            data-channelid=""
            data-rendered="false"
          >
            {/* caster.fm's widget script refuses to render without these --
               it is their "powered by" attribution requirement, not
               decoration, so it stays exactly as their dashboard gives it. */}
            <a href="https://www.caster.fm">Shoutcast Hosting</a>{" "}
            <a href="https://www.caster.fm">Stream Hosting</a>{" "}
            <a href="https://www.caster.fm">Radio Server Hosting</a>
          </div>
        </div>
        <Script src="https://cdn.cloud.caster.fm//widgets/embed.js"
               strategy="afterInteractive" />
      </div>
    );
  }

  if (!streamUrl && embed?.kind === "iframe") {
    return (
      <div className="block block--hot">
        <h2 className="block__label">
          <span>Listen</span>
        </h2>
        <iframe
          className="embed"
          src={embed.url}
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
          link from caster.fm, or paste their embed code (the whole snippet)
          into <code>PLAYER_EMBED_URL</code>, and redeploy.
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
