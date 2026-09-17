"use client";

import Script from "next/script";

import type { EmbedConfig } from "@/lib/embed";

type Props = {
  streamUrl: string;
  embed: EmbedConfig;
};

/**
 * caster.fm's own embed (a widget or, on an older account, a plain iframe),
 * or a note that nothing is configured yet.
 *
 * There is no custom play button or volume control on this site: caster.fm's
 * embed is a cross-origin iframe, which the browser deliberately walls off
 * from anything on this page -- there is no way to reach its audio element
 * or its volume from out here, by design, the same way no page can reach
 * into an embedded ad or video from another site. Its own built-in controls
 * are the only ones that will ever exist for it.
 *
 * The `streamUrl` branch is kept for the day a real STREAM_URL exists (a
 * direct audio link this site could build its own player around), but none
 * of that UI exists today -- there was nothing to attach it to.
 */
export default function Player({ streamUrl, embed }: Props) {
  if (streamUrl) return null;

  if (embed?.kind === "widget") {
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

  if (embed?.kind === "iframe") {
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
