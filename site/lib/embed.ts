/**
 * caster.fm's embed comes in two shapes depending on when your account was
 * set up: an old-style `<iframe src="...">` (a plain URL), or their newer
 * widget -- a `<div class="cstrEmbed" data-publicToken="...">` that a loader
 * script scans the page for and renders into.
 *
 * PLAYER_EMBED_URL is meant to hold whatever caster.fm's dashboard gives you
 * to copy, unedited -- the whole snippet, not a hand-extracted piece of it.
 * This picks the right shape back apart from that.
 */

export type EmbedConfig =
  | { kind: "widget"; publicToken: string; theme: string; color: string }
  | { kind: "iframe"; url: string }
  | null;

const ATTR = (name: string) => new RegExp(`data-${name}="([^"]*)"`, "i");

export function parseEmbed(raw: string): EmbedConfig {
  const value = (raw || "").trim();
  if (!value) return null;

  const token = value.match(ATTR("publictoken"));
  if (token && token[1]) {
    return {
      kind: "widget",
      publicToken: token[1],
      theme: value.match(ATTR("theme"))?.[1] || "dark",
      color: value.match(ATTR("color"))?.[1] || "6929D7",
    };
  }

  // No widget markup found. If it is at least a real URL, honour it as a
  // plain iframe -- that covers an older caster.fm account, or a manually
  // typed link. Anything else (stray HTML with no token, plain text) is not
  // something we can safely point an iframe or a widget at.
  if (/^https?:\/\//i.test(value)) {
    return { kind: "iframe", url: value };
  }

  return null;
}
