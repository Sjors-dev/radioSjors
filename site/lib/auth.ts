/**
 * The front door.
 *
 * The station is deliberately private -- that is what keeps music licensing
 * out of the picture -- so the site sits behind one shared password. The
 * cookie holds a hash derived from that password, which means it cannot be
 * forged without knowing it, and no session store is needed anywhere.
 *
 * This runs in the Edge runtime (middleware), so it uses Web Crypto rather
 * than anything from Node.
 */

export const COOKIE = "radio_pass";

/** Fails closed. An unset password locks the site rather than opening it. */
export function passwordIsConfigured(): boolean {
  return Boolean(process.env.SITE_PASSWORD);
}

export async function tokenFor(password: string): Promise<string> {
  const data = new TextEncoder().encode(`radio-station:${password}`);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export async function isUnlocked(cookieValue?: string): Promise<boolean> {
  const password = process.env.SITE_PASSWORD;
  // No password configured means nobody gets in. Opening the station to the
  // whole internet because an environment variable was forgotten is not a
  // failure mode worth having.
  if (!password || !cookieValue) return false;
  return equal(cookieValue, await tokenFor(password));
}

/** Constant time, so the cookie cannot be guessed a character at a time. */
function equal(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let index = 0; index < a.length; index += 1) {
    diff |= a.charCodeAt(index) ^ b.charCodeAt(index);
  }
  return diff === 0;
}
