/**
 * Reads what the station is doing.
 *
 * The radio itself runs on a laptop behind a home router, so it cannot be
 * asked anything. It pushes instead: a small JSON document into a secret
 * GitHub Gist every twenty seconds. This is the other end of that.
 *
 * Everything here runs on the server so the GitHub token never reaches a
 * browser.
 */

export type NowPlaying = {
  text: string;
  artist: string;
  title: string;
  kind: "song" | "patter";
  duration: number;
  started_at: number;
};

export type QueueItem = {
  kind: "song" | "patter" | "banter";
  artist: string;
  title: string;
  duration: number;
  hosts: string;
};

export type HistoryItem = {
  artist: string;
  title: string;
  played_at: number;
};

export type StationState = {
  updated_at: number;
  station: { name: string; description: string; hosts: string[] };
  now_playing: NowPlaying;
  show: { note: string; slot: string; source: string; mood: string };
  queue: QueueItem[];
  history: HistoryItem[];
  library: { tracks: number; artists: number; seconds: number };
  weather?: {
    place: string;
    temp_c: number | null;
    condition: string;
    high_c: number | null;
    low_c: number | null;
  };
};

export type StationResult =
  | { ok: true; state: StationState }
  | { ok: false; error: string };

const GIST_API = "https://api.github.com/gists";

export async function readStation(): Promise<StationResult> {
  const gistId = process.env.GIST_ID;
  const token = process.env.GITHUB_TOKEN;
  const filename = process.env.GIST_FILENAME || "radio.json";

  if (!gistId) return { ok: false, error: "GIST_ID is not set" };

  const headers: Record<string, string> = {
    Accept: "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
  };
  // A secret gist is readable without a token if you know the id, so the
  // token is optional here even though the radio needs one to write.
  if (token) headers.Authorization = `Bearer ${token}`;

  let response: Response;
  try {
    response = await fetch(`${GIST_API}/${gistId}`, {
      headers,
      cache: "no-store",
    });
  } catch {
    return { ok: false, error: "could not reach GitHub" };
  }

  if (!response.ok) {
    return { ok: false, error: `GitHub said ${response.status}` };
  }

  let payload: { files?: Record<string, { content?: string }> };
  try {
    payload = await response.json();
  } catch {
    return { ok: false, error: "GitHub sent something unreadable" };
  }

  const file = payload.files?.[filename];
  if (!file?.content) {
    return { ok: false, error: `no ${filename} in that gist yet` };
  }

  try {
    return { ok: true, state: JSON.parse(file.content) as StationState };
  } catch {
    return { ok: false, error: "the station wrote something malformed" };
  }
}

/** Seconds since the current track started, clamped to its length. */
export function elapsed(now: NowPlaying, at: number = Date.now() / 1000) {
  if (!now.started_at) return 0;
  const seconds = Math.max(0, at - now.started_at);
  return now.duration > 0 ? Math.min(seconds, now.duration) : seconds;
}

export function clock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/** "4 minutes ago", for the recently-played list. */
export function ago(timestamp: number, at: number = Date.now() / 1000): string {
  const seconds = Math.max(0, at - timestamp);
  if (seconds < 90) return "just now";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} hr ago`;
  return `${Math.round(hours / 24)} d ago`;
}
