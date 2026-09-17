"""Publishes what the station is doing to a secret GitHub Gist.

The laptop lives behind a home router with no public address, so the website
cannot reach in to ask what is playing.  Instead the station pushes a small
JSON document out to a gist every few seconds, and the site reads it.

A gist rather than a database because it costs nothing, needs no new account,
and the GitHub token only ever has `gist` scope -- so the worst case if it
leaks is somebody editing a file that says what song is on.

Nothing here is allowed to be fatal.  If GitHub is down, the token is wrong or
the network is out, publishing quietly stops and the radio keeps playing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

import requests

from .config import Config
from .db import Database
from .library import Library
from .queueing import QueueManager

log = logging.getLogger("site")

API = "https://api.github.com"


class SitePublisher:
    def __init__(self, cfg: Config, db: Database, queue: QueueManager,
                 library: Library, weather=None):
        self.cfg = cfg
        self.db = db
        self.queue = queue
        self.library = library
        self.weather = weather

        self.token = Config.env("GITHUB_TOKEN")
        self.gist_id = Config.env("SITE_GIST_ID")
        self.filename = str(cfg.get("site.filename", "radio.json"))
        self.interval = float(cfg.get("site.publish_interval_seconds", 20))
        self.queue_length = int(cfg.get("site.queue_length", 8))
        self.history_length = int(cfg.get("site.history_length", 12))
        self._configured = bool(cfg.get("site.enabled", False))

        self._session = requests.Session()
        self._last_publish = 0.0
        self._last_digest = ""
        self._failures = 0
        self._quiet_until = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self._configured and self.token and self.gist_id)

    def why_disabled(self) -> str:
        if not self._configured:
            return "site.enabled is false in config"
        if not self.token:
            return "GITHUB_TOKEN is not set in .env"
        if not self.gist_id:
            return "SITE_GIST_ID is not set in .env (run: main.py site-init)"
        return ""

    # -- the document -------------------------------------------------------

    def payload(self, now_playing: str = "", since: float = 0.0) -> dict:
        """Everything the website shows, in one small document."""
        current = self._describe_now_playing(now_playing, since)
        upcoming = self._describe_queue(current)
        block = self.db.one(
            "SELECT mood_name, mood_note, source FROM blocks ORDER BY id DESC LIMIT 1")

        document = {
            "updated_at": time.time(),
            "station": {
                "name": self.cfg.get("station.name", "Radio"),
                "description": self.cfg.get("station.description", ""),
                "hosts": [str(host.get("name") or "") for host in
                          (self.cfg.get("dj.hosts", []) or [])
                          if isinstance(host, dict) and host.get("name")],
            },
            "now_playing": current,
            "show": {
                "note": (block["mood_note"] if block else "") or "",
                "slot": (block["mood_name"] if block else "") or "",
                "source": (block["source"] if block else "") or "",
                "mood": self.db.get_state("current_mood", ""),
            },
            "queue": upcoming,
            "history": self._describe_history(),
            "library": self._describe_library(),
        }

        reading = self.weather.current() if self.weather else None
        if reading:
            document["weather"] = {
                "place": reading.get("place"),
                "temp_c": reading.get("temp_c"),
                "condition": reading.get("condition"),
                "high_c": reading.get("high_c"),
                "low_c": reading.get("low_c"),
            }
        return document

    def _describe_now_playing(self, now_playing: str, since: float) -> dict:
        """Split the on-air string into fields the site can lay out.

        Liquidsoap only hands back one string, so the library is asked whether
        it recognises it -- that is where the duration comes from.
        """
        text = (now_playing or "").strip()
        current = {"text": text, "artist": "", "title": "", "kind": "song",
                   "duration": 0.0, "started_at": since or 0.0}
        if not text:
            return current

        if text.lower().endswith("station id") or " - station id" in text.lower():
            current.update(kind="patter", title="Station ID",
                           artist=str(self.cfg.get("station.name", "Radio")))
            return current

        # Try the exact split first. The fuzzy search is there for metadata
        # that does not match the library cleanly, but on a title like
        # "Artist 1 - Track 2" it will happily settle on Artist 2.
        track = None
        if " - " in text:
            artist, _, title = text.partition(" - ")
            track = self.library.has_track(artist.strip(), title.strip())
        track = track or self.library.find(text)
        if track:
            current.update(artist=track["artist"], title=track["title"],
                           duration=float(track.get("duration") or 0.0))
        elif " - " in text:
            artist, _, title = text.partition(" - ")
            current.update(artist=artist.strip(), title=title.strip())
        else:
            current["title"] = text
        return current

    def _describe_queue(self, current: dict) -> list[dict]:
        items = []
        for row in self.queue.preview(self.queue_length + 2, include_pending=True):
            if row["kind"] in ("patter", "banter"):
                items.append({
                    "kind": row["kind"],
                    "hosts": row.get("host") or "",
                    "artist": "", "title": "", "duration": 0.0,
                })
                continue
            # The first ready item is often the one already on air.
            if (items == [] and row["title"] == current.get("title")
                    and row["artist"] == current.get("artist")):
                continue
            items.append({
                "kind": "song",
                "artist": row["artist"] or "",
                "title": row["title"] or "",
                "duration": float(row["duration"] or 0.0),
                "hosts": "",
            })
        return items[:self.queue_length]

    def _describe_history(self) -> list[dict]:
        rows = self.db.query(
            "SELECT artist, title, last_played_at FROM tracks "
            "WHERE last_played_at IS NOT NULL ORDER BY last_played_at DESC LIMIT ?",
            (self.history_length,))
        return [{"artist": row["artist"], "title": row["title"],
                 "played_at": row["last_played_at"]} for row in rows]

    def _describe_library(self) -> dict:
        row = self.db.one(
            "SELECT COUNT(*) AS tracks, COUNT(DISTINCT artist) AS artists, "
            "COALESCE(SUM(duration), 0) AS seconds FROM tracks WHERE missing=0")
        return {
            "tracks": int(row["tracks"]) if row else 0,
            "artists": int(row["artists"]) if row else 0,
            "seconds": float(row["seconds"]) if row else 0.0,
        }

    # -- publishing ---------------------------------------------------------

    def maybe_publish(self, now_playing: str = "", since: float = 0.0) -> bool:
        """Publish if it is time and something actually changed."""
        if not self.enabled:
            return False
        now = time.time()
        if now - self._last_publish < self.interval:
            return False
        if now < self._quiet_until:
            return False
        self._last_publish = now

        document = self.payload(now_playing, since)
        # updated_at changes every call, so it is left out of the comparison:
        # otherwise every tick would look like a change and we would spend the
        # GitHub rate limit saying nothing new.
        digest = hashlib.sha1(
            json.dumps({k: v for k, v in document.items() if k != "updated_at"},
                       sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if digest == self._last_digest:
            return False

        if self._write(document):
            self._last_digest = digest
            return True
        return False

    def publish_now(self, document: dict | None = None) -> bool:
        """Push once, ignoring the interval. Used by the CLI."""
        if not self.enabled:
            return False
        return self._write(document if document is not None else self.payload())

    def _write(self, document: dict) -> bool:
        body = {"files": {self.filename: {
            "content": json.dumps(document, ensure_ascii=False, indent=1)}}}
        try:
            response = self._session.patch(
                f"{API}/gists/{self.gist_id}", json=body,
                headers=self._headers(), timeout=15)
            response.raise_for_status()
        except Exception as exc:
            self._failures += 1
            # Back off hard: a bad token fails identically every time, and
            # hammering it just burns the rate limit.
            self._quiet_until = time.time() + min(900, 30 * self._failures)
            level = log.warning if self._failures <= 2 else log.debug
            level("could not publish to the site gist (%s); retrying later", exc)
            return False

        if self._failures:
            log.info("site publishing recovered after %d failures", self._failures)
        self._failures = 0
        self._quiet_until = 0.0
        log.debug("published site state to gist %s", self.gist_id)
        return True

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    # -- setup --------------------------------------------------------------

    def create_gist(self) -> tuple[str, str]:
        """Create the secret gist this station will publish into.

        Returns (gist_id, raw_url). Raises on failure -- this one is
        interactive, so the person running it wants to hear about it.
        """
        if not self.token:
            raise RuntimeError("GITHUB_TOKEN is not set in .env")
        body = {
            "description": f"{self.cfg.get('station.name', 'Radio')} - now playing",
            # Secret, not private: unlisted, readable by anyone with the id.
            # Nothing in here is worth hiding, and the site has to read it.
            "public": False,
            "files": {self.filename: {"content": json.dumps(
                {"updated_at": time.time(), "now_playing": {}, "queue": []},
                indent=1)}},
        }
        response = self._session.post(f"{API}/gists", json=body,
                                      headers=self._headers(), timeout=20)
        response.raise_for_status()
        data = response.json()
        return data["id"], data.get("html_url", "")
