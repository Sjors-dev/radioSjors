"""Last.fm discovery.

The LLM is deliberately *not* used to name tracks: it hallucinates titles and
its music knowledge is frozen at its training cutoff.  Last.fm returns real,
existing tracks, and also gives us a reference duration that the downloader
uses as a sanity check.
"""

from __future__ import annotations

import logging
import random
import time

import requests

log = logging.getLogger("discovery")

API_ROOT = "https://ws.audioscrobbler.com/2.0/"


class LastFM:
    def __init__(self, api_key: str, timeout: int = 20):
        self.api_key = (api_key or "").strip()
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "ai-radio/1.0 (private station)"
        self._last_call = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    # -- low level ----------------------------------------------------------

    def _call(self, method: str, **params) -> dict:
        if not self.enabled:
            log.debug("Last.fm disabled (no LASTFM_API_KEY), skipping %s", method)
            return {}

        # Be polite: Last.fm asks for well under 5 requests/second.
        elapsed = time.time() - self._last_call
        if elapsed < 0.25:
            time.sleep(0.25 - elapsed)

        query = {"method": method, "api_key": self.api_key, "format": "json"}
        query.update({k: v for k, v in params.items() if v not in (None, "")})
        try:
            response = self._session.get(API_ROOT, params=query, timeout=self.timeout)
            self._last_call = time.time()
            if response.status_code != 200:
                log.warning("Last.fm %s -> HTTP %s", method, response.status_code)
                return {}
            data = response.json()
            if "error" in data:
                log.warning("Last.fm %s -> error %s: %s", method,
                            data.get("error"), data.get("message"))
                return {}
            return data
        except Exception as exc:
            log.warning("Last.fm %s failed: %s", method, exc)
            return {}

    # -- discovery ----------------------------------------------------------

    def similar_artists(self, artist: str, limit: int = 12) -> list[str]:
        data = self._call("artist.getSimilar", artist=artist, limit=limit, autocorrect=1)
        items = (data.get("similarartists") or {}).get("artist") or []
        return [item["name"] for item in items if item.get("name")]

    def top_tracks(self, artist: str, limit: int = 15) -> list[dict]:
        data = self._call("artist.getTopTracks", artist=artist, limit=limit, autocorrect=1)
        items = (data.get("toptracks") or {}).get("track") or []
        out = []
        for item in items:
            name = item.get("name")
            who = (item.get("artist") or {}).get("name") or artist
            if name:
                out.append({"artist": who, "title": name})
        return out

    def similar_tracks(self, artist: str, title: str, limit: int = 15) -> list[dict]:
        data = self._call("track.getSimilar", artist=artist, track=title,
                          limit=limit, autocorrect=1)
        items = (data.get("similartracks") or {}).get("track") or []
        out = []
        for item in items:
            name = item.get("name")
            who = (item.get("artist") or {}).get("name")
            if name and who:
                out.append({"artist": who, "title": name})
        return out

    def track_info(self, artist: str, title: str) -> dict:
        """Returns {duration: seconds|None, tags: [...], listeners: int}."""
        data = self._call("track.getInfo", artist=artist, track=title, autocorrect=1)
        track = data.get("track") or {}
        duration_ms = track.get("duration")
        try:
            duration = float(duration_ms) / 1000.0 if duration_ms else None
        except (TypeError, ValueError):
            duration = None
        if duration is not None and duration <= 1:
            duration = None  # Last.fm uses 0 for "unknown"

        tags = [t["name"] for t in ((track.get("toptags") or {}).get("tag") or [])
                if t.get("name")]
        listeners = 0
        try:
            listeners = int((track.get("listeners") or 0))
        except (TypeError, ValueError):
            pass
        return {
            "duration": duration,
            "tags": tags[:6],
            "listeners": listeners,
            "artist": (track.get("artist") or {}).get("name") or artist,
            "title": track.get("name") or title,
        }

    def artist_tags(self, artist: str) -> list[str]:
        data = self._call("artist.getTopTags", artist=artist, autocorrect=1)
        tags = (data.get("toptags") or {}).get("tag") or []
        return [t["name"] for t in tags[:6] if t.get("name")]

    def tag_top_tracks(self, tag: str, limit: int = 30) -> list[dict]:
        data = self._call("tag.getTopTracks", tag=tag, limit=limit)
        items = (data.get("tracks") or {}).get("track") or []
        out = []
        for item in items:
            name = item.get("name")
            who = (item.get("artist") or {}).get("name")
            if name and who:
                out.append({"artist": who, "title": name})
        return out

    # -- higher level -------------------------------------------------------

    def expand(self, seeds: list[str], want: int = 25,
               per_artist: int = 3) -> list[dict]:
        """Grow a pool of candidate tracks from a handful of seed artists.

        Breadth first, deliberately. Taking eight top tracks from one artist
        fills the library with that artist; taking two or three from many
        artists is what actually grows a station. The result is round-robined
        so the download queue alternates artists rather than working through
        one discography at a time.
        """
        if not self.enabled or not seeds:
            return []

        # Widen to every similar artist we can reach before fetching a single
        # track, so the artist list is broad before the track list is deep.
        artists: list[str] = []
        seen_artists: set[str] = set()
        shuffled = list(seeds)
        random.shuffle(shuffled)

        for seed in shuffled:
            for artist in [seed] + self.similar_artists(seed, limit=10):
                key = artist.lower().strip()
                if key and key not in seen_artists:
                    seen_artists.add(key)
                    artists.append(artist)
            # Enough artists to fill the request several times over at the
            # per-artist cap, without hammering Last.fm for all of them.
            if len(artists) >= max(12, want):
                break

        random.shuffle(artists)

        buckets: list[list[dict]] = []
        seen_tracks: set[tuple[str, str]] = set()
        collected = 0
        for artist in artists:
            bucket = []
            for track in self.top_tracks(artist, limit=per_artist * 2):
                key = (track["artist"].lower(), track["title"].lower())
                if key in seen_tracks:
                    continue
                seen_tracks.add(key)
                bucket.append(track)
                if len(bucket) >= per_artist:
                    break
            if bucket:
                buckets.append(bucket)
                collected += len(bucket)
            if collected >= want * 2:
                break

        # Round-robin: one track from each artist, then a second, and so on.
        pool: list[dict] = []
        for index in range(per_artist):
            for bucket in buckets:
                if index < len(bucket):
                    pool.append(bucket[index])

        log.info("Last.fm expansion: %d seeds -> %d artists -> %d candidates "
                 "(max %d per artist)",
                 len(seeds), len(buckets), len(pool[:want]), per_artist)
        return pool[:want]
