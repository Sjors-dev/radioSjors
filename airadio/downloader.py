"""yt-dlp wrapper with the quality filters that make it usable.

A bare yt-dlp search returns a lot of junk for music: live versions, nightcore
edits, ten-hour loops, reaction videos.  Every candidate is scored and anything
suspicious is rejected and logged rather than silently added to the library.

Only ever one download runs at a time (the brain loop is single threaded, and a
lock file guards against a CLI invocation racing it) so background fetching
never starves the stream on a two-core CPU.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
from pathlib import Path

from .config import Config
from .db import Database
from .discovery import LastFM
from .library import Library
from .util import (contains_word, dedupe_key, normalize, safe_filename,
                   strip_decorations)

log = logging.getLogger("downloader")


class DownloadResult:
    def __init__(self, ok: bool, track: dict | None = None, reason: str = "",
                 busy: bool = False):
        self.ok = ok
        self.track = track
        self.reason = reason
        # busy means "someone else holds the download lock", which is not a
        # failure of this track and must not count against its attempts.
        self.busy = busy

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"<DownloadResult ok={self.ok} busy={self.busy} "
                f"reason={self.reason!r}>")


class Downloader:
    def __init__(self, cfg: Config, db: Database, library: Library, lastfm: LastFM):
        self.cfg = cfg
        self.db = db
        self.library = library
        self.lastfm = lastfm
        self.lock_path = cfg.path("state") / "download.lock"

    # -- locking ------------------------------------------------------------

    def _acquire_lock(self) -> bool:
        try:
            if self.lock_path.exists():
                age = time.time() - self.lock_path.stat().st_mtime
                if age < 1800:
                    return False
                log.warning("clearing stale download lock (%.0fs old)", age)
                self.lock_path.unlink(missing_ok=True)
            self.lock_path.write_text(str(os.getpid()), encoding="utf-8")
            return True
        except Exception as exc:
            log.warning("download lock error: %s", exc)
            return True  # never let the lock itself block the station

    def _release_lock(self) -> None:
        try:
            self.lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    # -- scoring ------------------------------------------------------------

    def _reject_patterns(self, requested_title: str,
                         allow_remix: bool = False) -> list[str]:
        """Patterns to reject, minus any the listener explicitly asked for.

        Someone requesting "Live Forever" or a specific remix should still get
        it, so a pattern present in the request is not disqualifying.
        """
        wanted = normalize(requested_title)
        patterns = self.cfg.get("downloader.reject_title_patterns", []) or []
        patterns = [p for p in patterns if normalize(p) not in wanted]
        if allow_remix:
            relaxed = {"remix", "bootleg", "mashup", "extended"}
            patterns = [p for p in patterns if p.lower() not in relaxed]
        return patterns

    def score_candidate(self, candidate: dict, artist: str, title: str,
                        expected_duration: float | None,
                        allow_remix: bool = False) -> tuple[int, str]:
        """Return (score, reason). Score below zero means reject."""
        cand_title = candidate.get("title") or ""
        channel = (candidate.get("channel") or candidate.get("uploader") or "")
        duration = candidate.get("duration")
        low_title = cand_title.lower()
        low_channel = channel.lower()

        min_len = float(self.cfg.get("downloader.min_duration_seconds", 75))
        max_len = float(self.cfg.get("downloader.max_duration_seconds", 900))

        if duration is None:
            return -1, "no duration reported"
        if duration < min_len:
            return -1, f"too short ({duration:.0f}s < {min_len:.0f}s)"
        if duration > max_len:
            return -1, f"too long ({duration:.0f}s > {max_len:.0f}s)"

        for pattern in self._reject_patterns(title, allow_remix):
            if contains_word(cand_title, pattern):
                return -1, f"title contains {pattern!r}"

        if expected_duration:
            tolerance = float(self.cfg.get("downloader.duration_tolerance", 0.25))
            drift = abs(duration - expected_duration) / expected_duration
            if drift > tolerance:
                return -1, (f"duration {duration:.0f}s is {drift * 100:.0f}% off "
                            f"the expected {expected_duration:.0f}s")

        score = 0
        for pattern in self.cfg.get("downloader.prefer_channel_patterns", []) or []:
            if pattern.lower() in low_channel:
                # "Artist - Topic" is auto-generated official audio: best case.
                score += 40 if "topic" in pattern.lower() else 20
                break

        # The channel actually being the artist is a strong signal.
        if normalize(artist) and normalize(artist) in normalize(channel):
            score += 25

        norm_cand = normalize(cand_title)
        title_words = [w for w in normalize(title).split() if len(w) > 2]
        if title_words:
            hits = sum(1 for w in title_words if w in norm_cand)
            score += int(30 * hits / len(title_words))
        if normalize(artist) and normalize(artist) in norm_cand:
            score += 15
        if "official" in low_title and "video" not in low_title:
            score += 5
        if expected_duration:
            drift = abs(duration - expected_duration) / expected_duration
            score += int(20 * (1 - min(drift / 0.25, 1.0)))

        return score, "ok"

    # -- searching ----------------------------------------------------------

    def _search(self, artist: str, title: str) -> list[dict]:
        import yt_dlp

        count = int(self.cfg.get("downloader.search_results", 8))
        query = f"ytsearch{count}:{artist} {title}"
        options = {
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "extract_flat": "in_playlist",
            "skip_download": True,
            "socket_timeout": 30,
            "ignoreerrors": True,
        }
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                data = ydl.extract_info(query, download=False)
        except Exception as exc:
            log.warning("search failed for %s - %s: %s", artist, title, exc)
            return []

        entries = (data or {}).get("entries") or []
        results = []
        for entry in entries:
            if not entry:
                continue
            results.append({
                "id": entry.get("id"),
                "url": entry.get("url") or entry.get("webpage_url")
                       or f"https://www.youtube.com/watch?v={entry.get('id')}",
                "title": entry.get("title") or "",
                "channel": entry.get("channel") or entry.get("uploader") or "",
                "duration": entry.get("duration"),
            })
        return results

    # -- the main entry point ----------------------------------------------

    def fetch(self, artist: str, title: str, allow_remix: bool = False,
              max_seconds: float | None = None) -> DownloadResult:
        """Find, vet, download, tag and register one track."""
        artist = strip_decorations(artist or "").strip()
        title = strip_decorations(title or "").strip()
        if not artist or not title:
            return DownloadResult(False, reason="missing artist or title")

        if self.library.is_banned(artist, title):
            log.info("refusing to download banned track: %s - %s", artist, title)
            return DownloadResult(False, reason="that one is banned from the station")

        existing = self.library.has_track(artist, title)
        if existing:
            return DownloadResult(True, existing, "already in library")

        if not self._acquire_lock():
            return DownloadResult(False, reason="another download is running",
                                  busy=True)

        try:
            return self._fetch_locked(artist, title, allow_remix, max_seconds)
        finally:
            self._release_lock()

    def _fetch_locked(self, artist: str, title: str, allow_remix: bool,
                      max_seconds: float | None) -> DownloadResult:
        info = self.lastfm.track_info(artist, title) if self.lastfm.enabled else {}
        expected = info.get("duration")
        # Trust Last.fm's spelling of the artist/title when it corrected us.
        artist = info.get("artist") or artist
        title = info.get("title") or title

        candidates = self._search(artist, title)
        if not candidates:
            self._log_attempt(artist, title, None, "error", "no search results")
            return DownloadResult(False, reason="no search results")

        scored = []
        for candidate in candidates:
            score, reason = self.score_candidate(
                candidate, artist, title, expected, allow_remix)

            if max_seconds and (candidate.get("duration") or 0) > max_seconds:
                score, reason = -1, f"longer than the {max_seconds:.0f}s request cap"

            if score < 0:
                log.info("rejected %r (%s) - %s", candidate["title"],
                         candidate["channel"], reason)
                self._log_attempt(artist, title, candidate, "rejected", reason)
            else:
                scored.append((score, candidate))

        if not scored:
            return DownloadResult(False, reason="every candidate failed quality checks")

        scored.sort(key=lambda pair: pair[0], reverse=True)
        best_score, best = scored[0]
        log.info("picked %r (%s, %ss, score %d) for %s - %s", best["title"],
                 best["channel"], best.get("duration"), best_score, artist, title)

        path = self._download(best, artist, title)
        if path is None:
            self._log_attempt(artist, title, best, "error", "download failed")
            return DownloadResult(False, reason="download failed")

        tags = info.get("tags") or []
        if not tags and self.lastfm.enabled:
            tags = self.lastfm.artist_tags(artist)

        self.library.write_tags(
            path,
            title=title,
            artist=artist,
            genre=(tags[0] if tags else ""),
            airadio_tags=", ".join(tags),
        )

        actual = self.library.read_tags(path)
        track_id = self.db.execute(
            "INSERT INTO tracks(path, dedupe_key, title, artist, genre, tags, "
            "duration, source_url, added_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (str(path.resolve()), dedupe_key(artist, title), title, artist,
             (tags[0] if tags else ""), ", ".join(tags), actual.get("duration") or 0.0,
             best.get("url"), time.time()),
        )
        self._log_attempt(artist, title, best, "accepted", f"score {best_score}")
        log.info("added to library: %s - %s (%s)", artist, title, path.name)

        row = self.db.one("SELECT * FROM tracks WHERE id = ?", (track_id,))
        return DownloadResult(True, dict(row) if row else None, "downloaded")

    def _download(self, candidate: dict, artist: str, title: str) -> Path | None:
        import yt_dlp

        library_dir = self.cfg.path("library")
        quality = str(self.cfg.get("downloader.audio_quality", "0"))

        with tempfile.TemporaryDirectory(prefix="airadio-dl-") as tmp:
            tmp_dir = Path(tmp)
            options = {
                "format": "bestaudio/best",
                "outtmpl": str(tmp_dir / "track.%(ext)s"),
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": quality,
                }],
                "quiet": True,
                "no_warnings": True,
                "noprogress": True,
                "noplaylist": True,
                "retries": 3,
                "fragment_retries": 3,
                "socket_timeout": 30,
                "concurrent_fragment_downloads": 1,
                "ignoreerrors": False,
            }
            try:
                with yt_dlp.YoutubeDL(options) as ydl:
                    ydl.download([candidate["url"]])
            except Exception as exc:
                log.warning("yt-dlp failed for %s: %s", candidate.get("url"), exc)
                return None

            produced = sorted(tmp_dir.glob("track.mp3")) or sorted(tmp_dir.glob("*.mp3"))
            if not produced:
                log.warning("yt-dlp produced no mp3 for %s - %s (is ffmpeg installed?)",
                            artist, title)
                return None

            target = library_dir / f"{safe_filename(artist)} - {safe_filename(title)}.mp3"
            counter = 1
            while target.exists():
                target = library_dir / (
                    f"{safe_filename(artist)} - {safe_filename(title)} ({counter}).mp3")
                counter += 1
            shutil.move(str(produced[0]), str(target))
            return target

    def _log_attempt(self, artist: str, title: str, candidate: dict | None,
                     decision: str, reason: str) -> None:
        self.db.execute(
            "INSERT INTO download_log(artist, title, video_id, video_title, channel, "
            "duration, decision, reason, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (artist, title,
             (candidate or {}).get("id"),
             (candidate or {}).get("title"),
             (candidate or {}).get("channel"),
             (candidate or {}).get("duration"),
             decision, reason, time.time()),
        )

    # -- background growth --------------------------------------------------

    def artist_track_count(self, artist: str) -> int:
        row = self.db.one(
            "SELECT COUNT(*) AS n FROM tracks WHERE missing=0 AND artist=? "
            "COLLATE NOCASE", (artist,))
        return int(row["n"]) if row else 0

    def artist_cap(self, artist: str) -> int:
        """How many tracks this artist is allowed in the library.

        An artist you named in seed_artists is one you actually like, so it
        gets a much higher ceiling than something Last.fm suggested three hops
        out. The point of the cap was never to ration favourites, only to stop
        one name crowding everything else out.
        """
        seeds = {normalize(name) for name in
                 (self.cfg.get("discovery.seed_artists", []) or []) if name}
        if normalize(artist) in seeds:
            return int(self.cfg.get("discovery.max_tracks_per_seed_artist", 0) or 0)
        return int(self.cfg.get("discovery.max_tracks_per_artist", 0) or 0)

    def artist_is_full(self, artist: str) -> bool:
        cap = self.artist_cap(artist)
        return bool(cap) and self.artist_track_count(artist) >= cap

    def seed_candidates(self, wanted: int = 25) -> int:
        """Ask Last.fm for new tracks and stash them as pending candidates."""
        seeds = list(self.cfg.get("discovery.seed_artists", []) or [])
        per_artist = int(self.cfg.get("discovery.tracks_per_artist", 3))

        # Seed from the library's own artists too, so the station drifts with
        # the listener's taste -- but from the LEAST represented ones. Seeding
        # from the most played artists is a feedback loop: the artists that
        # already dominate the library pull in more of themselves.
        rows = self.db.query(
            "SELECT artist, COUNT(*) AS n FROM tracks WHERE missing=0 "
            "GROUP BY artist ORDER BY n ASC LIMIT 10"
        )
        seeds += [row["artist"] for row in rows]
        if not seeds:
            return 0

        added = 0
        skipped_full = 0
        for track in self.lastfm.expand(seeds, want=wanted, per_artist=per_artist):
            key = dedupe_key(track["artist"], track["title"])
            if self.artist_is_full(track["artist"]):
                # One artist should not be able to crowd out everything else.
                # Seeds get a far higher ceiling, and a chat request overrides
                # it entirely -- that path goes through fetch().
                skipped_full += 1
                continue
            if self.db.one("SELECT 1 FROM tracks WHERE dedupe_key=? AND missing=0", (key,)):
                continue
            if self.db.one("SELECT 1 FROM banned WHERE dedupe_key=?", (key,)):
                continue
            if self.db.one("SELECT 1 FROM candidates WHERE dedupe_key=?", (key,)):
                continue
            self.db.execute(
                "INSERT OR IGNORE INTO candidates(dedupe_key, artist, title, source, "
                "created_at) VALUES(?,?,?,?,?)",
                (key, track["artist"], track["title"], "lastfm", time.time()),
            )
            added += 1
        if added or skipped_full:
            log.info("queued %d new download candidates (%d skipped, artist "
                     "already at its track cap)", added, skipped_full)
        return added

    def fetch_next_candidate(self) -> DownloadResult | None:
        """Download one pending candidate. Returns None when there are none."""
        target = int(self.cfg.get("discovery.library_target", 0) or 0)
        if target and self.library.count() >= target:
            log.debug("library target of %d reached, not growing", target)
            return None

        row = self.db.one(
            "SELECT * FROM candidates WHERE status='new' AND attempts < 3 "
            "ORDER BY created_at LIMIT 1"
        )
        if row is not None and self.artist_is_full(row["artist"]):
            # The cap was clear when this was queued but has filled since.
            self.db.execute(
                "UPDATE candidates SET status='rejected', note=?, updated_at=? "
                "WHERE id=?",
                (f"artist at its cap of {self.artist_cap(row['artist'])}",
                 time.time(), row["id"]))
            return DownloadResult(False, reason=f"{row['artist']} is at its track cap")
        if row is None:
            if self.seed_candidates() == 0:
                return None
            row = self.db.one(
                "SELECT * FROM candidates WHERE status='new' AND attempts < 3 "
                "ORDER BY created_at LIMIT 1"
            )
            if row is None:
                return None

        result = self.fetch(row["artist"], row["title"])
        if result.busy:
            # Lost the lock race. Nothing to record: the candidate was never
            # actually tried, so burning an attempt on it would eventually mark
            # a perfectly good track as permanently rejected.
            return result

        self.db.execute(
            "UPDATE candidates SET attempts = attempts + 1, status=?, note=?, "
            "updated_at=? WHERE id=?",
            ("downloaded" if result.ok else
             ("rejected" if row["attempts"] >= 2 else "new"),
             result.reason, time.time(), row["id"]),
        )
        return result
