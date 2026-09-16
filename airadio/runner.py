"""The brain loop.

Deliberately single threaded: planning, TTS rendering and downloading all run
one after another, which is exactly the concurrency cap a two-core Celeron
needs.  Feeding liquidsoap is cheap and happens every tick, before and after any
heavy job, so a slow download can never empty the stream's queue.

Everything in here is wrapped so that a failure logs and the loop continues.
The stream is downstream of this process and keeps playing regardless.
"""

from __future__ import annotations

import logging
import signal
import time
from pathlib import Path

from .config import Config
from .db import Database
from .discovery import LastFM
from .downloader import Downloader
from .library import Library
from .liquidsoap import AI_QUEUE, REQUEST_QUEUE, LiquidsoapClient, annotate_uri
from .queueing import QueueManager
from .tts import TTS
from .util import human_duration, normalize
from .brain.intent import classify
from .brain.llm import LLM
from .brain.planner import Planner

log = logging.getLogger("runner")


class Runner:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = Database(cfg.db_path)
        self.db.init()

        self.library = Library(self.db, cfg.path("library"))
        self.lastfm = LastFM(Config.env("LASTFM_API_KEY"))
        self.llm = LLM(cfg)
        self.tts = TTS(cfg)
        self.planner = Planner(cfg, self.db, self.library, self.llm)
        self.downloader = Downloader(cfg, self.db, self.library, self.lastfm)
        self.queue = QueueManager(cfg, self.db, self.library, self.tts)
        self.ls = LiquidsoapClient(
            str(cfg.get("stream.telnet_host", "127.0.0.1")),
            int(cfg.get("stream.telnet_port", 1234)),
        )

        self.tick_seconds = float(cfg.get("runner.tick_seconds", 5))
        self.queue_depth = int(cfg.get("stream.liquidsoap_queue_depth", 4))
        self.download_interval = float(cfg.get("runner.download_interval_seconds", 1200))
        self.enrich_interval = float(cfg.get("runner.enrich_interval_seconds", 900))

        self._running = True
        self._last_download = 0.0
        self._last_enrich = 0.0
        self._last_housekeeping = 0.0
        self._last_connected: bool | None = None
        self._last_now_playing = ""
        # Per queue: was the item we pushed last a patter line? Used to
        # kill the crossfade on both sides of a spoken link.
        self._last_push_was_patter: dict[str, bool] = {}

    # -- lifecycle ----------------------------------------------------------

    def stop(self, *_args) -> None:
        log.info("shutdown requested")
        self._running = False

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)

        ok, detail = self.tts.check()
        log.info("tts: %s", detail)
        if not ok:
            log.warning("patter is disabled until TTS works; music will still play")
        log.info("llm providers available: %s",
                 ", ".join(self.llm.available_providers) or "none (fallback planner only)")
        log.info("last.fm: %s", "enabled" if self.lastfm.enabled else "disabled (no key)")
        log.info("library: %d tracks", self.library.count())

        while self._running:
            started = time.time()
            try:
                self.tick()
            except Exception:
                log.exception("tick failed, continuing")
            elapsed = time.time() - started
            if self._running:
                time.sleep(max(0.5, self.tick_seconds - elapsed))

        log.info("brain stopped")

    # -- one pass -----------------------------------------------------------

    def tick(self) -> None:
        self.feed_stream()
        self.handle_chat_requests()
        self.maintain_buffer()
        self.background_work()
        self.feed_stream()

    # -- stream feeding -----------------------------------------------------

    def feed_stream(self) -> None:
        """Keep liquidsoap's queues shallow but never empty.

        One telnet connection per call: liquidsoap logs every connect and
        disconnect, and this runs every few seconds for years.
        """
        state = self.ls.poll(AI_QUEUE, REQUEST_QUEUE)
        connected = state is not None
        if connected != self._last_connected:
            if connected:
                log.info("connected to liquidsoap telnet")
            else:
                log.warning("liquidsoap telnet unreachable - is the stream service up? "
                            "(queue is safe, items stay ready)")
            self._last_connected = connected
        if state is None:
            return

        depth = state["depths"].get(AI_QUEUE, 0)
        self._update_now_playing(state["now_playing"])

        # Liquidsoap owns playback, so its queue depth is the only signal for
        # what it has finished. Without this, pushed items count towards the
        # buffer forever and the planner eventually stops building blocks.
        # The +1 covers the track that has left the queue and is on air.
        self.queue.reconcile_pushed("ai", depth + 1)
        self.queue.reconcile_pushed(
            "request", state["depths"].get(REQUEST_QUEUE, 0))

        # Tier 1: listener requests, pushed immediately and in full.
        for item in self.queue.take_next("request", limit=4):
            self._push(REQUEST_QUEUE, item)

        # Tier 2: the AI-planned block, topped up to a shallow depth so a vibe
        # shift or a re-plan takes effect within a couple of tracks.
        missing = self.queue_depth - depth
        if missing > 0:
            for item in self.queue.take_next("ai", limit=missing):
                self._push(AI_QUEUE, item)

    def _update_now_playing(self, current: str) -> None:
        """Mirror liquidsoap's current track into a file the bot can read.

        Cheaper than giving the bot its own telnet dependency, and it survives
        the stream being restarted.
        """
        if not current or current == self._last_now_playing:
            return
        self._last_now_playing = current
        try:
            self.cfg.now_playing_path.write_text(current, encoding="utf-8")
        except Exception as exc:
            log.debug("could not write now_playing.txt: %s", exc)
        log.info("on air: %s", current)

    def _push(self, queue_id: str, item: dict) -> None:
        title = item.get("title") or ""
        artist = item.get("artist") or ""
        is_patter = item.get("kind") == "patter"
        if is_patter:
            title = "Station ID"

        # Hard-cut into a patter line, and out of it into the next track.
        # A crossfade dissolves the DJ into the music at both ends, which over
        # a ten-second link is most of the link.
        hard_cut = is_patter or self._last_push_was_patter.get(queue_id, False)

        uri = annotate_uri(item["path"], title=title, artist=artist,
                           no_crossfade=hard_cut)
        if self.ls.push(queue_id, uri):
            self._last_push_was_patter[queue_id] = is_patter
            log.info("-> %s: %s%s", queue_id,
                     f"{artist} - {title}" if title else Path(item["path"]).name,
                     "  [patter]" if is_patter else "")
        else:
            # Put it back so it is not silently lost.
            self.db.execute("UPDATE queue_items SET status='ready' WHERE id=?",
                            (item["id"],))

    # -- buffer -------------------------------------------------------------

    def maintain_buffer(self) -> None:
        if not self.queue.needs_block():
            return
        if self.library.count() == 0:
            log.debug("library empty, nothing to plan yet")
            return

        log.info("buffer down to %s, planning a new block",
                 human_duration(self.queue.ready_seconds()))
        plan = self.planner.plan_block()
        if not plan.get("items"):
            log.warning("planner produced nothing; safety playlist will cover")
            return
        self.queue.build_block(plan)

    # -- chat ---------------------------------------------------------------

    def handle_chat_requests(self) -> None:
        rows = self.db.query(
            "SELECT * FROM chat_requests WHERE status='new' ORDER BY id LIMIT 5")
        for row in rows:
            self.db.execute(
                "UPDATE chat_requests SET status='working' WHERE id=?", (row["id"],))
            try:
                note = self._handle_request(dict(row))
                status = "done"
            except Exception as exc:
                log.exception("chat request %s failed", row["id"])
                note = f"error: {exc}"
                status = "failed"
            self.db.execute(
                "UPDATE chat_requests SET status=?, note=?, handled_at=? WHERE id=?",
                (status, note[:500], time.time(), row["id"]),
            )

    def _handle_request(self, row: dict) -> str:
        intent = classify(self.llm, row["text"])
        kind = intent["kind"]
        log.info("chat (%s) from %s: %r -> %s", intent.get("source"),
                 row.get("user"), row["text"][:80], kind)
        self.db.execute("UPDATE chat_requests SET kind=? WHERE id=?",
                        (kind, row["id"]))

        if kind == "track":
            return self._handle_track_request(intent)
        if kind == "vibe":
            return self._handle_vibe_request(intent)
        if kind == "ban":
            return self._handle_ban_request(intent)
        if kind == "question":
            return self.status_line()
        return intent.get("reply") or "Got it."

    def _handle_ban_request(self, intent: dict) -> str:
        """Take a track off the station for good."""
        artist = intent.get("artist", "")
        title = intent.get("title", "")
        named = bool(artist or title)

        track = None
        if named:
            if artist and title:
                track = self.library.has_track(artist, title)
            if track is None:
                track = self.library.find(f"{artist} {title}".strip())
            if track is None:
                return (f"Could not find {(artist + ' ' + title).strip()!r} in the "
                        "library, so there is nothing to ban.")
        else:
            # "delete this" means whatever is on air right now.
            current = read_now_playing(self.cfg)
            if not current:
                return ("I do not know what is playing right now, so I cannot ban "
                        "it. Name the track and I will.")
            track = self.library.find(current)
            if track is None:
                return f"Could not match {current!r} to a track in the library."

        was_on_air = _same_track(read_now_playing(self.cfg), track)

        self.library.ban(track, reason="banned from chat",
                         banned_dir=self.cfg.path("banned"))

        # Pull it out of anything already planned but not yet played.
        dropped = self.db.execute(
            "DELETE FROM queue_items WHERE status='ready' AND track_id=?",
            (track["id"],))

        if was_on_air:
            self.ls.skip()

        log.info("banned %s - %s (dropped %d queued, on air: %s)",
                 track["artist"], track["title"], dropped, was_on_air)
        return (f"Banned {track['artist']} - {track['title']}. "
                f"{'Skipping it now. ' if was_on_air else ''}"
                "It will not be played or re-downloaded again.")

    def _handle_track_request(self, intent: dict) -> str:
        artist = intent.get("artist", "")
        title = intent.get("title", "")
        query = f"{artist} {title}".strip()

        track = self.library.has_track(artist, title) if artist and title else None
        if track is None:
            track = self.library.find(query)

        if track is None:
            if not artist:
                return (f"Could not find {title!r} in the library, and I need the "
                        "artist name to go and fetch it.")
            log.info("request not in library, downloading: %s - %s", artist, title)
            max_minutes = float(self.cfg.get("bot.max_request_minutes", 12))
            result = self.downloader.fetch(artist, title, allow_remix=True,
                                           max_seconds=max_minutes * 60)
            if not result.ok or not result.track:
                return f"Could not get {artist} - {title}: {result.reason}"
            track = result.track

        self.queue.enqueue_request(track)
        self.feed_stream()
        return f"Queued {track['artist']} - {track['title']}, up after this track."

    def _handle_vibe_request(self, intent: dict) -> str:
        mood = intent.get("mood") or intent.get("reply") or ""
        if not mood:
            return "Not sure what mood you meant."
        self.planner.set_mood(mood)

        if bool(self.cfg.get("bot.replan_on_vibe", True)):
            dropped = self.queue.clear_pending("ai")
            log.info("vibe shift: dropped %d unplayed items, re-planning", dropped)
            plan = self.planner.plan_block()
            if plan.get("items"):
                self.queue.build_block(plan)
            return (f"Shifting to: {mood}. Takes effect within a track or two "
                    f"(a couple are already queued up).")
        return f"Shifting to: {mood} from the next block."

    # -- background ---------------------------------------------------------

    def background_work(self) -> None:
        now = time.time()
        min_tracks = int(self.cfg.get("discovery.min_tracks_to_program", 25))
        bootstrapping = self.library.count() < min_tracks

        # While the library is too small to program, fetch as fast as one
        # download at a time allows; afterwards settle into the hourly cadence.
        interval = 45.0 if bootstrapping else self._download_gap()
        if now - self._last_download >= interval:
            self._last_download = now
            self._safe(self._download_one, "background download")

        if now - self._last_enrich >= self.enrich_interval:
            self._last_enrich = now
            self._safe(self.planner.enrich_tracks, "track enrichment")

        if now - self._last_housekeeping >= 3600:
            self._last_housekeeping = now
            # Picks up anything dropped into the library folder by hand.
            # Cheap: unchanged files are skipped on a size check.
            self._safe(self.library.scan, "library rescan")
            self._safe(self.queue.prune_history, "queue history prune")
            self._safe(self.tts.prune, "patter prune")

    def _download_gap(self) -> float:
        per_hour = max(1, int(self.cfg.get("discovery.downloads_per_hour", 3)))
        return max(120.0, 3600.0 / per_hour)

    def _download_one(self) -> None:
        if not self.lastfm.enabled:
            log.debug("no Last.fm key, skipping library growth")
            return
        result = self.downloader.fetch_next_candidate()
        if result is None:
            log.debug("no download candidates pending")
        elif result.ok and result.track:
            log.info("library grew to %d tracks", self.library.count())

    def _safe(self, func, label: str):
        try:
            return func()
        except Exception:
            log.exception("%s failed", label)
            return None

    # -- status -------------------------------------------------------------

    def status_line(self) -> str:
        now_playing = read_now_playing(self.cfg)
        upcoming = self.queue.preview(4)
        parts = [f"Now: {now_playing or 'unknown'}"]
        nxt = [f"{item['artist']} - {item['title']}" for item in upcoming
               if item["kind"] == "song"]
        if nxt:
            parts.append("Next: " + "; ".join(nxt[:3]))
        parts.append(f"Buffer: {human_duration(self.queue.ready_seconds())}")
        return " | ".join(parts)


def _same_track(now_playing: str, track: dict) -> bool:
    """Is this on-air string the same track? Metadata spelling varies, so match
    on the normalised title and artist rather than on an exact string."""
    if not now_playing:
        return False
    haystack = normalize(now_playing)
    title = normalize(track.get("title", ""))
    artist = normalize(track.get("artist", ""))
    return bool(title) and title in haystack and (not artist or artist in haystack)


def read_now_playing(cfg: Config) -> str:
    """Liquidsoap writes the current track here on every metadata change."""
    try:
        text = cfg.now_playing_path.read_text(encoding="utf-8", errors="replace")
        return text.strip()
    except Exception:
        return ""
