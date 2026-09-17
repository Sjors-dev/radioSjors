"""Queue manager: turns a planned block into ready-to-play files.

Nothing reaches the stream until it exists on disk.  Patter is rendered here;
if a render fails the line is dropped and the music carries on, because the one
thing that must never happen is a gap.
"""

from __future__ import annotations

import json
import logging
import time
import wave
from pathlib import Path

from .config import Config
from .db import Database
from .library import Library
from .tts import TTS
from .util import human_duration

log = logging.getLogger("queue")


class QueueManager:
    def __init__(self, cfg: Config, db: Database,
                 library: Library | None = None, tts: TTS | None = None):
        # library and tts are only needed to build blocks; read-only callers
        # (the Discord bot, the status CLI) can leave them out.
        self.cfg = cfg
        self.db = db
        self.library = library
        self.tts = tts

    # -- measurements -------------------------------------------------------

    def ready_seconds(self) -> float:
        """Audio rendered and not yet played, in seconds.

        Only counts items still waiting: 'ready' (held here) plus 'pushed'
        (handed to liquidsoap but not yet played). Items liquidsoap has
        finished with are reconciled to 'done' by the feeder -- without that
        this figure grows forever and the planner stops building blocks.
        """
        row = self.db.one(
            "SELECT COALESCE(SUM(duration), 0) AS total FROM queue_items "
            "WHERE status IN ('ready', 'pushed', 'pending')"
        )
        return float(row["total"]) if row else 0.0

    def reconcile_pushed(self, tier: str, still_queued: int) -> int:
        """Mark pushed items liquidsoap has actually finished playing as done.

        Liquidsoap owns playback, so the only way to know what it has finished
        is to compare what we pushed against how deep its queue still is: the
        `still_queued` most recently pushed items are still ahead of the
        speakers, everything pushed before that has actually been on air.

        This is also where play_count and last_played_at get stamped, not at
        push time. Liquidsoap can hold several items in its own buffer ahead
        of whatever is actually audible (stream.liquidsoap_queue_depth), so
        stamping "played" the moment a track was merely handed off made a
        track look like it had already aired minutes before anyone heard it
        -- while it was still sitting in the upcoming queue.
        """
        rows = self.db.query(
            "SELECT id, track_id FROM queue_items WHERE status='pushed' "
            "AND tier=? ORDER BY seq", (tier,))
        cutoff = max(0, len(rows) - max(0, still_queued))
        finished = rows[:cutoff]
        if not finished:
            return 0

        ids = [row["id"] for row in finished]
        placeholders = ",".join("?" * len(ids))
        changed = self.db.execute(
            f"UPDATE queue_items SET status='done' WHERE id IN ({placeholders})",
            ids)
        for row in finished:
            if row["track_id"]:
                self.library.mark_played(row["track_id"])
        return changed

    def ready_count(self) -> int:
        row = self.db.one(
            "SELECT COUNT(*) AS n FROM queue_items WHERE status = 'ready'")
        return int(row["n"]) if row else 0

    def needs_block(self) -> bool:
        buffer_minutes = float(self.cfg.get("planner.buffer_minutes", 60))
        return self.ready_seconds() < buffer_minutes * 60

    def _next_seq(self) -> int:
        row = self.db.one("SELECT COALESCE(MAX(seq), 0) AS top FROM queue_items")
        return int(row["top"]) + 1 if row else 1

    # -- building -----------------------------------------------------------

    def build_block(self, plan: dict) -> int:
        """Render and enqueue a planned block. Returns the number of items queued."""
        items = plan.get("items") or []
        if not items:
            return 0

        block_id = self.db.execute(
            "INSERT INTO blocks(created_at, hour, mood_name, mood_note, source, "
            "item_count) VALUES(?,?,?,?,?,?)",
            (time.time(), time.localtime().tm_hour, plan.get("mood_name"),
             plan.get("show_note"), plan.get("source"), len(items)),
        )

        seq = self._next_seq()
        queued = 0
        songs = 0
        patter_ok = 0

        for item in items:
            if item["kind"] == "song":
                track = item["track"]
                path = Path(track["path"])
                if not path.exists():
                    log.warning("skipping missing file: %s", path)
                    continue
                self.db.execute(
                    "INSERT INTO queue_items(block_id, seq, kind, tier, path, "
                    "track_id, title, artist, duration, status, created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (block_id, seq, "song", "ai", str(path), track["id"],
                     track["title"], track["artist"], track.get("duration") or 0.0,
                     "ready", time.time()),
                )
                seq += 1
                queued += 1
                songs += 1

            elif item["kind"] in ("patter", "banter"):
                # Recorded as 'pending' and rendered later, one item at a time,
                # by the brain loop. Rendering a whole hour of patter inline
                # blocks everything else -- chat requests included -- for as
                # long as the voice takes, which on a slow CPU is minutes.
                #
                # A conversation is stored as one row holding every turn, so it
                # reaches the stream as a single push. Split across rows, a
                # song could land in the middle of it.
                if item["kind"] == "banter":
                    text = json.dumps(item["lines"], ensure_ascii=False)
                    host = " & ".join(
                        dict.fromkeys(line["host"] for line in item["lines"]))
                else:
                    text = item["text"]
                    host = item.get("host") or ""
                self.db.execute(
                    "INSERT INTO queue_items(block_id, seq, kind, tier, text, "
                    "host, title, artist, duration, status, created_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (block_id, seq, item["kind"], "ai", text, host,
                     "", self.cfg.get("station.name", "Radio"),
                     0.0, "pending", time.time()),
                )
                seq += 1
                queued += 1
                patter_ok += 1

        log.info("block %d queued: %d songs, %d patter lines awaiting render, "
                 "source=%s, buffer now %s",
                 block_id, songs, patter_ok, plan.get("source"),
                 human_duration(self.ready_seconds()))
        return queued

    def render_pending(self, limit: int = 1) -> int:
        """Render the next queued patter line(s). Called once per tick.

        Earliest first, because the front of the queue is what liquidsoap needs
        next. A line that will not render is marked failed and dropped from the
        running order rather than blocking it.
        """
        if self.tts is None:
            return 0
        rows = self.db.query(
            "SELECT * FROM queue_items WHERE status='pending' ORDER BY seq LIMIT ?",
            (limit,))
        rendered = 0
        for row in rows:
            hint = f"b{row['block_id']}"
            if row["kind"] == "banter":
                audio = self.tts.render_exchange(_turns(row["text"]),
                                                 name_hint=hint)
            else:
                audio = self.tts.render(row["text"] or "", name_hint=hint,
                                        host=row["host"])
            if audio is None:
                self.db.execute(
                    "UPDATE queue_items SET status='failed' WHERE id=?", (row["id"],))
                log.info("patter dropped from the running order (render failed): %.50s",
                         row["text"] or "")
                continue
            self.db.execute(
                "UPDATE queue_items SET status='ready', path=?, duration=? WHERE id=?",
                (str(audio), wav_duration(audio), row["id"]))
            rendered += 1
        return rendered

    def pending_count(self) -> int:
        row = self.db.one(
            "SELECT COUNT(*) AS n FROM queue_items WHERE status='pending'")
        return int(row["n"]) if row else 0

    # -- requests -----------------------------------------------------------

    def enqueue_request(self, track: dict) -> int:
        """Top-priority item. Plays after the current track finishes."""
        item_id = self.db.execute(
            "INSERT INTO queue_items(seq, kind, tier, path, track_id, title, artist, "
            "duration, status, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (self._next_seq(), "song", "request", track["path"], track["id"],
             track["title"], track["artist"], track.get("duration") or 0.0,
             "ready", time.time()),
        )
        log.info("request queued: %s - %s", track["artist"], track["title"])
        return item_id

    # -- consumption --------------------------------------------------------

    def take_next(self, tier: str, limit: int = 1) -> list[dict]:
        """Claim the next ready items of a tier and mark them pushed."""
        rows = self.db.query(
            "SELECT * FROM queue_items WHERE status IN ('ready','pending') "
            "AND tier=? ORDER BY seq LIMIT ?", (tier, limit),
        )
        claimed = []
        for row in rows:
            if row["status"] != "ready":
                # The next item is still being voiced. Wait for it rather than
                # playing the song that was meant to come after the link.
                break
            path = Path(row["path"] or "")
            if not path.exists():
                log.warning("queued file vanished, dropping: %s", row["path"])
                self.db.execute("UPDATE queue_items SET status='failed' WHERE id=?",
                                (row["id"],))
                continue
            changed = self.db.execute(
                "UPDATE queue_items SET status='pushed', pushed_at=? "
                "WHERE id=? AND status='ready'", (time.time(), row["id"]),
            )
            # play_count / last_played_at are stamped later, in
            # reconcile_pushed, once liquidsoap's own queue depth confirms
            # this has actually aired -- not here, when it has only just been
            # handed off and may still be minutes from the speakers.
            if changed:
                claimed.append(dict(row))
        return claimed

    def preview(self, limit: int = 8, include_pending: bool = False) -> list[dict]:
        """What is coming up.

        `include_pending` adds items that are queued but not yet voiced. The
        stream feeder must not see those -- it would push a file that does not
        exist yet -- but a display wants them, because they are genuinely next.
        """
        statuses = ("'ready','pushed','pending'" if include_pending
                    else "'ready','pushed'")
        rows = self.db.query(
            "SELECT kind, tier, title, artist, text, host, duration FROM queue_items "
            f"WHERE status IN ({statuses}) ORDER BY "
            "CASE tier WHEN 'request' THEN 0 ELSE 1 END, seq LIMIT ?", (limit,)
        )
        return [dict(row) for row in rows]

    def clear_pending(self, tier: str | None = None) -> int:
        """Drop not-yet-played items, e.g. after an immediate re-plan."""
        if tier:
            return self.db.execute(
                "DELETE FROM queue_items WHERE status='ready' AND tier=?", (tier,))
        return self.db.execute("DELETE FROM queue_items WHERE status='ready'")

    def prune_history(self, keep: int = 500) -> int:
        """Keep the table small; played items are only kept for the log."""
        row = self.db.one("SELECT COUNT(*) AS n FROM queue_items WHERE status='done'")
        if not row or row["n"] <= keep:
            return 0
        return self.db.execute(
            "DELETE FROM queue_items WHERE status='done' AND id NOT IN "
            "(SELECT id FROM queue_items WHERE status='done' ORDER BY id DESC LIMIT ?)",
            (keep,),
        )


def _turns(raw: str | None) -> list[dict]:
    """Decode a stored conversation. A malformed row is simply not spoken."""
    try:
        data = json.loads(raw or "[]")
    except (ValueError, TypeError):
        return []
    return [turn for turn in data if isinstance(turn, dict)]


def wav_duration(path: Path) -> float:
    """Length of a wav file in seconds, 0.0 if it cannot be read."""
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or 1
            return frames / float(rate)
    except Exception:
        return 0.0
