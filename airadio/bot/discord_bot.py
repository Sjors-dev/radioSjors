"""Discord front end.

The bot never plans, downloads or renders anything itself.  It writes the
message into SQLite and the brain loop picks it up, so a wedged download can
never take the chat down with it -- and the bot can be restarted freely.
"""

from __future__ import annotations

import asyncio
import logging
import time

import discord

from ..config import Config
from ..db import Database
from ..liquidsoap import LiquidsoapClient
from ..queueing import QueueManager
from ..util import human_duration

log = logging.getLogger("bot")

HELP = """\
Just talk to me normally:
  `play bohemian rhapsody by queen`  - queues a specific track (downloads it if needed)
  `make it darker and slower`        - shifts the station's mood
  `never play this again`            - bans whatever is on air right now
  `what's playing?`                  - now playing and what's next

Commands:
  `!np`            what is on air right now
  `!queue`         the next few tracks
  `!status`        library, buffer, mood, stream health
  `!skip`          skip the current track
  `!mood`          show the current mood
  `!mood <text>`   set the mood, e.g. `!mood darker and slower`
  `!mood reset`    back to the time-of-day schedule
  `!ban`           ban whatever is playing now
  `!ban <track>`   ban a named track
  `!banned`        list what is banned
  `!help`          this

`!ban` moves the audio to `banned/` rather than deleting it, so a mistake is
undone with `main.py unban "<artist>" "<title>"`.
"""

# How long to wait for the brain to handle a message before giving up on a
# reply. The brain is single threaded, so a request can queue behind a
# whole block of patter renders, which on a slow CPU with a high-quality
# voice is several minutes. The request is never lost, only the reply.
REPLY_TIMEOUT = 420.0


class RadioBot(discord.Client):
    def __init__(self, cfg: Config):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)

        self.cfg = cfg
        self.db = Database(cfg.db_path)
        self.db.init()
        self.ls = LiquidsoapClient(
            str(cfg.get("stream.telnet_host", "127.0.0.1")),
            int(cfg.get("stream.telnet_port", 1234)),
        )
        channel = Config.env("DISCORD_CHANNEL_ID").strip()
        self.channel_id = int(channel) if channel.isdigit() else None
        self.confirm = bool(cfg.get("bot.confirm_requests", True))

    async def on_ready(self) -> None:
        log.info("connected to Discord as %s", self.user)
        if self.channel_id:
            log.info("listening in channel %s only", self.channel_id)
        else:
            log.info("listening in every channel the bot can see")

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        if self.channel_id and message.channel.id != self.channel_id:
            return

        text = (message.content or "").strip()
        if not text:
            return

        lowered = text.lower()
        if lowered in ("!help", "help", "!commands"):
            await message.channel.send(HELP)
            return
        if lowered in ("!np", "!nowplaying"):
            await message.channel.send(self._now_playing())
            return
        if lowered in ("!queue", "!next"):
            await message.channel.send(self._queue_preview())
            return
        if lowered in ("!banned", "!bans"):
            await message.channel.send(self._banned_list())
            return
        if lowered == "!status":
            await message.channel.send(self._status())
            return
        if lowered == "!skip":
            ok = self.ls.skip()
            await message.channel.send("Skipped." if ok else
                                       "Could not reach the stream engine.")
            return
        if lowered in ("!mood reset", "!mood clear", "!clearmood", "!reset"):
            # Handled here rather than sent to the classifier, which would read
            # "reset" as a vibe and set the mood to the literal word.
            self.db.set_state("current_mood", "")
            await message.channel.send(
                "Mood cleared - back to the time-of-day schedule.")
            return
        if lowered in ("!mood", "!vibe"):
            current = self.db.get_state("current_mood", "")
            await message.channel.send(
                f"Mood: {current}" if current else
                "Mood: following the time-of-day schedule.")
            return
        if lowered.startswith("!mood "):
            text = text[len("!mood "):].strip()

        request_id = self.db.execute(
            "INSERT INTO chat_requests(user, text, status, created_at) "
            "VALUES(?,?,?,?)",
            (str(message.author.display_name), text, "new", time.time()),
        )
        log.info("queued chat request %s: %r", request_id, text[:100])

        try:
            await message.add_reaction("\N{HOURGLASS WITH FLOWING SAND}")
        except Exception:
            pass

        if not self.confirm:
            return

        note = await self._await_result(request_id)
        if note:
            await message.channel.send(note[:1900])

    # -- helpers ------------------------------------------------------------

    async def _await_result(self, request_id: int) -> str:
        """Poll for the brain's answer without blocking the event loop."""
        deadline = time.time() + REPLY_TIMEOUT
        while time.time() < deadline:
            await asyncio.sleep(1.5)
            row = await asyncio.to_thread(
                self.db.one,
                "SELECT status, note FROM chat_requests WHERE id=?", (request_id,),
            )
            if row is None:
                return ""
            if row["status"] in ("done", "failed"):
                return row["note"] or "Done."
        return ("The brain is taking a while on that one - it will still go "
                "through when it finishes.")

    def _now_playing(self) -> str:
        try:
            current = self.cfg.now_playing_path.read_text(
                encoding="utf-8", errors="replace").strip()
        except Exception:
            current = ""
        return f"Now playing: {current}" if current else "Nothing reported yet."

    def _queue_preview(self) -> str:
        queue = QueueManager(self.cfg, self.db)  # read-only view
        items = queue.preview(10)
        songs = [item for item in items if item["kind"] == "song"]
        if not songs:
            return "Queue is empty - the safety playlist is covering."
        lines = []
        for item in songs[:6]:
            marker = "*" if item["tier"] == "request" else " "
            lines.append(f"{marker} {item['artist']} - {item['title']}")
        return "Coming up:\n```\n" + "\n".join(lines) + "\n```"

    def _banned_list(self) -> str:
        rows = self.db.query(
            "SELECT artist, title FROM banned ORDER BY created_at DESC LIMIT 15")
        if not rows:
            return "Nothing is banned."
        lines = [f"{row['artist']} - {row['title']}" for row in rows]
        total = self.db.one("SELECT COUNT(*) AS n FROM banned")
        header = f"Banned ({total['n'] if total else len(lines)}):"
        return header + "\n```\n" + "\n".join(lines) + "\n```"

    def _status(self) -> str:
        queue = QueueManager(self.cfg, self.db)
        tracks = self.db.one("SELECT COUNT(*) AS n FROM tracks WHERE missing=0")
        mood = self.db.get_state("current_mood", "") or "(time of day only)"
        stream = "up" if self.ls.connected else "unreachable"
        return (f"Library: {tracks['n'] if tracks else 0} tracks\n"
                f"Buffer: {human_duration(queue.ready_seconds())} ready\n"
                f"Mood: {mood}\n"
                f"Stream engine: {stream}\n"
                f"{self._now_playing()}")


def run(cfg: Config) -> None:
    token = Config.env("DISCORD_TOKEN").strip()
    if not token:
        raise SystemExit("DISCORD_TOKEN is not set in .env - the bot cannot start.")
    RadioBot(cfg).run(token, log_handler=None)
