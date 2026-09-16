"""Talk to the running liquidsoap over its local telnet control socket.

The stream never pulls from Python; Python pushes into liquidsoap's own request
queues.  That keeps the audio path entirely inside liquidsoap, so a crashed or
wedged brain cannot stall playback -- liquidsoap just drains its queue and then
drops to the safety playlist.
"""

from __future__ import annotations

import logging
import socket
from pathlib import Path

log = logging.getLogger("liquidsoap")

AI_QUEUE = "aiqueue"
# Not "requests": liquidsoap already owns the request.* telnet namespace.
REQUEST_QUEUE = "reqqueue"
OUTPUT_ID = "caster"


class LiquidsoapClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 1234,
                 timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    # -- transport ----------------------------------------------------------

    def command(self, text: str) -> str | None:
        """Run one telnet command. Returns the response, or None if unreachable."""
        results = self.commands([text])
        return results[0] if results else None

    def commands(self, texts: list[str]) -> list[str] | None:
        """Run several commands down one connection.

        Liquidsoap logs a line per telnet connect and disconnect, so opening a
        socket per command turns a 5-second poll into a flooded journal.  One
        connection per tick, closed politely with `quit`, keeps the stream log
        readable.

        Returns one response per command, or None if liquidsoap is unreachable.
        """
        if not texts:
            return []
        try:
            with socket.create_connection((self.host, self.port), self.timeout) as sock:
                sock.settimeout(self.timeout)
                sock.sendall("".join(f"{text}\n" for text in texts).encode("utf-8"))

                buffer = ""
                responses: list[str] = []
                while len(responses) < len(texts):
                    data = sock.recv(4096)
                    if not data:
                        break
                    buffer += data.decode("utf-8", "replace").replace("\r", "")
                    buffer = _drain(buffer, responses)

                try:
                    # Without this liquidsoap logs "disconnected without saying
                    # goodbye" on every single poll.
                    sock.sendall(b"quit\n")
                except OSError:
                    pass

                while len(responses) < len(texts):
                    responses.append("")
                return responses
        except (socket.timeout, OSError) as exc:
            log.debug("liquidsoap telnet %s:%s unreachable (%s)",
                      self.host, self.port, exc)
            return None

    @property
    def connected(self) -> bool:
        return self.command("uptime") is not None

    # -- queues -------------------------------------------------------------

    def push(self, queue_id: str, uri: str) -> bool:
        response = self.command(f"{queue_id}.push {uri}")
        if response is None:
            return False
        if "ERROR" in response.upper():
            log.warning("liquidsoap rejected push: %s", response)
            return False
        return True

    def queue_length(self, queue_id: str) -> int:
        """How many requests are still sitting in that liquidsoap queue."""
        response = self.command(f"{queue_id}.queue")
        if response is None:
            return -1
        return len([part for part in response.split() if part.strip()])

    def skip(self) -> bool:
        return self.command(f"{OUTPUT_ID}.skip") is not None

    def now_playing(self) -> str:
        """Current track as 'Artist - Title', read back from the output.

        Parsed from telnet rather than written by the .liq script, so the
        liquidsoap file stays minimal and version-portable.
        """
        response = self.command(f"{OUTPUT_ID}.metadata")
        if not response:
            return ""
        return format_metadata(parse_metadata(response))

    def poll(self, *queue_ids: str) -> dict | None:
        """Everything the feeder needs, in one connection per tick.

        Returns {"now_playing": str, "depths": {queue_id: int}}, or None if
        liquidsoap is unreachable -- which is also how "connected" is
        established, so there is no separate ping.
        """
        queue_ids = queue_ids or (AI_QUEUE, REQUEST_QUEUE)
        responses = self.commands(
            [f"{OUTPUT_ID}.metadata"] + [f"{q}.queue" for q in queue_ids])
        if responses is None:
            return None

        padded = responses + [""] * (1 + len(queue_ids) - len(responses))
        depths = {
            queue_id: len([part for part in padded[index + 1].split() if part.strip()])
            for index, queue_id in enumerate(queue_ids)
        }
        return {"now_playing": format_metadata(parse_metadata(padded[0])),
                "depths": depths}


def parse_metadata(response: str) -> dict:
    """Pull the most recent metadata block out of an output's telnet dump.

    Liquidsoap prints a history of blocks headed '--- 1 ---', '--- 2 ---', in
    the order they went to air -- so the LAST block is what is playing now.
    Reading the first one instead pins now-playing to whatever was on when the
    stream started.
    """
    blocks: list[dict] = []
    current: dict = {}
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("---"):
            if current:
                blocks.append(current)
                current = {}
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        current[key.strip().lower()] = value.strip().strip('"')
    if current:
        blocks.append(current)
    return blocks[-1] if blocks else {}


def format_metadata(meta: dict) -> str:
    artist = meta.get("artist", "").strip()
    title = meta.get("title", "").strip()
    if artist and title:
        return f"{artist} - {title}"
    if title:
        return title
    filename = meta.get("filename", "")
    return Path(filename).stem if filename else ""


def annotate_uri(path: str | Path, title: str = "", artist: str = "",
                 no_crossfade: bool = False) -> str:
    """Build an annotate: URI so caster.fm shows sane now-playing metadata.

    Patter wav files carry no tags at all, so without this the stream would
    announce the filename.

    `no_crossfade` sets liquidsoap's per-track crossfade overrides, which is how
    a spoken link gets a hard cut while music keeps its blend. A two-second
    crossfade over a ten-second patter line spends a fifth of the DJ dissolved
    into the song.
    """
    path = str(Path(path).resolve())
    fields = []
    if title:
        fields.append(f'title="{_escape(title)}"')
    if artist:
        fields.append(f'artist="{_escape(artist)}"')
    if no_crossfade:
        fields.append('liq_cross_duration="0."')
        fields.append('liq_fade_in="0."')
        fields.append('liq_fade_out="0."')
    if not fields:
        return path
    return f"annotate:{','.join(fields)}:{path}"


def _escape(value: str) -> str:
    # Liquidsoap annotate values are double-quoted; keep it simple and safe.
    return (value.replace("\\", "")
                 .replace('"', "'")
                 .replace("\n", " ")
                 .replace(":", " -")
                 .strip())


def _drain(buffer: str, responses: list[str]) -> str:
    """Pull every complete response out of the buffer, leaving the remainder.

    Liquidsoap terminates each command's reply with a line containing just
    "END", so responses are split on that rather than on connection close.
    """
    while True:
        marker = _find_end(buffer)
        if marker is None:
            return buffer
        start, stop = marker
        body = buffer[:start]
        lines = [line.strip() for line in body.split("\n")]
        responses.append("\n".join(line for line in lines if line))
        buffer = buffer[stop:]


def _find_end(buffer: str) -> tuple[int, int] | None:
    """Locate a whole-line 'END' terminator: (body end, next response start)."""
    position = 0
    while True:
        index = buffer.find("END\n", position)
        if index == -1:
            return None
        if index == 0 or buffer[index - 1] == "\n":
            return index, index + 4
        position = index + 4
