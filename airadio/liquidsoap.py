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

    def poll(self, queue_id: str) -> tuple[bool, str, int] | None:
        """Everything the feeder needs, in one connection per tick.

        Returns (connected, now playing, queue depth). None means unreachable,
        which is also how "connected" is established -- no separate ping.
        """
        responses = self.commands([f"{OUTPUT_ID}.metadata", f"{queue_id}.queue"])
        if responses is None:
            return None
        metadata, queue = (responses + ["", ""])[:2]
        depth = len([part for part in queue.split() if part.strip()])
        return True, format_metadata(parse_metadata(metadata)), depth


def parse_metadata(response: str) -> dict:
    """Pull the most recent metadata block out of an output's telnet dump.

    Liquidsoap prints blocks headed '--- 1 ---', newest first, each a series of
    key="value" lines.
    """
    current: dict = {}
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("---"):
            if current:
                break  # the first block is the newest one
            continue
        key, sep, value = line.partition("=")
        if not sep:
            continue
        current[key.strip().lower()] = value.strip().strip('"')
    return current


def format_metadata(meta: dict) -> str:
    artist = meta.get("artist", "").strip()
    title = meta.get("title", "").strip()
    if artist and title:
        return f"{artist} - {title}"
    if title:
        return title
    filename = meta.get("filename", "")
    return Path(filename).stem if filename else ""


def annotate_uri(path: str | Path, title: str = "", artist: str = "") -> str:
    """Build an annotate: URI so caster.fm shows sane now-playing metadata.

    Patter wav files carry no tags at all, so without this the stream would
    announce the filename.
    """
    path = str(Path(path).resolve())
    fields = []
    if title:
        fields.append(f'title="{_escape(title)}"')
    if artist:
        fields.append(f'artist="{_escape(artist)}"')
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
