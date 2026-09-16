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
        try:
            with socket.create_connection((self.host, self.port), self.timeout) as sock:
                sock.settimeout(self.timeout)
                sock.sendall((text + "\n").encode("utf-8"))

                chunks: list[bytes] = []
                while True:
                    data = sock.recv(4096)
                    if not data:
                        break
                    chunks.append(data)
                    if b"\nEND\r\n" in b"".join(chunks) or b"\nEND\n" in b"".join(chunks):
                        break

                raw = b"".join(chunks).decode("utf-8", "replace")
                lines = [line.strip() for line in raw.replace("\r", "").split("\n")]
                return "\n".join(line for line in lines if line and line != "END")
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
