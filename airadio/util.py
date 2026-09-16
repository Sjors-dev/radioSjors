"""Small shared helpers: text normalisation, logging, filesystem safety."""

from __future__ import annotations

import logging
import logging.handlers
import re
import unicodedata
from pathlib import Path

_PAREN_JUNK = re.compile(
    r"\s*[\(\[][^\)\]]*?"
    r"(official|audio|video|lyric|lyrics|hd|hq|4k|remaster(ed)?|explicit|visualizer|mv)"
    r"[^\)\]]*[\)\]]",
    re.IGNORECASE,
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def strip_decorations(text: str) -> str:
    """Remove '(Official Video)'-style noise from a title."""
    cleaned = _PAREN_JUNK.sub("", text or "")
    cleaned = re.sub(r"\s*[\(\[]\s*[\)\]]", "", cleaned)
    return cleaned.strip(" -–—_|")


def normalize(text: str) -> str:
    """Aggressive normalisation used for de-duplication and matching."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = strip_decorations(text).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\bfeat\.?\b.*$", "", text)
    text = re.sub(r"\bft\.?\b.*$", "", text)
    text = _NON_ALNUM.sub(" ", text)
    return " ".join(text.split())


def dedupe_key(artist: str, title: str) -> str:
    return f"{normalize(artist)}|{normalize(title)}"


def safe_filename(text: str, max_length: int = 80) -> str:
    """Filesystem-safe name that still reads like the original."""
    text = strip_decorations(text)
    text = re.sub(r'[<>:"/\|?*\x00-\x1f]', "", text)
    text = text.replace("\n", " ").strip(" .")
    text = re.sub(r"\s+", " ", text)
    return (text[:max_length].strip() or "untitled")


def contains_word(haystack: str, phrase: str) -> bool:
    """Case-insensitive whole-word match.

    Substring matching is wrong for the quality filters: "mix" would match
    "remix", "live" would match "Livewire", "cover" would match "Coverdale".
    """
    if not haystack or not phrase:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in phrase.split())
    return re.search(pattern + r"(?!\w)", haystack, re.IGNORECASE) is not None


def quick_hash(path: Path, chunk: int = 1 << 20) -> str:
    """Cheap content fingerprint: file size plus a hash of the first chunk.

    Enough to spot a byte-identical re-download without reading whole files off
    a spinning disk.
    """
    import hashlib

    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            head = handle.read(chunk)
        return f"{size}-{hashlib.sha1(head).hexdigest()[:16]}"
    except Exception:
        return ""


def human_duration(seconds: float | None) -> str:
    if not seconds or seconds <= 0:
        return "0:00"
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def setup_logging(log_dir: Path, name: str, level: str = "INFO") -> logging.Logger:
    """Root logger writing to both stdout (journald picks it up) and a file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in list(root.handlers):
        root.removeHandler(handler)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)-18s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    rotating = logging.handlers.RotatingFileHandler(
        log_dir / f"{name}.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    rotating.setFormatter(fmt)
    root.addHandler(rotating)

    # These are chatty and never interesting at INFO.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("discord").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)

    return logging.getLogger(name)
