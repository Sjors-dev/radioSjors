"""Music library: scan the folder, read/write tags, keep SQLite in sync."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from mutagen import File as MutagenFile
from mutagen.easyid3 import EasyID3
from mutagen.id3 import ID3NoHeaderError
from mutagen.mp3 import MP3

from .db import Database
from .util import dedupe_key, normalize, quick_hash

log = logging.getLogger("library")

AUDIO_SUFFIXES = {".mp3", ".m4a", ".ogg", ".opus", ".flac", ".wav"}

# Custom free-text tags we own.  Stored in ID3 TXXX frames via EasyID3.
for _key in ("mood", "energy", "airadio_tags"):
    try:
        EasyID3.RegisterTXXXKey(_key, _key)
    except Exception:  # pragma: no cover - registration is idempotent enough
        pass


class Library:
    def __init__(self, db: Database, library_dir: Path):
        self.db = db
        self.dir = Path(library_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    # -- tag helpers --------------------------------------------------------

    @staticmethod
    def read_tags(path: Path) -> dict:
        """Best-effort metadata for any audio file. Falls back to the filename."""
        info: dict = {
            "title": "",
            "artist": "",
            "album": "",
            "genre": "",
            "mood": "",
            "energy": None,
            "tags": "",
            "duration": 0.0,
        }
        try:
            audio = MutagenFile(path, easy=True)
        except Exception as exc:
            log.warning("unreadable audio file %s: %s", path.name, exc)
            audio = None

        if audio is not None:
            def first(key: str) -> str:
                value = audio.tags.get(key) if audio.tags else None
                if isinstance(value, list) and value:
                    return str(value[0])
                return str(value) if value else ""

            info["title"] = first("title")
            info["artist"] = first("artist")
            info["album"] = first("album")
            info["genre"] = first("genre")
            info["mood"] = first("mood")
            info["tags"] = first("airadio_tags")
            energy = first("energy")
            if energy.strip().isdigit():
                info["energy"] = int(energy.strip())
            if getattr(audio, "info", None) is not None:
                info["duration"] = float(getattr(audio.info, "length", 0.0) or 0.0)

        if not info["title"] or not info["artist"]:
            # "Artist - Title.mp3" is what the downloader writes.
            stem = path.stem
            if " - " in stem:
                artist, _, title = stem.partition(" - ")
            else:
                artist, title = "Unknown Artist", stem
            info["artist"] = info["artist"] or artist.strip()
            info["title"] = info["title"] or title.strip()

        return info

    @staticmethod
    def write_tags(path: Path, **fields) -> None:
        """Write tags onto an mp3. Silently skips non-mp3 formats."""
        if path.suffix.lower() != ".mp3":
            return
        try:
            try:
                audio = EasyID3(path)
            except ID3NoHeaderError:
                mp3 = MP3(path)
                mp3.add_tags()
                mp3.save()
                audio = EasyID3(path)
            for key, value in fields.items():
                if value in (None, ""):
                    continue
                audio[key] = str(value)
            audio.save()
        except Exception as exc:
            log.warning("could not tag %s: %s", path.name, exc)

    # -- scanning -----------------------------------------------------------

    def scan(self, prune: bool = True) -> dict:
        """Index every audio file under the library dir. Returns a summary."""
        added = updated = skipped = unchanged = 0
        seen_paths: set[str] = set()
        seen_keys: dict[str, str] = {}
        seen_hashes: dict[str, str] = {}

        for path in sorted(self.dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
                continue
            rel = str(path.resolve())
            seen_paths.add(rel)

            # Fast path for files we already know: if the size still matches the
            # one baked into the stored fingerprint, nothing has changed, so
            # skip reading tags and hashing. Keeps a periodic rescan of a
            # thousand-track library off a spinning disk essentially free.
            known = self.db.one(
                "SELECT id, dedupe_key, content_hash FROM tracks WHERE path = ?",
                (rel,))
            if known and known["content_hash"]:
                try:
                    same_size = known["content_hash"].startswith(
                        f"{path.stat().st_size}-")
                except OSError:
                    same_size = False
                if same_size:
                    seen_keys.setdefault(known["dedupe_key"], rel)
                    seen_hashes.setdefault(known["content_hash"], rel)
                    unchanged += 1
                    continue

            info = self.read_tags(path)
            key = dedupe_key(info["artist"], info["title"])
            digest = quick_hash(path)

            # Two ways of being a duplicate: same artist+title after
            # normalisation, or byte-identical (a re-download under a slightly
            # different name, which tag-based matching would miss).
            existing_path = seen_keys.get(key)
            if existing_path and existing_path != rel:
                log.info("duplicate skipped: %s (already have %s)", path.name,
                         Path(existing_path).name)
                skipped += 1
                continue
            if digest:
                twin = seen_hashes.get(digest)
                if twin and twin != rel:
                    log.info("identical file skipped: %s (same bytes as %s)",
                             path.name, Path(twin).name)
                    skipped += 1
                    continue
                seen_hashes[digest] = rel
            seen_keys[key] = rel

            row = known
            if row:
                self.db.execute(
                    "UPDATE tracks SET title=?, artist=?, album=?, genre=?, tags=?, "
                    "mood=?, energy=COALESCE(?, energy), duration=?, dedupe_key=?, "
                    "content_hash=?, missing=0 WHERE id=?",
                    (info["title"], info["artist"], info["album"], info["genre"],
                     info["tags"], info["mood"], info["energy"], info["duration"],
                     key, digest, row["id"]),
                )
                updated += 1
            else:
                dup = self.db.one(
                    "SELECT id, path FROM tracks WHERE (dedupe_key=? OR "
                    "(content_hash IS NOT NULL AND content_hash=?)) AND missing=0",
                    (key, digest or "\x00"),
                )
                if dup and Path(dup["path"]).exists():
                    log.info("duplicate of existing track, skipping: %s", path.name)
                    skipped += 1
                    continue
                self.db.execute(
                    "INSERT INTO tracks(path, dedupe_key, content_hash, title, artist, "
                    "album, genre, tags, mood, energy, duration, added_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (rel, key, digest, info["title"], info["artist"], info["album"],
                     info["genre"], info["tags"], info["mood"], info["energy"],
                     info["duration"], time.time()),
                )
                added += 1

        pruned = 0
        if prune:
            for row in self.db.query("SELECT id, path FROM tracks WHERE missing=0"):
                if row["path"] not in seen_paths and not Path(row["path"]).exists():
                    self.db.execute("UPDATE tracks SET missing=1 WHERE id=?", (row["id"],))
                    pruned += 1

        summary = {"added": added, "updated": updated, "unchanged": unchanged,
                   "skipped": skipped, "pruned": pruned, "total": self.count()}
        log.info("scan complete: %s", summary)
        return summary

    # -- queries ------------------------------------------------------------

    def count(self) -> int:
        row = self.db.one("SELECT COUNT(*) AS n FROM tracks WHERE missing=0")
        return int(row["n"]) if row else 0

    def has_track(self, artist: str, title: str) -> dict | None:
        row = self.db.one(
            "SELECT * FROM tracks WHERE dedupe_key=? AND missing=0 LIMIT 1",
            (dedupe_key(artist, title),),
        )
        return dict(row) if row else None

    def find(self, text: str) -> dict | None:
        """Loose lookup for '<artist> <title>' free text coming from chat."""
        needle = normalize(text)
        if not needle:
            return None
        best, best_score = None, 0.0
        for row in self.db.query(
            "SELECT * FROM tracks WHERE missing=0 ORDER BY id DESC LIMIT 5000"
        ):
            haystack = normalize(row["artist"] + " " + row["title"])
            score = _overlap(needle, haystack)
            if score > best_score:
                best, best_score = row, score
        # Require most of the query's words to be present before claiming a hit.
        return dict(best) if best is not None and best_score >= 0.8 else None

    def mark_played(self, track_id: int) -> None:
        self.db.execute(
            "UPDATE tracks SET play_count = play_count + 1, last_played_at = ? "
            "WHERE id = ?",
            (time.time(), track_id),
        )


def _overlap(needle: str, haystack: str) -> float:
    words = needle.split()
    if not words:
        return 0.0
    hits = sum(1 for word in words if word in haystack)
    return hits / len(words)
