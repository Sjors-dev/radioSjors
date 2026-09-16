"""The hourly planner.

One LLM pass per block, not per song: the model picks the running order from a
pre-filtered candidate pool and writes every patter line for the hour in one go.
That keeps the show coherent, keeps call volume inside any free tier, and means
a slow or failed call costs us one block, not one song.

If the LLM is unavailable the deterministic fallback planner takes over and the
station keeps sounding like a station.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime
from pathlib import Path

from ..config import Config
from ..db import Database
from ..library import Library
from .llm import LLM, LLMUnavailable
from .prompts import PLANNER_SYSTEM, PLANNER_USER, TAGGER_SYSTEM, TAGGER_USER

log = logging.getLogger("planner")


class Planner:
    def __init__(self, cfg: Config, db: Database, library: Library, llm: LLM):
        self.cfg = cfg
        self.db = db
        self.library = library
        self.llm = llm

    # -- mood ---------------------------------------------------------------

    def current_mood(self) -> str:
        """The listener's standing vibe instruction, if they set one."""
        return self.db.get_state("current_mood", "").strip()

    def set_mood(self, mood: str) -> None:
        self.db.set_state("current_mood", mood.strip())
        self.db.set_state("current_mood_set_at", str(time.time()))
        log.info("mood state updated: %s", mood.strip() or "(cleared)")

    # -- candidate selection ------------------------------------------------

    def select_candidates(self, energy_range: list[int], pool_size: int) -> list[dict]:
        """Tracks eligible for this hour, newest-cooldown and energy aware."""
        cooldown_hours = float(self.cfg.get("planner.repeat_cooldown_hours", 8))
        cutoff = time.time() - cooldown_hours * 3600
        low, high = (energy_range + [1, 5])[:2]

        rows = self.db.query(
            "SELECT * FROM tracks WHERE missing=0 "
            "AND (last_played_at IS NULL OR last_played_at < ?) "
            "AND (energy IS NULL OR energy BETWEEN ? AND ?) "
            "ORDER BY RANDOM() LIMIT ?",
            (cutoff, low, high, pool_size),
        )

        if len(rows) < max(8, pool_size // 4):
            # Small or heavily played library: relax the energy band first,
            # then the cooldown. Playing something twice beats silence.
            log.info("only %d candidates in energy %d-%d, relaxing filters",
                     len(rows), low, high)
            rows = self.db.query(
                "SELECT * FROM tracks WHERE missing=0 "
                "AND (last_played_at IS NULL OR last_played_at < ?) "
                "ORDER BY RANDOM() LIMIT ?",
                (cutoff, pool_size),
            )
        if not rows:
            rows = self.db.query(
                "SELECT * FROM tracks WHERE missing=0 "
                "ORDER BY COALESCE(last_played_at, 0) LIMIT ?",
                (pool_size,),
            )

        return [dict(row) for row in rows if Path(row["path"]).exists()]

    def _recent_titles(self, limit: int = 12) -> list[str]:
        rows = self.db.query(
            "SELECT artist, title FROM tracks WHERE last_played_at IS NOT NULL "
            "ORDER BY last_played_at DESC LIMIT ?", (limit,)
        )
        return [f"{row['artist']} - {row['title']}" for row in rows]

    # -- planning -----------------------------------------------------------

    def plan_block(self, now: datetime | None = None) -> dict:
        """Produce the next block. Always returns something playable."""
        now = now or datetime.now()
        profile = self.cfg.mood_for_hour(now.hour)
        block_minutes = float(self.cfg.get("planner.block_minutes", 60))
        pool_size = int(self.cfg.get("planner.candidate_pool", 60))
        min_tracks = int(self.cfg.get("discovery.min_tracks_to_program", 25))

        candidates = self.select_candidates(profile.get("energy", [1, 5]), pool_size)
        if not candidates:
            log.warning("library is empty, cannot plan a block")
            return {"items": [], "source": "empty", "mood_name": profile.get("name"),
                    "show_note": ""}

        average = _average_duration(candidates)
        track_count = max(3, min(len(candidates), round(block_minutes * 60 / average)))

        if self.llm.enabled and self.library.count() >= min_tracks:
            try:
                plan = self._plan_with_llm(now, profile, candidates, track_count)
                if plan["items"]:
                    return plan
                log.warning("LLM plan had no usable items, falling back")
            except LLMUnavailable as exc:
                log.warning("LLM planning unavailable (%s), using fallback planner", exc)
            except Exception as exc:
                log.exception("LLM planning blew up (%s), using fallback planner", exc)

        return self._plan_fallback(now, profile, candidates, track_count)

    def _plan_with_llm(self, now: datetime, profile: dict, candidates: list[dict],
                       track_count: int) -> dict:
        patter_every = int(self.cfg.get("dj.patter_every_n_tracks", 2))
        max_words = int(self.cfg.get("dj.max_patter_words", 45))
        opening = bool(self.cfg.get("dj.patter_at_block_start", True))
        mood = self.current_mood()

        listing = "\n".join(
            "{id}. {artist} - {title}{extra}".format(
                id=track["id"], artist=track["artist"], title=track["title"],
                extra=_describe(track))
            for track in candidates
        )

        system = PLANNER_SYSTEM.format(
            persona=self.cfg.get("dj.persona", "You are a radio DJ."),
            language=self.cfg.get("dj.language", "english"),
            max_words=max_words,
        )
        user = PLANNER_USER.format(
            clock=now.strftime("%H:%M"),
            day=now.strftime("%A"),
            slot_name=profile.get("name", "default"),
            slot_mood=profile.get("mood", "varied"),
            mood_override=(f"The listener has asked for: {mood}\n"
                           "Weight the picks and your tone towards that."
                           if mood else ""),
            recent=", ".join(self._recent_titles()) or "nothing yet",
            track_count=track_count,
            patter_every=patter_every,
            patter_plural="" if patter_every == 1 else "s",
            opening_note=("Start the hour with a patter line that sets the scene."
                          if opening else "Do not open with patter."),
            candidates=listing,
        )

        data = self.llm.complete_json(system, user, temperature=0.85,
                                      max_tokens=3000)
        items = self._validate(data.get("items") or [], candidates, track_count)

        return {
            "items": items,
            "source": "llm",
            "mood_name": profile.get("name"),
            "show_note": str(data.get("show_note") or "")[:300],
        }

    def _validate(self, raw_items: list, candidates: list[dict],
                  track_count: int) -> list[dict]:
        """Trust nothing: drop invented ids, repeats and malformed entries."""
        by_id = {track["id"]: track for track in candidates}
        max_words = int(self.cfg.get("dj.max_patter_words", 45))
        used: set[int] = set()
        items: list[dict] = []

        for raw in raw_items:
            if not isinstance(raw, dict):
                continue
            kind = str(raw.get("type", "")).lower()

            if kind == "song":
                try:
                    track_id = int(raw.get("id"))
                except (TypeError, ValueError):
                    log.debug("dropping song item with bad id: %r", raw)
                    continue
                track = by_id.get(track_id)
                if track is None:
                    log.info("LLM referenced unknown track id %s, dropped", track_id)
                    continue
                if track_id in used:
                    continue
                if not Path(track["path"]).exists():
                    log.warning("planned track file vanished: %s", track["path"])
                    continue
                used.add(track_id)
                items.append({"kind": "song", "track": track})

            elif kind == "patter":
                text = _clean_patter(str(raw.get("text") or ""), max_words)
                if text:
                    items.append({"kind": "patter", "text": text})

        if len(used) < max(2, track_count // 3):
            log.warning("LLM only produced %d valid songs, treating plan as failed",
                        len(used))
            return []

        # Never end a block on patter: it would leave dead air pointing nowhere.
        while items and items[-1]["kind"] == "patter":
            items.pop()
        return items

    def _plan_fallback(self, now: datetime, profile: dict, candidates: list[dict],
                       track_count: int) -> dict:
        """Deterministic programming for when the LLM is down.

        Weighted-random picks inside the energy band with artist spacing, plus
        plain template patter so the hour still sounds hosted.
        """
        spacing = int(self.cfg.get("planner.artist_spacing", 4))
        patter_every = int(self.cfg.get("dj.patter_every_n_tracks", 2))
        opening = bool(self.cfg.get("dj.patter_at_block_start", True))

        pool = list(candidates)
        random.shuffle(pool)
        # Least recently played first, with a random jitter so it is not a cycle.
        pool.sort(key=lambda t: (t.get("last_played_at") or 0) + random.uniform(0, 3600))

        chosen: list[dict] = []
        recent_artists: list[str] = []
        for track in pool:
            if len(chosen) >= track_count:
                break
            if track["artist"] in recent_artists[-spacing:]:
                continue
            chosen.append(track)
            recent_artists.append(track["artist"])

        if not chosen:
            chosen = pool[:track_count]

        items: list[dict] = []
        if opening:
            items.append({"kind": "patter",
                          "text": _template_opening(now, profile, chosen[0])})
        for index, track in enumerate(chosen):
            if index > 0 and patter_every > 0 and index % patter_every == 0:
                items.append({"kind": "patter",
                              "text": _template_link(chosen[index - 1], track)})
            items.append({"kind": "song", "track": track})

        log.info("fallback planner built %d songs for the %s slot",
                 len(chosen), profile.get("name"))
        return {"items": items, "source": "fallback",
                "mood_name": profile.get("name"),
                "show_note": "fallback programming (no LLM)"}

    # -- track enrichment ---------------------------------------------------

    def enrich_tracks(self, batch: int = 25) -> int:
        """Give untagged tracks a mood word and 1-5 energy rating via the LLM."""
        if not self.llm.enabled:
            return 0
        rows = self.db.query(
            "SELECT id, artist, title, genre, tags FROM tracks "
            "WHERE missing=0 AND energy IS NULL LIMIT ?", (batch,)
        )
        if not rows:
            return 0

        listing = "\n".join(
            "{id}. {artist} - {title} [{genre}]".format(
                id=row["id"], artist=row["artist"], title=row["title"],
                genre=(row["tags"] or row["genre"] or "unknown"))
            for row in rows
        )
        try:
            data = self.llm.complete_json(TAGGER_SYSTEM,
                                          TAGGER_USER.format(tracks=listing),
                                          temperature=0.2, max_tokens=1500)
        except LLMUnavailable as exc:
            log.info("track enrichment skipped: %s", exc)
            return 0

        valid_ids = {row["id"] for row in rows}
        updated = 0
        for entry in (data.get("tracks") or []):
            if not isinstance(entry, dict):
                continue
            try:
                track_id = int(entry.get("id"))
                energy = int(entry.get("energy"))
            except (TypeError, ValueError):
                continue
            if track_id not in valid_ids or not 1 <= energy <= 5:
                continue
            mood = str(entry.get("mood") or "").strip().lower()[:24]
            self.db.execute("UPDATE tracks SET mood=?, energy=? WHERE id=?",
                            (mood, energy, track_id))
            row = self.db.one("SELECT path FROM tracks WHERE id=?", (track_id,))
            if row:
                self.library.write_tags(Path(row["path"]), mood=mood, energy=energy)
            updated += 1

        log.info("enriched %d/%d tracks with mood and energy", updated, len(rows))
        return updated


# -- helpers ----------------------------------------------------------------


def _average_duration(tracks: list[dict]) -> float:
    durations = [t["duration"] for t in tracks if (t.get("duration") or 0) > 30]
    return (sum(durations) / len(durations)) if durations else 240.0


def _describe(track: dict) -> str:
    bits = []
    if track.get("tags"):
        bits.append(str(track["tags"])[:60])
    elif track.get("genre"):
        bits.append(str(track["genre"])[:40])
    if track.get("mood"):
        bits.append(str(track["mood"]))
    if track.get("energy"):
        bits.append(f"energy {track['energy']}")
    return f"  ({'; '.join(bits)})" if bits else ""


def _clean_patter(text: str, max_words: int) -> str:
    """Strip anything that would sound wrong coming out of a TTS voice."""
    text = text.replace("*", "").replace("#", "").replace("`", "")
    text = text.replace("\n", " ").strip().strip('"')
    text = " ".join(text.split())
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words + 15:
        text = " ".join(words[:max_words]).rstrip(",;: ") + "."
    return text


def _template_opening(now: datetime, profile: dict, first: dict) -> str:
    hour = now.hour
    if hour < 5:
        greeting = "Still up, then. Good."
    elif hour < 12:
        greeting = "Morning."
    elif hour < 18:
        greeting = "Afternoon."
    else:
        greeting = "Evening."
    return (f"{greeting} You're listening to the usual. "
            f"Starting off with {first['artist']}, {first['title']}.")


def _template_link(previous: dict, upcoming: dict) -> str:
    options = [
        f"That was {previous['artist']}. Next up, {upcoming['artist']} "
        f"with {upcoming['title']}.",
        f"{previous['title']} there, from {previous['artist']}. "
        f"Staying with it: {upcoming['artist']}.",
        f"Coming up, {upcoming['title']} by {upcoming['artist']}.",
    ]
    return random.choice(options)
