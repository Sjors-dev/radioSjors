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
from datetime import datetime, timedelta
from pathlib import Path

from ..config import Config
from ..db import Database
from ..library import Library
from ..news import News
from ..util import human_duration, normalize
from ..weather import Weather
from .llm import LLM, LLMUnavailable
from .prompts import PLANNER_SYSTEM, PLANNER_USER, TAGGER_SYSTEM, TAGGER_USER

log = logging.getLogger("planner")


class Planner:
    def __init__(self, cfg: Config, db: Database, library: Library, llm: LLM,
                 weather: Weather | None = None, news: News | None = None):
        self.cfg = cfg
        self.db = db
        self.library = library
        self.llm = llm
        self.weather = weather
        self.news = news

    # -- mood ---------------------------------------------------------------

    def current_mood(self) -> str:
        """The listener's standing vibe instruction, if they set one."""
        return self.db.get_state("current_mood", "").strip()

    def set_mood(self, mood: str) -> None:
        self.db.set_state("current_mood", mood.strip())
        self.db.set_state("current_mood_set_at", str(time.time()))
        log.info("mood state updated: %s", mood.strip() or "(cleared)")

    # -- candidate selection ------------------------------------------------

    def artists_named_in(self, text: str) -> list[str]:
        """Library artists the listener mentioned, if any.

        "something moodier" names nobody; "in the mood for some Westside Gunn"
        names one, and that has to change which tracks are offered, not just
        what the prompt says.
        """
        needle = normalize(text or "")
        if not needle:
            return []
        found = []
        for row in self.db.query(
                "SELECT DISTINCT artist FROM tracks WHERE missing=0"):
            name = normalize(row["artist"] or "")
            # Short names ("Air", "Sor") match far too much inside a sentence.
            if len(name) >= 5 and name in needle:
                found.append(row["artist"])
        if found:
            log.info("mood names library artist(s): %s", ", ".join(found))
        return found

    def select_candidates(self, energy_range: list[int], pool_size: int,
                          focus_artists: list[str] | None = None,
                          exemplar_artists: list[str] | None = None,
                          genres: list[str] | None = None) -> list[dict]:
        """Tracks eligible for this hour, newest-cooldown and energy aware.

        Two ways an artist can be named, and they mean different things:

        `focus_artists` come from the listener asking for someone right now
        ("in the mood for some Westside Gunn"). Build the hour around them.

        `exemplar_artists` come from the time-of-day profile ("Soft Rock, think
        Smashing Pumpkins, Radiohead"). Those are illustrations of a genre, not
        a running order -- make sure a few are available to pick, no more.

        Either way they have to reach the pool. An instruction to lean towards
        an artist is worthless if the model is only shown two of their tracks.

        `genres` (from the mood profile -- an entry's own `genres:`, or
        Config's planner.default_genres fallback by slot name, see
        Config._with_default_genres) narrows the pool to tracks tagged
        anything close to it before energy is even considered. Energy alone
        cannot tell a soft rock hour from a jazz one that happens to sit at
        the same tempo -- without this, the model is just handed a random
        slice of the whole library and has to guess at genre from the mood
        text alone, which is exactly how a jazz or Christmas track ends up in
        a rock hour once the library has any of either in it.
        """
        cooldown_hours = float(self.cfg.get("planner.repeat_cooldown_hours", 8))
        cutoff = time.time() - cooldown_hours * 3600
        low, high = (energy_range + [1, 5])[:2]

        rows: list = []
        if genres:
            rows = self._query_by_genre(genres, cutoff, low, high, pool_size)
            if len(rows) < max(8, pool_size // 4):
                # Not enough of the library is tagged anything close to this
                # genre yet. An hour that ignores the genre beats one with
                # only three songs in it.
                log.info("only %d candidates match genre %s, widening past "
                         "genre for this hour", len(rows), ", ".join(genres))
                rows = []

        if not rows:
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

        tracks = [dict(row) for row in rows if Path(row["path"]).exists()]
        pool = self._diversify(tracks, pool_size)

        if exemplar_artists:
            pool = self._add_focus(
                pool, exemplar_artists, pool_size,
                want=int(self.cfg.get("planner.exemplar_artist_tracks", 3)),
                label="exemplars")
        if focus_artists:
            pool = self._add_focus(
                pool, focus_artists, pool_size,
                want=int(self.cfg.get("planner.focus_artist_tracks", 10)),
                label="focus")
        return pool

    def _query_by_genre(self, genres: list[str], cutoff: float, low: int,
                        high: int, pool_size: int) -> list:
        """Tracks whose stored genre/tags look like any of `genres`.

        Substring matching against whatever Last.fm actually returned at
        download time -- there is no controlled vocabulary here, so this is
        deliberately loose (a genre list of "rock" matches a tag of
        "alternative rock" or "90s rock" too) rather than trying to be exact.
        """
        conditions = []
        params: list = [cutoff, low, high]
        for genre in genres:
            needle = str(genre).strip().lower()
            if not needle:
                continue
            conditions.append(
                "(LOWER(COALESCE(tags,'')) LIKE ? OR LOWER(COALESCE(genre,'')) LIKE ?)")
            params.extend([f"%{needle}%", f"%{needle}%"])
        if not conditions:
            return []
        params.append(pool_size)
        return self.db.query(
            "SELECT * FROM tracks WHERE missing=0 "
            "AND (last_played_at IS NULL OR last_played_at < ?) "
            "AND (energy IS NULL OR energy BETWEEN ? AND ?) "
            f"AND ({' OR '.join(conditions)}) "
            "ORDER BY RANDOM() LIMIT ?",
            params,
        )

    def _add_focus(self, pool: list[dict], focus_artists: list[str],
                   pool_size: int, want: int = 10,
                   label: str = "focus") -> list[dict]:
        """Put plenty of the named artist in front of the model.

        Ignores the energy band and the play cooldown: an explicit request for
        an artist outranks the time-of-day profile, and refusing because they
        were played two hours ago is not what the listener meant.
        """
        have = {track["id"] for track in pool}
        added = 0
        for artist in focus_artists:
            for row in self.db.query(
                    "SELECT * FROM tracks WHERE missing=0 AND artist=? "
                    "COLLATE NOCASE ORDER BY COALESCE(last_played_at, 0) LIMIT ?",
                    (artist, want)):
                if row["id"] in have or not Path(row["path"]).exists():
                    continue
                have.add(row["id"])
                pool.insert(0, dict(row))
                added += 1
        if added:
            log.info("%s %s: %d extra tracks in a pool of %d",
                     label, ", ".join(focus_artists), added, len(pool))
        # Trim from the far end so the focus tracks survive the cap.
        return pool[:max(pool_size, added + 20)]

    def _diversify(self, tracks: list[dict], pool_size: int) -> list[dict]:
        """Cap how many tracks per artist reach the LLM.

        A random pool from a library dominated by a few artists hands the model
        eight tracks by one of them, and it dutifully programmes them together.
        Round-robin by artist instead, so the menu itself is spread out.
        """
        by_artist: dict[str, list[dict]] = {}
        for track in tracks:
            by_artist.setdefault(track.get("artist", ""), []).append(track)

        if len(by_artist) < 2:
            return tracks

        cap = int(self.cfg.get("planner.max_per_artist_in_pool", 0) or 0)
        if cap <= 0:
            # Enough of each artist to give the model choice, never enough to
            # fill the hour from one of them.
            cap = max(2, pool_size // max(4, len(by_artist)))

        ordered: list[dict] = []
        for round_index in range(cap):
            for artist_tracks in by_artist.values():
                if round_index < len(artist_tracks):
                    ordered.append(artist_tracks[round_index])
        log.debug("candidate pool: %d tracks across %d artists (max %d each)",
                  len(ordered), len(by_artist), cap)
        return ordered[:pool_size]

    def _recent_titles(self, limit: int = 12) -> list[str]:
        rows = self.db.query(
            "SELECT artist, title FROM tracks WHERE last_played_at IS NOT NULL "
            "ORDER BY last_played_at DESC LIMIT ?", (limit,)
        )
        return [f"{row['artist']} - {row['title']}" for row in rows]

    # -- hosts and segments -------------------------------------------------

    def hosts(self) -> list[dict]:
        """Configured hosts, first one leading. Always at least one."""
        entries = [host for host in (self.cfg.get("dj.hosts", []) or [])
                   if isinstance(host, dict) and str(host.get("name") or "").strip()]
        if not entries:
            return [{"name": "the host", "character": "warm, dry, never oversells"}]
        return entries

    def host_names(self, speaking: list[str] | None = None) -> list[str]:
        """Host names, optionally narrowed to those with a working voice.

        A co-host whose voice model never got downloaded must not appear in the
        running order: the line would be rendered in the wrong voice and the
        conversation would sound like one person arguing with themselves.
        """
        names = [str(host["name"]).strip() for host in self.hosts()]
        if speaking is None:
            return names
        allowed = [name for name in names if name in speaking]
        return allowed or names[:1]

    def _host_block(self, names: list[str]) -> str:
        characters = {str(host["name"]).strip(): str(host.get("character") or "")
                      for host in self.hosts()}
        lines = []
        for index, name in enumerate(names):
            role = "leads the hour" if index == 0 else "co-hosts"
            detail = characters.get(name, "").strip()
            lines.append(f"- {name} {role}." + (f" {detail}" if detail else ""))
        if len(names) == 1:
            lines.append("- There is no co-host tonight, so never write a "
                         "conversation: every spoken item is this one voice.")
        return "\n".join(lines)

    def segment_brief(self, airtime: datetime, names: list[str],
                      has_weather: bool, has_news: bool = False) -> dict:
        """Decide how much the hosts talk this hour, and about what.

        Rolled per block rather than fixed, because "talks more, but only
        sometimes" is the whole point: an hour that always has exactly one
        weather moment and exactly two chats is just a longer format, not a
        looser one.
        """
        segments = self.cfg.section("dj").get("segments") or {}

        banter = 0
        turns = 0
        if len(names) > 1:
            banter = _roll(segments.get("banter_per_block", [0, 2]))
            turns = max(2, _roll(segments.get("banter_turns", [4, 6])))
        notes = _roll(segments.get("music_notes_per_block", [0, 2]))

        weather = ""
        if has_weather and random.random() < float(
                segments.get("weather_chance", 0.5)):
            window = segments.get("weather_outlook_hours", [5, 11])
            try:
                start, end = int(window[0]), int(window[1])
            except (TypeError, ValueError, IndexError):
                start, end = 5, 11
            # The whole-day forecast is only useful before the day happens.
            weather = "outlook" if start <= airtime.hour < end else "now"

        news = has_news and random.random() < float(
            segments.get("news_chance", 0.4))

        return {"banter": banter, "banter_turns": turns,
                "music_notes": notes, "weather": weather, "news": news}

    def _segment_plan_text(self, brief: dict, names: list[str]) -> str:
        wants = []
        if brief["banter"] and len(names) > 1:
            pair = " and ".join(names[:2])
            wants.append(
                f"- Include {_count(brief['banter'], 'conversation')} between "
                f"{pair}, about {brief['banter_turns']} turns each time. Put "
                f"them where they fit the music, not at the very start.")
        if brief["music_notes"]:
            wants.append(
                f"- Include {_count(brief['music_notes'], 'music note')}: a "
                f"host saying one true, concrete thing about a record you are "
                f"playing. If you are not sure it is true, talk about how it "
                f"sounds instead.")
        if brief["weather"] == "outlook":
            wants.append("- Include exactly one weather moment, using the "
                         "brief below, covering how the day is shaping up. "
                         "Somewhere in the first half of the hour.")
        elif brief["weather"] == "now":
            wants.append("- Include exactly one weather moment, using the "
                         "brief below. Just the conditions outside right now, "
                         "in a sentence or two. Do not read out a forecast.")
        if brief.get("news"):
            wants.append("- Include exactly one news moment, using the "
                         "headlines below. Pick one or two that actually fit "
                         "this station, say them in your own natural spoken "
                         "words rather than reading a headline verbatim, and "
                         "never add a fact, number or opinion a headline does "
                         "not already contain.")

        if not wants:
            return ("This hour is a quiet one: every spoken item is a short "
                    "link. No conversations, no weather, no music trivia.")
        return ("Plan for this hour, on top of the usual short links:\n"
                + "\n".join(wants)
                + "\nEverything else stays a short link. Do not stack these "
                  "against each other; spread them through the hour.")

    # -- planning -----------------------------------------------------------

    def plan_block(self, now: datetime | None = None, airs_in: float = 0.0,
                   speaking_hosts: list[str] | None = None) -> dict:
        """Produce the next block. Always returns something playable.

        `airs_in` is how many seconds of audio are already queued ahead of this
        block, so it plans for when the listener will actually HEAR it. The
        buffer is an hour deep by design: planning against the current clock
        means the time-of-day genre arrives an hour late and the DJ announces
        a time that passed before anyone heard it.

        `speaking_hosts` is who actually has a voice on disk. A host that
        cannot be rendered must not be written into the show.
        """
        now = now or datetime.now()
        airtime = now + timedelta(seconds=max(0.0, airs_in))
        if airs_in > 60:
            log.info("planning for %s, which is when this block reaches air "
                     "(%s of audio queued ahead of it)",
                     airtime.strftime("%H:%M"), human_duration(airs_in))

        profile = self.cfg.mood_for_hour(airtime.hour)
        block_minutes = float(self.cfg.get("planner.block_minutes", 60))
        pool_size = int(self.cfg.get("planner.candidate_pool", 60))
        min_tracks = int(self.cfg.get("discovery.min_tracks_to_program", 25))

        # The listener asking right now outranks the time-of-day profile, so
        # an artist named in both is treated as focus, not as an exemplar.
        mood_override = self.current_mood()
        focus = self.artists_named_in(mood_override)
        exemplars = [artist for artist in
                     self.artists_named_in(profile.get("mood", ""))
                     if artist not in focus]
        # A standing mood locks the candidate pool to the time-of-day genre
        # otherwise -- "upbeat hard rap" during a jazz hour used to hand the
        # LLM 51 jazz tracks and ask it to sound excited about rap, which it
        # obviously cannot do with nothing but jazz on offer, and the
        # fallback planner (no free-text understanding at all) fared even
        # worse. An explicit override means the listener wants something
        # other than the default for this hour, so the genre lock -- which
        # exists to serve that default -- gets out of the way; named artists
        # and the mood text in the prompt still steer it.
        candidates = self.select_candidates(
            profile.get("energy", [1, 5]), pool_size,
            focus_artists=focus, exemplar_artists=exemplars,
            genres=(profile.get("genres") if not mood_override else None))
        if not candidates:
            log.warning("library is empty, cannot plan a block")
            return {"items": [], "source": "empty", "mood_name": profile.get("name"),
                    "show_note": ""}

        average = _average_duration(candidates)
        track_count = max(3, min(len(candidates), round(block_minutes * 60 / average)))

        names = self.host_names(speaking_hosts)
        briefing = self.weather.briefing() if self.weather else ""
        has_news = bool(self.news and self.news.headlines())
        brief = self.segment_brief(airtime, names, bool(briefing), has_news)
        log.info("hour brief: hosts=%s, banter=%d, music notes=%d, weather=%s, "
                 "news=%s", "/".join(names), brief["banter"],
                 brief["music_notes"], brief["weather"] or "none",
                 "yes" if brief.get("news") else "no")

        if self.llm.enabled and self.library.count() >= min_tracks:
            try:
                plan = self._plan_with_llm(airtime, profile, candidates,
                                           track_count, focus, names, brief)
                if plan["items"]:
                    return plan
                log.warning("LLM plan had no usable items, falling back")
            except LLMUnavailable as exc:
                log.warning("LLM planning unavailable (%s), using fallback planner", exc)
            except Exception as exc:
                log.exception("LLM planning blew up (%s), using fallback planner", exc)

        return self._plan_fallback(airtime, profile, candidates,
                                   track_count, focus, names, brief)

    def _plan_with_llm(self, airtime: datetime, profile: dict,
                       candidates: list[dict], track_count: int,
                       focus_artists: list[str] | None = None,
                       names: list[str] | None = None,
                       brief: dict | None = None) -> dict:
        """`airtime` is when the listener hears this, not when it was planned."""
        patter_every = int(self.cfg.get("dj.patter_every_n_tracks", 2))
        max_words = int(self.cfg.get("dj.max_patter_words", 45))
        max_segment_words = int(self.cfg.get("dj.max_segment_words", 90))
        opening = bool(self.cfg.get("dj.patter_at_block_start", True))
        mood = self.current_mood()
        names = names or self.host_names()
        brief = brief or {"banter": 0, "banter_turns": 4, "music_notes": 0,
                          "weather": ""}

        listing = "\n".join(
            "{id}. {artist} - {title}{extra}".format(
                id=track["id"], artist=track["artist"], title=track["title"],
                extra=_describe(track))
            for track in candidates
        )

        system = PLANNER_SYSTEM.format(
            persona=self.cfg.get("dj.persona", "You are a radio DJ."),
            host_block=self._host_block(names),
            first_host=names[0],
            second_host=names[1] if len(names) > 1 else names[0],
            language=self.cfg.get("dj.language", "english"),
            max_words=max_words,
            max_segment_words=max_segment_words,
            segment_plan=self._segment_plan_text(brief, names),
            artist_spacing=int(self.cfg.get("planner.artist_spacing", 4)),
        )
        user = PLANNER_USER.format(
            clock=_round_clock(airtime),
            day=airtime.strftime("%A"),
            slot_name=profile.get("name", "default"),
            slot_mood=profile.get("mood", "varied"),
            mood_override=_mood_note(mood, focus_artists),
            recent=", ".join(self._recent_titles()) or "nothing yet",
            weather_brief=(f"Weather brief (facts only, use nothing else):\n"
                           f"{self._weather_text(brief)}\n"
                           if brief["weather"] else ""),
            news_brief=(f"News brief (facts only, use nothing else):\n"
                       f"{self._news_text(brief)}\n"
                       if brief.get("news") else ""),
            track_count=track_count,
            patter_every=patter_every,
            patter_plural="" if patter_every == 1 else "s",
            opening_note=("Start the hour with a short link that sets the scene."
                          if opening else "Do not open with talk."),
            candidates=listing,
        )

        data = self.llm.complete_json(system, user, temperature=0.85,
                                      max_tokens=self.llm.budget("plan"),
                                      label="plan")
        items = self._validate(data.get("items") or [], candidates, track_count,
                               names, max_banter_turns=brief["banter_turns"] + 2)

        return {
            "items": items,
            "source": "llm",
            "mood_name": profile.get("name"),
            "show_note": str(data.get("show_note") or "")[:300],
        }

    def _weather_text(self, brief: dict) -> str:
        """The facts for this hour's weather moment, or nothing at all.

        An hour with no weather slot must not be handed a briefing: the model
        will use anything it is given, and then every hour has weather in it.
        """
        if not self.weather or not brief.get("weather"):
            return ""
        return self.weather.briefing(include_outlook=brief["weather"] == "outlook")

    def _news_text(self, brief: dict) -> str:
        """The facts for this hour's news moment, or nothing at all.

        Same reasoning as _weather_text: an hour with no news slot must not
        be handed headlines, or every hour ends up mentioning the news.
        """
        if not self.news or not brief.get("news"):
            return ""
        return self.news.briefing()

    def _validate(self, raw_items: list, candidates: list[dict],
                  track_count: int, names: list[str] | None = None,
                  max_banter_turns: int | None = None) -> list[dict]:
        """Trust nothing: drop invented ids, repeats and malformed entries."""
        by_id = {track["id"]: track for track in candidates}
        max_segment_words = int(self.cfg.get("dj.max_segment_words", 90))
        names = names or self.host_names()
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

            elif kind in ("patter", "link", "weather", "note", "news"):
                # The generous cap is the runaway guard, not the target; the
                # prompt asks for links under max_words and only lets weather,
                # news and music notes run longer.
                text = _clean_patter(str(raw.get("text") or ""), max_segment_words)
                if text:
                    items.append({"kind": "patter", "text": text,
                                  "host": _pick_host(raw.get("host"), names, 0)})

            elif kind in ("banter", "conversation", "exchange"):
                lines = _clean_exchange(raw.get("lines"), names,
                                        max_segment_words,
                                        max_turns=max_banter_turns)
                if len(lines) >= 2:
                    items.append({"kind": "banter", "lines": lines})
                elif lines:
                    # One usable turn is not a conversation, but it is still a
                    # line somebody can read out.
                    items.append({"kind": "patter", "text": lines[0]["text"],
                                  "host": lines[0]["host"]})

        if len(used) < max(2, track_count // 3):
            log.warning("LLM only produced %d valid songs, treating plan as failed",
                        len(used))
            return []

        items = _drop_consecutive_talk(items)

        # Never end a block on talk: it would leave the hosts introducing
        # something that is not there, and then dead air.
        while items and items[-1]["kind"] != "song":
            items.pop()

        self._report_artist_spacing(items)
        return items

    def _report_artist_spacing(self, items: list[dict]) -> None:
        """Log artist repeats the model slipped in.

        Not enforced by dropping tracks: the patter lines name the songs around
        them, so removing one would leave the DJ introducing a track that no
        longer plays. The prompt and the diversified pool do the work; this is
        how you find out when they were not enough.
        """
        spacing = int(self.cfg.get("planner.artist_spacing", 4))
        artists = [item["track"]["artist"] for item in items
                   if item["kind"] == "song"]
        clashes = [
            artist for index, artist in enumerate(artists)
            if artist in artists[max(0, index - spacing):index]
        ]
        if clashes:
            log.warning("LLM repeated %d artist(s) inside the %d-track spacing "
                        "window: %s", len(clashes), spacing,
                        ", ".join(sorted(set(clashes))))

    def _plan_fallback(self, airtime: datetime, profile: dict,
                       candidates: list[dict], track_count: int,
                       focus_artists: list[str] | None = None,
                       names: list[str] | None = None,
                       brief: dict | None = None) -> dict:
        """Deterministic programming for when the LLM is down.

        Weighted-random picks inside the energy band with artist spacing, plus
        plain template patter so the hour still sounds hosted.
        """
        spacing = int(self.cfg.get("planner.artist_spacing", 4))
        patter_every = int(self.cfg.get("dj.patter_every_n_tracks", 2))
        opening = bool(self.cfg.get("dj.patter_at_block_start", True))
        focus = {artist.lower() for artist in (focus_artists or [])}
        names = names or self.host_names()
        brief = brief or {"banter": 0, "banter_turns": 4, "music_notes": 0,
                          "weather": "", "news": False}

        pool = list(candidates)
        random.shuffle(pool)
        # Least recently played first, with a random jitter so it is not a cycle.
        # An artist the listener asked for sorts to the front regardless: the
        # LLM being down is no reason to ignore an explicit request.
        pool.sort(key=lambda t: (
            0 if (t.get("artist") or "").lower() in focus else 1,
            (t.get("last_played_at") or 0) + random.uniform(0, 3600),
        ))

        chosen: list[dict] = []
        used: set[int] = set()
        recent_artists: list[str] = []
        # Walking the pool once starves on a narrow library: with three artists
        # and a spacing of four, everything after the third pick is refused and
        # the "hour" comes out three tracks long. Relax the spacing until the
        # block is full instead.
        gap = spacing
        while len(chosen) < track_count and gap >= 0:
            progressed = False
            for track in pool:
                if len(chosen) >= track_count:
                    break
                if track["id"] in used:
                    continue
                if gap and track["artist"] in recent_artists[-gap:]:
                    continue
                chosen.append(track)
                used.add(track["id"])
                recent_artists.append(track["artist"])
                progressed = True
            if not progressed:
                gap -= 1
        if gap < spacing and chosen:
            log.info("fallback relaxed artist spacing to %d to fill the block "
                     "(library is narrow for this slot)", max(gap, 0))

        if not chosen:
            chosen = pool[:track_count]

        # Where the two extras go, if this hour has any. Kept away from the
        # opening so the hour still starts with a normal link.
        slots = [index for index in range(patter_every or 1, len(chosen))
                 if patter_every > 0 and index % patter_every == 0]
        random.shuffle(slots)
        banter_at = set(slots[:brief["banter"]] if len(names) > 1 else [])
        claimed = len(banter_at)
        weather_at = (slots[claimed] if brief["weather"] and
                      len(slots) > claimed else None)
        claimed += 1 if weather_at is not None else 0
        news_at = (slots[claimed] if brief.get("news") and
                  len(slots) > claimed else None)

        items: list[dict] = []
        if opening:
            items.append({"kind": "patter", "host": names[0],
                          "text": _template_opening(airtime, profile, chosen[0])})
        for index, track in enumerate(chosen):
            if index > 0 and patter_every > 0 and index % patter_every == 0:
                previous = chosen[index - 1]
                if index in banter_at:
                    items.append({"kind": "banter",
                                  "lines": _template_banter(
                                      names, previous, track,
                                      brief["banter_turns"])})
                elif index == weather_at and self.weather:
                    text = _template_weather(
                        self.weather.current(),
                        outlook=brief["weather"] == "outlook")
                    items.append({"kind": "patter", "host": names[0],
                                  "text": text or _template_link(previous, track)})
                elif index == news_at and self.news:
                    text = _template_news(self.news.headlines())
                    items.append({"kind": "patter", "host": names[0],
                                  "text": text or _template_link(previous, track)})
                else:
                    # Alternate per LINK, not per track index: with a link
                    # every second song the track index is always even, so
                    # index % len(names) would hand every line to one host.
                    turn = index // max(1, patter_every)
                    items.append({"kind": "patter",
                                  "host": names[turn % len(names)],
                                  "text": _template_link(previous, track)})
            items.append({"kind": "song", "track": track})

        log.info("fallback planner built %d songs for the %s slot",
                 len(chosen), profile.get("name"))
        note = self._fallback_show_note(profile)
        return {"items": items, "source": "fallback",
                "mood_name": profile.get("name"),
                "show_note": note[:300]}

    def _fallback_show_note(self, profile: dict) -> str:
        """What "This hour" should say when there is no LLM to write one.

        A standing mood the listener set with !mood is only ever partly
        honoured here -- the genre lock lifts for an override (see
        plan_block), so the pool is no longer walled off to the wrong genre,
        but the fallback planner still has no way to understand free text: it
        can only weight towards a library artist named in it (see
        artists_named_in/focus_artists) and otherwise picks close to at
        random from whatever the pool now contains. Saying just the slot's
        own mood text while a standing override is active looked like the
        request had been silently dropped; this says plainly that it hasn't
        fully taken effect rather than leaving that to be inferred.
        """
        mood_text = str(profile.get("mood") or "").strip()
        override = self.current_mood().strip()
        base = f"{mood_text} (fallback programming, no LLM)" if mood_text \
            else "fallback programming (no LLM)"
        if not override:
            return base
        return (f"Still playing the {profile.get('name')} slot's own "
               f"rotation -- \"{override}\" needs the LLM to fully apply "
               f"beyond artists named in it. {base}")

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
                                          temperature=0.2,
                                          max_tokens=self.llm.budget("tag"),
                                          label="tag")
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


def _drop_consecutive_talk(items: list[dict]) -> list[dict]:
    """Never let two spoken items run back to back.

    The prompt asks the model to interleave talk with songs, but nothing
    stops it front-loading an opening link, a music note, a weather moment
    and a whole conversation before playing a single track -- which is
    exactly what happens once several "extra" segment types are possible in
    one hour and the model just writes all of them first. The fallback
    planner can't do this: it inserts one song right after every talk item,
    structurally. The LLM plan gets that same guarantee here instead of
    just a prompt asking nicely for it.

    Dropping the excess is safe -- the hour just has one fewer aside. Trying
    to fix it by moving songs earlier would still leave a back-announce
    pointing at a track that no longer plays next to it.
    """
    fixed: list[dict] = []
    for item in items:
        if item["kind"] != "song" and fixed and fixed[-1]["kind"] != "song":
            log.info("dropping a %s item stacked on top of another talk item",
                     item["kind"])
            continue
        fixed.append(item)
    return fixed


def _mood_note(mood: str, focus_artists: list[str] | None) -> str:
    """The listener's standing instruction, as the prompt sees it."""
    if not mood:
        return ""
    note = (f"The listener has asked for: {mood}\n"
            "Weight the picks and your tone towards that.")
    if focus_artists:
        names = " and ".join(focus_artists)
        note += (f"\nThey named {names} specifically, and there are plenty of "
                 f"their tracks in the list below. Build the hour around them: "
                 f"play several and let that be the spine of the show. The "
                 f"artist spacing rule does not apply to {names}, only to "
                 f"everyone else -- though still never two of their songs back "
                 f"to back.")
    return note


def _round_clock(moment: datetime) -> str:
    """Nearest five minutes.

    An exact time reads badly out loud ("twenty-eight minutes past eleven"),
    and it is approximate anyway -- the block takes an hour to play out.
    """
    minute = int(round(moment.minute / 5.0) * 5)
    if minute >= 60:
        moment = moment + timedelta(hours=1)
        minute = 0
    return moment.replace(minute=minute).strftime("%H:%M")


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


def _template_banter(names: list[str], previous: dict, upcoming: dict,
                     turns: int = 4) -> list[dict]:
    """A two-host handover for when the LLM cannot write one.

    Short and factual on purpose. This only runs while the model is
    unreachable, and a stilted exchange that states what is playing beats an
    ambitious one that lands badly every single time it repeats.
    """
    first = names[0]
    second = names[1] if len(names) > 1 else names[0]
    scripts = [
        [(first, f"That was {previous['artist']}, {previous['title']}."),
         (second, "Not heard that one in a while."),
         (first, f"Next up is {upcoming['artist']}."),
         (second, f"{upcoming['title']}. Good call.")],
        [(second, f"So that was {previous['title']}."),
         (first, f"{previous['artist']}. Still holds up."),
         (second, "What are we doing after this?"),
         (first, f"{upcoming['artist']}, {upcoming['title']}.")],
        [(first, f"{previous['artist']} there."),
         (second, "Nice way to keep it moving."),
         (first, f"Staying in it. {upcoming['artist']} next."),
         (second, f"{upcoming['title']}. Let it run.")],
    ]
    script = random.choice(scripts)[:max(2, min(int(turns), 4))]
    return [{"host": host, "text": text} for host, text in script]


def _template_weather(reading: dict | None, outlook: bool = False) -> str:
    """A spoken weather line built straight from the reading."""
    if not reading:
        return ""
    temp = _spoken_number(reading.get("temp_c"))
    if not temp:
        return ""
    place = reading.get("place") or "town"
    condition = reading.get("condition") or ""
    line = f"Outside in {place} it is {temp} degrees"
    line += f", {condition}." if condition else "."

    if outlook:
        high = _spoken_number(reading.get("high_c"))
        low = _spoken_number(reading.get("low_c"))
        if high and low:
            line += f" Later on, a high of {high} and a low of {low}."
        chance = reading.get("rain_chance")
        if chance is not None and chance >= 40:
            line += f" About {_spoken_number(chance)} percent chance of rain."
    return line


def _template_news(headlines: list[dict]) -> str:
    """A spoken news line built straight from one real headline.

    No rewriting: the fallback planner has no LLM to safely paraphrase
    with, so it reads the headline as fetched rather than risk putting
    words in a story's mouth.
    """
    if not headlines:
        return ""
    headline = random.choice(headlines).get("title") or ""
    if not headline:
        return ""
    return f"In the news: {headline}."


def _roll(spec, default: int = 0) -> int:
    """Read a config value that may be a range [low, high] or a fixed number."""
    if isinstance(spec, (list, tuple)) and len(spec) >= 2:
        try:
            low, high = int(spec[0]), int(spec[1])
        except (TypeError, ValueError):
            return default
        if high < low:
            low, high = high, low
        return random.randint(max(0, low), max(0, high))
    try:
        return max(0, int(spec))
    except (TypeError, ValueError):
        return default


_SMALL = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def _count(number: int, noun: str) -> str:
    word = _SMALL.get(number, str(number))
    return f"{word} {noun}" + ("" if number == 1 else "s")


def _spoken_number(value) -> str:
    """Digits read badly out loud in some Piper voices; words never do."""
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return ""
    if number < 0:
        return "minus " + _spoken_number(-number)
    if number < 20:
        return _ONES[number]
    if number < 100:
        tens, rest = divmod(number, 10)
        return _TENS[tens] + (f" {_ONES[rest]}" if rest else "")
    if number == 100:
        return "a hundred"
    return str(number)


def _pick_host(raw, names: list[str], index: int = 0) -> str:
    """Map whatever the model wrote in "host" onto a real, speakable host."""
    wanted = str(raw or "").strip().lower()
    for name in names:
        if wanted == name.lower():
            return name
    return names[index % len(names)]


def _clean_exchange(raw_lines, names: list[str], max_words: int,
                    max_turns: int | None = None) -> list[dict]:
    """Turn the model's "lines" array into an alternating, speakable script.

    `max_turns` is a backstop, not the target -- the prompt already asks for
    a specific turn count, but nothing stopped the model writing twice that
    many, and a conversation left to run long is also a conversation left to
    drift off whatever it started about.
    """
    if not isinstance(raw_lines, list):
        return []
    if max_turns is not None:
        raw_lines = raw_lines[:max_turns]
    per_turn = max(12, max_words // 3)

    turns: list[dict] = []
    for index, raw in enumerate(raw_lines):
        if isinstance(raw, dict):
            host = _pick_host(raw.get("host"), names, index)
            text = _clean_patter(str(raw.get("text") or ""), per_turn)
        elif isinstance(raw, str):
            host = names[index % len(names)]
            text = _clean_patter(raw, per_turn)
        else:
            continue
        if not text:
            continue
        if turns and turns[-1]["host"] == host:
            # Two turns in a row from one host is not a conversation. Join
            # them rather than dropping half of what was written.
            turns[-1]["text"] = _clean_patter(
                turns[-1]["text"] + " " + text, per_turn * 2)
            continue
        turns.append({"host": host, "text": text})

    if len(names) < 2 and turns:
        # Nobody to bounce off, so read the whole thing as one voice.
        merged = _clean_patter(" ".join(turn["text"] for turn in turns), max_words)
        return [{"host": names[0], "text": merged}] if merged else []
    return turns
